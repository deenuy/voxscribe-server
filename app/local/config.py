"""Configuration management for the engine.

Loads from environment variables, then from a dotenv file (if VOXSCRIBE_ENV_FILE is set),
then from built-in defaults. Environment variables take precedence.
"""

import dataclasses
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Literal

# Turn detectors the live path can run: the two energy gates in core's LiveTranscriber, and the two real detectors
# of core's speech capability. Order is the order a person reads in an error message.
LIVE_VAD_MODES = ("adaptive", "fixed", "silero", "heuristic")
MAX_TICKET_TTL_S = 300  # a ticket is a one-shot credential, not a session: keep its lifetime short


def live_vad_mode_unavailable(mode: str) -> str | None:
    """Why `mode` cannot run on this machine, or None when it can.

    The energy gates always run. "heuristic" needs numpy; "silero" also needs onnxruntime and the model that
    faster-whisper ships. The check is the detector's own (`SileroDetector.unavailable_reason`), so the message
    names the missing piece.
    """
    if mode in ("adaptive", "fixed"):
        return None
    try:
        import numpy  # noqa: F401
    except ImportError:
        return "numpy is not installed (uv sync)"
    if mode == "silero":
        from voxscribe.speech.silero import SileroDetector

        return SileroDetector.unavailable_reason()
    return None


_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _hhmm(value: str) -> dtime:
    match = _HHMM.match(value)
    if not match:
        raise ValueError(f"expected HH:MM in 24-hour local time, got {value!r}")
    return dtime(int(match.group(1)), int(match.group(2)))


@dataclass(frozen=True)
class QuietHours:
    """A daily window of local time in which the assistant stays silent (no sound, no session).

    The window may cross midnight ("22:00" to "07:00"). start equal to end is an empty window.
    """

    enabled: bool = False
    start: str = "22:00"
    end: str = "07:00"

    def __post_init__(self) -> None:
        _hhmm(self.start)
        _hhmm(self.end)

    def active(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        t = (now or datetime.now()).time().replace(second=0, microsecond=0)
        start, end = _hhmm(self.start), _hhmm(self.end)
        if start == end:
            return False
        if start < end:
            return start <= t < end
        return t >= start or t < end


@dataclass
class Config:
    """Typed configuration for the voxscribe engine."""

    # Database
    database_url: str = "postgresql://localhost:5432/ideascribe_dev"

    # Providers
    providers: Literal["fake", "local"] = "local"

    # Data directory
    data_dir: str = ""

    # CORS
    allowed_origins: str = ""

    # Privacy mode
    privacy_mode: Literal["offline", "connectors"] = "offline"

    # Logging
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"

    # LLM (main model for enrichment, summaries, Ask)
    llm_model: str = "gemma4:12b"
    llm_temperature: float = 0.2
    llm_num_ctx: int = 8192
    llm_max_tokens: int = 1024
    llm_timeout_s: int = 120
    llm_keep_alive: str = "30m"

    # Live LLM (for live voice and live assistance)
    live_model: str = "gemma4:e4b"
    live_temperature: float = 0.3
    live_max_tokens: int = 256
    live_timeout_s: int = 20

    # Embedding model
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # ASR (speech recognition)
    asr_backend: str = "faster_whisper"
    asr_model: str = "small.en"
    asr_language: str = "en"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"

    # VAD (voice activity detection)
    vad_endpoint_ms: int = 700
    # Live transcription turn detection (VOXSCRIBE_LIVE_VAD_MODE): "adaptive" follows the room's noise floor so
    # quiet or far-field audio is still heard; "fixed" is the original absolute RMS gate; "silero" and "heuristic"
    # segment with core's shared speech capability (voxscribe.speech, ADR 0001) under the SEGMENTATION policy.
    # The default stays "adaptive" until the evaluation (docs/eval-live-vad.md) wins a change. Long continuous
    # speech is cut into a stable line every live_max_segment_ms (0 turns the cut off).
    live_vad_mode: str = "adaptive"
    live_max_segment_ms: int = 15000

    # Retrieval
    retrieval_top_k: int = 6

    # Event stream (/v1/events): how many recent events each tenant's stream keeps for a client that
    # reconnects (VOXSCRIBE_EVENTS_REPLAY_BUFFER), and how many streams are kept at once
    # (VOXSCRIBE_EVENTS_MAX_STREAMS). In memory only; a gap older than the buffer gets a stream.reset.
    events_replay_buffer: int = 1024
    events_max_streams: int = 256

    # Identity: when all three are set, every request must carry a verified person and each person's data is kept
    # apart. With none set, one person uses the engine at a time (the local single-user mode).
    identity_issuer: str = ""
    identity_audience: str = ""
    identity_jwks_url: str = ""
    identity_jwks_file: str = ""

    @property
    def tenancy_enforced(self) -> bool:
        return bool(self.identity_issuer and self.identity_audience and self.identity_jwks_url)

    # How long a ticket for the live socket or the event stream stays valid, in seconds (VOXSCRIBE_TICKET_TTL_S).
    # A ticket is used within moments of being issued; the lifetime only covers the round trip.
    ticket_ttl_s: int = 30

    # What this engine's own audio devices are called on a voice session's row (VOXSCRIBE_DEVICE_ID).
    # Empty: the voice scope adapter's name for an unnamed device.
    device_id: str = ""

    # Voice
    voice_enabled: bool = False
    tts_engine: str = "piper"
    tts_voice: str = ""
    voice_barge_in: bool = True
    voice_echo_guard: bool = True
    # Capture clean-up on the engine microphone (T12): echo cancellation against the playback
    # reference (VOXSCRIBE_AEC_ENABLED) and noise suppression (VOXSCRIBE_NOISE_SUPPRESS_ENABLED).
    # voice_echo_guard keeps the time-based mute behind them. Noise suppression is off by
    # default: on the fan-noise fixture at 10 dB SNR it raised recognizer word error
    # (tools/voice_bench/capture_wer.py, results/t12-capture/capture_wer.json).
    aec_enabled: bool = True
    noise_suppress_enabled: bool = False
    voice_max_reply_seconds: int = 20
    voice_mode: str = "default"          # default | classroom | seniorcare
    voice_reply_model: str = "llama3.2"  # 3B-4B class for spoken replies

    # Wake word — any phrase, no model training required (VOXSCRIBE_WAKE_PHRASE)
    wake_phrase: str = "hey vox"

    # Quiet hours (VOXSCRIBE_QUIET_HOURS_ENABLED, _START, _END; local time, HH:MM): a wake inside the
    # window is recorded as rejected and nothing is played.
    quiet_hours: QuietHours = dataclasses.field(default_factory=QuietHours)

    # Context header injected into every reply turn
    vox_timezone: str = "UTC"
    vox_location: str = ""
    vox_user_name: str = ""

    # Retention: how many days to keep wake candidate rows and tool-call records before the
    # periodic sweep deletes them. 0 (the default) means keep forever; set to a positive integer
    # to enable deletion.
    wake_candidate_retention_days: int = 0
    tool_call_retention_days: int = 0

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Config":
        """Load configuration from environment.

        Precedence:
        1. Environment variables
        2. Dotenv file (if env_file or VOXSCRIBE_ENV_FILE points to one)
        3. Built-in defaults (in the dataclass)
        """
        # Load dotenv file: explicit arg > VOXSCRIBE_ENV_FILE env var > .env in CWD
        env_vars = {}
        if env_file is None:
            env_file = os.environ.get("VOXSCRIBE_ENV_FILE")
        if env_file is None:
            candidate = Path.cwd() / ".env"
            if candidate.exists():
                env_file = str(candidate)

        if env_file and Path(env_file).exists():
            env_vars = _parse_env_file(env_file)

        # Merge with OS environment (OS env takes precedence)
        for key, value in os.environ.items():
            if key.startswith("VOXSCRIBE_") or key in ("OLLAMA_HOST",):
                env_vars[key] = value

        # Build config with validated values
        config_dict = {}

        # Database URL
        db_url = env_vars.get("VOXSCRIBE_DATABASE_URL")
        if db_url:
            config_dict["database_url"] = db_url

        # Identity
        for env_name, field in (
            ("VOXSCRIBE_IDENTITY_ISSUER", "identity_issuer"),
            ("VOXSCRIBE_IDENTITY_AUDIENCE", "identity_audience"),
            ("VOXSCRIBE_IDENTITY_JWKS_URL", "identity_jwks_url"),
            ("VOXSCRIBE_IDENTITY_JWKS_FILE", "identity_jwks_file"),
        ):
            value = env_vars.get(env_name, "").strip()
            if value:
                config_dict[field] = value
        configured = [bool(config_dict.get(f)) for f in ("identity_issuer", "identity_audience", "identity_jwks_url")]
        if any(configured) and not all(configured):
            _fail("VOXSCRIBE_IDENTITY_*", "set all of ISSUER, AUDIENCE and JWKS_URL, or none of them")

        # Tickets
        config_dict["ticket_ttl_s"] = _parse_int(env_vars, "VOXSCRIBE_TICKET_TTL_S", cls.ticket_ttl_s)
        if not 1 <= config_dict["ticket_ttl_s"] <= MAX_TICKET_TTL_S:
            _fail("VOXSCRIBE_TICKET_TTL_S", f"must be between 1 and {MAX_TICKET_TTL_S} seconds")

        # Device
        device_id = env_vars.get("VOXSCRIBE_DEVICE_ID", "").strip()
        if device_id:
            config_dict["device_id"] = device_id

        # Providers
        providers = env_vars.get("VOXSCRIBE_PROVIDERS", "").strip()
        if providers:
            if providers not in ("fake", "local"):
                _fail("VOXSCRIBE_PROVIDERS", f"must be 'fake' or 'local', got {providers!r}")
            config_dict["providers"] = providers

        # Data directory
        data_dir = env_vars.get("VOXSCRIBE_DATA_DIR", "").strip()
        if data_dir:
            config_dict["data_dir"] = data_dir

        # Allowed origins
        allowed_origins = env_vars.get("VOXSCRIBE_ALLOWED_ORIGINS", "").strip()
        if allowed_origins:
            config_dict["allowed_origins"] = allowed_origins

        # Privacy mode
        privacy_mode = env_vars.get("VOXSCRIBE_PRIVACY_MODE", "").strip()
        if privacy_mode:
            if privacy_mode not in ("offline", "connectors"):
                _fail("VOXSCRIBE_PRIVACY_MODE", f"must be 'offline' or 'connectors', got {privacy_mode!r}")
            config_dict["privacy_mode"] = privacy_mode

        # Log level
        log_level = env_vars.get("VOXSCRIBE_LOG_LEVEL", "").strip().lower()
        if log_level:
            if log_level not in ("debug", "info", "warning", "error"):
                _fail("VOXSCRIBE_LOG_LEVEL", f"must be debug|info|warning|error, got {log_level!r}")
            config_dict["log_level"] = log_level

        # Ollama host
        ollama_host = env_vars.get("OLLAMA_HOST", "").strip()
        if ollama_host:
            config_dict["ollama_host"] = ollama_host

        # LLM (main model)
        llm_model = env_vars.get("VOXSCRIBE_LLM_MODEL", "").strip()
        if llm_model:
            config_dict["llm_model"] = llm_model

        config_dict["llm_temperature"] = _parse_float(env_vars, "VOXSCRIBE_LLM_TEMPERATURE", 0.2)
        config_dict["llm_num_ctx"] = _parse_int(env_vars, "VOXSCRIBE_LLM_NUM_CTX", 8192)
        config_dict["llm_max_tokens"] = _parse_int(env_vars, "VOXSCRIBE_LLM_MAX_TOKENS", 1024)
        config_dict["llm_timeout_s"] = _parse_int(env_vars, "VOXSCRIBE_LLM_TIMEOUT_S", 120)

        llm_keep_alive = env_vars.get("VOXSCRIBE_LLM_KEEP_ALIVE", "").strip()
        if llm_keep_alive:
            config_dict["llm_keep_alive"] = llm_keep_alive

        # Live LLM
        live_model = env_vars.get("VOXSCRIBE_LIVE_MODEL", "").strip()
        if live_model:
            config_dict["live_model"] = live_model

        config_dict["live_temperature"] = _parse_float(env_vars, "VOXSCRIBE_LIVE_TEMPERATURE", 0.3)
        config_dict["live_max_tokens"] = _parse_int(env_vars, "VOXSCRIBE_LIVE_MAX_TOKENS", 256)
        config_dict["live_timeout_s"] = _parse_int(env_vars, "VOXSCRIBE_LIVE_TIMEOUT_S", 20)

        # Embedding model
        embed_model = env_vars.get("VOXSCRIBE_EMBED_MODEL", "").strip()
        if embed_model:
            config_dict["embed_model"] = embed_model

        config_dict["embed_dim"] = _parse_int(env_vars, "VOXSCRIBE_EMBED_DIM", 768)

        # Validate embed_dim
        if config_dict["embed_dim"] <= 0:
            _fail("VOXSCRIBE_EMBED_DIM", "must be positive")

        # ASR
        asr_backend = env_vars.get("VOXSCRIBE_ASR_BACKEND", "").strip()
        if asr_backend:
            if asr_backend not in ("faster_whisper",):
                _fail("VOXSCRIBE_ASR_BACKEND", f"must be 'faster_whisper', got {asr_backend!r}")
            config_dict["asr_backend"] = asr_backend

        asr_model = env_vars.get("VOXSCRIBE_ASR_MODEL", "").strip()
        if asr_model:
            config_dict["asr_model"] = asr_model

        asr_language = env_vars.get("VOXSCRIBE_ASR_LANGUAGE", "").strip()
        if asr_language:
            config_dict["asr_language"] = asr_language

        asr_device = env_vars.get("VOXSCRIBE_ASR_DEVICE", "").strip()
        if asr_device:
            config_dict["asr_device"] = asr_device

        asr_compute_type = env_vars.get("VOXSCRIBE_ASR_COMPUTE_TYPE", "").strip()
        if asr_compute_type:
            config_dict["asr_compute_type"] = asr_compute_type

        # VAD
        config_dict["vad_endpoint_ms"] = _parse_int(env_vars, "VOXSCRIBE_VAD_ENDPOINT_MS", 700)
        live_vad_mode = env_vars.get("VOXSCRIBE_LIVE_VAD_MODE", "").strip().lower()
        if live_vad_mode:
            if live_vad_mode not in LIVE_VAD_MODES:
                _fail("VOXSCRIBE_LIVE_VAD_MODE", f"must be one of {', '.join(LIVE_VAD_MODES)}, got {live_vad_mode!r}")
            reason = live_vad_mode_unavailable(live_vad_mode)
            if reason:
                # No silent fallback: a detector that cannot run is a startup error, not a quiet switch to energy.
                _fail("VOXSCRIBE_LIVE_VAD_MODE", f"{live_vad_mode!r} is not available here: {reason}")
            config_dict["live_vad_mode"] = live_vad_mode
        config_dict["live_max_segment_ms"] = _parse_int(env_vars, "VOXSCRIBE_LIVE_MAX_SEGMENT_MS", 15000)

        # Retrieval
        config_dict["retrieval_top_k"] = _parse_int(env_vars, "VOXSCRIBE_RETRIEVAL_TOP_K", 6)

        # Event stream bounds
        for env_name, field in (
            ("VOXSCRIBE_EVENTS_REPLAY_BUFFER", "events_replay_buffer"),
            ("VOXSCRIBE_EVENTS_MAX_STREAMS", "events_max_streams"),
        ):
            value = _parse_int(env_vars, env_name, getattr(cls, field))
            if value < 1:
                _fail(env_name, f"must be at least 1, got {value}")
            config_dict[field] = value

        # Voice (planned, accepted but unused)
        voice_enabled = env_vars.get("VOXSCRIBE_VOICE_ENABLED", "").strip().lower()
        if voice_enabled:
            config_dict["voice_enabled"] = voice_enabled in ("true", "1", "yes")

        tts_engine = env_vars.get("VOXSCRIBE_TTS_ENGINE", "").strip()
        if tts_engine:
            config_dict["tts_engine"] = tts_engine

        tts_voice = env_vars.get("VOXSCRIBE_TTS_VOICE", "").strip()
        if tts_voice:
            config_dict["tts_voice"] = tts_voice

        voice_barge_in = env_vars.get("VOXSCRIBE_VOICE_BARGE_IN", "").strip().lower()
        if voice_barge_in:
            config_dict["voice_barge_in"] = voice_barge_in in ("true", "1", "yes")

        voice_echo_guard = env_vars.get("VOXSCRIBE_VOICE_ECHO_GUARD", "").strip().lower()
        if voice_echo_guard:
            config_dict["voice_echo_guard"] = voice_echo_guard in ("true", "1", "yes")

        for env_name, field in (("VOXSCRIBE_AEC_ENABLED", "aec_enabled"),
                                ("VOXSCRIBE_NOISE_SUPPRESS_ENABLED", "noise_suppress_enabled")):
            value = env_vars.get(env_name, "").strip().lower()
            if value:
                config_dict[field] = value in ("true", "1", "yes")

        config_dict["voice_max_reply_seconds"] = _parse_int(
            env_vars, "VOXSCRIBE_VOICE_MAX_REPLY_SECONDS", 20
        )

        voice_mode = env_vars.get("VOXSCRIBE_VOICE_MODE", "").strip().lower()
        if voice_mode:
            if voice_mode not in ("default", "classroom", "seniorcare"):
                _fail("VOXSCRIBE_VOICE_MODE", f"must be default|classroom|seniorcare, got {voice_mode!r}")
            config_dict["voice_mode"] = voice_mode

        voice_reply_model = env_vars.get("VOXSCRIBE_VOICE_REPLY_MODEL", "").strip()
        if voice_reply_model:
            config_dict["voice_reply_model"] = voice_reply_model

        wake_phrase = env_vars.get("VOXSCRIBE_WAKE_PHRASE", "").strip().lower()
        if wake_phrase:
            config_dict["wake_phrase"] = wake_phrase

        quiet_kw: dict = {}
        quiet_enabled = env_vars.get("VOXSCRIBE_QUIET_HOURS_ENABLED", "").strip().lower()
        if quiet_enabled:
            quiet_kw["enabled"] = quiet_enabled in ("true", "1", "yes")
        for env_name, key in (("VOXSCRIBE_QUIET_HOURS_START", "start"),
                              ("VOXSCRIBE_QUIET_HOURS_END", "end")):
            value = env_vars.get(env_name, "").strip()
            if value:
                try:
                    _hhmm(value)
                except ValueError as exc:
                    _fail(env_name, str(exc))
                quiet_kw[key] = value
        if quiet_kw:
            config_dict["quiet_hours"] = QuietHours(**quiet_kw)

        # Context header — read VOX_* from .env or os.environ
        for env_key, field in (
            ("VOX_TIMEZONE", "vox_timezone"),
            ("VOX_LOCATION", "vox_location"),
            ("VOX_USER_NAME", "vox_user_name"),
        ):
            val = env_vars.get(env_key, "") or os.environ.get(env_key, "")
            if val.strip():
                config_dict[field] = val.strip()

        # Retention
        config_dict["wake_candidate_retention_days"] = _parse_int(
            env_vars, "VOXSCRIBE_WAKE_CANDIDATE_RETENTION_DAYS", 0
        )
        config_dict["tool_call_retention_days"] = _parse_int(
            env_vars, "VOXSCRIBE_TOOL_CALL_RETENTION_DAYS", 0
        )

        return cls(**config_dict)


def _parse_env_file(path: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file.

    Ignores GOOGLE_* and keys with SECRET, TOKEN, PASSWORD, or KEY in the name.
    Returns a dict of {key: value} pairs.
    """
    env_vars = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Skip GOOGLE_* keys (they're secrets)
                if key.startswith("GOOGLE_"):
                    continue

                # Skip app-only keys (not for engine)
                if key.startswith("IDEASCRIBE_"):
                    continue
                if key in ("VOXSCRIBE_SERVER_DIR", "VOXSCRIBE_CORE_DIR"):
                    continue

                env_vars[key] = value
    except FileNotFoundError:
        pass

    return env_vars


def _parse_int(env_vars: dict[str, str], key: str, default: int) -> int:
    """Parse an integer from env_vars, return default if not set."""
    value = env_vars.get(key, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        _fail(key, f"must be an integer, got {value!r}")


def _parse_float(env_vars: dict[str, str], key: str, default: float) -> float:
    """Parse a float from env_vars, return default if not set."""
    value = env_vars.get(key, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        _fail(key, f"must be a float, got {value!r}")


def _fail(key: str, detail: str) -> None:
    """Print error and exit with status 2."""
    print(f"{key}: {detail}", file=sys.stderr)
    sys.exit(2)
