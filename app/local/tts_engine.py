"""TTS engine: macOS 'say' stub for P1.

Synthesizes text using macOS say command, saves to a temp WAV at 24 kHz,
reads as numpy float32, and enqueues to AudioEngine for playback.

LIMITS:
- macOS only. Requires AudioEngine to be started.
- say outputs 24 kHz mono WAV (LEF16). AudioEngine playback_rate must match.
- No streaming: each sentence is synthesized fully before playback starts.
  P1 target is first audio within 1.5 s end-to-end (LLM + TTS).
- NEEDS OWNER: replace with Kokoro MLX or Piper once TTS engine is decided.
  See docs/PRD/RESEARCH-kokoro-mlx.md and docs/PRD/RESEARCH-piper.md.
- Playback reference ring in AudioEngine is populated via play(); echo
  cancellation (AEC3) is deferred to P4.

Voice: configurable via SAY_VOICE env var or settings.voice_tts_voice.
Default: Samantha (English, natural rate).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

log = logging.getLogger("app.local.tts_engine")

_DEFAULT_VOICE    = os.environ.get("SAY_VOICE", "Samantha")
_PLAYBACK_RATE_HZ = 24_000
_SAY_TIMEOUT_S    = 15.0
_PHRASES_DIR      = Path(os.environ.get("VOXSCRIBE_PHRASES_DIR",
                         Path.home() / ".voxscribe" / "phrases"))


async def synthesize_and_play(
    text: str,
    audio_engine,
    *,
    voice: str = _DEFAULT_VOICE,
) -> float:
    """Synthesize text and play through AudioEngine. Returns duration in seconds.

    text: the sentence to speak.
    audio_engine: an AudioEngine instance (must be started).
    Returns the audio duration in seconds, or 0.0 on failure.
    """
    if not text.strip():
        return 0.0

    try:
        import numpy as np
    except ImportError:
        log.error("tts_engine: numpy required for TTS playback")
        return 0.0

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmppath = f.name

    try:
        await asyncio.to_thread(
            _say_to_wav, text, tmppath, voice=voice
        )
        samples = await asyncio.to_thread(_read_wav_float32, tmppath, np)
    except Exception:
        log.exception("tts_engine: synthesis failed text=%r", text[:60])
        return 0.0
    finally:
        try:
            os.unlink(tmppath)
        except OSError:
            pass

    if samples is None or len(samples) == 0:
        return 0.0

    duration_s = len(samples) / _PLAYBACK_RATE_HZ
    try:
        await audio_engine.play(samples, block=True)
    except Exception:
        log.exception("tts_engine: playback failed")
    return duration_s


def _say_to_wav(text: str, path: str, *, voice: str) -> None:
    subprocess.run(
        [
            "say", "-v", voice,
            "--file-format", "WAVE",
            "--data-format", f"LEI16@{_PLAYBACK_RATE_HZ}",
            "-o", path,
            text,
        ],
        check=True,
        timeout=_SAY_TIMEOUT_S,
    )


def _read_wav_float32(path: str, np) -> object:
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples_i16 = np.frombuffer(raw, dtype="<i2")
    return samples_i16.astype("float32") / 32768.0


def _phrase_cache_path(text: str, voice: str) -> Path:
    key = hashlib.sha1(f"{voice}:{text}".encode()).hexdigest()[:16]
    return _PHRASES_DIR / f"{key}.wav"


async def prerender_phrase(text: str, *, voice: str = _DEFAULT_VOICE) -> Path | None:
    """Pre-render a fixed phrase to a cached WAV file. Returns the path or None on error.

    Idempotent: returns the cached path if the file already exists.
    Used for gate refusal phrases so playback incurs zero synthesis time.
    """
    path = _phrase_cache_path(text, voice)
    if path.exists():
        return path
    try:
        _PHRASES_DIR.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_say_to_wav, text, str(path), voice=voice)
        log.info("tts_engine: pre-rendered phrase len=%d -> %s", len(text), path.name)
        return path
    except Exception:
        log.exception("tts_engine: prerender failed text=%r", text[:60])
        return None


async def play_prerendered(phrase_path: Path, audio_engine) -> float:
    """Play a pre-rendered WAV file through AudioEngine. Returns duration seconds.

    Falls back silently if the file is missing or playback fails.
    """
    try:
        import numpy as np
    except ImportError:
        return 0.0
    try:
        samples = await asyncio.to_thread(_read_wav_float32, str(phrase_path), np)
    except Exception:
        log.exception("tts_engine: failed to read prerendered %s", phrase_path)
        return 0.0
    if samples is None or len(samples) == 0:
        return 0.0
    duration_s = len(samples) / _PLAYBACK_RATE_HZ
    try:
        await audio_engine.play(samples, block=True)
    except Exception:
        log.exception("tts_engine: prerendered playback failed")
    return duration_s
