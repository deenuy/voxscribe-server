"""Voice host: wires the ten ports of voxscribe.session to server adapters and runs the cascade.

Enabled with --wake; replaces the old monolithic wake loop. The host owns the voice state
machine (off, armed, listening, error) and the five-stage cascade:

  capture    AudioIn delivers 32 ms float32 chunks to this host's subscriber queue
  recognize  armed: the two-stage wake gate (detector patience, voice check, recognizer
             confirm) over 80 ms frames; listening: a recognizer stream per turn, run as its own
             task so recognition overlaps capture
  turn       Speech decides when the turn ends; it is ticked with stream time from the Clock
             (samples fed this turn), never engine uptime, and a ClockDomainError is logged
  reply      the Delegate produces sentences; voxscribe.session.stages.reply gates each one
             before the Talker synthesises it
  playback   AudioOut plays the audio; while it plays (and for the echo guard tail) capture is
             muted from the turn stage and the wake gate does not run
  barge-in   while a reply generates and plays, capture energy (echo-cancelled, T12) is read by
             the BargeInDetector: an interruption stops playback and generation, the stored turn
             and the history keep only the clauses that were heard, the interjection becomes the
             next turn, and that turn's prompt says not to repeat the heard part. A backchannel
             keeps the floor. Off when capture is not echo-cancelled (see voice/barge_in.py)

This is the only module in app/local/voice that knows which adapter backs each port, and it
imports them only in build_ports(). Everything else goes through the Protocols.

After turn close (T10): the end cue plays as a background task and the final decode starts at once,
bounded by FINALIZE_TIMEOUT_MS; past it the reply starts on the turn's stable partial (drift gated)
and the late decode is stored before the session closes. Before any reply the speech guard (T03)
drops chime echoes, one-word turns, too little audio and decodes Whisper scores as not speech; on
turn 1 that is a silent close and no meeting is left behind. Cues are muted from capture by the
AudioOut adapter, except the pre-roll, which predates the acknowledgement chime.

Also carried over: quiet hours (a wake inside the window is recorded as rejected and nothing is
played or opened), the capture watchdog hook (after a stream reopen the gate is reset on the next
frame), the debug trigger (behind --debug-trigger, see routers/wake.py), barge-in while armed
(the playback-ratio rule, BARGE_RATIO), multi-turn conversations with stop phrases, and the
speculative reply start with its drift rule.
No transcript text, audio or prompt is logged.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from voxscribe.session.ports import (
    AudioIn,
    AudioOut,
    ClockDomainError,
    Delegate,
    Gate,
    Recognizer,
    SessionStore,
    Speech,
    Talker,
    TranscriptSegment,
    TurnDecision,
    WakeDetector,
)
from voxscribe.session.ports import ReplyStream
from voxscribe.session.stages.reply import speak_reply
from voxscribe.wake.gate import ASR_CONFIRM_PHRASE, FRAME_SAMPLES, GateConfig, WakeGate

from app.local.voice.barge_in import (
    BACKCHANNEL,
    BARGE_IN,
    BARGE_IN_POLL_S,
    BARGE_IN_PRE_ROLL_MS,
    OVERLAP,
    BargeInDetector,
    HeardReply,
    Interruption,
    interruption_note,
)

log = logging.getLogger("app.local.wake")
log_tts = logging.getLogger("app.local.tts")

# app.local.voice.controls and app.local.voice.memory are imported lazily inside the methods
# that use them so that this module satisfies the boundary test (no top-level app.* imports).

SAMPLE_RATE = 16_000
CHUNK_FRAMES = 512              # 32 ms at 16 kHz, the capture block
RMS_FLOOR = 0.005
SILENCE_S = 6.0                 # end-of-turn silence when the turn detector is unavailable
CAP_S = 300.0                   # hard cap per turn
CONV_IDLE_S = 20.0              # how long to wait for the next turn of a conversation
FIRST_TURN_GRACE_S = 2.0        # first turn: SILENCE_S plus this before giving up
IDLE_SPEECH_RMS_FACTOR = 4      # audio this far above RMS_FLOOR keeps a conversation open
QUEUE_POLL_S = 0.5              # a capture gap this long ends the turn as timed out
TICK_EVERY_FRAMES = 5           # poll the turn detector for a pause every this many chunks
TURN_RING_S = 60.0              # capture held for the recognizer, oldest dropped beyond this
RECOGNIZE_DRAIN_TIMEOUT_S = 120.0
BARGE_RATIO = 2.5               # mic_rms / playback_rms above this stops playback while armed
MIN_PLAYBACK_RMS = 1e-6
SPEC_MIN_WORDS = 4              # do not start the reply model early on fewer words than this
ARMED_POLL_S = 0.25
CRASH_RESET_S = 1.0
TRIGGER_POLL_S = 0.1
TRIGGER_POLL_TRIES = 20
INT16_SCALE = 32767

CHIME_ACK = "/System/Library/Sounds/Glass.aiff"
CHIME_END = "/System/Library/Sounds/Pop.aiff"
CHIME_ERR = "/System/Library/Sounds/Basso.aiff"

# Fast finalize (T10): the end cue never blocks the final decode, and the final decode is given
# this long after turn close; past it the turn's stable partial is used and the final decode is
# stored when it lands.
FINALIZE_TIMEOUT_MS = 2000
# A partial is stable when it revises at most this share of the words the previous partial had
# (words appended after them are new speech, not drift).
DRIFT_GATE_THRESHOLD = 0.15

# Speech guard (T03): a turn is answered only if it looks like a person spoke. Whisper hears the
# acknowledgement chime as one of CHIME_TRANSCRIPTS.
CHIME_TRANSCRIPTS = frozenset({"beep", "boop", "bing", ""})
MIN_REPLY_WORDS = 2
MIN_AUDIO_MS = 500              # unmuted audio the turn must hold
MAX_NO_SPEECH_PROB = 0.6        # Whisper's own no-speech threshold

TRIGGER_DETECTOR = "trigger"    # wake_events.detector for the debug trigger route
QUIET_HOURS_REASON = "quiet_hours"
STOP_HINT = "thank you vox"

# Clause-streaming constants (T11) -- keep in sync with clause_splitter.CLAUSE_GAP_MAX_MS
_CLAUSE_GAP_MAX_MS = 300        # gap above this is perceptible; log a warning

# Phrases that end the conversation, including Whisper mishearings of "vox" (wox, wux, walks).
STOP_PHRASES: dict[str, set[str]] = {
    "hey vox": {
        "thank you vox", "thanks vox",
        "thank you wox", "thanks wox",
        "thank you wux", "thanks wux",
        "thank you walks", "thanks walks",
        "goodbye vox", "bye vox", "goodbye wox", "bye wox",
        "hey vox stop", "vox stop", "wox stop", "wux stop",
        "stop listening", "stop vox", "stop wox",
    },
}

# Words a final flush may add without changing the meaning; any other added word cancels the
# speculative reply (owner-decided 2026-10-17).
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "it", "this", "that", "i", "you", "my",
    "your", "we", "me", "s", "ll", "d", "re", "ve", "m", "t",
})


def is_stop_phrase(text: str, wake_phrase: str) -> bool:
    norm = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    return any(p in norm for p in STOP_PHRASES.get(wake_phrase, set()))


def spec_text_changed(pre: str, final: str) -> bool:
    """True when the final text is not the pre-flush text plus function words only."""
    pre_words = re.sub(r"[^\w\s]", " ", pre.lower()).split()
    final_words = re.sub(r"[^\w\s]", " ", final.lower()).split()
    if final_words[:len(pre_words)] != pre_words:
        return True
    return any(w not in FUNCTION_WORDS for w in final_words[len(pre_words):])


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def partial_drift(previous: str, latest: str) -> float:
    """Word error of the latest partial against the previous one, over the previous one's words.

    Words the latest partial appends after them are new speech and do not count as drift."""
    ref = _words(previous)
    if not ref:
        return 0.0
    hyp = _words(latest)[:len(ref)]
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev_diag, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev_diag, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev_diag + (r != h))
    return row[-1] / len(ref)


def is_genuine_speech(text: str, audio_duration_ms: int,
                      no_speech_prob: float | None = None) -> bool:
    """False for a chime transcript, fewer than MIN_REPLY_WORDS words, under MIN_AUDIO_MS of
    unmuted audio, or a final decode Whisper itself scores as not speech."""
    stripped = text.strip()
    if stripped.lower().rstrip(".!?,;:") in CHIME_TRANSCRIPTS:
        return False
    if len(stripped.split()) < MIN_REPLY_WORDS:
        return False
    if audio_duration_ms < MIN_AUDIO_MS:
        return False
    if no_speech_prob is not None and no_speech_prob > MAX_NO_SPEECH_PROB:
        return False
    return True


class StablePartial:
    """The open utterance's last stable partial, the fallback when the final decode is late."""

    def __init__(self) -> None:
        self._previous = ""
        self.text = ""

    def update(self, partial: str) -> None:
        if not self._previous or partial_drift(self._previous, partial) <= DRIFT_GATE_THRESHOLD:
            self.text = partial
        self._previous = partial

    def clear(self) -> None:
        """A stable segment ended the utterance: its text is in the turn's stable texts now."""
        self._previous = self.text = ""


class WakeState(str, Enum):
    OFF = "off"
    ARMED = "armed"
    LISTENING = "listening"
    ERROR = "error"


class QuietHours:
    """Active during the configured window. start and end are 'HH:MM' strings.

    When start > end the window wraps midnight (e.g. 21:00-08:00).
    """

    def __init__(self, start: str, end: str) -> None:
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        self._start = datetime.time(sh, sm)
        self._end = datetime.time(eh, em)

    def active(self) -> bool:
        now = datetime.datetime.now().time().replace(second=0, microsecond=0)
        s, e = self._start, self._end
        if s <= e:
            return s <= now < e
        # wraps midnight
        return now >= s or now < e


_CONFIG_DEFAULT_WAKE_PHRASE = "hey vox"   # the Config dataclass default; used to detect explicit overrides


@dataclass(frozen=True)
class VoiceConfig:
    wake_phrase: str = "hey vox"
    trigger_phrase: str = ASR_CONFIRM_PHRASE    # what a person must say; shown in the UI
    detector: str = "openvocab"                 # display name until the detector is loaded
    quiet_hours: object | None = None           # anything with active() -> bool
    # When True, a wake inside the quiet-hours window is rejected and nothing plays.
    # Senior care always answers a wake (VX-H5; PRODUCT-MAP.md: "must still answer a wake at
    # night; only proactive speech and volume are restricted"), so this is False for that mode.
    # TODO: this belongs in the core profile schema; move it there with the next core wheel.
    quiet_hours_block_wake: bool = True
    max_turns: int = 0                          # 0 = no limit; from profile.reply.max_turns
    conv_idle_s: float = CONV_IDLE_S            # idle timeout per turn; from profile.reply.idle_timeout_s

    @classmethod
    def from_settings(cls, settings) -> VoiceConfig:
        from app.local.voice.wake_setup import DEFAULT_DETECTOR, DETECTOR_ENV

        return cls(
            wake_phrase=getattr(settings, "wake_phrase", "hey vox").lower().strip(),
            detector=os.environ.get(DETECTOR_ENV, "").strip() or DEFAULT_DETECTOR,
            quiet_hours=getattr(settings, "quiet_hours", None),
        )

    @classmethod
    def from_profile(cls, profile, settings=None) -> VoiceConfig:
        """Build from a voxscribe-core Profile; profile-driven fields replace defaults.

        When settings is supplied, explicitly configured server-config values win over
        the profile's defaults:
          - wake_phrase: settings.wake_phrase wins when it differs from the Config
            dataclass default (i.e. VOXSCRIBE_WAKE_PHRASE was set by the operator).
          - quiet_hours: settings.quiet_hours wins when settings.quiet_hours.enabled
            is True (i.e. VOXSCRIBE_QUIET_HOURS_ENABLED=true was set by the operator).
            Exception: settings can never turn on wake-blocking for senior_care mode.
        """
        from app.local.voice.wake_setup import DEFAULT_DETECTOR, DETECTOR_ENV

        mode = getattr(profile, "mode", "personal")

        # Senior care must always answer a wake (VX-H5); only proactive speech is suppressed.
        quiet_hours_block_wake = (mode != "senior_care")

        quiet_hours: QuietHours | None = None
        if getattr(profile, "quiet_hours_enabled", False):
            quiet_hours = QuietHours(
                getattr(profile, "quiet_hours_start", "22:00"),
                getattr(profile, "quiet_hours_end", "07:00"),
            )

        wake = getattr(profile, "wake", None)
        wake_phrase = (getattr(wake, "phrase", "hey vox") if wake else "hey vox").lower().strip()

        reply = getattr(profile, "reply", None)
        max_turns = int(getattr(reply, "max_turns", 0)) if reply else 0
        conv_idle_s = float(getattr(reply, "idle_timeout_s", CONV_IDLE_S)) if reply else CONV_IDLE_S

        # Apply explicit server-config overrides on top of profile defaults.
        if settings is not None:
            # Wake phrase: explicit env override wins when it differs from the config default.
            settings_phrase = getattr(settings, "wake_phrase", _CONFIG_DEFAULT_WAKE_PHRASE)
            if settings_phrase and settings_phrase != _CONFIG_DEFAULT_WAKE_PHRASE:
                wake_phrase = settings_phrase.lower().strip()

            # Quiet hours: operator env override wins, but can never enable wake-blocking
            # for senior care -- that would silence the person who needs it most.
            cfg_qh = getattr(settings, "quiet_hours", None)
            if cfg_qh is not None and getattr(cfg_qh, "enabled", False):
                quiet_hours = QuietHours(
                    getattr(cfg_qh, "start", "22:00"),
                    getattr(cfg_qh, "end", "07:00"),
                )
                # Still cannot block wakes in senior care.
                if mode == "senior_care":
                    quiet_hours_block_wake = False

        return cls(
            wake_phrase=wake_phrase,
            detector=os.environ.get(DETECTOR_ENV, "").strip() or DEFAULT_DETECTOR,
            quiet_hours=quiet_hours,
            quiet_hours_block_wake=quiet_hours_block_wake,
            max_turns=max_turns,
            conv_idle_s=conv_idle_s,
        )


class StreamClock:
    """The Clock port, inline in the host: stream time is samples fed this turn."""

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 mono: Callable[[], int] | None = None) -> None:
        self._rate = sample_rate
        self._samples = 0
        self._mono = mono or (lambda: int(time.monotonic() * 1000))

    def reset(self) -> None:
        self._samples = 0

    def advance(self, samples: int) -> None:
        self._samples += int(samples)

    def stream_ms(self) -> int:
        return self._samples * 1000 // self._rate

    def mono_ms(self) -> int:
        return int(self._mono())


class _ClauseTalker:
    """Talker wrapper that logs a warning when synthesis of clause N+1 is not ready
    before clause N's expected playback end (a perceptible gap between clauses).

    The gap estimate is conservative: it compares synthesis duration against the
    PCM length of the previous clause, treating play() as non-blocking (queued)
    which is the AudioOut contract.
    """

    def __init__(self, inner: Talker, clock: StreamClock) -> None:
        self._inner = inner
        self._clock = clock
        self._prev_pcm_duration_ms: int = 0
        self._prev_synth_done_ms: int | None = None
        self._clause_index: int = 0

    @property
    def sample_rate(self) -> int:
        return self._inner.sample_rate

    async def synthesize(self, text: str) -> np.ndarray:
        self._clause_index += 1
        t_start = self._clock.mono_ms()
        pcm = await self._inner.synthesize(text)
        t_done = self._clock.mono_ms()
        synth_ms = t_done - t_start

        # Gap = how much longer synthesis took than the previous clause's playback.
        if self._prev_synth_done_ms is not None and self._clause_index > 1:
            expected_prev_end = self._prev_synth_done_ms + self._prev_pcm_duration_ms
            gap_ms = max(0, t_done - expected_prev_end)
            if gap_ms > 0:
                log_tts.warning(
                    "tts: gap before clause %d dur_ms=%d synth_ms=%d (max=%d ms)",
                    self._clause_index, gap_ms, synth_ms, _CLAUSE_GAP_MAX_MS,
                )

        # Record timing for the next clause's gap check.
        pcm_len = getattr(pcm, "size", len(pcm))
        if pcm_len > 0:
            self._prev_pcm_duration_ms = int(pcm_len * 1000 / self._inner.sample_rate)
        self._prev_synth_done_ms = t_done
        return pcm

    async def fixed(self, phrase: str) -> np.ndarray:
        # Fixed phrases (cues, refusals) are not clauses; reset gap tracking.
        self._prev_synth_done_ms = None
        self._clause_index = 0
        return await self._inner.fixed(phrase)

    async def warm(self) -> None:
        await self._inner.warm()


async def _clause_sentence_stream(original_sentences):
    """Wrap an async sentence iterator to yield clauses instead of sentences.

    Each sentence from the delegate is split at valid clause boundaries so that
    the first clause can be synthesised and queued for playback before the rest
    of the sentence is ready. Gate and synthesis proceed per clause.
    """
    from app.local.voice.clause_splitter import split_into_clauses  # lazy: no model load

    async for sentence in original_sentences:
        for clause in split_into_clauses(sentence):
            yield clause


@dataclass
class VoicePorts:
    audio_in: AudioIn
    audio_out: AudioOut
    wake: WakeDetector
    recognizer: Recognizer
    talker: Talker
    delegate: Delegate
    speech: Speech
    gate: Gate
    store: SessionStore
    clock: StreamClock


def build_ports(app) -> VoicePorts:
    """The production adapters. Loads models: call off the event loop."""
    from app.local.voice.adapters import audio_portaudio
    from app.local.voice.adapters.asr_whisper import WhisperRecognizer
    from app.local.voice.adapters.delegate_ollama import OllamaDelegate
    from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy
    from app.local.voice.adapters.speech_heuristic import HeuristicSpeech
    from app.local.voice.adapters.store_postgres import PostgresSessionStore
    from app.local.voice.adapters.tts_kokoro import KokoroTalker
    from app.local.voice.wake_setup import make_detector

    settings = getattr(app.state, "settings", None)
    engine = getattr(app.state, "audio_engine", None)
    profile_store = getattr(getattr(app, "state", None), "profile_store", None)

    def _get_voice_speed():
        if profile_store is not None:
            asst = profile_store.get().assistant
            return (asst.voice_id, asst.speed)
        return ("", 1.0)

    recognizer = WhisperRecognizer(getattr(app.state, "providers", {}).get("asr"), settings)
    recognizer.warm()
    return VoicePorts(
        audio_in=audio_portaudio.PortAudioIn(engine),
        audio_out=audio_portaudio.PortAudioOut(engine),
        wake=make_detector(phrase=ASR_CONFIRM_PHRASE),
        recognizer=recognizer,
        talker=KokoroTalker(get_voice_speed=_get_voice_speed),
        delegate=OllamaDelegate(app),
        speech=HeuristicSpeech(),
        gate=CorePolicyGate(session_policy(app)),
        store=PostgresSessionStore(app),
        clock=StreamClock(mono=audio_portaudio.monotonic_ms),
    )


def make_wake_gate(ports: VoicePorts, phrase: str = ASR_CONFIRM_PHRASE) -> WakeGate:
    """The two-stage gate over the ports: WakeDetector, then Speech's voice check, then a
    Recognizer confirm pass."""
    return WakeGate(ports.wake, voice_ratio=ports.speech.voice_ratio,
                    transcribe=lambda window: ports.recognizer.transcribe(window).words,
                    config=GateConfig(phrase=phrase))


class PreRollQueue:
    """A session's audio source: audio already heard (after the wake phrase, or the start of an
    interjection that interrupted a reply), then live capture.

    `from_pre_roll` is True while the last chunk handed out was pre-roll: that audio predates
    the acknowledgement chime, so the chime's capture mute does not apply to it."""

    def __init__(self, pre_roll, live, chunk_frames: int = CHUNK_FRAMES) -> None:
        self._pre = deque(pre_roll[i: i + chunk_frames]
                          for i in range(0, len(pre_roll), chunk_frames))
        self._live = live
        self.from_pre_roll = False

    def pre_roll_pending(self) -> bool:
        """True while the next chunk is pre-roll: it predates any mute and is never muted."""
        return bool(self._pre)

    async def get(self):
        if self._pre:
            self.from_pre_roll = True
            return self._pre.popleft()
        self.from_pre_roll = False
        return await self._live.get()

    def get_nowait(self):
        if self._pre:
            self.from_pre_roll = True
            return self._pre.popleft()
        self.from_pre_roll = False
        return self._live.get_nowait()

    def empty(self) -> bool:
        return not self._pre and self._live.empty()


class _TurnRing:
    """Capture held for the recognizer task: drop-oldest, closed with a None sentinel."""

    def __init__(self, capacity_chunks: int) -> None:
        self._q: asyncio.Queue = asyncio.Queue(maxsize=max(1, capacity_chunks) + 1)
        self._cap = max(1, capacity_chunks)
        self.pushed_samples = 0         # unmuted audio this turn, for the speech guard

    @property
    def pushed_ms(self) -> int:
        return self.pushed_samples * 1000 // SAMPLE_RATE

    def push(self, chunk) -> None:
        if self._q.qsize() >= self._cap:
            self._q.get_nowait()
        self._q.put_nowait(chunk)
        self.pushed_samples += len(chunk)

    def close(self) -> None:
        self._q.put_nowait(None)

    async def pop(self):
        return await self._q.get()


def _rms(chunk) -> float:
    return float((np.asarray(chunk, dtype=np.float32) ** 2).mean() ** 0.5)


def _take_queued(q) -> list:
    out = []
    while not q.empty():
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


def _drain(q) -> None:
    _take_queued(q)


def _no_events(_kind: str, _data: dict) -> None:
    pass


@dataclass
class SessionContext:
    """One conversation's state across its turns."""

    session_id: str = ""
    history: list[dict] = field(default_factory=list)
    late_finals: list[asyncio.Task] = field(default_factory=list)   # final decodes past the window
    silent: bool = False            # turn 1 heard no genuine speech: leave no meeting behind
    barge_in: bool = False         # the user interrupted the previous reply
    played_text: str = ""          # what the user heard of it
    pre_roll: np.ndarray | None = None   # the interjection's start, captured during the reply
    held_empty: bool = False       # one wordless turn after a barge-in did not end the session


@dataclass
class ReplyResult:
    text: str                      # recorded as spoken: the heard text after an interruption
    interruption: Interruption | None = None
    pre_roll: np.ndarray | None = None


class VoiceHost:
    """Builds the ports (once, off the loop), then runs armed, listening, armed, ..."""

    def __init__(self, config: VoiceConfig, ports: VoicePorts | None = None, *,
                 build: Callable[[], VoicePorts] | None = None,
                 wake_gate: WakeGate | None = None,
                 publish: Callable[[str, dict], None] | None = None) -> None:
        if ports is None and build is None:
            raise ValueError("VoiceHost needs ports or a build function")
        self.config = config
        self.ports = ports
        self._build = build
        self._gate = wake_gate
        self._publish = publish or _no_events
        self._task: asyncio.Task | None = None
        self._cues: set[asyncio.Task] = set()      # cues playing off the hot path
        self._capture_reset_pending = False
        self._state: dict = {
            "state": WakeState.OFF,
            "model": config.detector,      # replaced by the detector's model_name once loaded
            "phrase": config.wake_phrase,
            "trigger": config.trigger_phrase,
            "meeting_id": None,            # set for the duration of a session
        }

    @classmethod
    def from_app(cls, app) -> VoiceHost:
        from app.local.voice.adapters.events_app import publish_broadcast

        profile = getattr(getattr(app, "state", None), "profile", None)
        settings = getattr(getattr(app, "state", None), "settings", None)
        if profile is not None:
            config = VoiceConfig.from_profile(profile, settings=settings)
        else:
            config = VoiceConfig.from_settings(settings)
        return cls(config, build=lambda: build_ports(app), publish=publish_broadcast)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> WakeState:
        return self._state["state"]

    def snapshot(self) -> dict:
        return dict(self._state)

    def set_state(self, s: WakeState) -> None:
        self._state["state"] = s

    def arm(self) -> bool:
        if self.state in (WakeState.ARMED, WakeState.LISTENING):
            return False
        self.set_state(WakeState.ARMED)
        self._publish("wake_state", self.snapshot())
        return True

    def disarm(self) -> bool:
        if self.state == WakeState.OFF:
            return False
        self.set_state(WakeState.OFF)
        self._publish("wake_state", self.snapshot())
        return True

    @property
    def ready(self) -> bool:
        return self.ports is not None

    def _publish_state(self, state: WakeState) -> None:
        self._publish("wake_state", self.snapshot() | {"state": state.value})

    def _play_cue(self, path: str) -> asyncio.Task:
        """Play a cue as a background task. The AudioOut adapter mutes capture for the cue and
        its decay (EchoGuard.notify_chime), so nothing waits for it to finish."""
        task = asyncio.create_task(self.ports.audio_out.play_file(path), name="voice-cue")
        self._cues.add(task)
        task.add_done_callback(self._cues.discard)
        return task

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="voice-host")
        return self._task

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def run(self) -> None:
        """Outer wrapper: any unhandled exception resets to armed after CRASH_RESET_S."""
        log.info("voice host: starting")
        self.set_state(WakeState.ARMED)
        try:
            while True:
                if self.state == WakeState.OFF:
                    await asyncio.sleep(ARMED_POLL_S)
                    continue
                try:
                    await self._armed_session_loop()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("voice host: crashed -- resetting in %.0f s", CRASH_RESET_S)
                    self.set_state(WakeState.ERROR)
                    await asyncio.sleep(CRASH_RESET_S)
                    self.set_state(WakeState.ARMED)
        except asyncio.CancelledError:
            log.info("voice host: stopped (cancelled)")
            self.set_state(WakeState.OFF)
            raise

    async def _ready(self) -> VoicePorts:
        if self.ports is None:
            log.info("voice host: loading adapters detector=%s", self.config.detector)
            self.ports = await asyncio.to_thread(self._build)
        return self.ports

    async def _wake_gate(self) -> WakeGate:
        p = await self._ready()
        if self._gate is None:
            self._gate = make_wake_gate(p, self.config.trigger_phrase)
            log.info("voice host: wake gate ready detector=%s", p.wake.model_name)
        return self._gate

    async def _armed_session_loop(self) -> None:
        p = await self._ready()
        gate = await self._wake_gate()
        self._state["model"] = gate.detector.model_name
        self.set_state(WakeState.ARMED)
        log.info("[Vox Initialized] armed detector=%s patience=%d threshold=%.2f; say %r to start",
                 gate.detector.model_name, gate.detector.patience_frames, gate.config.threshold,
                 self.config.trigger_phrase)
        self._publish_state(WakeState.ARMED)

        # Resident models before the first wake: the reply model and the talker.
        asyncio.create_task(p.delegate.warm(), name="delegate-warm")
        asyncio.create_task(p.talker.warm(), name="talker-warm")

        await p.audio_in.start()
        audio_q = p.audio_in.subscribe()
        try:
            await self._armed(p, gate, audio_q)
        finally:
            p.audio_in.unsubscribe(audio_q)
            await p.audio_in.stop()

    # ------------------------------------------------------------------ capture + wake

    async def _armed(self, p: VoicePorts, gate: WakeGate, audio_q) -> None:
        """Stages capture and recognize while armed: 80 ms frames through the wake gate.

        Every candidate is recorded with its score and reject reason; a rejected one plays no
        sound and opens no session.
        """
        frame_buf = np.zeros(0, dtype=np.float32)
        out = p.audio_out
        while True:
            if self.state == WakeState.OFF:
                log.info("voice host: disarmed -- closing mic")
                return

            chunk = await audio_q.get()
            if self._capture_reset_pending:
                self._capture_reset_pending = False
                gate.reset()
                frame_buf = np.zeros(0, dtype=np.float32)
            rms = _rms(chunk)

            # Barge-in: energy over the playback estimate stops playback.
            if out.is_playing and rms > RMS_FLOOR:
                ref = out.playback_ref()
                if ref.size:
                    pb_rms = _rms(ref)
                    if rms > BARGE_RATIO * max(pb_rms, MIN_PLAYBACK_RMS):
                        log.info("voice host: barge-in mic_rms=%.4f pb_rms=%.4f", rms, pb_rms)
                        out.stop_playback()
            # No wake detection while the speaker plays: it would hear itself.
            if out.is_playing:
                continue

            frame_buf = np.concatenate([frame_buf, np.asarray(chunk, dtype=np.float32)])
            while frame_buf.shape[0] >= FRAME_SAMPLES:
                frame, frame_buf = frame_buf[:FRAME_SAMPLES], frame_buf[FRAME_SAMPLES:]
                # Off the loop: a candidate frame includes the voice check and the confirm pass.
                decision = await asyncio.to_thread(gate.process, frame)

                tele = gate.pop_telemetry()
                if tele is not None:
                    log.info("wake: peak_score=%.3f frames=%d peak_rms=%.4f",
                             tele.peak_score, tele.frames, tele.peak_rms)
                if decision is None:
                    continue

                ratio = ("n/a" if decision.speech_ratio is None
                         else f"{decision.speech_ratio:.2f}")
                if not decision.accepted:
                    log.info("wake: candidate rejected detector=%s score=%.3f reason=%s "
                             "speech_ratio=%s", decision.detector, decision.score,
                             decision.reject_reason, ratio)
                    await p.store.record_wake(score=decision.score, accepted=False,
                                              reject_reason=decision.reject_reason,
                                              detector=decision.detector,
                                              speech_ratio=decision.speech_ratio)
                    continue

                # Quiet hours: reject the wake when the window is active AND this mode
                # blocks wakes during quiet hours.  Senior care never blocks a wake
                # (quiet_hours_block_wake=False for that mode; VX-H5).
                quiet = self.config.quiet_hours
                if (quiet is not None and quiet.active()
                        and self.config.quiet_hours_block_wake):
                    log.info("wake: rejected (quiet hours) score=%.3f", decision.score)
                    await p.store.record_wake(score=decision.score, accepted=False,
                                              reject_reason=QUIET_HOURS_REASON,
                                              detector=decision.detector,
                                              speech_ratio=decision.speech_ratio)
                    frame_buf = np.zeros(0, dtype=np.float32)
                    gate.reset()    # or the same phrase scores again on the next frames
                    continue

                if self.state == WakeState.LISTENING:
                    log.info("voice host: already listening -- wake ignored")
                    continue

                log.info("wake-word-listener: DETECTED detector=%s score=%.3f speech_ratio=%s",
                         decision.detector, decision.score, ratio)

                # Keep "hey vox, how are you" in one breath: the audio after the phrase, what is
                # left in the frame buffer and what queued during the confirm all predate the
                # chime and go to the session as pre-roll. The chime then fires without being
                # awaited; capture is muted for it and its decay (the adapter pre-arms the echo
                # guard), so the session drops what the mic hears of it.
                pre_roll = np.concatenate([decision.pre_roll, frame_buf,
                                           *_take_queued(audio_q)])
                frame_buf = np.zeros(0, dtype=np.float32)
                self._play_cue(CHIME_ACK)

                self._publish_state(WakeState.LISTENING)
                try:
                    await self.run_session(PreRollQueue(pre_roll, audio_q),
                                           score=decision.score, detector=decision.detector,
                                           speech_ratio=decision.speech_ratio)
                except Exception:
                    log.exception("voice host: session failed")
                    await out.play_file(CHIME_ERR)
                finally:
                    self.set_state(WakeState.ARMED)
                    self._publish_state(WakeState.ARMED)
                    log.info("[Vox Stopped] back to armed; say %r to start a new conversation",
                             self.config.trigger_phrase)
                    _drain(audio_q)
                    frame_buf = np.zeros(0, dtype=np.float32)
                    gate.reset()    # detector state and held audio predate the session

    # ------------------------------------------------------------------ session

    async def run_session(self, audio_q, *, score: float, detector: str = TRIGGER_DETECTOR,
                          speech_ratio: float | None = None) -> None:
        """One conversation: many turns until a stop phrase, an empty turn or idle timeout.

        Silent close (T03): when turn 1 hears no genuine speech the meeting row is deleted (the
        wake_events row is kept for audit with its link cleared) and nothing is stored."""
        p = await self._ready()
        self.set_state(WakeState.LISTENING)
        try:
            p.speech.reset()
            use_turns = True
            log.info("voice host: using the turn detector")
        except RuntimeError:
            use_turns = False
            log.info("voice host: turn detector unavailable, falling back to silence counter")

        session_id = await p.store.create_session(score, detector=detector,
                                                   speech_ratio=speech_ratio)
        session = SessionContext(session_id=session_id)
        self._state["meeting_id"] = session_id
        log.info("[Vox Initialized] meeting_id=%s", session_id)

        turn_n = 0
        # T15: per-session pacing, length and level state.
        from app.local.voice.controls import ControlState   # lazy: boundary test
        from app.local.voice.memory import MemoryStore      # lazy: boundary test
        control_state = ControlState()
        # T15: last reply text, for the REPEAT intent.  Stored as a one-element list so
        # _handle_control() can read it without an extra method parameter.
        last_reply: list[str] = [""]
        # T19: in-session fact store, cleared at session close.
        memory = MemoryStore()
        max_turns = self.config.max_turns  # 0 means no limit
        try:
            while True:
                turn_n += 1
                if max_turns > 0 and turn_n > max_turns:
                    log.info("voice host: max turns reached limit=%d session=%s",
                             max_turns, session_id)
                    break
                if not await self._turn(p, audio_q, session_id, turn_n, use_turns, session,
                                        control_state, last_reply, memory):
                    break
        finally:
            self._state["meeting_id"] = None
            await self._await_late_finals(session)
            if session.silent:
                log.info("wake: silent close -- no genuine speech in turn 1 meeting_id=%s",
                         session_id)
                # discard_session extends the SessionStore port in this server; a store
                # without it closes the session normally.
                discard = getattr(p.store, "discard_session", p.store.close_session)
                await discard(session_id)
            else:
                await p.store.close_session(session_id)
                log.info("voice host: session closed meeting_id=%s turns=%d",
                         session_id, turn_n)

    async def _turn(self, p: VoicePorts, audio_q, session_id: str, turn_n: int,
                    use_turns: bool, session: SessionContext,
                    control_state=None, last_reply: list[str] | None = None,
                    memory=None) -> bool:
        """One turn: listen, recognise, reply. False when the conversation is over."""
        history = session.history
        stream = p.recognizer.open_stream(session_id)
        ring = _TurnRing(int(TURN_RING_S * SAMPLE_RATE / CHUNK_FRAMES))
        texts: list[str] = []
        partial = StablePartial()
        recognize = asyncio.create_task(
            self._recognize(p, ring, stream, session_id, turn_n, texts, partial, use_turns),
            name="voice-recognize")

        # After a barge-in the interjection has already started: its first audio was captured
        # while the reply played, and opens this turn.
        listen_q = audio_q
        if session.pre_roll is not None and session.pre_roll.size:
            listen_q = PreRollQueue(session.pre_roll, audio_q)
        session.pre_roll = None

        if await self._listen(p, listen_q, ring, turn_n, use_turns):
            if turn_n > 1:
                log.info("[Vox Stopped] idle timeout after %d turn(s)", turn_n - 1)
            else:
                log.info("[Vox Stopped] first-turn audio timeout")
                session.silent = True
            ring.close()
            recognize.cancel()
            return False

        turn_end_ms = p.clock.mono_ms()

        # Start the reply model on the text recognised so far, so generation overlaps the
        # final decode; kept only if the decode adds function words (spec_text_changed).
        # Not after a barge-in: the speculative stream has no system prompt to carry the
        # interruption note.
        prefetch = " ".join(texts).strip()
        speculation = None
        if (not session.barge_in and len(prefetch.split()) >= SPEC_MIN_WORDS
                and not is_stop_phrase(prefetch, self.config.wake_phrase)):
            speculation = p.delegate.speculate(prefetch, history or None)
            log.info("voice host: speculative reply started words=%d", len(prefetch.split()))

        # Fast finalize (T10): the end cue plays off the hot path and the final decode starts
        # at once on the audio buffered at close. It gets FINALIZE_TIMEOUT_MS; past that the
        # stable partial stands in, and the decode keeps running and is stored when it lands.
        self._play_cue(CHIME_END)
        final = asyncio.create_task(
            self._finalize(p, ring, recognize, stream, session_id, turn_n, texts),
            name="voice-finalize")
        try:
            no_speech_prob = await asyncio.wait_for(asyncio.shield(final),
                                                    timeout=FINALIZE_TIMEOUT_MS / 1000)
            turn_text = " ".join(texts).strip()
            source = "final"
        except asyncio.TimeoutError:
            session.late_finals.append(final)
            no_speech_prob = None
            turn_text = " ".join([*texts, partial.text]).strip()
            source = "partial"
        asr_final_ms = p.clock.mono_ms()
        log.info("voice host: asr final +%d ms from turn end source=%s turn=%d",
                 asr_final_ms - turn_end_ms, source, turn_n)

        if not turn_text and session.barge_in and not session.held_empty:
            # A cough or a door can interrupt a reply; it should not also end the conversation.
            session.held_empty = True
            log.info("[Vox Listening] the interruption held no words (turn=%d); listening on",
                     turn_n)
            return True
        if not turn_text:
            log.info("[Vox Stopped] empty turn, no speech detected (turn=%d)", turn_n)
            session.silent = turn_n == 1
            if speculation is not None:
                speculation.cancel()
            return False
        if is_stop_phrase(turn_text, self.config.wake_phrase):
            log.info("[Vox Stopped] stop phrase detected (turn=%d)", turn_n)
            if speculation is not None:
                speculation.cancel()
            return False
        # Speech guard (T03): chime echoes, sub-threshold audio, one-word transcripts and
        # decodes Whisper scores as not speech get no reply.
        if not is_genuine_speech(turn_text, ring.pushed_ms, no_speech_prob):
            log.info("wake: speech guard blocked turn=%d words=%d audio_ms=%d no_speech=%s",
                     turn_n, len(turn_text.split()), ring.pushed_ms,
                     "n/a" if no_speech_prob is None else f"{no_speech_prob:.2f}")
            session.silent = turn_n == 1
            if speculation is not None:
                speculation.cancel()
            return False
        if speculation is not None:
            if spec_text_changed(prefetch, turn_text):
                log.info("voice host: speculation cancelled, content drift")
                speculation.cancel()
                speculation = None
            else:
                log.info("voice host: speculation accepted, no content drift")

        # T15: check for a spoken control intent before routing to the LLM.
        from app.local.voice.controls import ControlIntent, detect_control  # lazy
        intent = detect_control(turn_text)
        if intent != ControlIntent.NONE:
            if speculation is not None:
                speculation.cancel()
            log.info("voice host: control intent=%s turn=%d", intent.value, turn_n)
            return await self._handle_control(
                p, intent, control_state, last_reply or [""])

        # The reply generates and plays while the barge-in watcher reads capture; an
        # interruption ends both, and its audio opens the next turn.
        self._publish("wake.reply", {"meetingId": session_id, "turn": turn_n,
                                     "phase": "thinking", "userText": turn_text, "text": ""})
        note = interruption_note(session.played_text) if session.barge_in else None
        result = await self._reply(session_id, turn_n, turn_text, history=history or None,
                                   speculation=speculation, turn_end_ms=turn_end_ms,
                                   asr_final_ms=asr_final_ms, audio_q=audio_q, note=note,
                                   control_state=control_state, memory=memory)
        interrupted = result.interruption is not None
        self._publish("wake.reply", {"meetingId": session_id, "turn": turn_n,
                                     "phase": "done", "userText": turn_text,
                                     "text": result.text, "interrupted": interrupted})
        history.append({"role": "user", "content": turn_text})
        if result.text:
            history.append({"role": "assistant", "content": result.text})
        session.barge_in = interrupted
        session.played_text = result.text if interrupted else ""
        session.held_empty = False
        if interrupted:
            session.pre_roll = result.pre_roll
        else:
            _drain(audio_q)
        if result.text and last_reply is not None:
            last_reply[0] = result.text
        return True

    async def _handle_control(self, p: VoicePorts, intent, control_state, last_reply: list[str]) -> bool:
        """Execute a spoken control intent. Returns True (keep session open) always."""
        from app.local.voice.controls import ACK_CUE, ControlIntent  # lazy
        if intent == ControlIntent.HOLD:
            try:
                await p.audio_out.play_file(CHIME_END)
            except Exception:
                log.debug("voice host: hold cue failed (non-critical)")
            return True

        if intent == ControlIntent.REPEAT:
            if last_reply[0]:
                try:
                    pcm = await p.talker.synthesize(last_reply[0])
                    await p.audio_out.play(pcm, p.talker.sample_rate)
                except Exception:
                    log.exception("voice host: repeat synthesis failed")
            else:
                try:
                    await p.audio_out.play_file(CHIME_ACK)
                except Exception:
                    log.debug("voice host: repeat chime failed (non-critical)")
            return True

        if control_state is not None:
            if intent == ControlIntent.SLOWER:
                control_state.apply_slower()
            elif intent == ControlIntent.FASTER:
                control_state.apply_faster()
            elif intent == ControlIntent.SHORTER:
                control_state.apply_shorter()
            elif intent == ControlIntent.LONGER:
                control_state.apply_longer()
            elif intent == ControlIntent.VOLUME_UP:
                control_state.apply_volume_up()
            elif intent == ControlIntent.VOLUME_DOWN:
                control_state.apply_volume_down()
            log.info("voice host: control applied speed=%.2f volume=%.2f suffix=%r",
                     control_state.speed, control_state.volume, control_state.prompt_suffix)
        try:
            pcm = await p.talker.synthesize(ACK_CUE)
            await p.audio_out.play(pcm, p.talker.sample_rate)
        except Exception:
            log.debug("voice host: control ack failed (non-critical)")
        return True

    async def _finalize(self, p: VoicePorts, ring: _TurnRing, recognize: asyncio.Task, stream,
                        session_id: str, turn_n: int, texts: list[str]) -> float | None:
        """Drain the recognizer task, then the final decode of what it still buffers. Stores
        and appends the final segments; returns the decode's no_speech_prob when known."""
        ring.close()
        try:
            await asyncio.wait_for(recognize, timeout=RECOGNIZE_DRAIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            recognize.cancel()
            log.warning("voice host: recognizer drain timeout turn=%d", turn_n)
        if stream is None:
            return None
        try:
            for seg in await asyncio.to_thread(stream.flush):
                await self._segment(p, session_id, turn_n, seg, texts)
        except Exception:
            log.warning("voice host: final decode failed turn=%d", turn_n)
            return None
        return getattr(stream, "no_speech_prob", None)

    async def _await_late_finals(self, session: SessionContext) -> None:
        """Final decodes that passed the finalize window land before the session closes."""
        if not session.late_finals:
            return
        done, pending = await asyncio.wait(session.late_finals,
                                           timeout=RECOGNIZE_DRAIN_TIMEOUT_S)
        for task in pending:
            task.cancel()
        log.info("voice host: late final decodes stored=%d dropped=%d meeting_id=%s",
                 len(done), len(pending), session.session_id)

    async def _listen(self, p: VoicePorts, audio_q, ring: _TurnRing, turn_n: int,
                      use_turns: bool) -> bool:
        """Stage turn: feed capture to the recognizer and the turn policy until the turn ends.

        Returns True when the turn timed out (no audio, or idle past the deadline)."""
        silence_needed = int(SILENCE_S * SAMPLE_RATE / CHUNK_FRAMES)
        cap_chunks = int(CAP_S * SAMPLE_RATE / CHUNK_FRAMES)
        conv_idle_s = self.config.conv_idle_s
        idle_deadline = time.monotonic() + ((SILENCE_S + FIRST_TURN_GRACE_S) if turn_n == 1
                                            else conv_idle_s)
        if use_turns:
            p.speech.reset()
        p.clock.reset()
        silence_run = 0
        chunks = 0
        clock_error_logged = False
        pre_roll_pending = getattr(audio_q, "pre_roll_pending", lambda: False)
        log.info("[Vox Listening] turn=%d (say your question, or %r to stop)", turn_n, STOP_HINT)
        try:
            while True:
                remaining = idle_deadline - time.monotonic()
                if remaining <= 0:
                    return True
                from_pre_roll = pre_roll_pending()
                try:
                    chunk = await asyncio.wait_for(audio_q.get(),
                                                   timeout=min(remaining, QUEUE_POLL_S))
                except asyncio.TimeoutError:
                    return True

                rms = _rms(chunk)
                if turn_n > 1 and rms > RMS_FLOOR * IDLE_SPEECH_RMS_FACTOR:
                    idle_deadline = time.monotonic() + conv_idle_s

                # Echo guard: our own playback, cues and their tails never reach the recognizer.
                # Pre-roll predates the acknowledgement chime and is never muted.
                if not from_pre_roll and p.audio_in.capture_muted():
                    silence_run = 0
                    continue

                ring.push(chunk)
                chunks += 1

                if use_turns:
                    p.clock.advance(len(chunk))
                    decisions: list[TurnDecision] = p.speech.process(chunk)
                    if chunks % TICK_EVERY_FRAMES == 0:
                        try:
                            decisions = p.speech.tick(p.clock.stream_ms()) or decisions
                        except ClockDomainError:
                            if not clock_error_logged:
                                log.error("voice host: turn clock is not stream time; pause "
                                          "decision skipped turn=%d", turn_n, exc_info=True)
                                clock_error_logged = True
                    d = decisions[-1] if decisions else None
                    if d is not None and d.closes_turn:
                        log.info("wake: turn %d closed action=%s silence_ms=%d",
                                 turn_n, d.action, d.silence_ms)
                        return False
                else:
                    silence_run = silence_run + 1 if rms < RMS_FLOOR else 0
                    if silence_run >= silence_needed:
                        log.info("wake: turn %d end-of-silence chunks=%d", turn_n, chunks)
                        return False

                if chunks >= cap_chunks:
                    log.info("wake: turn %d hard cap chunks=%d", turn_n, chunks)
                    return False
        except Exception:
            log.exception("voice host: audio collection error turn=%d", turn_n)
            return True

    async def _recognize(self, p: VoicePorts, ring: _TurnRing, stream, session_id: str,
                         turn_n: int, texts: list[str], partial: StablePartial,
                         use_turns: bool) -> None:
        """Stage recognize during a turn: stable segments are stored and fed to the policy;
        partials are published and tracked for the fast-finalize fallback."""
        while True:
            chunk = await ring.pop()
            if chunk is None:
                return
            if stream is None:
                continue
            try:
                stable = await asyncio.to_thread(stream.push, chunk)
                for seg in stable:
                    self._publish_segment(session_id, turn_n, seg)
                if stable:
                    partial.clear()
                for seg in stream.pop_partials():
                    self._publish_segment(session_id, turn_n, seg)
                    if seg.text:
                        partial.update(seg.text)
                for seg in stable:
                    await p.store.add_segment(session_id, seg)
                    if seg.text:
                        texts.append(seg.text)
                        if use_turns:
                            try:
                                p.speech.update_text(" ".join(texts))
                            except Exception:
                                pass
            except Exception:
                log.exception("voice host: transcription error")

    async def _segment(self, p: VoicePorts, session_id: str, turn_n: int,
                       seg: TranscriptSegment, texts: list[str]) -> None:
        self._publish_segment(session_id, turn_n, seg)
        await p.store.add_segment(session_id, seg)
        if seg.text:
            texts.append(seg.text)

    def _publish_segment(self, session_id: str, turn_n: int, seg: TranscriptSegment) -> None:
        if not seg.text:
            return
        self._publish("wake.transcript", {
            "meetingId": session_id, "turn": turn_n,
            "segment": {"id": seg.id, "text": seg.text, "state": seg.state,
                        "startMs": seg.start_ms, "endMs": seg.end_ms},
        })

    # ------------------------------------------------------------------ reply + playback

    async def reply_turn(self, session_id: str, turn_n: int, text: str, *,
                         history: list[dict] | None = None, speculation=None,
                         turn_end_ms: int = 0, asr_final_ms: int | None = None,
                         control_state=None, memory=None) -> str:
        """Stages reply and playback for one turn, without the barge-in watch. Returns the text
        recorded as spoken."""
        result = await self._reply(session_id, turn_n, text, history=history,
                                   speculation=speculation, turn_end_ms=turn_end_ms,
                                   asr_final_ms=asr_final_ms,
                                   control_state=control_state, memory=memory)
        return result.text

    def _barge_in_enabled(self, p: VoicePorts) -> bool:
        # Capture energy is the user's voice only when the speaker is cancelled out of it.
        return bool(getattr(p.audio_in, "echo_cancelled", False))

    async def _reply(self, session_id: str, turn_n: int, text: str, *,
                     history: list[dict] | None = None, speculation=None,
                     turn_end_ms: int = 0, asr_final_ms: int | None = None,
                     audio_q=None, note: str | None = None,
                     control_state=None, memory=None) -> ReplyResult:
        """Stages reply, playback and barge-in for one turn.

        Sentences are split into clauses (T11) so synthesis of clause N+1 starts while N plays.
        With `audio_q` (and an echo-cancelled capture) barge-in is watched during playback.
        `note` appended to the system prompt (barge-in); `control_state`/`memory` build prefix (T15/T19).
        """
        p = await self._ready()
        try:
            log.info("[Vox Thinking] turn=%d words=%d history_turns=%d barge_in=%s", turn_n,
                     len(text.split()), len(history) // 2 if history else 0, note is not None)

            # T19: extract facts from this turn, then build the [Memory] block.
            system_prefix: str | None = None
            if memory is not None:
                from app.local.voice.memory import memory_block  # lazy: boundary test
                memory.extract(text, turn_n)
                mems = memory.retrieve()
                if mems:
                    system_prefix = memory_block(mems)
                    log.info("voice host: memory block facts=%d turn=%d", len(mems), turn_n)

            # T15: append the prompt suffix from spoken length controls.
            if control_state is not None and control_state.prompt_suffix:
                suffix = control_state.prompt_suffix
                system_prefix = f"{system_prefix}\n\n{suffix}" if system_prefix else suffix

            turn = await p.store.begin_turn(session_id, turn_n, text)
            # PL09: every gate evaluation in this reply is journalled for the gate records, and
            # a gate that raises or answers late fails the reply closed.
            from app.local.voice.adapters.gate_policy import TurnGate  # lazy: boundary test
            turn_gate = TurnGate.for_turn(p.gate, turn)
            extra: dict = {}
            if note:
                extra["system_note"] = note
            if system_prefix:
                extra["system_prefix"] = system_prefix
            stream = await p.delegate.reply(text, session_id, turn_id=turn.turn_id,
                                            history=history, speculation=speculation, **extra)

            # T11: split into clauses so synthesis of clause N+1 starts while N plays.
            clause_stream = ReplyStream(
                sentences=_clause_sentence_stream(stream.sentences),
                source=stream.source,
                cue=turn_gate.scripted(stream.cue),          # scripted lines pass the gate too
                lead_in=turn_gate.scripted(stream.lead_in),
                fallback=stream.fallback,
            )
            clause_talker = _ClauseTalker(p.talker, p.clock)

            # T13: wrap with HeardReply so an interruption records only played clauses.
            heard = HeardReply(p.clock, scripted=(clause_stream.cue, clause_stream.lead_in))
            sentences = heard.sentences(clause_stream.sentences)
            clause_stream.sentences = sentences
            speak = asyncio.create_task(speak_reply(
                clause_stream, gate=heard.gate(turn_gate), talker=heard.talker(clause_talker),
                audio_out=heard.audio_out(p.audio_out), clock=p.clock, user_text=text,
                session_tools=turn_gate.tools(lambda: p.store.session_tools(session_id))),
                name="voice-reply")

            pre_roll = None
            try:
                if audio_q is not None and self._barge_in_enabled(p):
                    pre_roll = await self._watch_barge_in(p, audio_q, speak)
                completed = await self._settle(speak, interrupted=pre_roll is not None)
            finally:
                if not speak.done():
                    speak.cancel()

            interruption = None
            if pre_roll is not None:
                await sentences.aclose()
                if speculation is not None:
                    speculation.cancel()
                interruption = heard.interruption()
                outcome = heard.outcome(completed, end_ms=p.clock.mono_ms())
                log.info("[Vox Interrupted] turn=%d heard=%d of %d clause(s)", turn_n,
                         len(heard.played_clauses), len(heard.released))
            else:
                outcome = completed
            if outcome.first_audio_ms is not None:
                log.info("voice host: first audio +%d ms from turn end",
                         outcome.first_audio_ms - turn_end_ms)
            extra_turn = {"interruption": interruption} if interruption is not None else {}
            await p.store.add_turn(turn, outcome, turn_end_ms=turn_end_ms,
                                   asr_final_ms=asr_final_ms, **extra_turn)
            log.info("[Vox Ready] turn=%d reply %s decision=%s", turn_n,
                     "interrupted" if interruption else "complete", outcome.decision)
            return ReplyResult(text=outcome.text, interruption=interruption,
                               pre_roll=pre_roll)
        except Exception:
            log.exception("voice host: reply failed meeting_id=%s", session_id)
            return ReplyResult(text="")

    @staticmethod
    async def _settle(speak: asyncio.Task, *, interrupted: bool):
        """The reply stage's outcome; after an interruption the stage is cancelled if it is still
        running, and None is returned for it."""
        if interrupted and not speak.done():
            speak.cancel()
        try:
            return await speak
        except asyncio.CancelledError:
            if not interrupted:
                raise
            return None

    async def _watch_barge_in(self, p: VoicePorts, audio_q, speak: asyncio.Task):
        """Stage barge-in: read capture until the reply has been generated and played.

        Returns the capture held from just before the interruption (the start of the
        interjection, for the next turn), or None when the reply played to its end.
        Playback is stopped here, on the frame that decides, before anything else runs.
        """
        out = p.audio_out
        detector = BargeInDetector(SAMPLE_RATE)
        held: deque = deque(maxlen=max(1, BARGE_IN_PRE_ROLL_MS * SAMPLE_RATE
                                       // (1000 * CHUNK_FRAMES)))
        while not (speak.done() and not out.is_playing):
            try:
                chunk = await asyncio.wait_for(audio_q.get(), timeout=BARGE_IN_POLL_S)
            except asyncio.TimeoutError:
                continue
            held.append(chunk)
            for event in detector.feed(chunk, playing=out.is_playing):
                if event == BARGE_IN:
                    out.stop_playback()
                    log.info("voice host: barge-in, playback stopped")
                    return np.concatenate([np.asarray(c, dtype=np.float32) for c in held])
                if event in (BACKCHANNEL, OVERLAP):
                    log.info("voice host: %s during reply, keeping the floor", event)
        return None

    # ------------------------------------------------------------------ hooks

    async def trigger(self) -> str | None:
        """Debug trigger: start a session without a wake. Returns the session id once known."""
        if self.ports is None:
            raise RuntimeError("voice host is not ready")
        p = self.ports
        audio_q = p.audio_in.subscribe()

        async def _session() -> None:
            try:
                await self.run_session(audio_q, score=1.0)
            except Exception:
                log.exception("voice host: triggered session failed")
            finally:
                # The trigger path never goes through the armed loop, so reset here.
                self.set_state(WakeState.ARMED)
                p.audio_in.unsubscribe(audio_q)

        asyncio.create_task(_session(), name="trigger-session")
        for _ in range(TRIGGER_POLL_TRIES):
            await asyncio.sleep(TRIGGER_POLL_S)
            if self._state.get("meeting_id"):
                return self._state["meeting_id"]
        return None

    def on_capture_health(self, event) -> None:
        """CaptureHealth from the capture watchdog. After a reopen the gate is reset on the next
        frame, since the detector still holds features from before the gap."""
        if event.action == "reopened":
            self._capture_reset_pending = True
            log.info("voice host: capture recovered issue=%s; wake gate reset on next frame",
                     event.issue)
        elif event.action == "reopen_failed":
            log.warning("voice host: capture reopen failed issue=%s; watchdog will retry",
                        event.issue)
        self._publish("capture_health", {"issue": event.issue, "action": event.action,
                                         "value": event.value})
