"""Fakes for the ten session ports, for host tests. No mic, model, database or network."""
from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Sequence

import numpy as np
from voxscribe.session.ports import (
    GATE_PASS,
    GateDecision,
    ReplyOutcome,
    ReplyStream,
    TranscriptionResult,
    TranscriptSegment,
    TurnDecision,
    TurnRecord,
)

from app.local.voice.host import (
    CHUNK_FRAMES,
    MIN_AUDIO_MS,
    SAMPLE_RATE,
    TICK_EVERY_FRAMES,
    StreamClock,
    VoicePorts,
)

TALKER_RATE = 24_000
SAMPLES_PER_CHAR = 10
# FakeSpeech closes a turn after this many chunks: enough audio for the speech guard, plus one
# turn-detector tick of margin.
FRAMES_PER_TURN = -(-MIN_AUDIO_MS * SAMPLE_RATE // (1000 * CHUNK_FRAMES)) + TICK_EVERY_FRAMES
TEN_HOURS_MS = 10 * 3600 * 1000


def chunk(value: float = 0.0) -> np.ndarray:
    return np.full(CHUNK_FRAMES, value, dtype=np.float32)


class EndlessQueue:
    """A capture subscription that never runs dry and never has a backlog to drain."""

    def __init__(self) -> None:
        self.reads = 0

    async def get(self):
        await asyncio.sleep(0)
        self.reads += 1
        return chunk()

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def empty(self) -> bool:
        return True


class FakeAudioIn:
    sample_rate = SAMPLE_RATE
    echo_cancelled = True          # capture energy during a reply is the user (T13 barge-in)

    def __init__(self) -> None:
        self.queues: list = []
        self.muted = False
        self.starts = self.stops = 0

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1

    def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        self.queues.append(q)
        return q

    def unsubscribe(self, queue) -> None:
        if queue in self.queues:
            self.queues.remove(queue)

    def capture_muted(self) -> bool:
        return self.muted


class FakeAudioOut:
    available = True

    def __init__(self) -> None:
        self.played: list[int] = []
        self.files: list[str] = []
        self.stops = 0

    async def play(self, pcm_float32_mono, sample_rate: int) -> None:
        assert sample_rate == TALKER_RATE
        self.played.append(len(pcm_float32_mono))

    async def play_file(self, path: str) -> None:
        self.files.append(path)

    def stop_playback(self) -> None:
        self.stops += 1

    @property
    def is_playing(self) -> bool:
        return False

    def playback_ref(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class FakeWake:
    model_name = "fake:hey_vox"
    patience_frames = 1

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def score(self, pcm_float32_mono) -> float:
        return 0.0


class _FakeStream:
    def __init__(self, text: str, start_ms: int = 0) -> None:
        self.text = text
        self.start_ms = start_ms
        self.pushed = 0

    def push(self, pcm_float32_mono) -> list[TranscriptSegment]:
        self.pushed += 1
        return []

    def pop_partials(self) -> list[TranscriptSegment]:
        return []

    def flush(self) -> list[TranscriptSegment]:
        if not self.text:
            return []
        return [TranscriptSegment(id=str(uuid.uuid4()), text=self.text, start_ms=self.start_ms,
                                  end_ms=self.start_ms + self.pushed * 32)]


class FakeRecognizer:
    """One scripted text per turn; a turn past the script hears nothing."""

    def __init__(self, turns: Sequence[str] = ()) -> None:
        self.turns = list(turns)
        self.opened = 0

    def transcribe(self, pcm_float32_mono) -> TranscriptionResult:
        return TranscriptionResult(text="hey vox")

    def open_stream(self, session_id: str) -> _FakeStream:
        text = self.turns[self.opened] if self.opened < len(self.turns) else ""
        self.opened += 1
        return _FakeStream(text, start_ms=self.opened * 1000)


class FakeTalker:
    sample_rate = TALKER_RATE

    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.fixed_phrases: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.synthesized.append(text)
        return np.zeros(len(text) * SAMPLES_PER_CHAR, dtype=np.float32)

    async def fixed(self, phrase: str) -> np.ndarray:
        self.fixed_phrases.append(phrase)
        return np.zeros(len(phrase) * SAMPLES_PER_CHAR, dtype=np.float32)

    async def warm(self) -> None:
        pass


async def _sentences(items):
    for s in items:
        yield s


class FakeSpeculation:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def __aiter__(self):
        return _sentences(())


class FakeDelegate:
    """Replies "Answer <n>." to turn n unless a scripted reply is given."""

    def __init__(self, replies: Sequence[Sequence[str]] | None = None) -> None:
        self.replies = [list(r) for r in replies] if replies else None
        self.calls: list[str] = []
        self.notes: list[str | None] = []
        self.prefixes: list[str | None] = []
        self.histories: list[list[dict] | None] = []

    async def warm(self) -> None:
        pass

    def speculate(self, partial_text, history) -> FakeSpeculation:
        return FakeSpeculation()

    async def reply(self, turn_text, session_id, *, turn_id=None, history=None,
                    speculation=None, system_note=None, system_prefix=None) -> ReplyStream:
        self.calls.append(turn_text)
        self.notes.append(system_note)
        self.prefixes.append(system_prefix)
        self.histories.append([dict(m) for m in history] if history else None)
        n = len(self.calls)
        items = self.replies[n - 1] if self.replies else [f"Answer {n}."]
        return ReplyStream(sentences=_sentences(items), fallback="nothing")


class PauseSpeech:
    """Closes a turn on quiet: FRAMES_PER_TURN quiet chunks after the user spoke, or
    UNVOICED_CHUNKS_PER_TURN when nothing was voiced (the scripted recognizer supplies the text).
    The second is longer than the barge-in pre-roll, whose start is the quiet before the
    interjection."""

    def __init__(self) -> None:
        self.quiet = 0
        self.voiced = False

    def reset(self) -> None:
        self.quiet, self.voiced = 0, False

    def process(self, pcm_float32_mono) -> list[TurnDecision]:
        if float(np.abs(pcm_float32_mono).max(initial=0.0)) > 0:
            self.voiced, self.quiet = True, 0
        else:
            self.quiet += 1
        if self.quiet >= (FRAMES_PER_TURN if self.voiced else UNVOICED_CHUNKS_PER_TURN):
            return [TurnDecision(closes_turn=True, action="close", silence_ms=0)]
        return []

    def tick(self, stream_ms: int) -> list[TurnDecision]:
        return []

    def update_text(self, text: str) -> None:
        pass

    def voice_ratio(self, pcm_float32_mono) -> float:
        return 1.0


class FakeSpeech:
    """Closes a turn after FRAMES_PER_TURN chunks; records the time every tick was given."""

    def __init__(self, *, tick_raises: Exception | None = None) -> None:
        self.ticks: list[int] = []
        self.frames = 0
        self.resets = 0
        self.tick_raises = tick_raises

    def reset(self) -> None:
        self.resets += 1
        self.frames = 0

    def process(self, pcm_float32_mono) -> list[TurnDecision]:
        self.frames += 1
        if self.frames >= FRAMES_PER_TURN:
            return [TurnDecision(closes_turn=True, action="close", silence_ms=0)]
        return []

    def tick(self, stream_ms: int) -> list[TurnDecision]:
        self.ticks.append(stream_ms)
        if self.tick_raises is not None:
            raise self.tick_raises
        return []

    def update_text(self, text: str) -> None:
        pass

    def voice_ratio(self, pcm_float32_mono) -> float:
        return 1.0


class FakeGate:
    fail_closed_phrase = "Sorry, I can't say that."

    def __init__(self, *, refuse: str | None = None, raises: bool = False) -> None:
        self.refuse, self.raises = refuse, raises
        self.checked: list[str] = []

    def needs_session_tools(self, sentence: str) -> bool:
        return False

    def check(self, sentence, session_tools=(), *, user_text="") -> GateDecision:
        self.checked.append(sentence)
        if self.raises:
            raise RuntimeError("gate fault injected")
        if self.refuse and self.refuse in sentence:
            return GateDecision("refuse", "blocked", "I can't help with that.")
        return GateDecision(GATE_PASS)


class FakeStore:
    def __init__(self) -> None:
        self.wakes: list[dict] = []
        self.sessions: list[str] = []
        self.segments: list[tuple[str, str]] = []
        self.turns: list[tuple[int, ReplyOutcome]] = []
        self.interruptions: dict[int, object] = {}     # turn_n -> Interruption (T13)
        self.closed: list[str] = []
        self.discarded: list[str] = []

    async def record_wake(self, **kw) -> None:
        self.wakes.append(kw)

    async def create_session(self, wake_score, *, detector=None, speech_ratio=None) -> str:
        sid = f"s{len(self.sessions) + 1}"
        self.sessions.append(sid)
        self.wakes.append({"score": wake_score, "accepted": True, "detector": detector,
                           "session_id": sid})
        return sid

    async def add_segment(self, session_id, segment) -> None:
        self.segments.append((session_id, segment.text))

    async def begin_turn(self, session_id, turn_n, text) -> TurnRecord:
        return TurnRecord(session_id=session_id, turn_n=turn_n, turn_id=f"t{turn_n}")

    async def session_tools(self, session_id) -> list[str]:
        return []

    async def add_turn(self, turn, outcome, *, turn_end_ms, asr_final_ms=None,
                       interruption=None) -> None:
        self.turns.append((turn.turn_n, outcome))
        if interruption is not None:
            self.interruptions[turn.turn_n] = interruption

    async def close_session(self, session_id) -> None:
        self.closed.append(session_id)

    async def discard_session(self, session_id) -> None:
        """Silent close: the meeting row and its segments go, the wake row stays unlinked."""
        self.discarded.append(session_id)
        self.sessions.remove(session_id)
        self.segments = [s for s in self.segments if s[0] != session_id]
        for w in self.wakes:
            if w.get("session_id") == session_id:
                w["session_id"] = None


def shipped_profile(name: str = "personal"):
    """One of the product profiles the core wheel ships: personal, ideascribe, senior_care or
    classroom."""
    from pathlib import Path

    import voxscribe.profile as core_profile
    from voxscribe.profile.loader import load_profile

    return load_profile(Path(core_profile.__file__).parent / "profiles" / f"{name}.yaml")


def policy_gate(profile_name: str = "personal", settings=None):
    """The production gate adapter (core's policy engine, host floor behind it) for a shipped
    profile. Reads the pack from the wheel; no model, database or network."""
    from app.local.voice.adapters.gate_policy import CorePolicyGate, load_policy

    return CorePolicyGate(load_policy(shipped_profile(profile_name), settings))


def fake_ports(**over) -> VoicePorts:
    ports = dict(
        audio_in=FakeAudioIn(), audio_out=FakeAudioOut(), wake=FakeWake(),
        recognizer=FakeRecognizer(), talker=FakeTalker(), delegate=FakeDelegate(),
        speech=FakeSpeech(), gate=FakeGate(), store=FakeStore(),
        clock=StreamClock(mono=lambda: TEN_HOURS_MS),
    )
    ports.update(over)
    return VoicePorts(**ports)


# ── T13 barge-in simulation: one clock for capture and playback ──────────────
#
# The capture queue advances the clock one chunk per read and plays the user's voice from a
# script; the speaker plays queued clauses in real time on the same clock and, when stopped,
# goes silent at the next PLAYBACK_SLICE_MS boundary, as the AudioEngine does.

CHUNK_MS = CHUNK_FRAMES * 1000 // SAMPLE_RATE
SPEECH_MS_PER_CHAR = 60          # synthetic talker: about the pace of Kokoro at speed 1
UNVOICED_CHUNKS_PER_TURN = 3 * FRAMES_PER_TURN


class SimClock:
    def __init__(self) -> None:
        self.now_ms = 0

    def mono(self) -> int:
        return self.now_ms


class SimPlayback:
    """AudioOut on the sim clock. Reports each clause played to its end (reports_played)."""

    available = True
    reports_played = True

    def __init__(self, clock: SimClock) -> None:
        from app.local.audio_engine import PLAYBACK_SLICE_MS

        self.clock = clock
        self.slice_ms = PLAYBACK_SLICE_MS
        self.queue: list[tuple[float, object]] = []     # (duration_ms, on_played)
        self.head_started_ms: float | None = None
        self.played_ms = 0.0                            # playback time heard, all replies
        self.files: list[str] = []
        # (stop requested ms, silent from ms, reply audio heard by then in ms)
        self.stops: list[tuple[int, float, float]] = []
        self.started: list[float] = []                  # start time of every clause

    async def play(self, pcm_float32_mono, sample_rate: int, *, on_played=None) -> None:
        assert sample_rate == TALKER_RATE
        self.advance()
        if not self.queue:
            self.head_started_ms = self.clock.now_ms
            self.started.append(self.clock.now_ms)
        self.queue.append((len(pcm_float32_mono) * 1000 / sample_rate, on_played))

    async def play_file(self, path: str) -> None:
        self.files.append(path)

    def advance(self) -> None:
        now = self.clock.now_ms
        while self.queue and self.head_started_ms is not None:
            dur, done = self.queue[0]
            end = self.head_started_ms + dur
            if now < end:
                break
            self.queue.pop(0)
            self.played_ms += dur
            if done is not None:
                done()
            self.head_started_ms = end if self.queue else None
            if self.queue:
                self.started.append(end)

    def head_elapsed_ms(self) -> float:
        if not self.queue or self.head_started_ms is None:
            return 0.0
        return max(0.0, self.clock.now_ms - self.head_started_ms)

    def stop_playback(self) -> None:
        # The slice being written still plays; when it is the clause's last, the clause has
        # been played to its end (the engine reports it then too).
        now = self.clock.now_ms
        silent = float(now)
        if self.queue and self.head_started_ms is not None:
            dur, done = self.queue[0]
            into = now - self.head_started_ms
            heard = min(dur, math.ceil(into / self.slice_ms) * self.slice_ms)
            silent = self.head_started_ms + heard
            self.played_ms += heard
            if heard >= dur and done is not None:
                done()
        self.stops.append((now, silent, self.played_ms))
        self.queue.clear()
        self.head_started_ms = None

    @property
    def is_playing(self) -> bool:
        return bool(self.queue)

    def playback_ref(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class Burst:
    """The user speaking at `amp` RMS for `dur_ms`, starting once `at_playback_ms` of reply
    audio has been heard. `onset_ms` is when it started on the sim clock."""

    def __init__(self, at_playback_ms: float, dur_ms: float, amp: float) -> None:
        self.at, self.dur, self.amp = at_playback_ms, dur_ms, amp
        self.onset_ms: int | None = None

    def level(self, now_ms: int, played_ms: float) -> float:
        if self.onset_ms is None and played_ms >= self.at:
            self.onset_ms = now_ms
        if self.onset_ms is not None and now_ms < self.onset_ms + self.dur:
            return self.amp
        return 0.0


class SimCapture:
    """A capture subscription on the sim clock: each read is the next CHUNK_MS of the room."""

    def __init__(self, clock: SimClock, out: SimPlayback, bursts: Sequence[Burst] = ()) -> None:
        self.clock, self.out, self.bursts = clock, out, list(bursts)
        self.reads = 0

    async def get(self):
        await asyncio.sleep(0)
        self.out.advance()
        level = max((b.level(self.clock.now_ms, self.out.played_ms + self.out.head_elapsed_ms())
                     for b in self.bursts), default=0.0)
        self.clock.now_ms += CHUNK_MS
        self.out.advance()
        self.reads += 1
        return chunk(level)

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def empty(self) -> bool:
        return True


class SimTalker(FakeTalker):
    """Audio as long as the text takes to say, at SPEECH_MS_PER_CHAR."""

    async def synthesize(self, text: str) -> np.ndarray:
        self.synthesized.append(text)
        return np.zeros(len(text) * SPEECH_MS_PER_CHAR * TALKER_RATE // 1000, dtype=np.float32)

    async def fixed(self, phrase: str) -> np.ndarray:
        self.fixed_phrases.append(phrase)
        return np.zeros(len(phrase) * SPEECH_MS_PER_CHAR * TALKER_RATE // 1000,
                        dtype=np.float32)


def sim_ports(bursts: Sequence[Burst] = (), **over):
    """Ports on one sim clock; returns (ports, capture)."""
    clock = SimClock()
    out = SimPlayback(clock)
    capture = SimCapture(clock, out, bursts)
    over.setdefault("speech", PauseSpeech())
    ports = fake_ports(audio_out=out, talker=SimTalker(), clock=StreamClock(mono=clock.mono),
                       **over)
    return ports, capture
