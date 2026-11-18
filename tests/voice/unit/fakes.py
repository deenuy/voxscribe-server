"""Fakes for the ten session ports, for host tests. No mic, model, database or network."""
from __future__ import annotations

import asyncio
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

from app.local.voice.host import CHUNK_FRAMES, SAMPLE_RATE, StreamClock, VoicePorts

TALKER_RATE = 24_000
SAMPLES_PER_CHAR = 10
FRAMES_PER_TURN = 8          # FakeSpeech closes a turn after this many chunks
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

    async def warm(self) -> None:
        pass

    def speculate(self, partial_text, history) -> FakeSpeculation:
        return FakeSpeculation()

    async def reply(self, turn_text, session_id, *, turn_id=None, history=None,
                    speculation=None) -> ReplyStream:
        self.calls.append(turn_text)
        n = len(self.calls)
        items = self.replies[n - 1] if self.replies else [f"Answer {n}."]
        return ReplyStream(sentences=_sentences(items), fallback="nothing")


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
        self.closed: list[str] = []

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

    async def add_turn(self, turn, outcome, *, turn_end_ms, asr_final_ms=None) -> None:
        self.turns.append((turn.turn_n, outcome))

    async def close_session(self, session_id) -> None:
        self.closed.append(session_id)


def fake_ports(**over) -> VoicePorts:
    ports = dict(
        audio_in=FakeAudioIn(), audio_out=FakeAudioOut(), wake=FakeWake(),
        recognizer=FakeRecognizer(), talker=FakeTalker(), delegate=FakeDelegate(),
        speech=FakeSpeech(), gate=FakeGate(), store=FakeStore(),
        clock=StreamClock(mono=lambda: TEN_HOURS_MS),
    )
    ports.update(over)
    return VoicePorts(**ports)
