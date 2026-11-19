"""Fast finalize (T10) and the T03 speech guard, cue muting and silent close on the T08 host.

Fakes only: no mic, recognizer, reply model, talker or database.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace

import numpy as np
from voxscribe.session.ports import TranscriptSegment

from app.local.voice import host as host_mod
from app.local.voice.adapters import audio_portaudio
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import (
    CHIME_END,
    DRIFT_GATE_THRESHOLD,
    MIN_AUDIO_MS,
    PreRollQueue,
    StablePartial,
    VoiceConfig,
    VoiceHost,
    _TurnRing,
    is_genuine_speech,
    partial_drift,
)

from .fakes import (
    FRAMES_PER_TURN,
    EndlessQueue,
    FakeAudioOut,
    FakeRecognizer,
    _FakeStream,
    chunk,
    fake_ports,
)

SESSION_TIMEOUT_S = 10
SHORT_FINALIZE_MS = 50          # the finalize window in the timeout test
SLOW_FLUSH_S = 0.3              # how long the slow final decode takes, well past that window
FLUSH_WAIT_S = 5                # upper bound on a test's blocked final decode


def _seg(text: str, state: str = "stable") -> TranscriptSegment:
    return TranscriptSegment(id=str(uuid.uuid4()), text=text, start_ms=0, end_ms=0, state=state)


async def _session(ports, **config) -> VoiceHost:
    host = VoiceHost(VoiceConfig(**config), ports=ports)
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9), SESSION_TIMEOUT_S)
    return host


# ── end cue off the hot path ─────────────────────────────────────────────────


class _SlowCueOut(FakeAudioOut):
    """The end cue does not finish until the test says so; every step is logged in order."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.release = asyncio.Event()

    async def play_file(self, path: str) -> None:
        await super().play_file(path)
        if path == CHIME_END:
            self.log.append("cue started")
            await self.release.wait()
            self.log.append("cue finished")


class _LoggedStream(_FakeStream):
    def __init__(self, text: str, log: list[str]) -> None:
        super().__init__(text)
        self.log = log

    def flush(self):
        self.log.append("final decode")
        return super().flush()


class _LoggedRecognizer(FakeRecognizer):
    def __init__(self, turns, log: list[str]) -> None:
        super().__init__(turns)
        self.log = log

    def open_stream(self, session_id):
        s = super().open_stream(session_id)
        return _LoggedStream(s.text, self.log)


async def test_end_cue_not_awaited():
    log: list[str] = []
    out = _SlowCueOut(log)
    ports = fake_ports(audio_out=out, recognizer=_LoggedRecognizer(["What is one?"], log))
    host = await _session(ports)

    # The whole turn (final decode, reply) ran while the first end cue was still playing.
    assert log[:2] == ["cue started", "final decode"]
    assert "cue finished" not in log
    assert ports.delegate.calls == ["What is one?"]
    assert [o.text for _, o in ports.store.turns] == ["Answer 1."]

    out.release.set()
    await asyncio.gather(*host._cues)
    assert log.count("cue finished") == out.files.count(CHIME_END)


# ── bounded finalize ─────────────────────────────────────────────────────────


class _SlowFinalStream:
    """Partials while the turn runs; the final decode blocks until released."""

    def __init__(self, partials, final: str) -> None:
        self._partials = list(partials)
        self._final = final
        self.release = threading.Event()
        self.pushed = 0

    def push(self, pcm):
        self.pushed += 1
        return []

    def pop_partials(self):
        return [_seg(self._partials.pop(0), "provisional")] if self._partials else []

    def flush(self):
        self.release.wait(FLUSH_WAIT_S)
        return [_seg(self._final)]


class _OneSlowTurn(FakeRecognizer):
    def __init__(self, stream) -> None:
        super().__init__()
        self.slow = stream

    def open_stream(self, session_id):
        self.opened += 1
        return self.slow if self.opened == 1 else _FakeStream("")


async def test_finalize_timeout_uses_partial(monkeypatch):
    monkeypatch.setattr(host_mod, "FINALIZE_TIMEOUT_MS", SHORT_FINALIZE_MS)
    stream = _SlowFinalStream(["what is the", "what is the weather today",
                               "what was a feather today"], final="What is the weather today?")
    ports = fake_ports(recognizer=_OneSlowTurn(stream))
    asyncio.get_running_loop().call_later(SLOW_FLUSH_S, stream.release.set)
    await _session(ports)

    # The reply started on the last stable partial; the drifting one after it was ignored.
    assert ports.delegate.calls == ["what is the weather today"]
    # The late final decode was still stored before the session closed.
    assert ("s1", "What is the weather today?") in ports.store.segments
    assert ports.store.closed == ["s1"]


def test_partial_drift_gate():
    assert partial_drift("what is the", "what is the weather today") == 0.0
    assert partial_drift("what is the weather today", "what was a feather today") > \
        DRIFT_GATE_THRESHOLD
    p = StablePartial()
    for text in ("what is the", "what is the weather today", "what was a feather today"):
        p.update(text)
    assert p.text == "what is the weather today"
    p.clear()
    assert p.text == ""


# ── speech guard ─────────────────────────────────────────────────────────────


async def test_speech_guard_blocks_chime_transcript():
    assert not is_genuine_speech("Beep.", MIN_AUDIO_MS * 2)
    assert not is_genuine_speech("Hello", MIN_AUDIO_MS * 2)                  # one word
    assert not is_genuine_speech("What is one?", MIN_AUDIO_MS - 1)          # too little audio
    assert not is_genuine_speech("Thank you for watching.", MIN_AUDIO_MS * 2, no_speech_prob=0.9)
    assert is_genuine_speech("What is one?", MIN_AUDIO_MS * 2, no_speech_prob=0.05)

    ports = fake_ports(recognizer=FakeRecognizer(["Beep."]))
    await _session(ports)
    assert ports.delegate.calls == []
    assert ports.store.turns == []
    assert ports.talker.synthesized == []


class _NoSpeechStream(_FakeStream):
    no_speech_prob = 0.9


class _NoSpeechRecognizer(FakeRecognizer):
    def open_stream(self, session_id):
        s = super().open_stream(session_id)
        return _NoSpeechStream(s.text, s.start_ms)


async def test_speech_guard_uses_whisper_no_speech_prob():
    ports = fake_ports(recognizer=_NoSpeechRecognizer(["Thank you for watching."]))
    await _session(ports)
    assert ports.delegate.calls == []
    assert ports.store.discarded == ["s1"]


# ── silent close ─────────────────────────────────────────────────────────────


async def test_silent_close_no_meeting():
    ports = fake_ports(recognizer=FakeRecognizer(["Beep."]))
    host = await _session(ports)
    store = ports.store
    assert store.discarded == ["s1"] and store.closed == []
    assert store.sessions == [] and store.segments == [] and store.turns == []
    (wake,) = store.wakes
    assert wake["accepted"] is True and wake["session_id"] is None     # kept for audit
    assert host.snapshot()["meeting_id"] is None


async def test_guard_after_turn_one_closes_the_meeting_normally():
    ports = fake_ports(recognizer=FakeRecognizer(["What is one?", "Beep."]))
    await _session(ports)
    assert ports.delegate.calls == ["What is one?"]
    assert ports.store.closed == ["s1"] and ports.store.discarded == []


class _Conn:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, params=None):
        self.rows.append((sql, params))

    async def commit(self):
        pass


class _Pool:
    def __init__(self):
        self.rows: list = []

    def connection(self):
        rows = self.rows

        class _Ctx:
            async def __aenter__(self):
                return _Conn(rows)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


async def test_postgres_silent_close_unlinks_the_wake_and_deletes_the_meeting():
    pool = _Pool()
    store = PostgresSessionStore(SimpleNamespace(state=SimpleNamespace(
        store=SimpleNamespace(pool=pool))))
    await store.discard_session("m1")
    (unlink, unlink_args), (delete, delete_args) = pool.rows
    assert "UPDATE wake_events SET started_meeting_id=NULL" in unlink and unlink_args == ("m1",)
    assert "DELETE FROM meetings" in delete and delete_args == ("m1",)


# ── cue muting ───────────────────────────────────────────────────────────────


async def test_cue_pre_arms_the_echo_guard(monkeypatch):
    armed: list[int] = []
    engine = SimpleNamespace(echo_guard=SimpleNamespace(notify_chime=armed.append))
    monkeypatch.setattr(audio_portaudio.subprocess, "run", lambda *a, **kw: None)
    await audio_portaudio.PortAudioOut(engine).play_file(CHIME_END)
    # The cue's own length where the file can be read (macOS), the stand-in elsewhere.
    assert armed == [audio_portaudio.cue_duration_ms(CHIME_END) + audio_portaudio.CHIME_DECAY_MS]
    assert armed[0] >= 1500


async def test_pre_roll_is_never_muted_by_the_chime():
    ports = fake_ports()
    ports.audio_in.muted = True             # the chime is playing for the whole turn
    pre_roll_chunks = FRAMES_PER_TURN // 2   # too short to close the turn on its own
    live: asyncio.Queue = asyncio.Queue()
    for _ in range(FRAMES_PER_TURN):
        live.put_nowait(chunk())
    q = PreRollQueue(np.concatenate([chunk()] * pre_roll_chunks), live)
    ring = _TurnRing(FRAMES_PER_TURN * 2)
    host = VoiceHost(VoiceConfig(), ports=ports)
    # The live audio is all muted, so the turn never closes and ends when capture runs dry.
    assert await host._listen(ports, q, ring, 1, True) is True
    assert ring.pushed_samples == pre_roll_chunks * len(chunk())


# ── no_speech_prob from faster-whisper ───────────────────────────────────────


class _FakeWhisperModel:
    def __init__(self, probs) -> None:
        self.probs = probs

    def transcribe(self, audio, **kw):
        segs = (SimpleNamespace(text=f"word{i}", start=i, end=i + 1, avg_logprob=0.0,
                                no_speech_prob=p, words=None) for i, p in enumerate(self.probs))
        return segs, None


def test_whisper_provider_reports_no_speech_prob_for_the_measured_decode_only():
    from app.local.providers import _NoSpeechWhisperProvider

    provider = _NoSpeechWhisperProvider(model_size="tiny")
    model = provider._model = _FakeWhisperModel([0.7, 0.2])     # stands in for WhisperModel
    audio = [0.0] * host_mod.SAMPLE_RATE

    def decode():
        return provider.transcribe(audio, host_mod.SAMPLE_RATE)

    chunks, prob = provider.measure_no_speech(decode)
    assert [c.text for c in chunks] == ["word0", "word1"]
    assert prob == 0.2                      # the most speech-like segment decides
    model.probs = []
    assert provider.measure_no_speech(decode) == ([], None)
    assert len(decode()) == 0               # outside a measurement nothing is recorded
