"""Voice host around the two-stage wake gate (T02 behaviour on the T08 host). Fakes only.

Covers: a rejected candidate plays no sound, opens no session and is recorded once; an accepted
one plays the chime, then starts the session with the pre-roll and without the audio captured
while the chime played; the once-per-minute telemetry line; 80 ms frames; the pre-roll queue;
the gate reset after a capture reopen (T06); and the wake_events row the Postgres store writes.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from voxscribe.wake.gate import FRAME_SAMPLES, RejectReason, TelemetryWindow, WakeDecision

from app.local.voice import host as host_mod
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import (
    CHIME_ACK,
    CHUNK_FRAMES,
    PreRollQueue,
    VoiceConfig,
    VoiceHost,
    WakeState,
)

from .fakes import fake_ports

CHUNK = CHUNK_FRAMES


class FakeGate:
    """Returns scripted decisions, one per processed frame, then None."""

    def __init__(self, decisions, telemetry=None):
        self._decisions = list(decisions)
        self._telemetry = list(telemetry or [])
        self.detector = SimpleNamespace(model_name="fake:hey_vox", patience_frames=2)
        self.config = SimpleNamespace(threshold=0.5)
        self.frames = 0
        self.resets = 0

    def process(self, frame):
        assert frame.shape == (FRAME_SAMPLES,)
        self.frames += 1
        return self._decisions.pop(0) if self._decisions else None

    def pop_telemetry(self):
        return self._telemetry.pop(0) if self._telemetry else None

    def reset(self):
        self.resets += 1


def reject(reason=RejectReason.NO_VOICE, score=0.8, ratio=0.1):
    return WakeDecision(accepted=False, score=score, detector="fake:hey_vox", at_sample=0,
                        reason=reason, speech_ratio=ratio)


def accept(pre_roll, score=0.95):
    return WakeDecision(accepted=True, score=score, detector="fake:hey_vox", at_sample=0,
                        speech_ratio=0.7, phrase_end_sample=0, pre_roll=pre_roll)


@pytest.fixture
def env(monkeypatch):
    ports = fake_ports()
    host = VoiceHost(VoiceConfig(), ports=ports)
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    sessions: list[dict] = []

    real_play_file = ports.audio_out.play_file

    async def chime(path):
        await real_play_file(path)
        # The microphone hears the chime while it plays.
        for _ in range(3):
            audio_q.put_nowait(np.full(CHUNK, 0.5, dtype=np.float32))

    async def fake_session(q, *, score, detector="trigger", speech_ratio=None):
        got = []
        while not q.empty():
            got.append(q.get_nowait())
        sessions.append({"confidence": score, "chunks": got, "detector": detector,
                         "speech_ratio": speech_ratio})

    monkeypatch.setattr(ports.audio_out, "play_file", chime)
    monkeypatch.setattr(host, "run_session", fake_session)
    monkeypatch.setattr(host_mod, "CHIME_DECAY_S", 0.0)
    host.set_state(WakeState.ARMED)
    yield SimpleNamespace(host=host, ports=ports, q=audio_q, sessions=sessions)
    host.set_state(WakeState.OFF)


async def drive(env, gate, n_chunks):
    """Feed n_chunks of quiet audio through the host's armed loop with the fake gate."""
    for _ in range(n_chunks):
        env.q.put_nowait(np.zeros(CHUNK, dtype=np.float32))
    task = asyncio.create_task(env.host._armed(env.ports, gate, env.q))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if env.q.empty():
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_rejected_candidate_plays_no_sound_opens_no_session_and_is_recorded(env):
    gate = FakeGate([None, reject(RejectReason.NO_MATCH, score=0.83, ratio=0.6)])
    await drive(env, gate, n_chunks=10)
    assert env.ports.audio_out.files == []
    assert env.sessions == []
    assert gate.resets == 0
    (row,) = env.ports.store.wakes
    assert row == {"score": 0.83, "accepted": False, "reject_reason": "no_match",
                   "detector": "fake:hey_vox", "speech_ratio": 0.6}


async def test_accepted_wake_chimes_then_starts_session_with_pre_roll_and_no_chime_audio(env):
    pre_roll = np.arange(CHUNK * 2, dtype=np.float32) / 1e6
    gate = FakeGate([None, accept(pre_roll)])
    await drive(env, gate, n_chunks=10)
    assert env.ports.audio_out.files == [CHIME_ACK]
    (session,) = env.sessions
    assert session["confidence"] == 0.95
    assert session["detector"] == "fake:hey_vox" and session["speech_ratio"] == 0.7
    audio = np.concatenate(session["chunks"])
    # The session starts with the audio heard after the phrase ...
    assert np.array_equal(audio[: pre_roll.shape[0]], pre_roll)
    # ... and nothing the microphone captured while the chime played (0.5 everywhere).
    assert not np.any(audio == 0.5)
    assert gate.resets == 1          # back to armed with fresh detector state


async def test_telemetry_line_once_per_period(env, caplog):
    gate = FakeGate([], telemetry=[TelemetryWindow(peak_score=0.4211, peak_rms=0.012, frames=750)])
    with caplog.at_level(logging.INFO, logger="app.local.wake"):
        await drive(env, gate, n_chunks=10)
    lines = [r.getMessage() for r in caplog.records if "peak_score=" in r.getMessage()]
    assert lines == ["wake: peak_score=0.421 frames=750 peak_rms=0.0120"]


async def test_frames_handed_to_the_gate_are_80_ms(env):
    gate = FakeGate([])
    n_chunks = 10
    await drive(env, gate, n_chunks=n_chunks)
    assert gate.frames == (n_chunks * CHUNK) // FRAME_SAMPLES


async def test_capture_reopen_resets_the_gate_on_the_next_frame(env):
    gate = FakeGate([])
    env.host.on_capture_health(SimpleNamespace(action="reopened", issue="stall", value=0.0))
    await drive(env, gate, n_chunks=3)
    assert gate.resets == 1
    env.host.on_capture_health(SimpleNamespace(action="health", issue="clip", value=1.0))
    await drive(env, gate, n_chunks=3)
    assert gate.resets == 1


async def test_pre_roll_queue_yields_pre_roll_then_live_audio():
    live: asyncio.Queue = asyncio.Queue()
    live.put_nowait("live-1")
    q = PreRollQueue(np.arange(5, dtype=np.float32), live, chunk_frames=2)
    assert not q.empty()
    assert (await q.get()).tolist() == [0, 1]
    assert q.get_nowait().tolist() == [2, 3]
    assert (await q.get()).tolist() == [4]
    assert await q.get() == "live-1"
    assert q.empty()
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, params=None):
        self.rows.append((sql, params))

    async def commit(self):
        pass


class _FakePool:
    def __init__(self):
        self.rows: list = []

    def connection(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _FakeConn(pool.rows)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


async def test_postgres_store_writes_one_wake_events_row_per_candidate():
    pool = _FakePool()
    store = PostgresSessionStore(SimpleNamespace(state=SimpleNamespace(
        store=SimpleNamespace(pool=pool))))
    await store.record_wake(score=0.83, accepted=False, reject_reason="no_match",
                            detector="fake:hey_vox", speech_ratio=0.6)
    (sql, params), = pool.rows
    assert "INSERT INTO wake_events" in sql
    _id, phrase, score, _at, meeting_id, accepted, reason, detector, ratio = params
    assert (phrase, score, meeting_id, accepted, reason, detector, ratio) == (
        "hey_vox", 0.83, None, False, "no_match", "fake:hey_vox", 0.6)
