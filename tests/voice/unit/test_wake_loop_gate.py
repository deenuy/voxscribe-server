"""Wake loop host logic around the two-stage gate (T02). Fakes only: no models, mic or database.

Covers: a rejected candidate plays no sound, opens no session and writes one wake_events row; an
accepted one plays the chime, then starts the session with the pre-roll and without the audio
captured while the chime played; the once-per-minute telemetry line; the pre-roll queue.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from voxscribe.wake.gate import FRAME_SAMPLES, RejectReason, TelemetryWindow, WakeDecision

from app.local import wake_loop
from app.local.wake_loop import WakeState

CHUNK = wake_loop._CHUNK_FRAMES


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


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, params=None):
        self.rows.append((sql, params))

    async def commit(self):
        pass


class FakePool:
    def __init__(self):
        self.rows: list = []

    def connection(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return FakeConn(pool.rows)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def reject(reason=RejectReason.NO_VOICE, score=0.8, ratio=0.1):
    return WakeDecision(accepted=False, score=score, detector="fake:hey_vox", at_sample=0,
                        reason=reason, speech_ratio=ratio)


def accept(pre_roll, score=0.95):
    return WakeDecision(accepted=True, score=score, detector="fake:hey_vox", at_sample=0,
                        speech_ratio=0.7, phrase_end_sample=0, pre_roll=pre_roll)


@pytest.fixture
def env(monkeypatch):
    chimes: list[str] = []
    sessions: list[dict] = []
    pool = FakePool()
    app = SimpleNamespace(state=SimpleNamespace(store=SimpleNamespace(pool=pool), settings=None))
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)

    async def fake_chime(path):
        chimes.append(path)
        # The microphone hears the chime while it plays.
        for _ in range(3):
            audio_q.put_nowait(np.full(CHUNK, 0.5, dtype=np.float32))

    async def fake_session(app_, q, confidence, np, wake_phrase="hey vox", **kw):
        got = []
        while not q.empty():
            got.append(q.get_nowait())
        sessions.append({"confidence": confidence, "chunks": got, **kw})

    monkeypatch.setattr(wake_loop, "_play_chime_async", fake_chime)
    monkeypatch.setattr(wake_loop, "_run_session", fake_session)
    monkeypatch.setattr(wake_loop, "_CHIME_DECAY_S", 0.0)
    wake_loop.set_state(WakeState.ARMED)
    yield SimpleNamespace(app=app, q=audio_q, pool=pool, chimes=chimes, sessions=sessions)
    wake_loop.set_state(WakeState.OFF)


async def drive(env, gate, n_chunks, monkeypatch):
    """Feed n_chunks of quiet audio through the real _mic_loop with the fake gate."""
    monkeypatch.setattr(wake_loop, "_gate", gate)
    for _ in range(n_chunks):
        env.q.put_nowait(np.zeros(CHUNK, dtype=np.float32))
    task = asyncio.create_task(wake_loop._mic_loop(env.app, env.q, "hey vox", np, engine=None))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if env.q.empty():
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_rejected_candidate_plays_no_sound_opens_no_session_and_is_recorded(env, monkeypatch):
    gate = FakeGate([None, reject(RejectReason.NO_MATCH, score=0.83, ratio=0.6)])
    await drive(env, gate, n_chunks=10, monkeypatch=monkeypatch)
    assert env.chimes == []
    assert env.sessions == []
    assert gate.resets == 0
    (sql, params), = env.pool.rows
    assert "INSERT INTO wake_events" in sql
    _id, phrase, score, _at, meeting_id, accepted, reason, detector, ratio = params
    assert (phrase, score, meeting_id, accepted, reason, detector, ratio) == (
        "hey_vox", 0.83, None, False, "no_match", "fake:hey_vox", 0.6)


async def test_accepted_wake_chimes_then_starts_session_with_pre_roll_and_no_chime_audio(
        env, monkeypatch):
    pre_roll = np.arange(CHUNK * 2, dtype=np.float32) / 1e6
    gate = FakeGate([None, accept(pre_roll)])
    await drive(env, gate, n_chunks=10, monkeypatch=monkeypatch)
    assert env.chimes == [wake_loop._CHIME_ACK]
    (session,) = env.sessions
    assert session["confidence"] == 0.95
    assert session["detector"] == "fake:hey_vox" and session["speech_ratio"] == 0.7
    audio = np.concatenate(session["chunks"])
    # The session starts with the audio heard after the phrase ...
    assert np.array_equal(audio[: pre_roll.shape[0]], pre_roll)
    # ... and nothing the microphone captured while the chime played (0.5 everywhere).
    assert not np.any(audio == 0.5)
    assert gate.resets == 1          # back to armed with fresh detector state


async def test_telemetry_line_once_per_period(env, monkeypatch, caplog):
    gate = FakeGate([], telemetry=[TelemetryWindow(peak_score=0.4211, peak_rms=0.012, frames=750)])
    with caplog.at_level(logging.INFO, logger="app.local.wake"):
        await drive(env, gate, n_chunks=10, monkeypatch=monkeypatch)
    lines = [r.getMessage() for r in caplog.records if "peak_score=" in r.getMessage()]
    assert lines == ["wake: peak_score=0.421 frames=750 peak_rms=0.0120"]


async def test_frames_handed_to_the_gate_are_80_ms(env, monkeypatch):
    gate = FakeGate([])
    n_chunks = 10
    await drive(env, gate, n_chunks=n_chunks, monkeypatch=monkeypatch)
    assert gate.frames == (n_chunks * CHUNK) // FRAME_SAMPLES


async def test_pre_roll_queue_yields_pre_roll_then_live_audio():
    live: asyncio.Queue = asyncio.Queue()
    live.put_nowait("live-1")
    q = wake_loop._PreRollQueue(np.arange(5, dtype=np.float32), live, chunk_frames=2)
    assert not q.empty()
    assert (await q.get()).tolist() == [0, 1]
    assert q.get_nowait().tolist() == [2, 3]
    assert (await q.get()).tolist() == [4]
    assert await q.get() == "live-1"
    assert q.empty()
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()
