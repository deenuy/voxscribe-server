"""T06 acceptance at SIM level: capture recovery, power wake, quiet hours.

Real time, real watchdog, the voice host's real armed loop and wake gate, real wake router.
Faked: the audio device (a feeder that delivers 32 ms chunks, or stops, or sends zeros), the
wake detector (stage 2 of the gate is off), and the chime. The quiet hours test writes to a
throwaway schema of the local database through the real Postgres session store.

    R11  stalled callback and all-zero stream each recover to armed within RECOVERY_BOUND_S
    R8   after sleep and wake (POST /v1/wake/power) the loop is armed within RECOVERY_BOUND_S
    R10  inside quiet hours, WAKES wakes give 0 sounds and WAKES rejected wake_events rows
"""

import asyncio
import sys
import time
import types
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from voxscribe.wake.gate import WakeGate

from app.local.audio_engine import AudioEngine
from app.local.config import QuietHours
from app.local.voice import capture_watchdog as cw
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.capture_watchdog import CaptureWatchdog
from app.local.voice.host import VoiceConfig, VoiceHost, WakeState
from tests.voice.unit.fakes import fake_ports

RATE = 16_000
CHUNK = 512
CHUNK_S = CHUNK / RATE
RECOVERY_BOUND_S = 10.0   # R8 and R11 bar
WAKES = 10                # R10 count


class FakeDevice:
    """Stands in for AudioEngine: delivers chunks to subscribers in real time until told otherwise."""

    capture_rate = RATE
    capture_device = None

    def __init__(self) -> None:
        self.mode = "noise"          # noise | zeros | stalled
        self.reopens: list[tuple[str, float]] = []
        self._qs: list[asyncio.Queue] = []
        self._rng = np.random.default_rng(1)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._qs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        if q in self._qs:
            self._qs.remove(q)

    async def reopen_stream(self, reason: str) -> bool:
        self.reopens.append((reason, time.monotonic()))
        self.mode = "noise"          # the new stream is live
        return True

    async def feed(self) -> None:
        while True:
            await asyncio.sleep(CHUNK_S)
            if self.mode == "stalled":
                continue
            if self.mode == "zeros":
                chunk = np.zeros(CHUNK, dtype=np.float32)
            else:
                chunk = self._rng.normal(0.0, 0.01, CHUNK).astype(np.float32)
            for q in list(self._qs):
                if q.full():
                    q.get_nowait()
                q.put_nowait(chunk)


class FakeWakeModel:
    """WakeDetector: scores the first `hits` frames as a wake, everything after as nothing."""

    model_name = "fake:hey_vox"
    patience_frames = 1

    def __init__(self, hits: int = 0) -> None:
        self.hits = hits
        self.predicts = 0
        self.resets = 0

    def score(self, _frame) -> float:
        self.predicts += 1
        return 0.9 if self.predicts <= self.hits else 0.0

    def reset(self) -> None:
        self.resets += 1


def make_host(model: FakeWakeModel, *, quiet=None, store=None) -> VoiceHost:
    """A host whose gate is stage 1 only (no voice check, no confirm) over the fake detector."""
    ports = fake_ports(wake=model, **({"store": store} if store is not None else {}))
    gate = WakeGate(model, voice_ratio=None, transcribe=None)
    host = VoiceHost(VoiceConfig(quiet_hours=quiet), ports=ports, wake_gate=gate)
    host.set_state(WakeState.ARMED)
    return host


async def wait_for(cond, timeout: float, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(step)
    return cond()


@pytest.fixture
def armed():
    """A fake device, a faked wake detector, and the host's armed loop and the watchdog on it."""
    model = FakeWakeModel()
    host = make_host(model)
    device = FakeDevice()
    health: list = []

    def on_event(event):
        health.append(event)
        host.on_capture_health(event)

    watchdog = CaptureWatchdog(device, on_event=on_event, default_input=lambda: None)
    return SimpleNamespace(model=model, device=device, watchdog=watchdog, health=health,
                           host=host)


async def start(env) -> list[asyncio.Task]:
    mic_q = env.device.subscribe()
    tasks = [
        asyncio.create_task(env.device.feed()),
        asyncio.create_task(env.host._armed(env.host.ports, env.host._gate, mic_q)),
    ]
    env.watchdog.start()
    assert await wait_for(lambda: env.model.predicts > 0, 2.0), "armed loop never scored a frame"
    return tasks


async def finish(env, tasks) -> None:
    await env.watchdog.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    env.host.set_state(WakeState.OFF)


async def assert_rearmed(env) -> None:
    """After a reopen the wake gate is reset and the armed loop scores fresh frames again."""
    before = env.model.predicts
    assert await wait_for(lambda: env.model.resets >= 1, 2.0), "wake gate not reset after reopen"
    assert await wait_for(lambda: env.model.predicts > before + 5, 2.0), "no frames scored after reopen"
    assert env.host.state == WakeState.ARMED


# ---------------------------------------------------------------- R11

async def test_r11_stalled_callback_recovers_to_armed(armed):
    tasks = await start(armed)
    try:
        stalled_at = time.monotonic()
        armed.device.mode = "stalled"
        assert await wait_for(lambda: armed.device.reopens, RECOVERY_BOUND_S)
        reason, reopened_at = armed.device.reopens[0]
        assert reason == "stall"
        assert reopened_at - stalled_at <= RECOVERY_BOUND_S
        assert reopened_at - stalled_at >= cw.STALL_TIMEOUT_S - CHUNK_S
        await assert_rearmed(armed)
        assert [e.action for e in armed.health] == ["reopened"]
    finally:
        await finish(armed, tasks)


async def test_r11_all_zero_stream_recovers_to_armed(armed):
    tasks = await start(armed)
    try:
        zeros_at = time.monotonic()
        armed.device.mode = "zeros"
        assert await wait_for(lambda: armed.device.reopens, RECOVERY_BOUND_S)
        reason, reopened_at = armed.device.reopens[0]
        assert reason == "silence"
        assert reopened_at - zeros_at <= RECOVERY_BOUND_S
        await assert_rearmed(armed)
        await asyncio.sleep(cw.SILENCE_CHECK_WINDOW_S + 0.5)
        assert len(armed.device.reopens) == 1, "a live stream after the reopen must not reopen again"
    finally:
        await finish(armed, tasks)


# ---------------------------------------------------------------- R8

async def test_r8_sleep_then_wake_is_armed_within_bound(armed):
    from app.local.routers import wake as wake_router

    api = FastAPI()
    api.include_router(wake_router.router)
    api.state.capture_watchdog = armed.watchdog
    api.state.voice_host = armed.host
    tasks = await start(armed)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api),
                                     base_url="http://test") as client:
            r = await client.post("/v1/wake/power", json={"event": "sleep"})
            assert r.status_code == 202 and r.json()["sleeping"] is True
            armed.device.mode = "stalled"          # asleep: CoreAudio delivers nothing
            await asyncio.sleep(1.0)

            woke_at = time.monotonic()
            r = await client.post("/v1/wake/power", json={"event": "wake"})
            assert r.status_code == 202 and r.json() == {"event": "wake", "handled": True,
                                                         "sleeping": False}
            assert await wait_for(lambda: armed.device.reopens, RECOVERY_BOUND_S)
            reason, reopened_at = armed.device.reopens[0]
            assert reason == "power_wake"
            assert reopened_at - woke_at <= RECOVERY_BOUND_S
            await assert_rearmed(armed)
            r = await client.get("/v1/wake/state")
            assert r.json()["state"] == "armed"

            r = await client.post("/v1/wake/power", json={"event": "hibernate"})
            assert r.status_code == 422
    finally:
        await finish(armed, tasks)


async def test_power_endpoint_without_audio_engine():
    from app.local.routers import wake as wake_router

    api = FastAPI()
    api.include_router(wake_router.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api),
                                 base_url="http://test") as client:
        r = await client.post("/v1/wake/power", json={"event": "wake"})
    assert r.status_code == 202 and r.json() == {"event": "wake", "handled": False}


# ---------------------------------------------------------------- R10

async def test_r10_quiet_hours_ten_wakes_no_sound_ten_rows(monkeypatch, test_db_url):
    from app.local.store import Store

    db_url, schema = test_db_url
    store = await Store.create(db_url, schema=schema)
    now = datetime.now()
    quiet = QuietHours(enabled=True, start=now.strftime("%H:%M"),
                       end=(now + timedelta(minutes=2)).strftime("%H:%M"))
    app = SimpleNamespace(state=SimpleNamespace(store=store, settings=None))

    sessions: list[float] = []

    async def session(*_a, **kw):
        sessions.append(kw.get("score"))

    model = FakeWakeModel(hits=WAKES)
    host = make_host(model, quiet=quiet, store=PostgresSessionStore(app))
    monkeypatch.setattr(host, "run_session", session)

    q: asyncio.Queue = asyncio.Queue()
    rng = np.random.default_rng(2)
    for _ in range(WAKES * 4):   # at least WAKES gate frames of 1280 samples
        q.put_nowait(rng.normal(0.0, 0.01, CHUNK).astype(np.float32))
    loop_task = asyncio.create_task(host._armed(host.ports, host._gate, q))
    try:
        assert await wait_for(lambda: q.empty() and model.predicts > WAKES, 10.0)
        await asyncio.sleep(0.2)
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
        host.set_state(WakeState.OFF)

    try:
        async with store.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT accepted, reject_reason, started_meeting_id, confidence FROM wake_events")
            rows = await cur.fetchall()
    finally:
        await store.close()

    assert host.ports.audio_out.files == []
    assert sessions == []
    assert len(rows) == WAKES
    assert all(r[0] is False and r[1] == "quiet_hours" and r[2] is None for r in rows)


async def test_outside_quiet_hours_a_wake_opens_a_session(monkeypatch):
    sessions: list[float] = []

    async def session(*_a, **kw):
        sessions.append(kw.get("score"))

    model = FakeWakeModel(hits=1)
    host = make_host(model)
    monkeypatch.setattr(host, "run_session", session)

    q: asyncio.Queue = asyncio.Queue()
    for _ in range(4):
        q.put_nowait(np.full(CHUNK, 0.01, dtype=np.float32))
    task = asyncio.create_task(host._armed(host.ports, host._gate, q))
    try:
        assert await wait_for(lambda: sessions, 5.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        host.set_state(WakeState.OFF)
    assert len(sessions) == 1


# ---------------------------------------------------------------- AudioEngine.reopen_stream

class _FakeStream:
    log: list = []

    def __init__(self, kind, **kw) -> None:
        self.kind, self.kw, self.written = kind, kw, []
        _FakeStream.log.append(("open", kind, id(self)))

    def start(self):
        pass

    def stop(self):
        _FakeStream.log.append(("stop", self.kind, id(self)))

    def abort(self):
        _FakeStream.log.append(("abort", self.kind, id(self)))

    def close(self):
        _FakeStream.log.append(("close", self.kind, id(self)))

    def write(self, chunk):
        self.written.append(chunk)


@pytest.fixture
def fake_sounddevice(monkeypatch):
    calls: list[str] = []
    sd = types.ModuleType("sounddevice")
    sd.InputStream = lambda **kw: _FakeStream("input", **kw)
    sd.OutputStream = lambda **kw: _FakeStream("output", **kw)
    sd._terminate = lambda: calls.append("terminate")
    sd._initialize = lambda: calls.append("initialize")
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    _FakeStream.log = []
    return calls


async def test_reopen_stream_replaces_both_streams_and_keeps_subscribers(fake_sounddevice):
    # Clean-up off: this checks the subscription survives a reopen, sample for sample.
    engine = AudioEngine(echo_cancellation=False, noise_suppression=False)
    await engine.start()
    try:
        q = engine.subscribe()
        old_in, old_out = engine._capture_stream, engine._playback_stream

        assert await engine.reopen_stream("stall") is True
        assert fake_sounddevice == ["terminate", "initialize"]
        assert ("abort", "input", id(old_in)) in _FakeStream.log
        assert ("close", "input", id(old_in)) in _FakeStream.log
        assert ("close", "output", id(old_out)) in _FakeStream.log
        new_in, new_out = engine._capture_stream, engine._playback_stream
        assert new_in is not old_in and new_out is not old_out
        assert engine.reopen_count == 1

        # The new stream's callback still reaches the old subscription.
        frame = np.full((CHUNK, 1), 0.25, dtype=np.float32)
        new_in.kw["callback"](frame, CHUNK, None, None)
        got = await asyncio.wait_for(q.get(), 1.0)
        assert float(got[0]) == pytest.approx(0.25)

        # The playback loop survived the swap and writes to the new stream.
        await engine.play(np.zeros(8, dtype=np.float32), block=True)
        assert await wait_for(lambda: new_out.written, 1.0)
    finally:
        await engine.stop()


async def test_reopen_stream_failure_keeps_engine_running(fake_sounddevice, monkeypatch):
    engine = AudioEngine()
    await engine.start()
    try:
        def broken(**_kw):
            raise RuntimeError("no device")

        monkeypatch.setattr(sys.modules["sounddevice"], "InputStream", broken)
        assert await engine.reopen_stream("device_change") is False
        assert engine._capture_stream is None
        await engine.play(np.zeros(8, dtype=np.float32))   # dropped, not raised
        assert not engine._playback_task.done()
    finally:
        await engine.stop()


async def test_reopen_stream_before_start_is_a_no_op():
    assert await AudioEngine().reopen_stream("stall") is False
