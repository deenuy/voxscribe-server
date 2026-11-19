"""--wake startup on the voice host: lifespan builds it, the routes read it, shutdown stops it.

The audio device is refused (so no microphone opens) and the host's adapters are fakes; the
app, its lifespan, the wake router and the debug trigger are real, on a throwaway schema.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.local import audio_engine
from app.local.app import create_app
from app.local.config import Config
from app.local.voice import host as host_mod
from tests.voice.unit.fakes import EndlessQueue, fake_ports

pytestmark = pytest.mark.integration

from tests.testdb import TEST_DATABASE_URL as DB_URL
READY_TIMEOUT_S = 5.0
POLL_S = 0.02


async def _no_device(self):
    raise RuntimeError("no audio device in tests")


async def test_wake_startup_runs_the_voice_host(throwaway_schema, monkeypatch):
    monkeypatch.setattr(audio_engine.AudioEngine, "start", _no_device)
    ports = fake_ports()
    ports.audio_in.subscribe = lambda: EndlessQueue()
    monkeypatch.setattr(host_mod, "build_ports", lambda app: ports)

    settings = Config(database_url=DB_URL, data_dir="/tmp/voxscribe_test", providers="fake")
    app = await create_app(settings=settings, auth_disabled=True, schema=throwaway_schema,
                           wake=True, debug_trigger=True)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        host = app.state.voice_host
        assert not hasattr(app.state, "audio_engine")
        assert not hasattr(app.state, "capture_watchdog")    # no engine, no watchdog

        async def until(cond):
            deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_S
            while not cond() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(POLL_S)
            return cond()

        assert await until(lambda: host.ready and host.snapshot()["model"] == "fake:hey_vox")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            state = (await client.get("/v1/wake/state")).json()
            assert state["state"] == "armed" and state["trigger"] == "hey vox"
            assert (await client.post("/v1/wake/disarm")).json() == {"state": "off",
                                                                     "changed": True}
            assert (await client.post("/v1/wake/arm")).json() == {"state": "armed",
                                                                  "changed": True}
            # The trigger route needs the audio engine, which did not start.
            assert (await client.post("/v1/wake/trigger")).status_code == 503
            assert (await client.post("/v1/wake/power", json={"event": "sleep"})).json() == {
                "event": "sleep", "handled": False}
    finally:
        await lifespan.__aexit__(None, None, None)
    assert host.state == host_mod.WakeState.OFF
