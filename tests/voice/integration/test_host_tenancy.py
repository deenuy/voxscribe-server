"""--wake startup with tenant isolation on (PL05): the lifespan builds the voice host unbound, it
stays off, and the verified person who arms it over HTTP is who it listens for.

As in test_host_app.py the audio device is refused and the host's adapters are fakes; the app,
its lifespan, the identity check and the wake router are real.
"""
from __future__ import annotations

import asyncio
import uuid

import httpx
import psycopg
import pytest

from app.local import audio_engine, tenancy
from app.local.app import create_app
from app.local.config import Config
from app.local.identity import IdentityVerifier
from app.local.migrations import run_migrations
from app.local.voice import host as host_mod
from tests.test_tenancy import AUDIENCE, DB_URL, ISSUER, Keys
from tests.voice.unit.fakes import EndlessQueue, fake_ports

pytestmark = pytest.mark.integration

READY_TIMEOUT_S = 5.0
POLL_S = 0.02


async def _no_device(self):
    raise RuntimeError("no audio device in tests")


async def test_wake_startup_with_isolation_on_is_off_until_a_verified_person_arms(monkeypatch):
    monkeypatch.setattr(audio_engine.AudioEngine, "start", _no_device)
    ports = fake_ports()
    ports.audio_in.subscribe = lambda: EndlessQueue()
    monkeypatch.setattr(host_mod, "build_ports", lambda app: ports)

    base = f"t_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {base}")
        await conn.commit()
    await run_migrations(DB_URL, schema=base)
    settings = Config(database_url=DB_URL, data_dir="/tmp/voxscribe_test", providers="fake",
                      identity_issuer=ISSUER, identity_audience=AUDIENCE,
                      identity_jwks_url="http://jwks.test/keys")
    app = await create_app(settings=settings, auth_disabled=True, schema=base, wake=True)
    keys = Keys()

    async def fetch():
        return {"keys": [keys.jwk]}

    app.state.identity = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE,
                                          jwks_url="http://jwks.test/keys", fetch=fetch)
    subject = f"alice-{base}"
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        host = app.state.voice_host

        async def until(cond):
            deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_S
            while not cond() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(POLL_S)
            return cond()

        await asyncio.sleep(10 * POLL_S)
        assert host.state == host_mod.WakeState.OFF and not host.bound
        assert ports.audio_in.starts == 0                  # the armed loop never began

        headers = {"X-Identity-Token": keys.token(subject)}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            assert (await client.get("/v1/wake/state", headers=headers)).json()["state"] == "off"
            assert (await client.post("/v1/wake/arm", headers=headers)).json() == {
                "state": "armed", "changed": True}
            assert host.binding.tenant == tenancy.tenant_key(subject)
            assert await until(lambda: ports.audio_in.starts == 1)
            assert (await client.get("/v1/wake/state", headers=headers)).json()["state"] == "armed"
    finally:
        await lifespan.__aexit__(None, None, None)
        async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{tenancy.tenant_key(subject)}" CASCADE')
            await conn.execute(f"DROP SCHEMA IF EXISTS {base} CASCADE")
    assert host.state == host_mod.WakeState.OFF
