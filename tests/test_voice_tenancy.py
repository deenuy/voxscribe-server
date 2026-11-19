"""Voice and tenant isolation, both on (PL05, gap T1).

The real engine with identity switched on, the real Postgres voice store, and a voice host whose
audio, wake detector, recognizer, turn policy, reply model and talker are fakes. A voice session
belongs to the tenant the device is bound to: its rows land in that tenant's schema and nowhere
else, every connection it borrows is borrowed inside that tenant's scope, and with nobody bound
the host stays off. With isolation off the same session writes to the one shared schema, as
before. The host-level half of this (no database) is tests/voice/unit/test_session_context.py.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import psycopg
import pytest
from voxscribe.wake.gate import RejectReason, WakeDecision

from app.local import finalize, tenancy
from app.local.config import Config
from app.local.identity import IdentityVerifier
from app.local.migrations import run_migrations
from app.local.tools import web_search
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.scope import LOCAL_DEVICE, TenantSessionScope
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import (
    SessionContext,
    SessionTenantRequiredError,
    VoiceConfig,
    VoiceHost,
    WakeState,
)
from tests.conftest import make_app
from tests.test_tenancy import AUDIENCE, DB_URL, ISSUER, Keys
from tests.voice.unit.fakes import EndlessQueue, FakeRecognizer, fake_ports

TURNS = ("What is one?", "And two?")
STEP_TIMEOUT_S = 20
POLL_S = 0.01
SESSION_TABLES = ("meetings", "wake_events", "transcript_segments", "turns", "replies",
                  "latency_ledger")
DELEGATE_TABLES = ("reminders", "tool_calls", "delegate_tasks")


@pytest.fixture
async def engine():
    """The engine with identity on and no wake listener of its own. Yields (app, base schema,
    tenant for a subject). Drops every schema it created."""
    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    base = f"t_{uuid.uuid4().hex[:8]}"       # the shared (platform) schema
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {base}")
        await conn.commit()
    await run_migrations(DB_URL, schema=base)
    settings = Config(database_url=DB_URL, data_dir=str(data_dir), providers="fake",
                      identity_issuer=ISSUER, identity_audience=AUDIENCE,
                      identity_jwks_url="http://jwks.test/keys")
    from app.local.app import create_app

    app = await create_app(settings=settings, auth_disabled=True, schema=base)
    ctx = app.router.lifespan_context(app)
    await ctx.__aenter__()
    created: list[str] = []

    def tenant(subject: str) -> str:
        key = tenancy.tenant_key(f"{subject}-{base}")      # unique per test run
        created.append(key)
        return key

    yield app, base, tenant
    await _settle()
    await ctx.__aexit__(None, None, None)
    async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
        for key in set(created):
            await conn.execute(f'DROP SCHEMA IF EXISTS "{key}" CASCADE')
        await conn.execute(f"DROP SCHEMA IF EXISTS {base} CASCADE")


async def _settle() -> None:
    """Let the finalisation a closed session scheduled finish before the pool goes away."""
    while finalize._running:
        await asyncio.gather(*list(finalize._running), return_exceptions=True)


async def _until(condition) -> None:
    async def poll():
        while not condition():
            await asyncio.sleep(POLL_S)

    await asyncio.wait_for(poll(), STEP_TIMEOUT_S)


async def _counts(schema: str, tables) -> dict[str, int]:
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        return {t: (await (await conn.execute(f'SELECT count(*) FROM "{schema}".{t}')).fetchone())[0]
                for t in tables}


async def _rows(schema: str, query: str) -> list[tuple]:
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        return await (await conn.execute(query.format(s=f'"{schema}"'))).fetchall()


@pytest.fixture
def borrows(monkeypatch):
    """Every connection borrowed from the engine's pool from here on, as the scope it was
    borrowed in (a tenant key, `<system>`, or None for no scope at all)."""
    seen: list[str | None] = []
    real = tenancy.TenantPool.connection

    def connection(self, **kwargs):
        seen.append(tenancy.current())
        return real(self, **kwargs)

    monkeypatch.setattr(tenancy.TenantPool, "connection", connection)
    return seen


def _tenant_errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if (r.exc_info and r.exc_info[0] is not None
                and issubclass(r.exc_info[0], tenancy.TenantRequiredError | SessionTenantRequiredError))
            or "no tenant in scope" in r.getMessage()]


class ScriptedWakeGate:
    """The wake gate's decisions, in order; nothing after them."""

    detector = SimpleNamespace(model_name="fake:hey_vox", patience_frames=1)
    config = SimpleNamespace(threshold=0.5)

    def __init__(self, decisions) -> None:
        self._decisions = list(decisions)

    def process(self, _frame):
        return self._decisions.pop(0) if self._decisions else None

    def pop_telemetry(self):
        return None

    def reset(self) -> None:
        pass


def _rejected() -> WakeDecision:
    return WakeDecision(accepted=False, score=0.6, detector="fake:hey_vox", at_sample=0,
                        reason=RejectReason.NO_VOICE, speech_ratio=0.1)


def _accepted() -> WakeDecision:
    return WakeDecision(accepted=True, score=0.95, detector="fake:hey_vox", at_sample=0,
                        speech_ratio=0.7, phrase_end_sample=0,
                        pre_roll=np.zeros(0, dtype=np.float32))


# ── a full voice turn with isolation on ───────────────────────────────────────


async def test_a_voice_session_with_isolation_on_writes_its_rows_in_the_tenants_schema(
        engine, borrows, caplog):
    """From the armed loop: a rejected wake, then a wake that opens a session of two turns."""
    app, base, tenant_for = engine
    alice = tenant_for("alice")
    scope = TenantSessionScope(app)
    ports = fake_ports(store=PostgresSessionStore(app), recognizer=FakeRecognizer(TURNS))
    ports.audio_in.subscribe = lambda: EndlessQueue()
    host = VoiceHost(VoiceConfig(), ports=ports, scope=scope, binding=scope.device_binding(),
                     wake_gate=ScriptedWakeGate([_rejected(), _accepted()]))
    closed: list[str] = []
    close_session = ports.store.close_session

    async def close_and_tell(session_id):
        await close_session(session_id)
        closed.append(session_id)

    ports.store.close_session = close_and_tell

    caplog.set_level(logging.WARNING)
    host.start()
    try:
        # Nobody is bound: the host is off, does not arm, and has not touched the database.
        await asyncio.sleep(10 * POLL_S)
        assert host.state == WakeState.OFF and not host.bound
        with pytest.raises(SessionTenantRequiredError):
            host.arm()
        assert borrows == []

        assert host.bind(scope.device_binding(alice)) and host.arm()
        await _until(lambda: closed and host.state == WakeState.ARMED)
        host.disarm()
    finally:
        await host.stop()
    await _settle()

    (session_id,) = closed
    assert await _counts(alice, SESSION_TABLES) == {
        "meetings": 1, "wake_events": 2, "transcript_segments": len(TURNS),
        "turns": len(TURNS), "replies": len(TURNS), "latency_ledger": len(TURNS)}
    assert await _counts(base, SESSION_TABLES) == dict.fromkeys(SESSION_TABLES, 0)

    # The session row says who and what it ran for.
    assert await _rows(alice, "SELECT id, origin, tenant, product, device, profile_version "
                              "FROM {s}.meetings") == [
        (session_id, "wake", alice, app.state.profile.mode, LOCAL_DEVICE,
         scope.profile_version())]
    assert scope.profile_version().startswith(f"{app.state.profile.mode}@")
    assert sorted(await _rows(alice, "SELECT accepted, started_meeting_id FROM {s}.wake_events"),
                  key=lambda r: r[0]) == [(False, None), (True, session_id)]
    assert await _rows(alice, "SELECT turn_n, user_text FROM {s}.turns ORDER BY turn_n") == [
        (1, TURNS[0]), (2, TURNS[1])]

    # No tenant error anywhere, and every borrow was made inside the tenant's scope.
    assert _tenant_errors(caplog) == []
    assert borrows and set(borrows) == {alice}


async def test_every_connection_a_session_and_its_tasks_borrow_is_inside_the_tenants_scope(
        engine, borrows, caplog, monkeypatch):
    """The runtime half of the boundary: the store, the reminder store, the delegate's task rows
    and the thinker's tool row (written by a task the session started) all borrow in scope."""
    app, base, tenant_for = engine
    alice = tenant_for("alice")

    async def search(_query: str) -> str:
        return "[SEARCH RESULTS]\n1. Tides: a table (http://example.test)\n[END SEARCH RESULTS]"

    async def stream_reply(_user_text, *, model=None, system_prompt=None, history=None):
        yield "High tide is at noon."

    monkeypatch.setattr(web_search, "search", search)
    monkeypatch.setattr("app.local.llm_adapter.stream_reply", stream_reply)
    # Alice allows connectors, so the network decision (PL41) lets the thinker search; the
    # decision reads her stored setting, inside her scope like every other borrow.
    with tenancy.tenant_scope(alice):
        async with app.state.store.pool.connection() as conn:
            await conn.execute("INSERT INTO settings (key, value) VALUES (%s, %s)",
                               ("privacyMode", '"connectors"'))
            await conn.commit()

    delegate = OllamaDelegate(app)
    turns = ("Search for the latest tide tables.", "Remind me to call Alex in 10 minutes.",
             "What reminders do I have?")
    scope = TenantSessionScope(app)
    ports = fake_ports(store=PostgresSessionStore(app), delegate=delegate,
                       recognizer=FakeRecognizer(turns))
    host = VoiceHost(VoiceConfig(), ports=ports, scope=scope,
                     binding=scope.device_binding(alice))

    caplog.set_level(logging.WARNING)
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                           STEP_TIMEOUT_S)
    for task in delegate.lane.active():
        await delegate.lane.wait(task.task_id)
    await delegate.drain()
    await _settle()

    counts = await _counts(alice, DELEGATE_TABLES)
    assert counts["reminders"] == 1 and counts["tool_calls"] == 1 and counts["delegate_tasks"] == 1
    assert await _counts(alice, ("turns",)) == {"turns": len(turns)}
    assert await _counts(base, (*SESSION_TABLES, *DELEGATE_TABLES)) == dict.fromkeys(
        (*SESSION_TABLES, *DELEGATE_TABLES), 0)

    assert _tenant_errors(caplog) == []
    assert borrows and set(borrows) == {alice}


async def test_a_session_opened_with_no_tenant_is_refused_and_touches_no_schema(engine, borrows):
    app, base, _tenant_for = engine
    scope = TenantSessionScope(app)
    host = VoiceHost(VoiceConfig(), scope=scope, binding=scope.device_binding(),
                     ports=fake_ports(store=PostgresSessionStore(app),
                                      recognizer=FakeRecognizer(TURNS)))
    with pytest.raises(SessionTenantRequiredError):
        await host.run_session(EndlessQueue(), score=0.9)
    assert borrows == []
    assert await _counts(base, SESSION_TABLES) == dict.fromkeys(SESSION_TABLES, 0)

    # A voice write attempted outside any scope is refused by the store's pool, not redirected.
    await host.ports.store.record_wake(score=0.5, accepted=False, reject_reason="no_voice",
                                       detector="fake", speech_ratio=0.1)
    with tenancy.system_scope():
        await host.ports.store.record_wake(score=0.5, accepted=False, reject_reason="no_voice",
                                           detector="fake", speech_ratio=0.1)
    assert borrows == []
    assert await _counts(base, ("wake_events",)) == {"wake_events": 0}


# ── who binds the device ──────────────────────────────────────────────────────


async def test_the_host_built_for_the_app_starts_unbound_and_arming_binds_the_verified_caller(
        engine):
    app, _base, _tenant_for = engine
    keys = Keys()

    async def fetch():
        return {"keys": [keys.jwk]}

    app.state.identity = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE,
                                          jwks_url="http://jwks.test/keys", fetch=fetch)
    host = VoiceHost.from_app(app)
    host.ports = fake_ports()                # built, never started: no audio, no models
    app.state.voice_host = host
    assert not host.bound and host.binding.tenant is None
    assert (host.binding.product, host.binding.device) == (app.state.profile.mode, LOCAL_DEVICE)

    def as_user(subject: str) -> dict:
        return {"X-Identity-Token": keys.token(subject)}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:
        assert (await client.post("/v1/wake/arm")).json()["error"] == "identity_required"
        assert not host.bound and host.state == WakeState.OFF

        res = await client.post("/v1/wake/arm", headers=as_user("alice"))
        assert res.status_code == 200 and res.json() == {"state": "armed", "changed": True}
        assert host.binding.tenant == tenancy.tenant_key("alice")

        # While a session runs, another person cannot take the device.
        host._open = SessionContext(binding=host.binding)
        res = await client.post("/v1/wake/arm", headers=as_user("bob"))
        assert res.status_code == 409 and res.json()["error"] == "voice_session_in_progress"
        assert host.binding.tenant == tenancy.tenant_key("alice")
        host._open = None

        res = await client.post("/v1/wake/arm", headers=as_user("bob"))
        assert res.status_code == 200 and host.binding.tenant == tenancy.tenant_key("bob")


# ── isolation off ─────────────────────────────────────────────────────────────


async def test_with_isolation_off_a_session_writes_to_the_shared_schema_as_before(
        throwaway_schema, caplog):
    app = await make_app(throwaway_schema)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        host = VoiceHost.from_app(app)
        host.ports = fake_ports(store=PostgresSessionStore(app), recognizer=FakeRecognizer(TURNS))
        assert host.bound and host.binding.tenant is None and host.arm()

        caplog.set_level(logging.WARNING)
        await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                               STEP_TIMEOUT_S)
        await _settle()
    finally:
        await lifespan.__aexit__(None, None, None)

    assert await _counts(throwaway_schema, SESSION_TABLES) == {
        "meetings": 1, "wake_events": 1, "transcript_segments": len(TURNS),
        "turns": len(TURNS), "replies": len(TURNS), "latency_ledger": len(TURNS)}
    # No tenant to record; the other three facts are still on the row.
    ((tenant, product, device, version),) = await _rows(
        throwaway_schema, "SELECT tenant, product, device, profile_version FROM {s}.meetings")
    assert tenant is None and (product, device) == (app.state.profile.mode, LOCAL_DEVICE)
    assert version == TenantSessionScope(app).profile_version()
    assert _tenant_errors(caplog) == []
