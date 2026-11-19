"""Erase and export with tenant isolation on, where erase meets the voice host, the pool and work in flight (PL32 E).

E1  erase touches the voice host only when it works for the erasing tenant, keeps product and device, and is refused
    only while that tenant's own session runs.
E2  a borrow never falls through to the public schema: not for a schema erase is dropping, not for a ready set that
    another process (or erase) made stale.
E3  erase stops the tenant's work and forgets what memory holds for them, before the drop.
E5  export compresses off the event loop and pages its rows.
"""
from __future__ import annotations

import asyncio
import io
import tarfile
import time
from types import SimpleNamespace

import psycopg
import pytest

from app.local import events, finalize, meeting_bus, tenancy
from app.local.event_schema import RESET
from app.local.voice.adapters.scope import TenantSessionScope
from app.local.voice.delegate_lane import DelegateLane
from app.local.voice.host import SessionBinding, SessionContext, VoiceConfig, VoiceHost, WakeState
from tests import test_tenancy
from tests.testdb import TEST_DATABASE_URL
from tests.voice.unit.fakes import fake_ports

keys, tenant_engine = test_tenancy.keys, test_tenancy.tenant_engine   # fixtures, shared with test_tenancy

ERASE = {"X-Confirm": "delete-all"}
EXPORT = {"X-Confirm": "export"}


def _host(app, binding: SessionBinding) -> VoiceHost:
    scope = TenantSessionScope(app)
    host = VoiceHost(VoiceConfig(), ports=fake_ports(), scope=scope, binding=binding)
    app.state.voice_host = host
    return host


def _binding(tenant: str) -> SessionBinding:
    return SessionBinding(tenant=tenant, product="personal", device="kitchen", organization="org-1", role="member")


def _run_session(host: VoiceHost, tenant: str) -> None:
    """A session of this tenant is running (what VoiceHost.run_session holds while one is open)."""
    host._open = SessionContext(binding=host.binding)
    host.set_state(WakeState.LISTENING)


async def _schema_exists(key: str) -> bool:
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        row = await (await conn.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (key,))).fetchone()
    return row is not None


# ── E1: the voice host ────────────────────────────────────────────────────────


async def test_erasing_another_tenant_leaves_the_device_armed_for_its_owner(tenant_engine):
    client, app, as_user = tenant_engine
    alice, bob = as_user("e1-alice"), as_user("e1-bob")
    assert (await client.get("/v1/meetings", headers=alice)).status_code == 200     # alice has data
    host = _host(app, _binding(tenancy.tenant_key("e1-alice")))
    host.set_state(WakeState.ARMED)
    before = host.binding

    res = await client.delete("/v1/data", headers={**bob, **ERASE})
    assert res.status_code == 204, res.text
    assert host.binding == before and host.state == WakeState.ARMED
    assert await _schema_exists(tenancy.tenant_key("e1-alice"))     # alice's data is still there


async def test_erasing_the_bound_tenant_unbinds_only_who_the_device_is_for(tenant_engine):
    client, app, as_user = tenant_engine
    alice = as_user("e1-owner")
    key = tenancy.tenant_key("e1-owner")
    host = _host(app, _binding(key))

    res = await client.delete("/v1/data", headers={**alice, **ERASE})
    assert res.status_code == 204, res.text
    assert host.binding == SessionBinding(tenant=None, product="personal", device="kitchen",
                                          organization=None, role=None)
    assert key not in app.state.store.pool.provisioner._ready


async def test_only_the_erasing_tenants_own_running_session_refuses_the_erase(tenant_engine):
    client, app, as_user = tenant_engine
    alice, bob = as_user("e1-talking"), as_user("e1-other")
    key = tenancy.tenant_key("e1-talking")
    host = _host(app, _binding(key))
    _run_session(host, key)
    assert (await client.get("/v1/meetings", headers=alice)).status_code == 200     # provision alice

    other = await client.delete("/v1/data", headers={**bob, **ERASE})
    assert other.status_code == 204, other.text                  # someone else's session does not block bob

    own = await client.delete("/v1/data", headers={**alice, **ERASE})
    assert own.status_code == 409 and own.json()["error"] == "session_in_progress"
    assert host.binding.tenant == key and await _schema_exists(key)   # nothing was erased or unbound


# ── E2: no fall-through to public ─────────────────────────────────────────────


async def test_a_stale_ready_set_raises_instead_of_resolving_to_public(tenant_engine):
    """Another engine process (or erase) dropped the schema; this process still lists the tenant as ready."""
    client, app, as_user = tenant_engine
    me = as_user("e2-stale")
    key = tenancy.tenant_key("e2-stale")
    assert (await client.get("/v1/meetings", headers=me)).status_code == 200
    pool = app.state.store.pool
    assert key in pool.provisioner._ready
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        await conn.execute(f'DROP SCHEMA "{key}" CASCADE')

    with tenancy.tenant_scope(key):
        with pytest.raises(tenancy.TenantRequiredError):
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
        assert key not in pool.provisioner._ready          # the next borrow provisions it again
        async with pool.connection() as conn:
            schema = (await (await conn.execute("SELECT current_schema()")).fetchone())[0]
    assert schema == key


async def test_concurrent_erase_and_export_never_return_a_shared_schema_row(tenant_engine):
    """A shared table with a row sits in public, where a missing tenant schema would resolve to."""
    client, app, as_user = tenant_engine
    me = as_user("e2-race")
    marker = "pl32-shared-schema-row"
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        await conn.execute("CREATE TABLE public.pl32_shared (v text)")
        await conn.execute("INSERT INTO public.pl32_shared VALUES (%s)", (marker,))
    try:
        for _ in range(15):
            assert (await client.get("/v1/meetings", headers=me)).status_code == 200
            erase, export = await asyncio.gather(
                client.delete("/v1/data", headers={**me, **ERASE}),
                client.get("/v1/data/export", headers={**me, **EXPORT}),
                return_exceptions=True)
            assert not isinstance(erase, BaseException) and erase.status_code == 204
            if isinstance(export, BaseException):
                continue                                   # a refused borrow mid-stream is allowed; a leak is not
            assert marker.encode() not in export.content
            if export.status_code == 200 and export.content:
                try:
                    with tarfile.open(fileobj=io.BytesIO(export.content), mode="r:gz") as tar:
                        names = tar.getnames()
                        body = b"".join(tar.extractfile(n).read() for n in names if n.startswith("tables/"))
                except (tarfile.TarError, EOFError):
                    continue
                assert "tables/pl32_shared.jsonl" not in names and marker.encode() not in body
    finally:
        async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
            await conn.execute("DROP TABLE IF EXISTS public.pl32_shared")


# ── E3: work in flight and memory ─────────────────────────────────────────────


async def test_after_erase_a_replay_from_an_old_event_id_gets_a_reset_and_no_old_event(tenant_engine):
    client, app, as_user = tenant_engine
    me = as_user("e3-replay")
    key = tenancy.tenant_key("e3-replay")
    with tenancy.tenant_scope(key):
        async with events.subscribe() as sub:
            events.publish("transcript.partial", {"text": "what the person said"})
            old = await asyncio.wait_for(sub.next(), 5)
    assert (await client.delete("/v1/data", headers={**me, **ERASE})).status_code == 204

    with tenancy.tenant_scope(key):
        async with events.subscribe(last_event_id=old.id) as sub:
            first = await asyncio.wait_for(sub.next(), 5)
            assert not sub.pending and sub.empty()          # nothing else waiting: no old event
    assert first.type == RESET and first.data["reason"] == "unknown_id"
    assert "what the person said" not in repr(first)


async def test_no_row_of_the_tenant_reappears_after_a_delayed_task_completes(tenant_engine, monkeypatch):
    """A finalisation and a delegate job of the tenant are still running when the person erases. Neither writes
    after the erase; without the cancel they would re-provision the schema and write into it."""
    client, app, as_user = tenant_engine
    me = as_user("e3-delayed")
    key = tenancy.tenant_key("e3-delayed")
    assert (await client.get("/v1/meetings", headers=me)).status_code == 200
    pool = app.state.store.pool
    started = asyncio.Event()

    async def write_later(row_key: str) -> None:
        started.set()
        await asyncio.sleep(0.5)
        async with pool.connection() as conn:
            await conn.execute("INSERT INTO settings (key, value) VALUES (%s, '1'::jsonb)", (row_key,))
            await conn.commit()

    async def slow_finalize(_app, _meeting_id):
        await write_later("written-by-finalize")

    monkeypatch.setattr(finalize, "finalize_meeting", slow_finalize)
    lane = DelegateLane(runner=lambda _task: write_later("written-by-delegate"))
    delegate = SimpleNamespace(lane=lane, forget=lambda: [lane.cancel(t.task_id) for t in lane.active()],
                               drain=lambda: asyncio.sleep(0))
    host = _host(app, _binding(key))
    host.ports.delegate = delegate

    with tenancy.tenant_scope(key):
        finalize.schedule_finalize(app, "m1")
        lane.submit("look something up")
    await asyncio.wait_for(started.wait(), 5)

    assert (await client.delete("/v1/data", headers={**me, **ERASE})).status_code == 204
    await asyncio.sleep(1.0)                                  # past when both would have written
    assert not await _schema_exists(key)
    assert key not in pool.provisioner._ready


async def test_erase_drops_the_tenants_meeting_buses_and_profile_cache_entry(tenant_engine):
    client, app, as_user = tenant_engine
    me = as_user("e3-memory")
    key = tenancy.tenant_key("e3-memory")
    assert (await client.patch("/v1/assistant/profile", headers=me, json={"name": "Robin"})).status_code == 200
    with tenancy.tenant_scope(key):
        await meeting_bus.publish("m1", "status", {"state": "recording"})
    assert (key, "m1") in meeting_bus._buses
    assert app.state.profile_store.revision(key) >= 1          # the tenant's entry is cached

    assert (await client.delete("/v1/data", headers={**me, **ERASE})).status_code == 204
    assert not any(k[0] == key for k in meeting_bus._buses)
    assert app.state.profile_store.revision(key) == 0


# ── E5: export off the loop ───────────────────────────────────────────────────


async def test_the_event_loop_stays_responsive_during_a_large_export(client):
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO settings (key, value) SELECT 'k' || g, to_jsonb(repeat(md5(g::text), 8))"
            " FROM generate_series(1, 60000) g")
        await conn.commit()

    gaps: list[float] = []
    done = asyncio.Event()

    async def ticker():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    try:
        res = await client.get("/v1/data/export", headers=EXPORT)
    finally:
        done.set()
        await tick
    assert res.status_code == 200
    with tarfile.open(fileobj=io.BytesIO(res.content), mode="r:gz") as tar:
        lines = tar.extractfile("tables/settings.jsonl").read().splitlines()
    assert len(lines) == 60000
    assert max(gaps) < 0.15, f"the loop stalled for {max(gaps):.3f} s during the export"
