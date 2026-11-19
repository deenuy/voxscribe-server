"""Reminders and timers against Postgres (migrations 019 and 026): the store's SQL, the claim that
never hands a row to two deliverers, the routes, and tenant isolation (PL45, PL46).

The test database only, in throwaway schemas; the engine is created in-process with fake
providers and no wake listener. No model, no audio."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.local import tenancy
from app.local.store import Store
from app.local.voice import reminder_store as rs
from tests.conftest import _DB_URL
from tests.test_tenancy import keys, tenant_engine  # noqa: F401 - fixtures

NOW = datetime(2026, 11, 19, 19, 0, tzinfo=UTC)
ZONE = "America/Toronto"


@pytest.fixture
async def store(throwaway_schema):
    db = await Store.create(_DB_URL, schema=throwaway_schema, enforced=False)
    yield rs.ReminderStore.for_pool(db.pool)
    await db.close()


async def _add(store, minutes: float, *, kind=rs.KIND_REMINDER, text="stretch"):
    return await store.add(kind=kind, text=text if kind == rs.KIND_REMINDER else "",
                           label="ten minute" if kind == rs.KIND_TIMER else None,
                           fire_at=NOW + timedelta(minutes=minutes), time_zone=ZONE,
                           created_at=NOW, session_id="s1", turn_id="t1")


async def test_a_row_round_trips_with_its_zone_and_kind(store):
    item = await _add(store, 3)
    assert item.status == rs.PENDING and item.kind == rs.KIND_REMINDER
    assert item.due_at == "2026-11-19T14:03:00-05:00" and item.state == "scheduled"
    timer = await _add(store, 10, kind=rs.KIND_TIMER)
    assert timer.subject is None and timer.label == "ten minute"
    assert [i.id for i in await store.list_scheduled(rs.KIND_TIMER)] == [timer.id]
    assert await store.next_due_at() == item.fire_at
    await store.set_receipt(item.id, "call-1")


async def test_a_due_row_is_claimed_once_even_by_racing_deliverers(store):
    rows = [await _add(store, -1), await _add(store, -2), await _add(store, 5)]
    claims = await asyncio.gather(*(store.claim_due(NOW) for _ in range(4)))
    claimed = [i.id for batch in claims for i in batch]
    assert sorted(claimed) == sorted([rows[0].id, rows[1].id])       # each once, the future none
    assert await store.claim_due(NOW) == []
    assert await store.mark_delivered(rows[0].id, at=NOW, spoken=True, owed=False, missed=False)
    assert not await store.mark_delivered(rows[0].id, at=NOW, spoken=True, owed=False,
                                          missed=False)              # only a claimed row
    assert (await store.get(rows[0].id)).state == "fired"


async def test_cancel_only_while_scheduled(store):
    item = await _add(store, 5)
    assert (await store.cancel(item.id, at=NOW)).status == rs.CANCELLED
    assert await store.cancel(item.id, at=NOW) is None
    assert await store.claim_due(NOW + timedelta(hours=1)) == []


async def test_restart_helpers(store):
    stuck = await _add(store, -5)
    old = await _add(store, -24 * 60)
    await store.claim_due(NOW - timedelta(minutes=4))         # claims both, then "crashes"
    assert await store.release_stale_claims() == 2
    assert await store.expire_missed(NOW - timedelta(hours=12)) == 1
    assert (await store.get(old.id)).state == "missed"
    (again,) = await store.claim_due(NOW)
    assert again.id == stuck.id
    await store.mark_delivered(stuck.id, at=NOW, spoken=False, owed=True, missed=True)
    owed = await store.claim_owed(NOW - timedelta(hours=1))
    assert [i.id for i in owed] == [stuck.id] and owed[0].missed
    assert await store.claim_owed(NOW - timedelta(hours=1)) == []   # spoken once


# ── routes and isolation ─────────────────────────────────────────────────────────────────


async def _seed(app, key: str, text: str, minutes: float = 10) -> rs.Item:
    with tenancy.tenant_scope(key):
        return await rs.ReminderStore.for_pool(app.state.store.pool).add(
            kind=rs.KIND_REMINDER, text=text, fire_at=datetime.now(UTC) + timedelta(
                minutes=minutes), time_zone=ZONE, created_at=datetime.now(UTC))


async def test_each_person_sees_and_cancels_only_their_own(tenant_engine):  # noqa: F811
    client, app, as_user = tenant_engine
    alice, bob = as_user("alice-uuid"), as_user("bob-uuid")
    # Each tenant's schema exists once that person has made a request.
    assert (await client.get("/v1/reminders", headers=alice)).json() == {"items": []}
    assert (await client.get("/v1/reminders", headers=bob)).json() == {"items": []}
    a = await _seed(app, tenancy.tenant_key("alice-uuid"), "call the dentist")

    mine = (await client.get("/v1/reminders", headers=alice)).json()["items"]
    assert mine == [{"id": a.id, "kind": "reminder", "text": "call the dentist",
                     "dueAt": a.due_at, "state": "scheduled"}]
    assert (await client.get("/v1/reminders", headers=bob)).json() == {"items": []}
    # The voice path reads through the same pool: in Bob's scope, nothing of Alice's.
    with tenancy.tenant_scope(tenancy.tenant_key("bob-uuid")):
        assert await rs.ReminderStore.for_voice(app).list_scheduled() == []

    assert (await client.delete(f"/v1/reminders/{a.id}", headers=bob)).status_code == 404
    assert (await client.delete(f"/v1/reminders/{a.id}", headers=alice)).status_code == 204
    assert (await client.delete(f"/v1/reminders/{a.id}", headers=alice)).status_code == 204
    states = [i["state"] for i in (await client.get("/v1/reminders", headers=alice)).json()[
        "items"]]
    assert states == ["cancelled"]


async def test_a_fired_reminder_cannot_be_cancelled(tenant_engine):  # noqa: F811
    client, app, as_user = tenant_engine
    alice = as_user("alice-uuid")
    await client.get("/v1/reminders", headers=alice)
    key = tenancy.tenant_key("alice-uuid")
    item = await _seed(app, key, "stretch", minutes=-1)
    with tenancy.tenant_scope(key):
        store = rs.ReminderStore.for_pool(app.state.store.pool)
        await store.claim_due(datetime.now(UTC))
        await store.mark_delivered(item.id, at=datetime.now(UTC), spoken=True, owed=False,
                                   missed=False)
    res = await client.delete(f"/v1/reminders/{item.id}", headers=alice)
    assert res.status_code == 409 and res.json()["error"] == "conflict"
