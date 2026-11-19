"""Erase versus the append-only gate ledger (PL32).

A person's right to erase their data wins over the ledger, through one door: the erase transaction sets
voxscribe.erase_tenant_data with SET LOCAL, and the ledger's trigger lets DELETE and TRUNCATE through only then.
Everywhere else the ledger stays closed: UPDATE never passes, and the setting ends with the erase transaction.
"""

import re
from pathlib import Path

import psycopg
import pytest

from app.local import tenancy
from tests import test_tenancy
from tests.testdb import TEST_DATABASE_URL

keys, tenant_engine = test_tenancy.keys, test_tenancy.tenant_engine   # fixtures, shared with test_tenancy

APP = Path(__file__).resolve().parent.parent / "app" / "local"
ERASE = {"X-Confirm": "delete-all"}
SETTING = "voxscribe.erase_tenant_data"


async def _seed_gate_record(conn, rid: str = "g1") -> None:
    await conn.execute(
        "INSERT INTO gate_records (id, event, session_id, mode, policy_version, recorded_at)"
        " VALUES (%s, 'outcome', 's1', 'default', 'v1', '2026-11-18T00:00:00Z')", (rid,))
    await conn.commit()


async def _count(conn, table: str) -> int:
    return (await (await conn.execute(f"SELECT count(*) FROM {table}")).fetchone())[0]


def test_only_the_erase_path_names_the_ledger_door():
    """No engine module but the erase router (and the migration that defines the door) mentions the setting."""
    holders = sorted(str(p.relative_to(APP)) for p in APP.rglob("*") if p.suffix in (".py", ".sql")
                     and SETTING in p.read_text())
    assert holders == ["migrations/021_gate_records.sql", "routers/data.py"]
    data = (APP / "routers" / "data.py").read_text()
    # Set only transaction-locally, never for the session.
    assert re.findall(r"SET\s+(?:LOCAL\s+)?\{_ERASE_SETTING\}", data) == ["SET LOCAL {_ERASE_SETTING}"]


async def test_ordinary_code_still_cannot_update_delete_or_truncate_the_ledger(client):
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await _seed_gate_record(conn)
    for statement in ("UPDATE gate_records SET decision = 'pass'", "DELETE FROM gate_records",
                      "TRUNCATE gate_records", "TRUNCATE gate_records CASCADE"):
        async with store.pool.connection() as conn:
            with pytest.raises(psycopg.errors.RestrictViolation):
                await conn.execute(statement)
            await conn.rollback()
    async with store.pool.connection() as conn:
        assert await _count(conn, "gate_records") == 1


async def test_the_door_never_opens_for_update(client):
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await _seed_gate_record(conn)
        async with conn.transaction():
            await conn.execute(f"SET LOCAL {SETTING} = 'on'")
            with pytest.raises(psycopg.errors.RestrictViolation):
                async with conn.transaction():
                    await conn.execute("UPDATE gate_records SET decision = 'pass'")


async def test_erase_leaves_no_gate_record_and_closes_the_door_behind_it(client):
    """Isolation off: erase empties the ledger with everything else; afterwards the ledger is closed again on every
    pooled connection, including the one erase used."""
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await _seed_gate_record(conn, "g1")
        await _seed_gate_record(conn, "g2")

    assert (await client.delete("/v1/data", headers=ERASE)).status_code == 204

    async with store.pool.connection() as conn:
        assert await _count(conn, "gate_records") == 0
        assert await _count(conn, "gate_decisions") == 0
        assert await _count(conn, "schema_migrations") > 0
        await _seed_gate_record(conn, "g3")
    for _ in range(store.pool.max_size + 1):       # every connection the pool may lend
        async with store.pool.connection() as conn:
            value = (await (await conn.execute("SELECT current_setting(%s, true)", (SETTING,))).fetchone())[0]
            assert value in (None, "")
            with pytest.raises(psycopg.errors.RestrictViolation):
                await conn.execute("DELETE FROM gate_records")
            await conn.rollback()


async def test_erase_with_isolation_on_drops_the_schema_and_its_gate_records(tenant_engine):
    client, app, as_user = tenant_engine
    me = as_user("erase-ledger-user")
    key = tenancy.tenant_key("erase-ledger-user")
    assert (await client.get("/v1/meetings", headers=me)).status_code == 200   # provisions the schema
    with tenancy.tenant_scope(key):
        async with app.state.store.pool.connection() as conn:
            await _seed_gate_record(conn)
            assert await _count(conn, "gate_records") == 1
    assert key in app.state.store.pool.provisioner._ready

    res = await client.delete("/v1/data", headers={**me, **ERASE})
    assert res.status_code == 204, res.text

    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        gone = await (await conn.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = %s", (key,))).fetchone()
    assert gone == (0,)
    assert key not in app.state.store.pool.provisioner._ready
    # The next request starts the person over in an empty schema: no ledger row came back.
    assert (await client.get("/v1/meetings", headers=me)).status_code == 200
    with tenancy.tenant_scope(key):
        async with app.state.store.pool.connection() as conn:
            assert await _count(conn, "gate_records") == 0
