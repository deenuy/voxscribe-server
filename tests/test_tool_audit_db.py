"""PL43 against Postgres: the audit row's columns (migrations 024 and 025) and the receipt lookup the
gate runs, on a throwaway schema in the test database. Isolation off; fake executors."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.local.store import Store
from app.local.tools import audit, builtin, runtime
from app.local.tools.runtime import CallContext, ToolResult
from tests.conftest import _DB_URL
from tests.voice.unit.tool_fakes import load_profile, settings

FIRE_AT = "2026-11-19T15:00:00+00:00"


@pytest.fixture
async def app(throwaway_schema):
    store = await Store.create(_DB_URL, schema=throwaway_schema, enforced=False)
    yield SimpleNamespace(state=SimpleNamespace(store=store, settings=settings(),
                                                profile=load_profile("personal")))
    await store.close()


async def _timer(app, turn_id: str, *, fails: bool = False):
    async def execute(_args):
        if fails:
            raise RuntimeError("store down")
        return ToolResult(fields={"timer_id": "r1"}, ref="r1")

    return await runtime.call(app, builtin.TIMER_START,
                              {"duration_s": 300, "fire_at": FIRE_AT, "length": "Five minute",
                               "time_zone": "America/Toronto"},
                              CallContext(session_id="s1", turn_id=turn_id, task_id=None),
                              execute)


async def test_rows_and_receipts_in_postgres(app):
    ok = await _timer(app, "t1")
    assert ok.executed and ok.row_written and ok.receipt is not None
    failed = await _timer(app, "t2", fails=True)
    assert failed.status == audit.FAILED and failed.row_written
    refused = await runtime.call(app, builtin.WEB_SEARCH, {"query": "weather"},
                                 CallContext(session_id="s1", turn_id="t3"), _never)
    assert refused.status == audit.REFUSED and refused.row_written   # privacy offline

    pool = app.state.store.pool
    assert await audit.turn_receipts(pool, "s1", "t1") == [builtin.TIMER_START]
    assert await audit.turn_receipts(pool, "s1", "t2") == []
    assert await audit.turn_receipts(pool, "s1", "t3") == []
    assert await audit.turn_receipts(pool, "other", "t1") == []
    assert await audit.session_receipts(pool, "s1") == [builtin.TIMER_START]

    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT tool, status, tool_version, side_effect, args, args_hash, result_ref,"
            " error_code, duration_ms, policy_version, finished_at, query"
            " FROM tool_calls ORDER BY called_at")
        rows = await cur.fetchall()
    assert [(r[0], r[1]) for r in rows] == [(builtin.TIMER_START, "executed"),
                                            (builtin.TIMER_START, "failed"),
                                            (builtin.WEB_SEARCH, "refused")]
    executed = rows[0]
    assert executed[2] and executed[3] == "write" and executed[4]["length"] == "Five minute"
    assert len(executed[5]) == 64 and executed[6] == "r1" and executed[8] >= 0
    assert executed[9] and executed[10] and executed[11] is None
    assert rows[1][7] == "raised:RuntimeError" and rows[2][7] == "privacy_offline"


async def _never(_args):
    raise AssertionError("a refused tool ran")
