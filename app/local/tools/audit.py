"""Tool audit rows and receipts (PL43, TOOLS-CONNECTORS.md section 2.6).

One tool_calls row per call through the tool runtime, whatever happened: executed, failed,
timed_out, cancelled, invalid_args or refused (proposed and declined are the read-back's, PL44).
A row with status `executed` is a receipt. The gate's action-claim rule (app/local/gate.py, G7)
asks turn_receipts() for the write receipts of the turn that is speaking; a failed call, a refusal,
a read, or a receipt from another turn backs nothing.

Rows live in the tenant's schema (the pool is the voice scope's, so a write outside the tenant's
scope is refused) and carry the tenant key too. A row holds arguments only as the tool's record
allows (stored fields, or a digest and a length) and never result text: a reference at most.
Nothing here logs an argument or a result.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import astuple, dataclass, fields

log = logging.getLogger("app.local.tools.audit")

EXECUTED = "executed"
FAILED = "failed"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"
INVALID_ARGS = "invalid_args"
REFUSED = "refused"
PROPOSED = "proposed"        # read-back offered, not yet confirmed (PL44)
DECLINED = "declined"        # read-back declined or abandoned (PL44)
STATUSES = (EXECUTED, FAILED, TIMED_OUT, CANCELLED, INVALID_ARGS, REFUSED, PROPOSED, DECLINED)

WRITE_SIDE_EFFECTS = ("write", "irreversible")


@dataclass(frozen=True)
class AuditRow:
    id: str
    meeting_id: str | None           # the session
    turn_id: str | None
    task_id: str | None
    tool: str
    tool_version: str | None
    side_effect: str | None
    connector: str | None
    status: str
    args: dict | None
    args_hash: str | None
    confirm_turn_id: str | None
    result_count: int | None
    result_ref: str | None
    error_code: str | None
    duration_ms: int
    policy_version: str | None
    called_at: str
    finished_at: str
    tenant: str | None


COLUMNS = tuple(f.name for f in fields(AuditRow))

_INSERT = (f"INSERT INTO tool_calls ({', '.join(COLUMNS)})"
           f" VALUES ({', '.join(['%s'] * len(COLUMNS))}) ON CONFLICT DO NOTHING")


@dataclass(frozen=True)
class Receipt:
    """The proof the gate checks: an executed call, by id, tool, version and turn."""

    call_id: str
    tool: str
    tool_version: str
    side_effect: str
    session_id: str | None
    turn_id: str | None
    confirm_turn_id: str | None
    result_ref: str | None
    finished_at: str


def new_id() -> str:
    return str(uuid.uuid4())


async def write(pool, row: AuditRow) -> bool:
    """Write one row. Never raises; False when the row could not be written."""
    if pool is None:
        log.warning("tools audit: no pool, row not written tool=%s status=%s", row.tool,
                    row.status)
        return False
    values = list(astuple(row))
    values[COLUMNS.index("args")] = json.dumps(row.args) if row.args is not None else None
    try:
        async with pool.connection() as conn:
            await conn.execute(_INSERT, tuple(values))
            await conn.commit()
    except Exception as exc:  # noqa: BLE001 - the audit never breaks the turn; the gate fails closed
        log.warning("tools audit: row not written tool=%s status=%s (%s)", row.tool, row.status,
                    type(exc).__name__)
        return False
    return True


_RECEIPTS = ("SELECT DISTINCT tool FROM tool_calls WHERE meeting_id=%s"
             " AND (turn_id=%s OR confirm_turn_id=%s)"
             f" AND status='{EXECUTED}' AND side_effect IN ('write', 'irreversible')")


async def turn_receipts(pool, session_id: str, turn_id: str | None) -> list[str]:
    """The tools with an executed write receipt for `turn_id` in this session: the turn the call
    was made in, or the turn whose yes confirmed it. Raises on a failed lookup, so the gate
    fails closed."""
    if not turn_id:
        return []
    async with pool.connection() as conn:
        cur = await conn.execute(_RECEIPTS, (session_id, turn_id, turn_id))
        rows = await cur.fetchall()
    return [r[0] for r in rows]


async def session_receipts(pool, session_id: str) -> list[str]:
    """Every tool with an executed write receipt in the session. Not what the gate reads."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT DISTINCT tool FROM tool_calls WHERE meeting_id=%s"
            f" AND status='{EXECUTED}' AND side_effect IN ('write', 'irreversible')",
            (session_id,))
        rows = await cur.fetchall()
    return [r[0] for r in rows]
