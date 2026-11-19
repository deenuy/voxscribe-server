"""PL43: the tool runtime, its audit rows and the receipts the gate checks (section 2.6).

Fake executors and the in-memory pool from tool_fakes; the real registry, records, gate and
session store. No database, no model, no network.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from app.local import gate as floor
from app.local.tools import audit, builtin, runtime
from app.local.tools.runtime import CallContext, ToolFailed, ToolResult
from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from tests.voice.unit.tool_fakes import ToolPool, load_profile, make_app

SECRET = "call the dentist about my tooth"
FIRE_AT = "2026-11-19T15:00:00+00:00"
CTX = CallContext(session_id="s1", turn_id="t1")


class Executor:
    """Counts its calls; returns `fields`, raises `raises`, or sleeps past the deadline."""

    def __init__(self, fields=None, *, raises: Exception | None = None, sleep_s: float = 0):
        self.fields = fields if fields is not None else {"reminder_id": "r1", "when": "3:00 PM"}
        self.raises, self.sleep_s = raises, sleep_s
        self.calls: list[dict] = []

    async def __call__(self, args: dict):
        self.calls.append(args)
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.raises is not None:
            raise self.raises
        return ToolResult(fields=self.fields, ref=self.fields.get("reminder_id"))


@pytest.fixture
def pool() -> ToolPool:
    return ToolPool()


@pytest.fixture
def app(pool):
    return make_app(pool, profile=load_profile("personal"))


def _call(app, args, execute, name=builtin.REMINDER_CREATE, ctx=CTX):
    return runtime.call(app, name, args, ctx, execute)


# ── one row per call, whatever happened ──────────────────────────────────────


async def test_an_executed_call_writes_a_receipt_row(app, pool):
    execute = Executor()
    outcome = await _call(app, {"text": SECRET, "fire_at": FIRE_AT}, execute)
    assert outcome.executed and outcome.receipt is not None
    assert outcome.line == f"Got it, I'll remind you at 3:00 PM: {SECRET}."
    (row,) = pool.tool_calls
    spec = app.state.tool_registry.get(builtin.REMINDER_CREATE)
    assert row["status"] == audit.EXECUTED and row["tool_version"] == spec.version
    assert row["side_effect"] == "write" and row["result_ref"] == "r1"
    assert row["meeting_id"] == "s1" and row["turn_id"] == "t1"
    assert isinstance(row["duration_ms"], int) and row["called_at"] and row["finished_at"]
    assert row["policy_version"] == session_policy(app).stamp.policy_version
    assert SECRET not in str(row)                     # hashed, per the record's audit list
    assert FIRE_AT in row["args"] and len(row["args_hash"]) == 64


@pytest.mark.parametrize("args,code", [
    ({"text": SECRET, "fire_at": FIRE_AT, "priority": "high"}, "unknown_field:priority"),
    ({"text": 42, "fire_at": FIRE_AT}, "wrong_type:text"),
    ({"text": "x" * 501, "fire_at": FIRE_AT}, "too_long:text"),
    ({"text": SECRET, "fire_at": "at nine"}, "pattern_mismatch:fire_at"),
])
async def test_bad_arguments_make_no_call_and_one_invalid_args_row(app, pool, args, code):
    execute = Executor()
    outcome = await _call(app, args, execute)
    assert execute.calls == []
    assert outcome.status == audit.INVALID_ARGS and outcome.receipt is None
    assert outcome.line == "I couldn't tell when to remind you."
    (row,) = pool.tool_calls
    assert (row["status"], row["error_code"]) == (audit.INVALID_ARGS, code)
    assert row["args"] is None and row["tool_version"]


async def test_an_out_of_range_value_is_invalid(pool):
    record = {**next(r for r in builtin.FIRST_PARTY if r["name"] == builtin.REMINDER_LIST),
              "name": "count.check",
              "args": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1,
                                                              "maximum": 9}},
                       "required": ["n"], "additionalProperties": False},
              "audit": {"store": ["n"], "hash": []}}
    from app.local.tools.registry import ToolRegistry

    app = make_app(pool, profile=load_profile("personal"))
    app.state.tool_registry = ToolRegistry([record])
    execute = Executor({"count": 1})
    outcome = await runtime.call(app, "count.check", {"n": 10}, CTX, execute)
    assert outcome.status == audit.INVALID_ARGS and execute.calls == []
    assert pool.tool_calls[0]["error_code"] == "out_of_range:n"


async def test_a_tool_not_allowed_in_the_mode_is_not_offered_and_a_direct_call_is_refused(pool):
    app = make_app(pool, profile=load_profile("senior_care"))
    assert builtin.WEB_SEARCH not in [s.name for s in await runtime.offered(app)]
    execute = Executor({"text": "", "result_count": 0})
    outcome = await runtime.call(app, builtin.WEB_SEARCH, {"query": "weather"}, CTX, execute)
    assert outcome.status == audit.REFUSED and execute.calls == []
    assert [(r["status"], r["error_code"]) for r in pool.tool_calls] == [
        (audit.REFUSED, "mode_restricted")]


async def test_an_unknown_tool_is_refused_and_recorded(app, pool):
    outcome = await _call(app, {}, Executor(), name="email.send")
    assert outcome.status == audit.REFUSED and outcome.line == runtime.UNKNOWN_TOOL_LINE
    assert pool.tool_calls[0]["error_code"] == "unknown_tool"


async def test_a_call_past_its_timeout_is_timed_out_and_says_so(app, pool):
    execute = Executor(sleep_s=0.2)
    from app.local.tools import registry as reg

    fast = reg.ToolRegistry([{**next(r for r in builtin.FIRST_PARTY
                                      if r["name"] == builtin.REMINDER_CREATE),
                              "timeout_ms": 20}])
    app.state.tool_registry = fast
    outcome = await _call(app, {"text": SECRET, "fire_at": FIRE_AT}, execute)
    assert outcome.status == audit.TIMED_OUT and outcome.receipt is None
    assert outcome.line == "I'm not sure that reminder was saved."
    assert floor.find_action_claim(outcome.line) is None
    (row,) = pool.tool_calls
    assert (row["status"], row["error_code"]) == (audit.TIMED_OUT, "timeout")
    # The timeout line passes the gate and is what would be spoken.
    gate = CorePolicyGate(session_policy(app))
    assert gate.check(outcome.line).decision == "pass"


@pytest.mark.parametrize("raises,code", [(ToolFailed("store_down"), "store_down"),
                                         (RuntimeError("boom"), "raised:RuntimeError")])
async def test_a_failing_executor_is_failed_and_never_confirmed(app, pool, raises, code):
    outcome = await _call(app, {"text": SECRET, "fire_at": FIRE_AT}, Executor(raises=raises))
    assert outcome.status == audit.FAILED and outcome.receipt is None
    assert outcome.line == "Sorry, I couldn't save that reminder."
    assert floor.find_action_claim(outcome.line) is None
    assert pool.tool_calls[0]["error_code"] == code


async def test_a_result_outside_its_schema_is_failed(app, pool):
    outcome = await _call(app, {"text": SECRET, "fire_at": FIRE_AT},
                          Executor({"reminder_id": "r1"}))           # no "when"
    assert outcome.status == audit.FAILED
    assert pool.tool_calls[0]["error_code"] == "bad_result:when"


async def test_a_cancelled_call_still_writes_its_row(app, pool):
    task = asyncio.create_task(_call(app, {"text": SECRET, "fire_at": FIRE_AT},
                                     Executor(sleep_s=1)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [r["status"] for r in pool.tool_calls] == [audit.CANCELLED]


async def test_an_executed_write_whose_row_was_not_written_is_never_claimed(app, pool):
    class NoInsert(ToolPool):
        async def execute(self, sql, params=()):
            if sql.startswith("INSERT INTO tool_calls"):
                raise OSError("ledger down")
            return await super().execute(sql, params)

    broken = make_app(NoInsert(), profile=load_profile("personal"))
    outcome = await _call(broken, {"text": SECRET, "fire_at": FIRE_AT}, Executor())
    assert outcome.executed and outcome.receipt is None and not outcome.row_written
    assert outcome.line == "I'm not sure that reminder was saved."


async def test_an_asking_tool_does_not_run_without_its_confirming_turn(app, pool):
    from app.local.tools.registry import ToolRegistry

    record = next(r for r in builtin.FIRST_PARTY if r["name"] == builtin.REMINDER_CREATE)
    record = {**record, "confirm": "ask",
              "templates": {**record["templates"], "readback": "{text}. Shall I set it?"}}
    app.state.tool_registry = ToolRegistry([record])
    execute = Executor()
    outcome = await _call(app, {"text": SECRET, "fire_at": FIRE_AT}, execute)
    assert outcome.status == audit.REFUSED and execute.calls == []
    assert pool.tool_calls[0]["error_code"] == "needs_confirmation"
    confirmed = CallContext(session_id="s1", turn_id="t1", confirm_turn_id="t2")
    outcome = await _call(app, {"text": SECRET, "fire_at": FIRE_AT}, execute, ctx=confirmed)
    assert outcome.executed and pool.tool_calls[-1]["confirm_turn_id"] == "t2"


async def test_no_argument_or_result_text_reaches_the_logs(app, caplog):
    with caplog.at_level(logging.DEBUG):
        await _call(app, {"text": SECRET, "fire_at": FIRE_AT}, Executor())
        await _call(app, {"text": SECRET, "fire_at": FIRE_AT, "x": 1}, Executor())
        await _call(app, {"text": SECRET, "fire_at": FIRE_AT},
                    Executor(raises=RuntimeError(SECRET)))
    assert caplog.records and SECRET not in caplog.text and "dentist" not in caplog.text


# ── the gate's proof ─────────────────────────────────────────────────────────


TIMER_CLAIM = "I've set a timer for five minutes."


async def _receipts(app, turn_id: str) -> list[str]:
    return await PostgresSessionStore(app).turn_receipts("s1", turn_id)


async def _timer(app, *, turn_id: str, raises: Exception | None = None):
    return await runtime.call(app, builtin.TIMER_START,
                              {"label": "5 minutes", "fire_at": FIRE_AT},
                              CallContext(session_id="s1", turn_id=turn_id),
                              Executor({"timer_id": "r9"}, raises=raises))


async def test_a_timer_claim_passes_only_with_an_executed_timer_receipt_in_this_turn(app):
    gate = CorePolicyGate(session_policy(app))
    assert gate.check(TIMER_CLAIM, await _receipts(app, "t1")).decision == "replace"
    await _timer(app, turn_id="t1")
    assert gate.check(TIMER_CLAIM, await _receipts(app, "t1")).decision == "pass"
    # The same receipt does not back the claim in a later turn.
    verdict = gate.check(TIMER_CLAIM, await _receipts(app, "t2"))
    assert (verdict.decision, verdict.category) == ("replace", floor.ACTION_CLAIM_REASON)


async def test_a_failed_timer_backs_no_claim(app):
    await _timer(app, turn_id="t1", raises=RuntimeError("store down"))
    gate = CorePolicyGate(session_policy(app))
    assert await _receipts(app, "t1") == []
    assert gate.check(TIMER_CLAIM, await _receipts(app, "t1")).decision == "replace"


async def test_a_read_and_a_search_back_no_claim(app, pool):
    await runtime.call(app, builtin.REMINDER_LIST, {}, CTX, Executor({"count": 2}))
    pool.privacy = "connectors"
    await runtime.call(app, builtin.WEB_SEARCH, {"query": "weather"}, CTX,
                       Executor({"text": "", "result_count": 0}))
    assert [r["status"] for r in pool.tool_calls] == [audit.EXECUTED, audit.EXECUTED]
    assert await _receipts(app, "t1") == []


async def test_a_receipt_from_the_confirming_turn_backs_the_claim(app):
    await runtime.call(app, builtin.TIMER_START, {"label": "5 minutes", "fire_at": FIRE_AT},
                       CallContext(session_id="s1", turn_id="t1", confirm_turn_id="t2"),
                       Executor({"timer_id": "r9"}))
    assert await _receipts(app, "t2") == [builtin.TIMER_START]


def test_the_templates_forms_are_claims_and_the_failure_lines_are_not():
    for line in ("Got it, timer set for 5 minutes.", "Reminder set for noon.",
                 "I've added it to your calendar. It's added to your calendar."):
        assert floor.find_action_claim(line) is not None, line
    for record in builtin.FIRST_PARTY:
        for key in ("failure", "timeout", "refused", "invalid"):
            assert floor.find_action_claim(record["templates"][key]) is None, (record["name"],
                                                                                key)
