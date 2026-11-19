"""The tool runtime (PL43): the one way the engine runs a tool.

    call(app, name, args, ctx, execute)

  1. the record       an unknown tool is refused
  2. the registry     the session's product mode and, for a remote tool, the network decision
                      (PL41); a tool that is not offered is refused, so calling it directly
                      runs nothing either
  3. confirmation     a tool that asks first runs only with the turn that confirmed it
  4. the arguments    checked strictly against the record; any miss is invalid_args, 0 calls
  5. the executor     bounded by the record's timeout_ms: executed, failed, timed_out, cancelled
  6. the result       checked against the record's result schema; a mismatch is failed
  7. the audit row    one per call, whatever happened (app/local/tools/audit.py)

It returns a ToolOutcome: the status, the scripted line to speak (rendered from the record's
templates with validated arguments and, on success, the receipt), the checked result, and the
receipt when the call executed and its row was written. The line is a candidate like any other:
the caller hands it to the reply stage, which gates it. A success line is spoken as a claim only
because its receipt is in the ledger; when the row could not be written, the line is the record's
timeout line instead ("I'm not sure that was saved"), since nothing proves the action happened.

Logs carry the tool, its status, the duration and an error code: never an argument or a result.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.local import tenancy
from app.local.tools import audit
from app.local.tools.network import network_decision, product_mode
from app.local.tools.registry import REASON_UNKNOWN_TOOL, ToolRegistry, tool_registry
from app.local.tools.spec import ArgsError, ToolSpec, args_digest

log = logging.getLogger("app.local.tools.runtime")

UNKNOWN_TOOL_LINE = "I can't do that here."
REASON_NEEDS_CONFIRMATION = "needs_confirmation"
REASON_BAD_RESULT = "bad_result"
REASON_RECEIPT_UNWRITTEN = "receipt_unwritten"


@dataclass(frozen=True)
class CallContext:
    session_id: str | None
    turn_id: str | None
    task_id: str | None = None
    confirm_turn_id: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """What an executor returns. `fields` is checked against the record's result schema and may
    fill the success line; `ref` goes on the audit row; `items` reach only the caller (a list
    the caller renders), never the row."""

    fields: Mapping[str, Any]
    ref: str | None = None
    items: tuple = ()


class ToolFailed(Exception):
    """An executor's own failure, with a code for the audit row (never text from the person)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


Execute = Callable[[dict], Awaitable[ToolResult | Mapping[str, Any]]]


@dataclass(frozen=True)
class ToolOutcome:
    tool: str
    status: str
    line: str
    result: Mapping[str, Any] | None = None
    items: tuple = ()
    receipt: audit.Receipt | None = None
    error_code: str | None = None
    row_written: bool = False
    spec: ToolSpec | None = field(default=None, repr=False)

    @property
    def executed(self) -> bool:
        return self.status == audit.EXECUTED


def _now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _audit_pool(app):
    """The voice scope's pool. A call outside the tenant's scope is refused here, before it runs;
    an app with no store has no ledger (the row is not written, and the gate fails closed)."""
    from app.local.voice.adapters.scope import scoped_pool  # lazy: no import cycle

    try:
        return scoped_pool(app)
    except tenancy.TenantRequiredError:
        raise
    except Exception:  # noqa: BLE001 - no store on this app
        return None


def _policy_version(app) -> str | None:
    try:
        from app.local.voice.adapters.gate_policy import session_policy

        return session_policy(app).stamp.policy_version
    except Exception:  # noqa: BLE001 - recorded as unknown
        return None


async def _refusal(app, registry: ToolRegistry, spec: ToolSpec) -> tuple[str, str] | None:
    """(code, line) when the registry does not offer `spec` to this session; None otherwise."""
    mode = product_mode(app)
    decision = await network_decision(app) if spec.network == "remote" else None
    code = registry.permits(spec, mode, decision)
    if code is None:
        return None
    if decision is not None and not decision.allowed:
        return code, decision.line
    return code, spec.templates["refused"]


async def offered(app) -> tuple[ToolSpec, ...]:
    """Which tools this session may use: the registry's answer for the product mode and, when a
    remote tool is in play, the network decision."""
    registry = tool_registry(app)
    mode = product_mode(app)
    decision = await network_decision(app) if registry.needs_network_decision(mode) else None
    return registry.offered(mode, decision)


async def preflight(app, name: str, ctx: CallContext) -> ToolOutcome | None:
    """The registry's check alone, before work is queued for a tool (the delegate lane). None
    when the tool may run; otherwise the refusal, already written to the audit."""
    registry = tool_registry(app)
    spec = registry.get(name)
    pool = _audit_pool(app)
    refusal = ((REASON_UNKNOWN_TOOL, UNKNOWN_TOOL_LINE) if spec is None
               else await _refusal(app, registry, spec))
    if refusal is None:
        return None
    return await _finish(app, pool, spec, name, ctx, status=audit.REFUSED, line=refusal[1],
                         error_code=refusal[0], called_at=_now_z(), started=time.monotonic())


async def call(app, name: str, raw_args: Any, ctx: CallContext,
               execute: Execute) -> ToolOutcome:
    """Run one tool through the registry. Returns the outcome; raises only when cancelled (after
    its row is written) or when called outside the tenant's scope (before anything runs)."""
    registry = tool_registry(app)
    spec = registry.get(name)
    pool = _audit_pool(app)
    called_at, started = _now_z(), time.monotonic()

    def done(status: str, line: str, **kw) -> Awaitable[ToolOutcome]:
        return _finish(app, pool, spec, name, ctx, status=status, line=line,
                       called_at=called_at, started=started, **kw)

    if spec is None:
        return await done(audit.REFUSED, UNKNOWN_TOOL_LINE, error_code=REASON_UNKNOWN_TOOL)
    refusal = await _refusal(app, registry, spec)
    if refusal is not None:
        return await done(audit.REFUSED, refusal[1], error_code=refusal[0])
    if spec.confirm in ("ask", "ask_strict") and not ctx.confirm_turn_id:
        return await done(audit.REFUSED, spec.templates["refused"],
                          error_code=REASON_NEEDS_CONFIRMATION)
    try:
        args = spec.validate_args(raw_args)
    except ArgsError as exc:
        return await done(audit.INVALID_ARGS, spec.templates["invalid"],
                          error_code=f"{exc.code}:{exc.field}" if exc.field else exc.code,
                          raw_args=raw_args)

    timeout_s = spec.timeout_ms / 1000
    try:
        async with asyncio.timeout(timeout_s) as deadline:
            out = await execute(dict(args))
    except asyncio.CancelledError:
        await asyncio.shield(done(audit.CANCELLED, spec.render("timeout", args), args=args,
                                  error_code="cancelled"))
        raise
    except TimeoutError:
        if deadline.expired():
            return await done(audit.TIMED_OUT, spec.render("timeout", args), args=args,
                              error_code="timeout")
        return await done(audit.FAILED, spec.render("failure", args), args=args,
                          error_code="raised:TimeoutError")
    except tenancy.TenantRequiredError:
        raise
    except ToolFailed as exc:
        return await done(audit.FAILED, spec.render("failure", args), args=args,
                          error_code=exc.code)
    except Exception as exc:  # noqa: BLE001 - an executor's failure is an outcome, not a crash
        return await done(audit.FAILED, spec.render("failure", args), args=args,
                          error_code=f"raised:{type(exc).__name__}")

    if not isinstance(out, ToolResult):
        out = ToolResult(fields=out)
    try:
        result = spec.validate_result(out.fields)
    except ArgsError as exc:
        return await done(audit.FAILED, spec.render("failure", args), args=args,
                          error_code=f"{REASON_BAD_RESULT}:{exc.field or exc.code}")
    return await done(audit.EXECUTED, spec.render("success", {**args, **result}), args=args,
                      result=result, ref=out.ref, items=out.items)


async def _finish(app, pool, spec: ToolSpec | None, name: str, ctx: CallContext, *,
                  status: str, line: str, called_at: str, started: float,
                  error_code: str | None = None, args: Mapping | None = None,
                  raw_args: Any = None, result: Mapping | None = None, ref: str | None = None,
                  items: tuple = ()) -> ToolOutcome:
    """Write the call's row and build its outcome."""
    duration_ms = int((time.monotonic() - started) * 1000)
    finished_at = _now_z()
    if args is not None:
        kept, digest = spec.audit_args(args), args_digest(args)
    else:
        kept, digest = None, _raw_digest(raw_args)
    count = None
    if result is not None:
        count = next((result[k] for k in ("result_count", "count") if k in result), None)
    row = audit.AuditRow(
        id=audit.new_id(), meeting_id=ctx.session_id, turn_id=ctx.turn_id, task_id=ctx.task_id,
        tool=name, tool_version=spec.version if spec else None,
        side_effect=spec.side_effect if spec else None, connector=None, status=status,
        args=kept, args_hash=digest, confirm_turn_id=ctx.confirm_turn_id, result_count=count,
        result_ref=ref, error_code=error_code, duration_ms=duration_ms,
        policy_version=_policy_version(app), called_at=called_at, finished_at=finished_at,
        tenant=tenancy.current_tenant())
    written = await audit.write(pool, row)
    receipt = None
    if status == audit.EXECUTED:
        if written:
            receipt = audit.Receipt(call_id=row.id, tool=name, tool_version=row.tool_version,
                                    side_effect=row.side_effect, session_id=ctx.session_id,
                                    turn_id=ctx.turn_id, confirm_turn_id=ctx.confirm_turn_id,
                                    result_ref=ref, finished_at=finished_at)
        elif spec.side_effect in audit.WRITE_SIDE_EFFECTS:
            # Done, but nothing proves it: never claim it (section 2.11).
            line, error_code = spec.render("timeout", args), REASON_RECEIPT_UNWRITTEN
    log.info("tools: call tool=%s status=%s duration_ms=%d error=%s receipt=%s", name, status,
             duration_ms, error_code or "none", receipt is not None)
    return ToolOutcome(tool=name, status=status, line=line, result=result, items=items,
                       receipt=receipt, error_code=error_code, row_written=written, spec=spec)


def _raw_digest(raw: Any) -> str | None:
    if raw is None:
        return None
    try:
        return args_digest(raw)
    except (TypeError, ValueError):
        return None
