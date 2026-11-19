"""The steps a scribe goes through after Stop, as the app shows them: transcript, summary, tasks.

Each step is in one state, `working`, `ready`, `failed` or `skipped`, with a reason: a short code (REASONS), never
model output or transcript text, and null when there is nothing to add. A state change is published as the
`meeting.pipeline` event and kept in `meeting_steps`, so GET /v1/meetings/{id}/pipeline gives a screen opened later
the same answer without waiting for an event.

Who hears it: events.publish() goes to the tenant in scope. Every caller here runs in the meeting's tenant's scope:
a request, the live socket (scoped at connect), a voice session (scoped by the host), or a finalisation or
regeneration task, which inherits the scope of the request or session that created it.

Where each step's state comes from (read()):
  transcript  the recorded row (written when finalisation starts, once every segment is stored); without one, the
              meeting's status: recording or processing is working, done is ready (or skipped, no_speech)
  summary     summaries.status, the state the summary has always had; the recorded row only adds its reason. A
              summary edited by hand is ready, edited_by_hand. One left pending or running by a run this process
              no longer has (a restart during a regeneration) is failed, interrupted
  tasks       the recorded row; without one, derived from the summary (tasks are drafted after it). A row left
              working by a run this process no longer has is failed, interrupted

A step that has not started yet is `working` with reason `queued` (the wire has no separate waiting state).

Runs in progress (summary and tasks) are kept per tenant and meeting in this process, so a second request to
regenerate attaches to the run that is writing (routers/meetings.py) instead of failing or starting another.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from psycopg.rows import dict_row

from . import events, tenancy

log = logging.getLogger(__name__)

EVENT = "meeting.pipeline"
STEPS = ("transcript", "summary", "tasks")
WORKING, READY, FAILED, SKIPPED = "working", "ready", "failed", "skipped"

# Reasons: short codes, the only thing besides the state that leaves this module.
QUEUED = "queued"                       # the step has not started; an earlier one is still working
RECORDING = "recording"                 # the scribe is still recording
NOT_STARTED = "not_started"             # the scribe was never started
TOO_SHORT = "too_short"                 # fewer words than the summary needs (VOXSCRIBE_SUMMARY_MIN_WORDS)
NO_SPEECH = "no_speech"                 # nothing was said
MODEL_UNAVAILABLE = "model_unavailable"  # the on-device AI is not running or could not be reached
MODEL_TIMEOUT = "model_timeout"         # the AI did not answer in time
MODEL_ERROR = "model_error"             # the AI failed in some other way
UNUSABLE_OUTPUT = "unusable_output"     # the AI answered, but not with something usable
EDITED_BY_HAND = "edited_by_hand"       # a person wrote or changed the summary
SUMMARY_FAILED = "summary_failed"       # tasks were not drafted because the summary failed
NOT_GENERATED = "not_generated"         # never asked for (an imported scribe); Generate can still do it
INTERRUPTED = "interrupted"             # the recording or the run was cut off (a restart)

REASONS = frozenset({QUEUED, RECORDING, NOT_STARTED, TOO_SHORT, NO_SPEECH, MODEL_UNAVAILABLE, MODEL_TIMEOUT,
                     MODEL_ERROR, UNUSABLE_OUTPUT, EDITED_BY_HAND, SUMMARY_FAILED, NOT_GENERATED, INTERRUPTED})

# summaries.status -> (state, the reason when none was recorded)
_SUMMARY = {
    "pending": (WORKING, QUEUED),
    "running": (WORKING, None),
    "ready": (READY, None),
    "skipped": (SKIPPED, TOO_SHORT),
    "unavailable": (FAILED, MODEL_UNAVAILABLE),
    "failed": (FAILED, MODEL_ERROR),
}


def summary_state(status: str, reason: str | None = None) -> tuple[str, str | None]:
    """The pipeline's (state, reason) for a summaries.status, with the reason the writer gave, if any."""
    state, default = _SUMMARY.get(status, (WORKING, QUEUED))
    return state, reason if reason is not None else default


def step(meeting_id: str, name: str, state: str, reason: str | None) -> dict:
    return {"meetingId": meeting_id, "step": name, "state": state, "reason": reason}


async def report(pool, meeting_id: str, name: str, state: str, reason: str | None = None) -> None:
    """Record a step's state and publish it to the tenant in scope. A database fault is logged, not raised: the work
    it reports on matters more than the report. Call it on the event loop."""
    if name not in STEPS or state not in (WORKING, READY, FAILED, SKIPPED) or (reason is not None and reason not in REASONS):
        raise ValueError(f"not a pipeline state: {name} {state} {reason}")
    try:
        if pool is None:            # a store without a database (a unit test's fake): publish only
            raise LookupError("no pool")
        async with pool.connection() as conn:
            await conn.execute(
                """INSERT INTO meeting_steps (meeting_id, step, state, reason, updated_at)
                   VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                   ON CONFLICT (meeting_id, step) DO UPDATE SET
                       state = EXCLUDED.state, reason = EXCLUDED.reason, updated_at = EXCLUDED.updated_at""",
                (meeting_id, name, state, reason),
            )
            await conn.commit()
    except LookupError:
        pass
    except Exception:  # noqa: BLE001 - a meeting deleted meanwhile, a database that went away
        log.warning("pipeline: could not record meeting=%s step=%s state=%s", meeting_id, name, state, exc_info=True)
    events.publish(EVENT, step(meeting_id, name, state, reason))
    log.info("pipeline: meeting=%s step=%s state=%s reason=%s", meeting_id, name, state, reason)


# ── runs in progress ──────────────────────────────────────────────────────────


class Run:
    """A summary or tasks run in progress for one meeting: its job id (what a regenerate request is answered with)
    and its task, when it has one."""

    def __init__(self, job_id: str | None = None):
        self.job_id = job_id or str(uuid.uuid4())
        self.task: asyncio.Task | None = None


_runs: dict[tuple[str | None, str, str], Run] = {}


def running(meeting_id: str, name: str = "summary") -> Run | None:
    """The run in progress for this meeting's step, in the tenant in scope, or None."""
    run = _runs.get((tenancy.current_tenant(), meeting_id, name))
    if run is not None and run.task is not None and run.task.done():
        return None
    return run


def start_run(meeting_id: str, name: str, task: asyncio.Task | None = None, *, job_id: str | None = None) -> Run:
    """Register a run for this meeting's step; it is let go when its task ends (or by end_run). Registering takes
    no await, so a check with running() followed by start_run() cannot be raced by another request."""
    run = Run(job_id)
    _runs[(tenancy.current_tenant(), meeting_id, name)] = run
    if task is not None:
        bind_task(meeting_id, name, run, task)
    return run


def bind_task(meeting_id: str, name: str, run: Run, task: asyncio.Task) -> None:
    """The task doing a registered run's work: the run ends with it."""
    key = (tenancy.current_tenant(), meeting_id, name)
    run.task = task
    task.add_done_callback(lambda _t: _runs.pop(key, None) if _runs.get(key) is run else None)


def end_run(meeting_id: str, name: str, run: Run) -> None:
    key = (tenancy.current_tenant(), meeting_id, name)
    if _runs.get(key) is run:
        _runs.pop(key, None)


def announce(entry: dict, meeting_id: str) -> None:
    """Publish a step as it stands, without recording anything (a request that attached to a run in progress)."""
    events.publish(EVENT, step(meeting_id, entry["step"], entry["state"], entry["reason"]))


def _live(meeting_id: str) -> bool:
    return any(running(meeting_id, name) is not None for name in ("summary", "tasks"))


# ── reading the state back ────────────────────────────────────────────────────


async def read(pool, meeting_id: str) -> list[dict] | None:
    """Every step of a meeting as {step, state, reason}, in STEPS order, or None when the meeting is not the
    tenant's."""
    async with pool.connection() as conn:
        conn.row_factory = dict_row
        meeting = await (await conn.execute("SELECT status FROM meetings WHERE id = %s", (meeting_id,))).fetchone()
        if meeting is None:
            return None
        summary = await (await conn.execute("SELECT status, edited FROM summaries WHERE meeting_id = %s",
                                            (meeting_id,))).fetchone()
        rows = {r["step"]: r for r in await (await conn.execute(
            "SELECT step, state, reason FROM meeting_steps WHERE meeting_id = %s", (meeting_id,))).fetchall()}
        spoken = (await (await conn.execute(
            "SELECT count(*) AS n FROM transcript_segments WHERE meeting_id = %s AND btrim(text) <> ''",
            (meeting_id,))).fetchone())["n"]
        tasks = (await (await conn.execute("SELECT count(*) AS n FROM tasks WHERE meeting_id = %s",
                                           (meeting_id,))).fetchone())["n"]
    status = meeting["status"]
    live = _live(meeting_id)

    def recorded(name):
        row = rows.get(name)
        return (row["state"], row["reason"]) if row else None

    # transcript
    if status == "scheduled":
        transcript = (WORKING, NOT_STARTED)
    elif status == "recording":
        transcript = (WORKING, RECORDING)
    elif status == "interrupted":
        transcript = (FAILED, INTERRUPTED)
    elif recorded("transcript"):
        transcript = recorded("transcript")
    elif status == "processing":
        transcript = (WORKING, None)
    else:
        transcript = (READY, None) if spoken else (SKIPPED, NO_SPEECH)

    # summary
    row = recorded("summary")
    if summary is None or summary["status"] == "none":
        if status in ("scheduled", "recording", "processing"):
            summ = (WORKING, QUEUED)
        elif status == "interrupted":
            summ = (SKIPPED, INTERRUPTED)
        else:
            summ = (SKIPPED, NOT_GENERATED)
    else:
        summ = summary_state(summary["status"])
        if summary["edited"] and summ[0] == READY:
            summ = (READY, EDITED_BY_HAND)
        elif row and row[0] == summ[0]:
            summ = row                                  # the reason the writer gave
        elif row and summ[0] == READY and row == (SKIPPED, NO_SPEECH):
            summ = row                                  # a "nothing was said" summary is stored as ready
        if summ[0] == WORKING and status == "done" and not live:
            summ = (FAILED, INTERRUPTED)

    # tasks
    row = recorded("tasks")
    if row is not None:
        tsk = (FAILED, INTERRUPTED) if row[0] == WORKING and not live and status == "done" else row
    elif tasks:
        tsk = (READY, None)
    elif summ[0] == WORKING:
        tsk = (WORKING, QUEUED)
    elif summ[0] == FAILED:
        tsk = (SKIPPED, SUMMARY_FAILED)
    elif summ[0] == SKIPPED:
        tsk = (SKIPPED, summ[1])
    else:
        tsk = (SKIPPED, NOT_GENERATED)

    return [{"step": n, "state": s, "reason": r} for n, (s, r) in zip(STEPS, (transcript, summ, tsk), strict=True)]
