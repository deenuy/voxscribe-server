"""Finishing a stopped meeting: enrich notes, write the summary, then publish.

One function, used by every way a meeting can stop (the Stop request, the silence auto-stop on the live
socket) and by recovery at startup, so no path can leave a meeting in 'processing' forever. It never raises:
whatever fails, the meeting is still published, because a transcript that exists must become editable and
visible. Tasks are kept in a set while they run, since the event loop only holds weak references to them.
"""

import asyncio
import logging
from contextlib import nullcontext

from . import meeting_bus, tenancy
from .store_meetings import MeetingStore

log = logging.getLogger(__name__)

_running: set[asyncio.Task] = set()


async def finalize_meeting(app, meeting_id: str) -> None:
    store = MeetingStore(app.state.store.pool)
    try:
        # Immediately mark summary as pending so the client sees a defined status.
        await store.set_summary_pending(meeting_id)
        providers = getattr(app.state, "providers", {})
        llm = providers.get("llm")
        if llm:
            from .enrichment import EnrichmentService
            from .live_support import build_retriever
            from .summary_service import SummaryService

            await EnrichmentService(store, llm, build_retriever(app)).finalize_meeting(meeting_id)
            await SummaryService(store, llm).generate_summary(meeting_id)
        else:
            await store.set_summary_status(meeting_id, "unavailable")
    except Exception:
        log.error("Finalization failed for meeting %s", meeting_id, exc_info=True)
    try:
        await store.mark_done(meeting_id)
        await meeting_bus.publish(meeting_id, "status", {"state": "done"})
    except Exception:
        log.error("Could not publish meeting %s", meeting_id, exc_info=True)


def schedule_finalize(app, meeting_id: str) -> asyncio.Task:
    task = asyncio.create_task(finalize_meeting(app, meeting_id))
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


async def recover_unfinished(app) -> int:
    """At startup, recover meetings a crash left in mid-flight states.

    'recording' sessions whose origin='wake' are closed as 'done' — they are voice-assistant
    turns the user never manually started, so there is nothing to resume. All other 'recording'
    sessions are marked 'interrupted' so the user can choose to resume them.
    'processing' sessions are scheduled for finalization (enrichment + summary) as normal.

    With tenant isolation on, every tenant's schema is checked and each meeting is handled inside its own
    tenant's scope. With it off there is one shared scope.

    Returns the total number of sessions recovered.
    """
    pool = app.state.store.pool
    store = MeetingStore(pool)
    enforced = getattr(pool, "enforced", False)
    total_interrupted = 0
    total_wake_closed = 0
    total_processing = 0
    for scope in (await pool.provisioner.known_tenants()) if enforced else [None]:
        with tenancy.tenant_scope(scope) if scope else nullcontext():
            async with pool.connection() as conn:
                recording_rows = await (
                    await conn.execute(
                        "SELECT id, origin FROM meetings WHERE status = 'recording'"
                    )
                ).fetchall()
                processing_rows = await (
                    await conn.execute("SELECT id FROM meetings WHERE status = 'processing'")
                ).fetchall()
            for (meeting_id, origin) in recording_rows:
                try:
                    if origin == "wake":
                        # Wake sessions are voice-assistant turns; the user never started them
                        # manually, so closing as done (not interrupted) prevents a spurious
                        # "resume recording?" dialog on next app launch.
                        async with pool.connection() as conn:
                            await conn.execute(
                                "UPDATE meetings SET status='done', ended_at=NOW() WHERE id=%s AND status='recording'",
                                (meeting_id,),
                            )
                            await conn.commit()
                        total_wake_closed += 1
                    else:
                        await store.mark_interrupted(meeting_id)
                        total_interrupted += 1
                except Exception:
                    log.error("Could not recover recording meeting %s", meeting_id, exc_info=True)
            for (meeting_id,) in processing_rows:
                schedule_finalize(app, meeting_id)
            total_processing += len(processing_rows)
    if total_wake_closed:
        log.info("Closed %d wake session(s) as done (server restart)", total_wake_closed)
    if total_interrupted:
        log.info("Marked %d recording session(s) as interrupted (server restart)", total_interrupted)
    if total_processing:
        log.info("Recovering %d meeting(s) left processing by a restart", total_processing)
    return total_interrupted + total_wake_closed + total_processing
