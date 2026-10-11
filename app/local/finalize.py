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
        providers = getattr(app.state, "providers", {})
        llm = providers.get("llm")
        if llm:
            from .enrichment import EnrichmentService
            from .live_support import build_retriever
            from .summary_service import SummaryService

            await EnrichmentService(store, llm, build_retriever(app)).finalize_meeting(meeting_id)
            await SummaryService(store, llm).generate_summary(meeting_id)
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
    """At startup, finish meetings a crash left in 'processing'. Returns how many were scheduled.

    With tenant isolation on, every tenant's schema is checked and each meeting is finished inside its own tenant's
    scope (the tasks it starts inherit that scope). With it off there is one shared scope."""
    pool = app.state.store.pool
    enforced = getattr(pool, "enforced", False)
    total = 0
    for scope in (await pool.provisioner.known_tenants()) if enforced else [None]:
        with tenancy.tenant_scope(scope) if scope else nullcontext():
            async with pool.connection() as conn:
                rows = await (await conn.execute("SELECT id FROM meetings WHERE status = 'processing'")).fetchall()
            for (meeting_id,) in rows:
                schedule_finalize(app, meeting_id)
            total += len(rows)
    if total:
        log.info("Recovering %d meeting(s) left processing by a restart", total)
    return total
