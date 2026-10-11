"""Stats endpoints: usage tracking and aggregation."""

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Request

from app.local.models import Stats, WeeklyStat

router = APIRouter(prefix="/v1", tags=["stats"])


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    """Get usage statistics.

    Returns stats for:
    - tokensSaved: total tokens in usage table
    - weekly: list of stats for last 5 ISO weeks (oldest first)
    - meetingsTranscribed: meetings with transcript
    - sourcesAdded: sources with status ready
    - assistsProvided: assist events + chat answers
    - cloudCalls: always 0
    """
    store = request.app.state.store

    async with store.pool.connection() as conn:
        # Get total tokens
        cursor = await conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM usage")
        total_tokens = (await cursor.fetchone())[0]

        # Get weekly stats for last 5 ISO weeks
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()

        weekly_stats = []
        for week_offset in range(4, -1, -1):  # Last 5 weeks, oldest first
            target_week = iso_week - week_offset
            target_year = iso_year

            # Handle year boundary
            if target_week < 1:
                target_year -= 1
                target_week += 52

            # Get Monday of that ISO week
            jan4 = datetime(target_year, 1, 4)
            week_start = jan4 - timedelta(days=jan4.weekday())
            week_start = week_start + timedelta(weeks=target_week - 1)

            # Sum tokens for that week
            cursor = await conn.execute(
                """SELECT COALESCE(SUM(tokens), 0) FROM usage
                WHERE day >= %s AND day < %s""",
                (week_start.date(), week_start.date() + timedelta(days=7)),
            )
            week_tokens = (await cursor.fetchone())[0]
            weekly_stats.append(
                WeeklyStat(
                    weekStart=week_start.date().isoformat(),
                    tokens=week_tokens,
                )
            )

        # Count meetings with transcript
        cursor = await conn.execute("SELECT COUNT(*) FROM meetings WHERE status = %s", ("done",))
        meetings_transcribed = (await cursor.fetchone())[0]

        # Count sources with status ready
        cursor = await conn.execute("SELECT COUNT(*) FROM sources WHERE status = %s", ("ready",))
        sources_added = (await cursor.fetchone())[0]

        # Count assists (for now, use assist feature in usage table)
        cursor = await conn.execute("SELECT COALESCE(COUNT(*), 0) FROM usage WHERE feature = %s", ("assist",))
        assists = (await cursor.fetchone())[0]

        # Count chat answers (future: when chat is implemented)
        assists_provided = assists

    return Stats(
        tokensSaved=total_tokens,
        weekly=weekly_stats,
        meetingsTranscribed=meetings_transcribed,
        sourcesAdded=sources_added,
        assistsProvided=assists_provided,
        cloudCalls=0,
    ).model_dump()
