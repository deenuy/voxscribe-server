"""Calendar endpoints: import from ICS or Google, list events."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Query, Request

from app.local import connector_tokens, ics, store_misc
from app.local.errors import ConflictError, InvalidRequestError, NetworkDisabledError
from app.local.models import CalendarImportRequest, CalendarImportResponse, CalendarStatusResponse

router = APIRouter(prefix="/v1/calendar", tags=["calendar"])


@router.get("/events")
async def get_calendar_events(
    request: Request, from_: str | None = Query(None, alias="from"), to: str | None = None
) -> dict:
    """Get calendar events in a date range.

    Query params:
        from: ISO 8601 start time (inclusive)
        to: ISO 8601 end time (exclusive)
    """
    store = request.app.state.store

    async with store.pool.connection() as conn:
        events = await store_misc.get_calendar_events(conn, from_, to)

    return {
        "items": [e.model_dump() for e in events],
    }


@router.post("/import")
async def import_calendar(request: Request, body: CalendarImportRequest) -> dict:
    """Import calendar events from ICS file or Google Calendar.

    Body:
        kind: "ics" or "google"
        path: file path (for ics)
        calendarId: calendar ID (for google, optional)

    Returns:
        {imported: int, lastImportedAt: ISO 8601}
    """
    store = request.app.state.store

    if body.kind == "ics":
        if not body.path:
            raise InvalidRequestError("path required for ics import")

        # Parse ICS file
        try:
            events = ics.parse_ics_file(body.path)
        except FileNotFoundError as e:
            raise InvalidRequestError(f"file not found: {body.path}") from e
        except Exception as e:
            raise InvalidRequestError(f"failed to parse ICS: {e}") from e

        # Upsert events into database and create scheduled meetings
        async with store.pool.connection() as conn:
            for event in events:
                # Upsert calendar event
                await store_misc.upsert_calendar_event(
                    conn,
                    event["uid"],
                    event["title"],
                    event["start"],
                    event["end"],
                    event.get("attendees"),
                )

                # Upsert scheduled meeting
                meeting_id = str(uuid.uuid4())
                await conn.execute(
                    """INSERT INTO meetings
                    (id, title, calendar_event_id, status, scheduled_start, scheduled_end, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING""",
                    (
                        meeting_id,
                        event["title"],
                        event["uid"],
                        "scheduled",
                        event["start"],
                        event["end"],
                    ),
                )
            await conn.commit()

        last_import_at = await _get_last_import_time(store)
        return CalendarImportResponse(
            imported=len(events),
            lastImportedAt=last_import_at or datetime.utcnow().isoformat() + "Z",
        ).model_dump()

    elif body.kind == "google":
        # Check if Google connector is connected
        settings = request.app.state.settings

        # Check privacy mode
        if settings.privacy_mode != "connectors":
            raise NetworkDisabledError("Google Calendar import requires privacyMode: connectors")

        # Check if token is available
        token_tuple = await connector_tokens.get_google_access_token()
        if not token_tuple:
            raise ConflictError("Google account not connected")

        token, _ = token_tuple

        # Fetch events from Google Calendar API
        import httpx

        calendar_id = body.calendarId or "primary"
        google_api_base = "https://www.googleapis.com/calendar/v3"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{google_api_base}/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"maxResults": 100},
                )

                if response.status_code == 401:
                    raise ConflictError("Google access token expired")
                elif response.status_code != 200:
                    raise Exception(f"Google API error: {response.status_code}")

                data = response.json()
                events_data = data.get("items", [])

                # Upsert events into database
                async with request.app.state.store.pool.connection() as conn:
                    for event_item in events_data:
                        event_id = event_item.get("id")
                        title = event_item.get("summary", "Event")
                        start = event_item.get("start", {}).get("dateTime") or event_item.get("start", {}).get(
                            "date"
                        )
                        end = event_item.get("end", {}).get("dateTime") or event_item.get("end", {}).get("date")
                        attendees = [a.get("email") for a in event_item.get("attendees", [])]

                        if start and end:
                            await store_misc.upsert_calendar_event(
                                conn, event_id, title, start, end, attendees
                            )

                            # Create scheduled meeting if not exists
                            meeting_id = str(uuid.uuid4())
                            await conn.execute(
                                """INSERT INTO meetings
                                (id, title, calendar_event_id, status, scheduled_start, scheduled_end, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                ON CONFLICT (id) DO NOTHING""",
                                (meeting_id, title, event_id, "scheduled", start, end),
                            )
                    await conn.commit()

                last_import_at = await _get_last_import_time(store)
                return CalendarImportResponse(
                    imported=len(events_data),
                    lastImportedAt=last_import_at or datetime.utcnow().isoformat() + "Z",
                ).model_dump()

        except httpx.RequestError as e:
            raise ConflictError(f"Failed to fetch from Google Calendar: {e}") from e
    else:
        raise InvalidRequestError(f"unknown kind: {body.kind}")


@router.get("/status")
async def calendar_status(request: Request) -> dict:
    """Get calendar import status and connected accounts."""
    store = request.app.state.store

    async with store.pool.connection() as conn:
        last_import_at = await store_misc.get_last_import_time(conn)
        # Check for connected Google account
        account = await store_misc.get_connector_account(conn, "google")

    connected_accounts = []
    if account:
        connected_accounts.append(account.get("email", ""))

    return CalendarStatusResponse(
        lastImportedAt=last_import_at,
        connectedAccounts=connected_accounts,
    ).model_dump()


async def _get_last_import_time(store) -> str | None:
    """Get the last import time."""
    async with store.pool.connection() as conn:
        return await store_misc.get_last_import_time(conn)
