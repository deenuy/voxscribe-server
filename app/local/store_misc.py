"""Store module for miscellaneous configuration and metrics.

Tables owned by this module:
- settings (app settings, key-value pairs)
- calendar_events (imported calendar events)
- usage (token tracking and feature metrics)
- auth_events (Google sign-in events)
- connector_accounts (Google account metadata, no tokens)
"""

import json

from .timefmt import iso_z
from datetime import date, datetime

from psycopg.rows import dict_row

from .models import AuthEvent, CalendarEvent, Settings


async def get_settings(conn) -> Settings:
    """Get all settings, merge with defaults, return Settings object."""
    cursor = await conn.execute("SELECT key, value FROM settings")
    rows = await cursor.fetchall()
    settings_dict = {}
    for key, value in rows:
        try:
            settings_dict[key] = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            settings_dict[key] = value

    # Create Settings with loaded values, defaults for anything missing
    return Settings(**{k: v for k, v in settings_dict.items() if k in Settings.model_fields})


async def put_settings(conn, settings: Settings) -> None:
    """Replace all settings with new values."""
    # Clear existing settings
    await conn.execute("DELETE FROM settings")

    # Insert new values
    for key, value in settings.model_dump().items():
        await conn.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s)",
            (key, json.dumps(value)),
        )

    await conn.commit()


async def record_usage(conn, feature: str, tokens: int, day: date | None = None) -> None:
    """Record token usage for a feature on a given day.

    This is the shared helper other agents use to record token counts.
    Transcription is counted in words converted at 150 words per 1000 tokens.
    Generative and embedding calls use their reported token counts.

    Args:
        conn: Database connection
        feature: Feature name (transcription, enrichment, embedding, summary, chat, etc.)
        tokens: Token count or word count (depending on feature)
        day: Date to record (default: today)
    """
    if day is None:
        day = date.today()

    # Upsert into usage table
    await conn.execute(
        """INSERT INTO usage (id, feature, day, tokens, created_at, updated_at)
        VALUES (gen_random_uuid()::text, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (feature, day) DO UPDATE
        SET tokens = usage.tokens + EXCLUDED.tokens,
            updated_at = CURRENT_TIMESTAMP""",
        (feature, day, tokens),
    )
    await conn.commit()


async def get_calendar_events(conn, from_dt: str | None = None, to_dt: str | None = None) -> list[CalendarEvent]:
    """Get calendar events in date range.

    Args:
        conn: Database connection
        from_dt: ISO 8601 start time (inclusive)
        to_dt: ISO 8601 end time (exclusive)

    Returns:
        List of CalendarEvent objects
    """
    query = "SELECT id, title, \"start\", \"end\", attendees FROM calendar_events"
    params = []

    if from_dt or to_dt:
        conditions = []
        if from_dt:
            conditions.append("\"start\" >= %s")
            params.append(from_dt)
        if to_dt:
            conditions.append("\"start\" < %s")
            params.append(to_dt)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY \"start\" ASC"

    conn.row_factory = dict_row
    cursor = await conn.execute(query, params)
    rows = await cursor.fetchall()

    events = []
    for row in rows:
        events.append(
            CalendarEvent(
                id=row["id"],
                title=row["title"],
                start=iso_z(row["start"]),
                end=iso_z(row["end"]),
                attendees=row.get("attendees") or [],
            )
        )
    return events


async def upsert_calendar_event(
    conn,
    event_id: str,
    title: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
) -> None:
    """Insert or update a calendar event."""
    attendees = attendees or []
    await conn.execute(
        """INSERT INTO calendar_events (id, title, "start", "end", attendees, created_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (id) DO UPDATE
        SET title = EXCLUDED.title,
            "start" = EXCLUDED."start",
            "end" = EXCLUDED."end",
            attendees = EXCLUDED.attendees""",
        (event_id, title, start, end, attendees),
    )
    await conn.commit()


async def get_last_import_time(conn) -> str | None:
    """Get the timestamp of the last calendar import.

    Returns:
        ISO 8601 timestamp or None if no import has occurred
    """
    conn.row_factory = dict_row
    cursor = await conn.execute(
        """SELECT MAX(created_at) as last_import FROM calendar_events"""
    )
    row = await cursor.fetchone()
    if row and row["last_import"]:
        dt = row["last_import"]
        return dt.isoformat() + "Z" if isinstance(dt, datetime) else dt
    return None


async def create_auth_event(conn, kind: str, detail: dict | None = None) -> AuthEvent:
    """Record an auth event (google_sign_in, google_sign_out, etc.)."""
    event_id = None
    import uuid
    event_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    detail = detail or {}

    await conn.execute(
        "INSERT INTO auth_events (id, at, kind, detail, created_at) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)",
        (event_id, now, kind, json.dumps(detail)),
    )
    await conn.commit()

    return AuthEvent(id=event_id, at=now, kind=kind, detail=detail)


async def list_auth_events(conn, limit: int = 50) -> list[AuthEvent]:
    """List recent auth events, newest first."""
    conn.row_factory = dict_row
    cursor = await conn.execute(
        "SELECT id, at, kind, detail FROM auth_events ORDER BY at DESC LIMIT %s",
        (limit,),
    )
    rows = await cursor.fetchall()

    events = []
    for row in rows:
        detail = row.get("detail") or {}
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except (json.JSONDecodeError, TypeError):
                detail = {}
        events.append(
            AuthEvent(
                id=row["id"],
                at=iso_z(row["at"]),
                kind=row["kind"],
                detail=detail,
            )
        )
    return events


async def get_connector_account(conn, connector: str) -> dict | None:
    """Get connector account metadata (not tokens)."""
    conn.row_factory = dict_row
    cursor = await conn.execute(
        """SELECT connector, sub, email, name, picture, scopes, token_expires_at, last_sync_at
        FROM connector_accounts
        WHERE connector = %s
        LIMIT 1""",
        (connector,),
    )
    return await cursor.fetchone()


async def put_connector_account(
    conn,
    connector: str,
    sub: str,
    email: str,
    name: str,
    picture: str | None,
    scopes: list[str],
    token_expires_at: str | None = None,
) -> None:
    """Store connector account metadata (not the access token)."""
    await conn.execute(
        """INSERT INTO connector_accounts (connector, sub, email, name, picture, scopes, token_expires_at, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (connector, sub) DO UPDATE
        SET email = EXCLUDED.email,
            name = EXCLUDED.name,
            picture = EXCLUDED.picture,
            scopes = EXCLUDED.scopes,
            token_expires_at = EXCLUDED.token_expires_at,
            updated_at = CURRENT_TIMESTAMP""",
        (connector, sub, email, name, picture, scopes, token_expires_at),
    )
    await conn.commit()


async def delete_connector_account(conn, connector: str) -> None:
    """Delete connector account metadata."""
    await conn.execute(
        "DELETE FROM connector_accounts WHERE connector = %s",
        (connector,),
    )
    await conn.commit()
