"""One way to write a moment in an API answer: UTC, ISO 8601, ending in Z.

PostgreSQL hands back timestamps in the session's time zone (for example -04:00). Writing `isoformat() + "Z"` on one
of those produced `2026-10-10T23:42:10-04:00Z`, which no client can parse. Convert to UTC first, then write it."""

from datetime import UTC, datetime


def iso_z(value):
    """A datetime as UTC ISO text with a Z. A naive datetime is taken to be UTC; anything else passes through."""
    if not isinstance(value, datetime):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
