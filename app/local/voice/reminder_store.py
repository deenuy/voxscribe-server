"""Reminder and timer rows (migrations 019 and 026): the engine's own store, the authority for
anything set by voice (TOOLS-CONNECTORS.md section 2.7).

The store records and reads rows and does nothing else. What a person said is parsed by
voice/spoken_time.py, confirmed by voice/reminder_dialog.py and written through the tool runtime
(app/local/tools/runtime.py), so a row exists only after a receipt-backed call. The due-item
scheduler (voice/due_scheduler.py) delivers rows; the routes (routers/reminders.py) list and
cancel them.

Rows live in the tenant's schema: every connection is borrowed from the pool the store was
given, which selects the schema of the tenant in scope. A row's `text` is what the person asked
to be reminded of: it is the tenant's data, returned to that tenant, and never logged. Logs here
carry ids, kinds, counts and times only.

Lifecycle (`status`):

    pending ──claim_due()──> firing ──mark_delivered()──> fired | missed
       │                       └──release_stale_claims() at start-up──> pending
       ├──cancel()──> cancelled
       └──expire_missed()──> missed (came due while the engine was off, outside the window)

claim_due() is one UPDATE ... WHERE status = 'pending' RETURNING, so two deliverers racing for
the same row never both get it. No time is read from the system clock here: every method that
needs one takes it.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("app.local.voice.reminders")

KIND_REMINDER = "reminder"
KIND_TIMER = "timer"
KINDS = (KIND_REMINDER, KIND_TIMER)

PENDING = "pending"
FIRING = "firing"
FIRED = "fired"
MISSED = "missed"
CANCELLED = "cancelled"

# What GET /v1/reminders calls each status.
API_STATE = {PENDING: "scheduled", FIRING: "scheduled", FIRED: "fired", MISSED: "missed",
             CANCELLED: "cancelled"}

LIST_LIMIT = 500

_COLUMNS = ("id", "kind", "text", "label", "fire_at", "time_zone", "status", "created_at",
            "fired_at", "spoken", "speak_owed", "missed")
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM reminders"
_RETURNING = f" RETURNING {', '.join(_COLUMNS)}"


def zone(name: str | None) -> ZoneInfo:
    """The zone a row was set in; UTC for a row written before migration 026 or a bad name."""
    try:
        return ZoneInfo(name) if name else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def device_zone(settings) -> str:
    """The device's time zone, as an IANA name: settings.vox_timezone (VOX_TIMEZONE) when set,
    else the operating system's (the TZ variable, or the zone /etc/localtime points at), else
    UTC. Nothing stores a zone per tenant today; each row keeps the zone it was set in."""
    import os

    for name in (getattr(settings, "vox_timezone", "") or "", os.environ.get("TZ", "")):
        name = name.strip().lstrip(":")
        if name and zone(name).key == name:
            return name
    try:
        target = os.path.realpath("/etc/localtime")
    except OSError:
        target = ""
    marker = "zoneinfo/"
    if marker in target:
        name = target.split(marker, 1)[1]
        if zone(name).key == name:
            return name
    return "UTC"


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True)
class Item:
    """One reminder or timer. Datetimes are aware, in UTC."""

    id: str
    kind: str
    text: str
    label: str | None
    fire_at: datetime
    time_zone: str | None
    status: str
    created_at: datetime
    fired_at: datetime | None = None
    spoken: bool = False
    speak_owed: bool = False
    missed: bool = False

    @property
    def tz(self) -> ZoneInfo:
        return zone(self.time_zone)

    @property
    def due_at(self) -> str:
        """ISO 8601 with the offset of the zone it was set in."""
        return self.fire_at.astimezone(self.tz).isoformat()

    @property
    def state(self) -> str:
        return API_STATE.get(self.status, "scheduled")

    @property
    def subject(self) -> str | None:
        """What the person will read in a notification: a reminder's words, a timer's none."""
        return (self.text or None) if self.kind == KIND_REMINDER else None


def _item(row) -> Item:
    (rid, kind, text, label, fire_at, time_zone, status, created_at, fired_at, spoken, owed,
     missed) = row
    return Item(id=rid, kind=kind or KIND_REMINDER, text=text or "", label=label,
                fire_at=_utc(fire_at), time_zone=time_zone, status=status,
                created_at=_utc(created_at), fired_at=_utc(fired_at), spoken=bool(spoken),
                speak_owed=bool(owed), missed=bool(missed))


class ReminderStore:
    """`pool` returns the connection pool to borrow from, called at each use (the voice scope's
    pool, or the request's), so the tenant in scope at that moment decides the schema."""

    def __init__(self, pool: Callable[[], object]) -> None:
        self._pool = pool

    @classmethod
    def for_voice(cls, app) -> ReminderStore:
        from app.local.voice.adapters.scope import scoped_pool  # lazy: this module stays light

        return cls(lambda: scoped_pool(app))

    @classmethod
    def for_pool(cls, pool) -> ReminderStore:
        return cls(lambda: pool)

    async def _rows(self, sql: str, params: tuple = ()) -> list[Item]:
        async with self._pool().connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
            await conn.commit()
        return [_item(r) for r in rows]

    # ---------------------------------------------------------------- writes

    async def add(self, *, kind: str, text: str, fire_at: datetime, time_zone: str,
                  created_at: datetime, label: str | None = None,
                  session_id: str | None = None, turn_id: str | None = None) -> Item:
        """Insert one pending row and return it. Raises when the row is not written."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        rid = str(uuid.uuid4())
        (item,) = await self._rows(
            "INSERT INTO reminders (id, kind, text, label, fire_at, time_zone, status,"
            " created_at, created_session_id, created_turn_id)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" + _RETURNING,
            (rid, kind, text, label, _utc(fire_at), time_zone, PENDING, _utc(created_at),
             session_id, turn_id))
        log.info("reminders: added id=%s kind=%s chars=%d fire_at=%s", rid, kind, len(text),
                 item.fire_at.isoformat())
        return item

    async def set_receipt(self, item_id: str, receipt_id: str) -> None:
        async with self._pool().connection() as conn:
            await conn.execute("UPDATE reminders SET receipt_id=%s WHERE id=%s",
                               (receipt_id, item_id))
            await conn.commit()

    async def cancel(self, item_id: str, *, at: datetime) -> Item | None:
        """Cancel a row that is still pending; None when there is none to cancel."""
        rows = await self._rows(
            "UPDATE reminders SET status=%s, cancelled_at=%s WHERE id=%s AND status=%s"
            + _RETURNING, (CANCELLED, _utc(at), item_id, PENDING))
        if rows:
            log.info("reminders: cancelled id=%s kind=%s", item_id, rows[0].kind)
        return rows[0] if rows else None

    async def claim_due(self, now: datetime) -> list[Item]:
        """Take every pending row due at `now` for delivery, oldest first. Atomic: a row is
        returned to one caller only."""
        rows = await self._rows(
            "UPDATE reminders SET status=%s, claimed_at=%s"
            " WHERE id IN (SELECT id FROM reminders WHERE status=%s AND fire_at <= %s"
            " ORDER BY fire_at FOR UPDATE SKIP LOCKED)" + _RETURNING,
            (FIRING, _utc(now), PENDING, _utc(now)))
        return sorted(rows, key=lambda i: i.fire_at)

    async def mark_delivered(self, item_id: str, *, at: datetime, spoken: bool, owed: bool,
                             missed: bool) -> bool:
        """The delivery attempt is over: fired (or missed), with what happened. Only a row this
        deliverer claimed changes."""
        rows = await self._rows(
            "UPDATE reminders SET status=%s, fired_at=%s, spoken=%s, speak_owed=%s, missed=%s"
            " WHERE id=%s AND status=%s" + _RETURNING,
            (MISSED if missed else FIRED, _utc(at), spoken, owed, missed, item_id, FIRING))
        return bool(rows)

    async def release_stale_claims(self) -> int:
        """At start-up: a row left `firing` by an engine that stopped mid-delivery is pending
        again (it is then delivered as missed, once more)."""
        rows = await self._rows(
            "UPDATE reminders SET status=%s, claimed_at=NULL WHERE status=%s" + _RETURNING,
            (PENDING, FIRING))
        return len(rows)

    async def expire_missed(self, before: datetime) -> int:
        """Pending rows due before `before` (outside the missed window): missed, undelivered."""
        rows = await self._rows(
            "UPDATE reminders SET status=%s, missed=TRUE WHERE status=%s AND fire_at < %s"
            + _RETURNING, (MISSED, PENDING, _utc(before)))
        return len(rows)

    async def claim_owed(self, since: datetime) -> list[Item]:
        """Rows delivered as a notification only since `since`, whose line is still to be
        spoken: taken once, so a second session does not speak them again."""
        rows = await self._rows(
            "UPDATE reminders SET speak_owed=FALSE WHERE speak_owed AND fired_at >= %s"
            + _RETURNING, (_utc(since),))
        return sorted(rows, key=lambda i: i.fire_at)

    async def mark_spoken(self, item_id: str) -> None:
        async with self._pool().connection() as conn:
            await conn.execute("UPDATE reminders SET spoken=TRUE WHERE id=%s", (item_id,))
            await conn.commit()

    # ---------------------------------------------------------------- reads

    async def get(self, item_id: str) -> Item | None:
        rows = await self._rows(_SELECT + " WHERE id=%s", (item_id,))
        return rows[0] if rows else None

    async def list_scheduled(self, kind: str | None = None) -> list[Item]:
        """Rows still to come (pending), soonest first; one kind or both."""
        if kind is None:
            return await self._rows(_SELECT + " WHERE status=%s ORDER BY fire_at", (PENDING,))
        return await self._rows(_SELECT + " WHERE status=%s AND kind=%s ORDER BY fire_at",
                                (PENDING, kind))

    async def list_all(self, limit: int = LIST_LIMIT) -> list[Item]:
        """Every row, soonest due first, at most `limit`."""
        return await self._rows(_SELECT + " ORDER BY fire_at LIMIT %s", (limit,))

    async def next_due_at(self) -> datetime | None:
        async with self._pool().connection() as conn:
            cur = await conn.execute(
                "SELECT min(fire_at) FROM reminders WHERE status=%s", (PENDING,))
            row = await cur.fetchone()
            await conn.commit()
        return _utc(row[0]) if row and row[0] is not None else None
