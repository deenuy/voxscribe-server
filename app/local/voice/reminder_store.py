"""Reminder store: native reminder and timer support backed by the local DB (T16).

The store is intentionally thin: it records and retrieves rows and does nothing else.
Routing, parsing, and read-back all live in the delegate adapter and the host.

parse_reminder(text) is the only entry point that accepts user text. It returns
(description, fire_at) on a match, or None when the text is not a reminder request.
Supported patterns (regex, no external library):

  remind me to X at HH:MM [am/pm]      -- time today, tomorrow if the slot is in the past
  remind me to X in N minutes/hours
  set a reminder for X at HH:MM [am/pm]
  set a timer for N minutes/seconds

All times are local time, converted to UTC.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

log = logging.getLogger("app.local.voice.reminders")

# ------------------------------------------------------------------ constants

REMINDER_SIGNALS = frozenset({
    "remind me",
    "reminder",
    "set a timer",
    "timer for",
    "alarm for",
})

REMINDER_LOOKAHEAD_H = 24   # schedule at most this many hours ahead

# The confirmations are the reminder.create and timer.start records' templates
# (app/local/tools/builtin.py), spoken only with a receipt.
REMINDER_FIRED_PHRASE = "Reminder: {text}."
REMINDER_LIST_INTRO = "You have {n} reminder{s} coming up."
REMINDER_LIST_ITEM = "{time_str}: {text}."
REMINDER_NONE_PHRASE = "You have no reminders coming up."
REMINDER_CANCELLED_PHRASE = "Reminder cancelled."

# Read-back detection: "what reminders do I have?", "my reminders", "show my reminders", etc.
_REMINDER_LIST_RE = re.compile(
    r"\b(?:"
    r"what(?:'s|\s+are|\s+do\s+i\s+have)?\s+(?:(?:my|upcoming)\s+)?reminders?"
    r"|my\s+reminders?"
    r"|show\s+(?:my\s+)?reminders?"
    r"|upcoming\s+reminders?"
    r"|what's\s+on\s+my\s+list"
    r")\b",
    re.IGNORECASE,
)

# ------------------------------------------------------------------ patterns

# "at HH:MM [am/pm]" or "for HH:MM am/pm" (for requires am/pm to avoid ambiguity)
_AT_TIME_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
    re.IGNORECASE,
)
# "for HH:MM am/pm" -- used in "set a reminder for 6:30 am" (no description)
_FOR_TIME_RE = re.compile(
    r"\bfor\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.IGNORECASE,
)

# "in N minutes/hours"
_IN_DELTA_RE = re.compile(
    r"\bin\s+(\d+)\s+(minutes?|hours?|seconds?)\b",
    re.IGNORECASE,
)

# "set a timer for N minutes/seconds"
_TIMER_RE = re.compile(
    r"\bset\s+(?:a\s+)?timer\s+for\s+(\d+)\s+(minutes?|seconds?)\b",
    re.IGNORECASE,
)

# Leading verb phrases to strip when extracting a description
_VERB_STRIP_RE = re.compile(
    r"^\s*(?:remind me (?:to|about)|set (?:a )?reminder (?:for|to|about))\s+",
    re.IGNORECASE,
)

# Trailing time clause to strip from description
_TRAILING_TIME_RE = re.compile(
    r"\s+(?:(?:at|for)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|in\s+\d+\s+(?:minutes?|hours?|seconds?))\s*$",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ helpers

def _local_offset() -> timedelta:
    return datetime.now().astimezone().utcoffset() or timedelta(0)


def _resolve_at_time(hour: int, minute: int) -> datetime:
    """Return a UTC datetime for the next occurrence of hour:minute local time.

    If that slot is in the past today, roll to tomorrow, up to REMINDER_LOOKAHEAD_H.
    """
    now_local = datetime.now().astimezone()
    candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now_local:
        candidate += timedelta(days=1)
    if (candidate - now_local).total_seconds() / 3600 > REMINDER_LOOKAHEAD_H:
        # Too far ahead; return as-is and let the caller decide.
        pass
    return candidate.astimezone(UTC).replace(tzinfo=UTC)


def format_local_time(dt: datetime) -> str:
    """Human-readable local time: '3:00 PM'."""
    local = dt.astimezone()
    return local.strftime("%-I:%M %p").lstrip("0") or local.strftime("%I:%M %p")


# ------------------------------------------------------------------ public parser

def parse_reminder(text: str) -> tuple[str, datetime] | None:
    """Parse a reminder or timer request.

    Returns (description, fire_at_utc) on a match, or None.
    """
    norm = text.strip()

    # "set a timer for N minutes/seconds" -- no description
    m = _TIMER_RE.search(norm)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if "second" in unit:
            delta = timedelta(seconds=n)
            desc = f"{n} second{'s' if n != 1 else ''}"
        else:
            delta = timedelta(minutes=n)
            desc = f"{n} minute{'s' if n != 1 else ''}"
        fire_at = datetime.now(UTC) + delta
        return desc, fire_at

    # "remind me to X at HH:MM [am/pm]", "set a reminder for X at HH:MM [am/pm]",
    # or "set a reminder for HH:MM am/pm" (time-only, no description after for)
    has_reminder_keyword = bool(re.search(r"\b(?:remind me|reminder)\b", norm, re.IGNORECASE))
    time_m = _AT_TIME_RE.search(norm) if has_reminder_keyword else None
    if time_m is None and has_reminder_keyword:
        time_m = _FOR_TIME_RE.search(norm)
    if time_m is not None and has_reminder_keyword:
        hour = int(time_m.group(1))
        minute = int(time_m.group(2)) if time_m.group(2) else 0
        ampm = (time_m.group(3) or "").lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if hour > 23:
            return None
        fire_at = _resolve_at_time(hour, minute)
        desc = _extract_description(norm)
        return desc, fire_at

    # "remind me to X in N minutes/hours"
    in_m = _IN_DELTA_RE.search(norm)
    if in_m and re.search(r"\b(?:remind me|reminder)\b", norm, re.IGNORECASE):
        n, unit = int(in_m.group(1)), in_m.group(2).lower()
        if "hour" in unit:
            delta = timedelta(hours=n)
        elif "minute" in unit:
            delta = timedelta(minutes=n)
        else:
            delta = timedelta(seconds=n)
        fire_at = datetime.now(UTC) + delta
        desc = _extract_description(norm)
        return desc, fire_at

    return None


def _extract_description(text: str) -> str:
    """Strip the leading verb phrase and trailing time clause from the user's request."""
    desc = _VERB_STRIP_RE.sub("", text, count=1)
    desc = _TRAILING_TIME_RE.sub("", desc).strip().rstrip(".!?").strip()
    return desc or text.strip()


def is_reminder_signal(text: str) -> bool:
    """True when the turn text looks like a reminder request."""
    norm = text.lower()
    return any(sig in norm for sig in REMINDER_SIGNALS)


def is_reminder_list_request(text: str) -> bool:
    """True when the user is asking what reminders they have."""
    return bool(_REMINDER_LIST_RE.search(text))


# ------------------------------------------------------------------ dataclasses

class ReminderStatus(Enum):
    PENDING = "pending"
    FIRED = "fired"
    CANCELLED = "cancelled"


@dataclass
class Reminder:
    id: str
    text: str
    fire_at: datetime       # UTC
    status: ReminderStatus
    created_at: datetime    # UTC


# ------------------------------------------------------------------ store

class ReminderStore:
    """In-process reminder registry backed by the local DB.

    All datetimes returned have tzinfo=UTC.
    """

    def __init__(self, app) -> None:
        self._app = app

    @property
    def _pool(self):
        from app.local.voice.adapters.scope import scoped_pool  # lazy: this module stays pure

        return scoped_pool(self._app)

    # ---------------------------------------------------------------- writes

    async def add(self, text: str, fire_at: datetime) -> Reminder:
        """Insert a new PENDING reminder and return it."""
        rid = str(uuid.uuid4())
        fire_utc = fire_at.astimezone(UTC).replace(tzinfo=UTC)
        created = datetime.now(UTC)
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO reminders (id, text, fire_at, status, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (rid, text.strip(), fire_utc, ReminderStatus.PENDING.value, created),
            )
            await conn.commit()
        log.info("reminders: added id=%s text_words=%d fire_at=%s",
                 rid, len(text.split()), fire_utc.isoformat())
        return Reminder(id=rid, text=text.strip(), fire_at=fire_utc,
                        status=ReminderStatus.PENDING, created_at=created)

    async def cancel(self, reminder_id: str) -> bool:
        """Mark a reminder CANCELLED. Returns False when not found or already non-pending."""
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "UPDATE reminders SET status = %s WHERE id = %s AND status = %s",
                (ReminderStatus.CANCELLED.value, reminder_id, ReminderStatus.PENDING.value),
            )
            await conn.commit()
            changed = result.rowcount if hasattr(result, "rowcount") else 0
        if changed:
            log.info("reminders: cancelled id=%s", reminder_id)
        return bool(changed)

    async def mark_fired(self, reminder_id: str) -> None:
        """Mark a reminder FIRED."""
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE reminders SET status = %s WHERE id = %s",
                (ReminderStatus.FIRED.value, reminder_id),
            )
            await conn.commit()
        log.info("reminders: fired id=%s", reminder_id)

    # ---------------------------------------------------------------- reads

    async def list_pending(self) -> list[Reminder]:
        """All PENDING reminders, ordered by fire_at ascending."""
        async with self._pool.connection() as conn:
            rows = await conn.execute(
                "SELECT id, text, fire_at, status, created_at FROM reminders "
                "WHERE status = %s ORDER BY fire_at",
                (ReminderStatus.PENDING.value,),
            )
            return [_row_to_reminder(r) for r in await rows.fetchall()]

    async def list_all(self) -> list[Reminder]:
        """All reminders, ordered by created_at descending."""
        async with self._pool.connection() as conn:
            rows = await conn.execute(
                "SELECT id, text, fire_at, status, created_at FROM reminders "
                "ORDER BY created_at DESC",
            )
            return [_row_to_reminder(r) for r in await rows.fetchall()]

    async def list_due(self) -> list[Reminder]:
        """PENDING reminders whose fire_at is at or before now, ordered by fire_at."""
        now = datetime.now(UTC)
        async with self._pool.connection() as conn:
            rows = await conn.execute(
                "SELECT id, text, fire_at, status, created_at FROM reminders "
                "WHERE status = %s AND fire_at <= %s ORDER BY fire_at",
                (ReminderStatus.PENDING.value, now),
            )
            return [_row_to_reminder(r) for r in await rows.fetchall()]


def is_timer_request(text: str) -> bool:
    """True when parse_reminder() read the text as a timer ("set a timer for N minutes")."""
    return bool(_TIMER_RE.search(text))


def build_reminder_list_sentences(pending: list[Reminder]) -> list[str]:
    """Sentences to speak when the user asks 'what reminders do I have?'."""
    if not pending:
        return [REMINDER_NONE_PHRASE]
    n = len(pending)
    s = "s" if n != 1 else ""
    sentences = [REMINDER_LIST_INTRO.format(n=n, s=s)]
    for r in pending:
        time_str = format_local_time(r.fire_at)
        sentences.append(REMINDER_LIST_ITEM.format(time_str=time_str, text=r.text))
    return sentences


def _row_to_reminder(row) -> Reminder:
    rid, text, fire_at, status, created_at = row
    # psycopg3 returns timezone-aware datetimes; normalise to UTC.
    fire_utc = fire_at.astimezone(UTC).replace(tzinfo=UTC) if fire_at.tzinfo else fire_at.replace(tzinfo=UTC)
    created_utc = created_at.astimezone(UTC).replace(tzinfo=UTC) if created_at.tzinfo else created_at.replace(tzinfo=UTC)
    return Reminder(
        id=rid,
        text=text,
        fire_at=fire_utc,
        status=ReminderStatus(status),
        created_at=created_utc,
    )
