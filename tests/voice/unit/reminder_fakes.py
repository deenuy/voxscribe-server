"""Fakes for the reminder and timer tests: a clock the test moves, and an in-memory store with the
interface of voice/reminder_store.ReminderStore (its SQL is tested against Postgres in
tests/test_reminders_db.py)."""
from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

from app.local.voice import reminder_store as rs
from app.local.voice.adapters.store_postgres import PostgresSessionStore

# Thursday 19 November 2026, 14:00 in Toronto.
START = datetime(2026, 11, 19, 19, 0, tzinfo=UTC)
ZONE = "America/Toronto"


class FakeClock:
    """Aware UTC now, moved only by the test."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw) -> datetime:
        self.now += timedelta(**kw)
        return self.now


class FakeReminderStore:
    def __init__(self, *, fail_add: bool = False) -> None:
        self.rows: dict[str, rs.Item] = {}
        self.receipts: dict[str, str] = {}
        self.fail_add = fail_add
        self.claims = 0

    def _set(self, item: rs.Item, **changes) -> rs.Item:
        new = dataclasses.replace(item, **changes)
        self.rows[item.id] = new
        return new

    async def add(self, *, kind, text, fire_at, time_zone, created_at, label=None,
                  session_id=None, turn_id=None) -> rs.Item:
        if self.fail_add:
            raise RuntimeError("reminder store down")
        item = rs.Item(id=str(uuid.uuid4()), kind=kind, text=text, label=label,
                       fire_at=fire_at.astimezone(UTC), time_zone=time_zone, status=rs.PENDING,
                       created_at=created_at)
        self.rows[item.id] = item
        return item

    async def set_receipt(self, item_id, receipt_id) -> None:
        self.receipts[item_id] = receipt_id

    async def cancel(self, item_id, *, at) -> rs.Item | None:
        item = self.rows.get(item_id)
        if item is None or item.status != rs.PENDING:
            return None
        return self._set(item, status=rs.CANCELLED)

    async def claim_due(self, now) -> list[rs.Item]:
        self.claims += 1
        due = sorted((i for i in self.rows.values()
                      if i.status == rs.PENDING and i.fire_at <= now), key=lambda i: i.fire_at)
        return [self._set(i, status=rs.FIRING) for i in due]

    async def mark_delivered(self, item_id, *, at, spoken, owed, missed) -> bool:
        item = self.rows.get(item_id)
        if item is None or item.status != rs.FIRING:
            return False
        self._set(item, status=rs.MISSED if missed else rs.FIRED, fired_at=at, spoken=spoken,
                  speak_owed=owed, missed=missed)
        return True

    async def release_stale_claims(self) -> int:
        stale = [i for i in self.rows.values() if i.status == rs.FIRING]
        for i in stale:
            self._set(i, status=rs.PENDING)
        return len(stale)

    async def expire_missed(self, before) -> int:
        old = [i for i in self.rows.values() if i.status == rs.PENDING and i.fire_at < before]
        for i in old:
            self._set(i, status=rs.MISSED, missed=True)
        return len(old)

    async def claim_owed(self, since) -> list[rs.Item]:
        owed = sorted((i for i in self.rows.values()
                       if i.speak_owed and i.fired_at is not None and i.fired_at >= since),
                      key=lambda i: i.fire_at)
        return [self._set(i, speak_owed=False) for i in owed]

    async def mark_spoken(self, item_id) -> None:
        self._set(self.rows[item_id], spoken=True)

    async def get(self, item_id) -> rs.Item | None:
        return self.rows.get(item_id)

    async def list_scheduled(self, kind=None) -> list[rs.Item]:
        return sorted((i for i in self.rows.values() if i.status == rs.PENDING
                       and (kind is None or i.kind == kind)), key=lambda i: i.fire_at)

    async def list_all(self, limit=rs.LIST_LIMIT) -> list[rs.Item]:
        return sorted(self.rows.values(), key=lambda i: i.fire_at)[:limit]

    async def next_due_at(self):
        pending = [i.fire_at for i in self.rows.values() if i.status == rs.PENDING]
        return min(pending) if pending else None

    def of(self, kind: str) -> list[rs.Item]:
        return [i for i in self.rows.values() if i.kind == kind]


class SessionRows(PostgresSessionStore):
    """The real session store's turn, gate and receipt rows (over the in-memory pool), with no
    meeting rows, so a whole session can run without a database."""

    async def create_session(self, *_a, **_kw):
        return "s1"

    async def add_segment(self, *_a):
        pass

    async def close_session(self, *_a):
        pass

    discard_session = close_session
