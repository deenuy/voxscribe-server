"""The due-item scheduler (PL46, TOOLS-CONNECTORS.md section 2.7): reminders and timers fire on
time, with or without a conversation open.

How it waits. One task for the engine. It asks each tenant's store for its earliest pending due
time and sleeps until the earliest of them, or until it is told a row changed (notify(): a
reminder set, a timer started, a cancel by voice or by DELETE /v1/reminders/{id}). There is no
polling interval. The clock is injected (SchedulerClock): the system one re-reads the wall
clock at least every MAX_SLEEP_S, so a machine that slept is not late by more than that.

How a row is never fired twice. A due row is taken by ReminderStore.claim_due(), one UPDATE
from pending to firing, so of two deliverers racing for it (two schedulers, a restart that
overlaps) only one gets it. It is marked fired (or missed) after the gate's decision and the
delivery attempt, not before (mark_delivered, from firing only).

Restarts (TC8). At start-up, rows left firing by an engine that stopped mid-delivery are pending
again; rows that came due before the start and within the missed window are delivered once, as
missed ("While I was off, ..."); older ones are marked missed and only listed.

Tenancy. Every tenant's rows are read and delivered inside that tenant's scope. An item is
spoken only on a device bound to its tenant (VoiceHost.speak_initiated); otherwise its tenant's
app is still sent the event and the line is owed to their next session. The scheduler never
creates a tenant's schema: with isolation on it reads only tenants whose schema this process has
already opened (a request or a bind did), and does a tenant's start-up recovery the first time
it sees them. Erase (DELETE /v1/data) holds the scheduler with erasing(): no pass runs and the
tenant's deliveries in flight are cancelled while their schema is dropped.

Delivering one item:

  1. The line for display and speech is put to the gate (no person's words in it). The event's
     `text` is the reminder's words when the gate passes them, else null (the app then shows a
     neutral line naming the time). A timer has no text.
  2. Inside quiet hours, by default (TC5): no sound, the notification only, and the line is
     spoken at the start of the next session. Otherwise the host says it: the chime, then the
     line through the gate (the neutral time line when the gate refuses the words), never over
     a person speaking, after the reply when a session is open.
  3. The event `reminder.due` goes to the tenant: {id, kind, text, dueAt, missed, spoken}.
  4. The row is marked fired or missed, with spoken and owed recorded.

Logs carry ids, kinds, tenants (their keys), outcomes and counts, never a reminder's words.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.local import tenancy
from app.local.voice import due_lines
from app.local.voice import reminder_store as rs
from app.local.voice.reminder_dialog import QUIET_DUE_NOTIFY, in_quiet_hours, quiet_hours_policy

log = logging.getLogger("app.local.voice.due_scheduler")

EVENT_KIND = "reminder.due"
MAX_SLEEP_S = 60.0
DEFAULT_MISSED_WINDOW = timedelta(hours=12)


class SchedulerClock(Protocol):
    def now(self) -> datetime:
        """Aware UTC now."""

    async def wait(self, changed: asyncio.Event, until: datetime | None) -> None:
        """Return when `changed` is set or the clock reaches `until` (never, for None)."""


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    async def wait(self, changed: asyncio.Event, until: datetime | None) -> None:
        timeout = MAX_SLEEP_S
        if until is not None:
            timeout = min(timeout, max(0.0, (until - self.now()).total_seconds()))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), timeout)


class Speaker(Protocol):
    """What says a due line on the device: the voice host."""

    async def speak_initiated(self, u): ...

    def on_session_open(self, listener: Callable[[str | None], None]) -> None: ...


@dataclass(frozen=True)
class Delivery:
    item_id: str
    tenant: str | None
    spoken: bool
    owed: bool
    missed: bool
    reason: str
    payload: dict


def _publish(kind: str, payload: dict) -> None:
    from app.local import events

    events.publish(kind, payload)


class DueScheduler:
    def __init__(self, app, *, clock: SchedulerClock | None = None,
                 speaker: Speaker | None = None,
                 store: Callable[[], object] | None = None,
                 publish: Callable[[str, dict], None] | None = None,
                 missed_window: timedelta = DEFAULT_MISSED_WINDOW) -> None:
        self._app = app
        self.clock = clock or SystemClock()
        self.speaker = speaker
        self._store = store or (lambda: rs.ReminderStore.for_voice(app))
        self._publish = publish or _publish
        self.missed_window = missed_window
        self._changed = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._deliveries: dict[asyncio.Task, str | None] = {}     # task -> tenant
        self._pass = asyncio.Lock()                 # one pass at a time; erase holds it
        self._recovered: set[str | None] = set()    # tenants whose start-up recovery ran
        self.delivered: deque[Delivery] = deque(maxlen=256)   # recent deliveries, newest last
        self.started_at: datetime | None = None
        if speaker is not None:
            speaker.on_session_open(self._session_opened)

    # ── lifecycle ──

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="due-scheduler",
                                             context=contextvars.Context())
        return self._task

    async def stop(self) -> None:
        tasks = [t for t in (self._task, *self._deliveries) if t is not None and not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._task = None

    def notify(self) -> None:
        """A row changed: re-plan the next wake."""
        self._changed.set()

    async def run(self) -> None:
        self.started_at = self.clock.now()
        log.info("due scheduler: started")
        while True:
            self._changed.clear()
            try:
                async with self._pass:
                    await self.recover(self.started_at)
                    await self.fire_due(self.clock.now())
                    until = await self.next_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("due scheduler: a pass failed; trying again at the next change")
                until = None
            await self.clock.wait(self._changed, until)

    @asynccontextmanager
    async def erasing(self, tenant: str | None):
        """Hold the scheduler while `tenant`'s data is erased: no pass runs, and the tenant's
        deliveries in flight are cancelled and waited out, so nothing of theirs is written (or
        their schema made again) after the erase. Their recovery runs again if they come back."""
        async with self._pass:
            mine = [t for t, owner in self._deliveries.items() if owner == tenant]
            for task in mine:
                task.cancel()
            await asyncio.gather(*mine, return_exceptions=True)
            self._recovered.discard(tenant)
            yield

    async def settle(self) -> None:
        """Wait for deliveries in flight (tests and shutdown)."""
        while self._deliveries:
            pending = list(self._deliveries)
            await asyncio.gather(*pending, return_exceptions=True)
            for task in pending:
                self._deliveries.pop(task, None)

    # ── tenants ──

    async def _tenants(self) -> list[str | None]:
        """The tenants to read: the one shared scope with isolation off; with it on, the tenants
        whose schema exists and this process has opened (never one it would have to create)."""
        pool = getattr(getattr(self._app.state, "store", None), "pool", None)
        if pool is None or not getattr(pool, "enforced", False):
            return [None]
        provisioner = pool.provisioner
        ready = getattr(provisioner, "_ready", None)
        known = list(await provisioner.known_tenants())
        return [t for t in known if ready is None or t in ready]

    @staticmethod
    def _scope(tenant: str | None):
        return tenancy.tenant_scope(tenant) if tenant else contextlib.nullcontext()

    # ── one pass ──

    async def recover(self, started_at: datetime) -> None:
        """Start-up, once per tenant: release stale claims, mark rows older than the window
        missed, deliver the rest that came due before the start, once, as missed."""
        for tenant in await self._tenants():
            if tenant in self._recovered:
                continue
            self._recovered.add(tenant)
            with self._scope(tenant):
                store = self._store()
                released = await store.release_stale_claims()
                expired = await store.expire_missed(started_at - self.missed_window)
                due = await store.claim_due(started_at)
            if released or expired or due:
                log.info("due scheduler: start-up tenant=%s released=%d expired=%d missed=%d",
                         tenant, released, expired, len(due))
            for item in due:
                self._spawn(item, tenant, missed=True)

    async def fire_due(self, now: datetime) -> int:
        count = 0
        for tenant in await self._tenants():
            with self._scope(tenant):
                due = await self._store().claim_due(now)
            for item in due:
                self._spawn(item, tenant, missed=False)
            count += len(due)
        return count

    async def next_due(self) -> datetime | None:
        times = []
        for tenant in await self._tenants():
            with self._scope(tenant):
                at = await self._store().next_due_at()
            if at is not None:
                times.append(at)
        return min(times) if times else None

    def _spawn(self, item: rs.Item, tenant: str | None, *, missed: bool) -> None:
        self._track(asyncio.create_task(self._deliver(item, tenant, missed=missed),
                                        name="due-delivery", context=contextvars.Context()),
                    tenant)

    def _track(self, task: asyncio.Task, tenant: str | None) -> None:
        self._deliveries[task] = tenant
        task.add_done_callback(lambda t: self._deliveries.pop(t, None))

    # ── one item ──

    def _gate(self):
        host = self.speaker
        gate = getattr(getattr(host, "ports", None), "gate", None)
        if gate is None:
            from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy

            gate = CorePolicyGate(session_policy(self._app))
        return gate

    def _passes(self, line: str) -> bool:
        from voxscribe.session.ports import GATE_PASS

        try:
            return self._gate().check(line, (), user_text="").decision == GATE_PASS
        except Exception:  # noqa: BLE001 - fail closed: nothing of the item's words is shown
            return False

    async def _deliver(self, item: rs.Item, tenant: str | None, *, missed: bool) -> None:
        with self._scope(tenant):
            now = self.clock.now()
            how = due_lines.MISSED if missed else due_lines.DUE
            line = due_lines.line(item, now=now, how=how)
            neutral = (due_lines.neutral(item, now=now, how=how)
                       if item.kind == rs.KIND_REMINDER else None)
            shown = item.subject if item.subject and self._passes(line) else None
            window, policy = quiet_hours_policy(self._app)
            spoken, owed, reason = False, False, "no_voice"
            if policy == QUIET_DUE_NOTIFY and in_quiet_hours(window, now.astimezone(item.tz)):
                owed, reason = True, "quiet_hours"
            elif self.speaker is not None:
                from app.local.voice.host import Initiated

                try:
                    result = await self.speaker.speak_initiated(
                        Initiated(item.id, tenant, line, neutral))
                    spoken, owed, reason = result.spoken, result.owed, result.reason
                except Exception:
                    log.exception("due scheduler: speaking failed id=%s", item.id)
                    owed, reason = True, "failed"
            payload = {"id": item.id, "kind": item.kind, "text": shown, "dueAt": item.due_at,
                       "missed": missed, "spoken": spoken}
            try:
                self._publish(EVENT_KIND, payload)
            except Exception:
                log.exception("due scheduler: event not published id=%s", item.id)
            try:
                await self._store().mark_delivered(item.id, at=self.clock.now(),
                                                   spoken=spoken, owed=owed, missed=missed)
            except Exception:
                log.exception("due scheduler: row not marked delivered id=%s", item.id)
        self.delivered.append(Delivery(item.id, tenant, spoken, owed, missed, reason, payload))
        log.info("due scheduler: delivered id=%s kind=%s tenant=%s missed=%s spoken=%s owed=%s"
                 " reason=%s", item.id, item.kind, tenant, missed, spoken, owed, reason)

    # ── owed lines, at the next session ──

    def _session_opened(self, tenant: str | None) -> None:
        """A session opened (the host calls this from inside it): the lines owed to its tenant
        are handed to the host, which says them at the end of the session's first reply."""
        self._track(asyncio.get_running_loop().create_task(
            self._speak_owed(tenant), name="due-owed", context=contextvars.Context()), tenant)

    async def _speak_owed(self, tenant: str | None) -> None:
        from app.local.voice.host import Initiated

        with self._scope(tenant):
            now = self.clock.now()
            store = self._store()
            try:
                owed = await store.claim_owed(now - self.missed_window)
            except Exception:
                log.exception("due scheduler: owed lines not read tenant=%s", tenant)
                return
            for item in owed:
                line = due_lines.line(item, now=now, how=due_lines.OWED)
                neutral = (due_lines.neutral(item, now=now, how=due_lines.OWED)
                           if item.kind == rs.KIND_REMINDER else None)
                try:
                    result = await self.speaker.speak_initiated(
                        Initiated(item.id, tenant, line, neutral))
                    if result.spoken:
                        await store.mark_spoken(item.id)
                except Exception:
                    log.exception("due scheduler: owed line failed id=%s", item.id)
                log.info("due scheduler: owed line id=%s tenant=%s", item.id, tenant)
