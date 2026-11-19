"""In-memory meeting event bus for live WebSocket pub/sub.

Every watcher of a meeting gets every event (event_schema.MEETING_EVENTS): each has its own
bounded queue, which drops its oldest event instead of blocking a publisher. Events published
while nobody watches are held (bounded the same way) for the first watcher, so the events of a
meeting's first moments are not lost to a socket that subscribes a moment late; a watcher that
joins later gets only what is published after it joined.
"""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from . import tenancy

_MAX_QUEUE = 256

# (tenant, meeting id) -> bus. A meeting id is only unique inside its tenant, and one tenant must never receive
# another's live events.
_buses: dict[tuple[str | None, str], "MeetingBus"] = {}


def _put(queue: asyncio.Queue, event: dict) -> None:
    if queue.full():
        try:
            queue.get_nowait()          # drop the oldest
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(event)


class MeetingBus:
    """Per-meeting pub/sub for live events (enhanced.update, summary.ready, status, ...)."""

    def __init__(self, meeting_id: str):
        self.meeting_id = meeting_id
        self._watchers: set[asyncio.Queue] = set()
        self._held: deque[dict] = deque(maxlen=_MAX_QUEUE)     # published while nobody watched

    async def subscribe(self) -> asyncio.Queue:
        """A new watcher's own queue of this meeting's events. Call unsubscribe() when done."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE)
        while self._held:
            _put(queue, self._held.popleft())
        self._watchers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._watchers.discard(queue)

    @asynccontextmanager
    async def watch(self):
        """A watcher's queue for the lifetime of the context."""
        queue = await self.subscribe()
        try:
            yield queue
        finally:
            self.unsubscribe(queue)

    def watcher_count(self) -> int:
        return len(self._watchers)

    async def publish(self, event: dict) -> None:
        """Publish an event to every watcher of this meeting."""
        if not self._watchers:
            self._held.append(event)
            return
        for queue in list(self._watchers):
            _put(queue, event)


def get_bus(meeting_id: str) -> MeetingBus:
    """Get or create the bus for a meeting."""
    key = (tenancy.current_tenant(), meeting_id)
    if key not in _buses:
        _buses[key] = MeetingBus(meeting_id)
    return _buses[key]


async def publish(meeting_id: str, event_type: str, data: Any) -> None:
    """Publish an event to a meeting's watchers."""
    bus = get_bus(meeting_id)
    event = {"type": event_type, "data": data}
    await bus.publish(event)


def drop_tenant(tenant: str | None) -> int:
    """Forget every bus of this tenant, with the events held for a first watcher (erase). Returns how many."""
    keys = [k for k in _buses if k[0] == tenant]
    for key in keys:
        _buses.pop(key)._held.clear()
    return len(keys)
