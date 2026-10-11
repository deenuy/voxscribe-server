"""In-memory meeting event bus for live WebSocket pub/sub."""

import asyncio
from typing import Any

_buses: dict[str, "MeetingBus"] = {}


class MeetingBus:
    """Per-meeting pub/sub for live events (enhanced.update, summary.ready, status)."""

    def __init__(self, meeting_id: str):
        self.meeting_id = meeting_id
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    async def subscribe(self) -> asyncio.Queue:
        """Get the event queue for this meeting."""
        return self.queue

    async def publish(self, event: dict) -> None:
        """Publish an event to this meeting's subscribers."""
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.queue.put(event)


def get_bus(meeting_id: str) -> MeetingBus:
    """Get or create the bus for a meeting."""
    if meeting_id not in _buses:
        _buses[meeting_id] = MeetingBus(meeting_id)
    return _buses[meeting_id]


async def publish(meeting_id: str, event_type: str, data: Any) -> None:
    """Publish an event to a meeting's subscribers."""
    bus = get_bus(meeting_id)
    event = {"type": event_type, "data": data}
    await bus.publish(event)
