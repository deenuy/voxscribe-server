"""In-process event bus for the /v1/events stream.

Routers publish small JSON events (source.update, model.progress,
connector.token_needed); the SSE endpoint subscribes and forwards them. One
process, no broker. Slow subscribers drop their oldest events instead of
blocking publishers.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

_MAX_QUEUE = 256
_subscribers: set[asyncio.Queue] = set()


def publish(event_type: str, data: Any) -> None:
    """Fan an event out to every subscriber. Safe to call from any coroutine."""
    event = {"type": event_type, "data": data}
    for queue in list(_subscribers):
        if queue.full():
            try:
                queue.get_nowait()  # drop the oldest
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(event)


@asynccontextmanager
async def subscribe():
    """Yield a queue of events for the lifetime of the context."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE)
    _subscribers.add(queue)
    try:
        yield queue
    finally:
        _subscribers.discard(queue)


def subscriber_count() -> int:
    return len(_subscribers)
