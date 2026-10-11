"""In-process event bus for the /v1/events stream.

Routers publish small JSON events (source.update, model.progress,
connector.token_needed); the SSE endpoint subscribes and forwards them. One
process, no broker. Slow subscribers drop their oldest events instead of
blocking publishers.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from . import tenancy

_MAX_QUEUE = 256
# queue -> the tenant it listens for (None is the single shared scope, used when isolation is off)
_subscribers: dict[asyncio.Queue, str | None] = {}


def publish(event_type: str, data: Any, *, broadcast: bool = False) -> None:
    """Fan an event out to the subscribers of the tenant in scope. Safe to call from any coroutine.

    Events carry whole records (a source, an idea, a meeting), so they only ever go to the tenant they belong to.
    `broadcast=True` is for facts about the machine, not a person (a model download's progress): everyone gets it."""
    event = {"type": event_type, "data": data}
    tenant = tenancy.current_tenant()
    for queue, listener in list(_subscribers.items()):
        if not broadcast and listener != tenant:
            continue
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
    _subscribers[queue] = tenancy.current_tenant()
    try:
        yield queue
    finally:
        _subscribers.pop(queue, None)


def subscriber_count() -> int:
    return len(_subscribers)
