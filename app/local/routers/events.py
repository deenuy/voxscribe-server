"""Server-Sent Events stream for real-time updates.

Every event has an `id:` line. A client that reconnects with the standard Last-Event-ID header
(or `lastEventId`, for a client that cannot set headers on a fresh connection) gets every event it
missed, once and in order, or a `stream.reset` event when they are no longer held (events.py).

`?schema=1` sends each event's whole envelope as its data (event_schema.Envelope); without it the
data is the payload alone, as before.
"""

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.local import events
from app.local.errors import InvalidRequestError
from app.local.event_schema import SCHEMA_VERSION

router = APIRouter(tags=["events"])

PING_S = 15     # an idle stream gets a comment line this often


def sse_frame(event: events.Event, *, envelope: bool = False) -> str:
    """One event as an SSE frame. With `envelope`, the data is the event's whole envelope."""
    data = event.envelope() if envelope else event.data
    data_json = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
    return f"id: {event.id}\nevent: {event.type}\ndata: {data_json}\n\n"


async def event_frames(last_event_id: str | None = None, *, envelope: bool = False,
                       ping_s: float = PING_S) -> AsyncIterator[str]:
    """The frames of the stream of the tenant in scope, resuming after `last_event_id`."""
    async with events.subscribe(last_event_id) as sub:
        while True:
            # Idle connections get a comment line, so proxies and the client can tell a quiet
            # stream from a dead one.
            try:
                event = await asyncio.wait_for(sub.next(), timeout=ping_s)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            yield sse_frame(event, envelope=envelope)


@router.get("/v1/events")
async def events_stream(request: Request, schema: int | None = None, lastEventId: str | None = None):
    """Stream server-sent events (event_schema.STREAM_EVENTS), each with an id; replays on reconnect."""
    if schema is not None and schema != SCHEMA_VERSION:
        raise InvalidRequestError(f"event schema {schema} is not served; this engine serves {SCHEMA_VERSION}")
    last = request.headers.get("last-event-id") or lastEventId
    return StreamingResponse(event_frames(last, envelope=schema is not None),
                             media_type="text/event-stream")
