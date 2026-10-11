"""Server-Sent Events stream for real-time updates."""

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.local import events

router = APIRouter(tags=["events"])


@router.get("/v1/events")
async def events_stream(request: Request):
    """Stream server-sent events: source.update, model.progress, connector.token_needed."""

    async def event_generator():
        async with events.subscribe() as queue:
            while True:
                try:
                    event = await queue.get()
                    event_type = event["type"]
                    data = event["data"]

                    # Format as SSE
                    if isinstance(data, dict):
                        data_json = json.dumps(data)
                    else:
                        data_json = str(data)

                    yield f"event: {event_type}\n"
                    yield f"data: {data_json}\n\n"

                except asyncio.CancelledError:
                    break
                except Exception:
                    break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
