"""Chat endpoints."""

import json
import logging
import uuid

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from voxscribe.knowledge.llm_stream import ProviderUnavailable

from app.local.chat_grounding import ground_in_scribe, ground_in_sources, stream_off_loop
from app.local.errors import NotFoundError, ProviderUnavailableError
from app.local.models import Citation, CreateThreadRequest, SendMessageRequest
from app.local.store_chat import ChatStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/threads", status_code=201)
async def create_thread(request: Request, body: CreateThreadRequest | None = None):
    """Create a new chat thread."""
    store = ChatStore(request.app.state.store.pool)
    thread_id = str(uuid.uuid4())

    scope = body.scope if body else None
    label_ids = scope.labelIds if scope else None
    meeting_id = scope.meetingId if scope else None
    source_ids = scope.sourceIds if scope else None

    thread = await store.create_thread(
        thread_id=thread_id,
        label_ids=label_ids,
        meeting_id=meeting_id,
        source_ids=source_ids,
    )
    return thread.model_dump(exclude_none=True)


@router.get("/threads")
async def list_threads(request: Request):
    """List chat threads."""
    store = ChatStore(request.app.state.store.pool)
    threads = await store.list_threads()
    return [t.model_dump(exclude_none=True) for t in threads]


@router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str):
    """Get a thread with messages."""
    store = ChatStore(request.app.state.store.pool)
    thread = await store.get_thread(thread_id)
    if not thread:
        raise NotFoundError(f"thread {thread_id} not found")
    messages = await store.get_messages(thread_id)
    return {
        "thread": thread.model_dump(exclude_none=True),
        "messages": [m.model_dump(exclude_none=True) for m in messages],
    }


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(request: Request, thread_id: str):
    """Delete a thread and its messages. Idempotent: deleting one that is already gone is still 204."""
    store = ChatStore(request.app.state.store.pool)
    await store.delete_thread(thread_id)
    return Response(status_code=204)


@router.post("/threads/{thread_id}/messages")
async def send_message(request: Request, thread_id: str, body: SendMessageRequest):
    """Send a message and stream the answer as server-sent events.

    `event: token` {"text"} for each piece of the answer, then `event: citations` {"citations"} and `event: done`
    {"messageId"}; or `event: error` {"error", "detail"}. A scribe thread (scope.meetingId) is answered from that
    scribe's transcript, any other from the person's sources (chat_grounding.py). Every database read goes through
    the tenant pool, and the model runs on a worker thread, never on the event loop."""
    pool = request.app.state.store.pool
    store = ChatStore(pool)
    providers = request.app.state.providers

    thread = await store.get_thread(thread_id)
    if not thread:
        raise NotFoundError(f"thread {thread_id} not found")

    text = body.text  # already trimmed, non-empty and bounded by the request model

    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())
    await store.add_message(user_message_id, thread_id, "user", text, [])

    scope = thread.scope
    llm = providers["llm_provider"]
    fake = getattr(request.app.state.settings, "providers", "") == "fake"

    async def response_generator():
        try:
            if scope and scope.meetingId:
                grounding = await ground_in_scribe(pool, llm, scope.meetingId, text)
            else:
                grounding = await ground_in_sources(pool, llm, providers["embedding_provider"], text,
                                                    scope.sourceIds if scope else None, fake_embedder=fake)
            pieces: list[str] = []
            async for token in stream_off_loop(grounding.tokens):
                token = (token or "").replace("\x00", "")    # PostgreSQL text cannot hold NUL; sent as stored
                if token:
                    pieces.append(token)
                    yield _sse("token", {"text": token})

            citation_models = [
                Citation(kind=c.kind, sourceId=c.source_id, chunkId=c.chunk_id, title=c.title, locator=c.locator,
                         segmentId=c.segment_id, startMs=c.start_ms, endMs=c.end_ms)
                for c in grounding.citations
            ]
            yield _sse("citations", {"citations": [c.model_dump(exclude_none=True) for c in citation_models]})
            answer = "".join(pieces)
            await store.add_message(assistant_message_id, thread_id, "assistant", answer, citation_models)
            log.info("chat: answered thread=%s scribe=%s model=%s answer_chars=%d citations=%d",
                     thread_id, bool(scope and scope.meetingId), grounding.model, len(answer), len(citation_models))
            yield _sse("done", {"messageId": assistant_message_id})
        except Exception as exc:  # noqa: BLE001 - the stream has started: the error goes in it
            # The exception's text can carry a prompt or a model's words: only its type is logged or sent.
            log.error("chat: answer failed thread=%s error=%s", thread_id, type(exc).__name__)
            code, detail = (("provider_unavailable", "the on-device AI could not be reached")
                            if isinstance(exc, (ProviderUnavailable, ProviderUnavailableError))
                            else ("internal_error", "the answer could not be generated"))
            yield _sse("error", {"error": code, "detail": detail})

    return StreamingResponse(response_generator(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    """One server-sent event. json.dumps escapes quotes, backslashes, newlines and every control character, so the
    data is always one valid JSON line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
