"""Chat endpoints."""

import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.local.chat_grounding import AsyncAnswerComposer
from app.local.errors import NotFoundError
from app.local.models import Citation, CreateThreadRequest
from app.local.pg_sync_index import PostgresKnowledgeIndexSync, PostgresSyncRetriever
from app.local.store_chat import ChatStore

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


@router.post("/threads/{thread_id}/messages")
async def send_message(request: Request, thread_id: str, body: dict):
    """Send a message and stream the response."""

    store = ChatStore(request.app.state.store.pool)
    providers = request.app.state.providers

    thread = await store.get_thread(thread_id)
    if not thread:
        raise NotFoundError(f"thread {thread_id} not found")

    text = body.get("text")
    if not text:
        raise ValueError("message text is required")

    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())

    # Save user message
    await store.add_message(user_message_id, thread_id, "user", text, [])

    # Determine scope for retrieval
    scope_source_ids = thread.scope.sourceIds if thread.scope else None

    # Sync index and retriever (the composer runs off the event loop)
    embedder = providers["embedding_provider"]
    llm_provider = providers["llm_provider"]
    db_url = request.app.state.settings.database_url

    # Use sync index for core's Retriever
    sync_index = PostgresKnowledgeIndexSync(db_url)
    sync_retriever = PostgresSyncRetriever(sync_index, embedder)

    # Wrap with async composer
    composer = AsyncAnswerComposer(llm_provider, sync_retriever)

    # Stream response
    async def response_generator():
        try:
            # Get token stream and citations (async)
            token_stream, citations = await composer.stream(text, scope_source_ids=scope_source_ids)

            # Stream tokens (stream is synchronous generator)
            collected_text = ""
            for token in token_stream:
                if token:
                    escaped = token.replace('"', '\\"').replace('\n', '\\n')
                    yield f'event: token\ndata: {{"text": "{escaped}"}}\n\n'
                    collected_text += token

            # Yield citations
            citation_models = [
                Citation(
                    kind=c.kind,
                    sourceId=c.source_id,
                    chunkId=c.chunk_id,
                    title=c.title,
                    locator=c.locator,
                    segmentId=c.segment_id,
                    startMs=c.start_ms,
                    endMs=c.end_ms,
                )
                for c in citations
            ]
            citations_data = [c.model_dump(exclude_none=True) for c in citation_models]
            yield f'event: citations\ndata: {json.dumps({"citations": citations_data})}\n\n'

            # Save assistant message with citations
            await store.add_message(assistant_message_id, thread_id, "assistant", collected_text, citation_models)

            # Emit done event
            yield f'event: done\ndata: {json.dumps({"messageId": assistant_message_id})}\n\n'

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f'event: error\ndata: {json.dumps({"error": "internal_error", "detail": str(e)})}\n\n'

    return StreamingResponse(response_generator(), media_type="text/event-stream")
