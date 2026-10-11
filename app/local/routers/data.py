"""Data endpoints: full data deletion."""

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from app.local.errors import InvalidRequestError

router = APIRouter(prefix="/v1", tags=["data"])


@router.delete("/data")
async def delete_all_data(request: Request, x_confirm: str | None = Header(None), models: str | None = None) -> JSONResponse:
    """Delete all data from all engine tables.

    Requires header: X-Confirm: delete-all

    Query params:
        models: if 'true', also delete downloaded Ollama models

    Returns 204 on success. Never drops tables, only deletes rows.
    Safe in throwaway test schemas.
    """
    if x_confirm != "delete-all":
        raise InvalidRequestError("X-Confirm: delete-all header required")

    store = request.app.state.store
    providers_mode = request.app.state.settings.providers

    async with store.pool.connection() as conn:
        # Delete all rows from engine tables (in reverse dependency order to avoid FK issues)
        tables = [
            "tasks",
            "transcript_edits",
            "chat_messages",
            "chat_threads",
            "meeting_attachments",
            "enhanced_blocks",
            "notes_blocks",
            "transcript_segments",
            "summaries",
            "idea_prds",
            "idea_activity",
            "idea_edges",
            "ideas",
            "features",
            "source_labels",
            "source_chunks",
            "sources",
            "labels",
            "meetings",
            "projects",
            "calendar_events",
            "settings",
            "usage",
            "connector_accounts",
            "auth_events",
        ]

        for table in tables:
            try:
                await conn.execute(f"DELETE FROM {table}")
            except Exception:
                # Table might not exist, skip
                pass

        await conn.commit()

    # Delete Ollama models if requested
    if models == "true" and providers_mode == "local":
        import httpx

        ollama_host = request.app.state.settings.ollama_host
        model_ids = ["gemma4:12b", "gemma4:e4b", "gemma4:e2b"]

        for model_id in model_ids:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.delete(f"{ollama_host}/api/delete", json={"name": model_id})
            except Exception:
                # Model might not exist, continue
                pass

    return JSONResponse(status_code=204, content={})
