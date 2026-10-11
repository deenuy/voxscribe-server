"""Models endpoints: listing, selection, and download."""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.local import events
from app.local.errors import ConflictError, InvalidRequestError, NotFoundError
from app.local.model_service import (
    build_model_list,
    assign_role,
    set_selected_roles,
)
from app.local.models import Model

router = APIRouter(prefix="/v1/models", tags=["models"])

# Track in-flight downloads: {model_id: True}
_downloading = {}


@router.get("")
async def list_models(request: Request) -> dict:
    """Get list of all models with current status and selection state."""
    providers_mode = request.app.state.settings.providers
    ollama_host = request.app.state.settings.ollama_host

    # Build model list from reality
    models = await build_model_list(
        request.app.state.settings,
        ollama_host,
        providers_mode,
    )

    return {
        "models": [m.model_dump(exclude_none=True) for m in models],
    }


@router.post("/{model_id}/select")
async def select_model(request: Request, model_id: str, body: dict) -> dict:
    """Select a model for a role (main or live).

    Body: {"role": "main" | "live"} (default: main)

    Returns the updated Model.
    """
    role = body.get("role", "main")
    if role not in ("main", "live"):
        raise InvalidRequestError("role must be 'main' or 'live'")

    # Get current model list to verify the model exists and is ready
    models = await build_model_list(
        request.app.state.settings,
        request.app.state.settings.ollama_host,
        request.app.state.settings.providers,
    )

    model = None
    for m in models:
        if m.id == model_id:
            model = m
            break

    if not model:
        raise NotFoundError(f"model {model_id} not found")

    # Expert models cannot be selected
    if model.kind == "expert":
        raise InvalidRequestError("expert models cannot be selected")

    # Model must be ready
    if model.status != "ready":
        raise ConflictError(f"model {model_id} is not ready (status: {model.status})")

    # Update selection: set this role for this model, clear it from others
    store = request.app.state.store

    # Get current roles for all models from database
    async with store.pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT key, value FROM settings WHERE key LIKE %s",
            ("model_role_%",),
        )
        rows = await cursor.fetchall()
        model_roles = {}
        for key, value in rows:
            # Key is "model_role_main" or "model_role_live"
            r = key.split("_")[-1]
            import json
            try:
                model_id_stored = json.loads(value)
                model_roles[r] = model_id_stored
            except (json.JSONDecodeError, ValueError):
                pass

        # Update: set role for this model, clear from others
        model_roles[role] = model_id

        # Store back
        import json
        for r, mid in model_roles.items():
            await conn.execute(
                "INSERT INTO settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (f"model_role_{r}", json.dumps(mid)),
            )
        await conn.commit()

    # Update in-memory selection
    assign_role(role, model_id)

    # Return updated model
    models = await build_model_list(
        request.app.state.settings,
        request.app.state.settings.ollama_host,
        request.app.state.settings.providers,
    )
    for m in models:
        if m.id == model_id:
            return m.model_dump(exclude_none=True)

    raise NotFoundError(f"model {model_id} not found after selection")


@router.post("/{model_id}/download")
async def download_model(request: Request, model_id: str) -> JSONResponse:
    """Start downloading a model.

    Returns 202 Accepted. Progress events are published to the event bus
    as model.progress with the model object.
    """
    if model_id in _downloading:
        raise ConflictError(f"model {model_id} is already downloading")

    # Get model list to find the model
    models = await build_model_list(
        request.app.state.settings,
        request.app.state.settings.ollama_host,
        request.app.state.settings.providers,
    )

    model = None
    for m in models:
        if m.id == model_id:
            model = m
            break

    if not model:
        raise NotFoundError(f"model {model_id} not found")

    # Start download in background
    providers_mode = request.app.state.settings.providers

    if providers_mode == "fake":
        # Fake mode: simulate download with deterministic progress
        asyncio.create_task(_simulate_download(model_id))
    else:
        # Local mode: download from Ollama
        ollama_host = request.app.state.settings.ollama_host
        if model_id.startswith("gemma") or model_id.startswith("nomic"):
            asyncio.create_task(_download_ollama_model(model_id, ollama_host))
        elif model_id == "faster-whisper":
            asyncio.create_task(_download_faster_whisper_model(model_id))

    return JSONResponse(status_code=202, content={})


async def _simulate_download(model_id: str) -> None:
    """Simulate a model download with deterministic progress."""
    _downloading[model_id] = True
    try:
        model_info = {
            "gemma4:12b": 7_000_000_000,
            "gemma4:e4b": 2_500_000_000,
            "gemma4:e2b": 1_300_000_000,
        }
        total_bytes = model_info.get(model_id, 1_000_000_000)

        # Simulate progress over 2 seconds
        for i in range(10):
            await asyncio.sleep(0.2)
            downloaded = int(total_bytes * (i + 1) / 10)
            progress_model = Model(
                id=model_id,
                kind="generative",
                name=model_id,
                description="",
                status="downloading",
                downloadedBytes=downloaded,
                required=False,
                selected=False,
                selectedRoles=[],
            )
            events.publish("model.progress", progress_model.model_dump(exclude_none=True))

        # Download complete
        progress_model = Model(
            id=model_id,
            kind="generative",
            name=model_id,
            description="",
            status="ready",
            downloadedBytes=total_bytes,
            required=False,
            selected=False,
            selectedRoles=[],
        )
        events.publish("model.progress", progress_model.model_dump(exclude_none=True))
    finally:
        _downloading.pop(model_id, None)


async def _download_ollama_model(model_id: str, ollama_host: str) -> None:
    """Download a model from Ollama via POST /api/pull."""
    import httpx

    _downloading[model_id] = True
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            # Stream the pull response
            async with client.stream(
                "POST",
                f"{ollama_host}/api/pull",
                json={"name": model_id},
            ) as response:
                if response.status_code != 200:
                    progress_model = Model(
                        id=model_id,
                        kind="generative",
                        name=model_id,
                        description="",
                        status="failed",
                        required=False,
                        selected=False,
                        selectedRoles=[],
                        error=f"HTTP {response.status_code}",
                    )
                    events.publish("model.progress", progress_model.model_dump(exclude_none=True))
                    return

                # Parse stream of JSON lines: {status, completed, total}
                total_bytes = 0
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        status = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        total_bytes = max(total_bytes, total)

                        if "error" in data:
                            progress_model = Model(
                                id=model_id,
                                kind="generative",
                                name=model_id,
                                description="",
                                status="failed",
                                required=False,
                                selected=False,
                                selectedRoles=[],
                                error=data["error"],
                            )
                            events.publish("model.progress", progress_model.model_dump(exclude_none=True))
                            return

                        if status == "success":
                            progress_model = Model(
                                id=model_id,
                                kind="generative",
                                name=model_id,
                                description="",
                                status="ready",
                                downloadedBytes=total_bytes,
                                required=False,
                                selected=False,
                                selectedRoles=[],
                            )
                            events.publish("model.progress", progress_model.model_dump(exclude_none=True))
                            return

                        # Throttle progress events
                        if completed % (10 * 1024 * 1024) < 100:  # ~every 10MB
                            progress_model = Model(
                                id=model_id,
                                kind="generative",
                                name=model_id,
                                description="",
                                status="downloading",
                                downloadedBytes=completed,
                                required=False,
                                selected=False,
                                selectedRoles=[],
                            )
                            events.publish("model.progress", progress_model.model_dump(exclude_none=True))
                    except Exception:
                        pass
    except Exception as e:
        progress_model = Model(
            id=model_id,
            kind="generative",
            name=model_id,
            description="",
            status="failed",
            required=False,
            selected=False,
            selectedRoles=[],
            error=str(e),
        )
        events.publish("model.progress", progress_model.model_dump(exclude_none=True))
    finally:
        _downloading.pop(model_id, None)


async def _download_faster_whisper_model(model_id: str) -> None:
    """Download a faster-whisper model using faster_whisper.utils.download_model."""
    try:
        import faster_whisper.utils

        model_name = "small.en"  # Default, could be configurable
        loop = asyncio.get_event_loop()

        # Run download in thread pool
        await loop.run_in_executor(
            None,
            faster_whisper.utils.download_model,
            model_name,
        )

        # Report success
        progress_model = Model(
            id=model_id,
            kind="expert",
            name="Faster Whisper",
            description="",
            status="ready",
            required=True,
            selected=False,
            selectedRoles=[],
        )
        events.publish("model.progress", progress_model.model_dump(exclude_none=True))
    except Exception as e:
        progress_model = Model(
            id=model_id,
            kind="expert",
            name="Faster Whisper",
            description="",
            status="failed",
            required=True,
            selected=False,
            selectedRoles=[],
            error=str(e),
        )
        events.publish("model.progress", progress_model.model_dump(exclude_none=True))
    finally:
        _downloading.pop(model_id, None)


@router.delete("/{model_id}")
async def delete_model(request: Request, model_id: str) -> JSONResponse:
    """Delete a downloaded model.

    Returns 204. Cannot delete required models (409 conflict).
    """
    # Get model list
    models = await build_model_list(
        request.app.state.settings,
        request.app.state.settings.ollama_host,
        request.app.state.settings.providers,
    )

    model = None
    for m in models:
        if m.id == model_id:
            model = m
            break

    if not model:
        raise NotFoundError(f"model {model_id} not found")

    if model.required:
        raise ConflictError("cannot delete required models")

    # Delete from Ollama
    providers_mode = request.app.state.settings.providers
    if providers_mode == "local":
        import httpx

        ollama_host = request.app.state.settings.ollama_host
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.delete(f"{ollama_host}/api/delete", json={"name": model_id})
                if response.status_code not in (200, 404):
                    raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            raise ConflictError(f"failed to delete model: {e}") from e

    return JSONResponse(status_code=204, content={})
