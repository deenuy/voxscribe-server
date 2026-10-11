"""Engine status for the app's status chip, and a way to start the AI runtime from it."""

import time

from fastapi import APIRouter, Request

from app.local import runtime_status as rt
from app.local.errors import EngineError

router = APIRouter(prefix="/v1/engine", tags=["engine"])


async def _checks(request: Request) -> tuple[bool, bool]:
    settings = request.app.state.settings
    store = request.app.state.store
    try:
        db_ok, _schema, _name = await store.health_check()
    except Exception:
        db_ok = False
    if settings.providers == "fake":
        return db_ok, True
    try:
        import faster_whisper  # noqa: F401

        asr_ok = True
    except ImportError:
        asr_ok = False
    return db_ok, asr_ok


async def _status(request: Request) -> dict:
    settings = request.app.state.settings
    probe = None if settings.providers == "fake" else await rt.probe_runtime(settings.ollama_host)
    db_ok, asr_ok = await _checks(request)
    return rt.build_status(
        settings=settings, probe=probe, db_ok=db_ok, asr_ok=asr_ok,
        start_requested_at=getattr(request.app.state, "runtime_start_requested_at", None),
    )


@router.get("/status")
async def engine_status(request: Request) -> dict:
    """State (ready, loading, degraded, offline) and one row per model role, from what the runtime reports."""
    return await _status(request)


@router.post("/start")
async def start_runtime(request: Request):
    """Start the local AI runtime if it is not running. Safe to repeat: it never starts a second one.

    Answers 200 with the state when it is already up, 202 when it has been (or just was) asked to start, and 409
    `runtime_not_installed` when there is nothing to start on this machine."""
    from fastapi.responses import JSONResponse

    status = await _status(request)
    if status["runtime"]["reachable"]:
        return status
    now = time.time()
    requested = getattr(request.app.state, "runtime_start_requested_at", None)
    if requested is None or now - requested >= rt.START_COOLDOWN_S:
        cmd = rt.runtime_command(request.app.state.settings)
        if cmd is None:
            raise EngineError("runtime_not_installed", "the AI runtime is not installed on this computer", 409)
        await rt.spawn_runtime(cmd)
        request.app.state.runtime_start_requested_at = now
        status = await _status(request)
    return JSONResponse(status_code=202, content=status)
