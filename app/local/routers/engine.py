"""Engine status for the app's status chip, and a way to start the AI runtime from it."""

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from app.local import runtime_status as rt
from app.local.errors import EngineError
from app.local.version import engine_version

router = APIRouter(prefix="/v1/engine", tags=["engine"])


async def _database(request: Request) -> dict:
    try:
        ok, schema, name = await request.app.state.store.health_check()
        return {"ok": bool(ok), "name": name, "schemaVersion": schema}
    except Exception:
        return {"ok": False, "name": None, "schemaVersion": None}


async def _checks(request: Request, db: dict) -> tuple[bool, bool]:
    settings = request.app.state.settings
    db_ok = db["ok"]
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
    db = await _database(request)
    db_ok, asr_ok = await _checks(request, db)
    status = rt.build_status(
        settings=settings, probe=probe, db_ok=db_ok, asr_ok=asr_ok,
        start_requested_at=getattr(request.app.state, "runtime_start_requested_at", None),
    )
    # The engine's own health, read from the running process and the database, not typed in anywhere.
    started = getattr(request.app.state, "started_at", None)
    status["engine"] = {
        "version": engine_version(),
        "uptimeSeconds": int(time.time() - started) if started else None,
        "startedAt": datetime.fromtimestamp(started, UTC).isoformat().replace("+00:00", "Z") if started else None,
        "database": db,
        "providers": settings.providers,
        "offline": settings.privacy_mode == "offline",
    }
    return status


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
