"""VoxScribe processing backend for IdeaScribe.

Docker hosts this service for LLM processing only. Mic capture lives in the
Electron client. Sessions are submitted, queued, and processed by a worker pool.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .errors import AppError
from .logging_config import configure, get_logger
from .queue import make_queue
from .routers import ideas, sessions
from .sessions import process_session

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure(
        level=settings.logging.level,
        fmt=settings.logging.format,
        log_file=settings.logging.log_file or None,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )
    log.info(
        "startup",
        extra={
            "extractor": settings.processing.extractor,
            "asr": settings.processing.asr_backend,
            "queue": settings.queue.backend,
            "workers": settings.queue.worker_concurrency,
        },
    )
    queue = make_queue(process_session)
    await queue.start()
    app.state.queue = queue
    try:
        yield
    finally:
        log.info("shutdown: draining queue workers")
        await queue.stop()


app = FastAPI(title="VoxScribe processing backend", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.tenant_id = request.headers.get("X-Tenant-Id", settings.auth.default_tenant)
    request.state.workspace_id = request.headers.get("X-Workspace-Id", settings.auth.default_workspace)
    request.state.device_id = request.headers.get("X-Device-Id", "default-device")
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    # Skip logging the SSE stream (connection stays open; we log it in the router).
    if "progress" not in request.url.path:
        log.info(
            "http.request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": duration_ms,
                "tenant": request.state.tenant_id,
                "device": request.state.device_id,
            },
        )
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    log.warning(
        "app_error",
        extra={"code": exc.code, "detail": exc.detail, "path": str(request.url.path)},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled_error", extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "an unexpected error occurred"},
    )


@app.get("/health")
def health(request: Request):
    queue = getattr(request.app.state, "queue", None)
    return {
        "status": "ok",
        "queue": settings.queue.backend,
        "queueRunning": queue is not None,
        "workers": settings.queue.worker_concurrency,
        "extractor": settings.processing.extractor,
        "asr": settings.processing.asr_backend,
    }


app.include_router(ideas.router)
app.include_router(sessions.router)
