"""Tests for the request body size limit middleware.

The middleware refuses requests whose Content-Length exceeds the configured limit
before the body is read into memory, with a 413 payload_too_large response.

These are unit tests: they wire the middleware logic directly onto a stub FastAPI
app so no database or lifespan is needed.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport


# ---------------------------------------------------------------------------
# Shared helper: build a stub app with the size-limit middleware
# ---------------------------------------------------------------------------


def _make_stub_app(limit: int) -> FastAPI:
    """A FastAPI app with the same size-limit middleware as app.py uses."""
    app = FastAPI()

    @app.post("/v1/meetings/import")
    async def fake_import():
        return JSONResponse({"ok": True})

    @app.post("/v1/sources")
    async def fake_sources():
        return JSONResponse({"ok": True})

    @app.put("/v1/meetings/{meeting_id}/notes")
    async def fake_notes(meeting_id: str):
        return JSONResponse({"ok": True})

    @app.post("/v1/meetings")
    async def fake_create():
        return JSONResponse({"ok": True})

    if limit > 0:
        _UPLOAD_PATHS = frozenset({"/v1/meetings/import", "/v1/sources"})

        @app.middleware("http")
        async def size_limit_middleware(request, call_next):
            path = request.url.path
            if path in _UPLOAD_PATHS or path.endswith("/notes"):
                cl = request.headers.get("content-length")
                if cl is not None:
                    try:
                        size = int(cl)
                    except ValueError:
                        size = 0
                    if size > limit:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": "payload_too_large",
                                "detail": f"request body exceeds the {limit}-byte limit",
                            },
                        )
            return await call_next(request)

    return app


def _client(app: FastAPI):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


LIMIT = 100  # bytes, used throughout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_import_under_limit_passes():
    """A request under the limit is forwarded to the route handler."""
    app = _make_stub_app(LIMIT)
    body = b'{"kind":"text","text":"hi"}'
    assert len(body) < LIMIT
    async with _client(app) as c:
        resp = await c.post(
            "/v1/meetings/import",
            content=body,
            headers={"content-type": "application/json", "content-length": str(len(body))},
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.anyio
async def test_import_over_limit_rejected():
    """POST /v1/meetings/import with Content-Length over the limit gets 413."""
    app = _make_stub_app(LIMIT)
    body = b"x" * 200
    assert len(body) > LIMIT
    async with _client(app) as c:
        resp = await c.post(
            "/v1/meetings/import",
            content=body,
            headers={"content-type": "application/octet-stream", "content-length": str(len(body))},
        )
    assert resp.status_code == 413
    data = resp.json()
    assert data["error"] == "payload_too_large"
    assert str(LIMIT) in data["detail"]


@pytest.mark.anyio
async def test_sources_over_limit_rejected():
    """POST /v1/sources with Content-Length over the limit gets 413."""
    app = _make_stub_app(LIMIT)
    body = b"y" * 200
    async with _client(app) as c:
        resp = await c.post(
            "/v1/sources",
            content=body,
            headers={"content-type": "application/octet-stream", "content-length": str(len(body))},
        )
    assert resp.status_code == 413
    assert resp.json()["error"] == "payload_too_large"


@pytest.mark.anyio
async def test_notes_over_limit_rejected():
    """PUT /v1/meetings/{id}/notes with Content-Length over the limit gets 413."""
    app = _make_stub_app(LIMIT)
    body = b"n" * 200
    async with _client(app) as c:
        resp = await c.put(
            "/v1/meetings/abc123/notes",
            content=body,
            headers={"content-type": "application/json", "content-length": str(len(body))},
        )
    assert resp.status_code == 413
    assert resp.json()["error"] == "payload_too_large"


@pytest.mark.anyio
async def test_unmonitored_route_not_checked():
    """Routes not in the upload list are never blocked, even with a large Content-Length."""
    app = _make_stub_app(LIMIT)
    body = b"z" * 200  # over the limit
    async with _client(app) as c:
        resp = await c.post(
            "/v1/meetings",
            content=body,
            headers={"content-type": "application/json", "content-length": str(len(body))},
        )
    # May be 200 or 422 (bad body) but never 413
    assert resp.status_code != 413


@pytest.mark.anyio
async def test_missing_content_length_not_blocked():
    """A request with no Content-Length header is not refused."""
    app = _make_stub_app(LIMIT)
    async with _client(app) as c:
        # Send without content-length header; httpx will omit it for small bodies
        resp = await c.post(
            "/v1/meetings/import",
            content=b"short",
            headers={"content-type": "application/json"},
        )
    # 200 or any error from missing/bad body — but not 413
    assert resp.status_code != 413


@pytest.mark.anyio
async def test_zero_limit_disables_middleware():
    """max_upload_bytes=0 means no limit: the middleware is not installed."""
    app = _make_stub_app(0)  # 0 = disabled
    body = b"z" * 1_000_000
    async with _client(app) as c:
        resp = await c.post(
            "/v1/meetings/import",
            content=body,
            headers={"content-type": "application/octet-stream", "content-length": str(len(body))},
        )
    # With the middleware absent the route still runs; 200 expected
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_exact_limit_boundary_passes():
    """A request whose Content-Length equals the limit exactly is allowed."""
    app = _make_stub_app(LIMIT)
    body = b"e" * LIMIT  # exactly at the limit
    async with _client(app) as c:
        resp = await c.post(
            "/v1/meetings/import",
            content=body,
            headers={"content-type": "application/octet-stream", "content-length": str(len(body))},
        )
    assert resp.status_code != 413


@pytest.mark.anyio
async def test_one_over_limit_rejected():
    """A request whose Content-Length is limit+1 is refused."""
    app = _make_stub_app(LIMIT)
    body = b"e" * (LIMIT + 1)
    async with _client(app) as c:
        resp = await c.post(
            "/v1/meetings/import",
            content=body,
            headers={"content-type": "application/octet-stream", "content-length": str(len(body))},
        )
    assert resp.status_code == 413
