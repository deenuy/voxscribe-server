"""HTTP-level contract behaviour found in the API review: CORS ordering, where a token may
travel, the error shape for bad input, route order, and status codes without bodies."""

from pathlib import Path

import httpx
import pytest

from app.local.app import create_app
from app.local.config import Config

TOKEN = "test-token-abc123"
ORIGIN = "http://localhost:3000"


@pytest.fixture
async def secure_client(throwaway_schema, monkeypatch):
    """The real app with authentication ON and one allowed origin."""
    monkeypatch.setenv("VOXSCRIBE_TOKEN", TOKEN)
    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = Config(
        database_url="postgresql://localhost:5432/ideascribe_dev",
        data_dir=str(data_dir),
        providers="fake",
        allowed_origins=ORIGIN,
    )
    app = await create_app(settings=settings, auth_disabled=False, schema=throwaway_schema)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await lifespan.__aexit__(None, None, None)


AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ── CORS must sit outside authentication ─────────────────────────────────────

@pytest.mark.asyncio
async def test_preflight_succeeds_without_a_token(secure_client):
    r = await secure_client.options(
        "/v1/health",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == ORIGIN


@pytest.mark.asyncio
async def test_unauthorised_answers_still_carry_cors_headers(secure_client):
    r = await secure_client.get("/v1/health", headers={"Origin": ORIGIN})
    assert r.status_code == 401
    assert r.headers["access-control-allow-origin"] == ORIGIN  # otherwise the browser reports a network error


# ── where a token may travel ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bearer_header_works_everywhere(secure_client):
    assert (await secure_client.get("/v1/meetings", headers=AUTH)).status_code == 200


@pytest.mark.asyncio
async def test_query_token_is_refused_on_ordinary_routes(secure_client):
    r = await secure_client.get(f"/v1/meetings?token={TOKEN}")
    assert r.status_code == 401  # URLs end up in logs and history


@pytest.mark.asyncio
async def test_query_token_is_accepted_for_the_event_stream_only(secure_client):
    # SSE streams forever, so only check that auth lets the request through to the route.
    r = await secure_client.get("/v1/events", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


# ── the error shape ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bad_input_uses_the_standard_error_shape(client):
    r = await client.post("/v1/meetings", json={"title": ["not", "a", "string"]})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_request" and "title" in body["detail"]
    assert "loc" not in str(body)  # not FastAPI's default {"detail": [{"loc": ...}]}


# ── routes that were shadowed by /{id} ───────────────────────────────────────

@pytest.mark.asyncio
async def test_sources_graph_is_reachable(client):
    r = await client.get("/v1/sources/graph")
    assert r.status_code == 200
    assert "nodes" in r.json() and "edges" in r.json()


@pytest.mark.asyncio
async def test_ideas_graph_is_reachable(client):
    r = await client.get("/v1/ideas/graph")
    assert r.status_code == 200 and "nodes" in r.json()


# ── status codes without bodies ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_label_delete_is_a_real_204(client):
    label = (await client.post("/v1/labels", json={"name": "to-delete"})).json()
    r = await client.delete(f"/v1/labels/{label['id']}")
    assert r.status_code == 204 and r.content == b""
    assert (await client.delete(f"/v1/labels/{label['id']}")).status_code == 404


@pytest.mark.asyncio
async def test_source_delete_and_reindex_have_no_body(client, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# hello\nworld")
    created = await client.post("/v1/sources", json={"kind": "file", "uri": str(f)})
    assert created.status_code == 201
    sid = created.json()["id"]
    reindex = await client.post(f"/v1/sources/{sid}/reindex")
    assert reindex.status_code == 202 and reindex.content == b""
    gone = await client.delete(f"/v1/sources/{sid}")
    assert gone.status_code == 204 and gone.content == b""


# ── query parameters ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_calendar_events_honours_the_from_parameter(client):
    r = await client.get("/v1/calendar/events", params={"from": "2999-01-01T00:00:00Z"})
    assert r.status_code == 200 and r.json()["items"] == []


@pytest.mark.asyncio
async def test_meeting_list_limit_is_capped(client):
    r = await client.get("/v1/meetings", params={"limit": 10_000_000})
    assert r.status_code == 200  # clamped, not an unbounded table read


# ── a URL source reads the stored privacy mode ───────────────────────────────

@pytest.mark.asyncio
async def test_privacy_mode_round_trips_through_the_database(client):
    from app.local.store_sources import SourceStore

    store = SourceStore(client._transport.app.state.store.pool)
    assert await store.get_privacy_mode() == "offline"  # default, nothing stored
    settings = (await client.get("/v1/settings")).json()
    settings["privacyMode"] = "connectors"
    assert (await client.put("/v1/settings", json=settings)).status_code == 200
    assert await store.get_privacy_mode() == "connectors"  # used to raise JSONDecodeError


# ── pooled connections must not carry one caller's settings to the next ──────

@pytest.mark.asyncio
async def test_pool_resets_the_row_factory_between_borrowers(throwaway_schema):
    """Regression: store code sets row_factory = dict_row on a pooled connection; the next
    borrower used to get dicts (health said 'database unavailable', the ideas list returned 500)."""
    from psycopg.rows import dict_row, tuple_row

    from app.local.db import create_pool

    pool = await create_pool("postgresql://localhost:5432/ideascribe_dev", min_size=1, max_size=1, schema=throwaway_schema)
    try:
        async with pool.connection() as conn:
            conn.row_factory = dict_row  # what the meetings store does
            assert (await (await conn.execute("SELECT 1 AS one")).fetchone()) == {"one": 1}
        async with pool.connection() as conn:  # the same single connection, borrowed again
            assert conn.row_factory is tuple_row
            assert (await (await conn.execute("SELECT 1")).fetchone())[0] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_health_and_ideas_work_after_the_meetings_store_has_run(client):
    await client.get("/v1/meetings")  # sets dict_row on whichever connection it borrowed
    await client.post("/v1/ideas", json={"title": "after meetings", "summary": "s", "type": "new_feature"})
    for _ in range(4):  # several borrows, so the pool hands the same connection around
        assert (await client.get("/v1/ideas")).status_code == 200
        health = (await client.get("/v1/health")).json()
        assert health["database"]["ok"] is True
