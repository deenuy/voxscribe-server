"""PL41 over HTTP: the Google calendar import obeys the network decision.

The real engine on a throwaway schema; the privacy mode is stored through PUT /v1/settings, as the
app stores it. No request leaves the device: a refused import never reaches the connector, and an
allowed one stops at "not connected" because no token was pushed.
"""
from __future__ import annotations

import httpx
import pytest

from app.local.config import Config
from tests.voice.unit.tool_fakes import load_profile


@pytest.fixture
async def engine(throwaway_schema, throwaway_data_dir):
    from app.local.app import create_app

    made = []

    async def start(mode: str, *, env_privacy: str = "offline"):
        settings = Config(database_url=_db_url(), data_dir=str(throwaway_data_dir),
                          providers="fake", privacy_mode=env_privacy)
        app = await create_app(settings=settings, auth_disabled=True, schema=throwaway_schema)
        app.state.profile = load_profile(mode)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                   base_url="http://test")
        made.append((client, ctx))
        return client

    yield start
    for client, ctx in made:
        await client.aclose()
        await ctx.__aexit__(None, None, None)


def _db_url() -> str:
    from tests.conftest import _DB_URL

    return _DB_URL


async def _import(client, privacy: str) -> httpx.Response:
    put = await client.put("/v1/settings", json={"privacyMode": privacy})
    assert put.status_code == 200, put.text
    return await client.post("/v1/calendar/import", json={"kind": "google"})


@pytest.mark.parametrize("mode", ["senior_care", "classroom"])
async def test_google_import_in_a_restricted_mode_is_network_disabled_even_with_connectors(
        engine, mode):
    client = await engine(mode, env_privacy="connectors")
    resp = await _import(client, "connectors")
    assert resp.status_code == 403
    assert resp.json()["error"] == "network_disabled"


async def test_google_import_reads_the_stored_privacy_setting_not_the_environment(engine):
    client = await engine("personal", env_privacy="connectors")
    resp = await _import(client, "offline")
    assert resp.status_code == 403 and resp.json()["error"] == "network_disabled"


async def test_google_import_with_connectors_stored_passes_the_decision(engine):
    client = await engine("personal", env_privacy="offline")
    resp = await _import(client, "connectors")
    # Past the decision: the import now needs a connected account, which this engine has none of.
    assert resp.status_code == 409, resp.text
