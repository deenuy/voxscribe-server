"""The public health route with tenant isolation on.

Health carries no person, so there is no tenant in scope when it runs. Its database check reads only the
platform's own schema version, so it must still answer with the database's real state: a status page, the
desktop app before sign-in and `make status` all call it with no identity.
"""
from __future__ import annotations

import httpx

from app.local.app import create_app
from app.local.config import Config
from tests.testdb import TEST_DATABASE_URL


async def test_health_reports_the_database_with_isolation_on_and_nobody_signed_in(throwaway_schema, throwaway_data_dir):
    app = await create_app(
        settings=Config(
            database_url=TEST_DATABASE_URL,
            data_dir=str(throwaway_data_dir),
            providers="fake",
            identity_issuer="https://accounts.test",
            identity_audience="voxscribe-engine",
            identity_jwks_url="http://jwks.test/keys",
        ),
        auth_disabled=True,
        schema=throwaway_schema,
    )
    assert app.state.settings.tenancy_enforced
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://engine.test") as http:
            resp = await http.get("/v1/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["database"]["ok"] is True
            assert body["database"]["schemaVersion"] > 0
            # A route that does carry a person's data still refuses a caller with no identity.
            assert (await http.get("/v1/meetings")).status_code == 401
