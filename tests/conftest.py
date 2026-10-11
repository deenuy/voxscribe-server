"""Pytest configuration and fixtures."""

import uuid

import httpx
import psycopg
import pytest

from app.local.config import Config
from app.local.migrations import run_migrations


@pytest.fixture
async def throwaway_schema():
    """Create a test database schema and clean it up afterward.

    Yields the schema name. Creates t_<hex>, runs migrations, and always drops on teardown.
    """
    schema = f"t_{uuid.uuid4().hex[:8]}"
    db_url = "postgresql://localhost:5432/ideascribe_dev"

    # Create the test schema
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.commit()

    # Run migrations in the test schema
    await run_migrations(db_url, schema=schema)

    yield schema

    # Clean up (always, even on test failure)
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await conn.commit()
    except Exception:
        # If cleanup fails, don't fail the test
        pass


def make_app(schema: str, **options):
    """Helper factory to create app with test options.

    Args:
        schema: The database schema to use for test isolation
        **options: Overrides for settings (providers, auth_disabled, auth_token, settings)

    Returns: A coroutine that creates the app
    """
    async def _create():
        from app.local.app import create_app
        from pathlib import Path

        db_url = "postgresql://localhost:5432/ideascribe_dev"
        data_dir = Path("/tmp/voxscribe_test")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Use provided settings or create default test settings
        settings = options.get("settings") or Config(
            database_url=db_url,
            data_dir=str(data_dir),
            providers=options.get("providers", "fake"),
        )

        app = await create_app(
            settings=settings,
            auth_disabled=options.get("auth_disabled", True),
            schema=schema,
        )
        return app

    return _create()


@pytest.fixture
async def client(throwaway_schema):
    """Async HTTP client for the real app in the current event loop.

    Uses httpx.AsyncClient with ASGITransport so the app runs in the same event loop.
    Runs the lifespan explicitly. Closes everything on teardown.
    """
    from app.local.app import create_app
    from pathlib import Path

    schema = throwaway_schema
    db_url = "postgresql://localhost:5432/ideascribe_dev"
    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = Config(
        database_url=db_url,
        data_dir=str(data_dir),
        providers="fake",
    )

    app = await create_app(
        settings=settings,
        auth_disabled=True,
        schema=schema,
    )

    # Keep the lifespan context open for the entire test
    lifespan_context = app.router.lifespan_context(app)
    await lifespan_context.__aenter__()

    # Create async client with ASGI transport
    transport = httpx.ASGITransport(app=app)
    async_client = httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    )

    yield async_client

    # Shutdown
    await async_client.aclose()
    await lifespan_context.__aexit__(None, None, None)


@pytest.fixture
async def test_db_url(throwaway_schema):
    """Fixture for tests that directly create pools using create_pool.

    Yields the database URL with schema configured for the test.
    """
    db_url = "postgresql://localhost:5432/ideascribe_dev"
    schema = throwaway_schema
    # Return a URL with the schema in search_path via connection options
    # Unfortunately, we can't do this in the URL, so we return the base URL
    # and tests must call Store.create or create_pool with schema parameter if available
    # For now, just return the schema for the test to use
    yield (db_url, schema)


@pytest.fixture
async def authed_client(throwaway_schema):
    """Async HTTP client with authentication enabled.

    Sends bearer token with each request.
    """
    from app.local.app import create_app
    from pathlib import Path

    schema = throwaway_schema
    db_url = "postgresql://localhost:5432/ideascribe_dev"
    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = Config(
        database_url=db_url,
        data_dir=str(data_dir),
        providers="fake",
    )

    app = await create_app(
        settings=settings,
        auth_disabled=False,
        schema=schema,
    )

    # Get the token from app state
    token = app.state.auth_token

    # Keep the lifespan context open for the entire test
    lifespan_context = app.router.lifespan_context(app)
    await lifespan_context.__aenter__()

    # Create async client with auth headers and ASGI transport
    transport = httpx.ASGITransport(app=app)
    async_client = httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )

    yield async_client

    # Shutdown
    await async_client.aclose()
    await lifespan_context.__aexit__(None, None, None)

@pytest.fixture
def ws_client(throwaway_schema):
    """Synchronous TestClient for WebSocket testing.

    WebSockets require synchronous TestClient, not async httpx.AsyncClient.
    This client shares the same throwaway schema as the async client.
    """
    import asyncio
    from starlette.testclient import TestClient
    from app.local.app import create_app
    from pathlib import Path

    schema = throwaway_schema
    db_url = "postgresql://localhost:5432/ideascribe_dev"
    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create app using asyncio.run (blocking event loop)
    async def create_app_async():
        return await create_app(
            settings=Config(
                database_url=db_url,
                data_dir=str(data_dir),
                providers="fake",
            ),
            auth_disabled=True,
            schema=schema,
        )

    app = asyncio.run(create_app_async())

    # Return a synchronous TestClient (which has websocket_connect)
    client = TestClient(app)
    yield client
    client.close()
