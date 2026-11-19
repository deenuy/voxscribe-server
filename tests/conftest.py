"""Pytest configuration and fixtures."""

import tempfile
import uuid
from pathlib import Path

import httpx
import psycopg
import pytest

from app.local.config import Config
from app.local.migrations import run_migrations
from tests.testdb import TEST_DATABASE_URL as _DB_URL
from tests.testdb import unsafe_reason


def pytest_sessionstart(session):
    """Refuse the whole run before any test when the test database could be someone's live data (see tests/testdb.py)."""
    reason = unsafe_reason()
    if reason:
        pytest.exit(f"refusing to run the tests: {reason}", returncode=pytest.ExitCode.USAGE_ERROR)


@pytest.fixture
async def throwaway_schema():
    """Create a test database schema and clean it up afterward.

    Yields the schema name. Creates t_<hex>, runs migrations, and always drops on teardown.
    """
    schema = f"t_{uuid.uuid4().hex[:8]}"

    # Create the test schema
    async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.commit()

    # Run migrations in the test schema
    await run_migrations(_DB_URL, schema=schema)

    yield schema

    # Clean up (always, even on test failure)
    try:
        async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await conn.commit()
    except Exception:
        pass


@pytest.fixture
def throwaway_data_dir():
    """A temporary directory for a test's data files, removed on teardown.

    Every test that exercises file I/O should use this instead of /tmp or the
    system default; using a unique tmpdir per test keeps tests isolated and
    prevents the new default data path (~/.local/share/voxscribe) from being
    touched during the test run.
    """
    with tempfile.TemporaryDirectory(prefix="voxscribe_test_") as d:
        yield Path(d)


def make_app(schema: str, data_dir: Path | None = None, **options):
    """Helper factory to create app with test options.

    Args:
        schema: The database schema to use for test isolation
        data_dir: Directory for test data files; defaults to a fresh tempdir
        **options: Overrides for settings (providers, auth_disabled, auth_token, settings)

    Returns: A coroutine that creates the app
    """
    async def _create():
        from app.local.app import create_app

        _data_dir = data_dir or Path(tempfile.mkdtemp(prefix="voxscribe_test_"))
        _data_dir.mkdir(parents=True, exist_ok=True)

        settings = options.get("settings") or Config(
            database_url=_DB_URL,
            data_dir=str(_data_dir),
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
async def client(throwaway_schema, throwaway_data_dir):
    """Async HTTP client for the real app in the current event loop.

    Uses httpx.AsyncClient with ASGITransport so the app runs in the same event loop.
    Runs the lifespan explicitly. Closes everything on teardown.
    """
    from app.local.app import create_app

    settings = Config(
        database_url=_DB_URL,
        data_dir=str(throwaway_data_dir),
        providers="fake",
    )

    app = await create_app(
        settings=settings,
        auth_disabled=True,
        schema=throwaway_schema,
    )

    lifespan_context = app.router.lifespan_context(app)
    await lifespan_context.__aenter__()

    transport = httpx.ASGITransport(app=app)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    yield async_client

    await async_client.aclose()
    await lifespan_context.__aexit__(None, None, None)


@pytest.fixture
async def test_db_url(throwaway_schema):
    """Fixture for tests that directly create pools using create_pool."""
    yield (_DB_URL, throwaway_schema)


@pytest.fixture
async def authed_client(throwaway_schema, throwaway_data_dir):
    """Async HTTP client with authentication enabled."""
    from app.local.app import create_app

    settings = Config(
        database_url=_DB_URL,
        data_dir=str(throwaway_data_dir),
        providers="fake",
    )

    app = await create_app(
        settings=settings,
        auth_disabled=False,
        schema=throwaway_schema,
    )

    token = app.state.auth_token

    lifespan_context = app.router.lifespan_context(app)
    await lifespan_context.__aenter__()

    transport = httpx.ASGITransport(app=app)
    async_client = httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )

    yield async_client

    await async_client.aclose()
    await lifespan_context.__aexit__(None, None, None)


@pytest.fixture
def ws_client(throwaway_schema):
    """Synchronous TestClient for WebSocket testing."""
    import asyncio
    from starlette.testclient import TestClient
    from app.local.app import create_app

    _data_dir = Path(tempfile.mkdtemp(prefix="voxscribe_test_"))

    async def create_app_async():
        return await create_app(
            settings=Config(
                database_url=_DB_URL,
                data_dir=str(_data_dir),
                providers="fake",
            ),
            auth_disabled=True,
            schema=throwaway_schema,
        )

    app = asyncio.run(create_app_async())
    client = TestClient(app)
    yield client
    client.close()
