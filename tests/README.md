# VoxScribe Engine Test Harness

## Fixtures

### `throwaway_schema`
Creates a test database schema (`t_<hex>`), runs migrations, and drops it after the test. Handles cleanup even on test failure. Yields the schema name.

### `client`
HTTP client for the real FastAPI app running in the current event loop. Uses `httpx.AsyncClient` with `ASGITransport`, ensuring the pool is in the same event loop. The lifespan runs within the fixture scope, so database operations work. Access the app via `client._transport.app.state`.

### `authed_client`
Like `client`, but with authentication enabled. Sends a bearer token with each request.

### `test_db_url`
Returns `(database_url, schema_name)` tuple for tests that use `Store.create` or `create_pool` directly. Tests must unpack and pass both to pool creation functions.

## How to Write a Route Test

```python
@pytest.mark.asyncio
async def test_my_endpoint(client):
    """Test a route with proper exception handling."""
    resp = await client.get("/v1/my-endpoint")
    assert resp.status_code == 200
    data = resp.json()
    assert data["field"] == "expected"
```

Key points:
- Use `async def` and `@pytest.mark.asyncio`
- Use `await` on all client calls (it's httpx.AsyncClient, not TestClient)
- Access the store via `client._transport.app.state.store`
- Tests run in the same event loop as the pool, so no deadlocks

## How to Write a Store Test

```python
@pytest.mark.asyncio
async def test_store_operation(test_db_url):
    """Test a store method directly."""
    db_url, schema = test_db_url
    store = await Store.create(db_url, schema=schema)
    
    # Use the store
    meeting = await store.create_meeting("id", "Title")
    assert meeting.title == "Title"
    
    await store.close()
```

Key points:
- Unpack `test_db_url` tuple
- Pass `schema=schema` to `Store.create` or `create_pool`
- The schema is guaranteed to be fresh and isolated
- Remember to call `await store.close()` for resource cleanup

## The test database

Tests use one database, defined once in `tests/testdb.py`: `VOXSCRIBE_TEST_DATABASE_URL`, default
`postgresql://localhost:5432/ideascribe_test`. Never the engine's database. Create it once (safe to repeat):

```bash
make test-db                       # creates ideascribe_test and installs the vector extension in its public schema
make test-db-reset CONFIRM=yes     # drop and recreate it
```

Inside that database every test works in a throwaway schema (`t_<hex>`), created by the fixture and dropped afterwards.
The public schema holds only extension objects (the vector type) that resolve through the search path.

A session guard in `conftest.py` refuses the run, before any test, when the test database is the one the engine's own
configuration resolves to (the default `Config` database, or `VOXSCRIBE_DATABASE_URL` in a `.env` at the repository
root or, from a linked worktree, the main checkout's root), or when its public schema holds rows. Its message names a
database by name, host and port only. `tests/test_boot.py` also fails when any test starts an engine process without
`--database-url`.

## Running Tests

Run all tests:
```bash
uv run --extra dev pytest -q
```

Run one module:
```bash
uv run --extra dev pytest tests/test_data.py -q
```

Run one test:
```bash
uv run --extra dev pytest tests/test_data.py::test_delete_data_requires_confirmation_header -q
```

Tests complete in under 2 minutes and leave zero schema leaks.
