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

## Schema Isolation Rule

**Tests NEVER touch the public schema of ideascribe_dev.** All queries run against the throwaway schema created by the fixture. This is enforced by the pool configuration which sets `search_path` to `<test_schema>, public`.

The public schema is used only for system objects (vector type, etc.) that resolve from the search_path. Data rows always live in the test schema and are cleaned up automatically.

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
