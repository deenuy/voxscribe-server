"""Tests for data deletion endpoint."""

import pytest

from app.local import store_misc


@pytest.mark.asyncio
async def test_delete_data_requires_confirmation_header(client):
    """Test that DELETE /data requires X-Confirm header."""
    # Without header
    resp = await client.delete("/v1/data")
    assert resp.status_code == 400
    assert "required" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_data_wrong_header_value(client):
    """Test that wrong X-Confirm value is rejected."""
    resp = await client.delete("/v1/data", headers={"X-Confirm": "wrong-value"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_data_clears_all_tables(client):
    """Test that DELETE /data with correct header clears all data."""
    # Add some data
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await store_misc.record_usage(conn, "test", 100)
        await store_misc.create_auth_event(conn, "test_event", {"detail": "test"})

    # Verify data exists
    async with store.pool.connection() as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM usage")
        usage_count = (await cursor.fetchone())[0]
        assert usage_count > 0

    # Delete
    resp = await client.delete("/v1/data", headers={"X-Confirm": "delete-all"})
    assert resp.status_code == 204

    # Verify data is gone
    async with store.pool.connection() as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM usage")
        usage_count = (await cursor.fetchone())[0]
        assert usage_count == 0

        cursor = await conn.execute("SELECT COUNT(*) FROM auth_events")
        auth_count = (await cursor.fetchone())[0]
        assert auth_count == 0


@pytest.mark.asyncio
async def test_delete_data_keeps_schema_migrations(client):
    """Test that DELETE /data does not drop schema_migrations table."""
    store = client._transport.app.state.store

    # Delete all data
    resp = await client.delete("/v1/data", headers={"X-Confirm": "delete-all"})
    assert resp.status_code == 204

    # Verify schema_migrations still exists with data
    async with store.pool.connection() as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM schema_migrations")
        migration_count = (await cursor.fetchone())[0]
        # Should have at least 2 migrations (001_init, 002_add_auth)
        assert migration_count >= 2
