"""Tests for stats endpoints."""

import pytest
from datetime import date, timedelta

from app.local import store_misc


@pytest.mark.asyncio
async def test_stats_empty(client):
    """Test GET /stats returns zeros when no data."""
    resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tokensSaved"] == 0
    assert data["meetingsTranscribed"] == 0
    assert data["sourcesAdded"] == 0
    assert data["assistsProvided"] == 0
    assert data["cloudCalls"] == 0
    assert len(data["weekly"]) == 5


@pytest.mark.asyncio
async def test_stats_weekly_zero_fill(client):
    """Test weekly stats are zero-filled for weeks with no data."""
    # Add usage only for today
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await store_misc.record_usage(conn, "test", 100)

    resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["weekly"]) == 5
    # All weeks should be represented, even with zero tokens
    for week in data["weekly"]:
        assert "weekStart" in week
        assert "tokens" in week


@pytest.mark.asyncio
async def test_stats_token_accumulation(client):
    """Test token counting."""
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        await store_misc.record_usage(conn, "feature1", 100)
        await store_misc.record_usage(conn, "feature1", 50)
        await store_misc.record_usage(conn, "feature2", 75)

    resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tokensSaved"] == 225


@pytest.mark.asyncio
async def test_record_usage_helper(client):
    """Test the record_usage helper function."""
    store = client._transport.app.state.store
    async with store.pool.connection() as conn:
        # Record usage for different days
        today = date.today()
        await store_misc.record_usage(conn, "transcription", 1000, day=today)
        await store_misc.record_usage(conn, "transcription", 500, day=today)
        await store_misc.record_usage(conn, "enrichment", 200, day=today - timedelta(days=1))

        # Check accumulated
        cursor = await conn.execute(
            "SELECT SUM(tokens) FROM usage WHERE feature = %s AND day = %s",
            ("transcription", today),
        )
        total = (await cursor.fetchone())[0]
        assert total == 1500
