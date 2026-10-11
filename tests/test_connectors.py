"""Tests for Google connector endpoints."""

import pytest
from datetime import datetime, timedelta

from app.local import store_misc, connector_tokens


@pytest.mark.asyncio
async def test_google_connector_not_connected_initially(client):
    """Test that Google connector is not connected initially."""
    resp = await client.get("/v1/connectors/google")
    assert resp.status_code == 200
    data = resp.json()
    assert data["connected"] is False
    assert data["account"] is None
    assert data["scopes"] == []


@pytest.mark.asyncio
async def test_google_connector_put_stores_account(client):
    """Test PUT /connectors/google stores account metadata."""
    now = datetime.utcnow()
    expires_at = (now + timedelta(hours=1)).isoformat() + "Z"

    body = {
        "account": {
            "sub": "test-sub-123",
            "email": "test@example.com",
            "name": "Test User",
            "picture": "https://example.com/pic.jpg",
        },
        "accessToken": "test-token-xyz",
        "expiresAt": expires_at,
        "scopes": ["openid", "email", "profile"],
    }

    resp = await client.put("/v1/connectors/google", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["connected"] is True
    assert data["account"]["email"] == "test@example.com"
    assert set(data["scopes"]) == {"openid", "email", "profile"}


@pytest.mark.asyncio
async def test_google_connector_token_not_returned(client):
    """Test that access tokens are never returned in responses."""
    expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
    body = {
        "account": {
            "sub": "test-sub",
            "email": "test@example.com",
            "name": "Test User",
        },
        "accessToken": "secret-token-xyz",
        "expiresAt": expires_at,
        "scopes": ["openid"],
    }

    resp = await client.put("/v1/connectors/google", json=body)
    response_str = resp.text
    # Token should never appear in response
    assert "secret-token-xyz" not in response_str
    assert "accessToken" not in response_str


@pytest.mark.asyncio
async def test_google_connector_delete(client):
    """Test DELETE /connectors/google clears account."""
    # Set up connector
    expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
    body = {
        "account": {"sub": "test-sub", "email": "test@example.com", "name": "Test User"},
        "accessToken": "test-token",
        "expiresAt": expires_at,
        "scopes": ["openid"],
    }
    resp = await client.put("/v1/connectors/google", json=body)
    assert resp.status_code == 200

    # Delete
    resp = await client.delete("/v1/connectors/google")
    assert resp.status_code == 204

    # Verify cleared
    resp = await client.get("/v1/connectors/google")
    assert resp.json()["connected"] is False
    assert resp.json()["account"] is None


@pytest.mark.asyncio
async def test_connector_tokens_storage():
    """Test in-memory token storage."""
    # Set token
    expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
    await connector_tokens.set_google_token("test-token", expires_at)

    # Get token
    result = await connector_tokens.get_google_access_token()
    assert result is not None
    token, exp = result
    assert token == "test-token"

    # Clear token
    await connector_tokens.clear_google_token()
    result = await connector_tokens.get_google_access_token()
    assert result is None


@pytest.mark.asyncio
async def test_connector_tokens_expiry():
    """Test that expired tokens return None."""
    # Set already-expired token
    past_time = (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
    await connector_tokens.set_google_token("expired-token", past_time)

    # Get should return None (expired)
    result = await connector_tokens.get_google_access_token()
    assert result is None


@pytest.mark.asyncio
async def test_google_connector_auth_events(client):
    """Test that auth events are recorded."""
    # Sign in
    expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"
    body = {
        "account": {"sub": "test-sub", "email": "test@example.com", "name": "Test User"},
        "accessToken": "test-token",
        "expiresAt": expires_at,
        "scopes": ["openid"],
    }
    resp = await client.put("/v1/connectors/google", json=body)
    assert resp.status_code == 200

    # Get auth events
    resp = await client.get("/v1/connectors/google/events")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) > 0

    # Check for google_sign_in event
    events_list = data["items"]
    event_kinds = {e["kind"] for e in events_list}
    assert "google_sign_in" in event_kinds
