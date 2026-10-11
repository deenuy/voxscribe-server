"""Tests for settings endpoints."""

import pytest


@pytest.mark.asyncio
async def test_settings_get(client):
    """Test GET /settings returns current settings."""
    resp = await client.get("/v1/settings")
    assert resp.status_code == 200
    data = resp.json()

    # Check expected fields exist
    assert "openAtLogin" in data
    assert "reminders" in data
    assert "enhanceStyle" in data
    assert "privacyMode" in data


@pytest.mark.asyncio
async def test_settings_put(client):
    """Test PUT /settings updates settings."""
    new_settings = {
        "openAtLogin": True,
        "reminders": True,
        "privacyMode": "connectors",
        "enhanceStyle": "concise",
    }

    resp = await client.put("/v1/settings", json=new_settings)
    assert resp.status_code == 200
    data = resp.json()
    assert data["openAtLogin"] is True
    assert data["privacyMode"] == "connectors"

    # Verify persisted
    resp2 = await client.get("/v1/settings")
    assert resp2.json()["openAtLogin"] is True


@pytest.mark.asyncio
async def test_settings_invalid_enum(client):
    """Test that invalid enum values are rejected."""
    resp = await client.put(
        "/v1/settings",
        json={"privacyMode": "invalid"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_settings_unknown_key_rejected(client):
    """Test that unknown settings keys are rejected."""
    resp = await client.put(
        "/v1/settings",
        json={"unknownKey": "value"},
    )
    assert resp.status_code == 400
    assert "unknown" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_settings_validation_ranges(client):
    """Test that ranges are validated."""
    # Invalid autoStopMinutes (must be positive)
    resp = await client.put(
        "/v1/settings",
        json={"autoStopMinutes": -1},
    )
    assert resp.status_code == 400
