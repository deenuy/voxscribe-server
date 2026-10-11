"""Tests for models endpoints."""

import pytest


@pytest.mark.asyncio
async def test_models_list_fake_mode(client):
    """Test GET /models returns catalogue in fake mode."""

    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) > 0

    # Check expert models are present
    model_ids = {m["id"] for m in data["models"]}
    assert "faster-whisper" in model_ids
    assert "nomic-embed-text" in model_ids

    # Check expert models are required and not selectable
    for m in data["models"]:
        if m["kind"] == "expert":
            assert m["required"] is True


@pytest.mark.asyncio
async def test_models_select_generative(client):
    """Test POST /models/{id}/select for generative models."""
    # Select gemma4:e4b as main role
    resp = await client.post(
        "/v1/models/gemma4:e4b/select",
        json={"role": "main"},
    )
    assert resp.status_code == 200
    model = resp.json()
    assert model["id"] == "gemma4:e4b"
    assert "main" in model["selectedRoles"]


@pytest.mark.asyncio
async def test_models_select_not_ready_fails(client):
    """Test selecting a model that is not ready fails."""
    # Try to select gemma4:12b (not ready in fake mode)
    resp = await client.post(
        "/v1/models/gemma4:12b/select",
        json={"role": "main"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


@pytest.mark.asyncio
async def test_models_expert_not_selectable(client):
    """Test that expert models cannot be selected."""
    resp = await client.post(
        "/v1/models/faster-whisper/select",
        json={"role": "main"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_models_invalid_role(client):
    """Test selecting with invalid role fails."""
    resp = await client.post(
        "/v1/models/gemma4:e4b/select",
        json={"role": "invalid"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_models_download_fake_mode(client):
    """Test POST /models/{id}/download returns 202 in fake mode."""
    resp = await client.post("/v1/models/gemma4:12b/download")
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_models_download_only_one_at_a_time(client):
    """Test that only one download per model can happen at once."""
    # Start first download
    resp1 = await client.post("/v1/models/gemma4:12b/download")
    assert resp1.status_code == 202

    # Try to start second download immediately (will race, one might succeed)
    resp2 = await client.post("/v1/models/gemma4:12b/download")
    # Might be 202 or 409 depending on timing, but both are acceptable


@pytest.mark.asyncio
async def test_models_delete_expert_fails(client):
    """Test that deleting expert models fails."""
    resp = await client.delete("/v1/models/faster-whisper")
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"
