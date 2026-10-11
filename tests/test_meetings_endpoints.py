"""Comprehensive tests for meetings endpoints and contract compliance."""

import json
import uuid

import pytest


@pytest.mark.asyncio
async def test_create_meeting_minimal(client):
    """Test creating a meeting with minimal fields."""
    response = await client.post("/v1/meetings", json={})
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "New meeting"
    assert data["status"] == "scheduled"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_meeting_with_fields(client):
    """Test creating a meeting with all optional fields."""
    response = await client.post(
        "/v1/meetings",
        json={
            "title": "Team sync",
            "context": "Weekly standup",
            "scheduledStart": "2024-01-01T10:00:00Z",
            "scheduledEnd": "2024-01-01T11:00:00Z",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Team sync"
    assert data["context"] == "Weekly standup"


@pytest.mark.asyncio
async def test_get_meeting(client):
    """Test getting a single meeting."""
    # Create a meeting first
    create_resp = await client.post(
        "/v1/meetings", json={"title": "Test meeting"}
    )
    meeting_id = create_resp.json()["id"]

    # Get it
    response = await client.get(f"/v1/meetings/{meeting_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == meeting_id
    assert data["title"] == "Test meeting"


@pytest.mark.asyncio
async def test_get_nonexistent_meeting(client):
    """Test getting a meeting that doesn't exist."""
    response = await client.get(f"/v1/meetings/{uuid.uuid4()}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_meetings(client):
    """Test listing meetings."""
    # Create a couple meetings
    for i in range(3):
        await client.post("/v1/meetings", json={"title": f"Meeting {i}"})

    response = await client.get("/v1/meetings")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "nextCursor" in data
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_list_meetings_with_query(client):
    """Test listing meetings with search query."""
    # Create meetings
    await client.post("/v1/meetings", json={"title": "Team meeting"})
    await client.post("/v1/meetings", json={"title": "Planning session"})

    response = await client.get("/v1/meetings?query=team")
    assert response.status_code == 200
    data = response.json()
    assert any("Team" in m.get("title", "") for m in data["items"])


@pytest.mark.asyncio
async def test_list_meetings_with_limit(client):
    """Test listing meetings with limit."""
    response = await client.get("/v1/meetings?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) <= 5


@pytest.mark.asyncio
async def test_update_meeting(client):
    """Test updating a meeting."""
    # Create a meeting
    create_resp = await client.post(
        "/v1/meetings", json={"title": "Original"}
    )
    meeting_id = create_resp.json()["id"]

    # Update it
    response = await client.patch(
        f"/v1/meetings/{meeting_id}",
        json={"title": "Updated", "context": "New context"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated"
    assert data["context"] == "New context"


@pytest.mark.asyncio
async def test_delete_meeting(client):
    """Test deleting a meeting."""
    # Create a meeting
    create_resp = await client.post(
        "/v1/meetings", json={"title": "To delete"}
    )
    meeting_id = create_resp.json()["id"]

    # Delete it
    response = await client.delete(f"/v1/meetings/{meeting_id}")
    assert response.status_code == 204

    # Verify it's gone
    get_resp = await client.get(f"/v1/meetings/{meeting_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_start_meeting_consent_required(client):
    """Test starting a meeting without consent fails."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.post(
        f"/v1/meetings/{meeting_id}/start",
        json={"consentAcknowledged": False},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "consent_required"


@pytest.mark.asyncio
async def test_start_meeting_success(client):
    """Test successfully starting a meeting."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.post(
        f"/v1/meetings/{meeting_id}/start",
        json={"consentAcknowledged": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["meeting"]["status"] == "recording"
    assert "liveUrl" in data


@pytest.mark.asyncio
async def test_start_meeting_conflict_already_recording(client):
    """Test starting a meeting that's already recording fails."""
    # Start first meeting
    create_resp1 = await client.post("/v1/meetings", json={})
    meeting_id1 = create_resp1.json()["id"]
    await client.post(
        f"/v1/meetings/{meeting_id1}/start",
        json={"consentAcknowledged": True},
    )

    # Try to start another
    create_resp2 = await client.post("/v1/meetings", json={})
    meeting_id2 = create_resp2.json()["id"]
    response = await client.post(
        f"/v1/meetings/{meeting_id2}/start",
        json={"consentAcknowledged": True},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "conflict"


@pytest.mark.asyncio
async def test_stop_meeting(client):
    """Test stopping a meeting."""
    # Create and start
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]
    await client.post(
        f"/v1/meetings/{meeting_id}/start",
        json={"consentAcknowledged": True},
    )

    # Stop
    response = await client.post(f"/v1/meetings/{meeting_id}/stop", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"


@pytest.mark.asyncio
async def test_put_notes(client):
    """Test replacing notes for a meeting."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.put(
        f"/v1/meetings/{meeting_id}/notes",
        json={
            "blocks": [
                {"blockId": str(uuid.uuid4()), "text": "First note", "offsetMs": 0},
                {"blockId": str(uuid.uuid4()), "text": "Second note", "offsetMs": 1000},
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] == 2


@pytest.mark.asyncio
async def test_get_notes(client):
    """Test getting notes for a meeting."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    block_id = str(uuid.uuid4())
    await client.put(
        f"/v1/meetings/{meeting_id}/notes",
        json={
            "blocks": [
                {"blockId": block_id, "text": "Test note", "offsetMs": 0},
            ]
        },
    )

    response = await client.get(f"/v1/meetings/{meeting_id}/notes")
    assert response.status_code == 200
    data = response.json()
    assert len(data["blocks"]) == 1
    assert data["blocks"][0]["text"] == "Test note"


@pytest.mark.asyncio
async def test_get_transcript(client):
    """Test getting transcript for a meeting."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.get(f"/v1/meetings/{meeting_id}/transcript")
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data


@pytest.mark.asyncio
async def test_get_enhanced(client):
    """Test getting enhanced notes for a meeting."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.get(f"/v1/meetings/{meeting_id}/enhanced")
    assert response.status_code == 200
    data = response.json()
    assert "blocks" in data
    assert "style" in data


@pytest.mark.asyncio
async def test_edit_enhanced_block(client):
    """Test editing an enhanced block."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    block_id = str(uuid.uuid4())

    # First create a note
    await client.put(
        f"/v1/meetings/{meeting_id}/notes",
        json={"blocks": [{"blockId": block_id, "text": "Test", "offsetMs": 0}]},
    )

    # Edit the enhanced version
    response = await client.put(
        f"/v1/meetings/{meeting_id}/enhanced/{block_id}",
        json={"text": "Edited text"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_enhance_returns_202(client):
    """Test enhance endpoint returns 202 Accepted."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.post(
        f"/v1/meetings/{meeting_id}/enhance",
        json={"style": "standard"},
    )
    assert response.status_code == 202
    data = response.json()
    assert "jobId" in data


@pytest.mark.asyncio
async def test_get_summary(client):
    """Test getting summary for a meeting."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.get(f"/v1/meetings/{meeting_id}/summary")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("none", "pending", "ready")


@pytest.mark.asyncio
async def test_regenerate_summary_returns_202(client):
    """Test regenerate summary endpoint returns 202 Accepted."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.post(
        f"/v1/meetings/{meeting_id}/summary/regenerate",
        json={},
    )
    assert response.status_code == 202
    data = response.json()
    assert "jobId" in data


@pytest.mark.asyncio
async def test_export_meeting(client):
    """Test exporting a meeting as markdown."""
    create_resp = await client.post(
        "/v1/meetings", json={"title": "Export test"}
    )
    meeting_id = create_resp.json()["id"]

    response = await client.get(
        f"/v1/meetings/{meeting_id}/export?format=markdown"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/markdown; charset=utf-8"


@pytest.mark.asyncio
async def test_meeting_computed_fields(client):
    """Test that computed fields are included in meeting response."""
    create_resp = await client.post("/v1/meetings", json={})
    meeting_id = create_resp.json()["id"]

    response = await client.get(f"/v1/meetings/{meeting_id}")
    data = response.json()

    # Check for computed fields
    assert "attachmentCount" in data
    assert "hasTranscript" in data
    assert "hasSummary" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
