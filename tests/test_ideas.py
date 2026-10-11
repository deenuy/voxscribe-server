"""Tests for ideas endpoints."""


import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_vocabulary(client: AsyncClient):
    """Test vocabulary endpoint."""
    response = await client.get("/v1/ideas/vocabulary")
    assert response.status_code == 200

    data = response.json()
    assert "types" in data
    assert "statuses" in data
    assert "rubric" in data
    assert len(data["types"]) > 0
    assert len(data["statuses"]) == 5
    assert len(data["rubric"]) == 10

    # Check status thresholds
    for status in data["statuses"]:
        assert "key" in status
        assert "label" in status
        assert "min" in status
        assert "max" in status


@pytest.mark.asyncio
async def test_create_idea(client: AsyncClient):
    """Test creating an idea."""
    response = await client.post(
        "/v1/ideas",
        json={
            "title": "Test Idea",
            "summary": "A test idea",
            "type": "new_feature",
        },
    )
    assert response.status_code == 201

    data = response.json()
    assert data["title"] == "Test Idea"
    assert data["summary"] == "A test idea"
    assert data["type"] == "new_feature"
    assert data["status"] == "raw"
    assert data["score"] == 0


@pytest.mark.asyncio
async def test_list_ideas(client: AsyncClient):
    """Test listing ideas."""
    # Create a few ideas
    for i in range(3):
        await client.post(
            "/v1/ideas",
            json={
                "title": f"Idea {i}",
                "summary": f"Summary {i}",
            },
        )

    response = await client.get("/v1/ideas")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert "nextCursor" in data
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_get_idea(client: AsyncClient):
    """Test getting a single idea."""
    # Create an idea
    create_response = await client.post(
        "/v1/ideas",
        json={"title": "Test", "summary": "Test summary"},
    )
    idea_id = create_response.json()["id"]

    response = await client.get(f"/v1/ideas/{idea_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == idea_id
    assert data["title"] == "Test"


@pytest.mark.asyncio
async def test_update_idea(client: AsyncClient):
    """Test updating an idea."""
    # Create an idea
    create_response = await client.post(
        "/v1/ideas",
        json={"title": "Original", "summary": "Original summary"},
    )
    idea_id = create_response.json()["id"]

    # Update it
    response = await client.patch(
        f"/v1/ideas/{idea_id}",
        json={"title": "Updated", "summary": "Updated summary"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["title"] == "Updated"
    assert data["summary"] == "Updated summary"
    assert data["edited"] is True


@pytest.mark.asyncio
async def test_delete_idea(client: AsyncClient):
    """Test deleting an idea."""
    # Create an idea
    create_response = await client.post(
        "/v1/ideas",
        json={"title": "To Delete"},
    )
    idea_id = create_response.json()["id"]

    # Delete it
    response = await client.delete(f"/v1/ideas/{idea_id}")
    assert response.status_code == 204

    # Verify it's gone
    get_response = await client.get(f"/v1/ideas/{idea_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_create_project(client: AsyncClient):
    """Test creating a project."""
    response = await client.post(
        "/v1/projects",
        json={
            "name": "Test Project",
            "description": "A test project",
        },
    )
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Test Project"
    assert data["description"] == "A test project"


@pytest.mark.asyncio
async def test_list_projects(client: AsyncClient):
    """Test listing projects."""
    # Create a few projects
    for i in range(2):
        await client.post(
            "/v1/projects",
            json={"name": f"Project {i}"},
        )

    response = await client.get("/v1/projects")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert len(data["items"]) >= 2


@pytest.mark.asyncio
async def test_create_feature(client: AsyncClient):
    """Test creating a feature."""
    # Create a project
    proj_response = await client.post(
        "/v1/projects",
        json={"name": "Test Project"},
    )
    project_id = proj_response.json()["id"]

    # Create a feature
    response = await client.post(
        f"/v1/projects/{project_id}/features",
        json={"name": "Feature 1"},
    )
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Feature 1"
    assert data["projectId"] == project_id


@pytest.mark.asyncio
async def test_list_features(client: AsyncClient):
    """Test listing features."""
    # Create a project
    proj_response = await client.post(
        "/v1/projects",
        json={"name": "Test Project"},
    )
    project_id = proj_response.json()["id"]

    # Create features
    for i in range(2):
        await client.post(
            f"/v1/projects/{project_id}/features",
            json={"name": f"Feature {i}"},
        )

    response = await client.get(f"/v1/projects/{project_id}/features")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_get_prd(client: AsyncClient):
    """Test getting PRD status."""
    # Create an idea
    create_response = await client.post(
        "/v1/ideas",
        json={"title": "Test", "summary": "Test"},
    )
    idea_id = create_response.json()["id"]

    response = await client.get(f"/v1/ideas/{idea_id}/prd")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "none"
    assert data["markdown"] is None


@pytest.mark.asyncio
async def test_get_activity(client: AsyncClient):
    """Test getting activity log."""
    # Create an idea
    create_response = await client.post(
        "/v1/ideas",
        json={"title": "Test"},
    )
    idea_id = create_response.json()["id"]

    response = await client.get(f"/v1/ideas/{idea_id}/activity")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    # Should have at least the creation event
    assert len(data["items"]) > 0
