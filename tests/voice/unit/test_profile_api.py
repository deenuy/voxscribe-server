"""Unit tests for the assistant profile API (T17).

Fakes only: no database, no audio engine, no real TTS or LLM.
Tests cover:
  - PATCH with a new wake_phrase is refused: the wake phrase is fixed (PL35)
  - Validation errors: speed out of range, empty name
  - GET returns current profile
  - POST /reset reverts to startup
  - voice and speed are read per synthesis call
  - system prompt is read per reply call
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal app fixture with the assistant router and a fake profile store
# ---------------------------------------------------------------------------

def _make_app():
    """Return a minimal FastAPI app with the assistant router and profile_store."""
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from voxscribe.profile.schema import Profile
    from app.local.errors import EngineError
    from app.local.voice.profile_store import AssistantProfileStore
    from app.local.routers.assistant import router

    app = FastAPI()

    @app.exception_handler(EngineError)
    async def engine_error_handler(request: Request, exc: EngineError):
        return JSONResponse(
            status_code=exc.status,
            content={"error": exc.code, "detail": exc.detail, **exc.extra},
        )

    app.include_router(router)
    app.state.profile_store = AssistantProfileStore(Profile())
    return app


@pytest.fixture()
def client():
    app = _make_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def app_instance():
    return _make_app()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_patch_wake_phrase_is_refused_because_it_is_fixed(client):
    """PL35: the wake phrase is fixed. A new one is refused by name; GET still has the fixed
    one and says so."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"wake_phrase": "hey aria"}},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "assistant_field_fixed"
    assert resp.json()["field"] == "wake_phrase"

    get_resp = client.get("/v1/assistant/profile")
    assert get_resp.status_code == 200
    assert get_resp.json()["assistant"]["wake_phrase"] == "hey vox"
    assert get_resp.json()["fields"]["wake_phrase"]["state"] == "fixed"


def test_patch_wake_phrase_single_word_is_refused_as_fixed(client):
    """A one-word phrase was invalid_request (D18); every other phrase is now fixed."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"wake_phrase": "aria"}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body.get("error") == "assistant_field_fixed"
    assert "wake_phrase" in body.get("detail", "")


def test_patch_speed_out_of_range(client):
    """Speed 3.0 (above maximum) must return 400 invalid_request (D18)."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"speed": 3.0}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body.get("error") == "invalid_request"
    assert "speed" in body.get("detail", "")


def test_patch_name_empty(client):
    """Empty name (after stripping) must return 400 invalid_request (D18)."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"name": "   "}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body.get("error") == "invalid_request"
    assert "name" in body.get("detail", "")


def test_reset_reverts(client):
    """PATCH then POST /reset; GET returns startup values."""
    assert client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"name": "aria", "speed": 1.5}},
    ).status_code == 200
    reset_resp = client.post("/v1/assistant/profile/reset")
    assert reset_resp.status_code == 200
    data = reset_resp.json()
    # Default startup values
    assert data["assistant"]["name"] == "vox"
    assert data["assistant"]["wake_phrase"] == "hey vox"
    assert data["assistant"]["speed"] == 1.0

    get_resp = client.get("/v1/assistant/profile")
    assert get_resp.json()["assistant"]["wake_phrase"] == "hey vox"


def test_get_returns_defaults(client):
    """GET before any PATCH returns the default profile."""
    resp = client.get("/v1/assistant/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["assistant"]["name"] == "vox"
    assert data["assistant"]["wake_phrase"] == "hey vox"
    assert data["assistant"]["speed"] == 1.0
    assert data["assistant"]["voice_id"] == ""


def test_voice_speed_read_per_turn(app_instance):
    """After patching speed, the profile_store returns the new speed immediately."""
    from app.local.voice.profile_store import AssistantProfileStore
    from voxscribe.profile.schema import Profile

    store: AssistantProfileStore = app_instance.state.profile_store

    # Initial speed
    assert store.get().assistant.speed == 1.0

    store.patch({"assistant": {"speed": 1.8}})

    # Next call to get() returns the patched speed
    assert store.get().assistant.speed == 1.8


def test_system_prompt_read_per_turn(app_instance):
    """After patching system_prompt, the store returns the new value."""
    store = app_instance.state.profile_store

    assert store.get().assistant.system_prompt == ""

    store.patch({"assistant": {"system_prompt": "You are a pirate assistant."}})

    assert store.get().assistant.system_prompt == "You are a pirate assistant."
    # Reset should clear it
    store.reset()
    assert store.get().assistant.system_prompt == ""


def test_patch_speed_at_minimum_boundary(client):
    """Speed at the minimum boundary (0.5) must be accepted."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"speed": 0.5}},
    )
    assert resp.status_code == 200
    assert resp.json()["assistant"]["speed"] == 0.5


def test_patch_speed_at_maximum_boundary(client):
    """Speed at the maximum boundary (2.0) must be accepted."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"speed": 2.0}},
    )
    assert resp.status_code == 200
    assert resp.json()["assistant"]["speed"] == 2.0


def test_patch_speed_below_minimum(client):
    """Speed below 0.5 must return 400 invalid_request (D18)."""
    resp = client.patch(
        "/v1/assistant/profile",
        json={"assistant": {"speed": 0.1}},
    )
    assert resp.status_code == 400
    assert resp.json().get("error") == "invalid_request"


def test_patch_preserves_unmentioned_fields(client):
    """PATCH with only speed preserves name and persona_label."""
    client.patch("/v1/assistant/profile",
                 json={"assistant": {"name": "aria", "persona_label": "companion"}})
    client.patch("/v1/assistant/profile", json={"assistant": {"speed": 1.3}})
    data = client.get("/v1/assistant/profile").json()
    assert data["assistant"]["name"] == "aria"
    assert data["assistant"]["persona_label"] == "companion"
    assert data["assistant"]["speed"] == 1.3
