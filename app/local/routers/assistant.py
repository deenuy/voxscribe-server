"""Assistant profile API.

GET  /v1/assistant/profile       -- return the current profile (tenant-aware)
PATCH /v1/assistant/profile      -- merge partial update; returns updated profile
POST /v1/assistant/profile/reset -- revert to startup profile
GET  /v1/assistant/voices        -- list available Kokoro voice IDs

The profile is persisted in the assistant_profiles table (migration 022), one row per
tenant schema. app.state.profile_store holds the in-memory cache.

With a database pool in app.state.store the router reads and writes through the DB via the
scope adapter helpers (voice/adapters/scope.py). Without a pool (unit tests) it falls back
to the in-memory store. The 422 status codes on validation errors are left as-is (a later
task aligns all validation errors to 400).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("app.local.routers.assistant")

router = APIRouter(prefix="/v1/assistant", tags=["assistant"])

# Standard Kokoro voice IDs shipped with the kokoro-onnx model bundle.
# Format: <language_gender>_<name>  (a=American, b=British; f=female, m=male)
KNOWN_KOKORO_VOICE_IDS: tuple[str, ...] = (
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_fable",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)


def _profile_to_dict(profile) -> dict:
    """Serialize a profile view to a plain dict for JSON responses."""
    cfg = profile.assistant
    return {
        "assistant": {
            "name": cfg.name,
            "wake_phrase": cfg.wake_phrase,
            "end_phrases": list(cfg.end_phrases),
            "voice_id": cfg.voice_id,
            "speed": cfg.speed,
            "system_prompt": cfg.system_prompt,
            "persona_label": cfg.persona_label,
        }
    }


def _get_store(request: Request):
    store = getattr(request.app.state, "profile_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="profile store not initialised")
    return store


def _has_pool(request: Request) -> bool:
    """True when the app has a live database pool (not running in a unit-test minimal app)."""
    store_obj = getattr(request.app.state, "store", None)
    return getattr(getattr(store_obj, "pool", None), "connection", None) is not None


def _current_tenant(request: Request):
    """The tenant key for the caller, or None with isolation off."""
    from app.local import tenancy
    return tenancy.current_tenant()


@router.get("/profile")
async def get_profile(request: Request):
    """Return the current assistant profile."""
    _get_store(request)   # validate store is present

    if _has_pool(request):
        from app.local.voice.adapters.scope import load_profile_for_request
        tenant = _current_tenant(request)
        view = await load_profile_for_request(request.app, tenant)
    else:
        store = _get_store(request)
        view = store.get()
    return JSONResponse(_profile_to_dict(view))


@router.patch("/profile")
async def patch_profile(request: Request):
    """Merge a partial update into the current profile.

    Accepts a JSON body with the same shape as the profile (any subset of fields).
    Returns the full updated profile.
    Returns 422 when any field fails validation.

    Example::
        PATCH /v1/assistant/profile
        {"assistant": {"name": "aria", "wake_phrase": "hey aria", "speed": 1.2}}
    """
    from app.local.voice.profile_store import ProfileValidationError

    _get_store(request)   # validate store is present

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="request body must be valid JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")

    try:
        if _has_pool(request):
            from app.local.voice.adapters.scope import save_profile_for_request
            tenant = _current_tenant(request)
            updated = await save_profile_for_request(request.app, body, tenant)
        else:
            store = _get_store(request)
            updated = store.patch(body)
    except ProfileValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "detail": str(exc)},
        )
    except Exception:
        log.exception("patch_profile: unexpected error")
        raise HTTPException(status_code=500, detail="failed to update profile")

    return JSONResponse(_profile_to_dict(updated))


@router.post("/profile/reset")
async def reset_profile(request: Request):
    """Revert the assistant profile to the startup defaults.

    Returns the reverted profile.
    """
    _get_store(request)   # validate store is present

    if _has_pool(request):
        from app.local.voice.adapters.scope import reset_profile_for_request
        tenant = _current_tenant(request)
        reverted = await reset_profile_for_request(request.app, tenant)
    else:
        store = _get_store(request)
        reverted = store.reset()
    return JSONResponse(_profile_to_dict(reverted))


@router.get("/voices")
async def list_voices():
    """Return the list of available Kokoro voice IDs.

    The list is static: it reflects the voices bundled with the kokoro-onnx model.
    The first entry is the default voice used when no voice_id is set in the profile.
    """
    return JSONResponse({"voices": list(KNOWN_KOKORO_VOICE_IDS)})
