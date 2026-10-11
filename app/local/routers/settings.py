"""Settings endpoints: get and update app settings."""

from fastapi import APIRouter, Request

from app.local import store_misc
from app.local.errors import InvalidRequestError
from app.local.models import Settings

router = APIRouter(prefix="/v1", tags=["settings"])


@router.get("/settings")
async def get_settings(request: Request) -> dict:
    """Get all settings."""
    store = request.app.state.store

    async with store.pool.connection() as conn:
        settings = await store_misc.get_settings(conn)

    return settings.model_dump()


@router.put("/settings")
async def put_settings(request: Request, body: dict) -> dict:
    """Replace all settings.

    Validates enum values and ranges, rejects unknown keys.
    """
    store = request.app.state.store

    # Validate known fields only
    known_fields = Settings.model_fields
    for key in body:
        if key not in known_fields:
            raise InvalidRequestError(f"unknown setting: {key}")

    # Validate enums
    if "privacyMode" in body and body["privacyMode"] not in ("offline", "connectors"):
        raise InvalidRequestError("privacyMode must be 'offline' or 'connectors'")

    if "enhanceStyle" in body and body["enhanceStyle"] not in ("standard", "concise"):
        raise InvalidRequestError("enhanceStyle must be 'standard' or 'concise'")

    # Validate ranges
    if "autoStopMinutes" in body and body["autoStopMinutes"] is not None:
        if not isinstance(body["autoStopMinutes"], int) or body["autoStopMinutes"] < 1:
            raise InvalidRequestError("autoStopMinutes must be a positive integer")

    # Create Settings object with defaults for missing fields
    try:
        settings = Settings(**body)
    except Exception as e:
        raise InvalidRequestError(f"invalid settings: {e}") from e

    # Store
    async with store.pool.connection() as conn:
        await store_misc.put_settings(conn, settings)

    return settings.model_dump()
