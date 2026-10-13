"""POST /v1/wake — external wake-detected and transcript injection.

Useful for testing without a mic, and for a future MacBook-side client
that detects a wake phrase locally and notifies the engine.

Auth: same Bearer token as every other engine endpoint (enforced by the
app-level middleware). No additional checks here.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.local import events

router = APIRouter(prefix="/v1/wake", tags=["wake"])


class WakeDetectedRequest(BaseModel):
    phrase: str = "hey_vox"


class WakeTranscriptRequest(BaseModel):
    text: str


@router.post("/detected", status_code=204)
async def wake_detected(body: WakeDetectedRequest):
    """Publish a wake_detected event to all SSE subscribers."""
    events.publish("wake_detected", {"phrase": body.phrase}, broadcast=True)


@router.post("/transcript", status_code=204)
async def wake_transcript(body: WakeTranscriptRequest):
    """Publish a wake_transcript event to all SSE subscribers."""
    events.publish("wake_transcript", {"text": body.text}, broadcast=True)


@router.get("/status")
async def wake_status(request: Request):
    """Report whether the wake loop is running in this engine process."""
    return {"wake_enabled": bool(getattr(request.app.state, "wake_enabled", False))}
