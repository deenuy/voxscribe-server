"""Wake word state API.

GET  /v1/wake/state     → {state, model, phrase}   states: armed | listening | off | error
POST /v1/wake/arm       → arm (no-op if already armed/listening)
POST /v1/wake/disarm    → set state to off; mic stream closes within ~32 ms
POST /v1/wake/trigger   → DEBUG ONLY (--no-auth). Bypass OWW, start a session immediately.
                          Returns {meeting_id}. Use for automated roomplay tests.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.local import events
from app.local.wake_loop import WakeState, get_state, set_state

router = APIRouter(prefix="/v1/wake", tags=["wake"])


@router.get("/state")
async def wake_state(_: Request):
    s = get_state()
    v = s["state"]
    return {
        "state":  v.value if isinstance(v, WakeState) else str(v),
        "model":  s["model"],
        "phrase": s["phrase"],
    }


@router.post("/arm", status_code=200)
async def arm_wake(_: Request):
    current = get_state()["state"]
    if current in (WakeState.ARMED, WakeState.LISTENING):
        return {"state": current.value if isinstance(current, WakeState) else str(current),
                "changed": False}
    set_state(WakeState.ARMED)
    events.publish("wake_state", get_state(), broadcast=True)
    return {"state": WakeState.ARMED.value, "changed": True}


@router.post("/disarm", status_code=200)
async def disarm_wake(_: Request):
    current = get_state()["state"]
    if current == WakeState.OFF:
        return {"state": WakeState.OFF.value, "changed": False}
    set_state(WakeState.OFF)
    events.publish("wake_state", get_state(), broadcast=True)
    return {"state": WakeState.OFF.value, "changed": True}


@router.post("/trigger", status_code=202)
async def trigger_wake_session(request: Request):
    """DEBUG ONLY: bypass OWW and start a wake session immediately.

    Only reachable when the server was started with --no-auth.
    Use for automated roomplay tests where TTS voice doesn't trigger OWW.
    Returns {meeting_id} immediately; session runs as a background task.
    """
    from app.local.wake_loop import _run_session
    import numpy as np

    app = request.app
    auth_disabled = getattr(app.state, "auth_disabled", False)
    if not auth_disabled:
        raise HTTPException(status_code=403, detail="trigger only available with --no-auth")

    current = get_state()["state"]
    if current == WakeState.LISTENING:
        raise HTTPException(status_code=409, detail="session already in progress")

    audio_engine = getattr(app.state, "audio_engine", None)
    if audio_engine is None:
        raise HTTPException(status_code=503, detail="audio engine not running")

    # Subscribe to the AudioEngine capture stream for this session
    audio_q = audio_engine.subscribe()

    async def _session_task():
        try:
            await _run_session(app, audio_q, confidence=1.0, np=np)
        finally:
            # Mirror _mic_loop's finally: reset state so /trigger is not stuck
            # in LISTENING forever (trigger path never goes through _mic_loop).
            set_state(WakeState.ARMED)
            audio_engine.unsubscribe(audio_q)

    task = asyncio.create_task(_session_task(), name="trigger-session")

    # Wait briefly for the session to create its meeting_id (it appears in DB within ~200ms)
    await asyncio.sleep(0.3)

    # Return state so caller knows it started
    return {"triggered": True, "state": get_state()["state"].value}
