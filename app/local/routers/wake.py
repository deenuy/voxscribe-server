"""Wake word state API.

GET  /v1/wake/state   → {state, model, phrase}   states: armed | listening | off | error
POST /v1/wake/arm     → arm (no-op if already armed/listening)
POST /v1/wake/disarm  → set state to off; mic stream closes within ~32 ms
"""

from fastapi import APIRouter, Request

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
