"""Wake word state API, read from the voice host (app.state.voice_host).

GET  /v1/wake/state     → {state, model, phrase, trigger, meeting_id}
                          states: armed | listening | off | error
GET  /v1/wake/profile   → {mode, delegate, quiet_hours_enabled}  current product profile summary
POST /v1/wake/arm       → arm (no-op if already armed/listening)
POST /v1/wake/disarm    → set state to off; the armed loop closes its subscription
POST /v1/wake/trigger   → DEBUG ONLY (--debug-trigger). Bypass the wake gate, start a session
                          immediately. Returns {meeting_id}. Use for automated roomplay tests.
POST /v1/wake/power     → sleep | wake | lock | unlock from the app, for the capture watchdog

Without --wake there is no voice host: the state is off and arm changes nothing.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.local.voice.host import VoiceConfig, WakeState

router = APIRouter(prefix="/v1/wake", tags=["wake"])

_OFF_CONFIG = VoiceConfig()


def _host(request: Request):
    return getattr(request.app.state, "voice_host", None)


def _value(v) -> str:
    return v.value if isinstance(v, WakeState) else str(v)


@router.get("/profile")
async def wake_profile(request: Request):
    """Return a summary of the current product profile (mode and key settings)."""
    profile = getattr(request.app.state, "profile", None)
    if profile is None:
        return {"mode": "personal", "delegate": "ollama", "quiet_hours_enabled": False}
    return {
        "mode": profile.mode,
        "delegate": profile.reply.delegate,
        "quiet_hours_enabled": profile.quiet_hours_enabled,
    }


@router.get("/state")
async def wake_state(request: Request):
    host = _host(request)
    if host is None:
        return {"state": WakeState.OFF.value, "model": _OFF_CONFIG.detector,
                "phrase": _OFF_CONFIG.wake_phrase, "trigger": _OFF_CONFIG.trigger_phrase,
                "meeting_id": None}
    s = host.snapshot()
    return {
        "state":      _value(s["state"]),
        "model":      s["model"],
        "phrase":     s["phrase"],
        "trigger":    s.get("trigger"),
        "meeting_id": s.get("meeting_id"),
    }


@router.post("/arm", status_code=200)
async def arm_wake(request: Request):
    host = _host(request)
    if host is None:
        return {"state": WakeState.OFF.value, "changed": False}
    changed = host.arm()      # the host publishes wake_state on a change
    return {"state": _value(host.state), "changed": changed}


@router.post("/disarm", status_code=200)
async def disarm_wake(request: Request):
    host = _host(request)
    if host is None:
        return {"state": WakeState.OFF.value, "changed": False}
    changed = host.disarm()
    return {"state": WakeState.OFF.value, "changed": changed}


@router.post("/trigger", status_code=202)
async def trigger_wake_session(request: Request):
    """DEBUG ONLY: bypass the wake gate and start a wake session immediately.

    Only reachable when the server was started with --debug-trigger. --no-auth
    alone does not enable it: the normal launch runs with --no-auth, and any
    local process could otherwise start a fake wake.
    Use for automated roomplay tests where a synthetic voice does not wake the gate.
    Returns {meeting_id} once the session has one (up to about 2 s).
    """
    app = request.app
    if not getattr(app.state, "debug_trigger", False):
        raise HTTPException(status_code=403, detail="trigger only available with --debug-trigger")

    host = _host(request)
    if host is not None and host.state == WakeState.LISTENING:
        raise HTTPException(status_code=409, detail="session already in progress")
    if getattr(app.state, "audio_engine", None) is None:
        raise HTTPException(status_code=503, detail="audio engine not running")
    if host is None or not host.ready:
        raise HTTPException(status_code=503, detail="voice host not ready")

    mid = await host.trigger()
    return {"triggered": True, "state": _value(host.state), "meeting_id": mid}


class PowerEvent(BaseModel):
    event: Literal["sleep", "wake", "lock", "unlock"]


@router.post("/power", status_code=202)
async def power_event(body: PowerEvent, request: Request):
    """Power and session events forwarded by the app (Electron powerMonitor).

    sleep pauses the capture watchdog; wake (or an unlock that follows a missed wake) reopens the
    microphone stream after a short delay so the voice host is armed on a live stream. Lock alone
    changes nothing. Returns handled=False when the engine runs without an audio engine.
    """
    watchdog = getattr(request.app.state, "capture_watchdog", None)
    if watchdog is None:
        return {"event": body.event, "handled": False}
    watchdog.on_power(body.event)
    return {"event": body.event, "handled": True, "sleeping": watchdog.sleeping}
