"""Wake word state API, read from the voice host (app.state.voice_host).

GET  /v1/wake/state     → {state, model, phrase, trigger, meeting_id, bound_to_caller}
                          states: armed | listening | off | error
                          meeting_id is the running session's, and only the tenant the device is
                          bound to gets it. Any other verified caller still learns whether the
                          device is in use (state), with meeting_id null and bound_to_caller
                          false: the device is listening, but not for them
GET  /v1/wake/profile   → {mode, delegate, quiet_hours_enabled}  current product profile summary
POST /v1/wake/arm       → arm (no-op if already armed/listening). With tenant isolation on, the
                          verified person who arms is who the device listens for from then on
POST /v1/wake/disarm    → set state to off; the armed loop closes its subscription
                          With isolation on, arm, disarm and trigger need a role that may control
                          the device (identity.DEVICE_CONTROL_ROLES: owner, admin, member, user)
                          and a token not tied to another device: else 403 role_not_permitted or
                          device_mismatch. With isolation off nothing is checked, as before
POST /v1/wake/trigger   → DEBUG ONLY (--debug-trigger). Bypass the wake gate, start a session
                          immediately. Returns {meeting_id}. Use for automated roomplay tests.
POST /v1/wake/power     → sleep | wake | lock | unlock from the app, for the capture watchdog

Without --wake there is no voice host: the state is off and arm changes nothing.

Who the device host's sessions belong to (PL05): nobody makes a request when a person says the
wake phrase, so the tenant is bound ahead of time by a request that does carry a verified
person. Arming is that request (it is also where consent to listen is given). With isolation on
the host is off until someone arms it, and another person cannot take the device while a
session runs (409 voice_session_in_progress). A host that is asked to arm or to start a session
with no tenant bound refuses with voice_tenant_unbound.
"""

from dataclasses import replace
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.local.errors import EngineError
from app.local.identity import require_device_control
from app.local.voice.host import (
    SessionInProgressError,
    SessionTenantRequiredError,
    VoiceConfig,
    WakeState,
)

router = APIRouter(prefix="/v1/wake", tags=["wake"])

_OFF_CONFIG = VoiceConfig()


def _host(request: Request):
    return getattr(request.app.state, "voice_host", None)


def _value(v) -> str:
    return v.value if isinstance(v, WakeState) else str(v)


def _device_id(request: Request, host) -> str:
    if host is not None:
        return host.binding.device
    from app.local.voice.adapters.scope import LOCAL_DEVICE

    return getattr(getattr(request.app.state, "settings", None), "device_id", "") or LOCAL_DEVICE


def _require_device_control(request: Request, host) -> None:
    """Binding or unbinding the audio device needs a role that may do it (identity.py). Without
    tenant isolation there is no person on the request and nothing is checked."""
    require_device_control(getattr(request.state, "identity", None), _device_id(request, host))


def _bind_caller(request: Request, host) -> None:
    """The verified person making this request is who the device listens for, if their role
    allows it. Without tenant isolation there is no person on the request and the binding is
    left as it is."""
    _require_device_control(request, host)
    tenant = getattr(request.state, "tenant", None)
    if tenant:
        identity = getattr(request.state, "identity", None)
        host.bind(replace(host.binding, tenant=tenant,
                          organization=getattr(identity, "organization", None),
                          role=getattr(identity, "role", None)))


def _refused(exc: SessionTenantRequiredError | SessionInProgressError) -> EngineError:
    return EngineError(exc.code, str(exc), 409)


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
                "meeting_id": None, "bound_to_caller": False}
    s = host.snapshot()
    # A session is its tenant's: the meeting id goes to the tenant the device is bound to and to
    # nobody else. With isolation off both sides are None, the one shared scope.
    mine = getattr(request.state, "tenant", None) == host.binding.tenant
    return {
        "state":      _value(s["state"]),
        "model":      s["model"],
        "phrase":     s["phrase"],
        "trigger":    s.get("trigger"),
        "meeting_id": s.get("meeting_id") if mine else None,
        "bound_to_caller": mine,
    }


@router.post("/arm", status_code=200)
async def arm_wake(request: Request):
    host = _host(request)
    _require_device_control(request, host)
    if host is None:
        return {"state": WakeState.OFF.value, "changed": False}
    try:
        _bind_caller(request, host)
        changed = host.arm()      # the host publishes wake_state on a change
    except (SessionTenantRequiredError, SessionInProgressError) as exc:
        raise _refused(exc) from None
    return {"state": _value(host.state), "changed": changed}


@router.post("/disarm", status_code=200)
async def disarm_wake(request: Request):
    host = _host(request)
    _require_device_control(request, host)
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

    try:
        _bind_caller(request, host)
        mid = await host.trigger()
    except (SessionTenantRequiredError, SessionInProgressError) as exc:
        raise _refused(exc) from None
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
