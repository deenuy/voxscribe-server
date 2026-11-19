"""POST /v1/wake/stop and /v1/wake/mute need what arming needs, and only the tenant the device is bound to may use
them (PL36 C). The real engine with identity on; the voice host is built on fake ports and never started."""

from __future__ import annotations

import asyncio

from app.local.voice.host import SessionContext, WakeState
from tests import test_identity_access
from tests.test_identity_access import _host_for

make_engine = test_identity_access.make_engine      # the engine with identity on (fixture)


async def test_only_the_bound_tenant_with_a_device_role_may_end_or_mute(make_engine):
    engine = await make_engine()
    host = _host_for(engine.app)
    c, h = engine.client, engine.headers
    assert (await c.post("/v1/wake/arm", headers=h("ann", role="owner"))).status_code == 200

    # No session running: nothing to end or mute, said plainly.
    assert (await c.post("/v1/wake/stop", headers=h("ann", role="owner"))).json() == {"stopped": False,
                                                                                        "state": "armed"}
    res = await c.post("/v1/wake/mute", json={"muted": True}, headers=h("ann", role="owner"))
    assert res.status_code == 200 and res.json() == {"muted": False, "state": "armed"}

    # A session runs for ann.
    session = SessionContext(binding=host.binding, session_id="s1")
    host._open = session
    host.set_state(WakeState.LISTENING)
    res = await c.post("/v1/wake/mute", json={"muted": True}, headers=h("ann", role="owner"))
    assert res.json() == {"muted": True, "state": "listening"}
    assert (await c.get("/v1/wake/state", headers=h("ann", role="owner"))).json()["muted"] is True

    # Another tenant, even with a device role, is refused and learns nothing: the same answer with or without a
    # session, and the state route tells them nothing about mute.
    for path, body in (("/v1/wake/stop", None), ("/v1/wake/mute", {"muted": False})):
        res = await c.post(path, json=body, headers=h("ben", role="admin"))
        assert res.status_code == 409 and res.json()["error"] == "voice_session_in_progress", path
        res = await c.post(path, json=body, headers=h("vic", role="viewer"))
        assert res.status_code == 403 and res.json()["error"] == "role_not_permitted", path
    other = (await c.get("/v1/wake/state", headers=h("ben", role="admin"))).json()
    assert other["muted"] is False and other["meeting_id"] is None and other["bound_to_caller"] is False
    assert session.muted and not session.stop_requested

    # ann ends it: the session is asked to stop and playback stops at once.
    async def session_closes():
        await asyncio.sleep(0.05)
        host._open = None
        host.set_state(WakeState.ARMED)

    closing = asyncio.ensure_future(session_closes())
    stops = host.ports.audio_out.stops
    res = await c.post("/v1/wake/stop", headers=h("ann", role="owner"))
    await closing
    assert res.status_code == 200 and res.json() == {"stopped": True, "state": "armed"}
    assert session.stop_requested and not session.muted and host.ports.audio_out.stops == stops + 1

    host._open = SessionContext(binding=host.binding)
    res = await c.post("/v1/wake/stop", headers=h("ben", role="admin"))
    assert res.status_code == 409 and not host._open.stop_requested
    host._open = None


async def test_a_mute_body_must_say_true_or_false(make_engine):
    engine = await make_engine()
    _host_for(engine.app)
    h = engine.headers("ann", role="owner")
    await engine.client.post("/v1/wake/arm", headers=h)
    for body in ({}, {"muted": "maybe"}):
        res = await engine.client.post("/v1/wake/mute", json=body, headers=h)
        assert res.status_code == 400 and res.json()["error"] == "invalid_request", body


async def test_without_the_voice_assistant_end_and_mute_change_nothing(make_engine):
    engine = await make_engine(isolation=False)
    auth = {"Authorization": "Bearer pl08-engine-token"}
    assert (await engine.client.post("/v1/wake/stop", headers=auth)).json() == {"stopped": False, "state": "off"}
    res = await engine.client.post("/v1/wake/mute", json={"muted": True}, headers=auth)
    assert res.json() == {"muted": False, "state": "off"}
    assert (await engine.client.get("/v1/wake/state", headers=auth)).json()["muted"] is False
