"""Voice events are scoped, typed and sequenced (PL06).

Host level with every port a fake, but the real event bus (app.local.events), the real event
sink (voice/adapters/events_app.py) and the real tenant scope (voice/adapters/scope.py) over an
app whose pool says isolation is on. What is shown: a session's events reach its tenant only,
carry the session id and a sequence number with no gap, and fit the event schema; device events
(wake_state, capture_health) go to the tenant the device is bound to whoever's scope they are
published from, and to nobody when no tenant is bound; with isolation off a subscriber with no
tenant gets everything with the same fields as before.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from app.local import event_schema, events, tenancy
from app.local.voice.adapters import events_app
from app.local.voice.adapters.scope import TenantSessionScope
from app.local.voice.host import SessionTenantRequiredError, VoiceConfig, VoiceHost

from .fakes import EndlessQueue, FakeRecognizer, fake_ports

TURNS = ["What is one?", "And two?"]
SESSION_TIMEOUT_S = 10
A, B = tenancy.tenant_key("pl06-alice"), tenancy.tenant_key("pl06-bob")


@pytest.fixture(autouse=True)
def fresh_streams(monkeypatch):
    monkeypatch.setattr(events, "_streams", type(events._streams)())


def _app(enforced: bool):
    pool = SimpleNamespace(enforced=enforced)
    return SimpleNamespace(state=SimpleNamespace(store=SimpleNamespace(pool=pool), profile=None,
                                                 settings=None))


def _host(*, enforced: bool, tenant: str | None) -> VoiceHost:
    scope = TenantSessionScope(_app(enforced))
    return VoiceHost(VoiceConfig(), ports=fake_ports(recognizer=FakeRecognizer(TURNS)),
                     scope=scope, binding=scope.device_binding(tenant), publish=events_app.publish)


async def _subscribe(stack: AsyncExitStack, tenant: str | None) -> events.Subscription:
    if tenant is None:
        return await stack.enter_async_context(events.subscribe())
    with tenancy.tenant_scope(tenant):
        return await stack.enter_async_context(events.subscribe())


def _drain(sub: events.Subscription) -> list[events.Event]:
    out = []
    while not sub.empty():
        out.append(sub.get_nowait())
    return out


async def test_tenant_b_receives_zero_wake_events_while_tenant_a_holds_a_voice_session():
    host = _host(enforced=True, tenant=A)
    async with AsyncExitStack() as stack:
        sub_a, sub_b = await _subscribe(stack, A), await _subscribe(stack, B)

        with tenancy.tenant_scope(B):           # whoever's request arms it, the state is A's
            assert host.arm()
        await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                               SESSION_TIMEOUT_S)
        host.on_capture_health(SimpleNamespace(issue="stall", action="reopened", value=0.0))

        got_a, got_b = _drain(sub_a), _drain(sub_b)

    assert got_b == []
    kinds = [e.type for e in got_a]
    assert kinds[0] == "wake_state" and kinds[-1] == "capture_health"
    assert {"wake.transcript", "wake.reply"} <= set(kinds)

    session_events = [e for e in got_a if e.type in event_schema.SESSION_EVENTS]
    assert [e.seq for e in session_events] == list(range(1, len(session_events) + 1))
    assert {e.session_id for e in session_events} == {"s1"}
    for e in got_a:
        if e.type in event_schema.DEVICE_EVENTS:
            assert e.seq is None and e.session_id is None
        event_schema.validate(e.type, e.data)                 # every payload fits its model
        event_schema.Envelope.model_validate(e.envelope())


async def test_with_isolation_on_and_nobody_bound_device_events_reach_nobody():
    host = _host(enforced=True, tenant=None)
    async with AsyncExitStack() as stack:
        subs = [await _subscribe(stack, A), await _subscribe(stack, B), await _subscribe(stack, None)]
        with pytest.raises(SessionTenantRequiredError):
            host.arm()
        host.on_capture_health(SimpleNamespace(issue="stall", action="reopen_failed", value=None))
        assert [_drain(s) for s in subs] == [[], [], []]


async def test_with_isolation_off_a_client_with_no_tenant_gets_every_event_with_its_fields():
    host = _host(enforced=False, tenant=None)
    async with AsyncExitStack() as stack:
        sub = await _subscribe(stack, None)
        assert host.arm()
        await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                               SESSION_TIMEOUT_S)
        got = _drain(sub)

    by_kind = {}
    for e in got:
        by_kind.setdefault(e.type, e.data)
    # The fields these events had before PL06, unchanged: nothing added to or removed from data.
    assert set(by_kind["wake_state"]) == {"state", "model", "phrase", "trigger", "meeting_id"}
    assert set(by_kind["wake.transcript"]) == {"meetingId", "turn", "segment"}
    assert set(by_kind["wake.reply"]) == {"meetingId", "turn", "phase", "userText", "text"}
    assert all(set(e) == {"type", "data"} for e in got)
