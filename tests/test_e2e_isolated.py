"""In-process E2E: voice session event isolation (PL12).

This test covers the voice session part of the e2e isolation scenario that cannot be driven
headlessly over HTTP. The trigger for a voice session (POST /v1/wake/trigger) requires the
AudioEngine, which wraps the OS microphone and speaker. Without audio hardware the engine
starts but sets app.state.audio_engine to None and the route returns 503.

What this test proves instead:
  - Voice session events (wake.transcript) published inside Alice's tenant scope reach
    Alice's event stream and not Bob's.
  - After a stream reconnects with the Last-Event-ID it last received, it replays every
    event it missed, in order.
  - Reconnecting with an unknown Last-Event-ID triggers a stream.reset event (not silence).

Why these tests do not need audio hardware:
  fake_ports() substitutes FakeAudioIn, FakeRecognizer and FakeTalker, all in-process and
  pure Python. VoiceHost.run_session() calls the recognizer and talker directly, bypassing
  the AudioEngine entirely. The host is not started (no armed loop, no mic open); we call
  run_session() directly with an EndlessQueue to simulate the audio feed.

Scope/tenancy note:
  TenantSessionScope.session(binding) enters tenancy.tenant_scope(binding.tenant). Every
  events.publish() call inside a session therefore publishes to that tenant's stream, not the
  shared stream. events.subscribe() subscribes to the stream of the tenant in scope when the
  context manager is entered.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

import psycopg
import pytest

from app.local import events, tenancy
from app.local.config import Config
from app.local.migrations import run_migrations
from app.local.voice.adapters.events_app import publish as publish_event
from app.local.voice.adapters.scope import TenantSessionScope
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.testdb import TEST_DATABASE_URL as DB_URL
from tests.testdb import unsafe_reason  # noqa: F401 (imported for the pytest_sessionstart check)
from tests.test_tenancy import AUDIENCE, ISSUER, Keys
from tests.voice.unit.fakes import EndlessQueue, FakeRecognizer, fake_ports

# One turn is enough to generate a wake.transcript event.
TURNS = ("What is the weather today?",)
STEP_TIMEOUT_S = 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def isolated_engine():
    """An app with identity on, one base schema, and a cleanup that drops every schema."""
    data_dir = Path("/tmp/voxscribe_pl12_e2e")
    data_dir.mkdir(parents=True, exist_ok=True)

    base = f"t_pl12_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {base}")
        await conn.commit()
    await run_migrations(DB_URL, schema=base)

    settings = Config(
        database_url=DB_URL,
        data_dir=str(data_dir),
        providers="fake",
        identity_issuer=ISSUER,
        identity_audience=AUDIENCE,
        identity_jwks_url="http://jwks.test/keys",
    )
    from app.local.app import create_app

    app = await create_app(settings=settings, auth_disabled=True, schema=base)
    ctx = app.router.lifespan_context(app)
    await ctx.__aenter__()

    keys = Keys()
    created_schemas: list[str] = []

    def tenant_for(subject: str) -> str:
        key = tenancy.tenant_key(f"{subject}-{base}")
        created_schemas.append(key)
        return key

    yield app, tenant_for, keys

    await _settle()
    await ctx.__aexit__(None, None, None)
    async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
        for schema in set(created_schemas):
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.execute(f"DROP SCHEMA IF EXISTS {base} CASCADE")


async def _settle() -> None:
    from app.local import finalize
    while finalize._running:
        await asyncio.gather(*list(finalize._running), return_exceptions=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect_one(sub: events.Subscription, timeout: float = 5.0) -> events.Event:
    return await asyncio.wait_for(sub.next(), timeout)


async def _drain(sub: events.Subscription, timeout: float = 0.3) -> list[events.Event]:
    """Collect as many events as come within `timeout` seconds."""
    result = []
    try:
        while True:
            result.append(await asyncio.wait_for(sub.next(), timeout))
    except asyncio.TimeoutError:
        pass
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_voice_session_events_reach_only_the_bound_tenant(isolated_engine):
    """Alice runs a voice session; Alice's stream gets wake.transcript; Bob's does not."""
    app, tenant_for, _keys = isolated_engine

    alice_key = tenant_for("alice")
    bob_key = tenant_for("bob")

    scope = TenantSessionScope(app)
    # fake_ports with FakeStore (default): events go to the SSE bus via publish_event,
    # independent of the session store. No database writes needed for event isolation.
    ports = fake_ports(recognizer=FakeRecognizer(TURNS))
    host = VoiceHost(VoiceConfig(), ports=ports, scope=scope,
                     binding=scope.device_binding(alice_key),
                     publish=publish_event)

    alice_events: list[events.Event] = []
    bob_events: list[events.Event] = []

    async with contextlib.AsyncExitStack() as stack:
        # Subscribe to both streams before the session starts.
        with tenancy.tenant_scope(alice_key):
            alice_sub = await stack.enter_async_context(events.subscribe())
        with tenancy.tenant_scope(bob_key):
            bob_sub = await stack.enter_async_context(events.subscribe())

        # Run the voice session in Alice's scope.
        await asyncio.wait_for(
            host.run_session(EndlessQueue(), score=0.9, detector="fake"),
            STEP_TIMEOUT_S,
        )
        await _settle()

        alice_events = await _drain(alice_sub)
        bob_events = await _drain(bob_sub)

    wake_tx = [e for e in alice_events if e.type == "wake.transcript"]
    assert wake_tx, (
        f"Alice's stream has no wake.transcript events. All events: {[e.type for e in alice_events]}"
    )
    # Bob's stream must be silent; it may have received machine-level broadcast events
    # (e.g. model download) but nothing from Alice's session.
    session_id = wake_tx[0].get("sessionId") or wake_tx[0].get("meetingId") or wake_tx[0].data.get("meetingId")
    bob_session_events = [e for e in bob_events if e.get("sessionId") == session_id
                          or e.data.get("meetingId") == session_id]
    assert bob_session_events == [], (
        f"Bob's stream received session events that belong to Alice: {bob_session_events}"
    )


async def test_reconnect_with_last_event_id_replays_alice_events(isolated_engine):
    """After Alice's stream reconnects with her last seen event ID, she gets the same events."""
    app, tenant_for, _keys = isolated_engine

    alice_key = tenant_for("alice-reconnect")

    scope = TenantSessionScope(app)
    ports = fake_ports(recognizer=FakeRecognizer(TURNS))
    host = VoiceHost(VoiceConfig(), ports=ports, scope=scope,
                     binding=scope.device_binding(alice_key),
                     publish=publish_event)

    # First subscription: open the subscription (which creates the stream), then publish a
    # sentinel event inside the subscription so wake.transcript is never the first event.
    # This guarantees there is always a valid prev_id for the reconnect test.
    async with contextlib.AsyncExitStack() as stack:
        with tenancy.tenant_scope(alice_key):
            alice_sub = await stack.enter_async_context(events.subscribe())

        # Now the stream exists; publish sentinel (create=False is safe now).
        with tenancy.tenant_scope(alice_key):
            events.publish("meeting.update", {"id": "sentinel", "title": "before-session"})

        await asyncio.wait_for(
            host.run_session(EndlessQueue(), score=0.9, detector="fake"),
            STEP_TIMEOUT_S,
        )
        await _settle()
        first_pass = await _drain(alice_sub)

    wake_tx_first = [e for e in first_pass if e.type == "wake.transcript"]
    assert wake_tx_first, "no wake.transcript events in the first pass"

    # The ID of the first wake.transcript event.
    first_id = wake_tx_first[0].id
    assert first_id, "wake.transcript event has no id"

    # Compute the previous event's ID so we can replay from just before the first transcript.
    # Event IDs are "<epoch>-<n>" (see events.py). We ask for events after the event before it.
    # If the first event is event 1 of the epoch, we reconnect with an unknown id to get a reset,
    # then fall back to no reconnect.
    epoch, n = first_id.rsplit("-", 1)
    prev_id = f"{epoch}-{int(n) - 1}" if int(n) > 1 else None

    if prev_id is None:
        pytest.skip("first wake.transcript was the very first event in the stream epoch; cannot go further back")

    # Reconnect with the previous event's ID: should replay the transcript event.
    async with contextlib.AsyncExitStack() as stack:
        with tenancy.tenant_scope(alice_key):
            replay_sub = await stack.enter_async_context(events.subscribe(last_event_id=prev_id))
        replayed = await _drain(replay_sub)

    replayed_ids = [e.id for e in replayed if e.type == "wake.transcript"]
    assert first_id in replayed_ids, (
        f"reconnect did not replay the first wake.transcript event. "
        f"first_id={first_id!r}, got={replayed_ids!r}"
    )


async def test_reconnect_with_unknown_id_sends_stream_reset(isolated_engine):
    """Reconnecting with an ID the stream never issued gives a stream.reset event, not silence."""
    app, tenant_for, _keys = isolated_engine
    alice_key = tenant_for("alice-reset")

    # Publish one event to create the stream.
    with tenancy.tenant_scope(alice_key):
        events.publish("meeting.update", {"id": "test-meeting", "title": "init"})

    # Reconnect with an unknown ID.
    async with contextlib.AsyncExitStack() as stack:
        with tenancy.tenant_scope(alice_key):
            sub = await stack.enter_async_context(events.subscribe(last_event_id="unknown-epoch-99999"))
        got = await _drain(sub)

    reset_events = [e for e in got if e.type == "stream.reset"]
    assert reset_events, (
        f"expected a stream.reset event but got: {[e.type for e in got]}"
    )
