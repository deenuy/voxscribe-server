"""A three-turn voice session on the host with the real Postgres store and the real policy gate.

Faked: audio, wake detector, recognizer (scripted text per turn), turn policy, reply model and
talker. Real: PostgresSessionStore (meetings, wake_events, segments, turns, replies,
gate_records, latency_ledger) and CorePolicyGate, against a throwaway schema.
"""
from __future__ import annotations

import asyncio

import pytest

from app.local.voice.adapters.gate_policy import CorePolicyGate
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.conftest import make_app
from tests.voice.unit.fakes import EndlessQueue, FakeRecognizer, fake_ports

pytestmark = pytest.mark.integration

TURNS = ("What is one?", "And two?", "And three?")
SESSION_TIMEOUT_S = 20


async def test_three_turn_session_writes_every_row(throwaway_schema):
    app = await make_app(throwaway_schema)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        ports = fake_ports(store=PostgresSessionStore(app), gate=CorePolicyGate(),
                           recognizer=FakeRecognizer(TURNS))
        host = VoiceHost(VoiceConfig(), ports=ports)
        await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.92,
                                                detector="fake:hey_vox", speech_ratio=0.7),
                               SESSION_TIMEOUT_S)

        async with app.state.store.pool.connection() as conn:
            async def rows(sql):
                return await (await conn.execute(sql)).fetchall()

            meetings = await rows("SELECT id, origin FROM meetings")
            wakes = await rows("SELECT accepted, detector, started_meeting_id FROM wake_events")
            turns = await rows("SELECT turn_n, user_text, ended_at IS NOT NULL FROM turns "
                               "ORDER BY turn_n")
            gates = await rows("SELECT decision, category FROM gate_outcomes")    # the view
            events = await rows("SELECT event, count(*) FROM gate_records GROUP BY event")
            replies = await rows("SELECT turn_n, text FROM replies ORDER BY turn_n")
            ledger = await rows("SELECT count(*) FROM latency_ledger")
            segments = await rows("SELECT count(*) FROM transcript_segments")
    finally:
        await lifespan.__aexit__(None, None, None)

    ((meeting_id, origin),) = meetings
    assert origin == "wake"
    assert wakes == [(True, "fake:hey_vox", meeting_id)]
    assert turns == [(1, TURNS[0], True), (2, TURNS[1], True), (3, TURNS[2], True)]
    assert gates == [("pass", "ok")] * len(TURNS)
    assert dict(events) == {"policy_applied": 1, "heard": len(TURNS), "decision": len(TURNS),
                            "outcome": len(TURNS)}
    assert replies == [(1, "Answer 1."), (2, "Answer 2."), (3, "Answer 3.")]
    assert ledger == [(len(TURNS),)]
    assert segments == [(len(TURNS),)]
