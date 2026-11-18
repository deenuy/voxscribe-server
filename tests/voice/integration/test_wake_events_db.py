"""wake_events rows for every candidate, against a throwaway Postgres schema (T02 migration 016)."""
from __future__ import annotations

import pytest

from app.local import wake_loop
from tests.conftest import make_app

pytestmark = pytest.mark.integration


async def test_every_candidate_writes_a_row_with_score_and_reason(throwaway_schema):
    app = await make_app(throwaway_schema)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        await wake_loop._record_wake_event(app, score=0.83, accepted=False, reject_reason="no_match",
                                           detector="openvocab:hey_vox", speech_ratio=0.61)
        await wake_loop._record_wake_event(app, score=0.91, accepted=False, reject_reason="no_voice",
                                           detector="openvocab:hey_vox", speech_ratio=0.05)
        await wake_loop._record_wake_event(app, score=1.0, accepted=True, reject_reason=None,
                                           detector="openvocab:hey_vox", speech_ratio=0.7)
        async with app.state.store.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT phrase, confidence, accepted, reject_reason, detector, speech_ratio, "
                "started_meeting_id FROM wake_events ORDER BY confidence")
            rows = await cur.fetchall()
    finally:
        await lifespan.__aexit__(None, None, None)

    assert [(r[2], r[3]) for r in rows] == [(False, "no_match"), (False, "no_voice"), (True, None)]
    assert all(r[0] == "hey_vox" and r[4] == "openvocab:hey_vox" and r[6] is None for r in rows)
    assert [round(r[1], 2) for r in rows] == [0.83, 0.91, 1.0]
    assert [round(r[5], 2) for r in rows] == [0.61, 0.05, 0.7]
