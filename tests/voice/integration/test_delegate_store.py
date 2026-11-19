"""The delegate lane against the real Postgres store (T14, migration 017).

Faked: audio, talker, the search engine and the reply model. Real: PostgresSessionStore, the
delegate adapter with its production lane and thinker, CorePolicyGate, and the delegate_tasks and
tool_calls rows in a throwaway schema.
"""
from __future__ import annotations

import asyncio

import pytest

from app.local import llm_adapter
from app.local.tools import web_search
from app.local.voice import delegate_lane as dl
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.gate_policy import CorePolicyGate
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.delegate_lane import TaskStatus
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.conftest import make_app
from tests.voice.unit.fakes import fake_ports

pytestmark = pytest.mark.integration

REQUEST = "search the web for standing desks"
SUMMARY = "The Uplift desk is the top pick."
STEP_TIMEOUT_S = 20


async def test_a_delegated_search_writes_its_task_row_and_a_tool_row_with_the_task_id(
        throwaway_schema, monkeypatch):
    async def search(_query: str) -> str:
        return "[SEARCH RESULTS]\n1. Desk: a desk (http://example.test)\n[END SEARCH RESULTS]"

    async def stream_reply(_user_text, **_kw):
        yield SUMMARY

    monkeypatch.setattr(web_search, "search", search)
    monkeypatch.setattr(llm_adapter, "stream_reply", stream_reply)

    app = await make_app(throwaway_schema)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        delegate = OllamaDelegate(app)
        ports = fake_ports(store=PostgresSessionStore(app), gate=CorePolicyGate(), delegate=delegate)
        host = VoiceHost(VoiceConfig(), ports=ports)
        session_id = await ports.store.create_session(0.9, detector="fake:hey_vox")

        first = await host.reply_turn(session_id, 1, REQUEST, turn_end_ms=0)
        task = delegate.lane.latest_open()
        await asyncio.wait_for(delegate.lane.wait(task.task_id), STEP_TIMEOUT_S)
        second = await host.reply_turn(session_id, 2, "What did you find?", turn_end_ms=0)
        await asyncio.wait_for(delegate.drain(), STEP_TIMEOUT_S)

        async with app.state.store.pool.connection() as conn:
            async def rows(sql):
                return await (await conn.execute(sql)).fetchall()

            tasks = await rows("SELECT id::text, task_id, session_id, request, status, result, "
                               "created_at IS NOT NULL, delivered_at IS NOT NULL "
                               "FROM delegate_tasks")
            tools = await rows("SELECT tc.task_id, tc.tool, tc.meeting_id, t.turn_n "
                               "FROM tool_calls tc JOIN turns t ON t.id = tc.turn_id")
            orphans = await rows("SELECT count(*) FROM tool_calls tc LEFT JOIN delegate_tasks d "
                                 "ON d.task_id = tc.task_id WHERE d.task_id IS NULL")
            replies = await rows("SELECT turn_n, text FROM replies ORDER BY turn_n")
    finally:
        await lifespan.__aexit__(None, None, None)

    assert first == dl.DELEGATING_PHRASE
    assert task.status is TaskStatus.DELIVERED
    assert tasks == [(task.task_id, task.task_id, session_id, REQUEST, "delivered", SUMMARY,
                      True, True)]
    assert tools == [(task.task_id, "web_search", session_id, 1)]
    assert orphans == [(0,)]                         # every tool row has its delegate task
    assert second == f"{dl.DELIVERY_LEAD.format(topic='standing desks')} {SUMMARY}"
    assert replies == [(1, first), (2, second)]
