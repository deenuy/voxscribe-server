"""Chat is the tenant's, a scribe thread is answered from that scribe's transcript, and the reply stream is valid and
leaves the event loop free (PL36 A).

The real engine on a throwaway schema with fake providers. The model is a scripted fake that records what it was
given, so a test can see exactly what an answer rested on."""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

import psycopg

from app.local import tenancy
from app.local.chat_grounding import NOT_IN_TRANSCRIPT, SCRIBE_NOT_FOUND, select_segments
from app.local.models import Segment
from app.local.store_meetings import MeetingStore
from tests import test_tenancy

keys = test_tenancy.keys                     # the engine with identity on, and its signing key (fixtures)
tenant_engine = test_tenancy.tenant_engine

REPO = Path(__file__).resolve().parent.parent


class ScriptedLLM:
    """Answers "The codeword is X." when its prompt holds "codeword is X", else the not-in-transcript sentence.
    Records every (system, prompt) it was given. `delay_s` sleeps (blocking) before each token: a slow model."""

    def __init__(self, delay_s: float = 0.0, tokens: list[str] | None = None):
        self.calls: list[tuple[str, str]] = []
        self.delay_s = delay_s
        self.tokens = tokens

    name = "fake:scripted"

    def stream(self, system, prompt, max_tokens=2048):
        self.calls.append((system, prompt))
        found = re.search(r"codeword is (\w+)", prompt)
        out = self.tokens or (["The codeword is ", found.group(1), "."] if found else [NOT_IN_TRANSCRIPT])
        for token in out:
            if self.delay_s:
                time.sleep(self.delay_s)
            yield token

    def complete(self, system, prompt, max_tokens=2048):
        return "".join(self.stream(system, prompt, max_tokens))


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """(event, data) pairs; every data line must be valid JSON."""
    out = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2, block
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def answer_of(events) -> str:
    return "".join(d["text"] for e, d in events if e == "token")


async def ask(client, headers, scope, question):
    thread = (await client.post("/v1/chat/threads", json={"scope": scope}, headers=headers)).json()
    res = await client.post(f"/v1/chat/threads/{thread['id']}/messages", json={"text": question}, headers=headers)
    assert res.status_code == 200, res.text
    return thread, parse_sse(res.text)


async def seed_scribe(pool, text: str, *, stop: bool = True) -> str:
    """A recorded scribe with one stable segment, stopped (processing: finalisation has not run, nothing indexed)."""
    store = MeetingStore(pool)
    mid = str(uuid.uuid4())
    await store.create_meeting(meeting_id=mid, title="Planning")
    await store.start_meeting(mid)
    await store.append_segment(mid, Segment(id=f"seg-{uuid.uuid4().hex[:8]}", startMs=1000, endMs=4000, source="mic",
                                            text=text, state="stable"))
    if stop:
        await store.stop_once(mid)
    return mid


async def test_a_scribe_question_is_answered_from_that_scribe_and_nobody_elses(tenant_engine):
    client, app, as_user = tenant_engine
    llm = ScriptedLLM()
    app.state.providers["llm_provider"] = llm
    pool = app.state.store.pool
    alice, bob = as_user("alice"), as_user("bob")

    with tenancy.tenant_scope(tenancy.tenant_key("alice")):
        mid = await seed_scribe(pool, "We agreed that the launch codeword is PELICAN for Friday.")
    # The same meeting id, and a source, planted in the shared schema the pool was built on.
    with tenancy.system_scope():
        shared = MeetingStore(pool)
        await shared.create_meeting(meeting_id=mid, title="Planted")
        await shared.import_lines(mid, [{"startMs": 0, "endMs": 1000, "text": "the launch codeword is OSPREY"}])
        async with pool.connection() as conn:
            await conn.execute("INSERT INTO sources (id, kind, uri, name, status) VALUES ('planted', 'file', 'x', 'p', 'ready')")
            await conn.execute("INSERT INTO source_chunks (id, source_id, ordinal, text) VALUES "
                               "('planted#0', 'planted', 0, 'the launch codeword is OSPREY')")

    # Asked right after Stop, before any finalisation or indexing: answered from the transcript, cited to it.
    thread, events = await ask(client, alice, {"meetingId": mid}, "What is the launch codeword?")
    assert answer_of(events) == "The codeword is PELICAN."
    citations = dict(events)["citations"]["citations"]
    assert [c["kind"] for c in citations] == ["transcript"] and citations[0]["startMs"] == 1000
    assert "done" in dict(events)
    assert len(llm.calls) == 1 and "PELICAN" in llm.calls[0][1] and "OSPREY" not in llm.calls[0][1]

    # Bob asks about Alice's scribe by its id: not his, so a plain sentence and no model call.
    _, events = await ask(client, bob, {"meetingId": mid}, "What is the launch codeword?")
    assert answer_of(events) == SCRIBE_NOT_FOUND
    assert dict(events)["citations"]["citations"] == []
    assert len(llm.calls) == 1

    # Bob's own search over sources: neither Alice's scribe nor the shared schema's planted source reaches him.
    _, events = await ask(client, bob, None, "What is the launch codeword?")
    text = answer_of(events)
    assert "OSPREY" not in text and "PELICAN" not in text
    assert dict(events)["citations"]["citations"] == []
    assert all("OSPREY" not in prompt and "PELICAN" not in prompt for _, prompt in llm.calls[1:])

    # The question and the answer are stored in Alice's thread, which Bob cannot see.
    stored = (await client.get(f"/v1/chat/threads/{thread['id']}", headers=alice)).json()["messages"]
    assert [(m["role"], m["text"]) for m in stored] == [("user", "What is the launch codeword?"),
                                                        ("assistant", "The codeword is PELICAN.")]
    assert (await client.get(f"/v1/chat/threads/{thread['id']}", headers=bob)).status_code == 404


async def test_a_scribe_that_does_not_hold_the_answer_says_so(tenant_engine):
    client, app, as_user = tenant_engine
    app.state.providers["llm_provider"] = llm = ScriptedLLM()
    alice = as_user("alice")
    with tenancy.tenant_scope(tenancy.tenant_key("alice")):
        mid = await seed_scribe(app.state.store.pool, "We talked about the garden and the new fence.")
    _, events = await ask(client, alice, {"meetingId": mid}, "What is the launch codeword?")
    assert answer_of(events) == NOT_IN_TRANSCRIPT
    assert NOT_IN_TRANSCRIPT in llm.calls[0][0]          # the model is told the exact sentence to use


async def test_chat_opens_no_database_connection_outside_the_pool(tenant_engine, monkeypatch):
    """Runtime: with psycopg's own connect refused for the length of the request, a scribe question and a sources
    question both answer. Only the pool's connections (opened before) are used."""
    client, app, as_user = tenant_engine
    app.state.providers["llm_provider"] = ScriptedLLM()
    alice = as_user("alice")
    with tenancy.tenant_scope(tenancy.tenant_key("alice")):
        mid = await seed_scribe(app.state.store.pool, "the launch codeword is PELICAN")
        await client.get("/v1/meetings", headers=alice)    # the tenant's schema is provisioned before the trap

    opened = []

    def refuse(*a, **k):
        opened.append(a)
        raise AssertionError("chat opened a database connection outside the pool")

    async def refuse_async(*a, **k):
        refuse(*a, **k)

    monkeypatch.setattr(psycopg, "connect", refuse)
    monkeypatch.setattr(psycopg.Connection, "connect", classmethod(lambda cls, *a, **k: refuse(*a, **k)))
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", classmethod(lambda cls, *a, **k: refuse_async(*a, **k)))
    _, events = await ask(client, alice, {"meetingId": mid}, "What is the launch codeword?")
    assert answer_of(events) == "The codeword is PELICAN."
    _, events = await ask(client, alice, None, "What is the launch codeword?")
    assert "done" in dict(events)
    assert opened == []


def test_no_chat_module_can_open_a_connection_of_its_own():
    """Static: the chat router and its grounding import nothing that opens a connection (psycopg, the sync index)."""
    for rel in ("app/local/routers/chat.py", "app/local/chat_grounding.py"):
        tree = ast.parse((REPO / rel).read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names |= {a.name for a in node.names}
        assert not {n for n in names if n.startswith("psycopg") or "pg_sync_index" in n
                    or n in ("PostgresKnowledgeIndexSync", "PostgresSyncRetriever", "database_url")}, rel
        assert "database_url" not in (REPO / rel).read_text(), rel


async def test_tokens_with_quotes_backslashes_and_control_characters_stream_as_valid_json(client):
    app = client._transport.app
    tricky = ['He said "go"', " C:\\path\\to", "\nnew line\ttab", "\x00\x1f end", " é ✓"]
    app.state.providers["llm_provider"] = ScriptedLLM(tokens=tricky)
    store = MeetingStore(app.state.store.pool)
    mid = await seed_scribe(app.state.store.pool, "anything at all was said here")
    _, events = await ask(client, {}, {"meetingId": mid}, "what was said?")
    assert answer_of(events) == "".join(tricky).replace("\x00", "")     # NUL cannot be stored, so is not sent
    assert [e for e, _ in events] == ["token"] * len(tricky) + ["citations", "done"]
    assert (await store.get_meeting(mid)).id == mid


async def test_the_event_loop_keeps_running_while_a_slow_model_streams(client):
    """Each token takes 0.3 s of blocking work. On the loop that would stall it 0.3 s at a time; off it, a ticker
    on the loop never waits much more than its own period."""
    app = client._transport.app
    app.state.providers["llm_provider"] = ScriptedLLM(delay_s=0.3, tokens=["one ", "two ", "three ", "four"])
    mid = await seed_scribe(app.state.store.pool, "anything at all was said here")
    gaps: list[float] = []
    running = True

    async def ticker():
        last = time.monotonic()
        while running:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    started = time.monotonic()
    _, events = await ask(client, {}, {"meetingId": mid}, "what was said?")
    elapsed = time.monotonic() - started
    running = False
    await tick
    assert answer_of(events) == "one two three four"
    assert elapsed >= 1.2                       # the model really was slow
    assert max(gaps) < 0.2, max(gaps)           # and the loop was never held for a token


def test_a_long_transcript_is_cut_to_the_segments_that_share_words_with_the_question():
    segs = [Segment(id=f"s{i}", startMs=i * 1000, endMs=i * 1000 + 900, source="mic", state="stable",
                    text=("the budget is ninety thousand" if i == 50 else f"filler sentence number {i} about nothing"))
            for i in range(100)]
    chosen = select_segments(segs, "What is the budget?", limit=200)
    ids = [s.id for s in chosen]
    assert "s50" in ids and "s49" in ids and "s51" in ids     # the match and its neighbours
    assert sum(len(s.text) + 12 for s in chosen) <= 260
    assert ids == sorted(ids, key=lambda x: int(x[1:]))        # spoken order
    assert select_segments(segs[:3], "anything", limit=10_000) == segs[:3]
    both_ends = [s.id for s in select_segments(segs, "zzz", limit=200)]
    assert both_ends[0] == "s0" and both_ends[-1] == "s99"
