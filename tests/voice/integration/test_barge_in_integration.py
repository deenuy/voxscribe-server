"""Barge-in end to end on the host (T13, VX-D2).

test_interruption_to_silence_timing   O1 at SIM level: the host's reply and barge-in stages on a
                                      fake clock shared by capture and playback, ten
                                      interruptions at different points of a reply; time from
                                      the decision to silence, and from the user's first word to
                                      silence, median under O1_TARGET_MS.
test_interrupted_turn_writes_heard_and_full_text
                                      the Postgres store: replies.text and played_text are the
                                      heard clauses, full_text everything released, barge_in set.
test_reply_after_interruption_repeats_no_clause
                                      O14 with the real reply model (Ollama), skipped when it is
                                      not running: ten interjections after a reply cut short, and
                                      the next reply must repeat no clause the user heard.

What SIM does not cover: the output device's own buffer, which plays out after the engine's last
slice write. That needs the ROOM run.
"""
from __future__ import annotations

import asyncio
import re
import statistics

import httpx
import pytest

from app.local.voice.barge_in import BARGE_IN_ENERGY_THRESHOLD, interruption_note
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.voice.unit.fakes import SPEECH_MS_PER_CHAR, Burst, FakeDelegate, sim_ports

pytestmark = pytest.mark.integration

O1_TARGET_MS = 300
SCENARIOS = 10
LOUD = BARGE_IN_ENERGY_THRESHOLD * 3
INTERJECTION_MS = 1500
REPLY_TIMEOUT_S = 10
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_PROBE_TIMEOUT_S = 2
LIVE_TIMEOUT_S = 120
MIN_CLAUSE_WORDS = 3            # shorter clauses ("Sure.") repeat by accident, not by echo

CLAUSES = ["Sure, here is the forecast.", "Tomorrow will be sunny in Montreal,",
           "with a high of twelve degrees.", "Wind from the west at twenty kilometres an hour.",
           "On Friday clouds move in,", "and rain is likely by the evening."]


def _ms(text: str) -> int:
    return len(text) * SPEECH_MS_PER_CHAR


async def test_interruption_to_silence_timing():
    decide_to_silence, onset_to_silence = [], []
    total = sum(_ms(c) for c in CLAUSES)
    for case in range(SCENARIOS):
        burst = Burst(at_playback_ms=total * (case + 0.5) / (SCENARIOS + 1),
                      dur_ms=INTERJECTION_MS, amp=LOUD)
        ports, capture = sim_ports([burst], delegate=FakeDelegate([CLAUSES]))
        host = VoiceHost(VoiceConfig(), ports=ports)
        result = await asyncio.wait_for(
            host._reply("s1", 1, "what is the weather this week", audio_q=capture,
                        turn_end_ms=0), REPLY_TIMEOUT_S)
        assert result.interruption is not None, case
        (stop_ms, silent_ms, _heard), = ports.audio_out.stops
        decide_to_silence.append(silent_ms - stop_ms)
        onset_to_silence.append(silent_ms - burst.onset_ms)

    decide_median = statistics.median(decide_to_silence)
    onset_median = statistics.median(onset_to_silence)
    print(f"\nO1 SIM: decision to silence median {decide_median:.0f} ms "
          f"(max {max(decide_to_silence):.0f}); first word to silence median "
          f"{onset_median:.0f} ms (max {max(onset_to_silence):.0f}); n={SCENARIOS}")
    assert decide_median < O1_TARGET_MS
    assert onset_median < O1_TARGET_MS


async def test_interrupted_turn_writes_heard_and_full_text(throwaway_schema):
    from app.local.voice.adapters.gate_rule import RuleGate
    from app.local.voice.adapters.store_postgres import PostgresSessionStore
    from tests.conftest import make_app

    app = await make_app(throwaway_schema)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        clauses = CLAUSES[:3]
        burst = Burst(at_playback_ms=_ms(clauses[0]) + _ms(clauses[1]) + _ms(clauses[2]) / 2,
                      dur_ms=INTERJECTION_MS, amp=LOUD)
        ports, capture = sim_ports([burst], delegate=FakeDelegate([clauses]),
                                   store=PostgresSessionStore(app), gate=RuleGate())
        host = VoiceHost(VoiceConfig(), ports=ports)
        session_id = await ports.store.create_session(0.9, detector="fake")
        await asyncio.wait_for(host._reply(session_id, 1, "what is the weather",
                                           audio_q=capture, turn_end_ms=0), REPLY_TIMEOUT_S)
        async with app.state.store.pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT text, barge_in, played_text, full_text FROM replies")).fetchall()
    finally:
        await lifespan.__aexit__(None, None, None)

    heard = " ".join(clauses[:2])
    assert rows == [(heard, True, heard, " ".join(clauses))]


# ── O14 with the real reply model ────────────────────────────────────────────

# (question, the reply heard before the user cut in, the interjection)
LIVE_CASES = [
    ("What is the weather tomorrow?", "Tomorrow will be sunny with a high of twelve degrees.",
     "wait, what about Friday?"),
    ("Tell me about the Eiffel Tower.", "The Eiffel Tower was finished in 1889 for a world fair.",
     "how tall is it?"),
    ("How do I boil an egg?", "Put the egg in a pot of cold water and bring it to a boil.",
     "how long for a soft one?"),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare around 1600.",
     "what else did he write?"),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa, in Ontario.",
     "and the biggest city?"),
    ("How far is the moon?", "The moon is about 384,000 kilometres from the Earth.",
     "how long would it take to fly there?"),
    ("Give me a dinner idea.", "You could make a quick pasta with garlic, olive oil and spinach.",
     "something without pasta please"),
    ("What does a heart do?", "The heart pumps blood through the body about seventy times a minute.",
     "what about the lungs?"),
    ("When is the next full moon?", "The next full moon is in about two weeks.",
     "is it a supermoon?"),
    ("How do plants grow?", "Plants use sunlight to turn water and carbon dioxide into sugar.",
     "do they need soil?"),
]


def _clauses(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?,;])\s+", text)
    return [p for p in (_norm(x) for x in parts) if len(p.split()) >= MIN_CLAUSE_WORDS]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _ollama_up() -> bool:
    try:
        return httpx.get(OLLAMA_TAGS_URL, timeout=OLLAMA_PROBE_TIMEOUT_S).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="needs the reply model (Ollama on localhost)")
async def test_reply_after_interruption_repeats_no_clause():
    from types import SimpleNamespace

    from app.local import reply_chain

    app = SimpleNamespace(state=SimpleNamespace(settings=None))
    repeated = []
    for question, heard, interjection in LIVE_CASES:
        history = [{"role": "user", "content": question},
                   {"role": "assistant", "content": heard}]
        src = reply_chain.reply_sentences(app, "live", interjection, history=history,
                                          system_note=interruption_note(heard))
        async def collect(sentences):
            return " ".join([s async for s in sentences])

        reply = await asyncio.wait_for(collect(src.sentences), LIVE_TIMEOUT_S)
        said = _norm(reply)
        hits = [c for c in _clauses(heard) if c in said]
        if hits:
            repeated.append((interjection, hits))
    print(f"\nO14 live (reply model): {SCENARIOS - len(repeated)} of {SCENARIOS} "
          f"replies repeat no heard clause; repeats: {repeated}")
    assert not repeated
