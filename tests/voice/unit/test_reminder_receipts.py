"""Defects D-1, D-2 and D-3 (TOOLS-CONNECTORS.md section 1.3), through the voice host's reply stage
with the real gate, the real session store and reminder store over the in-memory pool, and the real
delegate. No database, no model, no network.

D-1  a reminder that was saved is confirmed, not refused as an unbacked claim
D-2  a reminder whose save failed is never confirmed
D-3  a timer confirmation is an action claim and is checked like one; the reminder line keeps
     its words whole ("remind you at 3:00 PM: call the dentist", never "remind you call")
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from app.local import gate
from app.local.tools import audit, builtin
from app.local.voice import reminder_store as rs
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.voice.unit.fakes import fake_ports
from tests.voice.unit.tool_fakes import ToolPool, load_profile, make_app

REMIND = "remind me to call the dentist at 3pm"
TIMER = "set a timer for 5 minutes"
TIMER_LINE = "Got it, timer set for 5 minutes."


@pytest.fixture(autouse=True)
def _behaviour_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_BEHAVIOUR_LOG_DIR", tmp_path)


class Session:
    """One host, one delegate, one pool: turns in order, each with its own turn id."""

    def __init__(self, pool: ToolPool, mode: str = "personal") -> None:
        self.pool = pool
        self.app = make_app(pool, profile=load_profile(mode))
        ports = fake_ports(gate=CorePolicyGate(session_policy(self.app)),
                           store=PostgresSessionStore(self.app),
                           delegate=OllamaDelegate(self.app))
        self.host = VoiceHost(VoiceConfig(), ports=ports)
        self.turn_n = 0

    async def say(self, text: str, model_reply: str | None = None) -> str:
        self.turn_n += 1
        speculation = _tokens(model_reply) if model_reply else None
        return await self.host.reply_turn("s1", self.turn_n, text, turn_end_ms=0,
                                          speculation=speculation)

    def outcome(self) -> dict:
        """The gate's outcome record for the last reply."""
        rows = [r for r in self.pool.table("gate_records") if r.get("event") == "outcome"]
        return {"decision": rows[-1]["decision"], "category": rows[-1]["category"]}

    def rows(self, tool: str) -> list[dict]:
        return [r for r in self.pool.tool_calls if r["tool"] == tool]


async def _tokens(text: str):
    for word in text.split(" "):
        yield word + " "


def _expected_when() -> str:
    """3 pm local, as parse_reminder resolves it and the record speaks it."""
    _desc, fire_at = rs.parse_reminder(REMIND)
    return rs.format_local_time(fire_at)


async def test_d1_a_saved_reminder_is_confirmed_and_not_refused():
    session = Session(ToolPool())
    reply = await session.say(REMIND)
    assert reply == f"Got it, I'll remind you at {_expected_when()}: call the dentist."
    assert gate.ACTION_CLAIM_PHRASE not in reply
    assert session.outcome() == {"decision": "pass", "category": "ok"}
    assert len(session.pool.table("reminders")) == 1
    (row,) = session.rows(builtin.REMINDER_CREATE)
    assert row["status"] == audit.EXECUTED and row["result_ref"] == \
        session.pool.table("reminders")[0]["id"]


async def test_d2_a_failed_save_is_never_confirmed():
    session = Session(ToolPool(fail_reminders=True))
    reply = await session.say(REMIND)
    assert reply == "Sorry, I couldn't save that reminder."
    assert "remind you" not in reply.lower()
    assert session.pool.table("reminders") == []
    (row,) = session.rows(builtin.REMINDER_CREATE)
    assert (row["status"], row["error_code"]) == (audit.FAILED, "raised:RuntimeError")


async def test_d3_a_timer_with_its_receipt_is_confirmed_unchanged():
    session = Session(ToolPool())
    assert await session.say(TIMER) == TIMER_LINE
    assert session.outcome() == {"decision": "pass", "category": "ok"}
    (row,) = session.rows(builtin.TIMER_START)
    assert row["status"] == audit.EXECUTED
    assert session.pool.receipt_lookups == 1          # the gate checked the claim


async def test_d3_a_timer_line_with_no_receipt_is_replaced():
    """The same sentence from the reply model, with no timer call this turn."""
    session = Session(ToolPool())
    reply = await session.say("thanks", model_reply=TIMER_LINE)
    assert reply == gate.ACTION_CLAIM_PHRASE
    assert session.outcome() == {"decision": "replace", "category": gate.ACTION_CLAIM_REASON}


async def test_a_receipt_from_an_earlier_turn_backs_nothing():
    session = Session(ToolPool())
    assert await session.say(TIMER) == TIMER_LINE
    reply = await session.say("thanks", model_reply="I've set a timer for five minutes.")
    assert reply == gate.ACTION_CLAIM_PHRASE


async def test_a_failed_timer_is_not_confirmed_and_a_claim_after_it_is_replaced():
    session = Session(ToolPool(fail_reminders=True))
    assert await session.say(TIMER) == "Sorry, I couldn't start that timer."
    assert [r["status"] for r in session.rows(builtin.TIMER_START)] == [audit.FAILED]


async def test_a_saved_timer_without_its_receipt_is_not_claimed():
    class NoLedger(ToolPool):
        async def execute(self, sql, params=()):
            if sql.startswith("INSERT INTO tool_calls"):
                raise OSError("ledger down")
            return await super().execute(sql, params)

    session = Session(NoLedger())
    reply = await session.say(TIMER)
    assert reply == "I'm not sure that timer started."
    assert len(session.pool.table("reminders")) == 1


async def test_the_reminder_list_reads_through_the_registry():
    session = Session(ToolPool())                 # the fake pool's reminder query finds none
    assert await session.say("what reminders do I have?") == rs.REMINDER_NONE_PHRASE
    assert [r["status"] for r in session.rows(builtin.REMINDER_LIST)] == [audit.EXECUTED]
    assert session.pool.receipt_lookups == 0         # a read claims nothing


async def test_reminders_work_in_senior_care_as_before():
    session = Session(ToolPool(), mode="senior_care")
    reply = await session.say(REMIND)
    assert reply == f"Got it, I'll remind you at {_expected_when()}: call the dentist."


async def test_no_reminder_words_reach_the_logs(caplog):
    session = Session(ToolPool())
    with caplog.at_level(logging.DEBUG):
        await session.say(REMIND)
        await session.say(TIMER)
    assert "dentist" not in caplog.text


def test_a_reminder_fire_time_is_a_utc_instant_the_record_accepts():
    from app.local.tools.registry import default_registry

    spec = default_registry().get(builtin.REMINDER_CREATE)
    fire_at = datetime.now(UTC) + timedelta(minutes=3)
    assert spec.validate_args({"text": "stretch", "fire_at": fire_at.isoformat()})
