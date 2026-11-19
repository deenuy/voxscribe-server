"""PL45: reminders and timers as registry tools, read back before they are saved.

Each test speaks turns through the voice host's reply stage with the real gate (core's pack and
the host floor), the real session store and tool runtime over the in-memory pool (audit rows,
gate records), the real delegate and dialog, a fake clock and the in-memory reminder store. No
database, no model (a model reply, where a turn reaches the talker, is a fixed token stream),
no audio device and no network.

Acceptance criteria (TOOLS-CONNECTORS.md section 5, PL45) and the test that checks each:
  the validation gate's first sentence, as text with a fake clock, reads back three minutes,
  then 1 row and 1 receipt after yes, and the confirmation is spoken unchanged by the gate
                                          test_the_validation_sentence_is_read_back_then_set_on_yes
  no, or silence, yields 0 rows           test_no_sets_nothing, test_silence_past_the_window_*,
                                          test_a_session_that_ends_unanswered_sets_nothing
  a store failure yields no confirmation, the failure line and a failed row
                                          test_d2_a_store_failure_is_never_confirmed
  "how long is left on my timer" answers from the row
                                          test_how_long_is_left_answers_from_the_row
  under classroom "remind me" answers the timers-only line and writes no reminder
                                          test_classroom_has_timers_only
  with isolation on, tenant B's list never holds tenant A's rows
                                          tests/test_reminders_db.py
  D-1, D-2, D-3 (section 1.3)             test_d1_*, test_d2_*, test_d3_*
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import pytest

from app.local import gate
from app.local.tools import audit, builtin
from app.local.voice import reminder_dialog as rd
from app.local.voice import reminder_store as rs
from app.local.voice import spoken_time as st
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.voice.unit.fakes import fake_ports
from tests.voice.unit.reminder_fakes import ZONE, FakeClock, FakeReminderStore, SessionRows
from tests.voice.unit.tool_fakes import ToolPool, load_profile, make_app

VALIDATION = "remind me in two minutes, no, three minutes, to stretch"
READBACK = "In three minutes, stretch. Shall I set it?"
DENTIST = "remind me tomorrow at 8 pm to call the dentist"
DENTIST_READBACK = "Tomorrow at eight in the evening, call the dentist. Shall I set it?"


@pytest.fixture(autouse=True)
def _behaviour_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_BEHAVIOUR_LOG_DIR", tmp_path)


class Session:
    """One host, one delegate, one pool, one clock: turns in order, each with its own turn id,
    and the history the host keeps (what was heard of each reply)."""

    def __init__(self, mode: str = "personal", *, store: FakeReminderStore | None = None,
                 pool: ToolPool | None = None, **settings) -> None:
        self.pool = pool or ToolPool()
        self.app = make_app(self.pool, profile=load_profile(mode), vox_timezone=ZONE,
                            **settings)
        self.clock = FakeClock()
        self.store = store or FakeReminderStore()
        self.delegate = OllamaDelegate(self.app, reminder_store=self.store, clock=self.clock)
        ports = fake_ports(gate=CorePolicyGate(session_policy(self.app)),
                           store=PostgresSessionStore(self.app), delegate=self.delegate)
        self.host = VoiceHost(VoiceConfig(), ports=ports)
        self.history: list[dict] = []
        self.turn_n = 0

    async def say(self, text: str, model_reply: str | None = None, *, after_s: float = 2) -> str:
        self.clock.advance(seconds=after_s)
        self.turn_n += 1
        speculation = _tokens(model_reply) if model_reply else None
        reply = await self.host.reply_turn("s1", self.turn_n, text, turn_end_ms=0,
                                           history=list(self.history) or None,
                                           speculation=speculation)
        self.history += [{"role": "user", "content": text}]
        if reply:
            self.history += [{"role": "assistant", "content": reply}]
        return reply

    def rows(self, tool: str | None = None) -> list[dict]:
        return [r for r in self.pool.tool_calls if tool is None or r["tool"] == tool]

    def statuses(self, tool: str) -> list[tuple[str, str | None]]:
        return [(r["status"], r["error_code"]) for r in self.rows(tool)]

    def outcome(self) -> dict:
        rows = [r for r in self.pool.table("gate_records") if r.get("event") == "outcome"]
        return {"decision": rows[-1]["decision"], "category": rows[-1]["category"]}

    def reminders(self) -> list[rs.Item]:
        return self.store.of(rs.KIND_REMINDER)

    def timers(self) -> list[rs.Item]:
        return self.store.of(rs.KIND_TIMER)


async def _tokens(text: str):
    for word in text.split(" "):
        yield word + " "


# ── the read-back, step by step ──────────────────────────────────────────────────────────


async def test_the_validation_sentence_is_read_back_then_set_on_yes():
    s = Session()
    assert await s.say(VALIDATION) == READBACK
    assert s.reminders() == []                                   # nothing before the yes
    assert s.statuses(builtin.REMINDER_CREATE) == [(audit.PROPOSED, None)]
    assert await s.say("yes") == "Reminder set."
    assert s.outcome() == {"decision": "pass", "category": "ok"}  # the claim, unchanged
    (row,) = s.reminders()
    assert row.text == "stretch" and row.time_zone == ZONE
    assert row.fire_at == s.clock.now - timedelta(seconds=2) + timedelta(minutes=3)
    proposed, executed = s.rows(builtin.REMINDER_CREATE)
    assert executed["status"] == audit.EXECUTED and executed["result_ref"] == row.id
    assert executed["confirm_turn_id"] == executed["turn_id"] != proposed["turn_id"]
    assert executed["args_hash"] == proposed["args_hash"]        # what was read back is what is set
    assert s.store.receipts[row.id] == executed["id"]
    assert "stretch" not in str(executed["args"])                # a digest and a length


async def test_the_read_back_is_built_from_validated_arguments_only():
    s = Session()
    await s.say(VALIDATION)
    (proposed,) = s.rows(builtin.REMINDER_CREATE)
    kept = json.loads(proposed["args"])
    spec = s.app.state.tool_registry.get(builtin.REMINDER_CREATE)
    args = {"text": "stretch", "fire_at": kept["fire_at"], "when": kept["when"],
            "time_zone": ZONE}
    assert spec.render("readback", spec.validate_args(args)) == READBACK


async def test_no_sets_nothing():
    s = Session()
    await s.say(VALIDATION)
    assert await s.say("no") == rd.DECLINED_LINE
    assert s.reminders() == []
    assert s.statuses(builtin.REMINDER_CREATE) == [(audit.PROPOSED, None),
                                                    (audit.DECLINED, "no")]


async def test_a_correction_reads_back_again_and_yes_sets_the_corrected_time():
    s = Session()
    assert await s.say(DENTIST) == DENTIST_READBACK
    assert await s.say("no, nine") == ("Tomorrow at nine at night, call the dentist."
                                       " Shall I set it?")
    assert await s.say("yes") == "Reminder set."
    (row,) = s.reminders()
    assert row.fire_at.astimezone(row.tz).hour == 21
    assert s.statuses(builtin.REMINDER_CREATE) == [
        (audit.PROPOSED, None), (audit.DECLINED, "corrected"), (audit.PROPOSED, None),
        (audit.EXECUTED, None)]


async def test_silence_past_the_window_declines_and_the_late_turn_is_answered_normally():
    s = Session()
    await s.say(VALIDATION)
    late = st.READBACK_WINDOW.total_seconds() + 1
    reply = await s.say("yes", model_reply="Hello again.", after_s=late)
    assert reply == "Hello again."
    assert s.reminders() == []
    assert s.statuses(builtin.REMINDER_CREATE)[-1] == (audit.DECLINED, "silence")


async def test_the_host_measured_silence_is_what_the_window_counts():
    """The window runs from the end of the read-back to the start of the reply (the host's
    measure), not from when the read-back was produced."""
    s = Session()
    await s.say(VALIDATION)
    from app.local.voice.host import SessionContext, _session

    token = _session.set(SessionContext(reply_ended_mono=100.0, speech_onset_mono=101.0))
    try:
        # Long after the read-back was produced, but one second of silence before the yes.
        assert await s.say("yes", after_s=30) == "Reminder set."
    finally:
        _session.reset(token)
    assert len(s.reminders()) == 1


async def test_a_session_that_ends_unanswered_sets_nothing():
    s = Session()
    await s.say(VALIDATION)
    await s.delegate.session_closed("s1")
    assert s.reminders() == []
    assert s.statuses(builtin.REMINDER_CREATE)[-1] == (audit.DECLINED, "session_ended")
    assert not s.delegate.reminders.has_pending("s1")


async def test_an_unrelated_turn_declines_and_is_answered_normally():
    s = Session()
    await s.say(VALIDATION)
    reply = await s.say("what's the capital of France", model_reply="Paris.")
    assert reply == "Paris."
    assert s.reminders() == []
    assert s.statuses(builtin.REMINDER_CREATE)[-1] == (audit.DECLINED, "unrelated")


async def test_a_read_back_that_was_not_heard_whole_confirms_nothing():
    s = Session()
    await s.say(VALIDATION)
    s.history[-1]["content"] = "In three minutes, stretch."     # interrupted before the question
    await s.say("yes", model_reply="Okay.")
    assert s.reminders() == []
    assert s.statuses(builtin.REMINDER_CREATE)[-1] == (audit.DECLINED, "readback_unheard")


async def test_an_ambiguous_time_asks_its_one_question_first():
    s = Session()
    question = await s.say("remind me at 8 to call mom")
    assert question == ("Do you mean today at eight in the evening, or tomorrow at eight in the"
                        " morning?")
    assert s.statuses(builtin.REMINDER_CREATE) == [(audit.INVALID_ARGS,
                                                    "time:ambiguous_meridiem")]
    assert await s.say("in the morning") == ("Tomorrow at eight in the morning, call mom."
                                             " Shall I set it?")
    assert await s.say("yes") == "Reminder set."
    assert s.reminders()[0].fire_at.astimezone(s.reminders()[0].tz).hour == 8


async def test_no_time_asks_for_one_and_the_answer_is_read_back():
    s = Session()
    assert await s.say("remind me to call mom") == st.QUESTION_WHEN
    assert s.statuses(builtin.REMINDER_CREATE) == [(audit.INVALID_ARGS, "time:not_understood")]
    assert await s.say("in ten minutes") == "In ten minutes, call mom. Shall I set it?"


async def test_a_reminder_with_no_words_asks_what_it_is_about():
    s = Session()
    assert await s.say("remind me in ten minutes") == st.QUESTION_SUBJECT
    assert await s.say("to check the oven") == "In ten minutes, check the oven. Shall I set it?"


async def test_a_repeating_reminder_is_not_set():
    s = Session()
    assert await s.say("remind me every day at eight to walk") == rd.RECURRING_LINE
    assert s.reminders() == []


async def test_the_read_back_warns_when_the_time_is_inside_quiet_hours():
    from app.local.config import QuietHours

    s = Session(quiet_hours=QuietHours(enabled=True, start="22:00", end="07:00"))
    reply = await s.say("remind me tomorrow at 6 am to take out the bins")
    assert reply == (f"{rd.QUIET_HOURS_WARNING} Tomorrow at six in the morning, take out the"
                     " bins. Shall I set it?")
    assert await s.say("yes") == "Reminder set."


# ── D-1, D-2, D-3 (TOOLS-CONNECTORS.md section 1.3) ──────────────────────────────────────


async def test_d1_a_saved_reminder_is_confirmed_and_not_refused():
    s = Session()
    await s.say(DENTIST)
    reply = await s.say("yes")
    assert reply == "Reminder set." and gate.ACTION_CLAIM_PHRASE not in reply
    assert s.pool.receipt_lookups == 1                       # the gate checked the claim


async def test_d2_a_store_failure_is_never_confirmed():
    s = Session(store=FakeReminderStore(fail_add=True))
    await s.say(DENTIST)
    reply = await s.say("yes")
    assert reply == "Sorry, I couldn't save that reminder."
    assert s.statuses(builtin.REMINDER_CREATE)[-1] == (audit.FAILED, "raised:RuntimeError")
    assert s.reminders() == []


async def test_d3_a_timer_is_announced_from_its_receipt_and_its_claim_is_checked():
    s = Session()
    assert await s.say("set a timer for one minute") == "One minute timer, starting now."
    assert s.outcome() == {"decision": "pass", "category": "ok"}
    assert s.pool.receipt_lookups == 1                       # the announcement is a claim
    (timer,) = s.timers()
    assert timer.label == "one minute" and timer.text == ""
    assert timer.fire_at == s.clock.now + timedelta(minutes=1)
    assert s.statuses(builtin.TIMER_START) == [(audit.EXECUTED, None)]


async def test_d3_the_same_announcement_with_no_receipt_is_replaced():
    s = Session()
    reply = await s.say("thanks", model_reply="Ten minute timer, starting now.")
    assert reply == gate.ACTION_CLAIM_PHRASE


async def test_a_receipt_from_an_earlier_turn_backs_nothing():
    s = Session()
    await s.say("set a timer for one minute")
    reply = await s.say("thanks", model_reply="I've set a timer for five minutes.")
    assert reply == gate.ACTION_CLAIM_PHRASE


# ── timers ───────────────────────────────────────────────────────────────────────────────


async def test_how_long_is_left_answers_from_the_row():
    s = Session()
    await s.say("set a timer for ten minutes")
    s.clock.advance(minutes=3, seconds=58)                  # 4 min into it (2 s per turn)
    assert await s.say("how long is left on my timer") == "Six minutes left on your ten minute" \
                                                          " timer."
    assert s.statuses(builtin.TIMER_STATUS) == [(audit.EXECUTED, None)]


async def test_cancel_the_timer_by_voice():
    s = Session()
    await s.say("set a timer for ten minutes")
    assert await s.say("cancel the timer") == "Ten minute timer cancelled."
    assert s.timers()[0].status == rs.CANCELLED
    assert await s.say("how long is left on my timer") == rd.NO_TIMERS_LINE


async def test_a_timer_with_no_length_asks_how_long():
    s = Session()
    assert await s.say("set a timer please") == st.QUESTION_DURATION
    assert await s.say("five minutes") == "Five minute timer, starting now."


# ── lists and cancelling reminders ──────────────────────────────────────────────────────


async def test_list_and_cancel_reminders_by_voice():
    s = Session()
    assert await s.say("what reminders do I have") == rd.NO_REMINDERS_LINE
    await s.say(DENTIST)
    await s.say("yes")
    listed = await s.say("what reminders do I have")
    assert listed == ("You have one reminder coming up. Tomorrow at eight in the evening, call"
                      " the dentist.")
    question = await s.say("cancel my reminder to call the dentist")
    assert question == ("Cancel your reminder for tomorrow at eight in the evening, call the"
                        " dentist?")
    assert s.reminders()[0].status == rs.PENDING             # not before the yes
    assert await s.say("yes") == "Okay, that reminder is cancelled."
    assert s.reminders()[0].status == rs.CANCELLED
    assert s.statuses(builtin.REMINDER_CANCEL) == [(audit.PROPOSED, None),
                                                    (audit.EXECUTED, None)]


async def test_a_cancel_question_answered_no_keeps_the_reminder():
    s = Session()
    await s.say(DENTIST)
    await s.say("yes")
    await s.say("cancel the reminder")
    assert await s.say("no") == rd.CANCEL_KEPT_LINE
    assert s.reminders()[0].status == rs.PENDING


# ── product modes ────────────────────────────────────────────────────────────────────────


async def test_classroom_has_timers_only():
    s = Session("classroom")
    assert await s.say(DENTIST) == builtin.TIMERS_ONLY_LINE
    assert await s.say("what reminders do I have") == builtin.TIMERS_ONLY_LINE
    assert s.reminders() == []
    assert [r["error_code"] for r in s.rows(builtin.REMINDER_CREATE)] == ["mode_not_allowed"]
    assert await s.say("set a timer for five minutes") == "Five minute timer, starting now."


async def test_senior_care_reminders_are_read_back_and_set():
    s = Session("senior_care")
    assert await s.say(DENTIST) == DENTIST_READBACK
    assert await s.say("yes") == "Reminder set."


async def test_senior_care_dosing_words_fail_closed():
    """TC6: the dosing rule refuses the read-back, so it is never offered and no yes can set it.
    The turn's words carry the dose, so the host floor refuses every line of that turn and the
    person hears its escalation line. Flagged for clinician review."""
    s = Session("senior_care")
    reply = await s.say("remind me to take 2 pills at 8 pm")
    assert reply == gate.SENIORCARE_ESCALATION_PHRASE
    assert s.statuses(builtin.REMINDER_CREATE) == [(audit.REFUSED, "gate_refused")]
    await s.say("yes", model_reply="Okay.")
    assert s.reminders() == []


# ── privacy ──────────────────────────────────────────────────────────────────────────────


async def test_no_reminder_words_reach_the_logs(caplog):
    s = Session()
    with caplog.at_level(logging.DEBUG):
        await s.say(DENTIST)
        await s.say("yes")
        await s.say("what reminders do I have")
        await s.say("cancel my reminder to call the dentist")
        await s.say("yes")
    assert caplog.records and "dentist" not in caplog.text


def test_the_intent_matcher():
    cases = {
        "remind me to set a timer at five": rd.SET_REMINDER,
        "set a reminder for tomorrow at noon to call mom": rd.SET_REMINDER,
        "set a timer for ten minutes": rd.SET_TIMER,
        "start a five minute timer": rd.SET_TIMER,
        "how long is left on my timer": rd.TIMER_STATUS,
        "cancel the timer": rd.CANCEL_TIMER,
        "cancel my reminder to call the dentist": rd.CANCEL_REMINDER,
        "what reminders do I have": rd.LIST_REMINDERS,
        "what's the weather like": None,
        "I set the table": None,
    }
    for text, intent in cases.items():
        assert rd.match_intent(text) == intent, text


# ── through the host's turn loop ─────────────────────────────────────────────────────────


async def test_a_spoken_yes_of_one_word_sets_the_reminder_through_the_host_loop():
    """The whole session: the recognizer hears the request, then "Yes." (one word, which the
    speech guard lets through only because a read-back waits for it), then the session ends."""
    import asyncio

    from tests.voice.unit.fakes import EndlessQueue, FakeRecognizer

    s = Session()
    turns = (VALIDATION + ".", "Yes.")
    ports = fake_ports(gate=CorePolicyGate(session_policy(s.app)),
                       store=SessionRows(s.app), delegate=s.delegate,
                       recognizer=FakeRecognizer(turns))
    host = VoiceHost(VoiceConfig(), ports=ports)
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"), 10)
    (row,) = s.reminders()
    assert row.text == "stretch"
    assert [r for r, _ in s.statuses(builtin.REMINDER_CREATE)] == [audit.PROPOSED,
                                                                    audit.EXECUTED]


async def test_a_lone_yes_with_nothing_asked_still_ends_the_session():
    from app.local.voice.host import is_genuine_speech

    assert not is_genuine_speech("Yes.", 2_000)
    assert is_genuine_speech("Yes.", 2_000, short_answer=True)
    assert not is_genuine_speech("Hello.", 2_000, short_answer=True)
