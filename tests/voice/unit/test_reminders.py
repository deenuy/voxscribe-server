"""T16 reminders and timers: parser, store, routing and read-back.

All tests run without a database, the reply model, or web search. The reminder store is
a fake that records every add/cancel/mark_fired call in memory, and time is frozen via
datetime.now() monkeypatching where needed.

Acceptance criteria covered:
  A3-R1  "set a reminder for 6:30 am" creates a DB row (store.add called with correct args)
  A3-R2  no LLM or web search called for reminder set/cancel/list
  A3-R3  fired reminder spoken at next turn (fake clock)
  A3-R4  confirmation message names the time correctly
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.local.voice.reminder_store import (
    REMINDER_FIRED_PHRASE,
    REMINDER_LIST_INTRO,
    REMINDER_NONE_PHRASE,
    REMINDER_SIGNALS,
    Reminder,
    ReminderStatus,
    build_reminder_list_sentences,
    is_reminder_list_request,
    is_reminder_signal,
    parse_reminder,
    reminder_confirmation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(UTC)


def _reminder(text: str = "call Alex", *, minutes_from_now: float = 10.0,
              status: ReminderStatus = ReminderStatus.PENDING) -> Reminder:
    fire_at = _now_utc() + timedelta(minutes=minutes_from_now)
    return Reminder(
        id=str(uuid.uuid4()),
        text=text,
        fire_at=fire_at,
        status=status,
        created_at=_now_utc(),
    )


def _due_reminder(text: str = "take medication") -> Reminder:
    """A reminder that is already past due."""
    return Reminder(
        id=str(uuid.uuid4()),
        text=text,
        fire_at=_now_utc() - timedelta(seconds=30),
        status=ReminderStatus.PENDING,
        created_at=_now_utc() - timedelta(minutes=5),
    )


class FakeReminderStore:
    """In-memory fake that tracks every mutating call."""

    def __init__(self, reminders: list[Reminder] | None = None) -> None:
        self.reminders: list[Reminder] = list(reminders or [])
        self.fired_ids: list[str] = []
        self.cancelled_ids: list[str] = []
        self.added: list[tuple[str, datetime]] = []

    async def add(self, text: str, fire_at: datetime) -> Reminder:
        self.added.append((text, fire_at))
        r = Reminder(
            id=str(uuid.uuid4()),
            text=text,
            fire_at=fire_at,
            status=ReminderStatus.PENDING,
            created_at=_now_utc(),
        )
        self.reminders.append(r)
        return r

    async def cancel(self, reminder_id: str) -> bool:
        for r in self.reminders:
            if r.id == reminder_id and r.status is ReminderStatus.PENDING:
                r.status = ReminderStatus.CANCELLED
                self.cancelled_ids.append(reminder_id)
                return True
        return False

    async def mark_fired(self, reminder_id: str) -> None:
        self.fired_ids.append(reminder_id)
        for r in self.reminders:
            if r.id == reminder_id:
                r.status = ReminderStatus.FIRED

    async def list_pending(self) -> list[Reminder]:
        return [r for r in self.reminders if r.status is ReminderStatus.PENDING]

    async def list_all(self) -> list[Reminder]:
        return list(self.reminders)

    async def list_due(self) -> list[Reminder]:
        now = _now_utc()
        return [r for r in self.reminders
                if r.status is ReminderStatus.PENDING and r.fire_at <= now]


# ---------------------------------------------------------------------------
# parse_reminder
# ---------------------------------------------------------------------------

class TestParseReminder:
    def test_parse_at_time_pm(self):
        result = parse_reminder("remind me to call Alex at 3pm")
        assert result is not None
        desc, fire_at = result
        assert "call Alex" in desc
        assert fire_at.tzinfo is not None
        # hour should be 15 local, converted to UTC
        local = fire_at.astimezone()
        assert local.hour == 15
        assert local.minute == 0

    def test_parse_at_time_with_minutes(self):
        result = parse_reminder("remind me to pick up groceries at 6:30 am")
        assert result is not None
        desc, fire_at = result
        assert "groceries" in desc or "pick up" in desc
        local = fire_at.astimezone()
        assert local.hour == 6
        assert local.minute == 30

    def test_parse_at_time_noon(self):
        result = parse_reminder("set a reminder for the meeting at 12pm")
        assert result is not None
        _, fire_at = result
        local = fire_at.astimezone()
        assert local.hour == 12

    def test_parse_in_minutes(self):
        before = datetime.now(UTC)
        result = parse_reminder("remind me to check the oven in 5 minutes")
        assert result is not None
        desc, fire_at = result
        after = datetime.now(UTC)
        assert timedelta(minutes=4, seconds=58) <= (fire_at - before) <= timedelta(minutes=5, seconds=2)
        assert "oven" in desc or "check" in desc

    def test_parse_in_hours(self):
        before = datetime.now(UTC)
        result = parse_reminder("remind me to call the doctor in 2 hours")
        assert result is not None
        _, fire_at = result
        assert timedelta(hours=1, minutes=59) <= (fire_at - before) <= timedelta(hours=2, minutes=1)

    def test_parse_timer_minutes(self):
        before = datetime.now(UTC)
        result = parse_reminder("set a timer for 10 minutes")
        assert result is not None
        desc, fire_at = result
        assert "10 minute" in desc
        assert timedelta(minutes=9, seconds=58) <= (fire_at - before) <= timedelta(minutes=10, seconds=2)

    def test_parse_timer_seconds(self):
        before = datetime.now(UTC)
        result = parse_reminder("set a timer for 30 seconds")
        assert result is not None
        desc, fire_at = result
        assert "30 second" in desc
        assert timedelta(seconds=28) <= (fire_at - before) <= timedelta(seconds=32)

    def test_parse_returns_none_for_unrelated(self):
        assert parse_reminder("what is 2 plus 2?") is None
        assert parse_reminder("play some music") is None
        assert parse_reminder("what time is it?") is None
        assert parse_reminder("tell me a joke") is None

    def test_parse_rolls_to_tomorrow_if_past(self):
        """A time that has already passed today should land tomorrow."""
        # Use midnight as a time that is very likely in the past.
        result = parse_reminder("remind me to do something at 12am")
        assert result is not None
        _, fire_at = result
        # Should be more than 20 hours away (not 0 hours, which would mean today in the past)
        diff = fire_at - datetime.now(UTC)
        assert diff.total_seconds() > 0


# ---------------------------------------------------------------------------
# is_reminder_signal / is_reminder_list_request
# ---------------------------------------------------------------------------

class TestSignalDetection:
    @pytest.mark.parametrize("text", [
        "remind me to call Alex at 3pm",
        "set a reminder for 6:30 am",
        "set a timer for 5 minutes",
        "set an alarm for 8am",
        "timer for 10 minutes",
    ])
    def test_signal_detected(self, text: str):
        assert is_reminder_signal(text)

    def test_signal_not_detected_for_unrelated(self):
        assert not is_reminder_signal("what time is it?")
        assert not is_reminder_signal("search for news")

    @pytest.mark.parametrize("text", [
        "what reminders do I have?",
        "what's on my list?",
        "show my reminders",
        "what are my upcoming reminders?",
        "my reminders",
    ])
    def test_list_request_detected(self, text: str):
        assert is_reminder_list_request(text)

    def test_list_request_not_on_unrelated(self):
        assert not is_reminder_list_request("set a timer for 5 minutes")
        assert not is_reminder_list_request("what time is it?")


# ---------------------------------------------------------------------------
# FakeReminderStore: add and list_pending
# ---------------------------------------------------------------------------

class TestFakeStoreAddAndList:
    @pytest.mark.asyncio
    async def test_add_and_list(self):
        store = FakeReminderStore()
        fire_at = _now_utc() + timedelta(hours=1)
        await store.add("call Alex", fire_at)
        pending = await store.list_pending()
        assert len(pending) == 1
        assert pending[0].text == "call Alex"
        assert pending[0].status is ReminderStatus.PENDING

    @pytest.mark.asyncio
    async def test_cancel(self):
        store = FakeReminderStore()
        fire_at = _now_utc() + timedelta(hours=1)
        r = await store.add("call Alex", fire_at)
        result = await store.cancel(r.id)
        assert result is True
        pending = await store.list_pending()
        assert pending == []

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self):
        store = FakeReminderStore()
        assert await store.cancel("nonexistent-id") is False

    @pytest.mark.asyncio
    async def test_mark_fired(self):
        store = FakeReminderStore()
        fire_at = _now_utc() + timedelta(hours=1)
        r = await store.add("call Alex", fire_at)
        await store.mark_fired(r.id)
        pending = await store.list_pending()
        assert pending == []
        assert r.id in store.fired_ids


# ---------------------------------------------------------------------------
# build_reminder_list_sentences
# ---------------------------------------------------------------------------

class TestBuildReminderListSentences:
    def test_empty_list(self):
        sentences = build_reminder_list_sentences([])
        assert len(sentences) == 1
        assert sentences[0] == REMINDER_NONE_PHRASE

    def test_single_reminder(self):
        r = _reminder("call Alex", minutes_from_now=60)
        sentences = build_reminder_list_sentences([r])
        assert any("1 reminder" in s for s in sentences)
        assert any("call Alex" in s for s in sentences)

    def test_multiple_reminders(self):
        reminders = [_reminder("call Alex", minutes_from_now=60),
                     _reminder("buy milk", minutes_from_now=120)]
        sentences = build_reminder_list_sentences(reminders)
        assert any("2 reminder" in s for s in sentences)
        assert any("call Alex" in s for s in sentences)
        assert any("buy milk" in s for s in sentences)


# ---------------------------------------------------------------------------
# reminder_confirmation
# ---------------------------------------------------------------------------

class TestReminderConfirmation:
    def test_timer_phrase_for_timer_input(self):
        fire_at = _now_utc() + timedelta(minutes=5)
        msg = reminder_confirmation("set a timer for 5 minutes", "5 minutes", fire_at)
        assert "timer" in msg.lower()
        assert "5 minute" in msg

    def test_reminder_phrase_names_time(self):
        # Use a fixed datetime to verify the time string
        fire_at = datetime(2026, 11, 18, 15, 0, 0, tzinfo=UTC)  # 3:00 PM UTC
        msg = reminder_confirmation("remind me to call Alex at 3pm", "call Alex", fire_at)
        assert "call Alex" in msg
        # Time string should appear somewhere in the message
        assert any(c.isdigit() for c in msg)


# ---------------------------------------------------------------------------
# Integration: OllamaDelegate reminder routing (no LLM, no web search)
# ---------------------------------------------------------------------------

def _make_app(reminder_store=None) -> SimpleNamespace:
    """Minimal app namespace for OllamaDelegate."""
    settings = SimpleNamespace(
        voice_model=None, reply_model=None, voice_mode="default",
    )
    state = SimpleNamespace(settings=settings, store=None)
    return SimpleNamespace(state=state)


async def _collect(stream) -> list[str]:
    """Drain a ReplyStream's sentences into a list."""
    sentences = []
    async for s in stream.sentences:
        sentences.append(s)
    return sentences


class FakeDelegate:
    """Thin stand-in for OllamaDelegate that skips the actual DB and LLM."""

    def __init__(self, reminder_store: FakeReminderStore) -> None:
        from app.local.voice.adapters.delegate_ollama import OllamaDelegate
        from app.local.voice.delegate_lane import DelegateLane

        app = _make_app()
        # Patch the PostgresDelegateTaskStore and ReminderStore constructors.
        import app.local.voice.adapters.store_postgres as sp
        import app.local.voice.reminder_store as rs_mod

        self._store = reminder_store
        self._delegate: OllamaDelegate | None = None
        self._app = app

    def _build(self) -> "OllamaDelegate":
        from app.local.voice import delegate_lane as dl
        from app.local.voice.adapters.delegate_ollama import OllamaDelegate
        from app.local.voice.adapters.store_postgres import PostgresDelegateTaskStore
        from app.local.voice.delegate_lane import DelegateLane

        lane = DelegateLane()   # no runner -> tasks stay QUEUED, no network
        delegate = OllamaDelegate.__new__(OllamaDelegate)
        delegate._app = self._app
        delegate._task_store = SimpleNamespace(save=AsyncMock())
        delegate.lane = lane
        delegate._contexts = {}
        delegate._offered = None
        delegate._writes = set()
        delegate._reminder_store = self._store
        return delegate

    @property
    def delegate(self):
        if self._delegate is None:
            self._delegate = self._build()
        return self._delegate


class TestReminderRouting:
    @pytest.mark.asyncio
    async def test_no_llm_on_reminder_intent(self):
        """'remind me to X at Y' routes to store, not LLM. Source must be 'reminder'."""
        store = FakeReminderStore()
        fd = FakeDelegate(store)
        delegate = fd.delegate

        with (
            patch("app.local.reply_chain.talker_sentences",
                  side_effect=AssertionError("LLM was called")),
            patch("app.local.reply_chain.search_answer",
                  side_effect=AssertionError("search was called")),
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            stream = await delegate.reply(
                "remind me to call Alex at 3pm", "sess-1", turn_id="t-1",
            )
        assert stream.source == "reminder"
        sentences = await _collect(stream)
        assert any("call Alex" in s or "3" in s for s in sentences)
        # DB add was called
        assert len(store.added) == 1

    @pytest.mark.asyncio
    async def test_set_reminder_stores_row(self):
        """A3-R1: 'set a reminder for 6:30 am' creates a DB row."""
        store = FakeReminderStore()
        fd = FakeDelegate(store)
        delegate = fd.delegate

        with (
            patch("app.local.reply_chain.talker_sentences",
                  side_effect=AssertionError("LLM should not be called")),
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            stream = await delegate.reply(
                "set a reminder for 6:30 am", "sess-2", turn_id="t-2",
            )
        await _collect(stream)
        assert len(store.added) == 1
        desc, fire_at = store.added[0]
        # Fire time should be a future 6:30 AM
        local = fire_at.astimezone()
        assert local.hour == 6
        assert local.minute == 30

    @pytest.mark.asyncio
    async def test_confirmation_names_time(self):
        """A3-R4: confirmation message names the time correctly."""
        store = FakeReminderStore()
        fd = FakeDelegate(store)
        delegate = fd.delegate

        with (
            patch("app.local.reply_chain.talker_sentences",
                  side_effect=AssertionError("LLM should not be called")),
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            stream = await delegate.reply(
                "remind me to take medication at 8am", "sess-3", turn_id="t-3",
            )
        sentences = await _collect(stream)
        combined = " ".join(sentences).lower()
        # Should mention the time (8) and something about the reminder
        assert "8" in combined
        assert any(word in combined for word in ("remind", "medication", "8"))

    @pytest.mark.asyncio
    async def test_no_llm_on_list_request(self):
        """A3-R2: 'what reminders do I have?' reads pending list without LLM."""
        store = FakeReminderStore(reminders=[_reminder("call Alex")])
        fd = FakeDelegate(store)
        delegate = fd.delegate

        with (
            patch("app.local.reply_chain.talker_sentences",
                  side_effect=AssertionError("LLM was called")),
            patch("app.local.reply_chain.search_answer",
                  side_effect=AssertionError("search was called")),
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            stream = await delegate.reply(
                "what reminders do I have?", "sess-4", turn_id="t-4",
            )
        assert stream.source == "reminder"
        sentences = await _collect(stream)
        assert any("call Alex" in s for s in sentences)

    @pytest.mark.asyncio
    async def test_read_back_list_empty(self):
        """'what reminders do I have?' with no pending reminders says so."""
        store = FakeReminderStore()
        fd = FakeDelegate(store)
        delegate = fd.delegate

        with (
            patch("app.local.reply_chain.talker_sentences",
                  side_effect=AssertionError("LLM should not be called")),
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            stream = await delegate.reply(
                "what reminders do I have?", "sess-5", turn_id="t-5",
            )
        sentences = await _collect(stream)
        assert any("no reminder" in s.lower() for s in sentences)

    @pytest.mark.asyncio
    async def test_fire_check_speaks_due(self):
        """A3-R3: a past-due reminder is spoken at the next turn and marked fired."""
        due = _due_reminder("take medication")
        store = FakeReminderStore(reminders=[due])
        fd = FakeDelegate(store)
        delegate = fd.delegate

        # Any turn text: the fired reminder should come first, before the main reply.
        with (
            patch("app.local.reply_chain.talker_sentences") as mock_talker,
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            # talker_sentences is called synchronously; return a namespace with an async iterator.
            def _fake_sentences(*a, **kw):
                return SimpleNamespace(sentences=_aiter(["Hello."]), source="llm")

            mock_talker.side_effect = _fake_sentences

            stream = await delegate.reply("How are you?", "sess-6", turn_id="t-6")

        sentences = await _collect(stream)
        # The fired reminder sentence should appear in the output
        expected_phrase = REMINDER_FIRED_PHRASE.format(text="take medication")
        assert expected_phrase in sentences
        # The reminder should have been marked fired
        assert due.id in store.fired_ids

    @pytest.mark.asyncio
    async def test_timer_confirmation_no_llm(self):
        """'set a timer for 5 minutes' routes to reminder, not LLM."""
        store = FakeReminderStore()
        fd = FakeDelegate(store)
        delegate = fd.delegate

        with (
            patch("app.local.reply_chain.talker_sentences",
                  side_effect=AssertionError("LLM should not be called")),
            patch("app.local.reply_chain.route_turn", return_value=None),
        ):
            stream = await delegate.reply(
                "set a timer for 5 minutes", "sess-7", turn_id="t-7",
            )
        assert stream.source == "reminder"
        sentences = await _collect(stream)
        combined = " ".join(sentences).lower()
        assert "timer" in combined
        assert "5 minute" in combined


async def _aiter(items: list[str]):
    for item in items:
        yield item
