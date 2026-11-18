"""T07 guardrails.

G1: POST /v1/wake/trigger is reachable only with --debug-trigger; --no-auth alone gives 403.
G2: a sentence claiming a completed action (reminder, alarm, timer, calendar) with no tool row
    in the session is replaced with a fixed phrase, and one gate decision row says why.

No real LLM, TTS, microphone or database: the reply chain runs against a fake pool and a
scripted token stream.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.local import gate, wake_loop
from app.local.engine import build_parser
from app.local.reply_chain import run_reply
from app.local.routers import wake as wake_router


# ── G1: debug trigger behind its own flag ───────────────────────────────────


class _FakeAudioEngine:
    def subscribe(self):
        return object()

    def unsubscribe(self, _q):
        pass


def _wake_app(*, auth_disabled: bool, debug_trigger: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(wake_router.router)
    app.state.auth_disabled = auth_disabled
    app.state.debug_trigger = debug_trigger
    app.state.audio_engine = _FakeAudioEngine()
    return app


async def _post_trigger(app: FastAPI) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/wake/trigger")


@pytest.fixture
def armed_wake_state():
    saved = dict(wake_loop._state)
    wake_loop.set_state(wake_loop.WakeState.ARMED)
    wake_loop._state["meeting_id"] = None
    yield
    wake_loop._state.clear()
    wake_loop._state.update(saved)


def test_debug_trigger_flag_defaults_off():
    # The normal launch command: --no-auth --wake, no --debug-trigger.
    args = build_parser().parse_args(
        ["--port", "0", "--providers", "local", "--wake", "--no-auth"]
    )
    assert args.no_auth is True
    assert args.debug_trigger is False
    assert build_parser().parse_args(["--debug-trigger"]).debug_trigger is True


async def test_trigger_requires_debug_flag(armed_wake_state):
    app = _wake_app(auth_disabled=True, debug_trigger=False)
    resp = await _post_trigger(app)
    assert resp.status_code == 403
    assert "--debug-trigger" in resp.json()["detail"]


async def test_trigger_forbidden_when_flag_state_missing(armed_wake_state):
    app = _wake_app(auth_disabled=True, debug_trigger=False)
    del app.state.debug_trigger
    assert (await _post_trigger(app)).status_code == 403


async def test_trigger_allowed_with_debug_flag(armed_wake_state, monkeypatch):
    started: list[float] = []

    async def fake_run_session(_app, _audio_q, *, confidence, np):
        started.append(confidence)
        wake_loop._state["meeting_id"] = "m-trigger-test"

    monkeypatch.setattr(wake_loop, "_run_session", fake_run_session)
    app = _wake_app(auth_disabled=True, debug_trigger=True)
    resp = await _post_trigger(app)
    assert resp.status_code == 202
    body = resp.json()
    assert body["triggered"] is True
    assert body["meeting_id"] == "m-trigger-test"
    assert started == [1.0]


async def test_created_app_without_flag_returns_403(throwaway_schema, armed_wake_state):
    """A real app built with auth disabled and no debug_trigger refuses the trigger."""
    from app.local.app import create_app
    from app.local.config import Config

    settings = Config(
        database_url="postgresql://localhost:5432/ideascribe_dev",
        data_dir="/tmp/voxscribe_test",
        providers="fake",
    )
    app = await create_app(settings=settings, auth_disabled=True, schema=throwaway_schema)
    try:
        assert app.state.debug_trigger is False
        assert (await _post_trigger(app)).status_code == 403
    finally:
        await app.state.store.close()


# ── G2: no false action claims ───────────────────────────────────────────────

# (prompt, what the reply model says without a tool) pairs, modelled on ROOT-CAUSE Defect F.
ACTION_PROMPTS: list[tuple[str, str]] = [
    ("Set a reminder for 6:30am tomorrow",
     "I've added a reminder for 6:30 am tomorrow. Your morning alarm is set."),
    ("Add an alarm for 7am",
     "Sure! I've set an alarm for 7 am."),
    ("Set a timer for ten minutes",
     "Okay, your timer is set for ten minutes."),
    ("Remind me to take my pills at noon",
     "Of course, I'll remind you to take your pills at noon."),
    ("Wake me up at six tomorrow",
     "Your alarm has been set for 6 am. Sleep well."),
    ("Put my dentist appointment on the calendar for Friday",
     "I've scheduled your dentist appointment for Friday."),
    ("Remind me to call my daughter tonight",
     "I have added a reminder to call your daughter tonight."),
    ("Start a timer for the pasta",
     "Got it, I'll set a timer for the pasta."),
    ("Save a reminder to water the plants",
     "I’ve saved a reminder to water the plants."),
    ("Create an alarm for 5:45 tomorrow",
     "I've created an alarm for 5:45 tomorrow morning."),
]

_COMPLETION_LANGUAGE = ("i've added", "i've set", "i've scheduled", "i've saved", "i've created",
                        "is set", "has been set", "i'll remind", "i'll set", "i have added")


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _FakePool:
    """Records reply-chain writes; answers the tool_calls lookup from `tools`."""

    def __init__(self, tools: list[str] | None = None):
        self.tools = tools or []
        self.gate_rows: dict[str, dict] = {}
        self.tool_lookups = 0

    @asynccontextmanager
    async def connection(self):
        yield self

    async def commit(self):
        pass

    async def execute(self, sql: str, params=()):
        if sql.startswith("INSERT INTO gate_decisions"):
            self.gate_rows[params[0]] = {"decision": params[5], "category": params[6]}
        elif sql.startswith("UPDATE gate_decisions"):
            self.gate_rows[params[3]].update(decision=params[0], category=params[1])
        elif "FROM tool_calls" in sql:
            self.tool_lookups += 1
            return _FakeCursor([(t,) for t in self.tools])
        return _FakeCursor([])


async def _tokens(text: str):
    for word in text.split(" "):
        yield word + " "


async def _run(prompt: str, model_reply: str, pool: _FakePool) -> str:
    app = SimpleNamespace(state=SimpleNamespace(store=SimpleNamespace(pool=pool)))
    return await run_reply(
        app, meeting_id="m-g2", user_text=prompt, turn_end_ms=0,
        pre_stream=_tokens(model_reply),
    )


@pytest.fixture(autouse=True)
def _behaviour_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_BEHAVIOUR_LOG_DIR", tmp_path)


def test_keywords_are_a_named_tuple():
    assert isinstance(gate.ACTION_CLAIM_KEYWORDS, tuple)
    assert all(isinstance(k, str) for k in gate.ACTION_CLAIM_KEYWORDS)


def test_replacement_phrase_is_not_itself_a_claim():
    assert gate.find_action_claim(gate.ACTION_CLAIM_PHRASE) is None
    assert gate.check_action_claim(gate.ACTION_CLAIM_PHRASE, []) == ("pass", "ok")


@pytest.mark.parametrize("_prompt,model_reply", ACTION_PROMPTS)
def test_rule_replaces_each_claim_sentence(_prompt, model_reply, caplog):
    first_sentence = model_reply.split(". ")[0].split("! ")[-1]
    with caplog.at_level(logging.INFO, logger="app.local.gate"):
        decision, reason = gate.check_action_claim(first_sentence, session_tools=[])
    assert (decision, reason) == ("replace", "no_tool_row")
    blocked = [r.getMessage() for r in caplog.records if "blocked action claim" in r.getMessage()]
    assert len(blocked) == 1
    assert "reason=no_tool_row" in blocked[0]


@pytest.mark.parametrize("sentence", [
    "It's 3:03 am.",
    "Good morning, it's a sunny day in Montreal.",
    "Photosynthesis turns sunlight into energy.",
    "I can't set reminders or alarms yet, but I can help answer questions.",
    "The settings menu is on the left.",
])
def test_ordinary_sentences_pass(sentence):
    assert gate.check_action_claim(sentence, session_tools=[]) == ("pass", "ok")


def test_claim_backed_by_matching_tool_row_passes():
    sentence = "I've set an alarm for 7 am."
    assert gate.check_action_claim(sentence, ["alarm_set"]) == ("pass", "tool_row")


def test_web_search_row_does_not_back_a_claim():
    sentence = "I've added a reminder for 6:30 am tomorrow."
    assert gate.check_action_claim(sentence, ["web_search"]) == ("replace", "no_tool_row")


def test_tool_row_of_another_type_does_not_back_a_claim():
    sentence = "I've set an alarm for 7 am."
    assert gate.check_action_claim(sentence, ["reminder_create"]) == ("replace", "no_tool_row")


async def test_ten_action_prompts_give_zero_completion_claims(caplog):
    """Acceptance: 10 prompts, 0 completion claims spoken, one no_tool_row decision row each."""
    spoken_claims = 0
    for prompt, model_reply in ACTION_PROMPTS:
        pool = _FakePool()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="app.local.gate"):
            reply = await _run(prompt, model_reply, pool)

        lowered = reply.lower().replace("’", "'")
        if any(c in lowered for c in _COMPLETION_LANGUAGE):
            spoken_claims += 1
        assert gate.ACTION_CLAIM_PHRASE in reply, prompt

        assert len(pool.gate_rows) == 1, prompt
        (row,) = pool.gate_rows.values()
        assert row == {"decision": "replace", "category": "no_tool_row"}, prompt

        blocked = [r for r in caplog.records if "blocked action claim" in r.getMessage()]
        assert len(blocked) == 1, prompt

    assert spoken_claims == 0


async def test_ordinary_reply_passes_without_tool_lookup():
    pool = _FakePool()
    reply = await _run("What is photosynthesis?",
                       "Plants turn sunlight into energy. It happens in the leaves.", pool)
    assert reply == "Plants turn sunlight into energy. It happens in the leaves."
    (row,) = pool.gate_rows.values()
    assert row == {"decision": "pass", "category": "ok"}
    assert pool.tool_lookups == 0


async def test_backed_claim_is_spoken_unchanged():
    pool = _FakePool(tools=["reminder_create"])
    reply = await _run("Remind me at noon", "I'll remind you at noon.", pool)
    assert reply == "I'll remind you at noon."
    (row,) = pool.gate_rows.values()
    assert row == {"decision": "pass", "category": "ok"}
    assert pool.tool_lookups == 1
