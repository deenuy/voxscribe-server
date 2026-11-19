"""PL41: network tools obey the product mode, the profile and the privacy mode (defect D-4).

The decision is one pure function; the voice path consults it before the delegate lane and again
before the search runs; the Google import consults it before any request. Every request that would
leave the device goes through a counting fake transport, so "0 requests" is measured, not assumed.
No database, no model, no network.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest

from app.local import gate, llm_adapter
from app.local.tools import network, web_search
from app.local.voice import delegate_lane as dl
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import VoiceConfig, VoiceHost
from tests.voice.unit.fakes import fake_ports
from tests.voice.unit.tool_fakes import CountingTransport, ToolPool, load_profile, make_app

WEATHER = "search the web for the weather"
SUMMARY = "It is sunny with a high of twenty."


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_BEHAVIOUR_LOG_DIR", tmp_path)


@pytest.fixture
def transport(monkeypatch) -> CountingTransport:
    fake = CountingTransport()
    monkeypatch.setattr(web_search, "TRANSPORT", fake)
    return fake


@pytest.fixture
def model(monkeypatch) -> list[str]:
    """The reply model, scripted: it only summarises search results here."""
    calls: list[str] = []

    async def stream_reply(user_text, *, model=None, system_prompt=None, history=None):
        calls.append(user_text)
        yield SUMMARY

    monkeypatch.setattr(llm_adapter, "stream_reply", stream_reply)
    return calls


# ── the rule ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["senior_care", "classroom"])
@pytest.mark.parametrize("profile_allowed", [True, False])
@pytest.mark.parametrize("privacy", ["connectors", "offline"])
def test_restricted_modes_are_refused_whatever_the_profile_or_privacy(mode, profile_allowed,
                                                                       privacy):
    decision = network.decide(mode=mode, profile_allowed=profile_allowed, privacy_mode=privacy)
    assert not decision.allowed and decision.reason == network.REASON_MODE


@pytest.mark.parametrize("mode", ["personal", "ideascribe"])
def test_open_modes_need_the_profile_and_connectors(mode):
    def decided(profile_allowed, privacy):
        d = network.decide(mode=mode, profile_allowed=profile_allowed, privacy_mode=privacy)
        return d.allowed, d.reason

    assert decided(True, "connectors") == (True, network.REASON_OK)
    assert decided(True, "offline") == (False, network.REASON_PRIVACY)
    assert decided(True, "anything else") == (False, network.REASON_PRIVACY)
    assert decided(False, "connectors") == (False, network.REASON_PROFILE)


def test_every_refusal_has_a_scripted_line_that_the_gate_allows():
    for reason, line in network.REFUSAL_LINES.items():
        assert line and gate.find_action_claim(line) is None, reason
        for mode in ("personal", "senior_care", "classroom"):
            verdict = CorePolicyGate(session_policy(make_app(ToolPool(),
                                                             profile=load_profile(mode))))
            for sentence in line.split(". "):
                assert verdict.check(sentence).decision == "pass", (reason, mode)


async def test_the_stored_privacy_setting_is_read_not_the_environment():
    """The environment says connectors, the tenant's stored setting says offline: refused. And
    the other way round: allowed. The stored setting is the one authority."""
    env_on = make_app(ToolPool(privacy="offline"), profile=load_profile("personal"),
                      privacy_mode="connectors")
    decision = await network.network_decision(env_on)
    assert (decision.allowed, decision.reason, decision.privacy_mode) == (
        False, network.REASON_PRIVACY, "offline")

    env_off = make_app(ToolPool(privacy="connectors"), profile=load_profile("personal"),
                       privacy_mode="offline")
    decision = await network.network_decision(env_off)
    assert (decision.allowed, decision.privacy_mode) == (True, "connectors")


async def test_no_stored_setting_and_an_unreadable_one_are_offline():
    assert not (await network.network_decision(
        make_app(ToolPool(privacy=None), profile=load_profile("personal")))).allowed

    class Broken(ToolPool):
        async def execute(self, sql, params=()):
            raise OSError("database down")

    assert (await network.network_decision(
        make_app(Broken(), profile=load_profile("personal")))).reason == network.REASON_PRIVACY


async def test_a_profile_with_the_network_off_is_refused_without_reading_privacy():
    profile = load_profile("personal")
    profile = dataclasses.replace(profile, network=dataclasses.replace(profile.network,
                                                                       allowed=False))
    pool = ToolPool(privacy="connectors")
    decision = await network.network_decision(make_app(pool, profile=profile))
    assert decision.reason == network.REASON_PROFILE and pool.settings_reads == 0


async def test_a_restricted_mode_set_only_by_the_legacy_voice_mode_is_refused():
    app = make_app(ToolPool(privacy="connectors"), profile=load_profile("personal"),
                   voice_mode="seniorcare")
    assert (await network.network_decision(app)).reason == network.REASON_MODE


# ── the voice path ───────────────────────────────────────────────────────────


async def _turn(app, text: str, turn_n: int = 1) -> tuple[str, OllamaDelegate]:
    """One turn through the voice host's reply stage with the real gate, store and delegate."""
    delegate = OllamaDelegate(app)
    ports = fake_ports(gate=CorePolicyGate(session_policy(app)),
                       store=PostgresSessionStore(app), delegate=delegate)
    host = VoiceHost(VoiceConfig(), ports=ports)
    reply = await host.reply_turn("s1", turn_n, text, turn_end_ms=0)
    for task in delegate.lane.active():
        await asyncio.wait_for(delegate.lane.wait(task.task_id), 5)
    await delegate.drain()
    return reply, delegate


@pytest.mark.parametrize("mode", ["senior_care", "classroom"])
async def test_a_search_in_a_restricted_mode_sends_nothing_and_says_why(mode, transport, model,
                                                                         caplog):
    pool = ToolPool(privacy="connectors")
    app = make_app(pool, profile=load_profile(mode))
    with caplog.at_level(logging.DEBUG):
        reply, delegate = await _turn(app, WEATHER)

    assert transport.requests == []
    assert reply == network.REFUSAL_LINES[network.REASON_MODE]
    refused = [r for r in pool.tool_calls if r["status"] == "refused"]
    assert len(pool.tool_calls) == 1 and len(refused) == 1
    assert refused[0]["error_code"] == network.REASON_MODE
    assert refused[0]["meeting_id"] == "s1"
    assert delegate.lane.latest_open() is None and model == []
    assert "weather" not in caplog.text.lower()


async def test_personal_with_privacy_offline_sends_nothing(transport, model):
    pool = ToolPool(privacy="offline")
    reply, _ = await _turn(make_app(pool, profile=load_profile("personal")), WEATHER)
    assert transport.requests == []
    assert reply == network.REFUSAL_LINES[network.REASON_PRIVACY]
    assert [r["error_code"] for r in pool.tool_calls] == [network.REASON_PRIVACY]


async def test_personal_with_connectors_sends_exactly_one_request(transport, model):
    pool = ToolPool(privacy="connectors")
    reply, delegate = await _turn(make_app(pool, profile=load_profile("personal")), WEATHER)
    assert reply.startswith(dl.DELEGATING_PHRASE)   # the fake answers within the turn
    assert len(transport.requests) == 1
    assert [r["status"] for r in pool.tool_calls] == ["executed"]
    assert model == [WEATHER]                       # the summary of the results


async def test_privacy_turned_off_after_submission_is_obeyed_when_the_search_runs(transport,
                                                                                   model):
    """The decision is taken again just before the search runs."""
    pool = ToolPool(privacy="connectors")
    app = make_app(pool, profile=load_profile("personal"))
    lane = dl.DelegateLane()                        # no runner: the task waits until we run it
    delegate = OllamaDelegate(app, lane=lane)
    stream = await delegate.reply(WEATHER, "s1", turn_id="t1")
    assert [s async for s in stream.sentences] == [dl.DELEGATING_PHRASE]
    task = lane.latest_open()

    pool.privacy = "offline"
    result = await delegate._think(task)
    assert result == network.REFUSAL_LINES[network.REASON_PRIVACY]
    assert transport.requests == []
    assert [(r["status"], r["error_code"]) for r in pool.tool_calls] == [
        ("refused", network.REASON_PRIVACY)]
