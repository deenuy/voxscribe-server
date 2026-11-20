"""A reply tells the app when it is being spoken, and with which words.

`wake.reply` had two phases: `thinking` when the turn closed and `done` once the reply had been
generated and played. An app could not tell thinking from speaking, and had no words to show
until the speech was over. The host now also publishes `speaking` each time a reply clause is
handed to the speaker, with the reply text handed over so far.

Host level with every port a fake (fakes.py). What is shown: the order per turn is thinking,
speaking (one per clause, the text growing), done; every payload fits the event schema; text the
gate refused is never in a `speaking` event; an interrupted reply's `speaking` text covers what
was heard; a listener that fails does not cost a clause its audio.
"""
from __future__ import annotations

import asyncio

from app.local import event_schema
from app.local.voice.barge_in import BARGE_IN_ENERGY_THRESHOLD
from app.local.voice.host import VoiceConfig, VoiceHost

from .fakes import (
    SPEECH_MS_PER_CHAR,
    Burst,
    EndlessQueue,
    FakeDelegate,
    FakeGate,
    FakeRecognizer,
    fake_ports,
    sim_ports,
)

TURNS = ["What is one?", "And two?"]
REPLIES = [["One is a number.", "It comes first."], ["Two comes next."]]
CLAUSES = ["Hello,", "I found", "three results for your search."]
REFUSED = ["Sure,", "here is how to pick a lock."]
LOUD = BARGE_IN_ENERGY_THRESHOLD * 3          # a person talking over the reply
INTERJECTION_MS = 1200
TIMEOUT_S = 10


def _collector():
    events: list[tuple[str, dict]] = []
    return events, lambda kind, data: events.append((kind, dict(data)))


def _replies(events) -> list[dict]:
    return [data for kind, data in events if kind == "wake.reply"]


async def _reply(ports, capture=None, *, publish):
    host = VoiceHost(VoiceConfig(), ports=ports, publish=publish)
    return await asyncio.wait_for(
        host._reply("s1", 1, "find me a bakery", audio_q=capture, turn_end_ms=0), TIMEOUT_S)


async def test_each_turn_says_thinking_then_speaking_per_clause_then_done():
    events, publish = _collector()
    ports = fake_ports(recognizer=FakeRecognizer(TURNS), delegate=FakeDelegate(REPLIES))
    host = VoiceHost(VoiceConfig(), ports=ports, publish=publish)
    assert host.arm()
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                           TIMEOUT_S)

    replies = _replies(events)
    for data in replies:
        event_schema.validate("wake.reply", data)             # the new phase fits the schema
    turns = sorted({d["turn"] for d in replies})
    assert len(turns) == len(TURNS)
    spoken = list(ports.talker.synthesized)                   # the clauses, in spoken order
    for turn, reply in zip(turns, REPLIES, strict=True):
        mine = [d for d in replies if d["turn"] == turn]
        phases = [d["phase"] for d in mine]
        speaking = [d["text"] for d in mine if d["phase"] == "speaking"]
        assert phases == ["thinking", *["speaking"] * len(speaking), "done"]
        assert mine[0]["text"] == ""                          # thinking carries no words
        # One per clause, each adding its clause to the words before it.
        clauses, spoken = spoken[:len(speaking)], spoken[len(speaking):]
        assert speaking == [" ".join(clauses[:n + 1]) for n in range(len(clauses))]
        assert speaking[-1] == mine[-1]["text"] == " ".join(reply)
        assert {d["userText"] for d in mine} == {mine[0]["userText"]}
    assert spoken == []                                       # every clause was announced


async def test_text_the_gate_refused_is_never_in_a_speaking_event():
    events, publish = _collector()
    ports = fake_ports(delegate=FakeDelegate([REFUSED]), gate=FakeGate(refuse="pick a lock"))
    result = await _reply(ports, publish=publish)

    speaking = [d["text"] for d in _replies(events) if d["phase"] == "speaking"]
    assert speaking and all("lock" not in text for text in speaking)
    assert "lock" not in result.text
    # The gate's own line is what was spoken last, so it is what the last event ends with.
    assert speaking[-1].endswith(result.text)


async def test_an_interrupted_reply_announced_every_clause_that_was_heard():
    events, publish = _collector()
    first_ms = len(CLAUSES[0]) * SPEECH_MS_PER_CHAR
    burst = Burst(at_playback_ms=first_ms + 1, dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=FakeDelegate([CLAUSES]))
    result = await _reply(ports, capture, publish=publish)

    assert result.interruption is not None
    speaking = [d["text"] for d in _replies(events) if d["phase"] == "speaking"]
    assert speaking
    # What was heard to its end is where the announced words begin: the app can show the
    # words as they are spoken and settle on the heard text when `done` arrives.
    assert speaking[-1].startswith(result.text)


async def test_a_failing_listener_does_not_cost_a_clause_its_audio():
    def publish(kind, data):
        raise RuntimeError("event sink down")

    ports = fake_ports(delegate=FakeDelegate([CLAUSES]))
    result = await _reply(ports, publish=publish)

    assert len(ports.audio_out.played) == len(ports.talker.synthesized) > 0
    assert result.text == " ".join(CLAUSES)
