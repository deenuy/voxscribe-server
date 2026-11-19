"""How a conversation ends (PL61): natural closings with a spoken farewell, "stop" over the
assistant, unusable turns that end nothing, and waits that only recognised speech extends."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.local.config import Config
from app.local.voice.expression import FAREWELLS, NATURAL, PLAIN, PLAIN_FAREWELL
from app.local.voice.host import (
    CHUNK_FRAMES,
    SAMPLE_RATE,
    STOP_HINT,
    VoiceConfig,
    VoiceHost,
    is_closing,
    is_stop_speaking,
)
from tests.voice.unit.fakes import (
    CHUNK_MS,
    Burst,
    FakeDelegate,
    FakeRecognizer,
    fake_ports,
    sim_ports,
)
from tests.voice.unit.test_barge_in import CLAUSES, LOUD, SESSION_INTERJECTION_MS, _ms

SESSION_TIMEOUT_S = 20
QUESTION = "What is one plus one?"
LATER = "And what about two plus two?"


class Room:
    """Capture that never runs dry, at `level`, counting what was read."""

    def __init__(self, level: float = 0.0) -> None:
        self.level = level
        self.reads = 0

    async def get(self):
        await asyncio.sleep(0)
        self.reads += 1
        return np.full(CHUNK_FRAMES, self.level, dtype=np.float32)

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def empty(self) -> bool:
        return True


def _host(turns, *, expressiveness=PLAIN, **config):
    events: list[tuple[str, dict]] = []
    ports = fake_ports(recognizer=FakeRecognizer(turns), delegate=FakeDelegate())
    host = VoiceHost(VoiceConfig(expressiveness=expressiveness, **config), ports=ports,
                     publish=lambda kind, data: events.append((kind, data)))
    return host, ports, events


async def _session(host, room=None):
    room = room or Room()
    await asyncio.wait_for(host.run_session(room, score=0.9), SESSION_TIMEOUT_S)
    return room


def _captions(events) -> list[str]:
    return [d["segment"]["text"] for k, d in events if k == "wake.transcript"]


def _farewells_said(ports) -> list[str]:
    said = " ".join(ports.talker.synthesized)
    return [line for line in (*FAREWELLS.values(), *PLAIN_FAREWELL.values()) if line in said]


# ── what closes a conversation ───────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "That's all.", "that's all", "That's it, thanks.", "That's it, thank you.", "Okay, I'm done.",
    "Goodbye!", "Bye.", "Bye-bye.", "Stop.", "Never mind, bye.", "Okay, thanks, that's all.",
    "Great, bye!", "Thank you so much, that's all for now.", "Thanks, bye.",
    "thank you vox", "Okay, goodbye vox",
])
def test_natural_closings_end_the_conversation(text):
    assert is_closing(text, "hey vox")


@pytest.mark.parametrize("text", [
    "I said goodbye to my sister", "Remind me to say goodbye to grandma",
    "That's all I wanted to ask about the weather in town today", "don't stop",
    "stop the timer", "That's it!", "never mind", "see you", "goodbyes are hard",
    "What is one plus one?",
])
def test_a_closing_word_inside_a_longer_sentence_ends_nothing(text):
    assert not is_closing(text, "hey vox")


def test_the_persons_own_end_phrases_still_count():
    assert is_closing("Okay, see you later.", "hey vox", ("see you later",))
    assert not is_closing("Okay, see you later.", "hey vox")


@pytest.mark.parametrize("text", ["Stop.", "stop", "Wait.", "Hold on.", "Okay, stop.",
                                  "Hang on.", "Stop, stop."])
def test_stop_wait_and_hold_on_stop_the_speech(text):
    assert is_stop_speaking(text)


@pytest.mark.parametrize("text", ["stop the timer", "wait until tomorrow to remind me",
                                  "what time is it"])
def test_other_turns_do_not_stop_the_speech(text):
    assert not is_stop_speaking(text)


# ── the farewell ─────────────────────────────────────────────────────────────


async def test_thats_all_thanks_gets_one_farewell_and_closes():
    host, ports, _events = _host([QUESTION, "That's all, thanks."], expressiveness=NATURAL)
    await _session(host)
    assert ports.delegate.calls == [QUESTION]                 # the closing went to no model
    assert _farewells_said(ports) == [FAREWELLS["anytime"]]
    assert ports.talker.synthesized[-1] == FAREWELLS["anytime"]
    assert ports.store.closed == ["s1"] and ports.store.discarded == []
    assert [o.text for _, o in ports.store.turns] == ["Answer 1.", FAREWELLS["anytime"]]
    assert FAREWELLS["anytime"] in ports.gate.checked         # through the gate


async def test_plain_says_one_plain_line():
    host, ports, _ = _host([QUESTION, "Goodbye."])
    await _session(host)
    assert _farewells_said(ports) == [PLAIN_FAREWELL["goodbye"]]


async def test_two_sessions_running_do_not_use_the_same_farewell():
    ports = fake_ports(recognizer=FakeRecognizer([QUESTION, "Bye.", QUESTION, "Bye.", "Bye."]))
    host = VoiceHost(VoiceConfig(expressiveness=NATURAL), ports=ports)
    for _ in range(3):
        await _session(host)
    said = [t for t in ports.talker.synthesized if t in FAREWELLS.values()]
    assert len(said) == 3 and said[0] != said[1] != said[2]


async def test_a_farewell_the_gate_refuses_is_not_said():
    from tests.voice.unit.fakes import FakeGate

    host, ports, _ = _host([QUESTION, "That's all."], expressiveness=NATURAL)
    ports.gate = FakeGate(refuse=FAREWELLS["anytime"])
    await _session(host)
    assert FAREWELLS["anytime"] not in ports.talker.synthesized
    assert ports.store.closed == ["s1"]


async def test_an_idle_end_says_nothing():
    host, ports, _ = _host([QUESTION], conv_idle_s=1.0)
    await _session(host)
    assert _farewells_said(ports) == [] and ports.store.closed == ["s1"]


async def test_stop_while_listening_ends_the_session_with_a_farewell():
    host, ports, _ = _host([QUESTION, "Stop."])
    await _session(host)
    assert ports.delegate.calls == [QUESTION]
    assert _farewells_said(ports) == [PLAIN_FAREWELL["goodbye"]]


# ── "stop" over the assistant ────────────────────────────────────────────────


async def test_stop_during_a_reply_stops_speech_and_keeps_the_session():
    clauses = CLAUSES[:3]
    burst = Burst(at_playback_ms=_ms(clauses[0]) + _ms(clauses[1]) / 2,
                  dur_ms=SESSION_INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports(
        [burst], delegate=FakeDelegate([clauses, ["It opens at nine."]]),
        recognizer=FakeRecognizer(["find me a bakery", "Stop.", "when does it open?",
                                   STOP_HINT]))
    host = VoiceHost(VoiceConfig(), ports=ports)
    await asyncio.wait_for(host.run_session(capture, score=0.9), SESSION_TIMEOUT_S)
    assert len(ports.audio_out.stops) == 1                     # the speech stopped
    assert ports.delegate.calls == ["find me a bakery", "when does it open?"]   # listening on
    texts = [o.text for _, o in ports.store.turns]
    assert texts[0] == clauses[0] and texts[1] == "It opens at nine."
    assert ports.store.closed == ["s1"]


# ── a turn the engine cannot use ─────────────────────────────────────────────


async def test_a_one_word_turn_in_turn_three_does_not_end_the_session():
    host, ports, _ = _host([QUESTION, LATER, "Why?", "And three plus three?", STOP_HINT])
    await _session(host)
    assert ports.delegate.calls == [QUESTION, LATER, "And three plus three?"]
    assert [n for n, _ in ports.store.turns] == [1, 2, 3, 4]     # the one word took no number
    assert ports.store.closed == ["s1"]


@pytest.mark.parametrize("noise", ["", "Beep.", "Um."])
async def test_noise_in_turn_one_is_ignored_and_the_question_after_it_answered(noise):
    host, ports, events = _host([noise, QUESTION, STOP_HINT])
    await _session(host)
    assert ports.delegate.calls == [QUESTION]
    assert QUESTION in _captions(events)
    assert all(c in (QUESTION, STOP_HINT) for c in _captions(events))   # the noise never shown


async def test_a_session_with_nothing_usable_closes_as_silent_and_shows_nothing():
    """The first-turn wait runs out with only noise heard: nothing is kept, and since nothing is
    kept no caption of it was ever published."""
    host, ports, events = _host(["Beep.", "Um.", ""])
    room = await _session(host)
    assert ports.delegate.calls == [] and ports.store.turns == []
    assert ports.store.discarded == ["s1"] and ports.store.segments == []
    assert _captions(events) == []
    listened_s = room.reads * CHUNK_MS / 1000
    assert listened_s == pytest.approx(8.0, abs=0.5)           # the first-turn wait, unchanged


# ── the waits ────────────────────────────────────────────────────────────────


async def test_loud_non_speech_for_longer_than_the_wait_ends_the_session():
    host, ports, _ = _host([QUESTION], conv_idle_s=2.0)
    room = await _session(host, Room(level=0.2))                # loud, and nothing recognised
    assert ports.delegate.calls == [QUESTION] and ports.store.closed == ["s1"]
    listened_s = room.reads * CHUNK_MS / 1000
    assert listened_s < 2.0 + 2.0                               # turn 1 and its reply, then 2 s


async def test_a_session_with_no_recognised_speech_ends_at_the_limit():
    """Even when the wait is longer: the limit holds however loud the room."""
    host, ports, _ = _host([QUESTION], conv_idle_s=600.0, no_speech_limit_s=3.0)
    room = await _session(host, Room(level=0.2))
    assert ports.store.closed == ["s1"]
    assert room.reads * CHUNK_MS / 1000 < 3.0 + 2.0


def test_the_limit_is_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Config().voice_no_speech_limit_s == 120.0
    monkeypatch.setenv("VOXSCRIBE_VOICE_NO_SPEECH_LIMIT_S", "90")
    assert VoiceConfig.from_settings(Config.from_env()).no_speech_limit_s == 90.0
    monkeypatch.setenv("VOXSCRIBE_VOICE_NO_SPEECH_LIMIT_S", "5")
    with pytest.raises(SystemExit):
        Config.from_env()


async def test_end_from_the_app_closes_at_once_with_no_farewell():
    from tests.voice.unit.test_end_and_mute import Feeder, until

    host, ports, _ = _host([QUESTION, LATER, "one more question please"])
    host.ports.audio_in.echo_cancelled = True
    feeder = Feeder(ports.audio_in)
    try:
        await asyncio.wait_for(host.start_now(), 10)
        await until(lambda: len(ports.delegate.calls) == 1, "turn 1 to be answered")
        assert await host.end_session() is True
    finally:
        feeder.stop()
    assert _farewells_said(ports) == []


def test_chunk_size_is_what_the_waits_count():
    assert CHUNK_MS == CHUNK_FRAMES * 1000 // SAMPLE_RATE
