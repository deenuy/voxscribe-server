"""Thinking lines only when a reply is late (PL67).

A short line from expression.THINKING is said when no reply speech has started THINKING_AFTER_S
after the turn closed, in natural only. These tests run on the event loop's own time with a
threshold of a few tens of milliseconds."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from voxscribe.session.ports import ReplyStream

from app.local.config import Config
from app.local.voice.delegate_lane import DELEGATING_PHRASE
from app.local.voice.expression import FAREWELLS, NATURAL, PLAIN, THINKING
from app.local.voice.host import (
    CHUNK_FRAMES,
    SessionBinding,
    StreamClock,
    VoiceConfig,
    VoiceHost,
)
from tests.voice.unit.fakes import (
    TALKER_RATE,
    FakeAudioOut,
    FakeDelegate,
    FakeGate,
    FakeTalker,
    fake_ports,
)

THRESHOLD_S = 0.05
LATE_S = 0.25
LINE_SAMPLES = TALKER_RATE // 2          # half a second, so it is still playing when tested


class LineTalker(FakeTalker):
    """The production talker's `line` for fixed lines in the reply voice, recorded."""

    def __init__(self, load_s: float = 0.0) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.load_s = load_s

    async def line(self, text: str) -> np.ndarray:
        if self.load_s:
            await asyncio.sleep(self.load_s)
        self.lines.append(text)
        return np.zeros(LINE_SAMPLES, dtype=np.float32)


class Speaker(FakeAudioOut):
    """Plays in real time: each play is queued behind the last, and is_playing holds until the
    queue has played out."""

    def __init__(self) -> None:
        super().__init__()
        self.queue_ends = time.monotonic()
        self.log: list[tuple[str, float, float]] = []      # (what, queued at, starts at)
        self.stopped_at: float | None = None

    async def play(self, pcm_float32_mono, sample_rate: int) -> None:
        now = time.monotonic()
        starts = max(now, self.queue_ends)
        self.queue_ends = starts + len(pcm_float32_mono) / sample_rate
        kind = "line" if len(pcm_float32_mono) == LINE_SAMPLES else "reply"
        self.log.append((kind, now, starts))
        self.played.append(len(pcm_float32_mono))

    def stop_playback(self) -> None:
        self.stops += 1
        self.stopped_at = time.monotonic()
        self.queue_ends = time.monotonic()

    @property
    def is_playing(self) -> bool:
        return time.monotonic() < self.queue_ends


class Late(FakeDelegate):
    """The reply's first sentence arrives `delay_s` after the call."""

    def __init__(self, delay_s: float, *, source: str = "llm",
                 sentences: tuple[str, ...] = ("The answer is four.",)) -> None:
        super().__init__()
        self.delay_s, self.source, self.sentences = delay_s, source, sentences

    async def reply(self, turn_text, session_id, **kw) -> ReplyStream:
        self.calls.append(turn_text)

        async def gen():
            await asyncio.sleep(self.delay_s)
            for s in self.sentences:
                yield s

        return ReplyStream(sentences=gen(), source=self.source, fallback="nothing")


def _host(delegate, *, expressiveness=NATURAL, talker=None, gate=None, out=None):
    ports = fake_ports(delegate=delegate, talker=talker or LineTalker(), gate=gate or FakeGate(),
                       audio_out=out or Speaker(),
                       clock=StreamClock(mono=lambda: int(time.monotonic() * 1000)))
    host = VoiceHost(VoiceConfig(expressiveness=expressiveness, thinking_after_s=THRESHOLD_S),
                     ports=ports)
    return host, ports


async def _turns(host, ports, texts, *, expressiveness=NATURAL, audio_q=None):
    with host._scope.session(SessionBinding()) as session:
        session.expressiveness = expressiveness
        out = []
        for n, text in enumerate(texts, 1):
            result = await host._reply("s1", n, text, turn_end_ms=ports.clock.mono_ms(),
                                       audio_q=audio_q)
            out.append(result)
        return out, session


def _kinds(out: Speaker) -> list[str]:
    return [k for k, _q, _s in out.log]


async def test_a_reply_that_starts_in_time_has_no_thinking_line():
    host, ports = _host(Late(0.0))
    await _turns(host, ports, ["what is two plus two"])
    assert ports.talker.lines == [] and _kinds(ports.audio_out) == ["reply"]


async def test_a_late_reply_is_preceded_by_exactly_one_line():
    host, ports = _host(Late(LATE_S))
    (result,), _ = await _turns(host, ports, ["what is two plus two"])
    assert ports.talker.lines == [THINKING["let_me_see"]]
    assert _kinds(ports.audio_out) == ["line", "reply"]
    assert result.text == "The answer is four."            # the line is not the reply


async def test_the_line_is_never_the_same_as_the_last_one_in_the_session():
    host, ports = _host(Late(LATE_S))
    _, session = await _turns(host, ports, ["first question here", "second question here",
                                            "third question here"])
    lines = ports.talker.lines
    assert len(lines) == 3 and all(a != b for a, b in zip(lines, lines[1:], strict=False))
    assert set(lines) <= set(THINKING.values())
    assert session.last_thinking == list(THINKING)[2]


@pytest.mark.parametrize("text", ["yes", "No.", "yeah sure", "actually, make it Tuesday",
                                  "no I said six"])
async def test_no_line_after_a_yes_a_no_or_a_correction(text):
    host, ports = _host(Late(LATE_S))
    await _turns(host, ports, [text])
    assert ports.talker.lines == [] and _kinds(ports.audio_out) == ["reply"]


@pytest.mark.parametrize("source", ["delegate", "reminder", "clock", "cancel", "delivery"])
async def test_a_scripted_route_says_its_own_first_line(source):
    """A lookup announces the action ("I'm looking that up") as its first sentence, with no
    model in the way: no thinking line goes before it."""
    host, ports = _host(Late(LATE_S, source=source, sentences=(DELEGATING_PHRASE,)))
    await _turns(host, ports, ["who won the game last night"])
    assert ports.talker.lines == [] and " ".join(ports.talker.synthesized) == DELEGATING_PHRASE


async def test_plain_has_no_late_line():
    host, ports = _host(Late(LATE_S), expressiveness=PLAIN)
    await _turns(host, ports, ["what is two plus two"], expressiveness=PLAIN)
    assert ports.talker.lines == []


async def test_the_line_passes_the_gate_and_a_refused_one_is_not_said():
    host, ports = _host(Late(LATE_S))
    await _turns(host, ports, ["what is two plus two"])
    assert THINKING["let_me_see"] in ports.gate.checked
    refusing = FakeGate(refuse="Let me see")
    host, ports = _host(Late(LATE_S), gate=refusing)
    (result,), _ = await _turns(host, ports, ["what is two plus two"])
    assert _kinds(ports.audio_out) == ["reply"] and result.text == "The answer is four."


async def test_the_line_is_played_like_any_reply_audio():
    """Through AudioOut.play, the engine's playback queue, which mutes capture for its audio
    and tail (EchoGuard); never as a cue file."""
    host, ports = _host(Late(LATE_S))
    await _turns(host, ports, ["what is two plus two"])
    assert ports.audio_out.files == [] and _kinds(ports.audio_out)[0] == "line"


async def test_the_reply_follows_the_line_with_no_gap_and_no_overlap():
    """The reply's first clause is ready while the line plays: it is queued behind it, so it
    starts the moment the line ends."""
    host, ports = _host(Late(THRESHOLD_S + 0.05))
    await _turns(host, ports, ["what is two plus two"])
    (k1, _q1, line_start), (k2, queued, reply_start) = ports.audio_out.log
    assert (k1, k2) == ("line", "reply")
    line_end = line_start + LINE_SAMPLES / TALKER_RATE
    assert queued < line_end                                # ready while the line played
    assert reply_start == pytest.approx(line_end)           # no gap, no overlap


async def test_a_reply_ready_while_the_line_loads_goes_first_and_the_line_is_dropped():
    host, ports = _host(Late(THRESHOLD_S + 0.02), talker=LineTalker(load_s=0.15))
    await _turns(host, ports, ["what is two plus two"])
    assert _kinds(ports.audio_out) == ["reply"]


class LoudRoom:
    """Capture: a person speaking loudly into the microphone, all the time."""

    def __init__(self) -> None:
        self.reads = 0

    async def get(self):
        await asyncio.sleep(0.002)
        self.reads += 1
        return np.full(CHUNK_FRAMES, 0.3, dtype=np.float32)

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def empty(self) -> bool:
        return True


async def test_a_person_speaking_over_the_line_stops_it_at_once():
    host, ports = _host(Late(1.0))
    started = time.monotonic()
    (result,), _ = await _turns(host, ports, ["what is two plus two"], audio_q=LoudRoom())
    out = ports.audio_out
    assert _kinds(out) == ["line"] and out.stops >= 1
    assert out.stopped_at - out.log[0][2] < 0.3            # well before the line's end
    assert result.interruption is not None and time.monotonic() - started < 1.0


async def test_the_lines_are_rendered_ahead_in_natural():
    host, ports = _host(Late(0.0))
    await host._warm_lines(ports)
    assert ports.talker.lines == [*THINKING.values(), *FAREWELLS.values()]
    plain, plain_ports = _host(Late(0.0), expressiveness=PLAIN)
    await plain._warm_lines(plain_ports)
    assert plain_ports.talker.lines == []


def test_the_threshold_is_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Config().voice_thinking_after_s == 1.5
    monkeypatch.setenv("VOXSCRIBE_VOICE_THINKING_AFTER_S", "2.0")
    assert Config.from_env().voice_thinking_after_s == 2.0
    assert VoiceConfig.from_settings(Config.from_env()).thinking_after_s == 2.0
    monkeypatch.setenv("VOXSCRIBE_VOICE_THINKING_AFTER_S", "0.1")
    with pytest.raises(SystemExit):
        Config.from_env()


async def test_a_line_is_rendered_once_in_the_reply_voice_and_cached(tmp_path, monkeypatch):
    from app.local import tts_engine
    from app.local.voice.adapters.tts_kokoro import KokoroTalker

    calls: list[tuple[str, str, float]] = []

    async def synthesize(text, *, voice="", speed=1.0):
        calls.append((text, voice, speed))
        return np.full(2400, 0.1, dtype=np.float32)

    monkeypatch.setattr(tts_engine, "_PHRASES_DIR", tmp_path)
    monkeypatch.setattr(tts_engine, "synthesize", synthesize)
    talker = KokoroTalker(get_voice_speed=lambda: ("af_test", 1.1))
    first = await talker.line(THINKING["let_me_see"])
    again = await talker.line(THINKING["let_me_see"])
    assert calls == [(THINKING["let_me_see"], "af_test", 1.1)]
    assert first.shape == again.shape == (2400,)
    assert len(list(tmp_path.glob("line-*.wav"))) == 1
