"""Barge-in, heard-context truncation and recovery (T13, VX-D2) at unit level.

The host runs on fakes from fakes.py with one sim clock for capture and playback (SimCapture,
SimPlayback): clauses play in real time, the user's voice is a scripted burst of capture energy
(capture is echo-cancelled, T12, so energy there is the user), and a stop silences the speaker at
the next playback slice, as the AudioEngine does. The engine's own slice-wise cancel is checked
against a fake output stream at the end.

Scenarios: O4 (10 backchannels, 0 interruptions), O12 (stored turn equals played audio within
one clause, 10 of 10), O14 (no clause repeated after an interruption, 10 of 10; at this level
that is the context the reply model receives, see test_no_clause_repeated).
"""
from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest
from voxscribe.session.ports import ReplyStream, TranscriptSegment

from app.local import audio_engine as ae
from app.local.voice.barge_in import (
    BACKCHANNEL,
    BACKCHANNEL_MAX_DURATION_MS,
    BACKCHANNEL_MAX_ENERGY,
    BARGE_IN,
    BARGE_IN_DEBOUNCE_MS,
    BARGE_IN_ENERGY_THRESHOLD,
    BARGE_IN_FRAME_MS,
    BARGE_IN_MIN_FRAMES,
    OVERLAP,
    BargeInDetector,
    interruption_note,
)
from app.local.voice.host import SAMPLE_RATE, VoiceConfig, VoiceHost

from .fakes import (
    CHUNK_MS,
    SPEECH_MS_PER_CHAR,
    Burst,
    FakeDelegate,
    FakeGate,
    FakeRecognizer,
    sim_ports,
)

LOUD = BARGE_IN_ENERGY_THRESHOLD * 3          # a person talking over the reply
MURMUR = BACKCHANNEL_MAX_ENERGY / 2           # "mm-hmm" under the backchannel ceiling
FRAME = SAMPLE_RATE * BARGE_IN_FRAME_MS // 1000
INTERJECTION_MS = 1200                        # "wait, what about tomorrow?"
# In a session the fake turn policy closes the next turn after a few chunks, so the interjection
# must be over by then or it would interrupt the next reply too.
SESSION_INTERJECTION_MS = 10 * CHUNK_MS
SHORT_BURST_MS = 2 * CHUNK_MS                 # spans at most BARGE_IN_MIN_FRAMES - 1 frames
REPLY_TIMEOUT_S = 10
SCENARIOS = 10

CLAUSES = ["Hello,", "I found", "three results for your search.",
           "The first is a bakery on Main Street.", "The second is a cafe.",
           "The third is closed today."]


def _ms(text: str) -> int:
    return len(text) * SPEECH_MS_PER_CHAR


def _host(ports) -> VoiceHost:
    return VoiceHost(VoiceConfig(), ports=ports)


def _frames(level: float, n: int) -> np.ndarray:
    return np.full(FRAME * n, level, dtype=np.float32)


async def _reply(ports, capture, text="find me a bakery"):
    host = _host(ports)
    return await asyncio.wait_for(
        host._reply("s1", 1, text, audio_q=capture, turn_end_ms=0), REPLY_TIMEOUT_S)


def _fully_played(clauses, interrupted_at_playback_ms: float) -> list[str]:
    """Ground truth: clauses whose audio ended before the speaker went silent."""
    out, t = [], 0.0
    for c in clauses:
        t += _ms(c)
        if t <= interrupted_at_playback_ms:
            out.append(c)
    return out


def _started(clauses, silent_at_playback_ms: float) -> list[str]:
    """Clauses with any audio heard."""
    out, t = [], 0.0
    for c in clauses:
        if t < silent_at_playback_ms:
            out.append(c)
        t += _ms(c)
    return out


# ── detector ─────────────────────────────────────────────────────────────────


def test_detector_fires_on_the_min_frames_th_loud_frame():
    det = BargeInDetector(SAMPLE_RATE)
    assert det.feed(_frames(LOUD, BARGE_IN_MIN_FRAMES - 1), playing=True) == []
    assert det.feed(_frames(LOUD, 1), playing=True) == [BARGE_IN]


def test_detector_ignores_energy_when_nothing_plays():
    det = BargeInDetector(SAMPLE_RATE)
    assert det.feed(_frames(LOUD, BARGE_IN_MIN_FRAMES * 3), playing=False) == []
    assert det.barge_ins == 0


def test_debounce():
    """Two barge-in signals within BARGE_IN_DEBOUNCE_MS produce one barge-in event."""
    det = BargeInDetector(SAMPLE_RATE)
    gap_frames = 2
    events = det.feed(_frames(LOUD, BARGE_IN_MIN_FRAMES), playing=True)
    events += det.feed(_frames(0.0, gap_frames), playing=True)
    events += det.feed(_frames(LOUD, BARGE_IN_MIN_FRAMES), playing=True)
    assert (2 * BARGE_IN_MIN_FRAMES + gap_frames) * BARGE_IN_FRAME_MS < (
        BARGE_IN_MIN_FRAMES * BARGE_IN_FRAME_MS + BARGE_IN_DEBOUNCE_MS)
    assert events.count(BARGE_IN) == 1 and det.barge_ins == 1
    assert BACKCHANNEL not in events       # the held-off burst is not a backchannel either

    # After the cooldown a new interruption counts again.
    det.feed(_frames(0.0, BARGE_IN_DEBOUNCE_MS // BARGE_IN_FRAME_MS), playing=True)
    assert det.feed(_frames(LOUD, BARGE_IN_MIN_FRAMES), playing=True) == [BARGE_IN]


def test_detector_classifies_short_and_long_bursts_that_keep_the_floor():
    det = BargeInDetector(SAMPLE_RATE)
    short = det.feed(_frames(LOUD, BARGE_IN_MIN_FRAMES - 1), playing=True)
    short += det.feed(_frames(0.0, 1), playing=True)
    assert short == [BACKCHANNEL]
    soft_long = BACKCHANNEL_MAX_DURATION_MS // BARGE_IN_FRAME_MS + 1
    mid = (BACKCHANNEL_MAX_ENERGY + BARGE_IN_ENERGY_THRESHOLD) / 2
    events = det.feed(_frames(mid, soft_long), playing=True)
    events += det.feed(_frames(0.0, 1), playing=True)
    assert events == [OVERLAP]
    assert det.barge_ins == 0


# ── 1. interruption stops playback and generation ────────────────────────────


async def test_barge_in_stops_playback():
    """Energy above threshold for BARGE_IN_MIN_FRAMES while a reply plays calls stop_playback."""
    burst = Burst(at_playback_ms=_ms(CLAUSES[0]) / 2, dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=FakeDelegate([CLAUSES]))
    result = await _reply(ports, capture)

    out = ports.audio_out
    assert len(out.stops) == 1
    assert result.interruption is not None
    assert result.pre_roll is not None and result.pre_roll.size
    stop_ms, _silent, _heard = out.stops[0]
    # Detection takes BARGE_IN_MIN_FRAMES frames of speech, plus at most one frame of alignment.
    assert stop_ms - burst.onset_ms <= (BARGE_IN_MIN_FRAMES + 1) * BARGE_IN_FRAME_MS + CHUNK_MS
    # Nothing was queued again after the stop.
    assert not out.is_playing


async def test_barge_in_stops_generation():
    """The sentence source is closed: the reply model's stream ends with it."""
    state = {"closed": False, "yielded": 0}
    never = asyncio.Event()

    async def endless():
        try:
            for c in CLAUSES[:2]:
                state["yielded"] += 1
                yield c
            await never.wait()          # the model is still generating the next clause
            yield "never spoken"
        finally:
            state["closed"] = True

    class SlowDelegate(FakeDelegate):
        async def reply(self, turn_text, session_id, **kw) -> ReplyStream:
            self.calls.append(turn_text)
            return ReplyStream(sentences=endless(), fallback="nothing")

    burst = Burst(at_playback_ms=_ms(CLAUSES[0]) + _ms(CLAUSES[1]) / 2,
                  dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=SlowDelegate())
    result = await _reply(ports, capture)

    assert state["closed"] and state["yielded"] == 2
    assert result.text == CLAUSES[0]
    (_n, outcome), = ports.store.turns
    assert outcome.text == CLAUSES[0] and outcome.playback_end_ms is not None


async def test_barge_in_off_without_echo_cancellation():
    """Without a canceller capture energy includes the reply itself: the reply plays out."""
    burst = Burst(at_playback_ms=0, dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=FakeDelegate([CLAUSES[:2]]))
    ports.audio_in.echo_cancelled = False
    result = await _reply(ports, capture)
    assert ports.audio_out.stops == [] and result.interruption is None
    assert result.text == " ".join(CLAUSES[:2])


# ── 2. heard-context truncation ──────────────────────────────────────────────


async def test_barge_in_heard_context():
    """played_clauses ["Hello,", "I found"]: the stored turn text is "Hello, I found"."""
    clauses = CLAUSES[:3]
    burst = Burst(at_playback_ms=_ms(clauses[0]) + _ms(clauses[1]) + _ms(clauses[2]) / 3,
                  dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=FakeDelegate([clauses]))
    result = await _reply(ports, capture)

    (_n, outcome), = ports.store.turns
    assert outcome.text == "Hello, I found" == result.text
    assert outcome.sentences == ["Hello,", "I found"]
    heard = ports.store.interruptions[1]
    assert heard.played_text == "Hello, I found"
    assert heard.full_text == " ".join(clauses)


async def test_cue_and_lead_in_are_not_part_of_the_heard_text():
    class CueDelegate(FakeDelegate):
        async def reply(self, turn_text, session_id, **kw) -> ReplyStream:
            stream = await super().reply(turn_text, session_id, **kw)
            stream.cue, stream.lead_in = "hm", "Let me check."
            return stream

    clauses = CLAUSES[:3]
    lead = _ms("hm") + _ms("Let me check.")
    burst = Burst(at_playback_ms=lead + _ms(clauses[0]) + _ms(clauses[1]) / 2,
                  dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=CueDelegate([clauses]))
    result = await _reply(ports, capture)
    assert result.text == clauses[0]
    assert ports.store.interruptions[1].full_text.startswith(clauses[0])


async def test_interrupted_refusal_keeps_the_gate_verdict():
    clauses = ["Sure,", "here is how to pick a lock."]
    ports, capture = sim_ports([Burst(at_playback_ms=0, dur_ms=INTERJECTION_MS, amp=LOUD)],
                               delegate=FakeDelegate([clauses]),
                               gate=FakeGate(refuse="pick a lock"))
    await _reply(ports, capture)
    (_n, outcome), = ports.store.turns
    assert outcome.decision == "refuse" and outcome.category == "blocked"
    assert "lock" not in outcome.text
    assert "lock" not in ports.store.interruptions[1].full_text


@pytest.mark.parametrize("case", range(SCENARIOS))
async def test_stored_turn_equals_played_audio_within_one_clause(case):
    """O12: interrupted at ten points through a six-clause reply."""
    total = sum(_ms(c) for c in CLAUSES)
    at = total * (case + 0.5) / (SCENARIOS + 1)
    burst = Burst(at_playback_ms=at, dur_ms=INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=FakeDelegate([CLAUSES]))
    result = await _reply(ports, capture)

    silent_at = ports.audio_out.stops[0][2]
    played = _fully_played(CLAUSES, silent_at)
    heard_any = _started(CLAUSES, silent_at)
    stored = ports.store.turns[0][1].sentences
    assert stored == played, (case, stored, played)
    assert len(heard_any) - len(stored) <= 1
    assert result.text == " ".join(played)


# ── 3. the next reply answers the interjection, no repetition ────────────────


def test_interruption_note_text():
    assert interruption_note("Hello, I found") == (
        '[The user interrupted. The assistant had said: "Hello, I found". '
        'Do not repeat it. Answer the interjection directly.]')


async def test_barge_in_no_repetition():
    """The turn after a barge-in gets the note in its system prompt, and the heard text only."""
    clauses = CLAUSES[:3]
    burst = Burst(at_playback_ms=_ms(clauses[0]) + _ms(clauses[1]) + _ms(clauses[2]) / 2,
                  dur_ms=SESSION_INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports(
        [burst], delegate=FakeDelegate([clauses, ["Tomorrow it opens at nine."]]),
        recognizer=FakeRecognizer(["find me a bakery", "wait, what about tomorrow?"]))
    host = _host(ports)
    await asyncio.wait_for(host.run_session(capture, score=0.9, detector="fake"),
                           REPLY_TIMEOUT_S)

    d = ports.delegate
    assert d.calls == ["find me a bakery", "wait, what about tomorrow?"]
    assert d.notes[0] is None
    assert d.notes[1] == interruption_note("Hello, I found")
    assert "[The user interrupted." in d.notes[1] and '"Hello, I found"' in d.notes[1]
    assert d.histories[1] == [{"role": "user", "content": "find me a bakery"},
                              {"role": "assistant", "content": "Hello, I found"}]
    assert [o.text for _, o in ports.store.turns] == ["Hello, I found",
                                                      "Tomorrow it opens at nine."]
    assert list(ports.store.interruptions) == [1]


async def test_reply_chain_appends_the_note_to_the_system_prompt(monkeypatch):
    from types import SimpleNamespace

    from app.local import llm_adapter, reply_chain

    seen: dict = {}

    async def fake_stream(user_text, *, model, history=None, system_prompt=None):
        seen["system_prompt"] = system_prompt
        seen["history"] = history
        yield "Tomorrow it opens at nine. "

    monkeypatch.setattr(llm_adapter, "stream_reply", fake_stream)
    note = interruption_note("Hello, I found")
    app = SimpleNamespace(state=SimpleNamespace(settings=None))
    src = reply_chain.reply_sentences(app, "m1", "what about tomorrow?",
                                      history=[{"role": "assistant",
                                                "content": "Hello, I found"}],
                                      system_note=note)
    assert [s async for s in src.sentences] == ["Tomorrow it opens at nine."]
    assert seen["system_prompt"].endswith(note)
    assert seen["system_prompt"].startswith("You are a helpful voice assistant.")

    src = reply_chain.reply_sentences(app, "m1", "hello")
    assert [s async for s in src.sentences]
    assert seen["system_prompt"] is None          # unchanged without a barge-in


class _StableFirst:
    """A recognizer stream that hears the whole turn as one stable segment on the first push,
    so the host has words to speculate on before the flush."""

    def __init__(self, text: str) -> None:
        self.text, self.sent = text, False

    def push(self, pcm):
        if self.sent or not self.text:
            return []
        self.sent = True
        return [TranscriptSegment(id=f"seg-{self.text}", text=self.text, start_ms=0, end_ms=0)]

    def pop_partials(self):
        return []

    def flush(self):
        return []


class _StableRecognizer(FakeRecognizer):
    def open_stream(self, session_id):
        text = self.turns[self.opened] if self.opened < len(self.turns) else ""
        self.opened += 1
        return _StableFirst(text)


async def test_no_speculation_on_the_turn_after_a_barge_in():
    """The speculative stream has no system prompt, so it cannot carry the note."""
    clauses = CLAUSES[:3]
    burst = Burst(at_playback_ms=_ms(clauses[0]) / 2, dur_ms=SESSION_INTERJECTION_MS, amp=LOUD)

    class Spec(FakeDelegate):
        def __init__(self, replies):
            super().__init__(replies)
            self.speculated: list[str] = []

        def speculate(self, partial_text, history):
            self.speculated.append(partial_text)
            return super().speculate(partial_text, history)

    turns = ["find me a bakery near here", "wait what about one tomorrow"]
    ports, capture = sim_ports([burst], delegate=Spec([clauses, ["Sure."]]),
                               recognizer=_StableRecognizer(turns))
    await asyncio.wait_for(_host(ports).run_session(capture, score=0.9, detector="fake"),
                           REPLY_TIMEOUT_S)
    assert ports.delegate.calls == turns
    assert ports.delegate.speculated == [turns[0]]
    assert ports.delegate.notes[1] is not None


@pytest.mark.parametrize("case", range(SCENARIOS))
async def test_no_clause_repeated(case):
    """O14 at unit level: the clause playing or queued when the user interrupted, never heard
    to its end, is not in the stored reply nor anywhere in the next turn's context; the clauses
    that were heard reach the next turn only as already said (history and the do-not-repeat
    note). Whether the reply model obeys the note is a model property, not checked here."""
    total = sum(_ms(c) for c in CLAUSES[:-1])
    at = total * (case + 0.5) / SCENARIOS
    burst = Burst(at_playback_ms=at, dur_ms=SESSION_INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports(
        [burst], delegate=FakeDelegate([CLAUSES, ["Sure, tomorrow then."]]),
        recognizer=FakeRecognizer(["find me a bakery", "and tomorrow?"]))
    await asyncio.wait_for(_host(ports).run_session(capture, score=0.9, detector="fake"),
                           REPLY_TIMEOUT_S)

    played = _fully_played(CLAUSES, ports.audio_out.stops[0][2])
    cut = CLAUSES[len(played)]                  # the clause the interruption cut off
    stored = ports.store.turns[0][1].text
    assert stored == " ".join(played)
    assert cut not in stored
    context = ports.delegate.histories[1]
    note = ports.delegate.notes[1]
    assert all(cut not in m["content"] for m in context)
    assert cut not in note
    assert note == interruption_note(" ".join(played))
    assert context[-1] == {"role": "assistant", "content": " ".join(played)} or not played


async def test_a_wordless_interruption_does_not_end_the_session():
    """A cough interrupts the reply; the next turn hears nothing; the session listens on once,
    and the turn after it still answers with the heard context."""
    clauses = CLAUSES[:3]
    burst = Burst(at_playback_ms=_ms(clauses[0]) + _ms(clauses[1]) / 2,
                  dur_ms=SESSION_INTERJECTION_MS, amp=LOUD)
    ports, capture = sim_ports(
        [burst], delegate=FakeDelegate([clauses, ["It opens at nine."]]),
        recognizer=FakeRecognizer(["find me a bakery", "", "when does it open?"]))
    await asyncio.wait_for(_host(ports).run_session(capture, score=0.9, detector="fake"),
                           REPLY_TIMEOUT_S)
    d = ports.delegate
    assert d.calls == ["find me a bakery", "when does it open?"]
    assert d.notes[1] == interruption_note(clauses[0])
    assert [o.text for _, o in ports.store.turns] == [clauses[0], "It opens at nine."]


# ── 4. backchannel keeps the floor ───────────────────────────────────────────


async def test_backchannel_no_barge_in():
    """Three frames of speech energy (under BARGE_IN_MIN_FRAMES) do not stop playback."""
    burst = Burst(at_playback_ms=_ms(CLAUSES[0]), dur_ms=SHORT_BURST_MS, amp=LOUD)
    ports, capture = sim_ports([burst], delegate=FakeDelegate([CLAUSES[:3]]))
    result = await _reply(ports, capture)
    assert burst.onset_ms is not None                  # it was heard
    assert ports.audio_out.stops == [] and result.interruption is None
    assert result.text == " ".join(CLAUSES[:3])


async def test_ten_backchannels_give_no_interruption():
    """O4: ten "mm-hmm"s, short or quiet, over one reply: 0 interruptions, all of it heard."""
    total = sum(_ms(c) for c in CLAUSES)
    bursts = []
    for i in range(SCENARIOS):
        at = total * (i + 1) / (SCENARIOS + 2)
        quiet = i % 2 == 1
        bursts.append(Burst(at_playback_ms=at,
                            dur_ms=BACKCHANNEL_MAX_DURATION_MS - BARGE_IN_FRAME_MS if quiet
                            else SHORT_BURST_MS,
                            amp=MURMUR if quiet else LOUD))
    ports, capture = sim_ports(bursts, delegate=FakeDelegate([CLAUSES]))
    result = await _reply(ports, capture)

    assert all(b.onset_ms is not None for b in bursts)
    assert ports.audio_out.stops == []
    assert result.interruption is None and result.text == " ".join(CLAUSES)


# ── engine: slice-wise cancel, played callback, unplayed reference dropped ────


class _SliceStream:
    def __init__(self, **kw) -> None:
        self.writes: list[int] = []

    def start(self):
        pass

    def stop(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass

    def write(self, chunk):
        self.writes.append(len(chunk))


class _Processor:
    def __init__(self, **kw) -> None:
        pass

    def process(self, near, far=None):
        return near

    def reset(self):
        pass


@pytest.fixture
def engine(monkeypatch):
    sd = types.SimpleNamespace(InputStream=_SliceStream, OutputStream=_SliceStream,
                               query_devices=lambda: [], _terminate=lambda: None,
                               _initialize=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(ae, "_load_audio_processor", lambda: _Processor)
    return ae.AudioEngine()


async def _spin(n: int = 50) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def test_engine_writes_in_slices_and_reports_played(engine):
    await engine.start()
    try:
        slice_n = engine.playback_rate * ae.PLAYBACK_SLICE_MS // 1000
        played: list[str] = []
        await engine.play(np.zeros(slice_n * 3, np.float32), block=True,
                          on_played=lambda: played.append("a"))
        await _spin()
        assert engine._playback_stream.writes == [slice_n] * 3
        assert played == ["a"] and not engine.is_playing
    finally:
        await engine.stop()


async def test_engine_cancel_stops_mid_chunk_and_skips_its_callback(engine):
    await engine.start()
    try:
        slice_n = engine.playback_rate * ae.PLAYBACK_SLICE_MS // 1000
        slices = 10
        played: list[str] = []
        await engine.play(np.zeros(slice_n * slices, np.float32), block=True,
                          on_played=lambda: played.append("a"))
        await engine.play(np.zeros(slice_n, np.float32), block=True,
                          on_played=lambda: played.append("b"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert engine.is_playing
        far_before = len(engine.aec._far)
        engine.cancel_playback()
        await _spin()
        writes = engine._playback_stream.writes
        assert 0 < len(writes) < slices
        assert played == [] and not engine.is_playing
        assert len(engine.aec._far) < far_before          # the unplayed tail is forgotten
        verdict, _ = engine.echo_guard.observe(None, None)
        assert verdict == "pass"                          # the user's next words are heard
    finally:
        await engine.stop()


def test_sample_fifo_drop_tail():
    fifo = ae._SampleFifo(np)
    fifo.push(np.arange(4, dtype=np.float32))
    fifo.push(np.arange(4, 8, dtype=np.float32))
    fifo.pop(1)
    assert fifo.drop_tail(5) == 5
    assert len(fifo) == 2 and list(fifo.pop(2)) == [1.0, 2.0]
    assert fifo.drop_tail(3) == 0
