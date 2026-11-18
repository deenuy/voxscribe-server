"""Unit tests for T11 clause-level streaming speech.

Tests cover:
  - split_into_clauses() edge cases
  - gate applied per clause, with failure stopping the sentence
  - pipeline ordering: clause 1 starts playing before clause 2 synthesis completes
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest
from voxscribe.session.ports import (
    GATE_PASS,
    GateDecision,
    ReplyStream,
)

from app.local.voice.clause_splitter import (
    CLAUSE_GAP_MAX_MS,
    MIN_CLAUSE_WORDS,
    split_into_clauses,
)
from app.local.voice.host import _clause_sentence_stream, _ClauseTalker, StreamClock

# ---------------------------------------------------------------------------
# Helpers

TALKER_RATE = 24_000
SAMPLES_PER_CHAR = 10


async def _sentences(*items: str) -> AsyncIterator[str]:
    for s in items:
        yield s


def _pcm(text: str) -> np.ndarray:
    return np.zeros(len(text) * SAMPLES_PER_CHAR, dtype=np.float32)


# ---------------------------------------------------------------------------
# split_into_clauses() unit tests


def test_no_split_short_sentence():
    """A short sentence with no valid split boundary stays as one clause."""
    result = split_into_clauses("Hello.")
    assert result == ["Hello."]


def test_split_at_comma():
    """A sentence with enough words before the comma is split at the comma."""
    sentence = "I found three results, and here they are."
    result = split_into_clauses(sentence)
    assert len(result) == 2
    assert result[0] == "I found three results,"
    assert result[1] == "and here they are."


def test_no_split_before_min_words():
    """Too few words before the comma means no split."""
    # "Yes" = 1 word, below MIN_CLAUSE_WORDS
    result = split_into_clauses("Yes, I can.")
    assert result == ["Yes, I can."]
    # Also check exactly at the boundary (MIN_CLAUSE_WORDS - 1 words = no split)
    few_words = " ".join(["word"] * (MIN_CLAUSE_WORDS - 1))
    result2 = split_into_clauses(f"{few_words}, and then some more words here.")
    assert len(result2) == 1


def test_split_at_min_words_exactly():
    """Exactly MIN_CLAUSE_WORDS words before the comma triggers a split."""
    words = " ".join(["word"] * MIN_CLAUSE_WORDS)
    sentence = f"{words}, and more words follow."
    result = split_into_clauses(sentence)
    assert len(result) == 2
    assert result[0] == f"{words},"


# ---------------------------------------------------------------------------
# Gate-per-clause tests


class _FakeGate:
    """Gate that refuses any clause containing the refuse_word."""

    fail_closed_phrase = "Sorry, I cannot say that."

    def __init__(self, refuse_word: str | None = None) -> None:
        self.refuse_word = refuse_word
        self.checked: list[str] = []

    def needs_session_tools(self, sentence: str) -> bool:
        return False

    def check(self, sentence: str, session_tools=(), *, user_text: str = "") -> GateDecision:
        self.checked.append(sentence)
        if self.refuse_word and self.refuse_word in sentence:
            return GateDecision("refuse", "blocked", self.fail_closed_phrase)
        return GateDecision(GATE_PASS)


class _FakeTalker:
    sample_rate = TALKER_RATE

    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.fixed_phrases: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.synthesized.append(text)
        return _pcm(text)

    async def fixed(self, phrase: str) -> np.ndarray:
        self.fixed_phrases.append(phrase)
        return _pcm(phrase)

    async def warm(self) -> None:
        pass


class _FakeAudioOut:
    available = True

    def __init__(self) -> None:
        self.played: list[int] = []
        self.stops = 0

    async def play(self, pcm_float32_mono, sample_rate: int) -> None:
        self.played.append(len(pcm_float32_mono))

    async def play_file(self, path: str) -> None:
        pass

    def stop_playback(self) -> None:
        self.stops += 1

    @property
    def is_playing(self) -> bool:
        return False

    def playback_ref(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


def _mono_clock(start_ms: int = 0) -> StreamClock:
    counter = [start_ms]

    def _tick() -> int:
        v = counter[0]
        counter[0] += 1       # advance 1 ms per call for distinct timestamps
        return v

    return StreamClock(mono=_tick)


@pytest.mark.asyncio
async def test_gate_per_clause():
    """A clause that fails the gate produces the refusal phrase; subsequent clauses not spoken."""
    from voxscribe.session.stages.reply import speak_reply

    # Sentence that splits into two clauses; second clause is refused.
    sentence = "First result looks good, second clause is FORBIDDEN."
    clauses = split_into_clauses(sentence)
    assert len(clauses) == 2, f"Expected 2 clauses, got {clauses}"

    gate = _FakeGate(refuse_word="FORBIDDEN")
    talker = _FakeTalker()
    audio_out = _FakeAudioOut()
    clock = _mono_clock()

    async def _gen():
        yield sentence

    stream = ReplyStream(
        sentences=_clause_sentence_stream(_gen()),
        fallback="",
    )

    async def _no_tools():
        return []

    outcome = await speak_reply(
        stream, gate=gate, talker=talker, audio_out=audio_out,
        clock=clock, user_text="test",
        session_tools=_no_tools,
    )

    # Gate saw clause 1 (passed) and clause 2 (refused).
    assert len(gate.checked) == 2
    assert gate.checked[0] == clauses[0]
    assert gate.checked[1] == clauses[1]

    # Only clause 1 was synthesised; refused clause was never sent to Talker.
    assert talker.synthesized == [clauses[0]]

    # The refusal phrase was spoken as a fixed phrase.
    assert gate.fail_closed_phrase in talker.fixed_phrases

    # Outcome records the refusal.
    assert outcome.decision == "refuse"


@pytest.mark.asyncio
async def test_pipeline_timing():
    """Clause 1 is queued for playback before clause 2 synthesis completes.

    With AudioOut.play() returning once queued (non-blocking), speak_reply
    queues clause 1 and immediately starts synthesising clause 2. This test
    verifies the ordering: play(clause1) precedes synth_done(clause2).
    """
    from voxscribe.session.stages.reply import speak_reply

    events: list[str] = []

    class _OrderTrackerTalker:
        sample_rate = TALKER_RATE

        def __init__(self) -> None:
            self.synthesized: list[str] = []
            self.fixed_phrases: list[str] = []

        async def synthesize(self, text: str) -> np.ndarray:
            # Yield so other tasks can run, simulating async synthesis work.
            await asyncio.sleep(0)
            events.append(f"synth_done:{text[:30]}")
            self.synthesized.append(text)
            return _pcm(text)

        async def fixed(self, phrase: str) -> np.ndarray:
            self.fixed_phrases.append(phrase)
            return _pcm(phrase)

        async def warm(self) -> None:
            pass

    class _OrderTrackerAudioOut:
        available = True

        def __init__(self) -> None:
            self.played: list[int] = []
            self.stops = 0

        async def play(self, pcm, sample_rate: int) -> None:
            # Non-blocking: records play call without awaiting any duration.
            pcm_label = f"pcm_len={len(pcm)}"
            events.append(f"play:{pcm_label}")
            self.played.append(len(pcm))

        async def play_file(self, path: str) -> None:
            pass

        def stop_playback(self) -> None:
            self.stops += 1

        @property
        def is_playing(self) -> bool:
            return False

        def playback_ref(self) -> np.ndarray:
            return np.zeros(0, dtype=np.float32)

    sentence = "I found three results, and here they are."
    clauses = split_into_clauses(sentence)
    assert len(clauses) == 2, f"expected 2 clauses but got {clauses}"

    gate = _FakeGate()
    talker = _OrderTrackerTalker()
    audio_out = _OrderTrackerAudioOut()
    clock = _mono_clock()

    async def _gen():
        yield sentence

    stream = ReplyStream(
        sentences=_clause_sentence_stream(_gen()),
        fallback="",
    )

    async def _no_tools():
        return []

    await speak_reply(
        stream, gate=gate, talker=talker, audio_out=audio_out,
        clock=clock, user_text="test",
        session_tools=_no_tools,
    )

    # Both clauses were synthesised and played.
    assert talker.synthesized == clauses
    assert len(audio_out.played) == 2

    # Ordering: play(clause1) must appear before synth_done(clause2).
    # This confirms clause 1 is queued (starts playing) before clause 2 synthesis completes.
    play_1_idx = events.index(f"play:pcm_len={len(_pcm(clauses[0]))}")
    synth_2_idx = events.index(f"synth_done:{clauses[1][:30]}")
    assert play_1_idx < synth_2_idx, (
        f"Expected play(clause1) before synth_done(clause2) but got events={events}"
    )
