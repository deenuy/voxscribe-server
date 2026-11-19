"""The pace of turn-taking is the product's choice, and a turn that begins with speech is seen.

Two defects found in the owner's first test of the assistant (2026-11-19): at the kernel's own
bands a finished question closes the turn only after six seconds of silence, and a detector that
has just been reset does not report speech that begins at once, which is how every turn opened
by a wake phrase begins.
"""
from __future__ import annotations

import numpy as np
import pytest
from voxscribe.silence_policy import PauseBandConfig
from voxscribe.turn_taking import TurnTakingConfig

from app.local.config import Config
from app.local.end_of_turn import PATIENT_PAUSE_S, UNFINISHED_EXTRA_MS, turn_pacing
from app.local.voice.adapters.speech_heuristic import HeuristicSpeech

RATE, FRAME = 16_000, 512


def voiced(seconds: float, seed: int = 1) -> np.ndarray:
    """A speech-like sound: a harmonic stack with a moving pitch and a syllable rhythm."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    pitch = 130 + 25 * np.sin(2 * np.pi * 1.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 12))
    x = x * (0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2)
    x = x + np.random.default_rng(seed).normal(0, 0.02, n)
    return (0.25 * x / np.max(np.abs(x))).astype(np.float32)


def closes_after(pause_s: float | None, speech_s: float, *, lead_s: float = 0.0,
                 text: str = "What time is it?", listen_s: float = 9.0) -> float | None:
    """Seconds from the end of the speech to the decision that closes the turn, or None."""
    sp = HeuristicSpeech(pause_s)
    sp.reset()
    lead = int(lead_s * RATE)
    speech = voiced(speech_s)
    y = np.concatenate([np.zeros(lead, np.float32), speech, np.zeros(int(listen_s * RATE), np.float32)])
    end_s = (lead + len(speech)) / RATE
    told = False
    for i in range(0, len(y) - FRAME + 1, FRAME):
        now_s = (i + FRAME) / RATE
        if not told and now_s > end_s:
            sp.update_text(text)
            told = True
        decisions = sp.process(y[i:i + FRAME])
        if (i // FRAME) % 3 == 0:
            decisions = sp.tick(sp._td.stream_ms) or decisions
        if decisions and decisions[-1].closes_turn:
            return now_s - end_s
    return None


def test_a_patient_pause_keeps_the_kernels_own_bands():
    for pause in (None, PATIENT_PAUSE_S, 10.0):
        assert turn_pacing(pause) == (PauseBandConfig(), TurnTakingConfig())


def test_a_shorter_pause_scales_the_bands_and_bounds_an_unfinished_sentence():
    bands, turns = turn_pacing(0.9)
    assert bands.extended_ms == 900.0 and bands.short_ms < bands.long_ms < bands.extended_ms
    assert turns.hold_cap_ms == 900.0 + UNFINISHED_EXTRA_MS


def test_a_finished_question_is_answered_in_about_a_second_at_a_conversational_pace():
    after = closes_after(0.9, 1.2, lead_s=1.0)
    assert after is not None and after < 1.5


def test_the_same_question_waits_about_six_seconds_at_the_patient_pace():
    after = closes_after(PATIENT_PAUSE_S, 1.2, lead_s=1.0)
    assert after is not None and 5.0 < after < 7.5


@pytest.mark.parametrize("speech_s", [0.5, 0.8, 1.2])
def test_speech_that_begins_the_moment_the_turn_opens_is_seen(speech_s):
    """The words right after a wake phrase are the first audio the turn is given."""
    after = closes_after(0.9, speech_s, lead_s=0.0)
    assert after is not None and after < 1.5


def test_a_sentence_that_looks_unfinished_is_still_taken_within_the_bound():
    after = closes_after(0.9, 1.2, lead_s=1.0, text="What I wanted to ask about the")
    assert after is not None and after < (900.0 + UNFINISHED_EXTRA_MS) / 1000 + 0.6


def test_the_pace_can_be_set_by_the_operator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOXSCRIBE_END_OF_TURN_PAUSE_S", "0.9")
    assert Config.from_env().end_of_turn_pause_s == 0.9
    monkeypatch.delenv("VOXSCRIBE_END_OF_TURN_PAUSE_S")
    assert Config.from_env().end_of_turn_pause_s is None


@pytest.mark.parametrize("bad", ["fast", "0.1", "90"])
def test_a_pace_that_makes_no_sense_stops_the_engine_at_start(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOXSCRIBE_END_OF_TURN_PAUSE_S", bad)
    with pytest.raises(SystemExit):
        Config.from_env()
