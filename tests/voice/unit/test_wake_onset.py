"""The speech-onset wake detector: an utterance that begins with the phrase raises a candidate,
anything else does not, and a name the recognizer only wrote because it was primed is refused."""
from __future__ import annotations

import numpy as np
import pytest
from voxscribe.models import WordTiming

from app.local.voice.adapters.wake_onset import (
    CHECK_AFTER_S,
    DETECTED_SCORE,
    HybridWakeDetector,
    OnsetPhraseDetector,
    hears_name,
    prompt_for,
    starts_with_phrase,
)

FRAME = 1280          # the wake gate's frame: 80 ms at 16 kHz
FRAME_S = FRAME / 16_000


def words(text: str, confidence: float = 0.9) -> list[WordTiming]:
    return [WordTiming(w, i * 300.0, i * 300.0 + 250.0, confidence) for i, w in enumerate(text.split())]


class Recognizer:
    """Hands back scripted words and remembers how it was asked (primed or not)."""

    def __init__(self, primed: str, unprimed: str | None = None, confidence: float = 0.9) -> None:
        self.primed, self.unprimed, self.confidence = primed, unprimed, confidence
        self.calls: list[str] = []

    def __call__(self, audio: np.ndarray, prompt: str):
        self.calls.append(prompt)
        if prompt:
            return words(self.primed, self.confidence)
        return words(self.unprimed if self.unprimed is not None else self.primed)


def detector(recognizer, speech: list[bool]) -> tuple[OnsetPhraseDetector, list[float]]:
    """Run one frame per entry of `speech` (True: the voice detector hears speech)."""
    pattern = iter(speech)
    det = OnsetPhraseDetector("hey vox", transcribe=recognizer, vad=lambda frame: 0.9 if next(pattern) else 0.0)
    scores = [det.score(np.zeros(FRAME, dtype=np.float32)) for _ in speech]
    return det, scores


def frames(seconds: float) -> int:
    return int(round(seconds / FRAME_S))


QUIET, TALK = [False] * frames(0.5), [True] * frames(1.6)


@pytest.mark.parametrize("heard", ["Hey Vox", "hey, voks!", "Hey Vocks what time is it", "Oh, hey Vox", "Heyvox"])
def test_an_utterance_that_begins_with_the_phrase_matches(heard):
    assert starts_with_phrase(words(heard), "hey vox")


@pytest.mark.parametrize("heard", [
    "hey fox", "hey box", "hey folks welcome back", "hey max", "hey socks", "say vox", "vox",
    "hey jarvis", "hey jarvis what time is it", "I told him hey vox", "a box of rocks", "hey", "",
])
def test_a_near_miss_or_a_phrase_later_in_the_sentence_does_not(heard):
    assert not starts_with_phrase(words(heard), "hey vox")


def test_the_recognizer_is_primed_with_the_phrase_as_written():
    assert prompt_for("hey vox") == "Hey Vox."


def test_a_wake_raises_one_candidate_shortly_after_the_utterance_starts():
    rec = Recognizer("Hey Vox how are you")
    _, scores = detector(rec, QUIET + TALK + QUIET)
    hits = [i for i, s in enumerate(scores) if s == DETECTED_SCORE]
    assert len(hits) == 1
    after_onset = (hits[0] - len(QUIET) + 1) * FRAME_S
    assert CHECK_AFTER_S[0] <= after_onset <= CHECK_AFTER_S[0] + 2 * FRAME_S
    assert rec.calls == ["Hey Vox."]          # sure of the name: one pass, primed


def test_a_short_wake_is_checked_when_the_utterance_ends():
    rec = Recognizer("Hey Vox")
    _, scores = detector(rec, QUIET + [True] * frames(0.6) + QUIET)
    assert scores.count(DETECTED_SCORE) == 1


def test_other_speech_never_raises_a_candidate_and_costs_at_most_two_passes():
    rec = Recognizer("what is the weather like today in the city")
    _, scores = detector(rec, QUIET + [True] * frames(4.0) + QUIET)
    assert DETECTED_SCORE not in scores
    assert len(rec.calls) == len(CHECK_AFTER_S)


def test_quiet_and_noise_cost_no_recognizer_pass():
    rec = Recognizer("Hey Vox")
    _, scores = detector(rec, [False] * frames(5.0))
    assert DETECTED_SCORE not in scores and rec.calls == []


def test_an_unsure_name_is_accepted_only_when_an_unprimed_pass_hears_it_too():
    heard_again = Recognizer("Hey Vox", unprimed="hey vox", confidence=0.0)
    _, scores = detector(heard_again, QUIET + TALK + QUIET)
    assert scores.count(DETECTED_SCORE) == 1
    assert heard_again.calls[:2] == ["Hey Vox.", ""]

    # Priming wrote the phrase over "hey folks": no confidence, and the unprimed pass disagrees.
    pulled = Recognizer("Hey Vox", unprimed="hey folks welcome back", confidence=0.0)
    _, scores = detector(pulled, QUIET + TALK + QUIET)
    assert DETECTED_SCORE not in scores


def test_the_unprimed_pass_must_hear_the_name_itself():
    assert hears_name(words("hey vox"), "hey vox") and hears_name(words("k voks"), "hey vox")
    # A speaker whose v and w are one sound: the unprimed pass writes the nearest common word.
    for same_sound in ("hey walks", "hey wox how are you", "hey wax", "Hey, Walks."):
        assert hears_name(words(same_sound), "hey vox")
    for other in ("hey folks", "hey box", "hey fox", "hey max", "hey von", "hey bots"):
        assert not hears_name(words(other), "hey vox")


def test_the_gate_gets_the_words_of_the_pass_that_raised_the_candidate():
    rec = Recognizer("Hey Vox how are you")
    pattern = QUIET + [True] * frames(CHECK_AFTER_S[0])
    it = iter(pattern)
    det = OnsetPhraseDetector("hey vox", transcribe=rec, vad=lambda f: 0.9 if next(it) else 0.0)
    fed = np.zeros(0, dtype=np.float32)
    score = 0.0
    for _ in pattern:
        frame = np.zeros(FRAME, dtype=np.float32)
        fed = np.concatenate([fed, frame])
        score = det.score(frame)
    assert score == DETECTED_SCORE
    passes = len(rec.calls)
    window = fed[-32_000:]                       # what the gate asks about: its last two seconds
    assert [w.word for w in det.confirm(window)] == ["Hey", "Vox", "how", "are", "you"]
    assert len(rec.calls) == passes              # no second pass for the gate


def test_a_recognizer_failure_is_no_candidate():
    def broken(audio, prompt):
        raise RuntimeError("model unavailable")

    _, scores = detector(broken, QUIET + TALK + QUIET)
    assert DETECTED_SCORE not in scores


def test_reset_forgets_the_utterance_in_progress():
    rec = Recognizer("Hey Vox")
    it = iter([False] * 6 + [True] * 5 + [False] * 6 + [True] * 30)
    det = OnsetPhraseDetector("hey vox", transcribe=rec, vad=lambda f: 0.9 if next(it) else 0.0)
    for _ in range(11):
        det.score(np.zeros(FRAME, dtype=np.float32))
    det.reset()
    scores = [det.score(np.zeros(FRAME, dtype=np.float32)) for _ in range(36)]
    assert scores.count(DETECTED_SCORE) == 1


class Spotter:
    model_name, patience_frames = "spotter", 1

    def __init__(self, fire_on: int) -> None:
        self.n, self.fire_on, self.resets = 0, fire_on, 0

    def reset(self) -> None:
        self.resets += 1

    def score(self, frame) -> float:
        self.n += 1
        return 1.0 if self.n == self.fire_on else 0.0


def test_a_candidate_the_spotter_raised_still_needs_the_name_to_be_heard():
    quiet = iter([False] * 100)
    sure = Recognizer("so I said hey Vox")
    hybrid = HybridWakeDetector(OnsetPhraseDetector("hey vox", transcribe=sure, vad=lambda f: 0.0), Spotter(3))
    assert [hybrid.score(np.zeros(FRAME, dtype=np.float32)) for _ in range(4)] == [0.0, 0.0, 1.0, 0.0]
    assert [w.word for w in hybrid.confirm(np.zeros(32_000, dtype=np.float32))][-2:] == ["hey", "Vox"]

    pulled = Recognizer("Hey Vox", unprimed="hey folks", confidence=0.0)
    hybrid = HybridWakeDetector(OnsetPhraseDetector("hey vox", transcribe=pulled, vad=lambda f: 0.0), Spotter(1))
    assert hybrid.confirm(np.zeros(32_000, dtype=np.float32)) == []
    del quiet


def test_the_default_detector_is_the_hybrid_and_works_without_the_spotter_model(monkeypatch, tmp_path):
    from app.local.voice import wake_setup
    from app.local.voice.adapters import wake_openvocab

    monkeypatch.delenv(wake_setup.DETECTOR_ENV, raising=False)
    monkeypatch.setenv(wake_openvocab.KWS_MODEL_DIR_ENV, str(tmp_path))     # no spotter model there
    det = wake_setup.make_detector(transcribe=Recognizer("Hey Vox"))
    assert det.model_name == "hybrid:hey_vox" and det.patience_frames == 1
    assert callable(det.confirm)


def test_a_speaker_whose_v_is_heard_as_w_still_wakes_it():
    """Measured on the owner's voice: the primed pass writes the phrase with almost no confidence
    in the name, and the unprimed pass writes "hey walks"."""
    rec = Recognizer("Hey Vox how are you", unprimed="hey walks how are you", confidence=0.002)
    _, scores = detector(rec, QUIET + TALK + QUIET)
    assert scores.count(DETECTED_SCORE) == 1
