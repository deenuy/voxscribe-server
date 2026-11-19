"""Stage 1 of the wake gate by speech onset: the recognizer decides, the spotter only helps.

The keyword spotter (wake_openvocab) was the only stage 1, and it is the wrong tool for a name it
was never trained on: on voices it had not been tuned against it fired on 8 of 20 "hey vox" clips,
and no boost, threshold or respelling moved that (measured 2026-11-18). The recognizer confirm that
sits behind it hears the same clips correctly once it is told what to listen for.

So this detector raises a candidate from the audio itself:

1. a streaming voice detector finds the start of an utterance after a stretch of quiet;
2. shortly after the start (and once more for a slow speaker, or when the utterance ends first)
   the last two seconds go through the recognizer, primed with the phrase;
3. the candidate is raised only when the utterance *begins* with the phrase, spelled the way the
   phrase is spelled or a spelling that sounds identical ("vox", "voks", "vocks");
4. and the recognizer was sure of the name, or hears the name again with no priming at all.

Rule 3 is deliberately stricter than the gate's own rhyme match. That match is right for a
candidate a spotter already picked out of a sentence; applied to every utterance it would wake on
"hey folks" and "a box of rocks". Priming pulls a true "hey vox" to its spelling, so the strict rule
loses nothing on the measured clips.

Rule 4 is the price of priming: it also pulls a near-miss ("hey folks", "hey box", "hey docs") to
the phrase, and it does so with almost no confidence in the name. A sure name passes at once; an
unsure one gets a second pass, unprimed, and passes only if that pass hears the name too. On the
clips measured 2026-11-18 this kept every near-miss and everyday phrase out and nine in ten wakes
in (unseen voices included), where the spotter alone let five of twenty unseen voices through.

The gate still runs its voice check and its confirm after this. `confirm()` hands the gate the
words of the decode that raised the candidate, so the gate adds no recognizer pass of its own.

No audio and no words are logged or stored here.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import deque
from collections.abc import Callable, Sequence

import numpy as np
from voxscribe.models import WordTiming
from voxscribe.wake.gate import ASR_CONFIRM_PHRASE
from voxscribe.wake.phrase import phonetic_key

log = logging.getLogger("app.local.voice.wake_onset")

SAMPLE_RATE = 16_000
SPEECH_PROB = 0.5
ONSET_S = 0.06                     # speech that starts an utterance (one gate frame is 80 ms)
QUIET_BEFORE_S = 0.3               # quiet needed before a new utterance can start
END_SILENCE_S = 0.25               # quiet that ends an utterance
MIN_SPEECH_S = 0.3                 # an utterance shorter than this is not checked when it ends
CHECK_AFTER_S = (1.1, 2.0)         # recognizer passes, measured from the start of the utterance
WINDOW_S = 2.0                     # audio handed to the recognizer; the gate's confirm window
MAX_LEADING_WORDS = 1              # words allowed before the phrase ("oh, hey vox")
NAME_CONFIDENCE = 0.02             # the primed pass's confidence in the name that needs no second pass
DETECTED_SCORE = 1.0
SILENT_SCORE = 0.0

Transcribe = Callable[[np.ndarray, str], Sequence[WordTiming]]


def prompt_for(phrase: str) -> str:
    """What the recognizer is primed with: the phrase as a person would write it ("Hey Vox.")."""
    return " ".join(w.capitalize() for w in phrase.split()) + "."


def _tokens(words: Sequence[WordTiming]) -> list[str]:
    return [p.lower() for w in words for p in re.split(r"[^A-Za-z]+", w.word) if p]


def _same_word(heard: str, want: str) -> bool:
    """Sounds the same and starts with the same letter: "voks" is "vox", "fox" and "box" are not."""
    return bool(heard) and heard[0] == want[0] and phonetic_key(heard) == phonetic_key(want)


def _phrase_words(phrase: str) -> list[str]:
    return [w for w in (re.sub(r"[^a-z]", "", p) for p in phrase.lower().split()) if w]


def _tokens_with_confidence(words: Sequence[WordTiming]) -> list[tuple[str, float]]:
    return [(p.lower(), float(w.confidence)) for w in words
            for p in re.split(r"[^A-Za-z]+", w.word) if p]


def phrase_start_confidence(words: Sequence[WordTiming],
                            phrase: str = ASR_CONFIRM_PHRASE) -> float | None:
    """The recognizer's confidence in the name when the utterance begins with `phrase` (allowing
    MAX_LEADING_WORDS before it), or None when it does not begin with it.

    Each word must match in sound and first letter; the phrase written as one word ("heyvox")
    counts. Anything else, however close it rhymes, is not the phrase."""
    want = _phrase_words(phrase)
    if not want:
        return None
    toks = _tokens_with_confidence(words)
    joined = "".join(want)
    for start in range(MAX_LEADING_WORDS + 1):
        run = toks[start:start + len(want)]
        if len(run) == len(want) and all(_same_word(h, w) for (h, _), w in zip(run, want, strict=True)):
            return run[-1][1]
        if len(want) > 1 and start < len(toks) and len(toks[start][0]) > len(want[-1]) \
                and _same_word(toks[start][0], joined):
            return toks[start][1]
    return None


def starts_with_phrase(words: Sequence[WordTiming], phrase: str = ASR_CONFIRM_PHRASE) -> bool:
    """True when the utterance begins with `phrase`; see phrase_start_confidence."""
    return phrase_start_confidence(words, phrase) is not None


def hears_name(words: Sequence[WordTiming], phrase: str = ASR_CONFIRM_PHRASE) -> bool:
    """True when any word is the phrase's name (its last word) by sound and first letter, or the
    whole phrase written as one word. For the unprimed second pass, which has no reason to write
    the name unless it was said."""
    want = _phrase_words(phrase)
    if not want:
        return False
    joined = "".join(want)
    return any(_same_word(t, want[-1]) or (len(want) > 1 and len(t) > len(want[-1])
                                           and _same_word(t, joined))
               for t in _tokens(words))


class _Vad:
    """Speech probability per hop from the streaming Silero model core ships."""

    def __init__(self) -> None:
        from voxscribe.speech.silero import SileroDetector

        self._det = SileroDetector()

    def reset(self) -> None:
        self._det.reset()

    def __call__(self, frame: np.ndarray) -> float:
        frames = self._det.process(frame)
        return max((f.probability for f in frames), default=0.0)


class OnsetPhraseDetector:
    """A WakeDetector that scores DETECTED on the frame where an utterance is found to begin with
    the phrase, and SILENT otherwise."""

    patience_frames = 1

    def __init__(self, phrase: str = ASR_CONFIRM_PHRASE, *, transcribe: Transcribe | None = None,
                 vad: Callable[[np.ndarray], float] | None = None) -> None:
        self.phrase = " ".join(phrase.lower().split())
        self.model_name = f"onset:{self.phrase.replace(' ', '_')}"
        self._prompt = prompt_for(self.phrase)
        self._transcribe = transcribe
        self._vad = vad
        self._lock = threading.Lock()
        self._window_samples = int(WINDOW_S * SAMPLE_RATE)
        self._ring: deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._reset_state()

    # ------------------------------------------------------------------ WakeDetector

    def reset(self) -> None:
        with self._lock:
            self._ring.clear()
            self._ring_samples = 0
            self._reset_state()
            if self._vad is not None and hasattr(self._vad, "reset"):
                self._vad.reset()

    def score(self, pcm_float32_mono: np.ndarray) -> float:
        frame = np.asarray(pcm_float32_mono, dtype=np.float32).reshape(-1)
        if frame.shape[0] == 0:
            return SILENT_SCORE
        with self._lock:
            self._ring.append(frame)
            self._ring_samples += frame.shape[0]
            while self._ring_samples - self._ring[0].shape[0] >= self._window_samples:
                self._ring_samples -= self._ring.popleft().shape[0]
            self._seen += frame.shape[0]
            speech = self._speech_prob(frame) >= SPEECH_PROB
            hit = self._step(speech, frame.shape[0] / SAMPLE_RATE)
            return DETECTED_SCORE if hit else SILENT_SCORE

    # ------------------------------------------------------------------ for the gate

    def confirm(self, window: np.ndarray) -> list[WordTiming]:
        """The gate's recognizer pass. When the gate asks about the audio this detector has just
        judged, the words of that pass are returned; otherwise the window is decoded, primed the
        same way."""
        with self._lock:
            cached = self._last
            if cached is not None and cached[0] == self._seen and cached[1] == int(np.size(window)):
                return list(cached[2])
        # A candidate the spotter raised, from the middle of a sentence. The gate's rhyme match
        # decides where the phrase is; the name still has to be heard surely, or again unprimed,
        # or the priming alone could have written it.
        audio = np.asarray(window, dtype=np.float32)
        words = list(self._decode(audio))
        name = _phrase_words(self.phrase)[-1:]
        sure = any(_same_word(tok, name[0]) and conf >= NAME_CONFIDENCE
                   for tok, conf in _tokens_with_confidence(words)) if name else False
        if not sure and not hears_name(self._decode(audio, primed=False), self.phrase):
            return []
        return words

    # ------------------------------------------------------------------ internal

    def _reset_state(self) -> None:
        self._seen = 0                 # samples scored since the last reset
        self._in_speech = False
        self._speech_run = 0.0         # seconds of speech without a quiet frame
        self._quiet_run = QUIET_BEFORE_S   # seconds of quiet without speech; starts "quiet enough"
        self._since_onset = 0.0        # seconds since the utterance began
        self._speech_s = 0.0           # seconds of speech in this utterance
        self._checks = 0               # recognizer passes spent on this utterance
        self._done = False             # this utterance has had its verdict
        self._last: tuple[int, int, list[WordTiming]] | None = None

    def _speech_prob(self, frame: np.ndarray) -> float:
        if self._vad is None:
            self._vad = _Vad()
        return float(self._vad(frame))

    def _step(self, speech: bool, dt: float) -> bool:
        """Advance the utterance state by one frame of `dt` seconds; True when a candidate is
        raised on this frame."""
        if speech:
            self._speech_run += dt
        else:
            self._speech_run = 0.0
        if not self._in_speech:
            if speech and self._speech_run >= ONSET_S and self._quiet_run >= QUIET_BEFORE_S:
                self._in_speech = True
                self._since_onset = self._speech_run
                self._speech_s = self._speech_run
                self._checks = 0
                self._done = False
                self._quiet_run = 0.0
            elif not speech:
                self._quiet_run += dt
            return False

        self._since_onset += dt
        if speech:
            self._speech_s += dt
            self._quiet_run = 0.0
        else:
            self._quiet_run += dt
        ended = self._quiet_run >= END_SILENCE_S
        due = (not self._done and self._checks < len(CHECK_AFTER_S)
               and self._since_onset >= CHECK_AFTER_S[self._checks])
        at_end = (ended and not self._done and self._checks == 0
                  and self._speech_s >= MIN_SPEECH_S)
        hit = False
        if due or at_end:
            self._checks += 1
            hit = self._check()
            if hit:
                self._done = True
        if ended:
            self._in_speech = False
        return hit

    def _check(self) -> bool:
        window = np.concatenate(list(self._ring)) if self._ring else np.zeros(0, dtype=np.float32)
        try:
            words = list(self._decode(window))
            confidence = phrase_start_confidence(words, self.phrase)
            second = None
            if confidence is not None and confidence < NAME_CONFIDENCE:
                second = hears_name(self._decode(window, primed=False), self.phrase)
        except Exception as exc:  # noqa: BLE001 - a recognizer failure is no candidate, never a wake
            log.warning("wake onset check failed: %s", type(exc).__name__)
            return False
        hit = confidence is not None and (confidence >= NAME_CONFIDENCE or bool(second))
        # What the check decided, never what was said: counts, a confidence and two verdicts.
        log.info("wake onset check: pass=%d speech_s=%.2f words=%d begins_with_phrase=%s "
                 "name_confidence=%s unprimed_hears_name=%s candidate=%s",
                 self._checks, self._speech_s, len(words), confidence is not None,
                 "n/a" if confidence is None else f"{confidence:.3f}",
                 "n/a" if second is None else second, hit)
        if not hit:
            return False
        self._last = (self._seen, int(window.shape[0]), words)
        return True

    def _decode(self, window: np.ndarray, *, primed: bool = True) -> Sequence[WordTiming]:
        if self._transcribe is None:
            from app.local.voice.adapters.wake_confirm import WhisperConfirm

            confirm = WhisperConfirm()
            self._transcribe = lambda audio, prompt: confirm(audio, prompt=prompt)
        return self._transcribe(window, self._prompt if primed else "")


class HybridWakeDetector:
    """Speech onset first, with the keyword spotter beside it for a phrase said mid-sentence.

    Either one raises the candidate; the gate's voice check and confirm run after both."""

    patience_frames = 1

    def __init__(self, onset: OnsetPhraseDetector, spotter=None) -> None:
        self._onset = onset
        self._spotter = spotter
        self.model_name = f"hybrid:{onset.phrase.replace(' ', '_')}"

    def reset(self) -> None:
        self._onset.reset()
        if self._spotter is not None:
            self._spotter.reset()

    def score(self, pcm_float32_mono: np.ndarray) -> float:
        spotted = self._spotter.score(pcm_float32_mono) if self._spotter is not None else SILENT_SCORE
        return max(self._onset.score(pcm_float32_mono), spotted)

    def confirm(self, window: np.ndarray) -> list[WordTiming]:
        return self._onset.confirm(window)
