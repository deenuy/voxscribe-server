"""Stage 2 helpers for the wake gate: the Silero voice check and the Whisper confirm pass.

Both are injected into `voxscribe.wake.gate.WakeGate` as plain callables:

* `SileroVoiceRatio(window) -> float`: fraction of 32 ms Silero hops in the window whose speech
  probability is at least VAD_SPEECH_PROB. Uses the Silero v6 model that faster-whisper ships,
  through voxscribe-core's `SileroDetector`.
* `WhisperConfirm(window) -> list[WordTiming]`: one greedy faster-whisper pass with word timings
  over the 2 s before a candidate. The gate does the phrase matching; this adapter only
  transcribes and never logs or stores the words.

The confirm model is separate from the session recognizer so a candidate does not queue behind
a live transcription; base.en keeps the pass near 300 ms on a laptop CPU.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
from voxscribe.models import WordTiming

log = logging.getLogger("app.local.voice.wake_confirm")

VAD_SPEECH_PROB = 0.5
CONFIRM_MODEL_ENV = "VOXSCRIBE_WAKE_CONFIRM_MODEL"
DEFAULT_CONFIRM_MODEL = "base.en"
CONFIRM_BEAM = 1
CONFIRM_LANGUAGE = "en"
_MS_PER_S = 1000.0


class SileroVoiceRatio:
    def __init__(self) -> None:
        from voxscribe.speech.silero import SileroDetector

        self._det = SileroDetector()
        self._lock = threading.Lock()

    def __call__(self, window: np.ndarray) -> float:
        with self._lock:
            self._det.reset()
            frames = self._det.process(np.asarray(window, dtype=np.float32))
        if not frames:
            return 0.0
        return sum(1 for f in frames if f.probability >= VAD_SPEECH_PROB) / len(frames)


class WhisperConfirm:
    def __init__(self, model_size: str | None = None) -> None:
        self.model_size = model_size or os.environ.get(CONFIRM_MODEL_ENV, "").strip() \
            or DEFAULT_CONFIRM_MODEL
        self._model = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Load the model now so the first candidate does not pay for it."""
        self._load()

    def _load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                log.info("wake confirm model loaded model=%s", self.model_size)
            return self._model

    def __call__(self, window: np.ndarray, prompt: str | None = None) -> list[WordTiming]:
        """`prompt` primes the recognizer with the phrase being listened for ("Hey Vox."), so an
        unusual name comes back spelled as written instead of as its nearest common word."""
        model = self._load()
        audio = np.asarray(window, dtype=np.float32)
        if audio.shape[0] == 0:
            return []
        segments, _ = model.transcribe(
            audio,
            beam_size=CONFIRM_BEAM,
            language=CONFIRM_LANGUAGE,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        out: list[WordTiming] = []
        for seg in segments:
            for w in seg.words or ():
                out.append(WordTiming(w.word, w.start * _MS_PER_S, w.end * _MS_PER_S,
                                      float(w.probability)))
        return out
