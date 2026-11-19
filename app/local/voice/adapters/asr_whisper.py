"""Recognizer over faster-whisper.

`transcribe` is the wake confirm pass (WhisperConfirm, a small model with word timings, separate
from the session recognizer so a candidate never queues behind a live turn). `open_stream` is a
voxscribe-core LiveTranscriber over the configured ASR provider, one per turn.
"""
from __future__ import annotations

import numpy as np
from voxscribe.session.ports import TranscriptionResult, TranscriptSegment

from app.local.voice.adapters.wake_confirm import WhisperConfirm

MIC_SOURCE_TAG = 0
INT16_SCALE = 32767
PARTIAL_UPDATE_INTERVAL_MS = 500
DEFAULT_MAX_SEGMENT_MS = 15_000


class WhisperRecognizer:
    def __init__(self, asr_provider=None, settings=None, confirm: WhisperConfirm | None = None):
        self._asr = asr_provider
        self._settings = settings
        self._confirm = confirm or WhisperConfirm()

    def warm(self) -> None:
        self._confirm.warm()

    def transcribe(self, pcm_float32_mono: np.ndarray) -> TranscriptionResult:
        words = tuple(self._confirm(pcm_float32_mono))
        return TranscriptionResult(text=" ".join(w.word.strip() for w in words), words=words)

    def confirm_words(self, pcm_float32_mono: np.ndarray, prompt: str | None = None):
        """The short confirm pass, primed with the phrase being listened for (the wake detector's
        recognizer pass; one model, already loaded)."""
        return self._confirm(pcm_float32_mono, prompt=prompt)

    def open_stream(self, session_id: str) -> _LiveStream | None:
        if self._asr is None:
            return None
        from voxscribe.meeting.live import LiveTranscriber

        s = self._settings
        return _LiveStream(LiveTranscriber(
            self._asr,
            session_id=session_id,
            vad_mode=getattr(s, "live_vad_mode", "adaptive"),
            max_segment_ms=getattr(s, "live_max_segment_ms", DEFAULT_MAX_SEGMENT_MS) or None,
            partial_update_interval_ms=PARTIAL_UPDATE_INTERVAL_MS,
        ), asr=self._asr)


class _LiveStream:
    """One turn's recognizer stream. After flush(), no_speech_prob is Whisper's probability that
    the final decode was not speech, or None when the ASR provider does not report it."""

    def __init__(self, transcriber, asr=None) -> None:
        self._tr = transcriber
        self._measure = getattr(asr, "measure_no_speech", None)
        self.no_speech_prob: float | None = None

    def push(self, pcm_float32_mono: np.ndarray) -> list[TranscriptSegment]:
        pcm16 = (np.asarray(pcm_float32_mono) * INT16_SCALE).clip(-INT16_SCALE - 1, INT16_SCALE)
        return _segments(self._tr.push_audio(MIC_SOURCE_TAG, pcm16.astype("<i2").tobytes()))

    def pop_partials(self) -> list[TranscriptSegment]:
        return _segments(self._tr.pop_partials())

    def flush(self) -> list[TranscriptSegment]:
        if self._measure is None:
            self.no_speech_prob = None
            return _segments(self._tr.flush())
        segs, self.no_speech_prob = self._measure(self._tr.flush)
        return _segments(segs)


def _segments(segs) -> list[TranscriptSegment]:
    return [TranscriptSegment(id=s.id, text=s.text, start_ms=s.start_ms, end_ms=s.end_ms,
                              state=s.state, source=s.source, speaker=s.speaker) for s in segs]
