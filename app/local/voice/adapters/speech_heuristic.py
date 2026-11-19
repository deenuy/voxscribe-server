"""Speech over the heuristic turn detector (end_of_turn.TurnDetector) and the Silero voice check.

`process`/`tick` drive the TURN policy on the detector's own stream timeline: samples fed since
`reset()`. `tick` raises ClockDomainError when handed any other clock. `voice_ratio` is the wake
gate's voice check (Silero, the model faster-whisper ships), loaded on first use.
"""
from __future__ import annotations

import numpy as np
from voxscribe.session.ports import TurnDecision

FRAME_SAMPLES = 512        # TurnDetector frame: 32 ms at 16 kHz
INT16_SCALE = 32767


class HeuristicSpeech:
    def __init__(self, pause_s: float | None = None) -> None:
        self._td = None
        self._voice = None
        self._pause_s = pause_s
        self._room = None

    def set_room(self, pcm_float32_mono: np.ndarray | None) -> None:
        """The room as it sounds when nobody speaks, for the detector to settle on at the next
        reset() (end_of_turn.TurnDetector._settle). None: settle on silence."""
        self._room = None if pcm_float32_mono is None else np.asarray(pcm_float32_mono,
                                                                      dtype=np.float32)

    def set_pause(self, pause_s: float | None) -> None:
        """The end-of-turn pause for the next turn (the product profile's pace). Takes effect at
        the next reset()."""
        if pause_s != self._pause_s:
            self._pause_s = pause_s
            self._td = None

    def reset(self) -> None:
        """Raises RuntimeError when the speech stack is not installed (the host falls back)."""
        if self._td is None:
            from app.local.end_of_turn import TurnDetector

            self._td = TurnDetector(self._pause_s)
        self._td.reset(self._room)

    def process(self, pcm_float32_mono: np.ndarray) -> list[TurnDecision]:
        x = np.asarray(pcm_float32_mono)
        if x.shape[0] != FRAME_SAMPLES:
            return []          # the detector takes whole 32 ms frames only
        pcm16 = (x * INT16_SCALE).clip(-INT16_SCALE - 1, INT16_SCALE).astype("<i2")
        return _decisions(self._td.push_pcm16(pcm16.tobytes(), self._td.stream_ms))

    def tick(self, stream_ms: int) -> list[TurnDecision]:
        return _decisions(self._td.tick(stream_ms))

    def update_text(self, text: str) -> None:
        if self._td is not None:
            self._td.update_text(text)

    def voice_ratio(self, pcm_float32_mono: np.ndarray) -> float:
        if self._voice is None:
            from app.local.voice.adapters.wake_confirm import SileroVoiceRatio

            self._voice = SileroVoiceRatio()
        return self._voice(pcm_float32_mono)


def _decisions(d) -> list[TurnDecision]:
    if d is None:
        return []
    return [TurnDecision(closes_turn=bool(d.closes_turn), action=str(d.action),
                         silence_ms=int(d.silence_ms))]
