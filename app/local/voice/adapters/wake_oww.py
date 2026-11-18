"""WakeDetector adapter: an openWakeWord model, by default the synthetic "hey vox" head.

The head in `app/local/voice/models/hey_vox.onnx` is produced by `tools/train_wake_model.py`
from clips made by `tools/wake_synth_clips.py`; its manifest records the data and the
validation numbers. Any other openWakeWord model (a stock name such as "hey_jarvis" or a path to
an .onnx file) can be set in VOXSCRIBE_WAKE_MODEL; the gate's recognizer confirm still listens
for the configured phrase, so a stock model alone cannot wake the product on its own phrase.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from voxscribe.wake.gate import PATIENCE_FRAMES

WAKE_MODEL_ENV = "VOXSCRIBE_WAKE_MODEL"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hey_vox.onnx"
_INT16_SCALE = 32767


def configured_model() -> str:
    return os.environ.get(WAKE_MODEL_ENV, "").strip() or str(DEFAULT_MODEL)


class OwwWakeDetector:
    """`voxscribe.session.ports.WakeDetector` over openwakeword.model.Model."""

    patience_frames = PATIENCE_FRAMES

    def __init__(self, model: str | Path | None = None) -> None:
        from openwakeword.model import Model

        spec = str(model) if model is not None else configured_model()
        self._model = Model(wakeword_models=[spec], inference_framework="onnx")
        # Take the prediction key from the loaded model: a name such as "hey_jarvis_v0.1" has a
        # dot that a file-stem guess would cut.
        self._key = next(iter(self._model.models))
        self.model_name = f"oww:{self._key}"

    def reset(self) -> None:
        self._model.reset()

    def score(self, pcm_float32_mono: np.ndarray) -> float:
        x = np.asarray(pcm_float32_mono, dtype=np.float32)
        pcm16 = (x * _INT16_SCALE).clip(-_INT16_SCALE - 1, _INT16_SCALE).astype(np.int16)
        return float(self._model.predict(pcm16).get(self._key, 0.0))
