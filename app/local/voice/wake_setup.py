"""Builds the wake gate from settings, for the live wake loop and the offline replay harness.

VOXSCRIBE_WAKE_DETECTOR picks the stage 1 adapter:

  openvocab  sherpa-onnx keyword spotting on the plain-text phrase (default, chosen by the T02
             bake-off: equal fixture recall, better held-out recall, no training step)
  oww        an openWakeWord model, by default the synthetic "hey vox" head; VOXSCRIBE_WAKE_MODEL
             overrides the model

Both run behind the same two-stage gate (voice check, then recognizer confirm on "hey vox").
"""
from __future__ import annotations

import os

from voxscribe.session.ports import WakeDetector
from voxscribe.wake.gate import ASR_CONFIRM_PHRASE, GateConfig, WakeGate

DETECTOR_ENV = "VOXSCRIBE_WAKE_DETECTOR"
OPENVOCAB = "openvocab"
OWW = "oww"
DEFAULT_DETECTOR = OPENVOCAB
DETECTORS = (OPENVOCAB, OWW)


def configured_detector() -> str:
    name = os.environ.get(DETECTOR_ENV, "").strip().lower() or DEFAULT_DETECTOR
    if name not in DETECTORS:
        raise ValueError(f"{DETECTOR_ENV}={name!r}; expected one of {', '.join(DETECTORS)}")
    return name


def make_detector(name: str | None = None, *, oww_model: str | None = None,
                  phrase: str = ASR_CONFIRM_PHRASE) -> WakeDetector:
    name = name or configured_detector()
    if name == OPENVOCAB:
        from app.local.voice.adapters.wake_openvocab import OpenVocabWakeDetector
        return OpenVocabWakeDetector(keyword=phrase)
    if name == OWW:
        from app.local.voice.adapters.wake_oww import OwwWakeDetector
        return OwwWakeDetector(oww_model)
    raise ValueError(f"unknown wake detector {name!r}")


def make_gate(detector: WakeDetector, *, voice: bool = True, confirm: bool = True,
              phrase: str = ASR_CONFIRM_PHRASE) -> WakeGate:
    """The production gate. `voice` and `confirm` exist for the bake-off's stage-1-only rows."""
    voice_ratio = None
    transcribe = None
    if voice:
        from app.local.voice.adapters.wake_confirm import SileroVoiceRatio
        voice_ratio = SileroVoiceRatio()
    if confirm:
        from app.local.voice.adapters.wake_confirm import WhisperConfirm
        transcribe = WhisperConfirm()
        transcribe.warm()
    return WakeGate(detector, voice_ratio=voice_ratio, transcribe=transcribe,
                    config=GateConfig(phrase=phrase))
