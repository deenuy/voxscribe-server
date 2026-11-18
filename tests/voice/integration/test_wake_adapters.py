"""Wake adapters and the full two-stage gate on real audio (T02). Integration: loads real models.

Audio comes from the PRD fixtures when the workspace has them, otherwise from macOS `say`.
Skipped when neither source or a model file is available (run `uv run tools/fetch_wake_models.py`).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from voxscribe.wake.gate import FRAME_SAMPLES, MIN_SPEECH_RATIO, SAMPLE_RATE, RejectReason
from voxscribe.wake.phrase import match_phrase

pytestmark = pytest.mark.integration

_WORKSPACE = Path(__file__).resolve().parents[4]
_FIXTURES = _WORKSPACE / "ideascribe-app" / "docs" / "PRD" / "fixtures"
_LEAD_S, _TAIL_S = 1.0, 1.5
_NOISE_RMS = 1e-4


def _load(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = resample_poly(data, SAMPLE_RATE, sr).astype(np.float32)
    return data


def _speech(text: str, tmp_path: Path, fixture: str | None = None) -> np.ndarray:
    if fixture and (_FIXTURES / fixture).is_file():
        return _load(_FIXTURES / fixture)
    if shutil.which("say") is None:
        pytest.skip("no fixture and no macOS `say`")
    out = tmp_path / f"{text.replace(' ', '_')}.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(out), "--data-format=LEI16@16000", text],
                   check=True)
    return _load(out)


def _stream(clip: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    lead, tail = int(_LEAD_S * SAMPLE_RATE), int(_TAIL_S * SAMPLE_RATE)
    out = (rng.standard_normal(lead + clip.shape[0] + tail) * _NOISE_RMS).astype(np.float32)
    out[lead: lead + clip.shape[0]] += clip
    return out


def _frames(audio: np.ndarray):
    for i in range(audio.shape[0] // FRAME_SAMPLES):
        yield audio[i * FRAME_SAMPLES: (i + 1) * FRAME_SAMPLES]


@pytest.fixture(scope="module")
def tmp_audio(tmp_path_factory):
    return tmp_path_factory.mktemp("wake_audio")


@pytest.fixture(scope="module")
def hey_vox(tmp_audio):
    return _speech("hey vox", tmp_audio, fixture="hey_vox_samantha.aiff")


@pytest.fixture(scope="module")
def hey_jarvis(tmp_audio):
    return _speech("hey jarvis", tmp_audio)


def _detector(name: str):
    from app.local.voice.wake_setup import make_detector
    try:
        return make_detector(name)
    except Exception as exc:  # model files missing on this machine
        pytest.skip(f"{name} detector unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("name", ["openvocab", "oww"])
def test_detector_satisfies_the_port_and_scores_hey_vox_not_hey_jarvis(name, hey_vox, hey_jarvis):
    from voxscribe.session.ports import WakeDetector
    det = _detector(name)
    assert isinstance(det, WakeDetector)

    def peak(audio):
        det.reset()
        return max(det.score(f) for f in _frames(_stream(audio)))

    assert peak(hey_vox) >= 0.5
    assert peak(hey_jarvis) < 0.5


def test_silero_voice_ratio_separates_speech_from_noise(hey_vox):
    from app.local.voice.adapters.wake_confirm import SileroVoiceRatio
    vr = SileroVoiceRatio()
    clip = hey_vox[-SAMPLE_RATE:]
    noise = (np.random.default_rng(1).standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
    assert vr(clip) >= MIN_SPEECH_RATIO
    assert vr(noise) < MIN_SPEECH_RATIO


def test_whisper_confirm_words_match_hey_vox_only(hey_vox, hey_jarvis):
    from app.local.voice.adapters.wake_confirm import WhisperConfirm
    confirm = WhisperConfirm()
    words = confirm(_stream(hey_vox)[-2 * SAMPLE_RATE:])
    assert match_phrase(words, "hey vox") is not None
    assert match_phrase(confirm(_stream(hey_jarvis)[-2 * SAMPLE_RATE:]), "hey vox") is None


def test_full_gate_accepts_hey_vox_and_rejects_hey_jarvis_behind_the_stock_jarvis_model(
        hey_vox, hey_jarvis):
    """With the stock hey_jarvis detector, "hey jarvis" still cannot wake: the confirm refuses it."""
    from app.local.voice.adapters.wake_oww import OwwWakeDetector
    from app.local.voice.wake_setup import make_gate
    try:
        jarvis = OwwWakeDetector("hey_jarvis_v0.1")
    except Exception as exc:
        pytest.skip(f"stock model unavailable: {exc}")
    gate = make_gate(jarvis)

    def decisions(audio):
        gate.reset()
        return [d for d in (gate.process(f) for f in _frames(_stream(audio))) if d is not None]

    jarvis_decisions = decisions(hey_jarvis)
    assert not any(d.accepted for d in jarvis_decisions)
    assert any(d.reason is RejectReason.NO_MATCH for d in jarvis_decisions)

    vox_gate = make_gate(_detector("openvocab"))
    vox_gate.reset()
    vox = [d for d in (vox_gate.process(f) for f in _frames(_stream(hey_vox))) if d is not None]
    assert [d.accepted for d in vox] == [True]
    assert vox[0].phrase_end_sample is not None
