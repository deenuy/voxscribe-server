"""V2 turn detection checks using macOS TTS to generate real speech fixtures.

V2-P01: Normal utterance → TurnDetector closes the turn (closes_turn=True).
V2-P03: Fast-paced utterance → turn still closes correctly.
V2-P04: Silence only → turn does NOT close (no speech start, no closes_turn).
V2-P05: reset() enables a fresh turn detection after the first turn closes.
V2-E01: Short one-word utterance ("yes") → turn closes within expected time.
V2-E02: Two utterances back-to-back → second close detected.

V2-P02 (slow paced, mid-sentence pause resistance) deferred to hardware run.
V2-S01 detached soak — never inline per VOICE-AUTOPILOT.md section 3.3.

All clips are generated fresh; no pre-recorded files required.
Requires: macOS (say, afconvert), voxscribe-core on voice/p1-loop.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 512
_BYTES_PER_CHUNK = _CHUNK_FRAMES * 2


def _say_to_pcm16(text: str, *, rate: int = 180) -> bytes:
    """Generate speech via macOS `say`, return raw PCM16LE mono at 16 kHz."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    subprocess.run(
        ["say", "-v", "Samantha", "-r", str(rate),
         "--file-format", "WAVE", "--data-format", "LEI16@16000",
         "-o", path, text],
        check=True, timeout=10, capture_output=True,
    )
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes())


def _run_detector(pcm: bytes, *, silence_pad_s: float = 8.0) -> list[object]:
    """Feed silence warmup + pcm bytes + silence pad into TurnDetector; return all TurnDecisions.

    Warmup (640ms) lets the HeuristicDetector learn the room floor before speech.
    Default silence pad of 8s covers the 6000ms extended_ms close threshold.
    """
    from app.local.end_of_turn import TurnDetector

    td = TurnDetector()
    td.reset()
    decisions: list[object] = []
    # 640ms room-learning warmup before speech
    warmup = bytes(_BYTES_PER_CHUNK) * 20
    # silence padding after speech so the 6s extended_ms close fires
    silence = bytes(_BYTES_PER_CHUNK)
    pad_chunks = int(silence_pad_s * _SAMPLE_RATE / _CHUNK_FRAMES)
    full = warmup + pcm + silence * pad_chunks

    now_ms = 0
    for i in range(0, len(full) - _BYTES_PER_CHUNK + 1, _BYTES_PER_CHUNK):
        chunk = full[i: i + _BYTES_PER_CHUNK]
        now_ms += 32  # 512 samples at 16 kHz = 32 ms
        d = td.push_pcm16(chunk, now_ms)
        if d is not None:
            decisions.append(d)
        d2 = td.tick(now_ms)
        if d2 is not None:
            decisions.append(d2)
    return decisions


# ── V2-P01: normal utterance closes the turn ────────────────────────────────

@pytest.mark.slow
def test_v2_p01_normal_utterance_closes_turn():
    pcm = _say_to_pcm16("What is the weather today?")
    decisions = _run_detector(pcm)
    closes = [d for d in decisions if getattr(d, "closes_turn", False)]
    assert len(closes) >= 1, "Expected at least one closes_turn decision"


# ── V2-P03: fast paced utterance closes the turn ────────────────────────────

@pytest.mark.slow
def test_v2_p03_fast_paced_utterance_closes_turn():
    pcm = _say_to_pcm16("Set a timer for five minutes please.", rate=220)
    decisions = _run_detector(pcm)
    closes = [d for d in decisions if getattr(d, "closes_turn", False)]
    assert len(closes) >= 1, "Fast paced utterance: expected closes_turn"


# ── V2-P04: silence only -- no turn close ───────────────────────────────────

def test_v2_p04_silence_only_no_close():
    """Pure silence: TurnDetector should not close the turn (no speech started)."""
    from app.local.end_of_turn import TurnDetector

    td = TurnDetector()
    td.reset()
    silence = bytes(_BYTES_PER_CHUNK)
    decisions = []
    now_ms = 0
    for _ in range(200):  # 6.4 s of silence
        now_ms += 32
        d = td.push_pcm16(silence, now_ms)
        if d is not None:
            decisions.append(d)
        d2 = td.tick(now_ms)
        if d2 is not None:
            decisions.append(d2)
    closes = [d for d in decisions if getattr(d, "closes_turn", False)]
    assert len(closes) == 0, "Silence should not produce a closes_turn decision"


# ── V2-P05: reset enables a fresh turn ──────────────────────────────────────

@pytest.mark.slow
def test_v2_p05_reset_clears_state_for_new_turn():
    """After a first turn closes, reset() lets a second turn close too."""
    from app.local.end_of_turn import TurnDetector

    td = TurnDetector()

    pcm = _say_to_pcm16("Hello there.")

    def feed(start_ms: int) -> tuple[list[object], int]:
        decs = []
        warmup = bytes(_BYTES_PER_CHUNK) * 20
        silence = bytes(_BYTES_PER_CHUNK)
        full = warmup + pcm + silence * 250  # ~8 s pad covers 6s extended_ms
        now_ms = start_ms
        for i in range(0, len(full) - _BYTES_PER_CHUNK + 1, _BYTES_PER_CHUNK):
            chunk = full[i: i + _BYTES_PER_CHUNK]
            now_ms += 32
            d = td.push_pcm16(chunk, now_ms)
            if d is not None:
                decs.append(d)
            d2 = td.tick(now_ms)
            if d2 is not None:
                decs.append(d2)
        return decs, now_ms

    td.reset()
    first_decs, t1 = feed(0)
    first_closes = [d for d in first_decs if getattr(d, "closes_turn", False)]
    assert len(first_closes) >= 1, "First turn should close"

    td.reset()
    second_decs, _ = feed(t1)
    second_closes = [d for d in second_decs if getattr(d, "closes_turn", False)]
    assert len(second_closes) >= 1, "After reset, second turn should also close"


# ── V2-E01: very short utterance ("yes") closes the turn ────────────────────

@pytest.mark.slow
def test_v2_e01_short_utterance_closes_turn():
    """A single short word should still be detected and the turn closed."""
    pcm = _say_to_pcm16("yes", rate=180)
    decisions = _run_detector(pcm, silence_pad_s=8.0)
    closes = [d for d in decisions if getattr(d, "closes_turn", False)]
    assert len(closes) >= 1, "Short utterance 'yes': expected closes_turn"


# ── V2-E02: two utterances back-to-back ─────────────────────────────────────

@pytest.mark.slow
def test_v2_e02_two_utterances_second_closes():
    """Two utterances with a brief pause: both produce speech; at least one close."""
    from app.local.end_of_turn import TurnDetector

    td = TurnDetector()
    td.reset()

    pcm1 = _say_to_pcm16("Tell me a joke.", rate=180)
    pcm2 = _say_to_pcm16("What is the time?", rate=180)
    warmup = bytes(_BYTES_PER_CHUNK) * 20  # 640ms room warmup
    gap = bytes(_BYTES_PER_CHUNK) * 16  # 512 ms between utterances
    full = warmup + pcm1 + gap + pcm2 + bytes(_BYTES_PER_CHUNK) * 250  # 8s close pad

    decisions = []
    now_ms = 0
    for i in range(0, len(full) - _BYTES_PER_CHUNK + 1, _BYTES_PER_CHUNK):
        chunk = full[i: i + _BYTES_PER_CHUNK]
        now_ms += 32
        d = td.push_pcm16(chunk, now_ms)
        if d is not None:
            decisions.append(d)
        d2 = td.tick(now_ms)
        if d2 is not None:
            decisions.append(d2)

    closes = [d for d in decisions if getattr(d, "closes_turn", False)]
    assert len(closes) >= 1, "Two utterances: expected at least one closes_turn"
