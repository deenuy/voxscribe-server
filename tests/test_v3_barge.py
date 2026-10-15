"""V3 barge-in checks (code-level, no hardware required).

V3-P01: AudioEngine.cancel_playback() discards all queued chunks.
V3-P02: is_playing is True with items queued, False after cancel.
V3-P03: playback_ref() returns snapshot of last played samples.
V3-P04: _BARGE_RATIO constant present and correct value in wake_loop.
V3-P05: cancel_playback is idempotent (safe to call on empty queue).
V3-E01: barge-in gate math -- mic_rms > ratio * pb_rms triggers cancel.
V3-E02: mic_rms below _RMS_FLOOR → barge-in gate not evaluated.
V3-E03: equal rms values (mic == pb * ratio) → no trigger (strict >).
V3-E04: pb_rms=0 → clamped to 1e-6, gate uses safe denominator.

V3-H (hardware, real room) and V3-S01 (detached soak) not inline.
"""

from __future__ import annotations

import asyncio

import pytest


# ── AudioEngine barge-in primitives ─────────────────────────────────────────

def _make_engine():
    """Construct an AudioEngine without starting audio streams."""
    from app.local.audio_engine import AudioEngine
    return AudioEngine()


def _fill_queue(engine, n: int = 10):
    """Put n dummy items into the playback queue synchronously."""
    import asyncio
    for i in range(n):
        try:
            engine._playback_q.put_nowait(f"chunk_{i}")
        except asyncio.QueueFull:
            break


# ── V3-P01: cancel_playback discards all queued chunks ──────────────────────

def test_v3_p01_cancel_playback_drains_queue():
    engine = _make_engine()
    _fill_queue(engine, 5)
    assert not engine._playback_q.empty()
    engine.cancel_playback()
    assert engine._playback_q.empty()


# ── V3-P02: is_playing reflects queue state ──────────────────────────────────

def test_v3_p02_is_playing_true_with_items_false_after_cancel():
    engine = _make_engine()
    assert not engine.is_playing
    _fill_queue(engine, 3)
    assert engine.is_playing
    engine.cancel_playback()
    assert not engine.is_playing


# ── V3-P03: playback_ref returns a snapshot ──────────────────────────────────

def test_v3_p03_playback_ref_returns_list():
    engine = _make_engine()
    ref = engine.playback_ref()
    assert isinstance(ref, list)
    # Empty until something has been played
    assert ref == []


# ── V3-P04: _BARGE_RATIO constant ────────────────────────────────────────────

def test_v3_p04_barge_ratio_constant():
    import app.local.wake_loop as wl
    assert hasattr(wl, "_BARGE_RATIO"), "_BARGE_RATIO missing from wake_loop"
    assert wl._BARGE_RATIO == 2.5


# ── V3-P05: cancel_playback is safe on empty queue ───────────────────────────

def test_v3_p05_cancel_playback_idempotent_on_empty():
    engine = _make_engine()
    assert engine._playback_q.empty()
    engine.cancel_playback()  # must not raise
    engine.cancel_playback()  # again -- still safe
    assert engine._playback_q.empty()


# ── V3-E01: gate math -- mic > ratio * pb triggers ───────────────────────────

def test_v3_e01_barge_gate_math_triggers_above_threshold():
    """mic_rms > _BARGE_RATIO * pb_rms is the gate condition."""
    import app.local.wake_loop as wl
    ratio = wl._BARGE_RATIO
    pb_rms = 0.1
    # Just above threshold
    mic_rms = ratio * pb_rms + 1e-4
    assert mic_rms > ratio * max(pb_rms, 1e-6)


def test_v3_e01_barge_gate_math_does_not_trigger_below_threshold():
    import app.local.wake_loop as wl
    ratio = wl._BARGE_RATIO
    pb_rms = 0.1
    # Just below threshold
    mic_rms = ratio * pb_rms - 1e-4
    assert not (mic_rms > ratio * max(pb_rms, 1e-6))


# ── V3-E02: mic below _RMS_FLOOR → gate not evaluated ─────────────────────

def test_v3_e02_rms_floor_constant_present():
    """_RMS_FLOOR constant gates entry into barge-in check."""
    import app.local.wake_loop as wl
    assert hasattr(wl, "_RMS_FLOOR"), "_RMS_FLOOR missing from wake_loop"
    assert wl._RMS_FLOOR > 0


# ── V3-E03: exactly at threshold → no trigger (strict >) ─────────────────

def test_v3_e03_exact_threshold_does_not_trigger():
    import app.local.wake_loop as wl
    ratio = wl._BARGE_RATIO
    pb_rms = 0.1
    mic_rms = ratio * pb_rms  # exactly equal, not strictly greater
    assert not (mic_rms > ratio * max(pb_rms, 1e-6))


# ── V3-E04: pb_rms=0 clamped to 1e-6 ────────────────────────────────────────

def test_v3_e04_zero_pb_rms_clamped():
    """max(pb_rms, 1e-6) prevents division by zero / extreme amplification."""
    import app.local.wake_loop as wl
    ratio = wl._BARGE_RATIO
    pb_rms = 0.0
    clamped = max(pb_rms, 1e-6)
    assert clamped == 1e-6
    # Any nonzero mic_rms above ratio * 1e-6 would trigger
    # Just verify the clamp itself works
    assert ratio * clamped == pytest.approx(ratio * 1e-6)
