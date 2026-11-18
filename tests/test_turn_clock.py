"""T04 clock contract on the server host: TurnDetector.tick() takes stream time.

Defect C (ROOT-CAUSE section 4, 2026-11-17): the wake loop ticked with engine
uptime while the detector stamps pauses on stream time, so every pause looked as
long as the uptime and closed the turn at the first tick. The wake loop now
ticks with samples fed since reset(); TurnDetector refuses a tick from another
clock so the mix-up cannot return silently.
"""

from __future__ import annotations

import pytest

from app.local.end_of_turn import ClockDomainError, TurnDetector

_SAMPLE_RATE = 16_000
_CHUNK_FRAMES = 512
_SILENCE = bytes(_CHUNK_FRAMES * 2)
_UPTIME_MS = int(17.8 * 3_600_000)  # the uptime in the logged evidence


def _feed(td: TurnDetector, chunks: int) -> int:
    stream_samples = 0
    for _ in range(chunks):
        stream_samples += _CHUNK_FRAMES
        td.push_pcm16(_SILENCE, stream_samples * 1000 // _SAMPLE_RATE)
    return stream_samples * 1000 // _SAMPLE_RATE


def test_stream_ms_counts_audio_fed_since_reset():
    td = TurnDetector()
    td.reset()
    stream_ms = _feed(td, 50)
    assert td.stream_ms == stream_ms
    td.tick(stream_ms)  # the wake loop's clock: accepted
    td.reset()
    assert td.stream_ms == 0


def test_tick_with_engine_uptime_raises_clock_domain_error():
    td = TurnDetector()
    td.reset()
    stream_ms = _feed(td, 50)
    with pytest.raises(ClockDomainError):
        td.tick(_UPTIME_MS + stream_ms)
