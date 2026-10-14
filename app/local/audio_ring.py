"""Per-session bounded audio ring: decouples WS receive from push_audio.

The WS receive coroutine pushes frames to the ring without blocking on ASR.
A separate asyncio Task drains the ring and calls push_audio. If the ring
overflows (consumer too slow), the OLDEST frames are dropped and reported
so the caller can warn the client and increment dropped_audio_ms.

Design mirrors the CaptureRing in voxscribe-core parked/s5-capture, ported
to the server live path so voxscribe-core's listen path stays frozen.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2


def frame_ms(frame: bytes) -> float:
    """Duration of a raw PCM16 audio payload in milliseconds."""
    n_samples = len(frame) / BYTES_PER_SAMPLE
    return n_samples / SAMPLE_RATE * 1000.0


@dataclass
class RingStats:
    dropped_ms: float = 0.0
    overflow_count: int = 0
    pushed_ms: float = 0.0
    peak_fill_ms: float = 0.0


class AudioRing:
    """Bounded FIFO of raw PCM16 audio frames for one session.

    Push is non-blocking and never raises. When a new frame would exceed
    capacity, the OLDEST unread frames are evicted first and their duration
    is added to stats.dropped_ms. The caller receives the evicted ms so it
    can emit a warning immediately.

    Pop is an awaitable that returns None only after close() is called and
    the ring is empty.
    """

    def __init__(self, capacity_ms: float = 60_000.0):
        self._capacity_ms = capacity_ms
        self._frames: deque[bytes] = deque()
        self._fill_ms: float = 0.0
        self.stats = RingStats()
        self._closed = False
        self._not_empty = asyncio.Event()

    # ------------------------------------------------------------------
    # Producer side (called from asyncio coroutine)
    # ------------------------------------------------------------------

    def push(self, frame: bytes) -> float:
        """Enqueue a PCM16 frame. Returns evicted ms (0 when no overflow)."""
        ms = frame_ms(frame)
        self.stats.pushed_ms += ms

        evicted_ms = 0.0
        while self._frames and self._fill_ms + ms > self._capacity_ms:
            oldest = self._frames.popleft()
            old_ms = frame_ms(oldest)
            self._fill_ms -= old_ms
            evicted_ms += old_ms

        if evicted_ms:
            self.stats.dropped_ms += evicted_ms
            self.stats.overflow_count += 1

        self._frames.append(frame)
        self._fill_ms += ms
        if self._fill_ms > self.stats.peak_fill_ms:
            self.stats.peak_fill_ms = self._fill_ms
        self._not_empty.set()
        return evicted_ms

    def close(self) -> None:
        """Signal that no more frames will be pushed. Wakes a waiting pop."""
        self._closed = True
        self._not_empty.set()

    # ------------------------------------------------------------------
    # Consumer side (awaited by a single asyncio Task)
    # ------------------------------------------------------------------

    async def pop(self) -> bytes | None:
        """Return next frame, or None when closed and empty."""
        while True:
            if self._frames:
                frame = self._frames.popleft()
                self._fill_ms -= frame_ms(frame)
                if not self._frames:
                    self._not_empty.clear()
                return frame
            if self._closed:
                return None
            self._not_empty.clear()
            await self._not_empty.wait()

    @property
    def fill_ms(self) -> float:
        return self._fill_ms

    @property
    def empty(self) -> bool:
        return not self._frames
