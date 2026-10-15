"""Per-session bounded audio ring: decouples WS receive from push_audio.

Two policies:
  "drop_oldest"   -- sync push(), evicts oldest frames on overflow (mic/wake path).
  "wait_for_space" -- async push_wait(), waits for the consumer to pop before adding;
                      falls back to drop-oldest only on timeout (WS path).

Design mirrors the CaptureRing in voxscribe-core parked/s5-capture, ported to the
server live path so voxscribe-core's listen path stays frozen.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

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

    Policy "drop_oldest" (mic/wake path):
      push() is sync and never blocks. When a new frame would exceed capacity,
      the oldest unread frames are evicted. The caller receives the evicted ms
      so it can emit a warning immediately.

    Policy "wait_for_space" (WS path):
      push_wait() is async. When the ring is full, it awaits the consumer to
      pop at least one frame (up to full_timeout_s), then inserts. If the
      timeout expires, it falls back to drop-oldest. This applies TCP
      backpressure to the WebSocket client and avoids discarding dictation.

    pop() is awaitable and returns None only after close() is called and the
    ring is empty.
    """

    def __init__(
        self,
        capacity_ms: float = 60_000.0,
        policy: Literal["drop_oldest", "wait_for_space"] = "drop_oldest",
        full_timeout_s: float = 5.0,
    ):
        self._capacity_ms = capacity_ms
        self._policy = policy
        self._full_timeout_s = full_timeout_s
        self._frames: deque[bytes] = deque()
        self._fill_ms: float = 0.0
        self.stats = RingStats()
        self._closed = False
        self._not_empty = asyncio.Event()
        self._not_full = asyncio.Event()
        self._not_full.set()  # ring starts with space

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def push(self, frame: bytes) -> float:
        """Sync push, drop-oldest policy. Returns evicted ms (0 = no overflow).

        Use for mic/wake path. For WS path, call push_wait() instead.
        """
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
        if self._fill_ms < self._capacity_ms:
            self._not_full.set()
        return evicted_ms

    async def push_wait(self, frame: bytes, timeout: float | None = None) -> float:
        """Async push, wait-for-space policy. Returns evicted ms (0 = no drop).

        Waits up to `timeout` seconds (default: full_timeout_s) for the consumer
        to pop a frame before inserting. Falls back to drop-oldest on timeout.
        Use for the WebSocket path.
        """
        ms = frame_ms(frame)
        if self._fill_ms + ms > self._capacity_ms:
            t = timeout if timeout is not None else self._full_timeout_s
            try:
                await asyncio.wait_for(self._wait_for_space(ms), timeout=t)
            except asyncio.TimeoutError:
                pass  # fall through to drop-oldest
        return self.push(frame)

    async def _wait_for_space(self, ms: float) -> None:
        while self._fill_ms + ms > self._capacity_ms and not self._closed:
            self._not_full.clear()
            # Double-check after clearing to avoid a race with pop()
            if self._fill_ms + ms > self._capacity_ms and not self._closed:
                await self._not_full.wait()

    def close(self) -> None:
        """Signal that no more frames will be pushed. Wakes a waiting pop."""
        self._closed = True
        self._not_empty.set()
        self._not_full.set()  # unblock any waiting push_wait so it can exit

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
                # Signal that space is available for a waiting push_wait()
                self._not_full.set()
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
