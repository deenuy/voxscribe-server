"""AudioIn and AudioOut over the process AudioEngine (sounddevice / PortAudio).

The AudioEngine's lifetime belongs to the app (app.py starts and stops it, the capture watchdog
reopens it), so with an engine `start()` and `stop()` do nothing. When the engine could not
start, `PortAudioIn` opens its own capture stream between `start()` and `stop()` (the old wake
loop's fallback) and `PortAudioOut` is unavailable: replies are not synthesised, cues still play.
`monotonic_ms` is the engine's clock, shared with the echo guard; the host's Clock uses it.

Barge-in (T13): `PortAudioIn.echo_cancelled` tells the host whether capture is cleaned against
playback, which is what lets it read capture energy during a reply as the user's voice, and
`PortAudioOut.play(..., on_played=)` reports when a sentence has been played to its end.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import struct
import subprocess
import wave

import numpy as np

from app.local.audio_engine import monotonic_ms

__all__ = ["PortAudioIn", "PortAudioOut", "monotonic_ms"]

log = logging.getLogger("app.local.voice.audio")

CAPTURE_RATE = 16_000
CHUNK_FRAMES = 512            # 32 ms at the capture rate, the AudioEngine block size
DIRECT_QUEUE_CHUNKS = 512
CUE_TIMEOUT_S = 3
CUE_PLAYER = "afplay"
# A cue is muted from capture for its own length, read from the file (cue_duration_ms), and
# CHIME_DECAY_MS of room reverb after it. CHIME_DURATION_MS stands in when the file cannot be read.
# The fixed 800 ms this used to be was shorter than the cues themselves (the acknowledgement and
# end cues run about 1.6 s), so their last half second reached the recognizer and the turn
# detector as sound to judge (PL59).
CHIME_DURATION_MS = 1700
CHIME_DECAY_MS = 300          # room reverb after the cue ends, still muted from capture


@functools.lru_cache(maxsize=16)
def cue_duration_ms(path: str) -> int:
    """How long the cue at `path` plays, from its header (AIFF or WAV); CHIME_DURATION_MS when
    it cannot be read."""
    try:
        if path.lower().endswith((".wav", ".wave")):
            with wave.open(path, "rb") as w:
                return round(w.getnframes() * 1000 / w.getframerate())
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] != b"FORM" or head[8:12] not in (b"AIFF", b"AIFC"):
                return CHIME_DURATION_MS
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    return CHIME_DURATION_MS
                kind, size = chunk[:4], struct.unpack(">I", chunk[4:])[0]
                if kind == b"COMM":
                    comm = f.read(18)
                    frames = struct.unpack(">I", comm[2:6])[0]
                    rate = _extended(comm[8:18])
                    return round(frames * 1000 / rate) if rate > 0 else CHIME_DURATION_MS
                f.seek(size + (size & 1), 1)
    except (OSError, EOFError, struct.error, wave.Error, ValueError, ZeroDivisionError):
        return CHIME_DURATION_MS


def _extended(b: bytes) -> float:
    """An AIFF sample rate: an 80-bit IEEE 754 extended float, big-endian."""
    exponent = ((b[0] & 0x7F) << 8 | b[1]) - 16383
    mantissa = int.from_bytes(b[2:10], "big")
    value = mantissa * 2.0 ** (exponent - 63)
    return -value if b[0] & 0x80 else value


class PortAudioIn:
    sample_rate = CAPTURE_RATE

    def __init__(self, engine=None) -> None:
        self._engine = engine
        self._direct_qs: list[asyncio.Queue] = []
        self._stream = None

    async def start(self) -> None:
        if self._engine is None and self._stream is None:
            import sounddevice as sd

            loop = asyncio.get_running_loop()

            def _cb(indata, frames, t, status):
                chunk = indata[:, 0].copy()
                for q in list(self._direct_qs):
                    loop.call_soon_threadsafe(_offer, q, chunk)

            self._stream = sd.InputStream(samplerate=CAPTURE_RATE, channels=1, dtype="float32",
                                          blocksize=CHUNK_FRAMES, callback=_cb)
            self._stream.start()
            log.info("voice audio: mic open (direct stream)")

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def subscribe(self) -> asyncio.Queue:
        if self._engine is not None:
            return self._engine.subscribe()
        q: asyncio.Queue = asyncio.Queue(maxsize=DIRECT_QUEUE_CHUNKS)
        self._direct_qs.append(q)
        return q

    def unsubscribe(self, queue) -> None:
        if self._engine is not None:
            self._engine.unsubscribe(queue)
        elif queue in self._direct_qs:
            self._direct_qs.remove(queue)

    @property
    def echo_cancelled(self) -> bool:
        """True when the engine's canceller removes the assistant's voice from capture."""
        aec = getattr(self._engine, "aec", None) if self._engine is not None else None
        return bool(aec is not None and aec.echo_cancellation)

    def capture_muted(self) -> bool:
        # EchoGuard: time-based mute for the playback plus its tail. No engine, no playback.
        if self._engine is None:
            return False
        return self._engine.echo_guard.observe(None, None)[0] == "muted"


class PortAudioOut:
    reports_played = True      # play() takes on_played (see voice/barge_in.py)

    def __init__(self, engine=None) -> None:
        self._engine = engine
        self.available = engine is not None

    async def play(self, pcm_float32_mono, sample_rate: int, *, on_played=None) -> None:
        """Queue audio; returns once queued. on_played runs when all of it has been played."""
        if self._engine is None:
            return
        if sample_rate != self._engine.playback_rate:
            raise ValueError(f"playback runs at {self._engine.playback_rate} Hz, got {sample_rate}")
        await self._engine.play(pcm_float32_mono, block=True, on_played=on_played)

    async def play_file(self, path: str) -> None:
        # The cue plays through afplay, outside the engine's playback queue, so the echo guard
        # is pre-armed for it here, before the device hears it: capture is muted for the cue and
        # its room decay, and the cue never reaches the recognizer or the wake gate.
        if self._engine is not None:
            self._engine.echo_guard.notify_chime(cue_duration_ms(path) + CHIME_DECAY_MS)
        try:
            await asyncio.to_thread(subprocess.run, [CUE_PLAYER, path], check=False,
                                    timeout=CUE_TIMEOUT_S)
        except Exception:
            pass

    def stop_playback(self) -> None:
        if self._engine is not None:
            self._engine.cancel_playback()

    @property
    def is_playing(self) -> bool:
        return self._engine is not None and self._engine.is_playing

    def playback_ref(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._engine.playback_ref(), dtype=np.float32)


def _offer(q: asyncio.Queue, chunk) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(chunk)
