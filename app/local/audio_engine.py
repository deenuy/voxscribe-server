"""Single audio I/O engine for the voxscribe-engine process.

One instance per process. Owns:
  - the capture stream  (microphone at 16 kHz, float32)
  - the playback stream (speaker, float32, sample rate matches TTS)
  - one monotonic clock (monotonic_ms()) shared by capture and playback timestamps
  - a playback reference ring (last 250 ms of played samples, for AEC in R6)
  - an EchoGuard passthrough stub (returns mic frames unchanged until R6 wires real AEC)

Usage:
    engine = AudioEngine(capture_device="MacBook Pro Microphone", playback_device="MacBook Pro Speakers")
    await engine.start()
    sub = engine.subscribe()          # returns asyncio.Queue[np.ndarray]
    await engine.play(samples_f32)
    engine.echo_guard(mic_frame)      # passthrough until R6
    await engine.stop()

One mic stream per process: the voice host (app/local/voice/host.py) reads it through the
AudioIn and AudioOut adapters in app/local/voice/adapters/audio_portaudio.py.

Playback reference ring:
    engine.playback_ref()  → np.ndarray of last PLAYBACK_REF_MS milliseconds of
                             float32 mono at playback_rate. Returns empty array
                             when nothing has been played.
"""

import asyncio
import collections
import logging
import time

log = logging.getLogger("app.local.audio_engine")

_CAPTURE_RATE     = 16_000   # Hz; matches faster-whisper and OWW
_PLAYBACK_RATE    = 24_000   # Hz; Kokoro MLX / Piper output rate
_CHUNK_FRAMES     = 512      # 32 ms at capture rate
_PLAYBACK_REF_MS  = 250      # milliseconds of playback kept for AEC

# Shared epoch for all monotonic timestamps in this process.
_EPOCH = time.monotonic()


def monotonic_ms() -> int:
    """Milliseconds since the engine process started. One clock for capture and playback."""
    return int((time.monotonic() - _EPOCH) * 1000)


def _offer_drop_oldest(q: asyncio.Queue, chunk) -> None:
    # Runs on the loop thread: the full check and the put must not be split across threads.
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(chunk)


class EchoGuard:
    """Time-based mic mute: suppresses mic frames during TTS playback + 500 ms tail.

    Call notify_playback_end(ms) whenever a playback chunk finishes writing.
    observe() returns ("muted", mic_frame) while within the tail window,
    ("pass", mic_frame) otherwise.

    Callers must discard "muted" frames — never push them to ASR.
    Full AEC (correlation-based) is deferred until after daily-use baseline.
    """

    _TAIL_MS = 500

    def __init__(self) -> None:
        self._mute_until_ms: int = -1

    def notify_playback_end(self, end_ms: int) -> None:
        """Record that playback finished at end_ms; extends mute window."""
        self._mute_until_ms = end_ms + self._TAIL_MS

    def observe(self, mic_frame, playback_ref) -> object:
        """Returns ("muted", mic_frame) or ("pass", mic_frame)."""
        if monotonic_ms() <= self._mute_until_ms:
            return ("muted", mic_frame)
        return ("pass", mic_frame)


class _PlaybackRefRing:
    """Circular buffer of played float32 samples for the AEC correlation window."""

    def __init__(self, capacity_samples: int) -> None:
        self._buf = collections.deque(maxlen=capacity_samples)

    def push(self, samples) -> None:
        self._buf.extend(samples)

    def snapshot(self):
        """Return a list of float32 samples (newest at end)."""
        return list(self._buf)


class AudioEngine:
    """One capture stream + one playback stream for the lifetime of the process.

    Not started on construction: call ``await engine.start()`` after asyncio
    is running. ``await engine.stop()`` closes both streams cleanly.
    """

    def __init__(
        self,
        *,
        capture_device: str | int | None = None,
        playback_device: str | int | None = None,
        capture_rate: int = _CAPTURE_RATE,
        playback_rate: int = _PLAYBACK_RATE,
        chunk_frames: int = _CHUNK_FRAMES,
    ) -> None:
        self.capture_device  = capture_device
        self.playback_device = playback_device
        self.capture_rate    = capture_rate
        self.playback_rate   = playback_rate
        self.chunk_frames    = chunk_frames

        self._echo_guard     = EchoGuard()
        self._ref_capacity   = int(_PLAYBACK_REF_MS / 1000 * playback_rate)
        self._playback_ref   = _PlaybackRefRing(self._ref_capacity)

        self._capture_qs: list[asyncio.Queue] = []
        self._playback_q: asyncio.Queue       = asyncio.Queue(maxsize=256)
        self._capture_stream                   = None
        self._playback_stream                  = None
        self._playback_task: asyncio.Task | None = None
        self._started = False
        self._capture_cb = None                  # set by start(), reused by reopen_stream()
        self._reopen_lock = asyncio.Lock()
        self.reopen_count = 0

    # ------------------------------------------------------------------
    # public: clock

    @staticmethod
    def monotonic_ms() -> int:
        return monotonic_ms()

    # ------------------------------------------------------------------
    # public: echo guard

    @property
    def echo_guard(self) -> EchoGuard:
        return self._echo_guard

    # ------------------------------------------------------------------
    # public: playback reference

    def playback_ref(self):
        """Snapshot of the last PLAYBACK_REF_MS ms of played audio (float32 list)."""
        return self._playback_ref.snapshot()

    # ------------------------------------------------------------------
    # public: subscriptions

    def subscribe(self) -> asyncio.Queue:
        """Return a queue that receives float32 capture chunks (512 frames each).

        The caller keeps a reference and passes it to ``unsubscribe`` when done.
        Slow subscribers drop their oldest frame instead of blocking the capture loop.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._capture_qs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._capture_qs.remove(q)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # public: playback

    @property
    def is_playing(self) -> bool:
        """True while the playback queue has samples or the task is draining."""
        return not self._playback_q.empty()

    def cancel_playback(self) -> None:
        """Discard all queued playback samples (barge-in gate, P1)."""
        discarded = 0
        while not self._playback_q.empty():
            try:
                self._playback_q.get_nowait()
                discarded += 1
            except asyncio.QueueEmpty:
                break
        if discarded:
            log.info("audio_engine: playback cancelled discarded=%d chunks", discarded)

    async def play(self, samples, *, block: bool = False) -> None:
        """Enqueue float32 samples for playback.

        samples: any sequence of float32 values at self.playback_rate.
        block=True: wait until the queue has room (streaming TTS path).
        block=False: drop oldest if full (non-critical sounds).
        """
        try:
            import numpy as np
        except ImportError:
            return

        arr = np.asarray(samples, dtype="float32")
        if block:
            await self._playback_q.put(arr)
        else:
            if self._playback_q.full():
                try:
                    self._playback_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._playback_q.put_nowait(arr)

    # ------------------------------------------------------------------
    # lifecycle

    async def start(self) -> None:
        if self._started:
            return
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("AudioEngine needs sounddevice and numpy") from exc

        loop = asyncio.get_running_loop()

        def _capture_cb(indata, frames, t, status):
            if status:
                log.debug("capture status: %s", status)
            chunk = indata[:, 0].copy()
            for q in list(self._capture_qs):
                try:
                    loop.call_soon_threadsafe(_offer_drop_oldest, q, chunk)
                except Exception:
                    pass

        self._capture_cb = _capture_cb
        self._open_streams(sd)

        self._playback_task = asyncio.create_task(self._playback_loop(np), name="audio-engine-playback")
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        self._close_streams(abort=False)
        log.info("audio_engine: stopped")

    async def reopen_stream(self, reason: str = "manual") -> bool:
        """Close both streams, refresh the device list, and open them again.

        Called by the capture watchdog after a stall, digital silence, an input device change or a
        system wake. Subscribers keep their queues, so consumers see a gap and nothing else.
        PortAudio is re-initialised in between, because it reads the device list (and so the
        default input) only once: without that, a reopen after a device change would reopen the
        old device. Queued playback is dropped. Runs on the loop thread, like start() and the
        playback writes, so no write can race the close. Returns True when both streams are open.
        """
        if not self._started:
            return False
        async with self._reopen_lock:
            import sounddevice as sd

            self.cancel_playback()
            self._close_streams(abort=True)
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                log.warning("audio_engine: device list refresh failed; reopening with the old list")
            try:
                self._open_streams(sd)
            except Exception:
                log.exception("audio_engine: reopen failed reason=%s", reason)
                return False
            self.reopen_count += 1
            log.info("audio_engine: streams reopened reason=%s count=%d", reason, self.reopen_count)
            return True

    # ------------------------------------------------------------------
    # internal

    def _open_streams(self, sd) -> None:
        capture_kw = {}
        if self.capture_device is not None:
            capture_kw["device"] = _resolve_device(self.capture_device, kind="input")

        self._capture_stream = sd.InputStream(
            samplerate=self.capture_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_frames,
            callback=self._capture_cb,
            **capture_kw,
        )
        self._capture_stream.start()
        log.info("audio_engine: capture open device=%r rate=%d chunk=%d",
                 self.capture_device or "default", self.capture_rate, self.chunk_frames)

        playback_kw = {}
        if self.playback_device is not None:
            playback_kw["device"] = _resolve_device(self.playback_device, kind="output")

        self._playback_stream = sd.OutputStream(
            samplerate=self.playback_rate,
            channels=1,
            dtype="float32",
            **playback_kw,
        )
        self._playback_stream.start()
        log.info("audio_engine: playback open device=%r rate=%d",
                 self.playback_device or "default", self.playback_rate)

    def _close_streams(self, *, abort: bool) -> None:
        # abort() does not wait for pending buffers: a stream whose device went away can hang in stop().
        for name in ("_capture_stream", "_playback_stream"):
            stream = getattr(self, name)
            setattr(self, name, None)
            if stream is None:
                continue
            try:
                stream.abort() if abort else stream.stop()
                stream.close()
            except Exception:
                log.warning("audio_engine: error closing %s", name.strip("_"))

    async def _playback_loop(self, np) -> None:
        try:
            while True:
                chunk = await self._playback_q.get()
                if self._playback_stream is None:
                    if self._started:
                        continue  # mid-reopen, or a reopen failed: drop the chunk, keep the loop
                    break
                self._playback_ref.push(chunk.tolist())
                try:
                    self._playback_stream.write(chunk)
                    self._echo_guard.notify_playback_end(monotonic_ms())
                except Exception:
                    log.exception("audio_engine: playback write error")
        except asyncio.CancelledError:
            pass


# ------------------------------------------------------------------
# helper: device selection by name

def _resolve_device(device: str | int, kind: str) -> str | int:
    """If device is an int, return it as-is. If it's a string, find the first
    device whose name contains it (case-insensitive). Falls back to the string
    itself if sounddevice is not available or no match is found."""
    if isinstance(device, int):
        return device
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        key = device.lower()
        for i, d in enumerate(devs):
            if key in d["name"].lower():
                if kind == "input" and d["max_input_channels"] > 0:
                    log.info("audio_engine: resolved capture device %r → index %d (%s)", device, i, d["name"])
                    return i
                if kind == "output" and d["max_output_channels"] > 0:
                    log.info("audio_engine: resolved playback device %r → index %d (%s)", device, i, d["name"])
                    return i
    except Exception:
        pass
    return device
