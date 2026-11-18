"""Single audio I/O engine for the voxscribe-engine process.

One instance per process. Owns:
  - the capture stream  (microphone at 16 kHz, float32)
  - the playback stream (speaker, float32, sample rate matches TTS)
  - one monotonic clock (monotonic_ms()) shared by capture and playback timestamps
  - a playback reference ring (last 250 ms of played samples, read by the barge-in gate)
  - capture clean-up (T12): every capture chunk passes through AcousticEchoCanceller, which
    subtracts what the speaker is playing (WebRTC AEC3) and, when enabled, then suppresses
    stationary noise, before any subscriber sees it
  - an EchoGuard time-based mute, kept behind the canceller as a fallback

Capture path:
    mic chunk -> echo cancellation -> noise suppression (opt-in) -> subscribers
    (the voice host then applies the EchoGuard mute before ASR: AudioIn.capture_muted)

Usage:
    engine = AudioEngine(capture_device="MacBook Pro Microphone", playback_device="MacBook Pro Speakers")
    await engine.start()
    sub = engine.subscribe()          # returns asyncio.Queue[np.ndarray]
    await engine.play(samples_f32)
    engine.echo_guard.observe(mic_frame, None)
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
import threading
import time

log = logging.getLogger("app.local.audio_engine")

_CAPTURE_RATE     = 16_000   # Hz; matches faster-whisper and OWW
_PLAYBACK_RATE    = 24_000   # Hz; Kokoro MLX / Piper output rate
_CHUNK_FRAMES     = 512      # 32 ms at capture rate
_PLAYBACK_REF_MS  = 250      # milliseconds of playback kept for the barge-in gate

# Capture clean-up (T12). WebRTC's audio processing runs on 10 ms frames.
_APM_FRAME_MS        = 10
# Far-end audio held for the canceller before the oldest is dropped. The playback loop pushes one
# sentence at a time, so this only fills if capture stops while playback continues.
_REFERENCE_BACKLOG_S = 60
# WebRTC noise suppression level: 0..3 is 6, 12, 18, 21 dB. 1 is the library default.
# Suppression is opt-in (Config.noise_suppress_enabled): it raised recognizer word error on the
# 10 dB SNR fan fixture, which T12's acceptance does not allow.
_NS_LEVEL            = 1

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
    Since T12 the capture path is echo-cancelled before this guard sees it
    (AcousticEchoCanceller); the mute stays as a second line for a burst the canceller misses.
    """

    _TAIL_MS = 500

    def __init__(self) -> None:
        self._mute_until_ms: int = -1

    def notify_playback_end(self, end_ms: int) -> None:
        """Record that playback finished at end_ms; extends mute window."""
        self._mute_until_ms = end_ms + self._TAIL_MS

    def notify_chime(self, duration_ms: int) -> None:
        """Pre-arm the mute window for a cue played outside the engine playback queue.

        Call when the cue fires (before the output device receives it), so the mic is muted
        while the cue plays through the room and decays. duration_ms is the whole mute (cue
        length plus room decay); TAIL_MS is not added. Never shortens a longer window.
        """
        mute_until = monotonic_ms() + duration_ms
        if mute_until > self._mute_until_ms:
            self._mute_until_ms = mute_until

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


class _SampleFifo:
    """First-in first-out float32 samples, read in exact lengths. Not thread-safe on its own."""

    def __init__(self, np, max_samples: int | None = None) -> None:
        self._np = np
        self._chunks: collections.deque = collections.deque()
        self._head = 0          # samples already read from the first chunk
        self._size = 0
        self._max = max_samples
        self.dropped = 0

    def __len__(self) -> int:
        return self._size

    def push(self, samples) -> None:
        arr = self._np.asarray(samples, dtype=self._np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._chunks.append(arr)
        self._size += arr.size
        if self._max is not None and self._size > self._max:
            self.dropped += self._size - self._max
            self._discard(self._size - self._max)

    def pop(self, n: int):
        """Return exactly n samples; zeros fill whatever the queue does not hold."""
        out = self._np.zeros(n, dtype=self._np.float32)
        filled = 0
        while filled < n and self._chunks:
            first = self._chunks[0]
            take = min(n - filled, first.size - self._head)
            out[filled: filled + take] = first[self._head: self._head + take]
            filled += take
            self._advance(take)
        return out

    def clear(self) -> None:
        self._chunks.clear()
        self._head = 0
        self._size = 0

    def _discard(self, n: int) -> None:
        while n > 0 and self._chunks:
            take = min(n, self._chunks[0].size - self._head)
            self._advance(take)
            n -= take

    def _advance(self, n: int) -> None:
        self._head += n
        self._size -= n
        if self._head >= self._chunks[0].size:
            self._chunks.popleft()
            self._head = 0


class _LinearResampler:
    """Streaming linear-interpolation resampler, continuous across chunk boundaries.

    Converts the playback reference to the capture rate. Linear interpolation folds a little of
    the band above the capture Nyquist back in; measured on TTS speech it cancels as well as a
    polyphase filter, so the simpler form is kept.
    """

    def __init__(self, np, src_rate: int, dst_rate: int) -> None:
        self._np = np
        self._step = src_rate / dst_rate
        self._t = 0.0       # next output position, in source samples, from the next chunk start
        self._last = 0.0    # last source sample of the previous chunk (position -1)

    def __call__(self, x):
        np = self._np
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self._step == 1.0 or x.size == 0:
            return x
        n = x.size
        if self._t > n - 1:
            self._t -= n
            self._last = float(x[-1])
            return np.zeros(0, dtype=np.float32)
        count = int((n - 1 - self._t) // self._step) + 1
        pos = self._t + np.arange(count) * self._step
        ext = np.concatenate([np.array([self._last], dtype=np.float32), x])
        out = np.interp(pos + 1.0, np.arange(n + 1), ext).astype(np.float32)
        self._t = float(pos[-1] + self._step - n)
        self._last = float(x[-1])
        return out

    def reset(self) -> None:
        self._t = 0.0
        self._last = 0.0


_apm_unavailable_logged = False


def _load_audio_processor():
    """The WebRTC AudioProcessor class, or None when pywebrtc-audio is not installed (logged once)."""
    global _apm_unavailable_logged
    try:
        from pywebrtc_audio import AudioProcessor
        return AudioProcessor
    except ImportError:
        if not _apm_unavailable_logged:
            _apm_unavailable_logged = True
            log.warning("audio_engine: pywebrtc-audio not installed; echo cancellation and noise "
                        "suppression are off and only the time-based echo guard runs "
                        "(install with: uv sync --extra aec)")
        return None


class AcousticEchoCanceller:
    """Subtract the playback reference from the capture signal, then suppress stationary noise.

    WebRTC's audio processing module (AEC3, then noise suppression, the order WebRTC runs them:
    suppressing noise first would make the echo path nonlinear and stop the adaptive filter from
    converging). The reference is what the playback loop is about to write; the canceller holds
    it in a queue and takes one 10 ms frame of it for every 10 ms of capture, so the two stay
    aligned by sample count while the speaker plays in real time. AEC3 estimates the remaining
    delay (output buffer, room, input buffer) itself.

    Thread use: push_reference() runs on the event loop thread, process_capture() on the
    PortAudio capture thread; one lock serialises them.

    Output is delayed by one 10 ms frame (plus the processor's own filter delay) so that every
    call returns exactly as many samples as it was given.

    Falls back to passthrough, leaving EchoGuard's time-based mute as the only guard, when
    pywebrtc-audio is not installed or both stages are disabled.
    """

    def __init__(
        self,
        *,
        capture_rate: int = _CAPTURE_RATE,
        playback_rate: int = _PLAYBACK_RATE,
        echo_cancellation: bool = True,
        noise_suppression: bool = False,
        processor_cls=None,
    ) -> None:
        import numpy as np

        self._np = np
        self.capture_rate = capture_rate
        self.frame_samples = capture_rate * _APM_FRAME_MS // 1000
        self._lock = threading.Lock()
        self._resample = _LinearResampler(np, playback_rate, capture_rate)
        self._far = _SampleFifo(np, max_samples=_REFERENCE_BACKLOG_S * capture_rate)
        self._near = _SampleFifo(np)
        self._out = _SampleFifo(np)

        self._apm = None
        if echo_cancellation or noise_suppression:
            cls = processor_cls or _load_audio_processor()
            if cls is not None:
                self._apm = cls(
                    sample_rate=capture_rate,
                    num_channels=1,
                    echo_cancellation=echo_cancellation,
                    noise_suppression=noise_suppression,
                    ns_level=_NS_LEVEL,
                )
        self.echo_cancellation = echo_cancellation and self._apm is not None
        self.noise_suppression = noise_suppression and self._apm is not None
        self._error_logged = False
        self._prime()
        log.info("audio_engine: capture clean-up echo_cancellation=%s noise_suppression=%s",
                 self.echo_cancellation, self.noise_suppression)

    @property
    def active(self) -> bool:
        """True when frames are processed; False means process_capture() is a passthrough."""
        return self._apm is not None

    def push_reference(self, pcm_float32) -> None:
        """Queue audio the speaker is about to play (the far-end reference), at playback rate."""
        if not self.echo_cancellation:
            return
        ref = self._resample(pcm_float32)
        with self._lock:
            self._far.push(ref)

    def process_capture(self, pcm_float32):
        """Return the capture chunk with the echo removed and noise suppressed, same length."""
        if self._apm is None:
            return pcm_float32
        np = self._np
        frame = np.asarray(pcm_float32, dtype=np.float32).reshape(-1)
        n = self.frame_samples
        try:
            with self._lock:
                self._near.push(frame)
                while len(self._near) >= n:
                    near = self._near.pop(n)
                    if self.echo_cancellation:
                        self._out.push(self._apm.process(near, self._far.pop(n)))
                    else:
                        self._out.push(self._apm.process(near))
                return self._out.pop(frame.size)
        except Exception:
            # Never lose capture audio to a processing fault: the echo guard still runs downstream.
            if not self._error_logged:
                self._error_logged = True
                log.exception("audio_engine: capture clean-up failed; passing raw audio")
            return frame

    def reset(self) -> None:
        """Forget the echo path and all queued audio (device change or stream reopen)."""
        with self._lock:
            self._far.clear()
            self._near.clear()
            self._out.clear()
            self._resample.reset()
            if self._apm is not None:
                self._apm.reset()
            self._prime()

    def _prime(self) -> None:
        # One frame of lead so a call can always return as many samples as it was given.
        self._out.push(self._np.zeros(self.frame_samples, dtype=self._np.float32))


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
        echo_cancellation: bool = True,
        noise_suppression: bool = False,
        echo_guard: bool = True,
    ) -> None:
        self.capture_device  = capture_device
        self.playback_device = playback_device
        self.capture_rate    = capture_rate
        self.playback_rate   = playback_rate
        self.chunk_frames    = chunk_frames

        self._echo_guard     = EchoGuard()
        # False: playback never arms the time-based mute (VOXSCRIBE_VOICE_ECHO_GUARD=false).
        self._echo_guard_on  = echo_guard
        self._aec_options    = {"echo_cancellation": echo_cancellation,
                                "noise_suppression": noise_suppression}
        self._aec: AcousticEchoCanceller | None = None   # built by start()
        self._ref_capacity   = int(_PLAYBACK_REF_MS / 1000 * playback_rate)
        self._playback_ref   = _PlaybackRefRing(self._ref_capacity)

        self._capture_qs: list[asyncio.Queue] = []
        self._raw_capture_qs: list[asyncio.Queue] = []   # before clean-up (device health checks)
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

    @property
    def aec(self) -> "AcousticEchoCanceller | None":
        """The capture clean-up stage; None until start()."""
        return self._aec

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

    def subscribe_raw(self) -> asyncio.Queue:
        """Like subscribe(), but chunks arrive as the device delivered them, before clean-up.

        For device health checks (the capture watchdog): noise suppression takes a quiet room
        close to digital silence and the canceller's high-pass removes mains hum, so neither can
        be judged on the cleaned signal. Never feed these chunks to the wake gate or ASR.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._raw_capture_qs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        for qs in (self._capture_qs, self._raw_capture_qs):
            try:
                qs.remove(q)
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
        if self._aec is None:
            self._aec = AcousticEchoCanceller(capture_rate=self.capture_rate,
                                              playback_rate=self.playback_rate,
                                              **self._aec_options)
        aec = self._aec

        def _capture_cb(indata, frames, t, status):
            if status:
                log.debug("capture status: %s", status)
            raw = indata[:, 0].copy()
            for q in list(self._raw_capture_qs):
                try:
                    loop.call_soon_threadsafe(_offer_drop_oldest, q, raw)
                except Exception:
                    pass
            # Echo cancellation and noise suppression run here, on the capture thread, so every
            # subscriber (wake gate, session ASR, debug routes) gets the cleaned signal.
            chunk = aec.process_capture(raw)
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
            if self._aec is not None:
                self._aec.reset()  # aborted output and possibly a new device: the echo path is new
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
                # Before the write, not after: write() blocks until the chunk is nearly played,
                # and the canceller must hold the reference before the echo of it is captured.
                if self._aec is not None:
                    self._aec.push_reference(chunk)
                try:
                    self._playback_stream.write(chunk)
                    if self._echo_guard_on:
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
