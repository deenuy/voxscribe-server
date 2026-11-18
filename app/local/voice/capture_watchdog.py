"""Capture watchdog: keeps the microphone stream alive and reports its health.

The audio engine opens one input stream at startup. After a hibernate, a lid-close sleep or a
change of input device, CoreAudio can stop calling the stream callback, or keep calling it with
all-zero buffers, and no error is raised. The watchdog subscribes to the engine like any other
consumer and watches what arrives:

    stall            no chunk for STALL_TIMEOUT_S                          reopen
    digital silence  nothing at or above DIGITAL_SILENCE_RMS for
                     SILENCE_CHECK_WINDOW_S while chunks keep arriving     reopen
    device change    the system default input device changed               reopen
    clip, dropout, hum                                                     health event, log only

After every reopen it emits a CaptureHealth event, so the wake loop can log the recovery and reset
its wake model. Power events forwarded by the app (POST /v1/wake/power) pause detection while the
machine sleeps and force a reopen shortly after it wakes. Repeated reopens that bring no signal
back are spaced out with a doubling backoff, so a muted or missing microphone is retried slowly
instead of in a tight loop.

Log lines carry the issue and a level, never audio.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("app.local.voice.capture_watchdog")

# Recovery triggers
STALL_TIMEOUT_S = 5.0          # no chunk at all for this long: the callback stopped firing
SILENCE_CHECK_WINDOW_S = 2.0   # every chunk this long below DIGITAL_SILENCE_RMS: digital silence
DIGITAL_SILENCE_RMS = 0.0001   # a real room never goes this low; all-zero buffers do
DEVICE_POLL_S = 2.0            # how often the default input device is read

# Health checks (log only)
CLIP_THRESHOLD = 0.98          # peak sample magnitude
DROPOUT_WINDOW_S = 0.1         # zero-sample share is compared between consecutive windows
DROPOUT_JUMP = 0.5             # rise in that share that counts as a dropout
HUM_WINDOW_S = 1.0             # 1 Hz bins at any sample rate
HUM_CHECK_INTERVAL_S = 10.0    # one spectrum every this many seconds
HUM_FUNDAMENTALS_HZ = (50.0, 60.0)
HUM_HARMONICS = 3              # fundamental plus the next two harmonics
HUM_BIN_HALF_WIDTH_HZ = 2.0
HUM_MIN_SHARE = 0.5            # share of spectral energy in the hum bins
HUM_MIN_RMS = 0.001            # ignore hum in a near-silent signal
VOICE_BAND_HZ = (300.0, 3400.0)
VOICE_BAND_MAX_SHARE = 0.05    # hum only counts when the voice band is close to empty
HEALTH_EVENT_INTERVAL_S = 30.0  # at most one event per health issue per interval

# Recovery pacing
POLL_INTERVAL_S = 0.5
WAKE_REOPEN_DELAY_S = 1.0      # let macOS reinitialise audio after a wake before reopening
REOPEN_BACKOFF_BASE_S = 2.0    # second reopen without signal waits this long, then doubles
REOPEN_BACKOFF_MAX_S = 60.0

POWER_EVENTS = ("sleep", "wake", "lock", "unlock")
RECOVERY_ISSUES = ("stall", "silence", "device_change", "power_wake")
HEALTH_ISSUES = ("clip", "dropout", "hum")


@dataclass(frozen=True)
class CaptureHealth:
    """One capture event. action is "reopened" or "reopen_failed" for a recovery, "logged" otherwise."""

    issue: str
    action: str
    value: float | None = None


class CaptureWatchdog:
    """Watches one AudioEngine's capture subscription and reopens the stream when it goes dead.

    engine needs subscribe(), unsubscribe(q), async reopen_stream(reason) and capture_rate.
    on_event receives every CaptureHealth. default_input returns an identifier for the system
    default input device, or None when it cannot tell; it is only used when the engine follows
    the default device (capture_device is None).
    """

    def __init__(
        self,
        engine,
        *,
        on_event: Callable[[CaptureHealth], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        default_input: Callable[[], object | None] | None = None,
    ) -> None:
        self._engine = engine
        self._on_event = on_event
        self._clock = clock
        self._rate = int(getattr(engine, "capture_rate", 16_000))
        if default_input is None and getattr(engine, "capture_device", None) is None:
            default_input = coreaudio_default_input
        self._default_input = default_input
        self._device: object | None = None
        self._next_device_t = 0.0

        self._sleeping = False
        self._consecutive_reopens = 0
        self._next_reopen_t = 0.0
        self._last_health_t: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._wake_task: asyncio.Task | None = None

        self._dropout_n = int(DROPOUT_WINDOW_S * self._rate)
        self._hum_n = int(HUM_WINDOW_S * self._rate)
        self._reset_signal_state()

    # ------------------------------------------------------------------
    # frame-level checks (pure; driven by run() or directly by tests)

    def observe(self, chunk) -> None:
        """Account for one capture chunk (float32 mono)."""
        now = self._clock()
        x = np.asarray(chunk, dtype=np.float32)
        self._last_chunk_t = now
        if x.size == 0:
            return
        rms = float(np.sqrt(np.mean(x * x)))
        if rms >= DIGITAL_SILENCE_RMS:
            self._last_signal_t = now
            self._consecutive_reopens = 0
            if self._sleeping:
                # A live signal means the machine is awake, whatever the last power event said.
                self._sleeping = False

        peak = float(np.max(np.abs(x)))
        if peak > CLIP_THRESHOLD:
            self._health("clip", peak)

        self._observe_dropout(x)
        self._observe_hum(x, now)

    def check(self) -> str | None:
        """The recovery reason that applies now ("stall" or "silence"), or None."""
        if self._sleeping:
            return None
        now = self._clock()
        if now < self._next_reopen_t:
            return None
        if now - self._last_chunk_t >= STALL_TIMEOUT_S:
            return "stall"
        # Measured on arrival times, so a stream that stops delivering is a stall, not silence.
        if self._last_chunk_t - self._last_signal_t >= SILENCE_CHECK_WINDOW_S:
            return "silence"
        return None

    def check_device(self) -> str | None:
        """'device_change' when the default input changed since the last poll, else None."""
        if self._default_input is None or self._sleeping:
            return None
        now = self._clock()
        if now < self._next_device_t:
            return None
        self._next_device_t = now + DEVICE_POLL_S
        current = self._default_input()
        if current is None:
            return None  # unreadable this time: keep the last known device
        previous, self._device = self._device, current
        if previous is not None and current != previous:
            return "device_change"
        return None

    # ------------------------------------------------------------------
    # recovery

    async def recover(self, issue: str) -> bool:
        """Reopen the engine's streams and report the result as a CaptureHealth event."""
        log.warning("capture_health issue=%s action=reopen", issue)
        try:
            ok = bool(await self._engine.reopen_stream(issue))
        except Exception:
            log.exception("capture watchdog: reopen_stream raised issue=%s", issue)
            ok = False
        now = self._clock()
        self._consecutive_reopens += 1
        backoff = 0.0
        if self._consecutive_reopens > 1:
            backoff = min(REOPEN_BACKOFF_BASE_S * 2 ** (self._consecutive_reopens - 2),
                          REOPEN_BACKOFF_MAX_S)
        self._next_reopen_t = now + backoff
        self._reset_signal_state()
        self._emit(CaptureHealth(issue, "reopened" if ok else "reopen_failed"))
        return ok

    def on_power(self, event: str) -> None:
        """Handle a power event forwarded by the app: sleep, wake, lock or unlock."""
        if event not in POWER_EVENTS:
            raise ValueError(f"unknown power event {event!r}")
        log.info("capture watchdog: power event=%s", event)
        if event == "sleep":
            self._sleeping = True
        elif event == "wake" or (event == "unlock" and self._sleeping):
            # An unlock while still marked asleep means the resume was missed: treat it as the wake.
            self._sleeping = False
            self._consecutive_reopens = 0
            self._next_reopen_t = 0.0
            self._reset_signal_state()  # time spent asleep is not a stall; the reopen below handles it
            if self._wake_task is None or self._wake_task.done():
                self._wake_task = asyncio.get_running_loop().create_task(
                    self._reopen_after_wake(), name="capture-watchdog-wake")
        # lock alone changes nothing about capture

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    # ------------------------------------------------------------------
    # lifecycle

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self.run(), name="capture-watchdog")

    async def stop(self) -> None:
        for task in (self._wake_task, self._task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = self._wake_task = None

    async def run(self) -> None:
        q = self._engine.subscribe()
        self._reset_signal_state()
        log.info("capture watchdog: running stall=%.1fs silence=%.1fs device_poll=%s",
                 STALL_TIMEOUT_S, SILENCE_CHECK_WINDOW_S,
                 "on" if self._default_input is not None else "off")
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=POLL_INTERVAL_S)
                    self.observe(chunk)
                    while not q.empty():
                        self.observe(q.get_nowait())
                except asyncio.TimeoutError:
                    pass
                issue = self.check() or self.check_device()
                if issue is not None:
                    await self.recover(issue)
        finally:
            self._engine.unsubscribe(q)

    # ------------------------------------------------------------------
    # internal

    async def _reopen_after_wake(self) -> None:
        await asyncio.sleep(WAKE_REOPEN_DELAY_S)
        await self.recover("power_wake")

    def _reset_signal_state(self) -> None:
        now = self._clock()
        self._last_chunk_t = now
        self._last_signal_t = now
        self._dropout_zeros = 0
        self._dropout_total = 0
        self._dropout_prev: float | None = None
        self._hum_buf: list[np.ndarray] = []
        self._hum_len = 0
        self._next_hum_t = now

    def _observe_dropout(self, x: np.ndarray) -> None:
        self._dropout_zeros += int(np.count_nonzero(x == 0.0))
        self._dropout_total += int(x.size)
        if self._dropout_total < self._dropout_n:
            return
        share = self._dropout_zeros / self._dropout_total
        prev = self._dropout_prev
        self._dropout_prev = share
        self._dropout_zeros = self._dropout_total = 0
        if prev is not None and share - prev >= DROPOUT_JUMP:
            self._health("dropout", share)

    def _observe_hum(self, x: np.ndarray, now: float) -> None:
        if now < self._next_hum_t:
            return
        self._hum_buf.append(x)
        self._hum_len += int(x.size)
        if self._hum_len < self._hum_n:
            return
        window = np.concatenate(self._hum_buf)[: self._hum_n]
        self._hum_buf, self._hum_len = [], 0
        self._next_hum_t = now + HUM_CHECK_INTERVAL_S
        share = hum_share(window, self._rate)
        if share is not None:
            self._health("hum", share)

    def _health(self, issue: str, value: float) -> None:
        now = self._clock()
        last = self._last_health_t.get(issue)
        if last is not None and now - last < HEALTH_EVENT_INTERVAL_S:
            return
        self._last_health_t[issue] = now
        log.info("capture_health issue=%s value=%.4f", issue, value)
        self._emit(CaptureHealth(issue, "logged", value))

    def _emit(self, event: CaptureHealth) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            log.exception("capture watchdog: event handler failed issue=%s", event.issue)


def hum_share(window: np.ndarray, rate: int) -> float | None:
    """Share of energy in the mains hum bins when the window is hum and nothing else, else None."""
    x = np.asarray(window, dtype=np.float64)
    if x.size == 0 or float(np.sqrt(np.mean(x * x))) < HUM_MIN_RMS:
        return None
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / rate)
    spectrum[0] = 0.0  # DC offset is not hum
    total = float(spectrum.sum())
    if total <= 0.0:
        return None
    hum_mask = np.zeros_like(freqs, dtype=bool)
    for f0 in HUM_FUNDAMENTALS_HZ:
        for k in range(1, HUM_HARMONICS + 1):
            hum_mask |= np.abs(freqs - f0 * k) <= HUM_BIN_HALF_WIDTH_HZ
    voice_mask = (freqs >= VOICE_BAND_HZ[0]) & (freqs <= VOICE_BAND_HZ[1])
    hum = float(spectrum[hum_mask].sum()) / total
    voice = float(spectrum[voice_mask].sum()) / total
    if hum >= HUM_MIN_SHARE and voice <= VOICE_BAND_MAX_SHARE:
        return hum
    return None


# ----------------------------------------------------------------------
# macOS default input device, read straight from CoreAudio (no pyobjc needed)

class _PropertyAddress(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_SYSTEM_OBJECT = 1                         # kAudioObjectSystemObject
_DEFAULT_INPUT = _fourcc("dIn ")           # kAudioHardwarePropertyDefaultInputDevice
_SCOPE_GLOBAL = _fourcc("glob")            # kAudioObjectPropertyScopeGlobal
_ELEMENT_MAIN = 0                          # kAudioObjectPropertyElementMain
_coreaudio = None


def coreaudio_default_input() -> int | None:
    """CoreAudio's current default input device id, or None off macOS or on any failure.

    PortAudio reads the device list once at initialisation, so it cannot see a change; CoreAudio
    always answers with the live value.
    """
    global _coreaudio
    if sys.platform != "darwin":
        return None
    try:
        if _coreaudio is None:
            lib = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
            fn = lib.AudioObjectGetPropertyData
            fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(_PropertyAddress), ctypes.c_uint32,
                           ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
            fn.restype = ctypes.c_int32
            _coreaudio = fn
        addr = _PropertyAddress(_DEFAULT_INPUT, _SCOPE_GLOBAL, _ELEMENT_MAIN)
        device = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(device))
        status = _coreaudio(_SYSTEM_OBJECT, ctypes.byref(addr), 0, None,
                            ctypes.byref(size), ctypes.byref(device))
        return int(device.value) if status == 0 else None
    except Exception:
        return None
