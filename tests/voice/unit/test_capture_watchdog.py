"""Capture watchdog and quiet hours: fakes only, no audio device, no database, no real time."""

import asyncio
from datetime import datetime

import numpy as np
import pytest

from app.local.config import Config, QuietHours
from app.local.voice import capture_watchdog as cw
from app.local.voice.capture_watchdog import CaptureHealth, CaptureWatchdog, hum_share

RATE = 16_000
CHUNK = 512
CHUNK_S = CHUNK / RATE


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeEngine:
    capture_rate = RATE
    capture_device = None

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.reopens: list[str] = []

    def subscribe(self):
        return asyncio.Queue()

    def unsubscribe(self, q) -> None:
        pass

    async def reopen_stream(self, reason: str) -> bool:
        self.reopens.append(reason)
        return self.ok


def noise(level: float = 0.01, n: int = CHUNK) -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, level, n).astype(np.float32)


def zeros(n: int = CHUNK) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def tone(freq: float, amp: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make(clock=None, engine=None, default_input=lambda: None):
    clock = clock or FakeClock()
    events: list[CaptureHealth] = []
    wd = CaptureWatchdog(engine or FakeEngine(), on_event=events.append, clock=clock,
                         default_input=default_input)
    return wd, clock, events


def feed(wd, clock, chunk_fn, seconds: float) -> None:
    for _ in range(int(seconds / CHUNK_S)):
        clock.advance(CHUNK_S)
        wd.observe(chunk_fn())


# ---------------------------------------------------------------- recovery triggers

def test_live_noise_never_triggers():
    wd, clock, _ = make()
    feed(wd, clock, noise, 10.0)
    assert wd.check() is None


def test_stall_after_timeout():
    wd, clock, _ = make()
    feed(wd, clock, noise, 1.0)
    clock.advance(cw.STALL_TIMEOUT_S - 0.1)
    assert wd.check() is None
    clock.advance(0.1)
    assert wd.check() == "stall"


def test_stopped_stream_is_a_stall_not_silence():
    wd, clock, _ = make()
    feed(wd, clock, noise, 1.0)
    clock.advance(cw.SILENCE_CHECK_WINDOW_S + 1.0)
    assert wd.check() is None


def test_digital_silence_after_window():
    wd, clock, _ = make()
    feed(wd, clock, noise, 1.0)
    feed(wd, clock, zeros, cw.SILENCE_CHECK_WINDOW_S - 0.1)
    assert wd.check() is None
    feed(wd, clock, zeros, 0.2)
    assert wd.check() == "silence"


def test_quiet_room_is_not_digital_silence():
    wd, clock, _ = make()
    feed(wd, clock, lambda: noise(cw.DIGITAL_SILENCE_RMS * 3), 5.0)
    assert wd.check() is None


def test_sleep_suppresses_detection_and_live_signal_clears_it():
    wd, clock, _ = make()
    feed(wd, clock, noise, 1.0)
    wd.on_power("sleep")
    clock.advance(cw.STALL_TIMEOUT_S * 4)
    assert wd.sleeping and wd.check() is None
    wd.observe(noise())
    assert not wd.sleeping


def test_lock_changes_nothing():
    wd, clock, _ = make()
    wd.on_power("lock")
    assert not wd.sleeping


def test_unknown_power_event_rejected():
    wd, _, _ = make()
    with pytest.raises(ValueError):
        wd.on_power("hibernate")


def test_device_change_detected_on_poll():
    devices = iter([7, 7, None, 9, 9])
    wd, clock, _ = make(default_input=lambda: next(devices))
    seen = []
    for _ in range(5):
        seen.append(wd.check_device())
        clock.advance(cw.DEVICE_POLL_S)
    assert seen == [None, None, None, "device_change", None]


def test_device_poll_is_paced():
    calls = []
    wd, clock, _ = make(default_input=lambda: calls.append(1) or 1)
    wd.check_device()
    wd.check_device()
    assert len(calls) == 1


def test_device_poll_off_when_engine_has_named_device():
    engine = FakeEngine()
    engine.capture_device = "MacBook Pro Microphone"
    wd = CaptureWatchdog(engine, clock=FakeClock())
    assert wd.check_device() is None


# ---------------------------------------------------------------- recovery and backoff

async def test_recover_reopens_and_emits():
    engine = FakeEngine()
    wd, clock, events = make(engine=engine)
    assert await wd.recover("stall") is True
    assert engine.reopens == ["stall"]
    assert events == [CaptureHealth("stall", "reopened")]


async def test_failed_reopen_reported_and_backed_off():
    engine = FakeEngine(ok=False)
    wd, clock, events = make(engine=engine)
    await wd.recover("stall")
    first_next = wd._next_reopen_t
    await wd.recover("stall")
    await wd.recover("stall")
    assert events[-1].action == "reopen_failed"
    assert first_next == clock()                                    # first retry is immediate
    backoff = cw.REOPEN_BACKOFF_BASE_S * 2
    assert wd._next_reopen_t - clock() == backoff
    feed(wd, clock, zeros, backoff - 0.5)                           # silent, but inside the backoff
    assert wd.check() is None
    feed(wd, clock, zeros, 1.0)
    assert wd.check() == "silence"
    for _ in range(20):
        await wd.recover("silence")
    assert wd._next_reopen_t - clock() == cw.REOPEN_BACKOFF_MAX_S


async def test_live_signal_resets_backoff():
    wd, clock, _ = make(engine=FakeEngine(ok=False))
    for _ in range(4):
        await wd.recover("silence")
    wd.observe(noise())
    await wd.recover("silence")
    assert wd._next_reopen_t == clock()


async def test_wake_reopens_after_delay(monkeypatch):
    monkeypatch.setattr(cw, "WAKE_REOPEN_DELAY_S", 0.01)
    engine = FakeEngine()
    wd, _, events = make(engine=engine)
    wd.on_power("sleep")
    wd.on_power("wake")
    assert not wd.sleeping
    await asyncio.sleep(0.05)
    assert engine.reopens == ["power_wake"]
    assert events[-1] == CaptureHealth("power_wake", "reopened")
    await wd.stop()


async def test_unlock_reopens_only_after_a_missed_wake(monkeypatch):
    monkeypatch.setattr(cw, "WAKE_REOPEN_DELAY_S", 0.01)
    engine = FakeEngine()
    wd, _, _ = make(engine=engine)
    wd.on_power("unlock")
    await asyncio.sleep(0.05)
    assert engine.reopens == []
    wd.on_power("sleep")
    wd.on_power("unlock")
    await asyncio.sleep(0.05)
    assert engine.reopens == ["power_wake"]
    await wd.stop()


async def test_event_handler_failure_does_not_break_recovery():
    def boom(_event):
        raise RuntimeError("handler")

    engine = FakeEngine()
    wd = CaptureWatchdog(engine, on_event=boom, clock=FakeClock(), default_input=lambda: None)
    assert await wd.recover("stall") is True


# ---------------------------------------------------------------- health checks (log only)

def test_clip_event_rate_limited():
    wd, clock, events = make()
    loud = noise()
    loud[10] = 0.995
    wd.observe(loud)
    wd.observe(loud)
    clock.advance(cw.HEALTH_EVENT_INTERVAL_S)
    wd.observe(loud)
    clips = [e for e in events if e.issue == "clip"]
    assert len(clips) == 2 and clips[0].action == "logged"


def test_dropout_on_sudden_zero_run():
    wd, clock, events = make()
    feed(wd, clock, noise, 0.5)
    gap = noise(n=int(cw.DROPOUT_WINDOW_S * RATE) * 2)
    gap[: int(cw.DROPOUT_WINDOW_S * RATE) * 2] = 0.0
    wd.observe(gap)
    assert [e.issue for e in events] == ["dropout"]


def test_no_dropout_on_steady_noise():
    wd, clock, events = make()
    feed(wd, clock, noise, 2.0)
    assert events == []


def test_hum_detected_for_mains_tone_only():
    assert hum_share(tone(60.0, 0.05, 1.0), RATE) is not None
    assert hum_share(tone(50.0, 0.05, 1.0) + tone(100.0, 0.02, 1.0), RATE) is not None
    assert hum_share(tone(60.0, 0.05, 1.0) + tone(1000.0, 0.05, 1.0), RATE) is None
    assert hum_share(noise(0.05, RATE), RATE) is None
    assert hum_share(tone(60.0, cw.HUM_MIN_RMS / 10, 1.0), RATE) is None


def test_hum_event_from_frames():
    wd, clock, events = make()
    signal = tone(60.0, 0.05, 1.2)
    for i in range(0, signal.size - CHUNK, CHUNK):
        clock.advance(CHUNK_S)
        wd.observe(signal[i:i + CHUNK])
    assert [e.issue for e in events] == ["hum"]


# ---------------------------------------------------------------- quiet hours policy

def at(hh: int, mm: int) -> datetime:
    return datetime(2026, 11, 17, hh, mm)


def test_quiet_hours_across_midnight():
    q = QuietHours(enabled=True, start="22:00", end="07:00")
    assert q.active(at(23, 30)) and q.active(at(3, 0)) and q.active(at(22, 0))
    assert not q.active(at(7, 0)) and not q.active(at(12, 0)) and not q.active(at(21, 59))


def test_quiet_hours_same_day_window():
    q = QuietHours(enabled=True, start="13:00", end="14:00")
    assert q.active(at(13, 30)) and not q.active(at(14, 0)) and not q.active(at(12, 59))


def test_quiet_hours_disabled_or_empty():
    assert not QuietHours(enabled=False).active(at(23, 0))
    assert not QuietHours(enabled=True, start="09:00", end="09:00").active(at(9, 0))


def test_quiet_hours_rejects_bad_time():
    with pytest.raises(ValueError):
        QuietHours(start="7am")
    with pytest.raises(ValueError):
        QuietHours(end="24:00")


def test_quiet_hours_default_off():
    assert Config().quiet_hours == QuietHours(enabled=False, start="22:00", end="07:00")


def test_quiet_hours_from_env(monkeypatch):
    monkeypatch.setenv("VOXSCRIBE_QUIET_HOURS_ENABLED", "true")
    monkeypatch.setenv("VOXSCRIBE_QUIET_HOURS_START", "23:15")
    monkeypatch.setenv("VOXSCRIBE_QUIET_HOURS_END", "06:45")
    cfg = Config.from_env(env_file="/nonexistent/.env")
    assert cfg.quiet_hours == QuietHours(enabled=True, start="23:15", end="06:45")


def test_quiet_hours_bad_env_fails_startup(monkeypatch):
    monkeypatch.setenv("VOXSCRIBE_QUIET_HOURS_START", "25:00")
    with pytest.raises(SystemExit):
        Config.from_env(env_file="/nonexistent/.env")
