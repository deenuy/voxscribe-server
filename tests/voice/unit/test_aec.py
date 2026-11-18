"""Capture clean-up (T12): echo cancellation, noise suppression and the echo guard behind them.

Fakes only: no microphone, no speaker. Tests that need the real WebRTC processor skip when
pywebrtc-audio is not installed (`uv sync --extra aec`).
"""

import asyncio
import math
import sys
import types

import numpy as np
import pytest

from app.local import audio_engine as ae
from app.local.audio_engine import AcousticEchoCanceller, AudioEngine
from app.local.config import Config

CAPTURE_RATE = 16_000
PLAYBACK_RATE = 24_000
CHUNK = 512                       # the engine's capture chunk, 32 ms
CONVERGE_S = 2.0                  # audio given to the canceller before it is judged
ECHO_DELAY_S = 0.06               # output buffer plus loudspeaker-to-mic path
ECHO_GAIN = 0.6
PURE_ECHO_MAX_RMS = 0.01          # T12 brief: a pure echo leaves less than this
ROOM_ECHO_MIN_ERLE_DB = 20.0      # echo return loss enhancement on a simulated room
PASSTHROUGH_MIN_CORR = 0.97      # high-pass filter and suppressor gain wobble, nothing more
PASSTHROUGH_GAIN_TOL = 0.1
HUM_HZ = 60.0
HUM_MIN_REDUCTION_DB = 6.0
MAX_LAG = 512                     # processing delay searched when comparing input and output
SPEECH_RMS = 0.1                  # about -20 dBFS, ordinary speech at a laptop microphone
ASSISTANT_F0 = 140.0


def _needs_webrtc():
    pytest.importorskip("pywebrtc_audio", reason="pywebrtc-audio not installed (uv sync --extra aec)")


def _speechlike(seconds: float, rate: int, seed: int, f0: float = ASSISTANT_F0) -> np.ndarray:
    """Voiced harmonics on a drifting pitch plus fricative noise, under a syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pitch = f0 * (1 + 0.2 * np.sin(2 * math.pi * 0.7 * t + rng.uniform(0, 2 * math.pi)))
    phase = 2 * math.pi * np.cumsum(pitch) / rate
    voiced = sum(np.sin(k * phase) / k for k in range(1, 40) if k * f0 * 1.2 < rate / 2 - 1000)
    hiss = rng.standard_normal(n) * 0.3
    envelope = np.clip(np.sin(2 * math.pi * 4.0 * t + rng.uniform(0, 2 * math.pi)), 0, None) ** 2
    x = (voiced + hiss) * envelope
    return (SPEECH_RMS * x / (_rms(x) + 1e-12)).astype(np.float32)


def _room(far_playback: np.ndarray) -> np.ndarray:
    """What the microphone hears: delayed, reverberant, attenuated, captured at 16 kHz."""
    from scipy.signal import resample_poly

    ir = np.zeros(int(0.08 * PLAYBACK_RATE))
    ir[0], ir[len(ir) // 3], ir[-1] = 1.0, 0.3, 0.1
    delay = int(ECHO_DELAY_S * PLAYBACK_RATE)
    wet = np.convolve(far_playback, ir)[: far_playback.size]
    echo = np.concatenate([np.zeros(delay), wet])[: far_playback.size] * ECHO_GAIN
    return resample_poly(echo, CAPTURE_RATE, PLAYBACK_RATE).astype(np.float32)


def _run(aec: AcousticEchoCanceller, capture: np.ndarray) -> np.ndarray:
    out = [aec.process_capture(capture[i: i + CHUNK]) for i in range(0, capture.size - CHUNK + 1, CHUNK)]
    for i, o in enumerate(out):
        assert o.shape == (CHUNK,), f"chunk {i} came back with shape {o.shape}"
    return np.concatenate(out)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _aligned(out: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """out and ref trimmed to the lag where they line up best (the processor delays its output)."""
    n = min(out.size, ref.size) - MAX_LAG
    lag = max(range(MAX_LAG), key=lambda d: float(np.dot(out[d: d + n], ref[:n])))
    return out[lag: lag + n], ref[:n]


# ---- echo cancellation -------------------------------------------------------------------------


def test_pure_echo_removed():
    _needs_webrtc()
    signal = _speechlike(6.0, CAPTURE_RATE, seed=1)
    aec = AcousticEchoCanceller(capture_rate=CAPTURE_RATE, playback_rate=CAPTURE_RATE)
    aec.push_reference(signal)
    out = _run(aec, signal)
    settled = out[int(CONVERGE_S * CAPTURE_RATE):]
    assert _rms(signal) > 5 * PURE_ECHO_MAX_RMS  # the input is loud, so the bound means something
    assert _rms(settled) < PURE_ECHO_MAX_RMS


def test_room_echo_removed_from_playback_rate_reference():
    """The engine path: a 24 kHz reference, a delayed and reverberant 16 kHz echo."""
    _needs_webrtc()
    pytest.importorskip("scipy")
    far = _speechlike(6.0, PLAYBACK_RATE, seed=2)
    capture = _room(far)
    aec = AcousticEchoCanceller(capture_rate=CAPTURE_RATE, playback_rate=PLAYBACK_RATE)
    aec.push_reference(far)
    out = _run(aec, capture)
    start = int(CONVERGE_S * CAPTURE_RATE)
    erle_db = 20 * math.log10(_rms(capture[start: out.size]) / max(_rms(out[start:]), 1e-9))
    assert erle_db > ROOM_ECHO_MIN_ERLE_DB


def test_capture_passthrough_no_reference():
    """Nothing playing: the user's audio comes through, delayed a few ms, level unchanged."""
    _needs_webrtc()
    t = np.arange(int(3.0 * CAPTURE_RATE)) / CAPTURE_RATE
    capture = sum(0.05 * np.sin(2 * math.pi * f * t) for f in (500.0, 1000.0, 2000.0)).astype(np.float32)
    aec = AcousticEchoCanceller(capture_rate=CAPTURE_RATE, noise_suppression=False)
    out = _run(aec, capture)
    settled = int(1.0 * CAPTURE_RATE)
    a, b = _aligned(out[settled:], capture[settled:])
    assert np.corrcoef(a, b)[0, 1] > PASSTHROUGH_MIN_CORR
    assert abs(_rms(a) / _rms(b) - 1.0) < PASSTHROUGH_GAIN_TOL


# ---- noise suppression -------------------------------------------------------------------------


def test_noise_suppress_reduces_rms():
    _needs_webrtc()
    t = np.arange(int(3.0 * CAPTURE_RATE)) / CAPTURE_RATE
    hum = (0.05 * np.sin(2 * math.pi * HUM_HZ * t)).astype(np.float32)
    settled = int(1.0 * CAPTURE_RATE)

    ns = AcousticEchoCanceller(capture_rate=CAPTURE_RATE, echo_cancellation=False,
                               noise_suppression=True)
    reduced_db = 20 * math.log10(_rms(hum[settled:]) / _rms(_run(ns, hum)[settled:]))
    assert reduced_db > HUM_MIN_REDUCTION_DB

    off = AcousticEchoCanceller(capture_rate=CAPTURE_RATE, echo_cancellation=False,
                                noise_suppression=False)
    assert np.array_equal(_run(off, hum), hum[: (hum.size // CHUNK) * CHUNK])


# ---- disabled and missing ----------------------------------------------------------------------


def test_aec_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("VOXSCRIBE_AEC_ENABLED", "false")
    monkeypatch.setenv("VOXSCRIBE_NOISE_SUPPRESS_ENABLED", "false")
    cfg = Config.from_env(env_file="/nonexistent/.env")
    assert cfg.aec_enabled is False and cfg.noise_suppress_enabled is False

    aec = AcousticEchoCanceller(echo_cancellation=cfg.aec_enabled,
                                noise_suppression=cfg.noise_suppress_enabled)
    frame = _speechlike(CHUNK / CAPTURE_RATE, CAPTURE_RATE, seed=5)
    aec.push_reference(frame)
    assert not aec.active
    assert aec.process_capture(frame) is frame


def test_config_defaults(monkeypatch):
    """Cancellation and the guard on; suppression opt-in (it raised word error at 10 dB SNR)."""
    for name in ("VOXSCRIBE_AEC_ENABLED", "VOXSCRIBE_NOISE_SUPPRESS_ENABLED",
                 "VOXSCRIBE_VOICE_ECHO_GUARD"):
        monkeypatch.delenv(name, raising=False)
    cfg = Config.from_env(env_file="/nonexistent/.env")
    assert cfg.aec_enabled and cfg.voice_echo_guard
    assert not cfg.noise_suppress_enabled
    monkeypatch.setenv("VOXSCRIBE_NOISE_SUPPRESS_ENABLED", "true")
    assert Config.from_env(env_file="/nonexistent/.env").noise_suppress_enabled


def test_missing_library_falls_back_to_passthrough(monkeypatch):
    monkeypatch.setattr(ae, "_load_audio_processor", lambda: None)
    aec = AcousticEchoCanceller()
    frame = _speechlike(CHUNK / CAPTURE_RATE, CAPTURE_RATE, seed=6)
    assert not aec.active and not aec.echo_cancellation
    assert aec.process_capture(frame) is frame


def test_processor_fault_passes_raw_audio():
    class Broken:
        def __init__(self, **_):
            pass

        def process(self, *_):
            raise RuntimeError("boom")

        def reset(self):
            pass

    aec = AcousticEchoCanceller(processor_cls=Broken)
    frame = _speechlike(CHUNK / CAPTURE_RATE, CAPTURE_RATE, seed=7)
    assert np.array_equal(aec.process_capture(frame), frame)


# ---- streaming helpers -------------------------------------------------------------------------


def test_resampler_is_continuous_across_chunks():
    x = _speechlike(1.0, PLAYBACK_RATE, seed=8)
    whole = ae._LinearResampler(np, PLAYBACK_RATE, CAPTURE_RATE)(x)
    r = ae._LinearResampler(np, PLAYBACK_RATE, CAPTURE_RATE)
    sizes = [1, 2, 7, 333, 480, 1024, 5000]
    parts, i = [], 0
    while i < x.size:
        n = sizes[len(parts) % len(sizes)]
        parts.append(r(x[i: i + n]))
        i += n
    pieces = np.concatenate(parts)
    assert abs(pieces.size - x.size * CAPTURE_RATE / PLAYBACK_RATE) <= 1
    m = min(pieces.size, whole.size)
    np.testing.assert_allclose(pieces[:m], whole[:m], atol=1e-6)


def test_fifo_pops_exact_lengths_and_zero_fills():
    f = ae._SampleFifo(np, max_samples=5)
    f.push([1, 2, 3])
    f.push([4, 5, 6, 7])         # over the cap: the oldest two go
    assert len(f) == 5 and f.dropped == 2
    assert f.pop(3).tolist() == [3, 4, 5]
    assert f.pop(4).tolist() == [6, 7, 0, 0]
    assert len(f) == 0


# ---- engine wiring (fake sounddevice) ----------------------------------------------------------


class _RecordingProcessor:
    """Stands in for the WebRTC processor: marks the output and records what it was given."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.resets = 0

    def process(self, near, far=None):
        self.calls.append((near.copy(), None if far is None else far.copy()))
        return near - (far if far is not None else 0.0)

    def reset(self):
        self.resets += 1


class _FakeInputStream:
    instances: list = []

    def __init__(self, callback=None, **_):
        self.callback = callback
        _FakeInputStream.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


class _FakeOutputStream(_FakeInputStream):
    def __init__(self, engine_ref, **kwargs):
        super().__init__(**kwargs)
        self.engine_ref = engine_ref
        self.writes: list[np.ndarray] = []
        self.reference_queued_at_write: list[int] = []

    def write(self, chunk):
        aec = self.engine_ref[0].aec
        self.reference_queued_at_write.append(len(aec._far))
        self.writes.append(np.asarray(chunk).copy())


@pytest.fixture
def fake_engine(monkeypatch):
    holder: list = []
    sd = types.SimpleNamespace(
        InputStream=_FakeInputStream,
        OutputStream=lambda **kw: _FakeOutputStream(holder, **kw),
        query_devices=lambda: [],
        _terminate=lambda: None,
        _initialize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    processors: list[_RecordingProcessor] = []

    def factory(**kwargs):
        p = _RecordingProcessor(**kwargs)
        processors.append(p)
        return p

    monkeypatch.setattr(ae, "_load_audio_processor", lambda: factory)

    def make(**kwargs):
        engine = AudioEngine(**kwargs)
        holder.append(engine)
        return engine

    return make, processors


async def _played(engine, samples) -> _FakeOutputStream:
    out = engine._playback_stream
    before = len(out.writes)
    await engine.play(samples, block=True)
    for _ in range(100):
        if len(out.writes) > before:
            break
        await asyncio.sleep(0)
    return out


def _capture(engine, samples: np.ndarray) -> None:
    engine._capture_cb(samples.reshape(-1, 1), samples.size, None, None)


async def test_engine_cleans_capture_against_what_it_plays(fake_engine):
    make, processors = fake_engine
    engine = make()
    await engine.start()
    try:
        q = engine.subscribe()
        far = _speechlike(0.1, PLAYBACK_RATE, seed=9)
        out = await _played(engine, far)
        assert out.reference_queued_at_write[0] > 0, "reference must be queued before the write"

        mic = _speechlike(CHUNK / CAPTURE_RATE, CAPTURE_RATE, seed=10)
        _capture(engine, mic)
        await asyncio.sleep(0)
        frame = q.get_nowait()
        assert frame.shape == (CHUNK,)

        proc = processors[0]
        assert proc.kwargs["echo_cancellation"] and not proc.kwargs["noise_suppression"]
        fed_far = np.concatenate([f for _, f in proc.calls])
        expected = ae._LinearResampler(np, PLAYBACK_RATE, CAPTURE_RATE)(far)
        np.testing.assert_allclose(fed_far[: expected.size], expected[: fed_far.size], atol=1e-6)
        assert not np.array_equal(frame, mic)  # subscribers get the processed signal
    finally:
        await engine.stop()


async def test_echoguard_still_mutes_during_playback(fake_engine):
    """Belt and suspenders: with the canceller on, playback still arms the time-based mute."""
    make, _ = fake_engine
    engine = make()
    await engine.start()
    try:
        assert engine.aec.active
        await _played(engine, _speechlike(0.1, PLAYBACK_RATE, seed=11))
        verdict, _ = engine.echo_guard.observe(np.zeros(CHUNK, np.float32), None)
        assert verdict == "muted"
    finally:
        await engine.stop()


async def test_echoguard_off_when_disabled(fake_engine):
    make, _ = fake_engine
    engine = make(echo_guard=False)
    await engine.start()
    try:
        await _played(engine, _speechlike(0.1, PLAYBACK_RATE, seed=12))
        verdict, _ = engine.echo_guard.observe(np.zeros(CHUNK, np.float32), None)
        assert verdict == "pass"
    finally:
        await engine.stop()


async def test_reopen_resets_the_echo_path(fake_engine):
    make, processors = fake_engine
    engine = make()
    await engine.start()
    try:
        engine.aec.push_reference(_speechlike(0.5, PLAYBACK_RATE, seed=13))
        assert len(engine.aec._far) > 0
        assert await engine.reopen_stream("test")
        assert len(engine.aec._far) == 0 and processors[0].resets == 1
    finally:
        await engine.stop()


async def test_raw_subscribers_get_the_device_signal(fake_engine):
    """The capture watchdog judges the device, not the clean-up: it gets the chunk unprocessed."""
    make, _ = fake_engine
    engine = make()
    await engine.start()
    try:
        cleaned, raw = engine.subscribe(), engine.subscribe_raw()
        mic = np.full(CHUNK, 0.1, dtype=np.float32)
        _capture(engine, mic)
        await asyncio.sleep(0)
        assert np.array_equal(raw.get_nowait(), mic)
        assert not np.array_equal(cleaned.get_nowait(), mic)
        engine.unsubscribe(raw)
        assert raw not in engine._raw_capture_qs
    finally:
        await engine.stop()


def test_watchdog_prefers_the_raw_subscription():
    from app.local.voice.capture_watchdog import CaptureWatchdog

    class Engine:
        capture_rate = CAPTURE_RATE
        capture_device = 0

        def __init__(self):
            self.used = None

        def subscribe(self):
            self.used = "cleaned"
            return asyncio.Queue()

        def subscribe_raw(self):
            self.used = "raw"
            return asyncio.Queue()

        def unsubscribe(self, q):
            pass

    async def go():
        engine = Engine()
        wd = CaptureWatchdog(engine, on_event=lambda e: None, default_input=lambda: None)
        task = asyncio.create_task(wd.run())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return engine.used

    assert asyncio.run(go()) == "raw"

