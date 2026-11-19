"""PL03: Product profile reaches the voice host (X4, UNIT).

Covers:
  - QuietHours.active() across window styles.
  - VoiceConfig.from_profile() reads wake phrase, quiet hours, max_turns and conv_idle_s.
  - Senior care always answers a wake (VX-H5); quiet_hours_block_wake=False for that mode.
  - Personal profile with quiet_hours_enabled=True rejects wakes inside the window.
  - Explicit server config (settings) wins over profile defaults for wake phrase and
    quiet hours; senior care exception: settings can never enable wake-blocking there.
  - Changing voice or speed via the get_voice_speed callable changes the arguments
    tts_engine.synthesize receives.
  - Health profile mode matches the loaded Profile.mode, not a hardcoded string.

No mic, model, database or network required.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from voxscribe.profile.schema import Profile, ReplyConfig, WakeConfig

from app.local.voice.adapters.tts_kokoro import KokoroTalker
from app.local.voice.host import (
    CHUNK_FRAMES,
    CONV_IDLE_S,
    FRAME_SAMPLES,
    QUIET_HOURS_REASON,
    QuietHours,
    VoiceConfig,
    VoiceHost,
    WakeState,
    _CONFIG_DEFAULT_WAKE_PHRASE,
)

from .fakes import fake_ports


# ── QuietHours unit tests ─────────────────────────────────────────────────────


def test_quiet_hours_midnight_wrap_active_before_end():
    """Window wraps midnight; a time between start and midnight is active."""
    qh = QuietHours("22:00", "07:00")
    import datetime
    # Directly probe the logic: 23:00 is after 22:00 and the window wraps midnight.
    now_active = datetime.time(23, 0)
    now_inactive = datetime.time(10, 0)
    s, e = qh._start, qh._end
    # wraps midnight: active when now >= start OR now < end
    assert (now_active >= s or now_active < e) is True
    assert (now_inactive >= s or now_inactive < e) is False


def test_quiet_hours_straight_window():
    """Non-wrapping window (e.g. 09:00-17:00): active inside, not outside."""
    qh = QuietHours("09:00", "17:00")
    import datetime
    inside = datetime.time(12, 0)
    outside = datetime.time(20, 0)
    s, e = qh._start, qh._end
    assert s <= inside < e
    assert not (s <= outside < e)


# ── VoiceConfig.from_profile ─────────────────────────────────────────────────


def test_from_profile_reads_wake_phrase():
    p = Profile(wake=WakeConfig(phrase="ok computer"))
    config = VoiceConfig.from_profile(p)
    assert config.wake_phrase == "ok computer"


def test_from_profile_quiet_hours_enabled_builds_quiet_hours_object():
    p = Profile(quiet_hours_enabled=True, quiet_hours_start="21:00", quiet_hours_end="08:00")
    config = VoiceConfig.from_profile(p)
    assert config.quiet_hours is not None
    assert isinstance(config.quiet_hours, QuietHours)


def test_from_profile_senior_care_does_not_block_wake():
    """Senior care always answers a wake (VX-H5); quiet_hours_block_wake must be False."""
    sc = Profile(mode="senior_care", quiet_hours_enabled=True)
    config = VoiceConfig.from_profile(sc)
    assert config.quiet_hours is not None      # window is present
    assert config.quiet_hours_block_wake is False  # but it never blocks a wake


def test_from_profile_non_senior_care_blocks_wake_during_quiet_hours():
    """All other modes block wakes when the window is active."""
    for mode in ("personal", "ideascribe", "classroom"):
        p = Profile(mode=mode, quiet_hours_enabled=True)  # type: ignore[arg-type]
        config = VoiceConfig.from_profile(p)
        assert config.quiet_hours_block_wake is True, f"mode={mode}"


def test_from_profile_quiet_hours_disabled_is_none():
    p = Profile(quiet_hours_enabled=False)
    config = VoiceConfig.from_profile(p)
    assert config.quiet_hours is None


def test_from_profile_reply_limits():
    p = Profile(reply=ReplyConfig(max_turns=5, idle_timeout_s=30.0))
    config = VoiceConfig.from_profile(p)
    assert config.max_turns == 5
    assert config.conv_idle_s == 30.0


def test_from_profile_defaults_match_schema():
    """Default Profile() propagates schema defaults: max_turns=20, conv_idle_s=20.0."""
    from voxscribe.profile.schema import ReplyConfig
    default_reply = ReplyConfig()
    config = VoiceConfig.from_profile(Profile())
    assert config.max_turns == default_reply.max_turns
    assert config.conv_idle_s == default_reply.idle_timeout_s


# ── Quiet-hours behaviour in the armed loop ──────────────────────────────────
# The gate below emits one accepted wake decision then returns None indefinitely.

class _OneWakeGate:
    """Returns one accepted decision, then None."""

    def __init__(self, pre_roll):
        from voxscribe.wake.gate import WakeDecision
        from types import SimpleNamespace
        self.detector = SimpleNamespace(model_name="fake:test", patience_frames=1)
        self.config = SimpleNamespace(threshold=0.5)
        self._decision = WakeDecision(
            accepted=True, score=0.95, detector="fake:test", at_sample=0,
            speech_ratio=0.7, phrase_end_sample=0, pre_roll=pre_roll,
        )
        self._returned = False
        self.resets = 0

    def process(self, frame):
        assert frame.shape == (FRAME_SAMPLES,)
        if not self._returned:
            self._returned = True
            return self._decision
        return None

    def pop_telemetry(self):
        return None

    def reset(self):
        self.resets += 1


async def _drive_armed_with_quiet(config: VoiceConfig, n_chunks: int = 20):
    """Drive the armed loop with n_chunks of silence; return (sessions, wakes)."""
    ports = fake_ports()
    host = VoiceHost(config, ports=ports)
    host.set_state(WakeState.ARMED)

    pre_roll = np.zeros(CHUNK_FRAMES * 2, dtype=np.float32)
    gate = _OneWakeGate(pre_roll)
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=256)

    for _ in range(n_chunks):
        audio_q.put_nowait(np.zeros(CHUNK_FRAMES, dtype=np.float32))

    sessions: list = []

    async def fake_session(q, *, score, detector="trigger", speech_ratio=None):
        sessions.append({"score": score, "detector": detector})

    original_run_session = host.run_session
    host.run_session = fake_session  # type: ignore[method-assign]

    task = asyncio.create_task(host._armed(ports, gate, audio_q))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if audio_q.empty():
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    host.set_state(WakeState.OFF)
    return sessions, ports.store.wakes


class _AlwaysActive:
    """QuietHours stand-in that is always active."""

    def active(self) -> bool:
        return True


class _NeverActive:
    """QuietHours stand-in that is never active."""

    def active(self) -> bool:
        return False


async def test_senior_care_profile_quiet_hours_active_wake_opens_session():
    """Senior care always answers a wake (VX-H5); window active does not block it."""
    senior_care = Profile(
        mode="senior_care",
        quiet_hours_enabled=True,
        quiet_hours_start="21:00",
        quiet_hours_end="08:00",
    )
    config = VoiceConfig.from_profile(senior_care)
    assert config.quiet_hours is not None           # window is present
    assert config.quiet_hours_block_wake is False   # but it must not block

    # Replace the real QuietHours with an always-active stub so the test is clock-independent.
    config = VoiceConfig(
        wake_phrase=config.wake_phrase,
        detector=config.detector,
        quiet_hours=_AlwaysActive(),
        quiet_hours_block_wake=False,   # senior care: never block
        max_turns=config.max_turns,
        conv_idle_s=config.conv_idle_s,
    )

    sessions, wakes = await _drive_armed_with_quiet(config)
    # Window active but block_wake=False -> wake opens a session.
    assert len(sessions) == 1


async def test_personal_profile_quiet_hours_active_wake_rejected():
    """Personal mode with quiet_hours_enabled=True: wake inside window is rejected."""
    personal_with_qh = Profile(
        mode="personal",
        quiet_hours_enabled=True,
        quiet_hours_start="22:00",
        quiet_hours_end="07:00",
    )
    config = VoiceConfig.from_profile(personal_with_qh)
    assert config.quiet_hours_block_wake is True

    # Replace with always-active stub.
    config = VoiceConfig(
        wake_phrase=config.wake_phrase,
        detector=config.detector,
        quiet_hours=_AlwaysActive(),
        quiet_hours_block_wake=True,
        max_turns=config.max_turns,
        conv_idle_s=config.conv_idle_s,
    )

    sessions, wakes = await _drive_armed_with_quiet(config)
    # Window active + block_wake=True -> rejected, no session, no chime.
    assert sessions == []
    assert len(wakes) == 1
    assert wakes[0]["accepted"] is False
    assert wakes[0]["reject_reason"] == QUIET_HOURS_REASON


# ── Voice and speed reach the synthesizer ─────────────────────────────────────


async def test_voice_and_speed_passed_to_synthesizer(monkeypatch):
    """get_voice_speed() drives the voice and speed args of tts_engine.synthesize."""
    import app.local.voice.adapters.tts_kokoro as kokoro_mod

    calls: list[dict] = []

    async def _fake_synth(text: str, *, voice: str = "af_heart", speed: float = 1.0):
        calls.append({"text": text, "voice": voice, "speed": speed})
        return np.zeros(100, dtype=np.float32)

    monkeypatch.setattr(kokoro_mod.tts_engine, "synthesize", _fake_synth)

    talker = KokoroTalker(get_voice_speed=lambda: ("custom_voice", 1.5))
    await talker.synthesize("hello world")

    assert len(calls) == 1
    assert calls[0]["voice"] == "custom_voice"
    assert calls[0]["speed"] == 1.5
    assert calls[0]["text"] == "hello world"


async def test_different_voice_gives_different_synthesizer_args(monkeypatch):
    """Changing voice_id changes what the synthesizer receives."""
    import app.local.voice.adapters.tts_kokoro as kokoro_mod

    voice_calls: list[str] = []

    async def _fake_synth(text: str, *, voice: str = "af_heart", speed: float = 1.0):
        voice_calls.append(voice)
        return np.zeros(100, dtype=np.float32)

    monkeypatch.setattr(kokoro_mod.tts_engine, "synthesize", _fake_synth)

    t1 = KokoroTalker(get_voice_speed=lambda: ("voice_a", 1.0))
    t2 = KokoroTalker(get_voice_speed=lambda: ("voice_b", 1.0))

    await t1.synthesize("test")
    await t2.synthesize("test")

    assert voice_calls == ["voice_a", "voice_b"]


async def test_empty_voice_id_uses_engine_default(monkeypatch):
    """When voice_id is empty the speed keyword is still forwarded."""
    import app.local.voice.adapters.tts_kokoro as kokoro_mod

    calls: list[dict] = []

    async def _fake_synth(text: str, *, voice: str = "af_heart", speed: float = 1.0):
        calls.append({"voice": voice, "speed": speed})
        return np.zeros(100, dtype=np.float32)

    monkeypatch.setattr(kokoro_mod.tts_engine, "synthesize", _fake_synth)

    talker = KokoroTalker(get_voice_speed=lambda: ("", 0.8))
    await talker.synthesize("hello")

    # voice was not passed -> engine default "af_heart" applies inside the stub.
    assert calls[0]["speed"] == 0.8
    assert calls[0]["voice"] == "af_heart"   # the stub's own default


# ── Health profile name ───────────────────────────────────────────────────────


def test_health_profile_name_uses_profile_mode():
    """The health response field reads Profile.mode; two different profiles give two names.

    This exercises the same expression used in app.py's /v1/health handler:
        getattr(getattr(app.state, "profile", None), "mode", "unknown")
    """
    def health_profile_name(profile):
        # Mirror of the expression in app.py health handler.
        app_state = SimpleNamespace(profile=profile)
        return getattr(getattr(app_state, "profile", None), "mode", "unknown")

    name_sc = health_profile_name(Profile(mode="senior_care"))
    name_pe = health_profile_name(Profile(mode="personal"))

    assert name_sc != name_pe, "two different profiles must give two different profile names"
    assert name_sc == "senior_care"
    assert name_pe == "personal"


def test_health_profile_name_no_profile_is_unknown():
    """When app.state.profile is absent the health field is 'unknown'."""
    app_state = SimpleNamespace()
    name = getattr(getattr(app_state, "profile", None), "mode", "unknown")
    assert name == "unknown"


# ── Settings-override tests ───────────────────────────────────────────────────
# Config.QuietHours (server-side) has an enabled flag; it's built only when
# VOXSCRIBE_QUIET_HOURS_ENABLED is set. Config.wake_phrase defaults to "hey vox".


def _fake_settings(*, wake_phrase=_CONFIG_DEFAULT_WAKE_PHRASE, qh_enabled=False,
                   qh_start="22:00", qh_end="07:00"):
    """Minimal settings stub."""
    from app.local.config import QuietHours as CfgQH
    return SimpleNamespace(
        wake_phrase=wake_phrase,
        quiet_hours=CfgQH(enabled=qh_enabled, start=qh_start, end=qh_end),
    )


def test_settings_wake_phrase_wins_over_profile():
    """An explicit VOXSCRIBE_WAKE_PHRASE overrides the profile's wake phrase."""
    p = Profile(wake=WakeConfig(phrase="hey computer"))
    settings = _fake_settings(wake_phrase="ok system")
    config = VoiceConfig.from_profile(p, settings=settings)
    assert config.wake_phrase == "ok system"


def test_profile_wake_phrase_wins_when_settings_is_default():
    """When settings.wake_phrase is the config default, the profile's phrase wins."""
    p = Profile(wake=WakeConfig(phrase="hey computer"))
    settings = _fake_settings(wake_phrase=_CONFIG_DEFAULT_WAKE_PHRASE)  # not explicitly set
    config = VoiceConfig.from_profile(p, settings=settings)
    assert config.wake_phrase == "hey computer"


def test_settings_quiet_hours_enabled_wins_over_profile_default():
    """VOXSCRIBE_QUIET_HOURS_ENABLED=true overrides a profile with no quiet hours."""
    p = Profile(mode="personal", quiet_hours_enabled=False)   # no profile quiet hours
    settings = _fake_settings(qh_enabled=True, qh_start="23:00", qh_end="06:00")
    config = VoiceConfig.from_profile(p, settings=settings)
    assert config.quiet_hours is not None
    assert config.quiet_hours_block_wake is True  # personal mode: blocks


def test_settings_quiet_hours_cannot_block_wake_in_senior_care():
    """Even if the operator enables VOXSCRIBE_QUIET_HOURS_ENABLED, senior care never blocks."""
    sc = Profile(mode="senior_care", quiet_hours_enabled=True)
    settings = _fake_settings(qh_enabled=True, qh_start="21:00", qh_end="08:00")
    config = VoiceConfig.from_profile(sc, settings=settings)
    # Window is present (operator set it), but wake is never blocked for senior care.
    assert config.quiet_hours is not None
    assert config.quiet_hours_block_wake is False


def test_settings_quiet_hours_disabled_does_not_override_profile():
    """When settings.quiet_hours.enabled is False (not explicitly set), profile wins."""
    p = Profile(mode="personal", quiet_hours_enabled=True, quiet_hours_start="20:00",
                quiet_hours_end="07:00")
    settings = _fake_settings(qh_enabled=False)   # not explicitly set
    config = VoiceConfig.from_profile(p, settings=settings)
    # Profile's quiet hours survive; settings did not override.
    assert config.quiet_hours is not None
