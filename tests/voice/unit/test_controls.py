"""Unit tests for spoken control detection and ControlState (T15).

No TTS, no LLM, no I/O.  Covers:
  - Each of the 8 intents detected from one sample phrase
  - ControlState speed clamped at SPEED_MIN and SPEED_MAX
  - ControlState volume clamped at VOLUME_MIN and VOLUME_MAX
  - REPEAT re-speaks last turn without calling the LLM (fake-only)
  - HOLD keeps session open without calling the LLM (fake-only)
"""
from __future__ import annotations

import pytest

from app.local.voice.controls import (
    ACK_CUE,
    SPEED_MAX,
    SPEED_MIN,
    VOLUME_MAX,
    VOLUME_MIN,
    ControlIntent,
    ControlState,
    detect_control,
)


# ---------------------------------------------------------------------------
# detect_control: one sample phrase per intent
# ---------------------------------------------------------------------------

class TestDetect:
    def test_hold(self):
        assert detect_control("hold on") == ControlIntent.HOLD

    def test_slower(self):
        assert detect_control("slow down please") == ControlIntent.SLOWER

    def test_faster(self):
        assert detect_control("speed up a bit") == ControlIntent.FASTER

    def test_shorter(self):
        assert detect_control("keep it brief") == ControlIntent.SHORTER

    def test_longer(self):
        assert detect_control("explain more about that") == ControlIntent.LONGER

    def test_repeat(self):
        assert detect_control("say that again") == ControlIntent.REPEAT

    def test_volume_up(self):
        assert detect_control("louder please") == ControlIntent.VOLUME_UP

    def test_volume_down(self):
        assert detect_control("quieter") == ControlIntent.VOLUME_DOWN

    def test_none_on_unrelated(self):
        assert detect_control("what is the weather today") == ControlIntent.NONE

    def test_case_insensitive(self):
        assert detect_control("SLOW DOWN") == ControlIntent.SLOWER

    def test_alternate_hold_phrase(self):
        assert detect_control("one moment please") == ControlIntent.HOLD

    def test_alternate_repeat_phrase(self):
        assert detect_control("what did you say") == ControlIntent.REPEAT


# ---------------------------------------------------------------------------
# ControlState: step and clamp behaviour
# ---------------------------------------------------------------------------

class TestControlState:
    def test_speed_decreases_on_slower(self):
        state = ControlState()
        old = state.speed
        state.apply_slower()
        assert state.speed < old

    def test_speed_increases_on_faster(self):
        state = ControlState()
        old = state.speed
        state.apply_faster()
        assert state.speed > old

    def test_speed_clamp_min(self):
        """Repeated slower requests cannot push speed below SPEED_MIN."""
        state = ControlState(speed=SPEED_MIN)
        for _ in range(20):
            state.apply_slower()
        assert state.speed >= SPEED_MIN

    def test_speed_clamp_max(self):
        """Repeated faster requests cannot push speed above SPEED_MAX."""
        state = ControlState(speed=SPEED_MAX)
        for _ in range(20):
            state.apply_faster()
        assert state.speed <= SPEED_MAX

    def test_volume_increases_on_up(self):
        state = ControlState()
        old = state.volume
        state.apply_volume_up()
        assert state.volume > old

    def test_volume_decreases_on_down(self):
        state = ControlState()
        old = state.volume
        state.apply_volume_down()
        assert state.volume < old

    def test_volume_clamp_min(self):
        """Repeated quieter requests cannot push volume below VOLUME_MIN."""
        state = ControlState(volume=VOLUME_MIN)
        for _ in range(20):
            state.apply_volume_down()
        assert state.volume >= VOLUME_MIN

    def test_volume_clamp_max(self):
        """Repeated louder requests cannot push volume above VOLUME_MAX."""
        state = ControlState(volume=VOLUME_MAX)
        for _ in range(20):
            state.apply_volume_up()
        assert state.volume <= VOLUME_MAX

    def test_shorter_sets_suffix(self):
        state = ControlState()
        state.apply_shorter()
        assert "2 sentences" in state.prompt_suffix

    def test_longer_sets_suffix(self):
        state = ControlState()
        state.apply_longer()
        assert "detailed" in state.prompt_suffix.lower()

    def test_profile_speed_floor_honored(self):
        """A spoken request cannot exceed the profile's minimum speed."""
        state = ControlState(speed=0.6)
        state.apply_slower(floor=0.6)
        assert state.speed >= 0.6

    def test_profile_speed_ceil_honored(self):
        """A spoken request cannot exceed the profile's maximum speed."""
        state = ControlState(speed=1.5)
        state.apply_faster(ceil=1.5)
        assert state.speed <= 1.5


# ---------------------------------------------------------------------------
# Host integration: REPEAT and HOLD do not call the LLM (fake-only)
# ---------------------------------------------------------------------------

class TestHostControlIntegration:
    @pytest.mark.asyncio
    async def test_repeat_does_not_call_delegate(self):
        """REPEAT re-speaks the last reply without an LLM call."""
        from tests.voice.unit.fakes import (
            FakeDelegate,
            FakeRecognizer,
            FakeSpeech,
            FakeTalker,
            FakeAudioOut,
            fake_ports,
            EndlessQueue,
        )
        from app.local.voice.host import VoiceConfig, VoiceHost

        # Turn 1: real question -> delegate answers "Paris."
        # Turn 2: "say that again" -> REPEAT detected -> delegate NOT called
        # Turn 3: "" (empty) -> session ends
        fake_delegate = FakeDelegate(replies=[["Paris."], ["should not reach"]])
        fake_talker = FakeTalker()
        fake_out = FakeAudioOut()
        recognizer = FakeRecognizer(turns=["what is the capital of France", "say that again"])
        speech = FakeSpeech()

        ports = fake_ports(
            talker=fake_talker,
            audio_out=fake_out,
            recognizer=recognizer,
            delegate=fake_delegate,
            speech=speech,
        )
        config = VoiceConfig()
        host = VoiceHost(config, ports=ports)

        await host.run_session(EndlessQueue(), score=1.0)

        # Delegate called only once (turn 1); "say that again" never reached it.
        assert len(fake_delegate.calls) == 1, (
            f"Expected 1 delegate call, got {len(fake_delegate.calls)}: {fake_delegate.calls}"
        )
        # The last reply ("Paris.") was synthesized again on the REPEAT turn.
        assert any("Paris" in s for s in fake_talker.synthesized), (
            f"Expected 'Paris' in synthesized list, got: {fake_talker.synthesized}"
        )

    @pytest.mark.asyncio
    async def test_hold_does_not_call_delegate(self):
        """HOLD keeps the session open without calling the LLM."""
        from tests.voice.unit.fakes import (
            FakeDelegate,
            FakeRecognizer,
            FakeSpeech,
            fake_ports,
            EndlessQueue,
        )
        from app.local.voice.host import VoiceConfig, VoiceHost

        # Turn 1: "hold on" -> HOLD intent -> no LLM call
        # Turn 2: real question -> LLM called
        # Turn 3: "" -> session ends
        fake_delegate = FakeDelegate(replies=[["The time is noon."]])
        recognizer = FakeRecognizer(turns=["hold on", "what time is it"])
        speech = FakeSpeech()

        ports = fake_ports(
            recognizer=recognizer,
            delegate=fake_delegate,
            speech=speech,
        )
        config = VoiceConfig()
        host = VoiceHost(config, ports=ports)

        await host.run_session(EndlessQueue(), score=1.0)

        # "hold on" did not reach the delegate; "what time is it" did.
        assert len(fake_delegate.calls) == 1, (
            f"Expected 1 delegate call, got {len(fake_delegate.calls)}: {fake_delegate.calls}"
        )
        assert "hold on" not in fake_delegate.calls[0].lower()
