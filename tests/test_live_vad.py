"""The live path's turn detection: the configured mode builds the right transcriber, the socket messages carry the
speech activity fields, and health reports the capability."""

import dataclasses

import pytest
from voxscribe.meeting.live import SpeechTransition
from voxscribe.speech import SEGMENTATION, SPEC_VERSION
from voxscribe.speech.silero import SileroDetector

from app.local import live_support
from app.local.config import Config
from app.local.providers import FakeASRProvider


def test_energy_modes_build_the_gates_they_always_did():
    for mode in ("adaptive", "fixed"):
        t = live_support.build_live_transcriber(Config(live_vad_mode=mode), FakeASRProvider(), "m1")
        assert t.vad_mode == mode and t.detector_name == "energy" and t.max_segment_ms == 15000
    t = live_support.build_live_transcriber(Config(live_vad_mode="adaptive", live_max_segment_ms=0), FakeASRProvider(), "m1")
    assert t.max_segment_ms is None


def test_heuristic_mode_runs_the_shared_segmentation_policy():
    t = live_support.build_live_transcriber(Config(live_vad_mode="heuristic"), FakeASRProvider(), "m1")
    assert t.vad_mode == "detector" and t.detector_name == "heuristic" and t.policy is SEGMENTATION
    assert t.pre_roll_ms == SEGMENTATION.pre_roll_ms and t.silence_duration_ms == SEGMENTATION.hangover_ms


def test_silero_mode_builds_a_silero_detector_or_fails_loudly(monkeypatch):
    if SileroDetector.is_available():
        t = live_support.build_live_transcriber(Config(live_vad_mode="silero"), FakeASRProvider(), "m1")
        assert t.vad_mode == "detector" and t.detector_name == "silero"
    from voxscribe.speech import SpeechDetectorUnavailable

    def unavailable(*args, **kwargs):
        raise SpeechDetectorUnavailable("no model here")

    monkeypatch.setattr(live_support, "create_detector", unavailable)
    with pytest.raises(SpeechDetectorUnavailable):
        live_support.build_live_transcriber(Config(live_vad_mode="silero"), FakeASRProvider(), "m1")


def test_the_vad_message_is_the_old_one_plus_three_additive_fields():
    assert live_support.vad_message(SpeechTransition(True, 1216, 0.9123, "silero")) == {
        "type": "vad", "speech": True, "at_ms": 1216, "p": 0.912, "detector": "silero",
    }
    assert live_support.vad_message(SpeechTransition(False, 2700, None, "energy")) == {
        "type": "vad", "speech": False, "at_ms": 2700, "p": None, "detector": "energy",
    }
    assert live_support.level_message(4352, 0.04149) == {"type": "vad.level", "at_ms": 4352, "p": 0.041}


def test_the_capability_names_the_real_detectors_the_mode_and_the_spec_version():
    cap = live_support.speech_activity_capability(Config(live_vad_mode="heuristic"))
    assert cap["live_mode"] == "heuristic" and cap["spec_version"] == SPEC_VERSION
    assert "heuristic" in cap["detectors"] and "fake" not in cap["detectors"]
    assert ("silero" in cap["detectors"]) == SileroDetector.is_available()


@pytest.mark.asyncio
async def test_health_reports_the_speech_activity_capability(client):
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, live_vad_mode="heuristic")
    body = (await client.get("/v1/health")).json()
    assert isinstance(body["capabilities"], list) and "speech_activity" in body["capabilities"]
    assert body["speech_activity"] == live_support.speech_activity_capability(app.state.settings)
    assert body["speech_activity"]["live_mode"] == "heuristic"
    assert set(body["speech_activity"]) == {"detectors", "live_mode", "spec_version"}
