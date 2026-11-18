"""Voice host wiring (T08): ten ports, the stream clock, quiet hours, the gate before the talker.

Every port is a fake from fakes.py except where a test names a real adapter that does no I/O
(the heuristic turn detector, the fixed-rule gate). No mic, recognizer, reply model or talker.
"""
from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from voxscribe.session import ports as P
from voxscribe.session.ports import ClockDomainError
from voxscribe.wake.gate import FRAME_SAMPLES, WakeDecision

from app.local.voice import host as host_mod
from app.local.voice.host import (
    CHIME_ACK,
    CHUNK_FRAMES,
    QUIET_HOURS_REASON,
    SAMPLE_RATE,
    TICK_EVERY_FRAMES,
    VoiceConfig,
    VoiceHost,
    VoicePorts,
    WakeState,
)

from .fakes import (
    FRAMES_PER_TURN,
    TEN_HOURS_MS,
    EndlessQueue,
    FakeDelegate,
    FakeGate,
    FakeRecognizer,
    FakeSpeech,
    chunk,
    fake_ports,
)

PORT_FIELDS = {
    "audio_in": P.AudioIn, "audio_out": P.AudioOut, "wake": P.WakeDetector,
    "recognizer": P.Recognizer, "talker": P.Talker, "delegate": P.Delegate,
    "speech": P.Speech, "gate": P.Gate, "store": P.SessionStore, "clock": P.Clock,
}
CHUNK_MS = CHUNK_FRAMES * 1000 // SAMPLE_RATE
HOST_FILE = Path(host_mod.__file__)


def _host(ports: VoicePorts, **config) -> VoiceHost:
    return VoiceHost(VoiceConfig(**config), ports=ports)


# ── wiring ────────────────────────────────────────────────────────────────────


def test_all_ports_wired():
    ports = fake_ports()
    host = _host(ports)
    assert host.ready and host.state == WakeState.OFF
    assert set(PORT_FIELDS) == {f for f in VoicePorts.__dataclass_fields__}
    assert len(PORT_FIELDS) == len(P.PORTS)
    for name, proto in PORT_FIELDS.items():
        assert isinstance(getattr(ports, name), proto), name


def test_production_adapters_satisfy_their_ports():
    """The real adapters, constructed without loading a model, a device or a database."""
    from app.local.voice.adapters.asr_whisper import WhisperRecognizer
    from app.local.voice.adapters.audio_portaudio import PortAudioIn, PortAudioOut
    from app.local.voice.adapters.delegate_ollama import OllamaDelegate
    from app.local.voice.adapters.gate_rule import RuleGate
    from app.local.voice.adapters.speech_heuristic import HeuristicSpeech
    from app.local.voice.adapters.store_postgres import PostgresSessionStore
    from app.local.voice.adapters.tts_kokoro import KokoroTalker

    app = SimpleNamespace(state=SimpleNamespace(settings=None))
    for adapter, proto in [
        (PortAudioIn(None), P.AudioIn), (PortAudioOut(None), P.AudioOut),
        (WhisperRecognizer(None), P.Recognizer), (KokoroTalker(), P.Talker),
        (OllamaDelegate(app), P.Delegate), (HeuristicSpeech(), P.Speech),
        (RuleGate(), P.Gate), (PostgresSessionStore(app), P.SessionStore),
        (host_mod.StreamClock(), P.Clock),
    ]:
        assert isinstance(adapter, proto), type(adapter).__name__


def test_host_imports_adapters_only_inside_builders():
    """Module level: stdlib, numpy and the engine. Server code only as adapters, lazily."""
    tree = ast.parse(HOST_FILE.read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in top:
        names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
        assert not any(n and n.startswith("app.") for n in names), names
    lazy = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            and n.module and n.module.startswith("app.")]
    assert lazy and all(m.startswith(("app.local.voice.adapters", "app.local.voice.wake_setup"))
                        for m in lazy), lazy


# ── a conversation with every port faked ─────────────────────────────────────


async def test_three_turn_conversation_runs_with_every_port_faked():
    ports = fake_ports(recognizer=FakeRecognizer(["What is one?", "And two?", "And three?"]))
    host = _host(ports)
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"), 10)

    store = ports.store
    assert store.sessions == ["s1"] and store.closed == ["s1"]
    assert [n for n, _ in store.turns] == [1, 2, 3]
    assert [o.text for _, o in store.turns] == ["Answer 1.", "Answer 2.", "Answer 3."]
    assert ports.talker.synthesized == ["Answer 1.", "Answer 2.", "Answer 3."]
    assert ports.delegate.calls == ["What is one?", "And two?", "And three?"]
    assert host.snapshot()["meeting_id"] is None


# ── T04: the turn detector runs on stream time ───────────────────────────────


async def test_clock_domain_checked():
    """The host ticks with stream time even when the latency clock reads ten hours."""
    speech = FakeSpeech()
    ports = fake_ports(speech=speech, recognizer=FakeRecognizer(["One question here."]))
    await asyncio.wait_for(_host(ports).run_session(EndlessQueue(), score=0.9), 10)

    expected = [k * TICK_EVERY_FRAMES * CHUNK_MS
                for k in range(1, FRAMES_PER_TURN // TICK_EVERY_FRAMES + 1)]
    assert speech.ticks[: len(expected)] == expected
    assert all(t < TEN_HOURS_MS // 1000 for t in speech.ticks)


def test_clock_domain_error_raised_for_mono_time_by_the_real_turn_detector():
    from app.local.voice.adapters.speech_heuristic import HeuristicSpeech

    speech = HeuristicSpeech()
    speech.reset()
    stream = host_mod.StreamClock(mono=lambda: TEN_HOURS_MS)
    for _ in range(FRAMES_PER_TURN):
        speech.process(np.zeros(CHUNK_FRAMES, dtype=np.float32))
        stream.advance(CHUNK_FRAMES)
    speech.tick(stream.stream_ms())                           # stream time is accepted
    with pytest.raises(ClockDomainError):
        speech.tick(stream.mono_ms())


async def test_clock_domain_error_is_logged_and_the_session_survives(caplog):
    speech = FakeSpeech(tick_raises=ClockDomainError("uptime passed as stream time"))
    ports = fake_ports(speech=speech, recognizer=FakeRecognizer(["Hello there friend."]))
    with caplog.at_level(logging.ERROR, logger="app.local.wake"):
        await asyncio.wait_for(_host(ports).run_session(EndlessQueue(), score=0.9), 10)
    assert [n for n, _ in ports.store.turns] == [1]
    errors = [r for r in caplog.records if "not stream time" in r.getMessage()]
    assert len(errors) == ports.recognizer.opened     # once per turn, not once per tick


# ── T06: quiet hours ─────────────────────────────────────────────────────────


class _Always:
    def active(self) -> bool:
        return True


class _OneDecisionGate:
    """The wake gate, scripted: one accepted candidate on the first frame, then nothing."""

    def __init__(self) -> None:
        self.detector = SimpleNamespace(model_name="fake:hey_vox", patience_frames=1)
        self.config = SimpleNamespace(threshold=0.5)
        self.resets = 0
        self._sent = False

    def process(self, frame):
        assert frame.shape == (FRAME_SAMPLES,)
        if self._sent:
            return None
        self._sent = True
        return WakeDecision(accepted=True, score=0.97, detector="fake:hey_vox", at_sample=0,
                            speech_ratio=0.8, phrase_end_sample=0)

    def pop_telemetry(self):
        return None

    def reset(self):
        self.resets += 1


async def _drive_armed(host: VoiceHost, gate, n_chunks: int) -> None:
    q: asyncio.Queue = asyncio.Queue()
    for _ in range(n_chunks):
        q.put_nowait(chunk())
    host.set_state(WakeState.ARMED)
    task = asyncio.create_task(host._armed(host.ports, gate, q))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if q.empty():
            break
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_quiet_hours_blocks_session():
    ports = fake_ports()
    host = _host(ports, quiet_hours=_Always())
    gate = _OneDecisionGate()
    await _drive_armed(host, gate, n_chunks=10)

    assert ports.store.sessions == []
    assert ports.audio_out.files == []                  # no chime, no sound at all
    (row,) = ports.store.wakes
    assert row["accepted"] is False and row["reject_reason"] == QUIET_HOURS_REASON
    assert row["detector"] == "fake:hey_vox" and row["score"] == 0.97
    assert gate.resets == 1


async def test_outside_quiet_hours_the_same_wake_opens_a_session(monkeypatch):
    monkeypatch.setattr(host_mod, "CHIME_DECAY_S", 0.0)
    ports = fake_ports()
    host = _host(ports)
    await _drive_armed(host, _OneDecisionGate(), n_chunks=10)
    assert ports.audio_out.files[0] == CHIME_ACK
    assert ports.store.sessions == ["s1"]


# ── T07 / ADR 0001: the gate runs before the talker ──────────────────────────


async def test_gate_checked_before_tts():
    ports = fake_ports(gate=FakeGate(refuse="forbidden"),
                       delegate=FakeDelegate([["Fine.", "A forbidden sentence.", "Later."]]))
    text = await _host(ports).reply_turn("s1", 1, "question", turn_end_ms=0)

    assert "A forbidden sentence." not in ports.talker.synthesized
    assert "Later." not in ports.talker.synthesized
    assert ports.talker.synthesized == ["Fine."]
    assert ports.talker.fixed_phrases == ["I can't help with that."]
    assert text == "I can't help with that."
    (_, outcome), = ports.store.turns
    assert (outcome.decision, outcome.category) == ("refuse", "blocked")


async def test_gate_fault_speaks_the_scripted_fallback_never_the_candidate():
    gate = FakeGate(raises=True)
    ports = fake_ports(gate=gate, delegate=FakeDelegate([["Candidate."]]))
    text = await _host(ports).reply_turn("s1", 1, "question", turn_end_ms=0)
    assert ports.talker.synthesized == []
    assert ports.talker.fixed_phrases == [gate.fail_closed_phrase]
    assert text == gate.fail_closed_phrase


async def test_real_rule_gate_refuses_before_the_talker_in_classroom_mode():
    from app.local import gate as rules
    from app.local.voice.adapters.gate_rule import RuleGate

    ports = fake_ports(gate=RuleGate("classroom"),
                       delegate=FakeDelegate([["Here is how to make a bomb."]]))
    text = await _host(ports).reply_turn("s1", 1, "tell me something", turn_end_ms=0)
    assert ports.talker.synthesized == []
    assert text == rules.CLASSROOM_REFUSAL_PHRASE
