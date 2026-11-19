"""End and Mute from the app (PL36 C): fake ports only, no microphone, model or database.

End stops playback at once and closes the session the way the stop phrase does, a reply in progress ending as an
interruption with its gate records; the wake listener goes back to what it was. Mute keeps the session open while
captured audio reaches neither the recognizer nor the turn detector, so nothing goes to the model, and the idle
timeout does not end it until the mute hold has passed."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from voxscribe.session.ports import ReplyStream

from app.local.voice import host as host_mod
from app.local.voice.host import VoiceConfig, VoiceHost, WakeState

from .fakes import (
    FakeAudioIn,
    FakeDelegate,
    FakeRecognizer,
    FakeSpeech,
    FakeStore,
    chunk,
    fake_ports,
)

WAIT_S = 10
O1_BAR_S = 0.3              # interruption to silence (the O1 bar)
TURNS = ["what is the weather like today", "and what about tomorrow then", "one more question please"]


class LedgerStore(FakeStore):
    """FakeStore that keeps what the host hands its gate ledger: (turn, how the reply ended)."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed: list[tuple[int, str]] = []

    async def flush_gate_records(self, turn, *, decision, category, ended) -> None:
        self.flushed.append((turn.turn_n, ended))


class SlowDelegate(FakeDelegate):
    """A reply of several sentences, each after a pause: still generating when End arrives."""

    async def reply(self, turn_text, session_id, **kw) -> ReplyStream:
        self.calls.append(turn_text)

        async def sentences():
            for i in range(20):
                await asyncio.sleep(0.1)
                yield f"Sentence {i}."
        return ReplyStream(sentences=sentences(), fallback="nothing")


class Feeder:
    """Capture: a chunk every few milliseconds into every subscriber queue of a FakeAudioIn."""

    def __init__(self, audio_in: FakeAudioIn) -> None:
        self.audio_in = audio_in
        self.task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        while True:
            for q in list(self.audio_in.queues):
                q.put_nowait(chunk())
            await asyncio.sleep(0.003)

    def stop(self) -> None:
        self.task.cancel()


async def until(check, what: str) -> None:
    deadline = time.monotonic() + WAIT_S
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


def _host(*, echo_cancelled: bool = True, delegate=None, config: VoiceConfig | None = None):
    audio_in = FakeAudioIn()
    audio_in.echo_cancelled = echo_cancelled
    store = LedgerStore()
    ports = fake_ports(audio_in=audio_in, recognizer=FakeRecognizer(TURNS), store=store,
                       delegate=delegate or FakeDelegate(), speech=FakeSpeech())
    return VoiceHost(config or VoiceConfig(), ports=ports), ports, store


@pytest.mark.parametrize("echo_cancelled", [True, False], ids=["barge-in watch", "no barge-in watch"])
async def test_end_during_a_reply_stops_it_at_once_and_closes_the_session_like_the_stop_phrase(echo_cancelled):
    host, ports, store = _host(echo_cancelled=echo_cancelled, delegate=SlowDelegate())
    feeder = Feeder(ports.audio_in)
    try:
        sid = await asyncio.wait_for(host.start_now(), WAIT_S)
        await until(lambda: ports.delegate.calls, "the first reply to start")
        await asyncio.sleep(0.25)                    # a sentence or two in
        stops_before = ports.audio_out.stops
        started = time.monotonic()
        assert await host.end_session() is True
        elapsed = time.monotonic() - started
    finally:
        feeder.stop()
    assert ports.audio_out.stops > stops_before     # playback stopped
    assert elapsed < O1_BAR_S, elapsed              # and the session closed, inside the interruption bar
    assert store.closed == [sid] and store.discarded == []      # closed, as the stop phrase closes it
    assert [n for n, _ in store.turns] == [1] and 1 in store.interruptions      # the reply: interrupted
    assert store.flushed and set(store.flushed) == {(1, host_mod.REPLY_INTERRUPTED)}
    assert len(ports.delegate.calls) == 1           # no further turn was answered
    assert host.state == WakeState.OFF and not host._claimed and host._open is None


async def test_end_while_listening_returns_an_armed_listener_to_armed():
    host, ports, store = _host()
    feeder = Feeder(ports.audio_in)
    try:
        assert host.arm()
        sid = await asyncio.wait_for(host.start_now(), WAIT_S)
        await until(lambda: len(ports.delegate.calls) == 1, "turn 1 to be answered")
        assert await host.end_session() is True
    finally:
        feeder.stop()
    assert store.closed == [sid]
    assert host.state == WakeState.ARMED
    assert await host.end_session() is False        # nothing running now


async def test_end_with_no_session_changes_nothing():
    host, ports, store = _host()
    assert await host.end_session() is False
    assert host.set_muted(True) is False and host.muted is False
    assert ports.audio_out.stops == 0 and store.closed == [] and host.state == WakeState.OFF


async def test_mute_keeps_the_session_open_and_sends_nothing_until_unmuted():
    host, ports, store = _host(config=VoiceConfig(conv_idle_s=0.3))
    feeder = Feeder(ports.audio_in)
    try:
        await asyncio.wait_for(host.start_now(), WAIT_S)
        await until(lambda: len(ports.delegate.calls) == 1, "turn 1 to be answered")
        assert host.set_muted(True) is True and host.muted
        frames, opened = ports.speech.frames, ports.recognizer.opened
        await asyncio.sleep(1.0)                    # past the idle timeout, with capture flowing
        assert host._open is not None and host.state == WakeState.LISTENING     # still open
        assert ports.speech.frames == frames        # the turn detector heard nothing
        assert ports.recognizer.opened == opened    # no turn closed, so no new recognizer stream
        assert len(ports.delegate.calls) == 1       # and nothing went to the model
        assert store.closed == []

        assert host.set_muted(False) is False and not host.muted
        await until(lambda: len(ports.delegate.calls) == 2, "the next turn after unmute")
        assert await host.end_session() is True
    finally:
        feeder.stop()
    assert host.muted is False and host.set_muted(True) is False      # mute ended with the session


async def test_a_muted_session_ends_when_the_mute_hold_runs_out(monkeypatch):
    monkeypatch.setattr(host_mod, "MUTE_HOLD_S", 0.3)
    host, ports, store = _host(config=VoiceConfig(conv_idle_s=0.1))
    feeder = Feeder(ports.audio_in)
    try:
        sid = await asyncio.wait_for(host.start_now(), WAIT_S)
        await until(lambda: len(ports.delegate.calls) == 1, "turn 1 to be answered")
        host.set_muted(True)
        muted_at = time.monotonic()
        await until(lambda: store.closed, "the session to close")
    finally:
        feeder.stop()
    assert store.closed == [sid] and time.monotonic() - muted_at >= 0.3
    assert len(ports.delegate.calls) == 1


async def test_a_muted_person_cannot_barge_in():
    """Loud capture during a reply interrupts it, unless the session is muted."""
    host, ports, store = _host(delegate=SlowDelegate())
    loud = np.full(512, 0.2, dtype=np.float32)
    q: asyncio.Queue = asyncio.Queue()

    async def speak_over():
        for _ in range(200):
            q.put_nowait(loud)
            await asyncio.sleep(0.005)

    session = host_mod.SessionContext()
    session.muted_until = time.monotonic() + 60
    token = host_mod._session.set(session)
    try:
        speak = asyncio.ensure_future(asyncio.sleep(0.6))
        feeding = asyncio.ensure_future(speak_over())
        pre_roll = await asyncio.wait_for(host._watch_barge_in(ports, q, speak), WAIT_S)
        feeding.cancel()
    finally:
        host_mod._session.reset(token)
    assert pre_roll is None and ports.audio_out.stops == 0
