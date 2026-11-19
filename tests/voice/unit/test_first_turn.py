"""The first turn closes on the person's pause (PL59).

In the owner's sessions of 2026-11-19 every first turn ended at the first-turn wait, eight seconds
after the wake, while later turns closed on a pause. These tests drive the host's real turn stage
(`_listen`) with the real turn detector over synthetic audio shaped like a first turn: the end of
the wake phrase in the pre-roll, the acknowledgement cue muted from capture, a short question,
then the room. Time on the host's wall clock advances 32 ms per chunk of live capture, as it does
with a microphone, so the first-turn wait is measured as it is in a session.

Two causes, both shown here:
- the turn detector settled on digital silence before every turn, so it measured the room against
  a floor near -120 dB and took the room after the person stopped for more speech: the pause that
  closes the turn started a second or more late;
- the first-turn wait was a deadline from the start of listening that words did not move, so a
  person still talking, or a close that came late, was cut off by it.
"""
from __future__ import annotations

import asyncio
import types
import wave

import numpy as np
import pytest

import app.local.voice.host as host_mod
from app.local.voice.adapters.audio_portaudio import (
    CHIME_DECAY_MS,
    CHIME_DURATION_MS,
    PortAudioOut,
    cue_duration_ms,
)
from app.local.voice.adapters.speech_heuristic import HeuristicSpeech
from app.local.voice.host import (
    CHUNK_FRAMES,
    FIRST_TURN_WAIT_S,
    ROOM_TONE_CHUNKS,
    SAMPLE_RATE,
    PreRollQueue,
    RoomTone,
    SessionContext,
    TurnHearing,
    VoiceConfig,
    VoiceHost,
    _TurnRing,
)
from tests.voice.unit.fakes import fake_ports

PAUSE_S = 0.9                       # the owner's pace (VOXSCRIBE_END_OF_TURN_PAUSE_S)
CHUNK_S = CHUNK_FRAMES / SAMPLE_RATE
PAUSE_BAND_S = PAUSE_S + 0.6        # the pause, the detector's hangover and a tick of margin
WORDS_EVERY_CHUNKS = 16             # the recognizer reports words about twice a second


def room(seconds: float, kind: str, seed: int, level: float = 0.002) -> np.ndarray:
    """Room noise: steady, with the falling spectrum of a real room (pink or brown), not white."""
    n = int(round(seconds * SAMPLE_RATE))
    x = np.random.default_rng(seed).normal(0, 1, max(n, 2))
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    freqs[0] = freqs[1]
    tilt = {"pink": 0.5, "brown": 1.0}[kind]
    y = np.fft.irfft(np.fft.rfft(x) / freqs ** tilt, len(x))
    return (level * y / np.std(y)).astype(np.float32)[:n]


def talk(seconds: float, seed: int, level: float = 0.08) -> np.ndarray:
    """Speech-like sound: a voiced harmonic stack in syllables, four or five a second, with the
    dips between syllables and the short gaps between words that speech has."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    pitch = 120 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 15))
    env = np.zeros(n)
    at = 0.0
    while at < seconds:
        dur = rng.uniform(0.12, 0.26)                  # one syllable
        i, j = int(at * SAMPLE_RATE), min(n, int((at + dur) * SAMPLE_RATE))
        env[i:j] = np.sin(np.linspace(0, np.pi, j - i)) * rng.uniform(0.5, 1.0)
        at += dur + (rng.uniform(0.08, 0.2) if rng.random() < 0.3 else rng.uniform(0.01, 0.04))
    y = x * (0.03 + env)
    return (level * y / np.max(np.abs(y))).astype(np.float32)


class FakeTime:
    """The host's wall clock: advances only as live capture arrives."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class FirstTurnCapture:
    """Live capture after the wake: (audio, muted) segments, one 32 ms chunk per read. While a
    segment marked `speech` plays, the recognizer reports words; at its end the turn's text is
    handed to the detector, as the recognizer task does."""

    def __init__(self, clock: FakeTime, speech: HeuristicSpeech, hearing: TurnHearing,
                 segments: list[tuple[np.ndarray, str]], text: str) -> None:
        self.clock, self.speech, self.hearing, self.text = clock, speech, hearing, text
        self.items: list[tuple[np.ndarray, str]] = []
        for audio, kind in segments:
            for i in range(0, len(audio) - CHUNK_FRAMES + 1, CHUNK_FRAMES):
                self.items.append((audio[i:i + CHUNK_FRAMES], kind))
        last = max(i for i, (_a, k) in enumerate(self.items) if k == "speech")
        self.speech_end_s = (last + 1) * CHUNK_S         # wall time from the start of listening
        self.reads = 0
        self.muted = False

    async def get(self):
        await asyncio.sleep(0)
        if self.reads >= len(self.items):
            await asyncio.sleep(3600)
        audio, kind = self.items[self.reads]
        self.reads += 1
        self.clock.now += CHUNK_S
        self.muted = kind == "muted"
        if kind == "speech" and self.reads % WORDS_EVERY_CHUNKS == 0:
            self.hearing.heard()
        if kind == "speech" and self.reads * CHUNK_S >= self.speech_end_s:
            self.hearing.heard()
            self.speech.update_text(self.text)
        return audio

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def empty(self) -> bool:
        return True


async def first_turn(monkeypatch, *, kind: str, question_s: float, seed: int = 7,
                     settle_on_room: bool = True, text: str = "How are you? Can you hear me?"):
    """Run turn 1 through the real `_listen`. Returns (timed_out, wall seconds from the start of
    listening to the end of the turn, wall seconds from the end of the question to it)."""
    clock = FakeTime()
    monkeypatch.setattr(host_mod, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    speech = HeuristicSpeech()
    ports = fake_ports(speech=speech)
    host = VoiceHost(VoiceConfig(end_of_turn_pause_s=PAUSE_S), ports=ports)
    if settle_on_room:
        # What the armed listener heard before the wake: the room.
        before = room(4.0, kind, seed + 1)
        for i in range(0, len(before) - CHUNK_FRAMES + 1, CHUNK_FRAMES):
            host._room.offer(before[i:i + CHUNK_FRAMES])
    hearing = TurnHearing()
    vox = talk(0.3, seed=2)                          # the end of "hey vox"
    pre_roll = room(0.6, kind, seed + 2)
    pre_roll[:len(vox)] += vox
    question = talk(question_s, seed=3) + room(question_s, kind, seed + 4)
    live = FirstTurnCapture(clock, speech, hearing, [
        (room(1.95, kind, seed + 3), "muted"),         # the cue and its decay, muted from capture
        (room(0.5, kind, seed + 5), "room"),           # the person begins
        (question, "speech"),
        (room(FIRST_TURN_WAIT_S + 10, kind, seed + 6), "room"),
    ], text)
    ports.audio_in.capture_muted = lambda: live.muted
    speech.reset()                                     # as the session does before turn 1
    started = clock.now
    timed_out = await asyncio.wait_for(
        host._listen(ports, PreRollQueue(pre_roll, live), _TurnRing(4000), 1, True,
                     SessionContext(), hearing), timeout=30)
    ended = clock.now - started
    return timed_out, ended, ended - live.speech_end_s


@pytest.mark.parametrize("kind", ["pink", "brown"])
async def test_a_short_first_question_closes_within_the_pause_band(monkeypatch, kind):
    timed_out, _ended, after = await first_turn(monkeypatch, kind=kind, question_s=2.0)
    assert timed_out is False
    assert 0 < after <= PAUSE_BAND_S


@pytest.mark.parametrize("kind", ["pink", "brown"])
async def test_settled_on_silence_the_same_turn_closes_late(monkeypatch, kind):
    """The cause, shown: the same turn with the detector settled on digital silence (what every
    turn got before PL59) closes well after the pause band, because the room after the question
    still reads as speech."""
    _t, _e, on_room = await first_turn(monkeypatch, kind=kind, question_s=2.0)
    timed_out, _ended, on_silence = await first_turn(monkeypatch, kind=kind, question_s=2.0,
                                                     settle_on_room=False)
    assert timed_out is False
    assert on_silence > PAUSE_BAND_S and on_silence > on_room + 0.6


async def test_a_first_question_longer_than_the_first_turn_wait_is_not_cut_off(monkeypatch):
    """The owner's log line, reproduced before the fix: a person still speaking when the first-turn
    wait ran out was cut off by it ("turn 1 closed at the first-turn wait with words heard")."""
    timed_out, ended, after = await first_turn(monkeypatch, kind="pink", question_s=7.5)
    assert ended > FIRST_TURN_WAIT_S
    assert timed_out is False
    assert 0 < after <= PAUSE_BAND_S


@pytest.mark.parametrize("settle_on_room", [True, False])
async def test_the_first_turn_wait_still_ends_a_turn_in_which_nothing_was_said(monkeypatch,
                                                                               settle_on_room):
    """Settled on the room, a turn of nothing but the room waits for the person and ends at the
    first-turn wait. Settled on silence, the room itself opened a turn and closed it, with
    nothing said."""
    clock = FakeTime()
    monkeypatch.setattr(host_mod, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    speech = HeuristicSpeech()
    ports = fake_ports(speech=speech)
    host = VoiceHost(VoiceConfig(end_of_turn_pause_s=PAUSE_S), ports=ports)
    hearing = TurnHearing()
    quiet = room(FIRST_TURN_WAIT_S + 10, "pink", 11)
    if settle_on_room:
        before = room(4.0, "pink", 12)
        for i in range(0, len(before) - CHUNK_FRAMES + 1, CHUNK_FRAMES):
            host._room.offer(before[i:i + CHUNK_FRAMES])

    class Room:
        def __init__(self):
            self.i = 0

        async def get(self):
            await asyncio.sleep(0)
            chunk = quiet[self.i * CHUNK_FRAMES:(self.i + 1) * CHUNK_FRAMES]
            self.i += 1
            clock.now += CHUNK_S
            return chunk

        def get_nowait(self):
            raise asyncio.QueueEmpty

        def empty(self):
            return True

    speech.reset()
    started = clock.now
    timed_out = await asyncio.wait_for(host._listen(ports, Room(), _TurnRing(4000), 1, True,
                                                    SessionContext(), hearing), timeout=30)
    if not settle_on_room:
        assert timed_out is False and clock.now - started < FIRST_TURN_WAIT_S
        return
    assert timed_out is True and not hearing.words
    assert FIRST_TURN_WAIT_S <= clock.now - started < FIRST_TURN_WAIT_S + 0.6


def test_the_room_tone_is_the_quietest_recent_capture_in_the_order_heard():
    tone = RoomTone(history=ROOM_TONE_CHUNKS * 3)
    assert tone.sample() is None
    loud = np.full(CHUNK_FRAMES, 0.3, np.float32)
    for i in range(ROOM_TONE_CHUNKS * 3):
        tone.offer(loud if i % 3 else np.full(CHUNK_FRAMES, i / 1e5, np.float32))
    sample = tone.sample()
    assert sample.shape == (ROOM_TONE_CHUNKS * CHUNK_FRAMES,)
    firsts = sample[::CHUNK_FRAMES]
    assert np.all(firsts < 0.01) and np.all(np.diff(firsts) > 0)
    tone.offer(np.zeros(10, np.float32))                # a partial chunk is not a room sample
    assert tone.sample().shape == (ROOM_TONE_CHUNKS * CHUNK_FRAMES,)


def _wav(path, seconds: float, rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))


def _aiff(path, seconds: float, rate: int = 44100) -> None:
    """A minimal AIFF: a COMM chunk with the sample rate as an 80-bit extended float."""
    import struct

    frames = int(seconds * rate)
    exponent, mantissa = 16383 + rate.bit_length() - 1, rate << (64 - rate.bit_length())
    comm = struct.pack(">hIh", 1, frames, 16) + struct.pack(">H", exponent) + \
        mantissa.to_bytes(8, "big")
    ssnd = struct.pack(">II", 0, 0) + b"\0\0" * frames
    body = b"AIFF" + b"COMM" + struct.pack(">I", len(comm)) + comm + \
        b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    path.write_bytes(b"FORM" + struct.pack(">I", len(body)) + body)


def test_a_cue_is_measured_from_its_file(tmp_path):
    cue_duration_ms.cache_clear()
    _wav(tmp_path / "a.wav", 1.65)
    _aiff(tmp_path / "b.aiff", 1.627)
    (tmp_path / "c.aiff").write_bytes(b"not a sound")
    assert cue_duration_ms(str(tmp_path / "a.wav")) == 1650
    assert cue_duration_ms(str(tmp_path / "b.aiff")) == 1627
    assert cue_duration_ms(str(tmp_path / "c.aiff")) == CHIME_DURATION_MS
    assert cue_duration_ms(str(tmp_path / "missing.aiff")) == CHIME_DURATION_MS


async def test_capture_is_muted_for_the_whole_cue_and_its_decay(tmp_path, monkeypatch):
    """The acknowledgement cue (1.65 s) used to be muted for 1.1 s: its last half second reached
    the turn detector as speech at the start of every first turn."""
    cue_duration_ms.cache_clear()
    cue = tmp_path / "ack.aiff"
    _aiff(cue, 1.65)
    muted: list[int] = []
    engine = types.SimpleNamespace(
        echo_guard=types.SimpleNamespace(notify_chime=muted.append))
    monkeypatch.setattr("app.local.voice.adapters.audio_portaudio.subprocess.run",
                        lambda *a, **k: None)
    await PortAudioOut(engine).play_file(str(cue))
    assert muted == [1650 + CHIME_DECAY_MS]
