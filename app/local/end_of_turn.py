"""End-of-turn detector powered by the parked S4 speech activity stack.

Replaces the fixed 6-second silence counter in the voice host's turn stage with
turn-taking policy decisions from SpeechActivityTracker + SpeechTurnAdapter
(voxscribe-core voice/p1-loop).

Detector choice: HeuristicDetector (model-free, only numpy).
Policy: TURN preset (hangover 300 ms, starts on 96 ms of speech).

API:
    td = TurnDetector()
    td.reset()
    decision = td.push_pcm16(frame_bytes, now_ms)  # frame_bytes = bytes, source_tag stripped
    if decision and decision.closes_turn:
        # end the session

frame_bytes must be raw PCM16LE mono at 16 kHz (512 samples = 1024 bytes).
source_tag byte (from AudioRing) must be stripped before calling push_pcm16.

Clock contract: now_ms is stream time, milliseconds of audio fed since reset(),
the timeline the detector stamps its speech events on. Engine uptime is a
different clock: a pause measured against it grows with uptime and closes every
turn at the first tick. tick() raises ClockDomainError when now_ms strays from
the audio actually fed by more than _CLOCK_SKEW_LIMIT_MS.
"""

from __future__ import annotations

import logging

# The engine's clock contract error, re-exported so callers that caught it here keep working and
# a host catches one type for every Speech adapter.
from voxscribe.session.ports import ClockDomainError

log = logging.getLogger("app.local.end_of_turn")

_SAMPLE_RATE  = 16_000
_CHUNK_FRAMES = 512
_SETTLE_FRAMES = 25             # 0.8 s of silence through a reset detector before a turn's audio
_BYTES_PER_FRAME = 2  # PCM16LE
# A correct caller is within one frame of the audio it fed; a minute apart means another clock.
_CLOCK_SKEW_LIMIT_MS = 60_000

__all__ = ["ClockDomainError", "TurnDetector"]


# How long a person may pause before their turn is taken to be over.
#
# The kernel's own bands are for dictation and for speakers who think aloud: a finished sentence
# closes the turn only after six seconds of silence, and an unfinished one holds for thirty. That is
# right for a companion that must never cut an elder off. It is wrong for a conversation with an
# assistant: nobody waits six seconds for an answer, they ask again, the silence never comes, and
# the turn never closes (the owner's first test, 2026-11-19). So the pace is the product's choice:
# a pause at or above PATIENT_PAUSE_S keeps the kernel's bands; a shorter one scales them, and a
# sentence that still looks unfinished gets UNFINISHED_EXTRA_MS more before it is taken anyway.
PATIENT_PAUSE_S = 6.0
UNFINISHED_EXTRA_MS = 1600.0
_MIN_PAUSE_MS = 300.0


def turn_pacing(pause_s: float | None):
    """(PauseBandConfig, TurnTakingConfig) for an end-of-turn pause of `pause_s` seconds."""
    from voxscribe.silence_policy import PauseBandConfig
    from voxscribe.turn_taking import TurnTakingConfig

    if pause_s is None or pause_s >= PATIENT_PAUSE_S:
        return PauseBandConfig(), TurnTakingConfig()
    extended = max(_MIN_PAUSE_MS, float(pause_s) * 1000.0)
    bands = PauseBandConfig(short_ms=min(400.0, extended * 0.3), long_ms=extended * 0.6,
                            extended_ms=extended)
    return bands, TurnTakingConfig(hold_cap_ms=extended + UNFINISHED_EXTRA_MS)


class TurnDetector:
    """Drive HeuristicDetector → SpeechActivityTracker → SpeechTurnAdapter.

    Instantiate once per wake session; call reset() before each new session;
    push PCM16 frames with push_pcm16(); check the returned TurnDecision.
    """

    def __init__(self, pause_s: float | None = None) -> None:
        try:
            import numpy as np
            from voxscribe.speech.heuristic import HeuristicDetector
            from voxscribe.speech.policy import TURN, SpeechActivityTracker
            from voxscribe.speech.turns import SpeechTurnAdapter
        except ImportError as exc:
            raise RuntimeError(
                "TurnDetector needs voxscribe-core on voice/p1-loop (numpy, speech stack)"
            ) from exc

        self._np = np
        self._detector = HeuristicDetector()
        self._tracker = SpeechActivityTracker(TURN)
        bands, turns = turn_pacing(pause_s)
        self._adapter = SpeechTurnAdapter(pause_config=bands, turn_config=turns,
                                          silence_offset_ms=TURN.hangover_ms)
        self.pause_s = pause_s
        self._now_ms: int = 0
        self._samples_fed: int = 0
        self._settle_ms: int = 0
        log.info("end_of_turn: TurnDetector initialized (heuristic detector, TURN policy, "
                 "closes a finished turn after %d ms of silence, an unfinished one after %d ms)",
                 int(bands.extended_ms), int(turns.hold_cap_ms))

    @property
    def stream_ms(self) -> int:
        """Milliseconds of audio fed since reset(): the clock tick() expects."""
        return self._samples_fed * 1000 // _SAMPLE_RATE

    def reset(self, room=None) -> None:
        """Start a new turn. `room` is the room as it sounds when nobody speaks (float32 mono
        at 16 kHz, any length): the detector settles on it. Without it the detector settles on
        digital silence, which no microphone hears (see _settle)."""
        self._detector.reset()
        self._tracker.reset()
        self._adapter.reset()
        self._now_ms = 0
        self._samples_fed = 0
        self._settle(room)
        log.debug("end_of_turn: reset")

    def _settle(self, room=None) -> None:
        """Let the detector find its footing before the turn's audio arrives.

        A freshly reset detector does not report speech that begins in its first half second or
        so. A turn opened by a wake phrase begins with speech at once: the words said right after
        the phrase ("hey vox, how are you") are the first audio the turn is given. Without this,
        that turn was never seen to open, so it never closed and the person was never answered
        (measured 2026-11-19: no start reported for speech in the first 0.3 s after a reset).
        The settling audio is silence and belongs to no turn; the time it takes is kept so the
        host's stream clock and the detector's own timeline still agree in tick().

        What the detector settles on is the room it then measures speech against: its floor is
        learned in the warm-up and afterwards rises at only 1.5 dB a second. Settled on digital
        silence, the floor sits near -120 dB for the whole turn, every sound a microphone picks
        up reads as far louder than the room, and only voicing and modulation are left to tell
        speech from the room. Room noise that is not white then counts as speech for a second
        and more after the person stops, so the pause that closes the turn starts late (the
        first-turn defect of 2026-11-19, PL59). Settled on the room's own sound, the floor is
        the room's and the pause starts when the person stops. With no room to hand the
        settling audio is silence, as before."""
        n = _SETTLE_FRAMES * _CHUNK_FRAMES
        pcm = None if room is None else self._np.asarray(room, dtype="float32").reshape(-1)
        if pcm is None or pcm.size < _CHUNK_FRAMES:
            pcm = self._np.zeros(n, dtype="float32")
        elif pcm.size < n:
            pcm = self._np.resize(pcm, n)
        self._detector.process(pcm[-n:])
        self._settle_ms = _SETTLE_FRAMES * _CHUNK_FRAMES * 1000 // _SAMPLE_RATE

    def push_pcm16(self, frame_bytes: bytes, now_ms: int) -> object | None:
        """Feed one PCM16LE frame (1024 bytes / 512 samples) and advance time.

        Returns a TurnDecision when the adapter produces one, else None.
        A TurnDecision with .closes_turn == True means the session should end.
        The host should also call tick(now_ms) every few frames to poll silence.
        """
        np = self._np
        expected = _CHUNK_FRAMES * _BYTES_PER_FRAME
        if len(frame_bytes) != expected:
            return None  # ignore partial frames

        samples_i16 = np.frombuffer(frame_bytes, dtype="<i2")
        samples_f32 = samples_i16.astype("float32") / 32768.0

        self._now_ms = now_ms
        self._samples_fed += len(samples_f32)
        frames = self._detector.process(samples_f32)
        decision = None
        for frame in frames:
            event = self._tracker.process(frame)
            if event is not None:
                d = self._adapter.observe(event)
                if d is not None:
                    decision = d
                    log.debug("end_of_turn: %s closes=%s silence_ms=%d",
                              d.action, d.closes_turn, d.silence_ms)
        return decision

    def tick(self, now_ms: int) -> object | None:
        """Poll the adapter for a silence decision without a new frame.

        Call this every ~100 ms during the session to catch long pauses that
        arrive between frames. Returns TurnDecision or None.
        now_ms must be stream time (see the module docstring); raises
        ClockDomainError when it is not.
        """
        skew_ms = int(now_ms) - self.stream_ms
        if abs(skew_ms) > _CLOCK_SKEW_LIMIT_MS:
            raise ClockDomainError(
                f"tick(now_ms={int(now_ms)}) is {skew_ms} ms from the {self.stream_ms} ms of audio "
                "fed since reset(); pass stream time, not engine uptime"
            )
        d = self._adapter.tick(int(now_ms) + self._settle_ms)
        if d is not None:
            log.debug("end_of_turn: tick %s closes=%s silence_ms=%d",
                      d.action, d.closes_turn, d.silence_ms)
        return d

    def update_text(self, text: str) -> None:
        """Pass the current transcript text for completeness assessment."""
        self._adapter.update_text(text)
