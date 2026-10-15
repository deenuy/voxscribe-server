"""End-of-turn detector powered by the parked S4 speech activity stack.

Replaces the fixed 6-second silence counter in wake_loop._run_session with
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
"""

from __future__ import annotations

import logging
import struct

log = logging.getLogger("app.local.end_of_turn")

_SAMPLE_RATE  = 16_000
_CHUNK_FRAMES = 512
_BYTES_PER_FRAME = 2  # PCM16LE


class TurnDetector:
    """Drive HeuristicDetector → SpeechActivityTracker → SpeechTurnAdapter.

    Instantiate once per wake session; call reset() before each new session;
    push PCM16 frames with push_pcm16(); check the returned TurnDecision.
    """

    def __init__(self) -> None:
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
        self._adapter = SpeechTurnAdapter(silence_offset_ms=TURN.hangover_ms)
        self._now_ms: int = 0
        log.info("end_of_turn: TurnDetector initialized (heuristic detector, TURN policy)")

    def reset(self) -> None:
        """Start a new session. Call once per wake trigger."""
        self._detector.reset()
        self._tracker.reset()
        self._adapter.reset()
        self._now_ms = 0
        log.debug("end_of_turn: reset")

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
        """
        d = self._adapter.tick(now_ms)
        if d is not None:
            log.debug("end_of_turn: tick %s closes=%s silence_ms=%d",
                      d.action, d.closes_turn, d.silence_ms)
        return d

    def update_text(self, text: str) -> None:
        """Pass the current transcript text for completeness assessment."""
        self._adapter.update_text(text)
