"""Barge-in: the user takes the floor back while a reply plays (T13, VX-D2).

Three parts, all used by the voice host's reply stage:

  BargeInDetector   reads capture energy while the speaker plays. Capture is cleaned against
                    the playback reference (T12), so energy there is the user, not the reply.
                    BARGE_IN_MIN_FRAMES consecutive 40 ms frames above BARGE_IN_ENERGY_THRESHOLD
                    is an interruption; a shorter burst, or one that never gets that loud, is a
                    backchannel ("mm-hmm", "yeah") and the reply keeps the floor. After an
                    interruption the detector holds off for BARGE_IN_DEBOUNCE_MS.
  HeardReply        wraps the talker, the gate, the speaker and the sentence source for one
                    reply, and records which clauses were released for speech (full text) and
                    which finished playing (heard text). After an interruption the stored turn
                    and the next turn's history carry only the heard text.
  interruption_note the line added to the next turn's system prompt, so the reply model answers
                    the interjection and does not say again what was already heard.

No transcript text or audio is logged.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from voxscribe.session.ports import (
    GATE_PASS,
    GATE_REFUSE,
    GATE_REPLACE,
    GateDecision,
    ReplyOutcome,
)
from voxscribe.session.stages.reply import GATE_ERROR

log = logging.getLogger("app.local.voice.barge_in")

SAMPLE_RATE = 16_000
BARGE_IN_FRAME_MS = 40              # detector frame; capture chunks are re-framed to this
BARGE_IN_ENERGY_THRESHOLD = 0.03    # RMS above this = speech
BARGE_IN_MIN_FRAMES = 4             # consecutive frames required (4 * 40 ms = 160 ms)
BARGE_IN_DEBOUNCE_MS = 500          # cooldown after a barge-in fires
BACKCHANNEL_MAX_ENERGY = 0.015      # "mm-hmm" stays below this (don't treat as barge-in)
BACKCHANNEL_MAX_DURATION_MS = 600   # a voiced burst shorter than this that did not interrupt
#                                     is a backchannel; a longer one is reported as overlap
BARGE_IN_PRE_ROLL_MS = 480          # capture kept before the trigger, handed to the next turn
BARGE_IN_POLL_S = 0.05              # capture wait while watching, so the end of playback is seen

BARGE_IN = "barge_in"
BACKCHANNEL = "backchannel"
OVERLAP = "overlap"                 # long but quiet speech under the reply: logged, keeps floor

INTERRUPTION_NOTE = ('[The user interrupted. The assistant had said: "{played_text}". '
                     'Do not repeat it. Answer the interjection directly.]')


def interruption_note(played_text: str) -> str:
    """The system prompt append for the turn after a barge-in."""
    return INTERRUPTION_NOTE.format(played_text=played_text.replace('"', "'"))


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)))) if frame.size else 0.0


class BargeInDetector:
    """Energy-based overlap classifier over capture while the assistant plays.

    Time is stream time (frames fed), not a wall clock, so the debounce behaves the same in a
    replay and on the device. Feed every capture chunk with whether the speaker is playing.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._frame = sample_rate * BARGE_IN_FRAME_MS // 1000
        self._ms_per_sample = 1000 / sample_rate
        self.barge_ins = 0
        self.backchannels = 0
        self.overlaps = 0
        self.reset()

    def reset(self) -> None:
        """Forget the run in progress and the debounce. Counters are kept."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._t_ms = 0.0
        self._loud = 0           # consecutive frames above BARGE_IN_ENERGY_THRESHOLD
        self._voiced = 0         # frames in the current burst above BACKCHANNEL_MAX_ENERGY
        self._fired = False      # the current burst already interrupted (or was debounced)
        self._hold_until_ms = float("-inf")

    def feed(self, chunk, *, playing: bool) -> list[str]:
        """Events for this chunk: BARGE_IN, BACKCHANNEL or OVERLAP, usually none."""
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not playing:
            self._t_ms += arr.size * self._ms_per_sample
            self._buf = np.zeros(0, dtype=np.float32)
            self._loud = 0
            return self._end_burst()
        self._buf = np.concatenate([self._buf, arr])
        events: list[str] = []
        while self._buf.size >= self._frame:
            frame, self._buf = self._buf[: self._frame], self._buf[self._frame:]
            events.extend(self._frame_event(_rms(frame)))
        return events

    def _frame_event(self, rms: float) -> list[str]:
        self._t_ms += BARGE_IN_FRAME_MS
        events: list[str] = []
        if rms > BACKCHANNEL_MAX_ENERGY:
            self._voiced += 1
        else:
            events.extend(self._end_burst())
        self._loud = self._loud + 1 if rms > BARGE_IN_ENERGY_THRESHOLD else 0
        if self._loud >= BARGE_IN_MIN_FRAMES and not self._fired:
            self._fired = True
            if self._t_ms >= self._hold_until_ms:
                self._hold_until_ms = self._t_ms + BARGE_IN_DEBOUNCE_MS
                self.barge_ins += 1
                events.append(BARGE_IN)
        return events

    def _end_burst(self) -> list[str]:
        voiced, fired = self._voiced, self._fired
        self._voiced, self._fired = 0, False
        if voiced == 0 or fired:
            return []
        if voiced * BARGE_IN_FRAME_MS < BACKCHANNEL_MAX_DURATION_MS:
            self.backchannels += 1
            return [BACKCHANNEL]
        self.overlaps += 1
        return [OVERLAP]


@dataclass(frozen=True)
class Interruption:
    """What the store writes beside the reply when a barge-in cut it short."""

    played_text: str     # what the user heard: the clauses that finished playing
    full_text: str       # every clause the reply stage released for speech


class HeardReply:
    """One reply's record of what was released for speech and what finished playing.

    `scripted` are the stream's cue and lead-in: spoken first, through Talker.fixed, and not
    part of the reply text (the reply stage does not record them either).
    """

    def __init__(self, clock, *, scripted: Sequence[str | None] = ()) -> None:
        self._clock = clock
        self._scripted = [s for s in scripted if s]
        self._pending: tuple[object, str | None] | None = None
        self.released: list[str] = []
        self.played_clauses: list[str] = []
        self.verdict: GateDecision | None = None     # the last verdict that was not a pass
        self.started_ms: int = clock.mono_ms()
        self.first_sentence_ms: int | None = None
        self.first_audio_ms: int | None = None

    @property
    def played_text(self) -> str:
        return " ".join(self.played_clauses)

    @property
    def full_text(self) -> str:
        return " ".join(self.released)

    def interruption(self) -> Interruption:
        return Interruption(played_text=self.played_text, full_text=self.full_text)

    # ---------------------------------------------------------------- wrappers

    def talker(self, inner) -> _HeardTalker:
        return _HeardTalker(inner, self)

    def gate(self, inner) -> _HeardGate:
        return _HeardGate(inner, self)

    def audio_out(self, inner) -> _HeardAudioOut | None:
        return None if inner is None else _HeardAudioOut(inner, self)

    async def sentences(self, source: AsyncIterator[str]) -> AsyncIterator[str]:
        """The delegate's sentences, passed through. Closing this closes the source, which is
        what stops generation: the reply model's HTTP stream ends with it."""
        try:
            async for sentence in source:
                if self.first_sentence_ms is None:
                    self.first_sentence_ms = self._clock.mono_ms()
                yield sentence
        finally:
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                await aclose()

    # ---------------------------------------------------------------- outcome

    def outcome(self, completed: ReplyOutcome | None, *, end_ms: int) -> ReplyOutcome:
        """The outcome recorded for an interrupted reply: the heard text, with the gate verdict
        the reply stage reached (or would have recorded) and the times seen so far."""
        if completed is not None:
            return replace(completed, text=self.played_text,
                           sentences=list(self.played_clauses), playback_end_ms=end_ms)
        decision, category = GATE_PASS, "ok"
        v = self.verdict
        if v is not None:
            decision = GATE_REPLACE if v.decision == GATE_REPLACE and v.replacement else GATE_REFUSE
            category = v.category
        return ReplyOutcome(decision=decision, category=category, text=self.played_text,
                            sentences=list(self.played_clauses), started_ms=self.started_ms,
                            first_sentence_ms=self.first_sentence_ms,
                            first_audio_ms=self.first_audio_ms, playback_end_ms=end_ms)

    # ---------------------------------------------------------------- bookkeeping

    def _release(self, clause: str | None) -> None:
        # At hand-off to the talker, so a clause cut off during synthesis is in the full text.
        if clause is not None:
            self.released.append(clause)

    def _synthesised(self, pcm, clause: str | None) -> None:
        self._pending = (pcm, clause)

    def _claim(self, pcm) -> str | None:
        """The clause this audio speaks, if it is the audio the talker just produced."""
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] is pcm:
            return pending[1]
        return None

    def _is_scripted(self, phrase: str) -> bool:
        if self._scripted and phrase == self._scripted[0]:
            self._scripted.pop(0)
            return True
        return False

    def _played(self, clause: str) -> None:
        self.played_clauses.append(clause)


class _HeardTalker:
    def __init__(self, inner, heard: HeardReply) -> None:
        self._inner, self._heard = inner, heard

    @property
    def sample_rate(self) -> int:
        return self._inner.sample_rate

    async def synthesize(self, text: str):
        self._heard._release(text)
        pcm = await self._inner.synthesize(text)
        self._heard._synthesised(pcm, text)
        return pcm

    async def fixed(self, phrase: str):
        clause = None if self._heard._is_scripted(phrase) else phrase
        self._heard._release(clause)
        pcm = await self._inner.fixed(phrase)
        self._heard._synthesised(pcm, clause)
        return pcm

    async def warm(self) -> None:
        await self._inner.warm()


class _HeardGate:
    def __init__(self, inner, heard: HeardReply) -> None:
        self._inner, self._heard = inner, heard

    @property
    def fail_closed_phrase(self) -> str:
        return self._inner.fail_closed_phrase

    def needs_session_tools(self, sentence: str) -> bool:
        return self._inner.needs_session_tools(sentence)

    def check(self, sentence: str, session_tools: Sequence[str] = (), *,
              user_text: str = "") -> GateDecision:
        try:
            verdict = self._inner.check(sentence, session_tools, user_text=user_text)
        except Exception:
            # The reply stage turns this into a refusal; record it the same way.
            self._heard.verdict = GateDecision(GATE_REFUSE, GATE_ERROR,
                                               self._inner.fail_closed_phrase)
            raise
        if verdict.decision != GATE_PASS:
            self._heard.verdict = verdict
        return verdict


class _HeardAudioOut:
    """The speaker, with each reply clause marked heard once its audio has played to the end.

    With an adapter that reports playback (`reports_played`), that is when the engine has
    written the last sample; otherwise it is when play() returns.
    """

    def __init__(self, inner, heard: HeardReply) -> None:
        self._inner, self._heard = inner, heard

    @property
    def available(self) -> bool:
        return getattr(self._inner, "available", True)

    async def play(self, pcm_float32_mono, sample_rate: int) -> None:
        clause = self._heard._claim(pcm_float32_mono)
        if clause is None:
            await self._inner.play(pcm_float32_mono, sample_rate)
            return
        done: Callable[[], None] = lambda: self._heard._played(clause)  # noqa: E731
        if getattr(self._inner, "reports_played", False):
            await self._inner.play(pcm_float32_mono, sample_rate, on_played=done)
        else:
            await self._inner.play(pcm_float32_mono, sample_rate)
            done()
        if self._heard.first_audio_ms is None:
            self._heard.first_audio_ms = self._heard._clock.mono_ms()

    async def play_file(self, path: str) -> None:
        await self._inner.play_file(path)

    def stop_playback(self) -> None:
        self._inner.stop_playback()

    @property
    def is_playing(self) -> bool:
        return self._inner.is_playing

    def playback_ref(self) -> np.ndarray:
        return self._inner.playback_ref()
