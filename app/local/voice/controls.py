"""Spoken control intents: rule-based phrase detection before the LLM call (T15).

Each phrase is matched case-insensitively after punctuation removal. First match wins.
No ML; no LLM; no I/O.

Named constants (SPEED_STEP, VOLUME_STEP, SPEED_MIN, SPEED_MAX, VOLUME_MIN, VOLUME_MAX)
are the single source of truth; never hardcode these values elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Named constants for step sizes and limits.
SPEED_STEP: float = 0.15        # factor change per spoken request (1 +/- SPEED_STEP)
VOLUME_STEP: float = 0.20       # factor change per spoken request (1 +/- VOLUME_STEP)
SPEED_MIN: float = 0.5
SPEED_MAX: float = 2.0
VOLUME_MIN: float = 0.3
VOLUME_MAX: float = 2.0

SPEED_INITIAL: float = 1.0
VOLUME_INITIAL: float = 1.0

# Acknowledgement cues spoken instead of an LLM reply on a control intent.
ACK_CUE: str = "Got it."
ALT_CUE: str = "Sure."


class ControlIntent(Enum):
    HOLD = "hold"
    SLOWER = "slower"
    FASTER = "faster"
    SHORTER = "shorter"
    LONGER = "longer"
    REPEAT = "repeat"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    NONE = "none"


# Trigger phrases in match order within each bucket.
# Longer / more specific phrases must come before shorter ones to avoid false substring matches.
_TRIGGERS: list[tuple[ControlIntent, list[str]]] = [
    (ControlIntent.HOLD, [
        "hold on", "one moment", "give me a second", "wait a moment", "just wait",
    ]),
    (ControlIntent.SLOWER, [
        "slow down", "speak slower", "too fast",
    ]),
    (ControlIntent.FASTER, [
        "speed up", "speak faster", "too slow",
    ]),
    (ControlIntent.SHORTER, [
        "be more concise", "shorter answers", "keep it brief",
    ]),
    (ControlIntent.LONGER, [
        "more detail", "explain more", "go deeper",
    ]),
    (ControlIntent.REPEAT, [
        "say that again", "repeat that", "what did you say",
    ]),
    (ControlIntent.VOLUME_UP, [
        "louder", "turn it up", "speak up",
    ]),
    (ControlIntent.VOLUME_DOWN, [
        "quieter", "turn it down", "softer",
    ]),
]


def _normalise(text: str) -> str:
    """Lowercase, collapse punctuation to spaces, strip."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def detect_control(text: str) -> ControlIntent:
    """Return the matching ControlIntent, or NONE if no phrase matches."""
    norm = _normalise(text)
    for intent, phrases in _TRIGGERS:
        for phrase in phrases:
            if phrase in norm:
                return intent
    return ControlIntent.NONE


@dataclass
class ControlState:
    """Mutable session state for pacing, length and level controls.

    One instance per conversation; created at session start in run_session() and passed
    to _turn() and reply_turn() so the system-prompt suffix and amplitude scale travel
    with each reply.

    Speed and volume are clamped to [SPEED_MIN, SPEED_MAX] and [VOLUME_MIN, VOLUME_MAX].
    If the profile imposes a tighter range pass floor= / ceil= to apply_slower / apply_faster.
    """

    speed: float = SPEED_INITIAL
    volume: float = VOLUME_INITIAL
    prompt_suffix: str = ""

    def apply_slower(self, *, floor: float = SPEED_MIN, ceil: float = SPEED_MAX) -> None:
        """Lower TTS speed by SPEED_STEP, clamped to [floor, ceil]."""
        new = self.speed * (1.0 - SPEED_STEP)
        self.speed = max(floor, min(ceil, new))

    def apply_faster(self, *, floor: float = SPEED_MIN, ceil: float = SPEED_MAX) -> None:
        """Raise TTS speed by SPEED_STEP, clamped to [floor, ceil]."""
        new = self.speed * (1.0 + SPEED_STEP)
        self.speed = max(floor, min(ceil, new))

    def apply_volume_up(self) -> None:
        """Raise TTS output amplitude by VOLUME_STEP, clamped to VOLUME_MAX."""
        self.volume = min(VOLUME_MAX, self.volume * (1.0 + VOLUME_STEP))

    def apply_volume_down(self) -> None:
        """Lower TTS output amplitude by VOLUME_STEP, clamped to VOLUME_MIN."""
        self.volume = max(VOLUME_MIN, self.volume * (1.0 - VOLUME_STEP))

    def apply_shorter(self) -> None:
        """Append the brevity instruction to the system-prompt suffix."""
        self.prompt_suffix = "Reply in 2 sentences or fewer."

    def apply_longer(self) -> None:
        """Append the detail instruction to the system-prompt suffix."""
        self.prompt_suffix = "Provide a detailed explanation."
