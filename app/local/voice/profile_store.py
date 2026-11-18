"""Thread-safe in-memory store for the mutable assistant profile.

The active profile starts from the loaded Profile at startup and can be patched
at runtime through the PATCH /v1/assistant/profile endpoint. Changes take effect
on the next turn -- no restart needed.

Usage::
    store = AssistantProfileStore(startup_profile)
    app.state.profile_store = store

    # In a request handler:
    current = app.state.profile_store.get()
    updated = app.state.profile_store.patch({"assistant": {"speed": 1.5}})
    original = app.state.profile_store.reset()
"""
from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger("app.local.voice.profile_store")

_SPEED_MIN = 0.5
_SPEED_MAX = 2.0


class ProfileValidationError(ValueError):
    """Raised when a profile field fails validation."""


@dataclass
class AssistantConfig:
    """Runtime-mutable assistant configuration.

    Stored separately from voxscribe-core's Profile because core's schema
    does not expose voice_id / speed / persona fields in this wheel version.
    """

    name: str = "vox"
    wake_phrase: str = "hey vox"
    end_phrases: tuple[str, ...] = ("goodbye", "stop listening")
    voice_id: str = ""
    speed: float = 1.0
    system_prompt: str = ""
    persona_label: str = "assistant"


class _ProfileView:
    """Thin wrapper that exposes profile.assistant for the router and adapters."""

    def __init__(self, assistant: AssistantConfig) -> None:
        self.assistant = assistant


class AssistantProfileStore:
    """Thread-safe in-memory store for the mutable assistant profile.

    Thread safety: a single RLock guards all mutations. Reads are copy-on-get
    so callers cannot modify the stored profile directly.
    """

    def __init__(self, startup=None) -> None:
        self._startup: AssistantConfig = AssistantConfig()
        self._current: AssistantConfig = copy.deepcopy(self._startup)
        self._lock = threading.RLock()

    def get(self) -> _ProfileView:
        """Return a view of the current profile (copy)."""
        with self._lock:
            return _ProfileView(copy.deepcopy(self._current))

    def patch(self, partial: dict) -> _ProfileView:
        """Merge partial dict over the current assistant config.

        Only keys present in partial are changed. Raises ValueError on
        out-of-range speed.
        """
        with self._lock:
            cfg = copy.deepcopy(self._current)
            ap = partial.get("assistant", {})
            if not isinstance(ap, dict):
                ap = {}

            if ap:
                old_phrase = cfg.wake_phrase
                if "name" in ap:
                    name = str(ap["name"]).strip()
                    if not name:
                        raise ProfileValidationError("name must not be empty")
                    cfg.name = name
                if "wake_phrase" in ap:
                    phrase = str(ap["wake_phrase"]).strip()
                    if len(phrase.split()) < 2:
                        raise ProfileValidationError(
                            "wake_phrase must contain at least two words"
                        )
                    cfg.wake_phrase = phrase
                if "end_phrases" in ap:
                    cfg.end_phrases = tuple(ap["end_phrases"])
                if "voice_id" in ap:
                    cfg.voice_id = str(ap["voice_id"])
                if "speed" in ap:
                    speed = float(ap["speed"])
                    if not (_SPEED_MIN <= speed <= _SPEED_MAX):
                        raise ProfileValidationError(
                            f"speed {speed} out of range [{_SPEED_MIN}, {_SPEED_MAX}]"
                        )
                    cfg.speed = speed
                if "system_prompt" in ap:
                    cfg.system_prompt = str(ap["system_prompt"])
                if "persona_label" in ap:
                    cfg.persona_label = str(ap["persona_label"])

                if cfg.wake_phrase != old_phrase:
                    log.info("wake: wake phrase updated from=%r to=%r",
                             old_phrase, cfg.wake_phrase)

                self._current = cfg

            return _ProfileView(copy.deepcopy(self._current))

    def reset(self) -> _ProfileView:
        """Revert to the startup profile. Returns a view of the startup profile."""
        with self._lock:
            self._current = copy.deepcopy(self._startup)
            log.info("profile_store: reverted to startup profile")
            return _ProfileView(copy.deepcopy(self._current))
