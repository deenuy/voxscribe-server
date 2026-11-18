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

log = logging.getLogger("app.local.voice.profile_store")


class AssistantProfileStore:
    """Thread-safe in-memory store for the mutable assistant profile.

    Thread safety: a single RLock guards all mutations. Reads are copy-on-get
    so callers cannot modify the stored profile directly.
    """

    def __init__(self, startup) -> None:
        from voxscribe.profile.schema import Profile

        if not isinstance(startup, Profile):
            raise TypeError(f"startup must be a Profile, got {type(startup)}")
        self._startup: "Profile" = copy.deepcopy(startup)
        self._current: "Profile" = copy.deepcopy(startup)
        self._lock = threading.RLock()

    def get(self):
        """Return a copy of the current profile."""
        with self._lock:
            return copy.deepcopy(self._current)

    def patch(self, partial: dict):
        """Merge partial dict over the current profile. Validates after merge.

        Only keys present in partial are changed; all other fields keep their
        current values. Raises ProfileValidationError on invalid combination.

        Returns a copy of the updated profile.
        """
        from voxscribe.profile.loader import ProfileValidationError, validate_profile
        from voxscribe.profile.schema import AssistantConfig, Profile

        with self._lock:
            current = copy.deepcopy(self._current)
            assistant_partial = partial.get("assistant", {})
            if not isinstance(assistant_partial, dict):
                assistant_partial = {}

            if assistant_partial:
                old_phrase = current.assistant.wake_phrase
                cfg = current.assistant

                updated_cfg = AssistantConfig(
                    name=assistant_partial.get("name", cfg.name),
                    wake_phrase=assistant_partial.get("wake_phrase", cfg.wake_phrase),
                    end_phrases=tuple(
                        assistant_partial.get("end_phrases", list(cfg.end_phrases))
                    ),
                    voice_id=assistant_partial.get("voice_id", cfg.voice_id),
                    speed=float(assistant_partial.get("speed", cfg.speed)),
                    system_prompt=assistant_partial.get("system_prompt", cfg.system_prompt),
                    persona_label=assistant_partial.get("persona_label", cfg.persona_label),
                )
                updated = Profile(assistant=updated_cfg)
                validate_profile(updated)

                new_phrase = updated_cfg.wake_phrase
                if new_phrase != old_phrase:
                    log.info(
                        "wake: wake phrase updated from=%r to=%r", old_phrase, new_phrase
                    )

                self._current = updated
                return copy.deepcopy(self._current)

            return copy.deepcopy(current)

    def reset(self):
        """Revert to the startup profile. Returns a copy of the startup profile."""
        with self._lock:
            self._current = copy.deepcopy(self._startup)
            log.info("profile_store: reverted to startup profile")
            return copy.deepcopy(self._current)
