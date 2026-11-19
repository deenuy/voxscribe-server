"""Per-tenant, persistent assistant profile store.

Defaults come from the loaded product profile, not from a bare AssistantConfig().
When an app has a database the actual row lives in each tenant's schema (migration 022);
this module holds the in-memory cache. The scope adapter (voice/adapters/scope.py) is the
only part of the voice code that reads and writes the database rows.

Sync API (get / patch / reset) is preserved for unit tests that run without a database.
The tenant argument selects whose profile is in scope; None means the single shared profile
used when isolation is off.

Limits enforced here (apply_product_clamp), on every patch and wherever a stored row becomes a
config (scope._row_to_config), so a row written under another product mode, or by hand, cannot
loosen one:
  - In senior_care or classroom mode, voice_id must be a known local voice or empty.
    A voice_id that is not in the known local set is considered a cloud voice and is
    silently clamped to an empty string.
  - In senior_care or classroom mode, system_prompt is clamped to an empty string: the
    product's prompt is the only one, and a tenant's text never joins it.
  - In every mode, speed is brought inside [_SPEED_MIN, _SPEED_MAX] (a patch outside the range
    is refused before it gets here; a stored value is clamped).
No log line names or wake phrases together with a tenant key.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import math
import threading
from dataclasses import dataclass, field

log = logging.getLogger("app.local.voice.profile_store")

_SPEED_MIN = 0.5
_SPEED_MAX = 2.0

# Every voice ID shipped with the kokoro-onnx bundle.  Any voice_id not in this
# set is treated as a cloud voice and is disallowed in restricted product modes.
_LOCAL_VOICE_IDS: frozenset[str] = frozenset({
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_fable", "am_liam", "am_michael", "am_onyx", "am_puck",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
})

# Product modes that must not reach a cloud voice endpoint.
_RESTRICTED_MODES: frozenset[str] = frozenset({"senior_care", "classroom"})


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


def _defaults_from_product_profile(product_profile) -> AssistantConfig:
    """Extract assistant config defaults from the loaded product profile.

    Falls back to AssistantConfig() defaults for fields the product profile does not cover.
    """
    cfg = AssistantConfig()
    if product_profile is None:
        return cfg
    # Wake phrase: use the profile's wake phrase when it differs from the hardcoded default.
    wake = getattr(product_profile, "wake", None)
    if wake is not None:
        phrase = getattr(wake, "phrase", None)
        if phrase and isinstance(phrase, str):
            cfg = copy.copy(cfg)
            cfg.wake_phrase = phrase
    return cfg


def _speed_in_range(speed: float) -> float:
    if not math.isfinite(speed):
        return AssistantConfig.speed
    return min(max(speed, _SPEED_MIN), _SPEED_MAX)


def apply_product_clamp(config: AssistantConfig, product_profile) -> AssistantConfig:
    """Enforce the limits a tenant's stored or patched values cannot override.

    In senior_care or classroom mode a voice_id that is not a known local Kokoro voice is
    silently clamped to an empty string (no cloud voices), and so is any system_prompt (no
    tenant text in the reply prompt). In every mode the speed is brought inside the range.
    """
    mode = getattr(product_profile, "mode", "") or ""
    restricted = mode in _RESTRICTED_MODES
    voice_id, system_prompt = config.voice_id, config.system_prompt
    if restricted and voice_id not in _LOCAL_VOICE_IDS:
        voice_id = ""
    if restricted:
        system_prompt = ""
    speed = _speed_in_range(config.speed)
    if (voice_id, system_prompt, speed) == (config.voice_id, config.system_prompt, config.speed):
        return config
    return dataclasses.replace(config, voice_id=voice_id, system_prompt=system_prompt, speed=speed)


def _apply_patch(base: AssistantConfig, partial: dict) -> AssistantConfig:
    """Return a new AssistantConfig with the fields from partial merged in.

    Raises ProfileValidationError for out-of-range or structurally invalid values.
    """
    cfg = copy.deepcopy(base)
    ap = partial.get("assistant", {})
    if not isinstance(ap, dict):
        ap = {}

    if not ap:
        return cfg

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
        log.info("wake: wake phrase updated (tenant omitted)")

    return cfg


class AssistantProfileStore:
    """Per-tenant in-memory cache for the mutable assistant profile.

    Thread safety: a single RLock guards all mutations. Reads return copies.

    Defaults come from the loaded product profile (passed at construction).  A
    bare AssistantConfig() is used when no product profile is available.

    The None tenant key represents the single shared profile used when
    isolation is off (or inside unit tests that have no identity configured).

    The scope adapter (scope.py) calls set_config() after a successful database
    load.  The router and prepare() use the per-tenant sync methods together
    with the DB helpers in scope.py.
    """

    def __init__(self, product_profile=None) -> None:
        self._product_profile = product_profile
        self._startup: AssistantConfig = _defaults_from_product_profile(product_profile)
        self._configs: dict[str | None, AssistantConfig] = {}
        self._revisions: dict[str | None, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ read

    def get(self, tenant: str | None = None) -> _ProfileView:
        """Return a view of the current profile for tenant (copy)."""
        with self._lock:
            cfg = self._configs.get(tenant, self._startup)
            return _ProfileView(copy.deepcopy(cfg))

    def revision(self, tenant: str | None = None) -> int:
        """Current revision for tenant; 0 means no custom row exists."""
        with self._lock:
            return self._revisions.get(tenant, 0)

    # ------------------------------------------------------------------ write

    def patch(self, partial: dict, *, tenant: str | None = None,
              product_profile=None) -> _ProfileView:
        """Merge partial dict over the current config for tenant (in-memory only).

        Used by unit tests and the sync fall-back path. For DB persistence call
        scope.save_profile_for_request() instead.

        product_profile overrides the constructor's profile for clamping when the
        router supplies the live loaded profile; None falls back to the constructor's.
        """
        effective_profile = product_profile if product_profile is not None else self._product_profile
        with self._lock:
            base = self._configs.get(tenant, self._startup)
            cfg = _apply_patch(base, partial)
            cfg = apply_product_clamp(cfg, effective_profile)
            self._configs[tenant] = cfg
            self._revisions[tenant] = self._revisions.get(tenant, 0) + 1
            return _ProfileView(copy.deepcopy(cfg))

    def reset(self, *, tenant: str | None = None) -> _ProfileView:
        """Revert to startup defaults for tenant (in-memory only)."""
        with self._lock:
            self._configs.pop(tenant, None)
            self._revisions.pop(tenant, None)
            log.info("profile_store: reverted to startup profile (tenant omitted)")
            return _ProfileView(copy.deepcopy(self._startup))

    # ------------------------------------------------------------------ DB sync

    def set_config(self, tenant: str | None, config: AssistantConfig, revision: int) -> None:
        """Cache a config loaded from the database.  Called by scope.py after a DB read."""
        with self._lock:
            self._configs[tenant] = config
            self._revisions[tenant] = revision

    # ------------------------------------------------------------------ defaults

    @property
    def startup(self) -> AssistantConfig:
        """The product-profile defaults (never mutated)."""
        return copy.deepcopy(self._startup)
