"""The network decision (PL41): may a tool that leaves the device run now?

One deterministic rule, consulted before anything that needs the network runs: web search on
the delegate lane, the Google calendar import, and every later network tool through the tool
registry. In order:

  refused  in senior care and classroom, always (no profile can turn it on there)
  refused  when the product profile says network.allowed is false
  refused  when the privacy mode is not "connectors"
  allowed  otherwise

Which privacy mode: the tenant's stored setting (settings.privacyMode, written by
PATCH /v1/settings and shown in the app's Settings) is the one authority for this decision. The
environment value (VOXSCRIBE_PRIVACY_MODE, Config.privacy_mode) is not read here. A setting that
cannot be read, or holds anything but "connectors", counts as "offline".

decide() is pure, with no I/O, shaped to move into voxscribe-core with the tool contract (PL58).
network_decision() is the server's reader around it. Nothing here logs what anyone said.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.local import tenancy

log = logging.getLogger("app.local.tools.network")

RESTRICTED_MODES = frozenset({"senior_care", "classroom", "seniorcare"})

PRIVACY_CONNECTORS = "connectors"
PRIVACY_OFFLINE = "offline"
PRIVACY_UNREAD = "unread"          # not needed: the decision was already a refusal

REASON_OK = "ok"
REASON_MODE = "mode_restricted"
REASON_PROFILE = "profile_network_off"
REASON_PRIVACY = "privacy_offline"

# Scripted, fixed text. Each passes the response gate like any other sentence.
REFUSAL_LINES: dict[str, str] = {
    REASON_MODE: "I can't look things up online here.",
    REASON_PROFILE: "Looking things up online is turned off for this assistant.",
    REASON_PRIVACY: ("I can't look that up while privacy is set to offline. "
                     "You can turn on connectors in Settings."),
}


@dataclass(frozen=True)
class NetworkDecision:
    allowed: bool
    reason: str
    mode: str
    profile_allowed: bool
    privacy_mode: str

    @property
    def line(self) -> str:
        """The scripted refusal line; empty when the network is allowed."""
        return REFUSAL_LINES.get(self.reason, "")


def decide(*, mode: str, profile_allowed: bool, privacy_mode: str) -> NetworkDecision:
    """The rule itself. Pure: the same inputs always give the same decision."""
    def made(allowed: bool, reason: str) -> NetworkDecision:
        return NetworkDecision(allowed=allowed, reason=reason, mode=mode,
                               profile_allowed=profile_allowed, privacy_mode=privacy_mode)

    if mode in RESTRICTED_MODES:
        return made(False, REASON_MODE)
    if not profile_allowed:
        return made(False, REASON_PROFILE)
    if privacy_mode != PRIVACY_CONNECTORS:
        return made(False, REASON_PRIVACY)
    return made(True, REASON_OK)


def product_mode(app) -> str:
    """The session's product mode as the response gate resolved it (the profile's mode, tightened
    by VOXSCRIBE_VOICE_MODE), so the gate and this decision never disagree about the mode."""
    from app.local.voice.adapters.gate_policy import session_policy  # lazy: no import cycle

    return session_policy(app).stamp.mode


def profile_network_allowed(app) -> bool:
    profile = getattr(app.state, "profile", None)
    return bool(getattr(getattr(profile, "network", None), "allowed", True))


async def stored_privacy_mode(pool) -> str:
    """The tenant's stored privacy mode. Fails closed to offline; a tenancy refusal is raised,
    so a call outside the tenant's scope stops before anything is sent."""
    from app.local.store_sources import SourceStore

    try:
        value = await SourceStore(pool).get_privacy_mode()
    except tenancy.TenantRequiredError:
        raise
    except Exception as exc:  # noqa: BLE001 - an unreadable setting is offline
        log.warning("network: privacy setting unreadable, treating as offline (%s)",
                    type(exc).__name__)
        return PRIVACY_OFFLINE
    return PRIVACY_CONNECTORS if value == PRIVACY_CONNECTORS else PRIVACY_OFFLINE


async def network_decision(app, *, pool=None) -> NetworkDecision:
    """The decision for this app's profile and the tenant in scope.

    pool: the pool to read the stored setting through; None means the voice path's
    scoped_pool(app). The setting is read only when mode and profile have not already refused.
    """
    mode = product_mode(app)
    allowed = profile_network_allowed(app)
    if mode in RESTRICTED_MODES or not allowed:
        decision = decide(mode=mode, profile_allowed=allowed, privacy_mode=PRIVACY_UNREAD)
    else:
        if pool is None:
            from app.local.voice.adapters.scope import scoped_pool  # lazy: no import cycle

            pool = scoped_pool(app)
        decision = decide(mode=mode, profile_allowed=allowed,
                          privacy_mode=await stored_privacy_mode(pool))
    log.info("network: decision allowed=%s reason=%s mode=%s", decision.allowed,
             decision.reason, decision.mode)
    return decision
