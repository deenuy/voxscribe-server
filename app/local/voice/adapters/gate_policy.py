"""Gate over voxscribe-core's policy engine: the host's one safety path (PL09).

    candidate -> core PolicyGate.evaluate   the versioned pack for the session's product mode
              -> host floor                 app.local.gate, only on what the engine allowed

Core's engine is the authority. A denial or a fallback from it is final and is spoken as the
pack's scripted line. The host floor runs behind it and can only tighten: it holds the rules
this server enforced before PL09 that core 0.5.0's packs cannot express yet (dosing and
forbidden-topic patterns matched on the person's words as well as the reply, a scripted line
per rule, and action claims that need a receipt in the turn). Its own scripted lines pass the
engine too.

The pack is read from the installed wheel, once per process. Anything short of a clean load
(unknown mode, missing or malformed pack, a senior care or classroom pack that is not in
allowlist posture) leaves core's fail-closed gate in place: every reply becomes the scripted
fallback and every record says `gate-unavailable`.

TurnGate is the per-reply wrapper the host puts around the Gate port. It journals every
evaluation for the gate records, and it turns a gate that raises or answers late into a raise,
which the reply stage speaks as the scripted fallback.

The ledger is part of the gate where the policy says so (invariant 5a). Under a restricted pack
(senior care, classroom) or any pack in allowlist posture, a turn whose heard record could not
be written is a gate fault: TurnGate refuses everything and the reply is the scripted fallback.
Under a denylist pack (personal, ideascribe) the reply proceeds: every clause is still
evaluated, the failed write is logged, and the records are written at the end of the reply if
the ledger is back by then (gate_records.flush_reply).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from importlib import metadata

from voxscribe import policy as core_policy
from voxscribe.session.ports import GATE_PASS, GATE_REFUSE, GATE_REPLACE, GateDecision

from app.local import gate as floor
from app.local.gate_records import (
    EVENT_CUE,
    EVENT_DECISION,
    RULE_PASSED,
    GateEvent,
    PolicyStamp,
    candidate_digest,
    now_z,
)

log = logging.getLogger("app.local.voice.gate")

SURFACE_TTS = "tts"                  # every clause the host speaks is evaluated on this surface

# Product mode (as the profile spells it) -> core policy pack (canonical pack filename).
# Core 0.6.0 ships a personal pack and accepts both "senior_care" and "seniorcare" for
# the senior care pack. PACK_ALIASES maps profile-layer spellings to canonical filenames
# for the case where a profile explicitly sets gate.policy_pack with an underscore spelling.
PACK_FOR_MODE = {
    "senior_care": "seniorcare",
    "classroom": "classroom",
    "ideascribe": "ideascribe",
    "personal": "personal",
}
# Explicit aliases for pack names a profile may spell with underscores (profile-layer convention).
# Replaces the previous generic .replace("_", "") normalisation; core handles both spellings too.
PACK_ALIASES = {"senior_care": "seniorcare"}
RESTRICTED_PACKS = ("seniorcare", "classroom")       # must load, and must load as allowlist
MODE_FOR_RESTRICTED_PACK = {"seniorcare": "senior_care", "classroom": "classroom"}
PROFILE_DEFAULT_PACK = "default"                     # GateConfig.policy_pack when a profile names none

CATEGORY_POLICY_DENY = "policy_deny"                 # the pack denied the candidate
CATEGORY_POLICY_FALLBACK = "policy_fallback"         # the engine could not allow it (fail closed)
CATEGORY_GATE_ERROR = "gate_error"                   # the gate raised
CATEGORY_GATE_TIMEOUT = "gate_timeout"               # the gate answered late, or its lookup hung
CATEGORY_NO_REPLY = "no_reply"                       # the source yielded nothing: the fallback line
CATEGORY_GATE_LEDGER = "gate_ledger"                 # the gate's ledger could not be written
RULE_HEARD_UNWRITTEN = "gate_ledger:heard_unwritten"

# A verdict that took longer than this is discarded and the reply fails closed. The port's
# check is synchronous, so this cannot interrupt a gate that never returns; it bounds what the
# host will act on. An evaluation costs microseconds (see the PL09 report for the measurement).
CHECK_DEADLINE_MS = 250
# The receipt lookup an action claim needs (this turn's receipts, PL43) is the gate's only I/O.
# Past this the reply fails closed instead of waiting on the database.
TOOLS_TIMEOUT_S = 1.0


class GateTimeout(RuntimeError):
    """The gate did not answer in time. Carries no text."""


class GateLedgerError(RuntimeError):
    """What was heard could not be put on record under a policy that requires it. Carries no
    text."""


def ledger_required(stamp) -> bool:
    """True when the session's policy makes the gate's records a condition of speaking: a
    restricted pack, or any pack in allowlist posture."""
    return stamp is not None and (getattr(stamp, "pack", "") in RESTRICTED_PACKS
                                  or getattr(stamp, "posture", "") == "allowlist")


@dataclass(frozen=True)
class PolicyDecision(GateDecision):
    """The port's verdict, with what the gate records need beside it."""

    rule: str = RULE_PASSED
    policy_version: str = ""
    candidate_hash: str = ""


@dataclass(frozen=True)
class LoadedPolicy:
    engine: core_policy.PolicyGate
    stamp: PolicyStamp
    floor_mode: str                  # the mode name app.local.gate's rules key on


def _engine_build() -> str:
    try:
        core = metadata.version("voxscribe-core")
    except metadata.PackageNotFoundError:
        core = "unknown"
    return f"voxscribe-core {core}; {floor.FLOOR_VERSION}"


def _resolve(profile, settings) -> tuple[str, str]:
    """(product mode, pack name) for the loaded profile. Raises PolicyPackError when the profile
    and the server's VOXSCRIBE_VOICE_MODE name two different restricted modes."""
    mode = getattr(profile, "mode", None) or "personal"
    named = getattr(getattr(profile, "gate", None), "policy_pack", None) or PROFILE_DEFAULT_PACK
    if named == PROFILE_DEFAULT_PACK:
        pack = PACK_FOR_MODE.get(mode, "")
    else:
        raw = str(named).strip().lower()
        pack = PACK_ALIASES.get(raw, raw)      # apply explicit aliases; core handles both spellings
    if mode in PACK_FOR_MODE and PACK_FOR_MODE[mode] in RESTRICTED_PACKS \
            and pack != PACK_FOR_MODE[mode]:
        raise core_policy.PolicyPackError(f"{mode} mode must run its own pack, profile names {pack!r}")
    # VOXSCRIBE_VOICE_MODE turned the classroom and senior care rules on before profiles
    # existed. It can still tighten a personal or ideascribe profile; it cannot contradict one.
    legacy = getattr(settings, "voice_mode", None) or "default"
    if legacy in RESTRICTED_PACKS and legacy != pack:
        if pack in RESTRICTED_PACKS:
            raise core_policy.PolicyPackError(
                f"profile mode {mode!r} and VOXSCRIBE_VOICE_MODE={legacy!r} disagree")
        mode, pack = MODE_FOR_RESTRICTED_PACK[legacy], legacy
    return mode, pack


def load_policy(profile=None, settings=None) -> LoadedPolicy:
    """The policy for the loaded profile, from the pack the wheel ships. Never raises: any
    failure returns core's fail-closed gate, stamped `gate-unavailable`."""
    mode = getattr(profile, "mode", None) or "personal"
    pack_name = ""
    try:
        mode, pack_name = _resolve(profile, settings)
        from voxscribe.resources import resolve as resolve_resource

        path = resolve_resource(f"resources_config/policies/{pack_name}.yaml")
        pack = core_policy.load_policy_pack(pack_name, path.parent)
        if pack_name in RESTRICTED_PACKS and pack.posture != "allowlist":
            raise core_policy.PolicyPackError(f"{pack_name} pack must be allowlist, is {pack.posture}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        engine = core_policy.PolicyGate(pack=pack)
    except Exception as exc:  # noqa: BLE001 - fail closed on anything
        log.error("gate policy: no usable pack for mode=%s pack=%s (%s: %s); every reply will"
                  " be the scripted fallback", mode, pack_name or "none", type(exc).__name__, exc)
        engine, digest = core_policy.fail_closed_gate(pack_name or mode), ""
    pack = engine.pack
    stamp = PolicyStamp(mode=mode, pack=pack_name or pack.mode, policy_version=pack.policy_version,
                        posture=pack.posture, digest=digest, engine=_engine_build())
    log.info("gate policy: mode=%s pack=%s version=%s posture=%s digest=%s", stamp.mode,
             stamp.pack, stamp.policy_version, stamp.posture, stamp.digest[:12] or "none")
    return LoadedPolicy(engine=engine, stamp=stamp,
                        floor_mode=pack_name if pack_name in RESTRICTED_PACKS else "default")


def session_policy(app) -> LoadedPolicy:
    """The process's loaded policy, resolved once from app.state.profile and kept on app.state.
    The gate adapter and the store's gate records both read it, so they stamp the same version."""
    state = app.state
    policy = getattr(state, "gate_policy", None)
    if policy is None:
        policy = load_policy(getattr(state, "profile", None), getattr(state, "settings", None))
        state.gate_policy = policy
    return policy


class CorePolicyGate:
    """The Gate port over core's PolicyGate, with the host floor behind it."""

    def __init__(self, policy: LoadedPolicy | None = None) -> None:
        self._policy = policy or load_policy()
        self._engine = self._policy.engine
        self._floor_mode = self._policy.floor_mode

    @property
    def stamp(self) -> PolicyStamp:
        return self._policy.stamp

    @property
    def policy_version(self) -> str:
        return self._policy.stamp.policy_version

    @property
    def fail_closed_phrase(self) -> str:
        """The pack's scripted fallback line: policy configuration, not host code."""
        return self._engine.pack.fallback_line

    def needs_session_tools(self, sentence: str) -> bool:
        return floor.find_action_claim(sentence) is not None

    def check(self, sentence: str, session_tools: Sequence[str] = (), *,
              user_text: str = "") -> PolicyDecision:
        verdict = self._engine.evaluate(sentence, surface=SURFACE_TTS)

        def decided(decision: str, category: str, replacement: str | None,
                    rule: str) -> PolicyDecision:
            return PolicyDecision(decision, category, replacement, rule=rule,
                                  policy_version=verdict.policy_version,
                                  candidate_hash=verdict.candidate_hash)

        if verdict.outcome != "allow":
            category = (CATEGORY_POLICY_DENY if verdict.outcome == "deny"
                        else CATEGORY_POLICY_FALLBACK)
            return decided(GATE_REFUSE, category, self.fail_closed_phrase, verdict.rule)
        if verdict.delivered_text != sentence:
            # An allow that is not the candidate is not an allow.
            return decided(GATE_REFUSE, CATEGORY_POLICY_FALLBACK, self.fail_closed_phrase,
                           "allow_text_mismatch")

        # Host floor: reached only by what the engine allowed, and it can only tighten.
        decision, category = floor.check(user_text, sentence, mode=self._floor_mode)
        if decision != GATE_PASS:
            return decided(GATE_REFUSE, category,
                           self._scripted(floor.refusal_phrase(category, self._floor_mode)),
                           f"{floor.FLOOR_VERSION}:{category}")
        if floor.find_action_claim(sentence) is not None:
            decision, reason = floor.check_action_claim(sentence, session_tools)
            if decision == GATE_REPLACE:
                return decided(GATE_REPLACE, reason, self._scripted(floor.ACTION_CLAIM_PHRASE),
                               f"{floor.FLOOR_VERSION}:{reason}")
        return decided(GATE_PASS, "ok", None, RULE_PASSED)

    def _scripted(self, phrase: str) -> str:
        """A floor phrase is spoken only if the engine allows it; otherwise the pack's line."""
        if self._engine.evaluate(phrase, surface=SURFACE_TTS).outcome == "allow":
            return phrase
        return self.fail_closed_phrase


SessionTools = Callable[[], Awaitable[Sequence[str]]]


class TurnGate:
    """One reply's view of the gate.

    Every evaluation is appended to `events` (the store writes them as gate records when the
    reply ends). A gate that raises, a verdict that arrives past the deadline, and a tool-row
    lookup that hangs are each journalled and then raised, so the reply stage refuses with the
    gate's scripted fallback and the candidate is never synthesised. Scripted lines (the cue,
    the lead-in and the line recorded when the source yields nothing) pass through it too.
    """

    def __init__(self, inner, events: list[GateEvent] | None = None, *,
                 deadline_ms: float | None = None, tools_timeout_s: float | None = None,
                 timer: Callable[[], int] = time.perf_counter_ns) -> None:
        self._inner = inner
        self.events: list[GateEvent] = events if events is not None else []
        self._deadline_us = int((CHECK_DEADLINE_MS if deadline_ms is None else deadline_ms) * 1000)
        self._tools_timeout_s = TOOLS_TIMEOUT_S if tools_timeout_s is None else tools_timeout_s
        self._timer = timer
        self._asked: str = ""            # the sentence the stage last asked about
        self._fault: str = ""            # the rule that faulted this turn's gate, if one did

    @classmethod
    def for_turn(cls, inner, turn) -> TurnGate:
        """Journal onto the store's record for this turn when it keeps one. When that record
        says the turn's heard row could not be written and its policy requires the ledger, the
        gate is faulted for the whole reply."""
        handle = getattr(turn, "handle", None)
        gate = cls(inner, getattr(handle, "gate_events", None))
        if getattr(handle, "heard_written", True) is False \
                and ledger_required(getattr(handle, "policy", None)):
            gate._fault = RULE_HEARD_UNWRITTEN
        return gate

    @property
    def faulted(self) -> bool:
        """True when nothing this turn may be spoken but the scripted fallback."""
        return bool(self._fault)

    @property
    def fail_closed_phrase(self) -> str:
        return self._inner.fail_closed_phrase

    def needs_session_tools(self, sentence: str) -> bool:
        self._asked = sentence
        if self._fault:
            return False                 # check() refuses; no lookup is spent on it
        try:
            return self._inner.needs_session_tools(sentence)
        except Exception as exc:
            self._journal(EVENT_DECISION, sentence, GATE_REFUSE, CATEGORY_GATE_ERROR,
                          f"gate_error:{type(exc).__name__}", cost_us=0)
            raise

    def check(self, sentence: str, session_tools: Sequence[str] = (), *,
              user_text: str = "") -> GateDecision:
        return self._evaluate(EVENT_DECISION, sentence, session_tools, user_text)

    def tools(self, fetch: SessionTools) -> SessionTools:
        """`fetch`, bounded: a lookup that does not return in time fails the reply closed."""
        async def bounded() -> Sequence[str]:
            started = self._timer()
            try:
                async with asyncio.timeout(self._tools_timeout_s):
                    return await fetch()
            except Exception as exc:
                timed_out = isinstance(exc, TimeoutError)
                self._journal(
                    EVENT_DECISION, self._asked, GATE_REFUSE,
                    CATEGORY_GATE_TIMEOUT if timed_out else CATEGORY_GATE_ERROR,
                    "gate_timeout:session_tools" if timed_out
                    else f"gate_error:session_tools:{type(exc).__name__}",
                    cost_us=(self._timer() - started) // 1000)
                if timed_out:
                    raise GateTimeout("session tools lookup timed out") from None
                raise
        return bounded

    def scripted(self, phrase: str | None) -> str | None:
        """A cue or lead-in, if the gate passes it; None otherwise, and nothing is spoken."""
        if not phrase:
            return phrase
        try:
            verdict = self._evaluate(EVENT_CUE, phrase, (), "")
        except Exception:  # noqa: BLE001 - journalled in _evaluate; a cue is simply dropped
            return None
        return phrase if verdict.decision == GATE_PASS else None

    def fallback(self, phrase: str) -> tuple[str, str, str]:
        """(text, decision, category) to record for a reply whose source yielded nothing.

        The stream's fallback line is not spoken by the reply stage, but it becomes the reply
        text: shown, stored, kept in the history and spoken if the person asks for a repeat.
        It is scripted text like a cue, so it is evaluated like one and journalled. Passed, it
        is recorded under CATEGORY_NO_REPLY (the journal row too), never as a reply that
        passed. Refused, or with the gate down, no text is recorded at all."""
        if not phrase:
            return "", GATE_PASS, CATEGORY_NO_REPLY
        try:
            verdict = self._evaluate(EVENT_CUE, phrase, (), "", passed_as=CATEGORY_NO_REPLY)
        except Exception:  # noqa: BLE001 - journalled in _evaluate; the line is dropped
            return "", GATE_REFUSE, self.events[-1].category
        if verdict.decision == GATE_PASS:
            return phrase, GATE_PASS, CATEGORY_NO_REPLY
        return "", GATE_REFUSE, verdict.category

    # ------------------------------------------------------------------ internals

    def _evaluate(self, event: str, text: str, session_tools: Sequence[str],
                  user_text: str, *, passed_as: str | None = None) -> GateDecision:
        if self._fault:
            self._journal(event, text, GATE_REFUSE, CATEGORY_GATE_LEDGER, self._fault, cost_us=0)
            raise GateLedgerError("the gate's ledger could not be written")
        started = self._timer()
        try:
            verdict = self._inner.check(text, session_tools, user_text=user_text)
            decision, category = verdict.decision, verdict.category
        except Exception as exc:
            self._journal(event, text, GATE_REFUSE, CATEGORY_GATE_ERROR,
                          f"gate_error:{type(exc).__name__}",
                          cost_us=(self._timer() - started) // 1000)
            raise
        cost_us = (self._timer() - started) // 1000
        if cost_us > self._deadline_us:
            self._journal(event, text, GATE_REFUSE, CATEGORY_GATE_TIMEOUT, "gate_timeout:check",
                          cost_us=cost_us)
            raise GateTimeout("gate verdict arrived past its deadline")
        rule = getattr(verdict, "rule", None) or (RULE_PASSED if decision == GATE_PASS
                                                  else category)
        if passed_as and decision == GATE_PASS:
            category = passed_as
        self._journal(event, text, decision, category, rule, cost_us=cost_us,
                      policy_version=getattr(verdict, "policy_version", ""),
                      candidate_hash=getattr(verdict, "candidate_hash", ""))
        return verdict

    def _journal(self, event: str, text: str, decision: str, category: str, rule: str, *,
                 cost_us: int, policy_version: str = "", candidate_hash: str = "") -> None:
        self.events.append(GateEvent(
            event=event, step_n=len(self.events) + 1, surface=SURFACE_TTS, decision=decision,
            category=category, rule=rule,
            policy_version=policy_version or getattr(self._inner, "policy_version", ""),
            candidate_hash=candidate_hash or candidate_digest(text),
            candidate_len=len(text or ""), cost_us=int(cost_us), recorded_at=now_z()))
