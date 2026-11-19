"""Host floor: the fixed rules that run behind core's policy engine.

The authority on what may be spoken is voxscribe-core's policy engine with the versioned pack
for the session's mode (app/local/voice/adapters/gate_policy.py). These rules run only after
the engine has allowed a sentence, and can only refuse or replace it: they never allow what
the engine denied. Each rule they fire is recorded under FLOOR_VERSION. They stay because
core 0.5.0's packs do not carry them yet; see the gate adapter for the list.

G1: a gate record exists before playback (the heard row, written by gate_records.py).
G2: classroom mode refuses forbidden-topic prompts with a fixed phrase.
G3: seniorcare mode refuses medical dosing with the escalation phrase.
G4: injection-resistant -- gate uses fixed regex, not LLM output or user text structure.
G5: gate failure never plays model text; caller uses refusal_phrase() only (enforced by reply_chain.py).
G6: behaviour log written per exchange to BEHAVIOUR_LOG_DIR (no transcript text).
G7: a sentence claiming a completed action (reminder, alarm, timer, calendar) is replaced with
    a fixed phrase unless an executed receipt of that action's type exists for the turn that
    speaks it, or for the turn whose yes confirmed it (Defect F; PL43, TOOLS-CONNECTORS.md 2.6).
    A failed call, a refusal, a read, or a receipt from an earlier turn backs nothing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("app.local.gate")

# Stamped on every record a floor rule decides. Bump it when a rule below changes.
FLOOR_VERSION = "host-floor-v3"      # v2: G7 needs a receipt in this turn (PL43); v3: the
                                     # timer and cancel claim forms, a truthful replacement (PL45)

# ── Fixed reply phrases (immutable per VOICE-AUTOPILOT.md) ──────────────────

CLASSROOM_REFUSAL_PHRASE = "I can't help with that in classroom mode."
SENIORCARE_ESCALATION_PHRASE = (
    "I'm not able to give medical advice. Please speak with a doctor or caregiver."
)
DEFAULT_REFUSAL_PHRASE = "I didn't catch that. Could you say it again?"

# ── Forbidden patterns (classroom mode) ─────────────────────────────────────
# Checked against both user_text and reply_text so injected model output is caught too.

_CLASSROOM_FORBIDDEN: list[re.Pattern] = [
    re.compile(r"\b(explicit|sexual|nude|naked)\b", re.IGNORECASE),
    re.compile(r"\b(bomb|explosiv|weapon\s+making)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+to\s+(make|build|create)\s+\w*\s*(weapon|gun|bomb|explosiv)\b", re.IGNORECASE),
    re.compile(r"\b(self[- ]?harm|suicid)\b", re.IGNORECASE),
    re.compile(r"\b(drug\s+(deal|use|traffick)|illegal\s+drug)\b", re.IGNORECASE),
]

# ── Dosing patterns (seniorcare mode) ───────────────────────────────────────

_SENIORCARE_DOSING: list[re.Pattern] = [
    re.compile(r"\b\d+\s*(mg|ml|mcg|g|tablet|pill|capsule)s?\b", re.IGNORECASE),
    re.compile(r"\b(dosage|overdose|how\s+much\s+to\s+take|how\s+many\s+(pill|tablet|capsule))\b", re.IGNORECASE),
    re.compile(r"\btake\s+\d+\b", re.IGNORECASE),
]

# ── Behaviour log (G6) ───────────────────────────────────────────────────────

_BEHAVIOUR_LOG_DIR = Path(
    os.environ.get("VOXSCRIBE_BEHAVIOUR_LOG_DIR", Path.home() / ".voxscribe" / "logs")
)


def _behaviour_log_path() -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return _BEHAVIOUR_LOG_DIR / f"behaviour_{today}.jsonl"


def write_behaviour_log(
    *,
    meeting_id: str,
    turn_n: int,
    mode: str,
    decision: str,
    category: str,
) -> None:
    """Append one line to the daily behaviour log. No transcript text."""
    try:
        _BEHAVIOUR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "meeting_id": meeting_id,
            "turn_n": turn_n,
            "mode": mode,
            "gate_decision": decision,
            "gate_category": category,
        }
        with _behaviour_log_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        log.exception("gate: failed to write behaviour log meeting_id=%s", meeting_id)


# ── Gate check (injection-resistant: fixed regex only) ──────────────────────


def check(user_text: str, reply_text: str, mode: str = "default") -> tuple[str, str]:
    """Return (decision, category).

    decision: "pass" | "refuse"
    category: "ok" | "empty" | "classroom_forbidden" | "seniorcare_dosing"

    Injection-resistant: only evaluates fixed regex patterns against both texts.
    Injected text in user_text or reply_text cannot override the policy.
    """
    if not reply_text.strip():
        return "refuse", "empty"

    if mode == "classroom":
        for pattern in _CLASSROOM_FORBIDDEN:
            if pattern.search(user_text) or pattern.search(reply_text):
                return "refuse", "classroom_forbidden"

    if mode == "seniorcare":
        for pattern in _SENIORCARE_DOSING:
            if pattern.search(user_text) or pattern.search(reply_text):
                return "refuse", "seniorcare_dosing"

    return "pass", "ok"


# ── G7: action claims must be backed by a receipt ───────────────────────────
# The reply model has no reminder, alarm, timer or calendar tool, yet it says
# "I've added a reminder" (ROOT-CAUSE Defect F). Any sentence claiming such an
# action is replaced unless the turn speaking it has an executed receipt of that
# action's type (app/local/tools/audit.py turn_receipts). The success lines of the
# engine's own tools are claims too, so they pass only after the store committed.
# Extend the tuples below; the control flow does not change.

# Reminders and timers exist now (PL45), so the replacement says only what is true: nothing was
# set by that sentence.
ACTION_CLAIM_PHRASE = "I haven't set anything."
ACTION_CLAIM_REASON = "no_receipt"
ACTION_CLAIM_BACKED = "receipt"

# Action-completion language, matched case-insensitively on word boundaries.
ACTION_CLAIM_KEYWORDS: tuple[str, ...] = (
    "I've added",
    "I've set",
    "I've scheduled",
    "I've saved",
    "I've created",
    "I have added",
    "I have set",
    "I have scheduled",
    "I have saved",
    "I have created",
    "reminder is set",
    "reminder has been",
    "alarm is set",
    "alarm has been",
    "timer is set",
    "timer has been",
    "I will remind you",
    "I'll remind you",
    "I'll set",
    "set a reminder",
    "set an alarm",
    "set a timer",
    "added a reminder",
    "added an alarm",
    # The forms the tools' templates and the models use (section 2.6).
    "timer set",
    "reminder set",
    "alarm set",
    "added to your calendar",
    # The timer announcement and the cancel lines of the PL45 tools (app/local/tools/builtin.py).
    "timer, starting now",
    "timer starting now",
    "timer cancelled",
    "timer is cancelled",
    "reminder cancelled",
    "reminder is cancelled",
)

# Tool types that can back an action claim, each with the words that name it in a sentence.
# A receipt's tool name backs a claim when it contains the tool type (e.g. "reminder.create").
ACTION_TOOL_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reminder", ("remind",)),
    ("alarm",    ("alarm",)),
    ("timer",    ("timer",)),
    ("calendar", ("calendar", "event", "appointment", "meeting")),
)

_ACTION_CLAIM_RE = re.compile(
    "|".join(rf"\b{re.escape(k.lower())}\b" for k in ACTION_CLAIM_KEYWORDS)
)


def _normalise(sentence: str) -> str:
    # LLMs emit typographic apostrophes; match them as plain ones.
    return sentence.replace("’", "'").replace("‘", "'").lower()


def find_action_claim(sentence: str) -> str | None:
    """Return the matched action-completion keyword, or None. Cheap: no I/O."""
    m = _ACTION_CLAIM_RE.search(_normalise(sentence))
    return m.group(0) if m else None


def _claimed_tool_types(sentence: str) -> set[str]:
    text = _normalise(sentence)
    return {tool for tool, words in ACTION_TOOL_TYPES if any(w in text for w in words)}


def _backing_tool_types(receipts: Iterable[str]) -> set[str]:
    names = [t.lower() for t in receipts if t]
    return {tool for tool, _ in ACTION_TOOL_TYPES if any(tool in n for n in names)}


def check_action_claim(sentence: str, receipts: Iterable[str]) -> tuple[str, str]:
    """Return (decision, reason) for one sentence about to be spoken.

    decision: "pass" | "replace"
    reason:   "ok" (no claim) | "receipt" (claim backed) | "no_receipt" (claim not backed)

    receipts: the tool names with an executed write receipt for the turn speaking (or the turn
    that confirmed the call). A claim naming a tool type (e.g. "alarm") needs a receipt of that
    type; a claim naming none ("I've saved it") needs a receipt of any action tool type. Web
    search and other reads never back a claim.
    On "replace", speak ACTION_CLAIM_PHRASE instead of the sentence.
    """
    keyword = find_action_claim(sentence)
    if keyword is None:
        return "pass", "ok"
    claimed = _claimed_tool_types(sentence)
    backing = _backing_tool_types(receipts)
    if (claimed & backing) if claimed else backing:
        return "pass", ACTION_CLAIM_BACKED
    log.info("gate: blocked action claim, sentence_len=%d, reason=%s",
             len(sentence), ACTION_CLAIM_REASON)
    return "replace", ACTION_CLAIM_REASON


def refusal_phrase(category: str, mode: str) -> str:
    """Return the fixed refusal phrase for this gate outcome."""
    if mode == "seniorcare" and category == "seniorcare_dosing":
        return SENIORCARE_ESCALATION_PHRASE
    if mode == "classroom" and category == "classroom_forbidden":
        return CLASSROOM_REFUSAL_PHRASE
    return DEFAULT_REFUSAL_PHRASE
