"""Safety gate v0 -- mode-aware reply filter.

G1: gate_decisions row written before playback (enforced by reply_chain.py).
G2: classroom mode refuses forbidden-topic prompts with a fixed phrase.
G3: seniorcare mode refuses medical dosing with the escalation phrase.
G4: injection-resistant -- gate uses fixed regex, not LLM output or user text structure.
G5: gate failure never plays model text; caller uses refusal_phrase() only (enforced by reply_chain.py).
G6: behaviour log written per exchange to BEHAVIOUR_LOG_DIR (no transcript text).
G7: a sentence claiming a completed action (reminder, alarm, timer, calendar) with no
    matching tool_calls row in the session is replaced with a fixed phrase (Defect F).
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


# ── G7: action claims must be backed by a tool row ──────────────────────────
# The reply model has no reminder, alarm, timer or calendar tool, yet it says
# "I've added a reminder" (ROOT-CAUSE Defect F). Any sentence claiming such an
# action is replaced unless the session has a tool_calls row for that tool type.
# Extend the tuples below; the control flow does not change.

ACTION_CLAIM_PHRASE = "I can't set reminders or alarms yet, but I can help answer questions."
ACTION_CLAIM_REASON = "no_tool_row"

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
)

# Tool types that can back an action claim, each with the words that name it in a sentence.
# A tool_calls.tool value backs a claim when it contains the tool type (e.g. "reminder_create").
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


def _backing_tool_types(session_tools: Iterable[str]) -> set[str]:
    names = [t.lower() for t in session_tools if t]
    return {tool for tool, _ in ACTION_TOOL_TYPES if any(tool in n for n in names)}


def check_action_claim(sentence: str, session_tools: Iterable[str]) -> tuple[str, str]:
    """Return (decision, reason) for one sentence about to be spoken.

    decision: "pass" | "replace"
    reason:   "ok" (no claim) | "tool_row" (claim backed) | "no_tool_row" (claim not backed)

    session_tools: tool_calls.tool values recorded in this session. A claim naming a tool type
    (e.g. "alarm") needs a row of that type; a claim naming none ("I've saved it") needs a row
    of any action tool type. Web search never backs a claim.
    On "replace", speak ACTION_CLAIM_PHRASE instead of the sentence.
    """
    keyword = find_action_claim(sentence)
    if keyword is None:
        return "pass", "ok"
    claimed = _claimed_tool_types(sentence)
    backing = _backing_tool_types(session_tools)
    if (claimed & backing) if claimed else backing:
        return "pass", "tool_row"
    log.info("gate: blocked action claim, sentence=%r, reason=%s",
             sentence[:50], ACTION_CLAIM_REASON)
    return "replace", ACTION_CLAIM_REASON


def refusal_phrase(category: str, mode: str) -> str:
    """Return the fixed refusal phrase for this gate outcome."""
    if mode == "seniorcare" and category == "seniorcare_dosing":
        return SENIORCARE_ESCALATION_PHRASE
    if mode == "classroom" and category == "classroom_forbidden":
        return CLASSROOM_REFUSAL_PHRASE
    return DEFAULT_REFUSAL_PHRASE
