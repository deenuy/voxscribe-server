"""Safety gate v0 -- mode-aware reply filter.

G1: gate_decisions row written before playback (enforced by reply_chain.py).
G2: classroom mode refuses forbidden-topic prompts with a fixed phrase.
G3: seniorcare mode refuses medical dosing with the escalation phrase.
G4: injection-resistant -- gate uses fixed regex, not LLM output or user text structure.
G5: gate failure never plays model text; caller uses refusal_phrase() only (enforced by reply_chain.py).
G6: behaviour log written per exchange to BEHAVIOUR_LOG_DIR (no transcript text).
"""

from __future__ import annotations

import json
import logging
import os
import re
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


def refusal_phrase(category: str, mode: str) -> str:
    """Return the fixed refusal phrase for this gate outcome."""
    if mode == "seniorcare" and category == "seniorcare_dosing":
        return SENIORCARE_ESCALATION_PHRASE
    if mode == "classroom" and category == "classroom_forbidden":
        return CLASSROOM_REFUSAL_PHRASE
    return DEFAULT_REFUSAL_PHRASE
