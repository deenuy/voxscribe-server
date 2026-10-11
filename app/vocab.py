"""Controlled vocabulary, status thresholds, and small shared helpers.

Mirrors the IdeaScribe canonical data model so the JSON this service emits
matches the client contract exactly.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

IDEA_TYPES = [
    "new_product",
    "new_feature",
    "feature_enhancement",
    "ai_capability",
    "workflow_automation",
    "data_product",
    "platform_capability",
    "governance_control",
    "experiment",
    "technical_enabler",
    "risk_compliance",
]

# (key, label, low, high) inclusive ranges
STATUS_THRESHOLDS = [
    ("raw", "Raw", 0, 39),
    ("needs_discovery", "Needs discovery", 40, 59),
    ("promising", "Promising", 60, 74),
    ("prd_candidate", "PRD candidate", 75, 84),
    ("delivery_ready", "Delivery-ready", 85, 100),
]

# Rubric dimension -> max points. Sum of maxes is 100.
RUBRIC_MAX = {
    "problem": 15,
    "persona": 10,
    "value": 10,
    "strategy": 10,
    "differentiation": 5,
    "feasibility": 10,
    "data_ai": 10,
    "scope": 10,
    "risk": 10,
    "execution": 10,
}


def status_for_score(score: int) -> str:
    for key, _label, low, high in STATUS_THRESHOLDS:
        if low <= score <= high:
            return key
    return "raw"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_time(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    # Example: 'Jun 5, 2026 / 09:31 AM' using the middle dot separator.
    return dt.strftime("%b %-d, %Y \u00b7 %I:%M %p")


def seeded_fraction(seed: str, salt: str) -> float:
    """Deterministic value in [0.4, 0.95] from a seed, for simulated scoring."""
    h = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    n = int(h[:8], 16) / 0xFFFFFFFF
    return 0.4 + n * 0.55
