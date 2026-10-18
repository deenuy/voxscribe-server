"""Pre-router: answers deterministic queries from the clock without an LLM call.

Patterns handled (case-insensitive, flexible phrasing):
  - "what day is it" / "what day is today" / "what's today"
  - "what's the date" / "what date is it" / "today's date"
  - "what time is it" / "what's the time"

Any of these get a one-sentence clock answer built from the context header,
with no Ollama round-trip. "And tomorrow" / follow-up phrasing is NOT caught
here — it goes to the LLM with history so C1 tests context resolution.

The caller writes a turns row with source='clock'.
"""

from __future__ import annotations

import re
from datetime import datetime


_DATE_PATTERNS = re.compile(
    r"""
    (?:
        (?:what['']?s?\s+(?:the\s+)?date)|       # what's the date, what date
        (?:what\s+date\s+is\s+it)|               # what date is it
        (?:today['']?s?\s+date)|                  # today's date
        (?:what\s+(?:is\s+)?today)               # what is today / what today
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TIME_PATTERNS = re.compile(
    r"""
    (?:
        (?:what(?:'s|\s+is)\s+(?:the\s+)?time)|  # what's the time / what is the time
        (?:what\s+time\s+is\s+it)                 # what time is it
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DAY_PATTERNS = re.compile(
    r"""
    (?:
        (?:what\s+day\s+(?:is\s+it|is\s+today))|  # what day is it / what day is today
        (?:what['']?s?\s+(?:the\s+)?day)           # what's the day / what's today's day
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def route(user_text: str, cfg) -> str | None:
    """Return a clock-sourced answer string, or None to continue to LLM.

    Only answers if the entire utterance (after stripping filler) is a
    pure date/time/day question. A sentence with any other substance
    (e.g. "and tomorrow") returns None.
    """
    text = user_text.strip().rstrip(".?!")

    # Reject if text has AND / follow-up content — let LLM handle those
    if re.search(r'\band\b', text, re.IGNORECASE):
        return None

    # Reject if longer than ~8 words — probably a compound query
    if len(text.split()) > 10:
        return None

    from app.local.context import _local_now
    tz_name = getattr(cfg, "vox_timezone", "UTC") or "UTC"
    now = _local_now(tz_name)

    if _DAY_PATTERNS.search(text):
        return _day_answer(now)
    if _DATE_PATTERNS.search(text):
        return _date_answer(now)
    if _TIME_PATTERNS.search(text):
        return _time_answer(now)
    return None


def _day_answer(now: datetime) -> str:
    return f"Today is {now.strftime('%A')}."


def _date_answer(now: datetime) -> str:
    return f"Today is {now.strftime('%A, %-d %B %Y')}."


def _time_answer(now: datetime) -> str:
    return f"It's {now.strftime('%-I:%M %p')} {now.strftime('%Z') or 'local time'}."
