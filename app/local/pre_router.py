"""Pre-router: answers deterministic queries without an LLM call, or routes to tools.

Patterns handled (case-insensitive, flexible phrasing):

  Clock (no LLM, no tool):
    - "what day is it" / "what day is today" / "what's today"
    - "what's the date" / "what date is it" / "today's date"
    - "what time is it" / "what's the time"

  Search (tool call required, filler phrase spoken first):
    - weather, news, latest, today's <X>, price of, who won, current <X>, score of

"And tomorrow" / follow-up phrasing is NOT caught here — it goes to the LLM
with history so C1 tests context resolution.

route() returns:
  - ("clock", answer_str)  for clock queries answered from the OS clock
  - ("search", query_str)  for queries that should go to web_search
  - None                   to continue to LLM

Return is a tuple (source, value) | None. Old callers that expected str | None
are updated; use route_clock() for the clock-only path.
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


_SEARCH_PATTERNS = re.compile(
    r"""
    (?:
        weather\s+\w|               # "weather in ..."
        (?:latest|current|recent)\s+\w|  # "latest news", "current score"
        (?:today'?s?\s+\w)|         # "today's weather"
        (?:who\s+won)|              # "who won the game"
        (?:price\s+of)|             # "price of gold"
        (?:what'?s?\s+(?:the\s+)?(?:score|news|weather|temperature))|
        (?:how\s+(?:is|are)\s+(?:the\s+)?(?:market|stocks|weather))
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def route(user_text: str, cfg) -> tuple[str, str] | None:
    """Return (source, value) or None.

    ("clock", answer) — answered from OS clock, no LLM call.
    ("search", query) — should be routed to web_search tool.
    None — continue to LLM unchanged.
    """
    text = user_text.strip().rstrip(".?!")

    # Clock path: short pure date/time/day questions only
    if not re.search(r'\band\b', text, re.IGNORECASE) and len(text.split()) <= 10:
        from app.local.context import _local_now
        tz_name = getattr(cfg, "vox_timezone", "UTC") or "UTC"
        now = _local_now(tz_name)

        if _DAY_PATTERNS.search(text):
            return ("clock", _day_answer(now))
        if _DATE_PATTERNS.search(text):
            return ("clock", _date_answer(now))
        if _TIME_PATTERNS.search(text):
            return ("clock", _time_answer(now))

    # Search path: queries about current facts, weather, news
    if _SEARCH_PATTERNS.search(text):
        return ("search", text)

    return None


def route_clock(user_text: str, cfg) -> str | None:
    """Legacy helper: returns clock answer str or None (no search routing)."""
    result = route(user_text, cfg)
    if result and result[0] == "clock":
        return result[1]
    return None


def _day_answer(now: datetime) -> str:
    return f"Today is {now.strftime('%A')}."


def _date_answer(now: datetime) -> str:
    return f"Today is {now.strftime('%A, %-d %B %Y')}."


def _time_answer(now: datetime) -> str:
    return f"It's {now.strftime('%-I:%M %p')} {now.strftime('%Z') or 'local time'}."
