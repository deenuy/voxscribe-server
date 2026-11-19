"""The first-party tools, as declarative records (TOOLS-CONNECTORS.md section 2.3).

These are the tools the engine had before the registry, registered without changing what they do:

  web.search       the thinker's search through SearXNG (delegate lane, remote network)
  reminder.create  "remind me to X at T": stored at once and announced
  timer.start      "set a timer for N minutes": a reminder row today, announced
  reminder.list    "what reminders do I have?"

reminder.create and timer.start announce today because that is what they did: the read-back that
asks first is PL45's, which raises reminder.create to `ask` and narrows classroom to timers.
Scripted lines are filled only from validated arguments and the receipt. No line but `success`
may claim the action happened (the gate's action-claim rule replaces it unless a receipt exists).
"""
from __future__ import annotations

WEB_SEARCH = "web.search"
REMINDER_CREATE = "reminder.create"
TIMER_START = "timer.start"
REMINDER_LIST = "reminder.list"

ALL_MODES = ["personal", "ideascribe", "senior_care", "classroom"]

# A UTC instant as datetime.isoformat() writes it.
UTC_INSTANT = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?\+00:00"
_INSTANT = {"type": "string", "maxLength": 40, "pattern": UTC_INSTANT,
            "description": "when it fires, UTC"}
_ROW_ID = {"type": "string", "maxLength": 64}
_LOCAL_TIME = {"type": "string", "maxLength": 32, "description": "the local time, spoken form"}

FIRST_PARTY: tuple[dict, ...] = (
    {
        "name": WEB_SEARCH,
        "summary": "Search the web for current facts, weather, news, prices or recent events.",
        "args": {
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": 500,
                                     "description": "a short question or keywords"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": 20_000},
                           "result_count": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["text", "result_count"],
            "additionalProperties": False,
        },
        "side_effect": "read",
        "confirm": "none",
        "modes": ["personal", "ideascribe"],
        "network": "remote",
        "lane": "delegate",
        "timeout_ms": 7_000,
        "templates": {
            "success": "Here's what I found.",
            "failure": "I can't look that up right now.",
            "timeout": "I couldn't finish looking that up.",
            "refused": "I can't look things up online here.",
            "invalid": "I can't look that up as asked.",
        },
        "audit": {"store": [], "hash": ["query"]},
        "untrusted_result": True,
    },
    {
        "name": REMINDER_CREATE,
        "summary": "Set a reminder that is spoken at the time asked for.",
        "args": {
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": 500,
                                    "description": "what to be reminded of"},
                           "fire_at": _INSTANT},
            "required": ["text", "fire_at"],
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {"reminder_id": _ROW_ID, "when": _LOCAL_TIME},
            "required": ["reminder_id", "when"],
            "additionalProperties": False,
        },
        "side_effect": "write",
        "confirm": "announce",
        "modes": ALL_MODES,
        "network": "none",
        "lane": "fast",
        "timeout_ms": 2_000,
        "templates": {
            "success": "Got it, I'll remind you at {when}: {text}.",
            "failure": "Sorry, I couldn't save that reminder.",
            "timeout": "I'm not sure that reminder was saved.",
            "refused": "I can't set reminders here.",
            "invalid": "I couldn't tell when to remind you.",
        },
        "audit": {"store": ["fire_at"], "hash": ["text"]},
        "untrusted_result": False,
    },
    {
        "name": TIMER_START,
        "summary": "Start a timer for a number of minutes or seconds.",
        "args": {
            "type": "object",
            "properties": {"label": {"type": "string", "maxLength": 40,
                                     "pattern": r"\d{1,6} (second|minute)s?",
                                     "description": "the duration as spoken back"},
                           "fire_at": _INSTANT},
            "required": ["label", "fire_at"],
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {"timer_id": _ROW_ID},
            "required": ["timer_id"],
            "additionalProperties": False,
        },
        "side_effect": "write",
        "confirm": "announce",
        "modes": ALL_MODES,
        "network": "none",
        "lane": "fast",
        "timeout_ms": 2_000,
        "templates": {
            "success": "Got it, timer set for {label}.",
            "failure": "Sorry, I couldn't start that timer.",
            "timeout": "I'm not sure that timer started.",
            "refused": "I can't run timers here.",
            "invalid": "I couldn't tell how long the timer should run.",
        },
        "audit": {"store": ["label", "fire_at"], "hash": []},
        "untrusted_result": False,
    },
    {
        "name": REMINDER_LIST,
        "summary": "Read out the reminders still to come.",
        "args": {"type": "object", "properties": {}, "required": [],
                 "additionalProperties": False},
        "result": {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 0, "maximum": 10_000}},
            "required": ["count"],
            "additionalProperties": False,
        },
        "side_effect": "read",
        "confirm": "none",
        "modes": ALL_MODES,
        "network": "none",
        "lane": "fast",
        "timeout_ms": 2_000,
        "templates": {
            "success": "You have {count} reminders coming up.",
            "failure": "I couldn't read your reminders just now.",
            "timeout": "I couldn't read your reminders just now.",
            "refused": "I can't read reminders here.",
            "invalid": "I couldn't read your reminders just now.",
        },
        "audit": {"store": [], "hash": []},
        "untrusted_result": False,
    },
)
