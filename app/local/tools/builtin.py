"""The first-party tools, as declarative records (TOOLS-CONNECTORS.md section 2.3).

  web.search       the thinker's search through SearXNG (delegate lane, remote network)
  reminder.create  "remind me to X at T": read back, set only on the next turn's yes (ask)
  reminder.list    "what reminders do I have?"
  reminder.cancel  "cancel my reminder to X": read back, cancelled only on a yes (ask)
  timer.start      "set a timer for ten minutes": started and announced at once (announce)
  timer.status     "how long is left on my timer?"
  timer.cancel     "cancel the timer": cancelled and announced at once (announce)

Product modes (TOOLS-CONNECTORS.md section 6). Classroom has timers only (TC12): no reminder
about a child is stored, so the reminder tools are not offered there and a direct call is
refused with the timers-only line. Senior care has reminders and timers; a reminder whose words
the senior care gate refuses (the dosing rule) is never proposed (TC6, voice/reminder_dialog.py).
Both rules are flagged for review: children's data (TC12) and clinician review (TC6).

Arguments are what voice/spoken_time.py made of the person's words: an instant in UTC, the zone
it was said in, and the words the read-back speaks (`when`, `length`), which carry nothing the
instant does not. A reminder's own words (`text`) are kept on the audit row only as a digest and
a length. Scripted lines are filled only from validated arguments and the receipt; no line but
`success` may claim the action happened (the gate's action-claim rule replaces it unless this
turn has the receipt).
"""
from __future__ import annotations

WEB_SEARCH = "web.search"
REMINDER_CREATE = "reminder.create"
REMINDER_LIST = "reminder.list"
REMINDER_CANCEL = "reminder.cancel"
TIMER_START = "timer.start"
TIMER_STATUS = "timer.status"
TIMER_CANCEL = "timer.cancel"

ALL_MODES = ["personal", "ideascribe", "senior_care", "classroom"]
REMINDER_MODES = ["personal", "ideascribe", "senior_care"]       # classroom: timers only (TC12)

MAX_TIMER_S = 24 * 3600              # spoken_time.MAX_TIMER_DURATION
MAX_SUBJECT = 200                    # spoken_time.MAX_SUBJECT_CHARS

# A UTC instant as datetime.isoformat() writes it.
UTC_INSTANT = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?\+00:00"
_INSTANT = {"type": "string", "maxLength": 40, "pattern": UTC_INSTANT,
            "description": "when it fires, UTC"}
_ROW_ID = {"type": "string", "maxLength": 64}
_ZONE = {"type": "string", "maxLength": 64, "pattern": r"[A-Za-z][A-Za-z0-9_+\-/]*",
         "description": "the IANA zone the time was said in"}
# Words a synthesizer says: letters, spaces, commas, hyphens and apostrophes. No digits.
_WORDS = r"[A-Za-z][A-Za-z ,'\-]*"
_WHEN = {"type": "string", "maxLength": 120, "pattern": _WORDS,
         "description": "the moment in words, as the read-back says it"}
_SUBJECT = {"type": "string", "maxLength": MAX_SUBJECT, "pattern": r"[^\r\n]+",
            "description": "what to be reminded of, in the person's words"}
_LENGTH = {"type": "string", "maxLength": 60, "pattern": _WORDS,
           "description": "the timer's length as said before 'timer'"}
_COUNT = {"type": "integer", "minimum": 0, "maximum": 10_000}
_NO_ARGS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

TIMERS_ONLY_LINE = "I can only set timers here."
REMINDER_REFUSED = TIMERS_ONLY_LINE

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
        "summary": "Set a reminder that is spoken and shown at the time asked for.",
        "args": {
            "type": "object",
            "properties": {"text": _SUBJECT, "fire_at": _INSTANT, "when": _WHEN,
                           "time_zone": _ZONE},
            "required": ["text", "fire_at", "when", "time_zone"],
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {"reminder_id": _ROW_ID},
            "required": ["reminder_id"],
            "additionalProperties": False,
        },
        "side_effect": "write",
        "confirm": "ask",
        "modes": REMINDER_MODES,
        "network": "none",
        "lane": "fast",
        "timeout_ms": 2_000,
        "templates": {
            # spoken_time.READ_BACK_REMINDER, filled from the same validated value.
            "readback": "{when}, {text}. Shall I set it?",
            "success": "Reminder set.",
            "failure": "Sorry, I couldn't save that reminder.",
            "timeout": "I'm not sure that reminder was saved.",
            "refused": REMINDER_REFUSED,
            "invalid": "I couldn't set that reminder as asked.",
        },
        "audit": {"store": ["fire_at", "when", "time_zone"], "hash": ["text"]},
        "untrusted_result": False,
    },
    {
        "name": REMINDER_LIST,
        "summary": "Read out the reminders still to come.",
        "args": _NO_ARGS,
        "result": {
            "type": "object",
            "properties": {"count": _COUNT},
            "required": ["count"],
            "additionalProperties": False,
        },
        "side_effect": "read",
        "confirm": "none",
        "modes": REMINDER_MODES,
        "network": "none",
        "lane": "fast",
        "timeout_ms": 2_000,
        "templates": {
            "success": "You have {count} reminders coming up.",
            "failure": "I couldn't read your reminders just now.",
            "timeout": "I couldn't read your reminders just now.",
            "refused": REMINDER_REFUSED,
            "invalid": "I couldn't read your reminders just now.",
        },
        "audit": {"store": [], "hash": []},
        "untrusted_result": False,
    },
    {
        "name": REMINDER_CANCEL,
        "summary": "Cancel one reminder that is still to come.",
        "args": {
            "type": "object",
            "properties": {"reminder_id": _ROW_ID, "when": _WHEN, "text": _SUBJECT},
            "required": ["reminder_id", "when", "text"],
            "additionalProperties": False,
        },
        "result": {
            "type": "object",
            "properties": {"reminder_id": _ROW_ID},
            "required": ["reminder_id"],
            "additionalProperties": False,
        },
        "side_effect": "write",
        "confirm": "ask",
        "modes": REMINDER_MODES,
        "network": "none",
        "lane": "fast",
        "timeout_ms": 2_000,
        "templates": {
            "readback": "Cancel your reminder for {when}, {text}?",
            "success": "Okay, that reminder is cancelled.",
            "failure": "Sorry, I couldn't cancel that reminder.",
            "timeout": "I'm not sure that reminder was cancelled.",
            "refused": REMINDER_REFUSED,
            "invalid": "I couldn't tell which reminder to cancel.",
        },
        "audit": {"store": ["reminder_id", "when"], "hash": ["text"]},
        "untrusted_result": False,
    },
    {
        "name": TIMER_START,
        "summary": "Start a timer for a length of time, announced at once.",
        "args": {
            "type": "object",
            "properties": {"duration_s": {"type": "integer", "minimum": 1,
                                          "maximum": MAX_TIMER_S},
                           "fire_at": _INSTANT, "length": _LENGTH, "time_zone": _ZONE},
            "required": ["duration_s", "fire_at", "length", "time_zone"],
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
            # spoken_time.ANNOUNCE_TIMER for a whole number of hours, minutes or seconds.
            "success": "{length} timer, starting now.",
            "failure": "Sorry, I couldn't start that timer.",
            "timeout": "I'm not sure that timer started.",
            "refused": "I can't run timers here.",
            "invalid": "I couldn't tell how long the timer should run.",
        },
        "audit": {"store": ["duration_s", "fire_at", "length", "time_zone"], "hash": []},
        "untrusted_result": False,
    },
    {
        "name": TIMER_STATUS,
        "summary": "Say how long is left on the timers that are running.",
        "args": _NO_ARGS,
        "result": {
            "type": "object",
            "properties": {"count": _COUNT},
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
            "success": "You have {count} timers running.",
            "failure": "I couldn't check your timers just now.",
            "timeout": "I couldn't check your timers just now.",
            "refused": "I can't run timers here.",
            "invalid": "I couldn't check your timers just now.",
        },
        "audit": {"store": [], "hash": []},
        "untrusted_result": False,
    },
    {
        "name": TIMER_CANCEL,
        "summary": "Cancel a running timer, announced at once.",
        "args": {
            "type": "object",
            "properties": {"timer_id": _ROW_ID, "length": _LENGTH},
            "required": ["timer_id", "length"],
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
            "success": "{length} timer cancelled.",
            "failure": "Sorry, I couldn't cancel that timer.",
            "timeout": "I'm not sure that timer was cancelled.",
            "refused": "I can't run timers here.",
            "invalid": "I couldn't tell which timer to cancel.",
        },
        "audit": {"store": ["timer_id", "length"], "hash": []},
        "untrusted_result": False,
    },
)
