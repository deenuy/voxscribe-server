"""What is said, and shown, when a reminder or a timer comes due (TOOLS-CONNECTORS.md 2.7).

Fixed templates filled from the stored row: the reminder's words, a timer's length, and the due
time in words (voice/spoken_time.py). A timer and a reminder are announced differently:

    due        "Reminder: stretch."                       "Your one minute timer is done."
    missed     "While I was off, you had a reminder today at three in the afternoon: call the
               dentist."                                  "While I was off, your ten minute
                                                          timer ended."
    owed       "Earlier, you had a reminder ..."           "Earlier, your ten minute timer ended."
    neutral    "You have a reminder you set for eight in the evening."

`missed` is for an item that came due while the engine was not running; `owed` for one that was
shown as a notification only (quiet hours, another person's device) and is spoken at the start
of the next session. The neutral line names only the time: it is spoken when the gate refuses a
reminder's own words (senior care's dosing rule, TC6), so those words are never heard.

Every line is a candidate like any other and passes the gate before it is spoken or shown.
Pure: no clock is read; `now` is passed in.
"""
from __future__ import annotations

from datetime import datetime

from app.local.voice import reminder_store as rs
from app.local.voice import spoken_time as st

DUE_REMINDER = "Reminder: {text}."
DUE_TIMER = "Your {length} timer is done."
MISSED_REMINDER = "While I was off, you had a reminder {when}: {text}."
MISSED_TIMER = "While I was off, your {length} timer ended."
OWED_REMINDER = "Earlier, you had a reminder {when}: {text}."
OWED_TIMER = "Earlier, your {length} timer ended."
NEUTRAL_DUE = "You have a reminder you set for {clock}."
NEUTRAL_MISSED = "While I was off, you had a reminder {when}."
NEUTRAL_OWED = "Earlier, you had a reminder {when}."

DUE, MISSED, OWED = "due", "missed", "owed"


def _length(item: rs.Item) -> str:
    return (item.label or "").strip().lower() or "timer's"


def line(item: rs.Item, *, now: datetime, how: str = DUE) -> str:
    """The line for `item` delivered `how` (due, missed or owed)."""
    if item.kind == rs.KIND_TIMER:
        template = {DUE: DUE_TIMER, MISSED: MISSED_TIMER, OWED: OWED_TIMER}[how]
        return template.format(length=_length(item)).replace("your timer's timer", "your timer")
    if not item.text:
        return neutral(item, now=now, how=how)
    if how == DUE:
        return DUE_REMINDER.format(text=item.text)
    when = st.say_moment(item.fire_at, now=now, tz=item.tz)
    template = MISSED_REMINDER if how == MISSED else OWED_REMINDER
    return template.format(when=when, text=item.text)


def neutral(item: rs.Item, *, now: datetime, how: str = DUE) -> str:
    """The line that names only the time, for a reminder whose words the gate refuses."""
    if how == DUE:
        return NEUTRAL_DUE.format(clock=st.say_clock(item.fire_at, tz=item.tz))
    when = st.say_moment(item.fire_at, now=now, tz=item.tz)
    return (NEUTRAL_MISSED if how == MISSED else NEUTRAL_OWED).format(when=when)


def due_line(item: rs.Item, *, now: datetime) -> str:
    return line(item, now=now, how=DUE)


def neutral_line(item: rs.Item, *, now: datetime) -> str:
    return neutral(item, now=now, how=DUE)
