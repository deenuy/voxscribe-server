"""Spoken times, durations and day ranges: what a person said aloud, read back in words (PL44).

Pure: no model, no database, no route, no logging of what was said, and no hidden clock. Every
function that needs the time takes `now` (an aware datetime) and `tz` (a tzinfo such as
`ZoneInfo("America/Toronto")`). Reminders and timers use it (voice/reminder_dialog.py, PL45); the
calendar answers (PL48).

Public entry points

    parse_when(text, now=, tz=)       a moment for a reminder           -> TimeParse
    parse_duration(text, now=, tz=)   a length of time for a timer      -> TimeParse
    correct(previous, text, now=)     "no, eight" said after a read-back -> TimeParse | None
    parse_day_range(text, now=, tz=)  "today", "tomorrow afternoon"     -> DayRange | None
    say_when(value) / say_duration(d) words a synthesizer says without digits
    read_back_reminder(value, subject) / announce_timer(value)   the scripted lines
    reminder_subject(remainder)       "remind me to call the dentist" -> Subject
    propose / reply / silence         the read-back state machine of design section 2.5
    classify_reply(text)              yes, no or neither, for a question that holds no time
    say_moment / say_clock            a stored moment in words (lists, due and missed lines)
    timer_length(d) / is_recurring    "ten minute" before "timer"; "every day at eight"

A TimeParse holds either a validated SpokenTime or a Refusal with its reason code, plus the
character spans of the time expression and the remainder of the text with those spans removed,
so "remind me tomorrow at 8 to call the dentist" leaves "remind me to call the dentist" (the
"to" is kept) and reminder_subject() gives "call the dentist" with its connector "to".
`question` is the one scripted clarifying question to ask, or None when nothing needs asking.

Supported forms (English). Anything outside this list is reported as not understood, never
guessed.

    relative     in two minutes, in 2 mins, in an hour, in an hour and a half, in half an hour,
                 in a quarter of an hour, in one and a half hours, in 1.5 hours, in 90 seconds,
                 in two hours and fifteen minutes, in 2 hours 30 minutes, ten minutes from now,
                 in two hours or so (taken as two hours),
                 in ten (a bare number from 1 to BARE_MINUTES_MAX, taken as minutes)
    durations    for ten minutes, ten minutes, a 45 minute timer, 45-minute, for an hour,
                 for a minute and a half, for ten (bare number, minutes)          [timers only]
    clock        at 8, at 8 30, at 8:30, at 8.30, at eight thirty, at eight oh five, 8am, 8 pm,
                 8 a.m., eight o'clock, half past eight, quarter past eight, quarter to nine,
                 quarter of nine, twenty to nine, ten after six, noon, midday, midnight,
                 twelve noon, 20:00; "at" is needed for a bare hour ("at 8"), not for a clock
                 that carries am, pm, o'clock, a period or a past/to form
    periods      in the morning, in the afternoon, in the evening, at night, tonight,
                 this morning, this afternoon, this evening, tomorrow morning, Friday night
    days         today, tonight, tomorrow, the day after tomorrow, in three days, in a week,
                 on Friday, Friday (next to a time), this Friday, next Friday,
                 Friday the twenty-first, on the 21st, on the twenty-first of November,
                 November 21, 21 November
    corrections  "<time>, no, <time or fragment>" with markers: no, nope, nah, sorry, I mean,
                 I meant, actually, make that, make it, rather, or rather, wait, no wait,
                 scratch that, correction; and a restatement with no marker ("at eight, at
                 eight thirty"). A fragment is the part that changed: "nine", "three minutes",
                 "6 pm", "Friday", "the 22nd", "in the evening".
    day ranges   today, tonight, this morning|afternoon|evening, tomorrow [period],
                 the day after tomorrow, [on|this|next] Friday [period], on the 21st,
                 this week, next week, this weekend

Not supported (reported as not understood): vague amounts (a few minutes, a couple of hours,
in a bit), approximate times (around eight, about eight, by eight), "in N days" without a time
of day (asked as "What time on ...?"), recurring times (every day at eight), years, months
without a day, and any second time that is neither a correction nor the other half of a day
and a clock ("at 5 or 6", "at 5 to call mom at 6").

Self-correction. The later value wins, the result says `corrected`, and each Correction keeps
the text that was superseded, so the trace still shows what was heard (voice design: a
thinking-repeat anchors a self-correction and is never deleted). A fragment changes only what it
names: "tomorrow at eight, sorry, nine" keeps tomorrow; "at 8 pm, no, 9" keeps the evening;
"in two minutes, no, three" keeps minutes.

Ambiguity: reported in `ambiguities`, resolved by the stated default below, with the other
readings in `alternatives` so the caller can ask the scripted question or read back the default.

    MERIDIEM               "at 8" with no am, pm or period: the next occurrence of 8 (on the
                           named day, the earlier of the two); flagged only when both readings
                           are still possible. A bare 12 means noon, or the midnight that ends
                           the day.
    PAST_TODAY             no day said and the time has passed today: tomorrow. With "today"
                           or "tonight" said, a passed time is refused (IN_THE_PAST).
    SAME_WEEKDAY           "next Friday" said on a Friday: a week from today. "on Friday" said
                           on a Friday: today if the time is still ahead, else a week from today.
    NEXT_WEEKDAY           "next Friday" when the coming Friday is in the same week as today
                           (weeks start on Monday): the coming Friday; the week after is the
                           alternative. "this Friday" is never flagged.
    MIDNIGHT               "tomorrow at midnight", "on Friday at midnight": the midnight that
                           ends that day; the one that starts it is the alternative.
    TOMORROW_AFTER_MIDNIGHT "tomorrow" said before EARLY_HOURS_END: the calendar tomorrow;
                           later today is the alternative.
    SKIPPED_WALL_TIME      a wall-clock time that does not exist (spring forward): moved
                           forward by the gap (2:30 becomes 3:30).
    REPEATED_WALL_TIME     a wall-clock time that occurs twice (fall back): the first one.
    UNIT_ASSUMED           "in ten", "for ten": minutes.

Calendar rules. A named day keeps its wall-clock time across a daylight-saving change ("tomorrow
at nine" is nine on tomorrow's clock). A relative amount in hours, minutes or seconds is elapsed
time ("in two hours" across the change is two real hours). "in N days" is N calendar days.
Months and years roll over by calendar arithmetic, never by adding 30 days. Clock times are to
the minute; a relative amount keeps the seconds of `now`.

Bounds, refused with a reason code: a moment at or before now (IN_THE_PAST), a moment further
ahead than MAX_REMINDER_AHEAD (TOO_FAR_AHEAD), a duration of zero (ZERO_DURATION) or longer
than MAX_TIMER_DURATION (DURATION_TOO_LONG). A date that does not exist ("February 30",
"Friday the 22nd" when the 22nd is not a Friday, "on the 30th" said on 31 January) is
INVALID_DATE: a day of the month is this month's or next month's, a month and day is the next
one within a year, and none of them is ever moved to a nearby date. Callers may pass a stricter
limit.

Read-back. Words only, no digits, no bare hour: "in three minutes", "tomorrow at eight in the
morning", "on Friday the twenty-first at noon", "on Monday the seventh of December at nine at
night". Periods: up to 11:59 "in the morning", 12:00 "noon", until 17:59 "in the afternoon",
until 20:59 "in the evening", from 21:00 "at night". The coming midnight is "tonight at
midnight"; a later one is "tomorrow at midnight, at the end of the day". Before EARLY_HOURS_END
the next day is named by weekday and date, never "tomorrow". Every accepted parse round-trips:
parsing its read-back with the same now gives the same instant, and the only ambiguity the
read-back can leave is REPEATED_WALL_TIME (words cannot tell the two 1:30s apart; the first is
kept). The read-back functions accept only a SpokenTime made by this module's parser and check
its bounds again, so a read-back can be built only from validated arguments, never from model
text. Compare instants with `.astimezone(UTC)`: Python compares two datetimes in the same zone by
wall clock, which ignores the fold.

A second language needs: its own lexicon tables (number and ordinal words, units, weekdays,
months, period words, correction markers, yes and no words, the subject's lead phrases), grammar
functions where word order differs (French "dans deux minutes", "huit heures moins le quart",
"à vingt heures"), the clock convention its speakers use (many use the 24-hour clock aloud, so
MERIDIEM rarely applies), its own period boundaries, its own read-back templates and number
words, and its own fixture table. The English tables are kept at the top of this module for that
reason; nothing else in the module is English-specific except the templates.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta, tzinfo
from enum import Enum

from app.local.voice.delegate_lane import is_affirmative

# ------------------------------------------------------------------ bounds and defaults

MAX_REMINDER_AHEAD = timedelta(days=365)   # a month and day read back without a year stay clear
MAX_TIMER_DURATION = timedelta(hours=24)
BARE_MINUTES_MAX = 60                      # "in ten" is minutes up to this; larger is not a time
EARLY_HOURS_END = 4                        # "tomorrow" said before 4 am may mean later today
READBACK_WINDOW = timedelta(seconds=8)     # silence longer than this after a read-back declines
MAX_SUBJECT_CHARS = 200

# ------------------------------------------------------------------ scripted lines

QUESTION_WHEN = "Sorry, what time should I set it for?"
QUESTION_DURATION = "Sorry, how long should the timer be?"
QUESTION_TIME_ON_DAY = "What time {day}?"
QUESTION_IN_THE_PAST = "That time has already passed. What time should I set it for?"
QUESTION_TOO_FAR = "That's further ahead than I can set a reminder. What time should I set it for?"
QUESTION_TOO_LONG = "The longest timer I can set is {limit}. How long should it be?"
QUESTION_ZERO = "How long should the timer be?"
QUESTION_INVALID_DATE = "I couldn't find that date. Which day do you mean?"
QUESTION_CONFLICTING = "I heard more than one time. Which time should I set it for?"
QUESTION_CHOICE = "Do you mean {first}, or {second}?"
QUESTION_SUBJECT = "What should I remind you about?"
READ_BACK_REMINDER = "{when}, {subject}. Shall I set it?"
READ_BACK_REMINDER_NO_SUBJECT = "{when}. Shall I set it?"
ANNOUNCE_TIMER = "{length} timer, starting now."
ANNOUNCE_TIMER_LONG = "Timer for {length}, starting now."

# ------------------------------------------------------------------ English lexicon

_UNIT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
               "eight": 8, "nine": 9}
_TEEN_WORDS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
               "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
               "seventy": 70, "eighty": 80, "ninety": 90}
_ORD_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
              "seventh": 7, "eighth": 8, "ninth": 9}
_ORD_OTHER = {"tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
              "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
              "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
_ORD_SUFFIXES = frozenset({"st", "nd", "rd", "th"})
_DUR_UNITS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1,
              "minute": 60, "minutes": 60, "min": 60, "mins": 60,
              "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600}
_DAY_UNITS = {"day": 1, "days": 1, "week": 7, "weeks": 7}
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
             "saturday": 5, "sunday": 6}
_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
           "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
           "december": 12}
_PERIOD_WORDS = frozenset({"morning", "afternoon", "evening", "night"})
_FILLERS = frozenset({"uh", "um", "umm", "uhm", "er", "erm", "ah", "hmm", "oh"})
_MARKERS = (
    ("or", "rather"), ("i", "mean"), ("i", "meant"), ("make", "that"), ("make", "it"),
    ("scratch", "that"), ("no",), ("nope",), ("nah",), ("sorry",), ("actually",),
    ("rather",), ("wait",), ("correction",),
)
# Leads a reply to a read-back may open with before the corrected time.
_REPLY_LEADS = _MARKERS + (
    ("change", "it", "to"), ("change", "that", "to"), ("set", "it", "for"), ("set", "it", "to"),
    ("yes",), ("yeah",), ("okay",), ("ok",), ("but",), ("how", "about"), ("what", "about"),
)
_REPLY_TRAILERS = frozenset({"please", "instead", "then", "thanks", "thank", "you", "ok",
                             "okay"})
_NO_WORDS = frozenset({"no", "nope", "nah", "cancel", "don't", "dont", "stop"})
_NO_PHRASES = ("never mind", "nevermind", "not now", "forget it", "no thanks", "do not")
_YES_PHRASES = ("set it", "that's right", "thats right", "correct", "sounds good", "please do")
_YES_VOCABULARY = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please", "go", "ahead", "do", "it",
    "set", "that's", "thats", "right", "correct", "sounds", "good", "thanks", "thank", "you",
    "great", "perfect", "uh", "um",
})
_RECURRING_WORDS = frozenset({"daily", "hourly", "weekly", "nightly", "weekdays", "weekends"})
_SUBJECT_LEAD_RE = re.compile(
    r"^\s*(?:hey\s+vox\b[\s,]*)?(?:(?:please|can you|could you|would you|will you)\s+)*"
    r"(?:remind\s+me|set\s+(?:me\s+)?(?:a\s+)?reminder|make\s+a\s+reminder|reminder)\b"
    r"\s*(?P<conn>to|about|that|for|of)?\b\s*",
    re.IGNORECASE,
)

_PERIOD_HOURS = {"morning": (5, 12), "afternoon": (12, 18), "evening": (18, 24),
                 "night": (18, 29)}

# ------------------------------------------------------------------ result types


class Kind(Enum):
    INSTANT = "instant"      # a moment, for a reminder
    DURATION = "duration"    # a length of time, for a timer


class Ambiguity(Enum):
    MERIDIEM = "meridiem"
    PAST_TODAY = "past_today"
    SAME_WEEKDAY = "same_weekday"
    NEXT_WEEKDAY = "next_weekday"
    MIDNIGHT = "midnight"
    TOMORROW_AFTER_MIDNIGHT = "tomorrow_after_midnight"
    SKIPPED_WALL_TIME = "skipped_wall_time"
    REPEATED_WALL_TIME = "repeated_wall_time"
    UNIT_ASSUMED = "unit_assumed"


class Refusal(Enum):
    NOT_UNDERSTOOD = "not_understood"
    CONFLICTING = "conflicting_times"
    NO_TIME_OF_DAY = "no_time_of_day"
    INVALID_DATE = "invalid_date"
    IN_THE_PAST = "in_the_past"
    TOO_FAR_AHEAD = "too_far_ahead"
    ZERO_DURATION = "zero_duration"
    DURATION_TOO_LONG = "duration_too_long"


@dataclass(frozen=True)
class Correction:
    """One self-correction: what was superseded and what replaced it. `marker` is None for a
    restatement with no correction word."""
    superseded: str
    replacement: str
    marker: str | None


@dataclass(frozen=True)
class Subject:
    """What a reminder is about, and the word that joins it to a rebuilt sentence:
    "I'll remind you {connector} {text}"."""
    text: str
    connector: str = "to"


_SEAL = object()


@dataclass(frozen=True)
class SpokenTime:
    """A validated time. Made only by this module's parsers; the read-back refuses any other.

    `instant` is aware, in `tz`: the reminder's moment, or when a timer started at `now` ends.
    `duration` is set for a timer and for a relative moment ("in three minutes").
    """
    kind: Kind
    instant: datetime
    duration: timedelta | None
    relative: bool
    ambiguities: tuple[Ambiguity, ...]
    alternatives: tuple[SpokenTime, ...]
    corrected: bool
    corrections: tuple[Correction, ...]
    now: datetime
    tz: tzinfo
    limit: timedelta
    _expr: _Expr = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class TimeParse:
    """The outcome of a parse. Exactly one of `value` and `refusal` is set."""
    text: str
    value: SpokenTime | None
    refusal: Refusal | None
    spans: tuple[tuple[int, int], ...]
    remainder: str
    question: str | None

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def span(self) -> tuple[int, int] | None:
        """From the start of the first time expression to the end of the last."""
        if not self.spans:
            return None
        return self.spans[0][0], self.spans[-1][1]


@dataclass(frozen=True)
class DayRange:
    """A stretch of calendar time: [start, end), aware, in the given zone."""
    start: datetime
    end: datetime
    words: str
    span: tuple[int, int]
    ambiguities: tuple[Ambiguity, ...] = ()


# ------------------------------------------------------------------ tokens


@dataclass(frozen=True)
class _Tok:
    kind: str    # word, int, dec, colon, punct
    text: str    # lower case
    start: int
    end: int


_TOKEN_RE = re.compile(
    r"(?P<ampm>(?<![a-z])[ap]\.\s?m\.?(?![a-z]))"
    r"|(?P<colon>\d{1,2}:\d{2})"
    r"|(?P<dec>\d+\.\d+)"
    r"|(?P<int>\d+)"
    r"|(?P<oclock>o['’]?\s?clock)"
    r"|(?P<word>[a-z]+(?:['’][a-z]+)?)"
    r"|(?P<punct>[,;!?.])",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[_Tok]:
    toks = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup or "word"
        raw = m.group().lower().replace("’", "'")
        if kind == "ampm":
            kind, raw = "word", raw[0] + "m"
        elif kind == "oclock":
            kind, raw = "word", "oclock"
        toks.append(_Tok(kind, raw, m.start(), m.end()))
    return toks


def _w(toks: list[_Tok], i: int) -> str:
    return toks[i].text if 0 <= i < len(toks) else ""


def _kind(toks: list[_Tok], i: int) -> str:
    return toks[i].kind if 0 <= i < len(toks) else ""


def _is_gap(tok: _Tok) -> bool:
    return tok.kind == "punct" or tok.text in _FILLERS


def _skip_gap(toks: list[_Tok], i: int) -> int:
    while i < len(toks) and _is_gap(toks[i]):
        i += 1
    return i


def _only_gap(toks: list[_Tok], i: int, j: int) -> bool:
    return all(_is_gap(t) for t in toks[i:j])


def _seq_at(toks: list[_Tok], i: int, seqs: tuple[tuple[str, ...], ...]) -> int | None:
    for seq in seqs:
        if all(_w(toks, i + k) == s for k, s in enumerate(seq)):
            return i + len(seq)
    return None


# ------------------------------------------------------------------ numbers


def _small_number(toks: list[_Tok], i: int) -> tuple[int, int] | None:
    """0 to 99 in words: seven, fifteen, forty five, forty-five."""
    w = _w(toks, i)
    if w == "zero":
        return 0, i + 1
    if w in _UNIT_WORDS:
        return _UNIT_WORDS[w], i + 1
    if w in _TEEN_WORDS:
        return _TEEN_WORDS[w], i + 1
    if w in _TENS_WORDS:
        if _w(toks, i + 1) in _UNIT_WORDS:
            return _TENS_WORDS[w] + _UNIT_WORDS[_w(toks, i + 1)], i + 2
        return _TENS_WORDS[w], i + 1
    return None


def _word_number(toks: list[_Tok], i: int, article: bool) -> tuple[int, int] | None:
    """A number in words below a million: "one hundred and twenty", "a hundred"."""
    total, j, found = 0, i, False
    for scale, word in ((1000, "thousand"), (100, "hundred")):
        if article and not found and _w(toks, j) in ("a", "an") and _w(toks, j + 1) == word:
            total, j, found = scale, j + 2, True
            continue
        head = _below_thousand(toks, j) if scale == 1000 else _small_number(toks, j)
        if head and _w(toks, head[1]) == word and head[0] > 0:
            total += head[0] * scale
            j, found = head[1] + 1, True
    if found:
        k = j + 1 if _w(toks, j) == "and" else j
        rest = _small_number(toks, k)
        if rest:
            return total + rest[0], rest[1]
        return total, j
    small = _small_number(toks, i)
    if small:
        return small
    if article and _w(toks, i) in ("a", "an"):
        return 1, i + 1
    return None


def _below_thousand(toks: list[_Tok], i: int) -> tuple[int, int] | None:
    head = _small_number(toks, i)
    if head and _w(toks, head[1]) == "hundred":
        k = head[1] + 1
        k = k + 1 if _w(toks, k) == "and" else k
        rest = _small_number(toks, k)
        if rest:
            return head[0] * 100 + rest[0], rest[1]
        return head[0] * 100, head[1] + 1
    return head


def _number(toks: list[_Tok], i: int, article: bool = False) -> tuple[float, int] | None:
    kind = _kind(toks, i)
    if kind == "int":
        return int(toks[i].text), i + 1
    if kind == "dec":
        return float(toks[i].text), i + 1
    return _word_number(toks, i, article)


def _ordinal(toks: list[_Tok], i: int) -> tuple[int, int] | None:
    """A day of the month said as an ordinal: 21st, twenty-first, thirtieth."""
    w = _w(toks, i)
    if _kind(toks, i) == "int" and _w(toks, i + 1) in _ORD_SUFFIXES:
        v = int(w)
        return (v, i + 2) if 1 <= v <= 31 else None
    if w in _ORD_UNITS:
        return _ORD_UNITS[w], i + 1
    if w in _ORD_OTHER:
        return _ORD_OTHER[w], i + 1
    if w in ("twenty", "thirty") and _w(toks, i + 1) in _ORD_UNITS:
        v = _TENS_WORDS[w] + _ORD_UNITS[_w(toks, i + 1)]
        return (v, i + 2) if v <= 31 else None
    return None


def _dom(toks: list[_Tok], i: int) -> tuple[int, int] | None:
    """A day of the month: an ordinal, or a plain 1 to 31 in digits."""
    o = _ordinal(toks, i)
    if o:
        return o
    if _kind(toks, i) == "int" and 1 <= int(toks[i].text) <= 31:
        return int(toks[i].text), i + 1
    return None


# ------------------------------------------------------------------ grammar pieces


@dataclass(frozen=True)
class _Clock:
    hour: int                 # as said, 0 to 23
    minute: int
    meridiem: str | None      # am, pm
    named: str | None         # noon, midnight
    period: str | None        # morning, afternoon, evening, night
    simple: bool = False      # a bare hour: usable as a count in a correction
    end_of_day: bool = False  # midnight said as "at the end of the day"


_END_OF_DAY = ("at", "the", "end", "of", "the", "day")
_MIDNIGHT_END_WORDS = "at midnight, at the end of the day"


@dataclass(frozen=True)
class _Day:
    offset: int | None = None       # days from today
    weekday: int | None = None      # 0 is Monday
    qualifier: str | None = None    # this, next
    month: int | None = None
    dom: int | None = None
    period: str | None = None
    word: str | None = None         # today, tonight, tomorrow
    weak: bool = False              # a bare weekday: kept only next to another time


@dataclass(frozen=True)
class _Comp:
    kind: str                       # rel, day, clock, period, frag
    start: int
    end: int
    rel: int | None = None          # seconds, for rel and duration
    unit: int | None = None         # seconds in the largest unit said
    assumed: bool = False
    day: _Day | None = None
    clock: _Clock | None = None
    period: str | None = None
    count: int | None = None        # frag: a bare number
    dom: int | None = None          # frag: "the 22nd"
    corrects: bool = False
    adjacent: bool = False
    marker: str | None = None


@dataclass(frozen=True)
class _Expr:
    rel: int | None = None
    unit: int | None = None
    assumed: bool = False
    day: _Day | None = None
    clock: _Clock | None = None
    period: str | None = None


def _dur_part(toks: list[_Tok], i: int) -> tuple[int, int, int] | None:
    """One "count unit" piece of a duration: (seconds, unit seconds, next index)."""
    j = i
    if _w(toks, j) in ("a", "an") and _w(toks, j + 1) in ("half", "quarter"):
        j += 1
    if _w(toks, j) == "half":
        k = j + 1
        k = k + 1 if _w(toks, k) in ("a", "an") else k
        return (1800, 3600, k + 1) if _w(toks, k) == "hour" else None
    if _w(toks, j) == "quarter":
        k = j + 1
        k = k + 1 if _w(toks, k) == "of" else k
        k = k + 1 if _w(toks, k) in ("a", "an") else k
        return (900, 3600, k + 1) if _w(toks, k) == "hour" else None
    n = _number(toks, i, article=True)
    if n is None:
        return None
    value, k = n
    half = ("and", "a", "half")
    if _seq_at(toks, k, (half,)) and _w(toks, k + 3) in _DUR_UNITS:
        value, k = value + 0.5, k + 3
    unit = _DUR_UNITS.get(_w(toks, k))
    if unit is None:
        return None
    k += 1
    if _seq_at(toks, k, (half,)) and value == int(value):
        value, k = value + 0.5, k + 3
    seconds = value * unit
    if seconds != int(seconds):
        return None
    return int(seconds), unit, k


def _duration(toks: list[_Tok], i: int) -> tuple[int, int, int] | None:
    """A duration: (total seconds, largest unit seconds, next index)."""
    total, largest, last, j, k = 0, None, None, i, i
    while True:
        part = _dur_part(toks, k)
        if part is None:
            break
        secs, unit, k2 = part
        if last is not None and unit >= last:
            break
        total, last, j = total + secs, unit, k2
        largest = largest or unit
        k = j + 1 if (_w(toks, j) == "and" or _w(toks, j) == ",") else j
    return (total, largest, j) if largest else None


def _bare_count(toks: list[_Tok], i: int) -> tuple[int, int] | None:
    """A bare 1 to BARE_MINUTES_MAX followed by the end, punctuation, "to" or "please"."""
    n = _number(toks, i)
    if n is None or n[0] != int(n[0]) or not 1 <= n[0] <= BARE_MINUTES_MAX:
        return None
    j = n[1]
    if j < len(toks) and toks[j].kind != "punct" and _w(toks, j) not in ("to", "please"):
        return None
    return int(n[0]), j


def _relative(toks: list[_Tok], i: int) -> _Comp | None:
    if _w(toks, i) == "in":
        d = _duration(toks, i + 1)
        if d:
            return _Comp("rel", i, d[2], rel=d[0], unit=d[1])
        b = _bare_count(toks, i + 1)
        if b:
            return _Comp("rel", i, b[1], rel=b[0] * 60, unit=60, assumed=True)
        return None
    d = _duration(toks, i)
    if d and _w(toks, d[2]) == "from" and _w(toks, d[2] + 1) == "now":
        return _Comp("rel", i, d[2] + 2, rel=d[0], unit=d[1])
    return None


def _timer_length(toks: list[_Tok], i: int) -> _Comp | None:
    """A timer's length: "for ten minutes", "ten minutes", "a 45 minute timer", "for ten"."""
    if _w(toks, i) == "for":
        d = _duration(toks, i + 1)
        if d:
            return _Comp("rel", i, d[2], rel=d[0], unit=d[1])
        b = _bare_count(toks, i + 1)
        if b:
            return _Comp("rel", i, b[1], rel=b[0] * 60, unit=60, assumed=True)
        return None
    rel = _relative(toks, i)
    if rel:
        return rel
    d = _duration(toks, i)
    if d is None:
        return None
    end = d[2] + 1 if _w(toks, d[2]) == "timer" else d[2]     # "a ten minute timer"
    return _Comp("rel", i, end, rel=d[0], unit=d[1])


def _attached_period(toks: list[_Tok], i: int) -> tuple[str, int] | None:
    if _w(toks, i) == "in" and _w(toks, i + 1) == "the" and _w(toks, i + 2) in _PERIOD_WORDS:
        return _w(toks, i + 2), i + 3
    if _w(toks, i) == "at" and _w(toks, i + 1) == "night":
        return "night", i + 2
    return None


def _period(toks: list[_Tok], i: int) -> _Comp | None:
    p = _attached_period(toks, i)
    return _Comp("period", i, p[1], period=p[0]) if p else None


def _hour_number(toks: list[_Tok], i: int) -> tuple[int, int] | None:
    if _kind(toks, i) == "int":
        v = int(toks[i].text)
        return (v, i + 1) if 0 <= v <= 23 and len(toks[i].text) <= 2 else None
    n = _small_number(toks, i)
    return n if n and 1 <= n[0] <= 12 else None


def _meridiem(toks: list[_Tok], i: int) -> tuple[str, int] | None:
    w = _w(toks, i)
    if w in ("am", "pm"):
        return w, i + 1
    if w in ("a", "p") and _w(toks, i + 1) == "m":
        return w + "m", i + 2
    return None


def _clock(toks: list[_Tok], i: int, weak_ok: bool) -> tuple[_Clock, int] | None:
    """A time of day. A bare hour ("8", "8 30", "eight thirty") only when weak_ok."""
    w = _w(toks, i)
    if w in ("noon", "midday"):
        return _Clock(12, 0, None, "noon", None), i + 1
    if w == "midnight":
        # "at midnight, at the end of the day" is how the read-back names a later midnight
        end = _seq_at(toks, _skip_gap(toks, i + 1), (_END_OF_DAY,))
        if end:
            return _Clock(0, 0, None, "midnight", None, end_of_day=True), end
        return _Clock(0, 0, None, "midnight", None), i + 1
    hn = _hour_number(toks, i)
    if hn and hn[0] == 12 and _w(toks, hn[1]) in ("noon", "midday", "midnight"):
        named = "midnight" if _w(toks, hn[1]) == "midnight" else "noon"
        return _Clock(0 if named == "midnight" else 12, 0, None, named, None), hn[1] + 1
    past_to = _past_to(toks, i)
    if past_to:
        hour, minute, j = past_to
        return _clock_tail(toks, hour, minute, j, strong=True, weak_ok=weak_ok, simple=False)
    tok = toks[i] if i < len(toks) else None
    if tok is None:
        return None
    if tok.kind == "colon" or (tok.kind == "dec" and re.fullmatch(r"\d{1,2}\.\d{2}", tok.text)):
        h, m = (int(x) for x in re.split(r"[:.]", tok.text))
        strong = tok.kind == "colon"
        return _clock_tail(toks, h, m, i + 1, strong=strong, weak_ok=weak_ok, simple=False)
    if hn is None:
        return None
    hour, j = hn
    minute, simple = 0, True
    if tok.kind == "int":
        nxt = toks[j] if j < len(toks) else None
        if nxt and nxt.kind == "int" and len(nxt.text) == 2 and int(nxt.text) <= 59:
            minute, j, simple = int(nxt.text), j + 1, False
    else:
        if _w(toks, j) in ("oh", "o") and _w(toks, j + 1) in _UNIT_WORDS:
            minute, j, simple = _UNIT_WORDS[_w(toks, j + 1)], j + 2, False
        else:
            mw = _small_number(toks, j)
            if mw and 10 <= mw[0] <= 59 and _w(toks, mw[1]) not in _DUR_UNITS:
                minute, j, simple = mw[0], mw[1], False
    return _clock_tail(toks, hour, minute, j, strong=False, weak_ok=weak_ok, simple=simple)


def _past_to(toks: list[_Tok], i: int) -> tuple[int, int, int] | None:
    """half past eight, quarter to nine, twenty to nine, ten after six: (hour, minute, next)."""
    k = i + 1 if _w(toks, i) == "a" and _w(toks, i + 1) == "quarter" else i
    if _w(toks, k) == "half":
        mins, kk = 30, k + 1
    elif _w(toks, k) == "quarter":
        mins, kk = 15, k + 1
    else:
        n = _number(toks, k)
        if n is None or n[0] != int(n[0]) or not 1 <= n[0] <= 59:
            return None
        mins, kk = int(n[0]), n[1]
        kk = kk + 1 if _w(toks, kk) in ("minute", "minutes") else kk
    rel = _w(toks, kk)
    hn = _hour_number(toks, kk + 1)
    if hn is None:
        return None
    hour, j = hn
    if rel in ("past", "after"):
        return hour, mins, j
    if rel in ("to", "till", "til", "before", "of") and mins != 30:
        before = 12 if hour == 1 else (23 if hour == 0 else hour - 1)
        return before, 60 - mins, j
    return None


def _clock_tail(toks: list[_Tok], hour: int, minute: int, j: int, *, strong: bool,
                weak_ok: bool, simple: bool) -> tuple[_Clock, int] | None:
    meridiem = period = None
    if _w(toks, j) == "oclock" and minute == 0:
        j, strong = j + 1, True
    mer = _meridiem(toks, j)
    if mer:
        meridiem, j, strong, simple = mer[0], mer[1], True, False
    ap = _attached_period(toks, j)
    if ap:
        period, j, strong, simple = ap[0], ap[1], True, False
    elif not strong:
        nxt = _day(toks, j)
        if nxt and nxt.day and nxt.day.period:
            strong = True
    if hour > 23 or minute > 59 or (meridiem and not 1 <= hour <= 12):
        return None
    if not strong and not weak_ok:
        return None
    return _Clock(hour, minute, meridiem, None, period, simple), j


def _clock_comp(toks: list[_Tok], i: int) -> _Comp | None:
    w = _w(toks, i)
    if w == "at":
        c = _clock(toks, i + 1, weak_ok=True)
    elif w == "for":
        c = _clock(toks, i + 1, weak_ok=False)
    else:
        c = _clock(toks, i, weak_ok=False)
    return _Comp("clock", i, c[1], clock=c[0]) if c else None


def _day_period(toks: list[_Tok], j: int) -> tuple[str | None, int]:
    if _w(toks, j) in _PERIOD_WORDS:
        return _w(toks, j), j + 1
    return None, j


def _day(toks: list[_Tok], i: int) -> _Comp | None:
    j, on = i, False
    if _w(toks, j) == "on":
        j, on = j + 1, True
    w = _w(toks, j)
    if not on:
        if w == "today":
            return _Comp("day", i, j + 1, day=_Day(offset=0, word="today"))
        if w == "tonight":
            return _Comp("day", i, j + 1, day=_Day(offset=0, word="tonight", period="night"))
        if w == "tomorrow":
            p, k = _day_period(toks, j + 1)
            return _Comp("day", i, k, day=_Day(offset=1, word="tomorrow", period=p))
        if _seq_at(toks, j, (("the", "day", "after", "tomorrow"),)):
            p, k = _day_period(toks, j + 4)
            return _Comp("day", i, k, day=_Day(offset=2, period=p))
        if w == "this" and _w(toks, j + 1) in ("morning", "afternoon", "evening"):
            return _Comp("day", i, j + 2, day=_Day(offset=0, word="today",
                                                   period=_w(toks, j + 1)))
        if w == "in":
            n = _number(toks, j + 1, article=True)
            if n and n[0] == int(n[0]) and _w(toks, n[1]) in _DAY_UNITS:
                days = int(n[0]) * _DAY_UNITS[_w(toks, n[1])]
                return _Comp("day", i, n[1] + 1, day=_Day(offset=days))
    qualifier = None
    if _w(toks, j) in ("this", "next"):
        qualifier, j = _w(toks, j), j + 1
    w = _w(toks, j)
    if w in _WEEKDAYS:
        p, k = _day_period(toks, j + 1)
        dom = month = None
        if _w(toks, k) == "the" and _ordinal(toks, k + 1):
            dom, k = _ordinal(toks, k + 1)
            if _w(toks, k) == "of" and _w(toks, k + 1) in _MONTHS:
                month, k = _MONTHS[_w(toks, k + 1)], k + 2
        weak = not (on or qualifier or p or dom)
        return _Comp("day", i, k, day=_Day(weekday=_WEEKDAYS[w], qualifier=qualifier,
                                           month=month, dom=dom, period=p, weak=weak))
    if qualifier:
        return None
    if w == "the" and _ordinal(toks, j + 1):
        dom, k = _ordinal(toks, j + 1)
        month = None
        if _w(toks, k) == "of" and _w(toks, k + 1) in _MONTHS:
            month, k = _MONTHS[_w(toks, k + 1)], k + 2
        if on or month:
            return _Comp("day", i, k, day=_Day(month=month, dom=dom))
        return None
    if w in _MONTHS:
        k = j + 1
        k = k + 1 if _w(toks, k) == "the" else k
        d = _dom(toks, k)
        if d:
            return _Comp("day", i, d[1], day=_Day(month=_MONTHS[w], dom=d[0]))
        return None
    d = _dom(toks, j)
    if d:
        k = d[1] + 1 if _w(toks, d[1]) == "of" else d[1]
        if _w(toks, k) in _MONTHS:
            return _Comp("day", i, k + 1, day=_Day(month=_MONTHS[_w(toks, k)], dom=d[0]))
    return None


def _component(toks: list[_Tok], i: int, kind: Kind) -> _Comp | None:
    if kind is Kind.DURATION:
        return _timer_length(toks, i)
    return _relative(toks, i) or _day(toks, i) or _period(toks, i) or _clock_comp(toks, i)


def _fragment(toks: list[_Tok], i: int, kind: Kind) -> _Comp | None:
    """The part that changed, said after a correction word: "three minutes", "nine"."""
    d = _duration(toks, i)
    if d:
        return _Comp("frag", i, d[2], rel=d[0], unit=d[1])
    if kind is Kind.INSTANT:
        k = i + 1 if _w(toks, i) == "the" else i
        o = _ordinal(toks, k)
        if o:
            return _Comp("frag", i, o[1], dom=o[0])
        c = _clock(toks, i, weak_ok=True)
        if c:
            count = c[0].hour if c[0].simple else None
            return _Comp("frag", i, c[1], clock=c[0], count=count)
    n = _number(toks, i)
    if n and n[0] == int(n[0]) and n[0] > 0:
        return _Comp("frag", i, n[1], count=int(n[0]))
    return None


def _markers(toks: list[_Tok], i: int, seqs: tuple[tuple[str, ...], ...]) -> int | None:
    """Consume a run of correction words and gaps; the index after them, or None."""
    j, found = i, False
    while True:
        g = _skip_gap(toks, j)
        m = _seq_at(toks, g, seqs)
        if m is None:
            break
        j, found = m, True
    return _skip_gap(toks, j) if found else None


# ------------------------------------------------------------------ scanning and folding


def _scan(toks: list[_Tok], kind: Kind) -> tuple[list[_Comp], Refusal | None]:
    comps: list[_Comp] = []
    i = 0
    while i < len(toks):
        c = _component(toks, i, kind)
        if c is None:
            i += 1
            continue
        adjacent = bool(comps) and _only_gap(toks, comps[-1].end, c.start)
        if _w(toks, c.end) == "or" and _w(toks, c.end + 1) == "so":
            c = replace(c, end=c.end + 2)        # "in two hours or so" is two hours
        comps.append(replace(c, adjacent=adjacent))
        i = c.end
        while True:
            k = _skip_gap(toks, i)
            if _w(toks, k) == "or" and _w(toks, k + 1) != "rather":
                k2 = _skip_gap(toks, k + 1)
                if _component(toks, k2, kind) or _fragment(toks, k2, kind):
                    return comps, Refusal.CONFLICTING
                break
            k2 = _markers(toks, k, _MARKERS)
            if k2 is None:
                break
            c2 = _component(toks, k2, kind) or _fragment(toks, k2, kind)
            if c2 is None:
                break
            marker = " ".join(t.text for t in toks[k:k2] if not _is_gap(t))
            comps.append(replace(c2, corrects=True, adjacent=True, marker=marker))
            i = c2.end
    return comps, None


def _keep(comps: list[_Comp]) -> list[_Comp]:
    """Drop a bare weekday that is not next to another time ("tell John about Friday")."""
    kept = []
    for n, c in enumerate(comps):
        if c.day is not None and c.day.weak and not c.corrects:
            nxt = comps[n + 1] if n + 1 < len(comps) else None
            if not c.adjacent and not (nxt and nxt.adjacent):
                continue
        kept.append(c)
    return kept


def _inherit(new: _Clock, old: _Clock | None) -> _Clock:
    if (old is not None and new.meridiem is None and new.period is None and new.named is None
            and 1 <= new.hour <= 12):
        return replace(new, meridiem=old.meridiem, period=old.period)
    return new


def _slot(c: _Comp) -> str:
    return c.kind


def _fill(expr: _Expr, c: _Comp) -> _Expr | None:
    if c.kind == "rel":
        if expr.rel is not None or expr.day or expr.clock or expr.period:
            return None
        return replace(expr, rel=c.rel, unit=c.unit, assumed=c.assumed)
    if expr.rel is not None:
        return None
    if c.kind == "day":
        return None if expr.day else replace(expr, day=c.day)
    if c.kind == "clock":
        return None if expr.clock else replace(expr, clock=c.clock)
    if c.kind == "period":
        return None if expr.period else replace(expr, period=c.period)
    return None


def _apply_correction(expr: _Expr, c: _Comp) -> _Expr | None:
    if c.kind == "rel":
        return _Expr(rel=c.rel, unit=c.unit, assumed=c.assumed)
    if c.kind == "day":
        base = _Expr(day=c.day, clock=expr.clock, period=expr.period)
        return base
    if c.kind == "clock":
        return _Expr(day=expr.day, clock=_inherit(c.clock, expr.clock), period=expr.period)
    if c.kind == "period":
        clock = replace(expr.clock, meridiem=None, period=None) if expr.clock else None
        day = replace(expr.day, period=None) if expr.day else None
        return _Expr(day=day, clock=clock, period=c.period)
    # a fragment changes only what it names
    if expr.rel is not None:
        if c.rel is not None:
            return _Expr(rel=c.rel, unit=c.unit)
        if c.count is not None:
            return replace(expr, rel=c.count * (expr.unit or 60))
        return None
    if c.clock is not None and (expr.clock is not None or expr.day is not None):
        return replace(expr, clock=_inherit(c.clock, expr.clock))
    if c.dom is not None and expr.day is not None and expr.day.dom is not None:
        return replace(expr, day=replace(expr.day, dom=c.dom))
    return None


def _comp_text(text: str, toks: list[_Tok], c: _Comp) -> str:
    return text[toks[c.start].start:toks[c.end - 1].end]


def _fold(text: str, toks: list[_Tok], comps: list[_Comp],
          start: _Expr | None = None) -> tuple[_Expr | None, list[Correction], Refusal | None]:
    expr = start or _Expr()
    corrections: list[Correction] = []
    prev: _Comp | None = None
    for c in comps:
        restated = prev is not None and c.adjacent and _slot(prev) == _slot(c)
        if c.corrects or restated or start is not None:
            new = _apply_correction(expr, c)
            if new is None:
                return None, corrections, Refusal.NOT_UNDERSTOOD
            superseded = _comp_text(text, toks, prev) if prev is not None else ""
            corrections.append(Correction(superseded, _comp_text(text, toks, c), c.marker))
            expr = new
        else:
            new = _fill(expr, c)
            if new is None:
                return None, corrections, Refusal.CONFLICTING
            expr = new
        prev = c
    return expr, corrections, None


def _spans(toks: list[_Tok], comps: list[_Comp]) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    prev_end_char = None
    for c in comps:
        s, e = toks[c.start].start, toks[c.end - 1].end
        if c.corrects and prev_end_char is not None:
            s = prev_end_char
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
        prev_end_char = e
    return tuple((a, b) for a, b in out)


def _remainder(text: str, spans: tuple[tuple[int, int], ...]) -> str:
    pieces, pos = [], 0
    for s, e in spans:
        pieces.append(text[pos:s])
        m = re.match(r"\s*,", text[e:])
        pos = e + m.end() if m and s > 0 and text[:s].strip() else e
    pieces.append(text[pos:])
    out = re.sub(r"\s+", " ", " ".join(pieces))
    out = re.sub(r"\s+([,.;!?])", r"\1", out)
    out = re.sub(r",{2,}", ",", out)
    return out.strip(" ,;")


# ------------------------------------------------------------------ resolving


def _localize(d: date, hour: int, minute: int, tz: tzinfo) -> tuple[datetime, Ambiguity | None]:
    naive = datetime(d.year, d.month, d.day, hour, minute)
    first = naive.replace(tzinfo=tz)
    canon = first.astimezone(UTC).astimezone(tz)
    if canon.replace(tzinfo=None) != naive:
        return canon, Ambiguity.SKIPPED_WALL_TIME
    if first.utcoffset() != naive.replace(tzinfo=tz, fold=1).utcoffset():
        return canon, Ambiguity.REPEATED_WALL_TIME
    return canon, None


def _hour_candidates(clock: _Clock, period: str | None) -> list[tuple[int, int, int]] | None:
    """(hour, minute, day shift) readings in time order, or None when they contradict."""
    if clock.named == "noon":
        return [(12, 0, 0)]
    if clock.named == "midnight":
        return [(0, 0, 1)]
    h, m = clock.hour, clock.minute
    if clock.meridiem:
        h24 = h % 12 + (12 if clock.meridiem == "pm" else 0)
        if period == "morning" and clock.meridiem == "pm":
            return None
        if period in ("afternoon", "evening") and clock.meridiem == "am":
            return None
        shift = 1 if period == "night" and h24 < 5 else 0
        return [(h24, m, shift)]
    if h == 0 or h >= 13:
        if period == "morning" and h >= 12:
            return None
        if period in ("afternoon", "evening") and h < 12:
            return None
        if period == "night":
            return [(h, m, 1)] if h < 5 else ([(h, m, 0)] if h >= 18 else None)
        return [(h, m, 0)]
    if period == "morning":
        return [(h % 12, m, 0)]
    if period == "afternoon":
        return [(h % 12 + 12, m, 0)]
    if period == "evening":
        return None if h == 12 else [(h + 12, m, 0)]
    if period == "night":
        if 6 <= h <= 11:
            return [(h + 12, m, 0)]
        if h == 12:
            return [(0, m, 1)]
        return [(h, m, 1)] if h <= 4 else None
    if h == 12:
        return [(12, m, 0), (0, m, 1)]
    return [(h, m, 0), (h + 12, m, 0)]


@dataclass
class _Resolved:
    instant: datetime | None = None
    ambiguities: list[Ambiguity] = field(default_factory=list)
    alternatives: list[datetime] = field(default_factory=list)
    refusal: Refusal | None = None
    ask_day: date | None = None


def _day_dates(day: _Day, today: date) -> tuple[list[date], list[Ambiguity], list[date]] | None:
    """Candidate dates in preference order, the flags, and alternative dates; None if the date
    does not exist."""
    if day.offset is not None:
        return [today + timedelta(days=day.offset)], [], []
    if day.weekday is not None and day.dom is None:
        delta = (day.weekday - today.weekday()) % 7
        if day.qualifier == "next":
            if delta == 0:
                return [today + timedelta(days=7)], [Ambiguity.SAME_WEEKDAY], [today]
            target = today + timedelta(days=delta)
            if today.weekday() + delta <= 6:
                return [target], [Ambiguity.NEXT_WEEKDAY], [target + timedelta(days=7)]
            return [target], [], []
        if day.qualifier == "this" or delta:
            return [today + timedelta(days=delta)], [], []
        return [today, today + timedelta(days=7)], [Ambiguity.SAME_WEEKDAY], []
    if day.dom is None:
        return None
    if day.weekday is not None:
        # "Friday the 21st": the next 21st must be a Friday; it is never searched for.
        found = _day_dates(replace(day, weekday=None, qualifier=None, period=None), today)
        if found is None or found[0][0].weekday() != day.weekday:
            return None
        return found
    if day.month is not None:
        for year in (today.year, today.year + 1):
            if day.dom <= calendar.monthrange(year, day.month)[1]:
                d = date(year, day.month, day.dom)
                if d >= today:
                    return [d], [], []
        return None
    year, month = today.year, today.month
    for _ in range(2):
        if day.dom <= calendar.monthrange(year, month)[1]:
            d = date(year, month, day.dom)
            if d >= today:
                return [d], [], []
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return None


def _merge_periods(*periods: str | None) -> tuple[bool, str | None]:
    found = {p for p in periods if p}
    if len(found) > 1:
        return False, None
    return True, (found.pop() if found else None)


def _resolve(expr: _Expr, now: datetime, tz: tzinfo) -> _Resolved:
    out = _Resolved()
    now_l = now.astimezone(tz)
    today = now_l.date()
    if expr.rel is not None:
        if expr.rel <= 0:
            out.refusal = Refusal.ZERO_DURATION
            return out
        out.instant = (now.astimezone(UTC) + timedelta(seconds=expr.rel)).astimezone(tz)
        if expr.assumed:
            out.ambiguities.append(Ambiguity.UNIT_ASSUMED)
        return out
    dates_info = _day_dates(expr.day, today) if expr.day else ([today], [], [])
    if dates_info is None:
        out.refusal = Refusal.INVALID_DATE
        return out
    dates, day_flags, alt_dates = dates_info
    if expr.clock is None:
        out.refusal = Refusal.NO_TIME_OF_DAY if (expr.day or expr.period) else \
            Refusal.NOT_UNDERSTOOD
        out.ask_day = dates[0] if expr.day else None
        return out
    ok, period = _merge_periods(expr.clock.period, expr.day.period if expr.day else None,
                                expr.period)
    cands = _hour_candidates(expr.clock, period) if ok else None
    if cands is None:
        out.refusal = Refusal.NOT_UNDERSTOOD
        return out

    def occurrences(base: date) -> list[tuple[datetime, int, Ambiguity | None]]:
        found = []
        for idx, (h, m, shift) in enumerate(cands):
            dt, flag = _localize(base + timedelta(days=shift), h, m, tz)
            if dt > now:
                found.append((dt, idx, flag))
        return sorted(found, key=lambda o: o[0])

    implicit = expr.day is None
    if implicit:
        both = occurrences(today) + occurrences(today + timedelta(days=1))
        bases = [(today, sorted(both, key=lambda o: o[0]))]
    else:
        bases = [(base, occurrences(base)) for base in dates]
    found = next(((base, options) for base, options in bases if options), None)
    if found is None:
        out.refusal = Refusal.IN_THE_PAST
        return out
    base_used, options = found
    dt, idx, flag = options[0]
    others = [o for o in options if o[1] != idx]
    out.instant = dt
    if flag:
        out.ambiguities.append(flag)
    if implicit and dt.date() > today and not expr.clock.named == "midnight" and \
            cands[idx][2] == 0:
        out.ambiguities.append(Ambiguity.PAST_TODAY)
    if len(cands) > 1 and others:
        out.ambiguities.append(Ambiguity.MERIDIEM)
        out.alternatives.append(others[0][0])
    if not implicit:
        out.ambiguities.extend(day_flags)
        h, m, shift = cands[idx]
        for alt_day in alt_dates:
            alt, _ = _localize(alt_day + timedelta(days=shift), h, m, tz)
            if alt > now:
                out.alternatives.append(alt)
        if Ambiguity.SAME_WEEKDAY in day_flags and len(dates) > 1:
            other = dates[1] if base_used == dates[0] else None
            if other is not None:
                alt, _ = _localize(other + timedelta(days=shift), h, m, tz)
                out.alternatives.append(alt)
        if (expr.clock.named == "midnight" and not expr.clock.end_of_day
                and expr.day.word not in ("today", "tonight")):
            start_of_day, _ = _localize(base_used, 0, 0, tz)
            if start_of_day > now:
                out.ambiguities.append(Ambiguity.MIDNIGHT)
                out.alternatives.append(start_of_day)
        if expr.day.word == "tomorrow" and now_l.hour < EARLY_HOURS_END:
            alt, _ = _localize(today + timedelta(days=shift), h, m, tz)
            if alt > now:
                out.ambiguities.append(Ambiguity.TOMORROW_AFTER_MIDNIGHT)
                out.alternatives.append(alt)
    return out


# ------------------------------------------------------------------ public parsers


def _check_now(now: datetime, tz: tzinfo) -> None:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be an aware datetime")
    if not isinstance(tz, tzinfo):
        raise ValueError("tz must be a tzinfo, for example ZoneInfo('America/Toronto')")


def _make(kind: Kind, expr: _Expr, res: _Resolved, now: datetime, tz: tzinfo,
          limit: timedelta, corrections: list[Correction]) -> SpokenTime:
    duration = timedelta(seconds=expr.rel) if expr.rel is not None else None
    alts = tuple(
        SpokenTime(kind, a, None, False, (), (), bool(corrections), tuple(corrections), now, tz,
                   limit, expr, _SEAL)
        for a in res.alternatives if now < a and a - now <= limit
    )
    return SpokenTime(kind, res.instant, duration, expr.rel is not None,
                      tuple(dict.fromkeys(res.ambiguities)), alts, bool(corrections),
                      tuple(corrections), now, tz, limit, expr, _SEAL)


def _question(kind: Kind, refusal: Refusal | None, value: SpokenTime | None,
              ask_day: date | None, now: datetime, tz: tzinfo, limit: timedelta) -> str | None:
    if value is not None:
        if value.alternatives:
            return QUESTION_CHOICE.format(first=say_when(value),
                                          second=say_when(value.alternatives[0]))
        return None
    if refusal is Refusal.NO_TIME_OF_DAY and ask_day is not None:
        return QUESTION_TIME_ON_DAY.format(day=_say_day(ask_day, now.astimezone(tz).date()))
    return {
        Refusal.IN_THE_PAST: QUESTION_IN_THE_PAST,
        Refusal.TOO_FAR_AHEAD: QUESTION_TOO_FAR,
        Refusal.ZERO_DURATION: QUESTION_ZERO,
        Refusal.DURATION_TOO_LONG: QUESTION_TOO_LONG.format(limit=say_duration(limit)),
        Refusal.INVALID_DATE: QUESTION_INVALID_DATE,
        Refusal.CONFLICTING: QUESTION_CONFLICTING,
    }.get(refusal, QUESTION_DURATION if kind is Kind.DURATION else QUESTION_WHEN)


def _finish(text: str, kind: Kind, expr: _Expr | None, refusal: Refusal | None,
            corrections: list[Correction], spans: tuple[tuple[int, int], ...], now: datetime,
            tz: tzinfo, limit: timedelta) -> TimeParse:
    remainder = _remainder(text, spans)
    value, ask_day = None, None
    if refusal is None and expr is not None:
        if kind is Kind.DURATION and expr.rel is None:
            refusal = Refusal.NOT_UNDERSTOOD
        else:
            res = _resolve(expr, now, tz)
            refusal, ask_day = res.refusal, res.ask_day
            if refusal is None:
                ahead = res.instant - now
                if kind is Kind.DURATION and timedelta(seconds=expr.rel) > limit:
                    refusal = Refusal.DURATION_TOO_LONG
                elif kind is Kind.INSTANT and ahead > limit:
                    refusal = Refusal.TOO_FAR_AHEAD
                else:
                    value = _make(kind, expr, res, now, tz, limit, corrections)
    if value is None and refusal is None:
        refusal = Refusal.NOT_UNDERSTOOD
    return TimeParse(text, value, refusal, spans, remainder,
                     _question(kind, refusal, value, ask_day, now, tz, limit))


def _recurring(toks: list[_Tok]) -> bool:
    """"every day at 8", "every other Friday", "daily at noon"."""
    for n, t in enumerate(toks):
        if t.text in _RECURRING_WORDS:
            return True
        nxt = _w(toks, n + 1)
        if t.text == "every" and (nxt in _WEEKDAYS or nxt in _PERIOD_WORDS or nxt in _DUR_UNITS
                                  or nxt in _DAY_UNITS or nxt in ("other", "single")
                                  or _number(toks, n + 1) is not None):
            return True
    return False


def _parse(text: str, kind: Kind, now: datetime, tz: tzinfo, limit: timedelta) -> TimeParse:
    _check_now(now, tz)
    toks = _tokenize(text)
    comps, refusal = _scan(toks, kind)
    comps = _keep(comps)
    spans = _spans(toks, comps)
    if _recurring(toks):
        refusal = Refusal.NOT_UNDERSTOOD          # recurring reminders are not built yet
    if refusal is not None or not comps:
        return _finish(text, kind, None, refusal or Refusal.NOT_UNDERSTOOD, [], spans, now, tz,
                       limit)
    expr, corrections, refusal = _fold(text, toks, comps)
    return _finish(text, kind, expr, refusal, corrections, spans, now, tz, limit)


def parse_when(text: str, *, now: datetime, tz: tzinfo,
               max_ahead: timedelta = MAX_REMINDER_AHEAD) -> TimeParse:
    """The moment a reminder is for, from the person's words."""
    return _parse(text, Kind.INSTANT, now, tz, max_ahead)


def parse_duration(text: str, *, now: datetime, tz: tzinfo,
                   max_duration: timedelta = MAX_TIMER_DURATION) -> TimeParse:
    """A timer's length, from the person's words. `instant` is when it ends if started now."""
    return _parse(text, Kind.DURATION, now, tz, max_duration)


def correct(previous: SpokenTime, text: str, *, now: datetime) -> TimeParse | None:
    """Apply a reply to a read-back that corrects the time ("no, eight", "make it nine pm",
    "three minutes"). None when the reply holds no time, so it is a yes, a no or another turn.
    A relative time is measured again from the new `now`."""
    _require_validated(previous)
    tz = previous.tz
    _check_now(now, tz)
    toks = _tokenize(text)
    k = _markers(toks, 0, _REPLY_LEADS)
    k = _skip_gap(toks, 0) if k is None else k
    first = _component(toks, k, previous.kind) or _fragment(toks, k, previous.kind)
    if first is None:
        return None
    chain, i = [first], first.end
    while True:
        g = _skip_gap(toks, i)
        k2 = _markers(toks, g, _MARKERS)
        nxt = _component(toks, k2 if k2 is not None else g, previous.kind)
        if nxt is None and k2 is not None:
            nxt = _fragment(toks, k2, previous.kind)
        if nxt is None:
            break
        chain.append(nxt)
        i = nxt.end
    tail = _skip_gap(toks, i)
    while tail < len(toks) and (toks[tail].text in _REPLY_TRAILERS or _is_gap(toks[tail])):
        tail += 1
    if tail != len(toks):
        return None
    spans = _spans(toks, chain)
    expr, corrections, refusal = _fold(text, toks, chain, start=previous._expr)
    corrections = list(previous.corrections) + [
        Correction(_say_value(previous), c.replacement, c.marker) for c in corrections]
    return _finish(text, previous.kind, expr, refusal, corrections, spans, now, tz,
                   previous.limit)


def reminder_subject(remainder: str) -> Subject:
    """What the reminder is about, from the text left after the time is removed. The lead
    ("remind me to") goes; its connector is kept for a rebuilt sentence."""
    m = _SUBJECT_LEAD_RE.match(remainder)
    text, connector = remainder, "to"
    if m:
        text = remainder[m.end():]
        conn = (m.group("conn") or "to").lower()
        connector = {"for": "about", "of": "about"}.get(conn, conn)
    text = re.sub(r"[\s,]*\bplease\b[\s.!?]*$", "", text.strip(), flags=re.IGNORECASE)
    text = text.strip().strip(" ,").rstrip(".!?").strip()
    return Subject(text[:MAX_SUBJECT_CHARS], connector)


# ------------------------------------------------------------------ day ranges


def parse_day_range(text: str, *, now: datetime, tz: tzinfo) -> DayRange | None:
    """The stretch of calendar time a question names ("what's on my calendar tomorrow"),
    or None when it names none."""
    _check_now(now, tz)
    toks = _tokenize(text)
    today = now.astimezone(tz).date()
    for i in range(len(toks)):
        week = _week_range(toks, i, today, tz)
        if week:
            return week
        c = _day(toks, i) or _period(toks, i)
        if c is None or (c.day is not None and c.day.weak):
            continue
        day = c.day or _Day(offset=0, word="today")
        end_tok = c.end
        if c.day is not None and c.day.period is None:
            p = _period(toks, c.end)
            if p:
                day, end_tok = replace(day, period=p.period), p.end
        period = c.period or day.period
        info = _day_dates(day, today)
        if info is None:
            return None
        dates, flags, _ = info
        d = dates[0]
        if period:
            lo, hi = _PERIOD_HOURS[period]
            start = _localize(d, lo, 0, tz)[0]
            end = _localize(d + timedelta(days=hi // 24), hi % 24, 0, tz)[0]
        else:
            start = _localize(d, 0, 0, tz)[0]
            end = _localize(d + timedelta(days=1), 0, 0, tz)[0]
        words = _say_day(d, today)
        if period:
            words = _period_words(words, period)
        span = (toks[c.start].start, toks[end_tok - 1].end)
        return DayRange(start, end, words, span, tuple(flags))
    return None


def _week_range(toks: list[_Tok], i: int, today: date, tz: tzinfo) -> DayRange | None:
    q, w = _w(toks, i), _w(toks, i + 1)
    if q not in ("this", "next") or w not in ("week", "weekend"):
        return None
    monday = today - timedelta(days=today.weekday())
    if w == "week":
        start_d = today if q == "this" else monday + timedelta(days=7)
        end_d = monday + timedelta(days=7 if q == "this" else 14)
    else:
        saturday = monday + timedelta(days=5) + (timedelta(days=7) if q == "next" else
                                                 timedelta(0))
        start_d, end_d = max(saturday, today), saturday + timedelta(days=2)
    span = (toks[i].start, toks[i + 1].end)
    return DayRange(_localize(start_d, 0, 0, tz)[0], _localize(end_d, 0, 0, tz)[0],
                    f"{q} {w}", span)


def _period_words(day_words: str, period: str) -> str:
    if day_words == "today":
        return "tonight" if period == "night" else f"this {period}"
    if day_words == "tomorrow":
        return f"tomorrow {period}"
    return f"{day_words}, in the {period}" if period != "night" else f"{day_words}, at night"


# ------------------------------------------------------------------ words


_UNIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEEN_NAMES = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
               "seventeen", "eighteen", "nineteen"]
_TENS_NAMES = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
               70: "seventy", 80: "eighty", 90: "ninety"}
_ORD_NAMES = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
              7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
              12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
              16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
              20: "twentieth", 30: "thirtieth"}
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"]


def number_words(n: int) -> str:
    """A whole number from zero to 999,999 in words: 45 -> "forty-five"."""
    if not 0 <= n < 1_000_000:
        raise ValueError("number out of range for words")
    if n < 10:
        return _UNIT_NAMES[n]
    if n < 20:
        return _TEEN_NAMES[n - 10]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS_NAMES[tens * 10] + (f"-{_UNIT_NAMES[unit]}" if unit else "")
    if n < 1000:
        head, rest = divmod(n, 100)
        return f"{_UNIT_NAMES[head]} hundred" + (f" and {number_words(rest)}" if rest else "")
    head, rest = divmod(n, 1000)
    tail = ""
    if rest:
        tail = f" and {number_words(rest)}" if rest < 100 else f" {number_words(rest)}"
    return f"{number_words(head)} thousand{tail}"


def _ordinal_words(n: int) -> str:
    if n in _ORD_NAMES:
        return _ORD_NAMES[n]
    tens, unit = divmod(n, 10)
    return f"{_TENS_NAMES[tens * 10]}-{_ORD_NAMES[unit]}"


def _say_clock(hour: int, minute: int) -> str:
    if (hour, minute) == (12, 0):
        return "noon"
    h12 = hour % 12 or 12
    words = number_words(h12)
    if minute:
        words += f" oh {number_words(minute)}" if minute < 10 else f" {number_words(minute)}"
    if hour < 12:
        part = "in the morning"
    elif hour < 18:
        part = "in the afternoon"
    elif hour < 21:
        part = "in the evening"
    else:
        part = "at night"
    return f"{words} {part}"


def _say_day(d: date, today: date, named_tomorrow: bool = True) -> str:
    dd = (d - today).days
    if dd == 0:
        return "today"
    if dd == 1 and named_tomorrow:
        return "tomorrow"
    base = f"on {_WEEKDAY_NAMES[d.weekday()]} the {_ordinal_words(d.day)}"
    if 1 <= dd <= 6:
        return base
    return f"{base} of {_MONTH_NAMES[d.month]}"


def say_duration(d: timedelta) -> str:
    """A length of time in words: "three minutes", "an hour and a half"."""
    total = int(d.total_seconds())
    if total <= 0 or total != d.total_seconds():
        raise ValueError("a duration to say must be a positive whole number of seconds")
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    if h and m == 30 and not s:
        return "an hour and a half" if h == 1 else f"{number_words(h)} and a half hours"
    parts = []
    if h:
        parts.append("an hour" if h == 1 else f"{number_words(h)} hours")
    if m:
        parts.append(f"{number_words(m)} minute" + ("s" if m != 1 else ""))
    if s:
        parts.append(f"{number_words(s)} second" + ("s" if s != 1 else ""))
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _require_validated(value: object) -> SpokenTime:
    if not isinstance(value, SpokenTime) or value._seal is not _SEAL:
        raise TypeError("a read-back is built only from a SpokenTime made by this parser")
    ahead = value.instant - value.now
    if value.kind is Kind.INSTANT and not timedelta(0) < ahead <= value.limit:
        raise ValueError("the time is outside its bounds")
    if value.kind is Kind.DURATION and (value.duration is None
                                        or not timedelta(0) < value.duration <= value.limit):
        raise ValueError("the duration is outside its bounds")
    return value


def say_when(value: SpokenTime, *, now: datetime | None = None) -> str:
    """The moment in words, relative to the parse's now (or the given one): "in three
    minutes", "tomorrow at eight in the morning", "on Friday the twenty-first at noon"."""
    _require_validated(value)
    now = value.now if now is None else now
    if value.relative and value.duration is not None and now == value.now:
        return f"in {say_duration(value.duration)}"
    local = value.instant.astimezone(value.tz)
    now_l = now.astimezone(value.tz)
    today = now_l.date()
    early = now_l.hour < EARLY_HOURS_END      # "tomorrow" is not said in the small hours
    if (local.hour, local.minute) == (0, 0):
        ref = local.date() - timedelta(days=1)
        if ref == today:
            return "tonight at midnight"
        return f"{_say_day(ref, today, named_tomorrow=not early)} {_MIDNIGHT_END_WORDS}"
    day = _say_day(local.date(), today, named_tomorrow=not early)
    return f"{day} at {_say_clock(local.hour, local.minute)}"


def _say_value(value: SpokenTime) -> str:
    return say_duration(value.duration) if value.kind is Kind.DURATION else say_when(value)


def _capital(s: str) -> str:
    return s[:1].upper() + s[1:]


def read_back_reminder(value: SpokenTime, subject: Subject | None) -> str:
    """The scripted read-back for a reminder: "In three minutes, stretch. Shall I set it?"."""
    _require_validated(value)
    if value.kind is not Kind.INSTANT:
        raise TypeError("a reminder's read-back needs a moment, not a duration")
    when = _capital(say_when(value))
    if subject is not None and not isinstance(subject, Subject):
        raise TypeError("the subject must be a Subject from reminder_subject()")
    if subject is None or not subject.text:
        return READ_BACK_REMINDER_NO_SUBJECT.format(when=when)
    if len(subject.text) > MAX_SUBJECT_CHARS or any(ch in subject.text for ch in "\r\n"):
        raise ValueError("the subject is not a validated reminder subject")
    return READ_BACK_REMINDER.format(when=when, subject=subject.text)


def announce_timer(value: SpokenTime) -> str:
    """The scripted announcement: "Ten minute timer, starting now."."""
    _require_validated(value)
    if value.kind is not Kind.DURATION or value.duration is None:
        raise TypeError("a timer announcement needs a duration")
    total = int(value.duration.total_seconds())
    if total % 3600 == 0:
        n, unit = total // 3600, "hour"
    elif total % 60 == 0:
        n, unit = total // 60, "minute"
    elif total < 60:
        n, unit = total, "second"
    else:
        return ANNOUNCE_TIMER_LONG.format(length=say_duration(value.duration))
    return ANNOUNCE_TIMER.format(length=_capital(f"{number_words(n)} {unit}"))


# ------------------------------------------------------------------ the read-back state machine


class ConfirmState(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DECLINED = "declined"


class ReplyKind(Enum):
    YES = "yes"
    NO = "no"
    CORRECTION = "correction"
    SILENCE = "silence"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class Proposal:
    """A reminder waiting for a yes (confirm: ask). Only CONFIRMED may be executed.

    `answer_normally` is set when the turn that declined it was about something else, so the
    caller answers that turn as usual. `question` is set when a correction could not be used.
    """
    when: SpokenTime
    subject: Subject | None
    read_back: str
    asked_at: datetime
    state: ConfirmState = ConfirmState.PENDING
    last_reply: ReplyKind | None = None
    answer_normally: bool = False
    refusal: Refusal | None = None
    question: str | None = None


def propose(when: SpokenTime, subject: Subject | None, *, at: datetime) -> Proposal:
    """Start a read-back: the scripted line to speak and the state waiting for the reply."""
    return Proposal(when, subject, read_back_reminder(when, subject), at)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower().replace("’", "'"))
                  ).strip()


def _is_no(norm: str) -> tuple[bool, bool]:
    """(is a no, carries more words than the no)."""
    words = norm.split()
    if not words:
        return False, False
    for phrase in _NO_PHRASES:
        if norm == phrase or norm.startswith(phrase + " "):
            rest = norm[len(phrase):].split()
            return True, any(w not in _REPLY_TRAILERS for w in rest)
    if words[0] in _NO_WORDS:
        rest = [w for w in words[1:] if w not in _REPLY_TRAILERS and w not in _NO_WORDS
                and w not in ("that", "it")]
        return True, bool(rest)
    return False, False


def _is_yes(text: str, norm: str) -> bool:
    """A short yes (delegate_lane.is_affirmative, or "set it", "that's right") made only of
    yes words: "okay, what's the weather" is a new turn, not a yes."""
    said = is_affirmative(text) or any(norm == p or norm.startswith(p + " ")
                                       for p in _YES_PHRASES)
    return said and all(w in _YES_VOCABULARY for w in norm.split())


def reply(proposal: Proposal, text: str, *, at: datetime,
          waited: timedelta | None = None) -> Proposal:
    """The person's next turn after the read-back. Yes confirms; no declines; a corrected time
    ("no, eight") reads back again; silence past the window or an unrelated turn declines, and
    the unrelated turn is answered normally.

    `waited` is the silence the caller measured between the end of the read-back and the start
    of the reply, when it knows it (a voice host does). Without it the silence is taken to be
    `at - asked_at`, which also counts the read-back's own playback and the reply's length."""
    if proposal.state is not ConfirmState.PENDING:
        raise ValueError("the proposal is no longer waiting for a reply")
    if (at - proposal.asked_at if waited is None else waited) > READBACK_WINDOW:
        return replace(proposal, state=ConfirmState.DECLINED, last_reply=ReplyKind.SILENCE,
                       answer_normally=bool(_norm(text)))
    corrected = correct(proposal.when, text, now=at)
    if corrected is not None:
        if corrected.value is None:
            return replace(proposal, state=ConfirmState.DECLINED,
                           last_reply=ReplyKind.CORRECTION, refusal=corrected.refusal,
                           question=corrected.question)
        return Proposal(corrected.value, proposal.subject,
                        read_back_reminder(corrected.value, proposal.subject), at,
                        last_reply=ReplyKind.CORRECTION)
    norm = _norm(text)
    no, more = _is_no(norm)
    if no:
        return replace(proposal, state=ConfirmState.DECLINED, last_reply=ReplyKind.NO,
                       answer_normally=more)
    if _is_yes(text, norm):
        return replace(proposal, state=ConfirmState.CONFIRMED, last_reply=ReplyKind.YES)
    return replace(proposal, state=ConfirmState.DECLINED, last_reply=ReplyKind.UNRELATED,
                   answer_normally=True)


def silence(proposal: Proposal, *, at: datetime, window: timedelta = READBACK_WINDOW
            ) -> Proposal:
    """No reply yet at `at`: still pending inside the window, declined after it."""
    if proposal.state is not ConfirmState.PENDING or at - proposal.asked_at <= window:
        return proposal
    return replace(proposal, state=ConfirmState.DECLINED, last_reply=ReplyKind.SILENCE)


def classify_reply(text: str) -> tuple[ReplyKind, bool]:
    """A reply to a question that holds no time (cancel this reminder?): (YES, NO or UNRELATED,
    whether the turn carries more than the answer and is to be answered normally). Uses the
    same yes and no words as reply()."""
    norm = _norm(text)
    no, more = _is_no(norm)
    if no:
        return ReplyKind.NO, more
    if _is_yes(text, norm):
        return ReplyKind.YES, False
    return ReplyKind.UNRELATED, True


def is_recurring(text: str) -> bool:
    """True when the words ask for a repeating time ("every day at eight", "daily at noon").
    The parsers refuse those as not understood; this lets a caller say why."""
    return _recurring(_tokenize(text))


def timer_length(d: timedelta) -> str:
    """A timer's length as the noun before "timer": "ten minute", "one hour", "ninety second".
    Whole hours, else whole minutes, else seconds."""
    total = int(d.total_seconds())
    if total <= 0 or total != d.total_seconds():
        raise ValueError("a timer length must be a positive whole number of seconds")
    if total % 3600 == 0:
        n, unit = total // 3600, "hour"
    elif total % 60 == 0:
        n, unit = total // 60, "minute"
    else:
        n, unit = total, "second"
    return f"{number_words(n)} {unit}"


def say_moment(instant: datetime, *, now: datetime, tz: tzinfo) -> str:
    """A stored moment in words, for a list, a cancel question or a missed reminder: "today at
    three in the afternoon", "yesterday at noon", "tomorrow at midnight", "on Friday the
    twenty-first at eight in the morning". Unlike say_when() it takes any aware instant (one read
    back from the store, already validated when it was set) and may be in the past."""
    _check_now(now, tz)
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise ValueError("instant must be an aware datetime")
    local = instant.astimezone(tz)
    today = now.astimezone(tz).date()
    d = local.date()
    day = "yesterday" if (d - today).days == -1 else _say_day(d, today)
    if (local.hour, local.minute) == (0, 0):
        return f"{day} at midnight"
    return f"{day} at {_say_clock(local.hour, local.minute)}"


def say_clock(instant: datetime, *, tz: tzinfo) -> str:
    """The time of day alone, in words: "eight in the evening", "noon", "midnight"."""
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise ValueError("instant must be an aware datetime")
    local = instant.astimezone(tz)
    if (local.hour, local.minute) == (0, 0):
        return "midnight"
    return _say_clock(local.hour, local.minute)


__all__ = [
    "ANNOUNCE_TIMER", "BARE_MINUTES_MAX", "EARLY_HOURS_END", "MAX_REMINDER_AHEAD",
    "MAX_SUBJECT_CHARS", "MAX_TIMER_DURATION", "READBACK_WINDOW", "READ_BACK_REMINDER",
    "Ambiguity", "ConfirmState", "Correction", "DayRange", "Kind", "Proposal", "Refusal",
    "ReplyKind", "SpokenTime", "Subject", "TimeParse", "announce_timer", "classify_reply",
    "correct", "is_recurring", "number_words", "parse_day_range", "parse_duration",
    "parse_when", "propose", "read_back_reminder", "reminder_subject", "reply", "say_duration",
    "say_clock", "say_moment", "say_when", "silence", "timer_length",
]
