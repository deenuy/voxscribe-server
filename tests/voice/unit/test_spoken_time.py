"""PL44 spoken times, durations, self-corrections and read-back.

Every case runs against a fixed `now` in America/Toronto, a zone with daylight saving; nothing
reads the system clock. The default now is Thursday 19 November 2026, 15:07:23.

Acceptance criteria covered (TOOLS-CONNECTORS.md section 5, PL44):
  fixture table of spoken forms, digits and number words, each resolving to the value its row
  states under the fake clock and zone (FORMS, DURATIONS, CORRECTIONS, AMBIGUITIES, EDGES);
  a time across a daylight-saving change keeps its wall-clock time (test_dst_*);
  an unparseable time yields one scripted clarifying question, never a guess (REFUSALS);
  yes confirms, no declines, silence past the window declines, "no, eight" reads back again,
  an unrelated turn declines and is answered normally (test_reply_*, test_silence_*);
  the read-back is built only from validated arguments (test_read_back_only_from_validated_*);
  the validation gate's sentence "remind me in two minutes, no, three minutes, to stretch"
  (test_validation_gate_sentence).
"""
from __future__ import annotations

import inspect
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.local.voice import spoken_time as st
from app.local.voice.spoken_time import Ambiguity as A
from app.local.voice.spoken_time import ConfirmState, Refusal, ReplyKind

TZ = ZoneInfo("America/Toronto")
NOW = datetime(2026, 11, 19, 15, 7, 23, tzinfo=TZ)          # a Thursday


def at(text: str) -> datetime:
    """A wall-clock time in TZ: "2026-11-20 08:00" or with seconds."""
    fmt = "%Y-%m-%d %H:%M:%S" if text.count(":") == 2 else "%Y-%m-%d %H:%M"
    return datetime.strptime(text, fmt).replace(tzinfo=TZ)


def utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC)


def when(text: str, now: datetime = NOW, **kw) -> st.TimeParse:
    return st.parse_when(text, now=now, tz=TZ, **kw)


def length(text: str, now: datetime = NOW, **kw) -> st.TimeParse:
    return st.parse_duration(text, now=now, tz=TZ, **kw)


def assert_round_trip(value: st.SpokenTime) -> None:
    """Parsing the read-back with the same now gives the same instant, with no ambiguity
    left except a repeated wall-clock time, which words cannot tell apart."""
    words = st.say_when(value)
    again = st.parse_when(words, now=value.now, tz=value.tz)
    assert again.ok, (words, again.refusal)
    assert utc(again.value.instant) == utc(value.instant), words
    assert set(again.value.ambiguities) <= {A.REPEATED_WALL_TIME}, (words,
                                                                    again.value.ambiguities)
    assert not re.search(r"\d", words), words


# ------------------------------------------------------------------ the fixture table of forms

# (spoken text, wall-clock result in TZ, ambiguities expected)
FORMS = [
    # relative: elapsed time from now, seconds kept
    ("in two minutes", "2026-11-19 15:09:23", set()),
    ("in 2 mins", "2026-11-19 15:09:23", set()),
    ("in twenty minutes", "2026-11-19 15:27:23", set()),
    ("in an hour", "2026-11-19 16:07:23", set()),
    ("in an hour and a half", "2026-11-19 16:37:23", set()),
    ("in one and a half hours", "2026-11-19 16:37:23", set()),
    ("in 1.5 hours", "2026-11-19 16:37:23", set()),
    ("in half an hour", "2026-11-19 15:37:23", set()),
    ("in a quarter of an hour", "2026-11-19 15:22:23", set()),
    ("in 90 seconds", "2026-11-19 15:08:53", set()),
    ("in two hours and fifteen minutes", "2026-11-19 17:22:23", set()),
    ("in 2 hours 30 minutes", "2026-11-19 17:37:23", set()),
    ("ten minutes from now", "2026-11-19 15:17:23", set()),
    ("in two hours or so", "2026-11-19 17:07:23", set()),
    ("in ten", "2026-11-19 15:17:23", {A.UNIT_ASSUMED}),
    # clock times, digits and number words
    ("tomorrow at 8am", "2026-11-20 08:00", set()),
    ("tomorrow at eight in the morning", "2026-11-20 08:00", set()),
    ("tomorrow at nine", "2026-11-20 09:00", {A.MERIDIEM}),
    ("at 8 30", "2026-11-19 20:30", {A.MERIDIEM}),
    ("at 8:30pm", "2026-11-19 20:30", set()),
    ("at 8.30 pm", "2026-11-19 20:30", set()),
    ("at 8 p.m.", "2026-11-19 20:00", set()),
    ("at eight thirty", "2026-11-19 20:30", {A.MERIDIEM}),
    ("at eight oh five pm", "2026-11-19 20:05", set()),
    ("at half past eight", "2026-11-19 20:30", {A.MERIDIEM}),
    ("half past six", "2026-11-19 18:30", {A.MERIDIEM}),
    ("quarter to nine tonight", "2026-11-19 20:45", set()),
    ("twenty to nine in the morning", "2026-11-20 08:40", {A.PAST_TODAY}),
    ("ten after six in the evening", "2026-11-19 18:10", set()),
    ("at eight o'clock in the evening", "2026-11-19 20:00", set()),
    ("at 20:00", "2026-11-19 20:00", set()),
    ("at six", "2026-11-19 18:00", {A.MERIDIEM}),
    ("at 6", "2026-11-19 18:00", {A.MERIDIEM}),
    ("at noon", "2026-11-20 12:00", {A.PAST_TODAY}),
    ("at midnight", "2026-11-20 00:00", set()),
    ("tonight at 1", "2026-11-20 01:00", set()),
    ("tomorrow at 11 at night", "2026-11-20 23:00", set()),
    ("tomorrow at two in the morning", "2026-11-20 02:00", set()),
    ("this afternoon at 4", "2026-11-19 16:00", set()),
    ("at 4 this afternoon", "2026-11-19 16:00", set()),
    ("this evening at 7", "2026-11-19 19:00", set()),
    ("at 9 tomorrow morning", "2026-11-20 09:00", set()),
    ("eight tomorrow morning", "2026-11-20 08:00", set()),
    # named days and dates
    ("on Friday at noon", "2026-11-20 12:00", set()),
    ("Saturday at 10 am", "2026-11-21 10:00", set()),
    ("this Saturday at 10 am", "2026-11-21 10:00", set()),
    ("on Saturday morning at 10", "2026-11-21 10:00", set()),
    ("Friday night at 11", "2026-11-20 23:00", set()),
    ("next Monday at 9", "2026-11-23 09:00", {A.MERIDIEM}),
    ("next Monday at 9 am", "2026-11-23 09:00", set()),
    ("the day after tomorrow at 7 pm", "2026-11-21 19:00", set()),
    ("in three days at 9 am", "2026-11-22 09:00", set()),
    ("in a week at noon", "2026-11-26 12:00", set()),
    ("on the 30th at 9 am", "2026-11-30 09:00", set()),
    ("on the 31st at noon", "2026-12-31 12:00", set()),
    ("on the twenty-first of December at noon", "2026-12-21 12:00", set()),
    ("on November 30th at noon", "2026-11-30 12:00", set()),
    ("November 30 at 3 pm", "2026-11-30 15:00", set()),
    ("30 November at 3 pm", "2026-11-30 15:00", set()),
    ("Saturday the 21st at 10 am", "2026-11-21 10:00", set()),
    ("on January 31st at noon", "2027-01-31 12:00", set()),
]


@pytest.mark.parametrize("text,expected,ambiguities", FORMS, ids=[f[0] for f in FORMS])
def test_forms(text, expected, ambiguities):
    p = when(text)
    assert p.ok, (p.refusal, p.question)
    assert p.value.instant == at(expected)
    assert set(p.value.ambiguities) == ambiguities
    assert p.value.kind is st.Kind.INSTANT
    assert not p.value.corrected
    assert_round_trip(p.value)


# (spoken text, seconds, ambiguities)
DURATIONS = [
    ("set a timer for ten minutes", 600, set()),
    ("for 10 minutes", 600, set()),
    ("set a timer for one minute", 60, set()),
    ("a 45 minute timer", 2700, set()),
    ("45-minute timer", 2700, set()),
    ("for an hour", 3600, set()),
    ("for a minute and a half", 90, set()),
    ("for 90 seconds", 90, set()),
    ("for thirty seconds", 30, set()),
    ("for two and a half minutes", 150, set()),
    ("for 1 hour 15 minutes", 4500, set()),
    ("for one hundred and twenty minutes", 7200, set()),
    ("in twenty minutes", 1200, set()),
    ("for 24 hours", 86400, set()),
    ("set a timer for ten", 600, {A.UNIT_ASSUMED}),
]


@pytest.mark.parametrize("text,seconds,ambiguities", DURATIONS, ids=[d[0] for d in DURATIONS])
def test_durations(text, seconds, ambiguities):
    p = length(text)
    assert p.ok, (p.refusal, p.question)
    v = p.value
    assert v.kind is st.Kind.DURATION
    assert v.duration == timedelta(seconds=seconds)
    assert utc(v.instant) == utc(NOW) + timedelta(seconds=seconds)
    assert set(v.ambiguities) == ambiguities
    words = st.say_duration(v.duration)
    assert not re.search(r"\d", words)
    for spoken in (words, st.announce_timer(v)):
        again = length(spoken)
        assert again.ok and again.value.duration == v.duration, spoken
        assert not again.value.ambiguities


# ------------------------------------------------------------------ self-corrections

# (spoken text, wall-clock result, the superseded text, the marker)
CORRECTIONS = [
    ("remind me in two minutes, no, three minutes, to stretch", "2026-11-19 15:10:23",
     "in two minutes", "no"),
    ("in two minutes, no, three", "2026-11-19 15:10:23", "in two minutes", "no"),
    ("in five, no wait, ten minutes", "2026-11-19 15:17:23", "in five", "no wait"),
    ("in an hour, actually, make that two", "2026-11-19 17:07:23", "in an hour",
     "actually make that"),
    ("tomorrow at eight, sorry, nine", "2026-11-20 09:00", "at eight", "sorry"),
    ("at 5, I mean 6 pm", "2026-11-19 18:00", "at 5", "i mean"),
    ("at 8 pm, no, 9", "2026-11-19 21:00", "at 8 pm", "no"),
    ("at six, no, seven", "2026-11-19 19:00", "at six", "no"),
    ("six pm, no, seven", "2026-11-19 19:00", "six pm", "no"),
    ("at eight, at eight thirty", "2026-11-19 20:30", "at eight", None),
    ("tomorrow at 9 am, no, Saturday", "2026-11-21 09:00", "at 9 am", "no"),
    ("on the 21st, no, the 22nd at 9 am", "2026-11-22 09:00", "on the 21st", "no"),
    ("at 8 in the morning, no, in the evening", "2026-11-19 20:00", "at 8 in the morning",
     "no"),
    ("at 5, no, in ten minutes", "2026-11-19 15:17:23", "at 5", "no"),
    ("in ten minutes, no, at 5 pm", "2026-11-19 17:00", "in ten minutes", "no"),
    ("remind me to call the dentist at 8, I mean at 9, tomorrow", "2026-11-20 09:00", "at 8",
     "i mean"),
]


@pytest.mark.parametrize("text,expected,superseded,marker", CORRECTIONS,
                         ids=[c[0] for c in CORRECTIONS])
def test_self_correction_later_value_wins(text, expected, superseded, marker):
    p = when(text)
    assert p.ok, (p.refusal, p.question)
    v = p.value
    assert v.instant == at(expected)
    assert v.corrected
    assert v.corrections[-1].superseded == superseded      # what was heard is kept, not deleted
    assert v.corrections[-1].marker == marker
    assert_round_trip(v)


@pytest.mark.parametrize("text,seconds", [
    ("set a timer for 10 minutes, no, 15", 900),
    ("a ten minute timer, sorry, twelve", 720),
    ("for an hour, I mean, half an hour", 1800),
])
def test_timer_self_correction(text, seconds):
    p = length(text)
    assert p.ok and p.value.corrected
    assert p.value.duration == timedelta(seconds=seconds)


def test_fragment_inherits_only_what_it_does_not_name():
    # "nine" replaces the hour; tomorrow stays, and the open morning-or-evening stays open
    p = when("tomorrow at eight, sorry, nine")
    assert A.MERIDIEM in p.value.ambiguities
    # "9" after "8 pm" keeps the evening
    assert when("at 8 pm, no, 9").value.ambiguities == ()


def test_a_correction_that_cannot_apply_is_not_guessed():
    p = when("at 8, no, ten minutes")
    assert p.refusal is Refusal.NOT_UNDERSTOOD
    assert p.question == st.QUESTION_WHEN


# ------------------------------------------------------------------ ambiguity and defaults

# (spoken text, now, default chosen, ambiguity, alternatives)
AMBIGUITIES = [
    ("at 8", NOW, "2026-11-19 20:00", A.MERIDIEM, ["2026-11-20 08:00"]),
    ("tomorrow at 7", NOW, "2026-11-20 07:00", A.MERIDIEM, ["2026-11-20 19:00"]),
    ("at 12", NOW, "2026-11-20 00:00", A.MERIDIEM, ["2026-11-20 12:00"]),
    ("tomorrow at 12", NOW, "2026-11-20 12:00", A.MERIDIEM, ["2026-11-21 00:00"]),
    ("at 9 am", NOW, "2026-11-20 09:00", A.PAST_TODAY, []),
    ("next Thursday at 5 pm", NOW, "2026-11-26 17:00", A.SAME_WEEKDAY, ["2026-11-19 17:00"]),
    ("next Thursday at 9 am", NOW, "2026-11-26 09:00", A.SAME_WEEKDAY, []),
    ("on Thursday at 4 pm", NOW, "2026-11-19 16:00", A.SAME_WEEKDAY, ["2026-11-26 16:00"]),
    ("on Thursday at 9 am", NOW, "2026-11-26 09:00", A.SAME_WEEKDAY, []),
    ("next Friday at 9 am", NOW, "2026-11-20 09:00", A.NEXT_WEEKDAY, ["2026-11-27 09:00"]),
    ("next Sunday at 9 am", NOW, "2026-11-22 09:00", A.NEXT_WEEKDAY, ["2026-11-29 09:00"]),
    ("tomorrow at midnight", NOW, "2026-11-21 00:00", A.MIDNIGHT, ["2026-11-20 00:00"]),
    ("tomorrow at 8 am", at("2026-11-20 00:30"), "2026-11-21 08:00",
     A.TOMORROW_AFTER_MIDNIGHT, ["2026-11-20 08:00"]),
    ("in ten", NOW, "2026-11-19 15:17:23", A.UNIT_ASSUMED, []),
]


@pytest.mark.parametrize("text,now,default,ambiguity,alternatives", AMBIGUITIES,
                         ids=[a[0] + " " + a[3].value for a in AMBIGUITIES])
def test_ambiguity_is_named_with_its_stated_default(text, now, default, ambiguity,
                                                    alternatives):
    p = when(text, now=now)
    assert p.ok
    v = p.value
    assert v.instant == at(default)
    assert ambiguity in v.ambiguities
    assert [a.instant for a in v.alternatives] == [at(x) for x in alternatives]
    if alternatives:
        assert p.question == st.QUESTION_CHOICE.format(
            first=st.say_when(v), second=st.say_when(v.alternatives[0]))
    for alt in v.alternatives:
        assert_round_trip(alt)
    assert_round_trip(v)


def test_meridiem_not_flagged_when_only_one_reading_is_left():
    p = when("today at 8")                                   # 8 am has passed
    assert p.value.instant == at("2026-11-19 20:00")
    assert p.value.ambiguities == ()
    assert p.question is None


def test_this_weekday_said_on_that_day_is_today_and_not_flagged():
    p = when("this Thursday at 4 pm")
    assert p.value.instant == at("2026-11-19 16:00")
    assert p.value.ambiguities == ()


def test_next_weekday_in_the_following_week_is_not_flagged():
    assert when("next Monday at 9 am").value.ambiguities == ()


def test_a_question_reply_resolves_the_meridiem():
    p = when("remind me at 8 to call the dentist")
    proposal = st.propose(p.value, st.reminder_subject(p.remainder), at=NOW)
    assert proposal.read_back == "Today at eight in the evening, call the dentist. Shall I set it?"
    out = st.reply(proposal, "no, in the morning", at=NOW + timedelta(seconds=2))
    assert out.state is ConfirmState.PENDING
    assert out.read_back == ("Tomorrow at eight in the morning, call the dentist. "
                             "Shall I set it?")


# ------------------------------------------------------------------ calendar edges and DST

FALL_BACK_EVE = datetime(2026, 10, 31, 20, 0, tzinfo=TZ)     # clocks go back at 2 am Nov 1
SPRING_EVE = datetime(2027, 3, 13, 20, 0, tzinfo=TZ)          # clocks go forward at 2 am Mar 14


def test_dst_named_day_keeps_its_wall_clock_time_after_fall_back():
    v = when("tomorrow at 9 am", now=FALL_BACK_EVE).value
    assert (v.instant.hour, v.instant.minute) == (9, 0)
    assert v.instant.utcoffset() == timedelta(hours=-5)
    assert utc(v.instant) - utc(FALL_BACK_EVE) == timedelta(hours=14)   # 13 wall hours + 1
    assert v.ambiguities == ()
    assert_round_trip(v)


def test_dst_named_day_keeps_its_wall_clock_time_after_spring_forward():
    v = when("tomorrow at 9 am", now=SPRING_EVE).value
    assert (v.instant.hour, v.instant.minute) == (9, 0)
    assert v.instant.utcoffset() == timedelta(hours=-4)
    assert utc(v.instant) - utc(SPRING_EVE) == timedelta(hours=12)
    assert_round_trip(v)


def test_dst_relative_hours_are_elapsed_time():
    v = when("in 6 hours", now=FALL_BACK_EVE).value
    assert utc(v.instant) - utc(FALL_BACK_EVE) == timedelta(hours=6)
    assert (v.instant.hour, v.instant.utcoffset()) == (1, timedelta(hours=-5))
    assert st.say_when(v) == "in six hours"


def test_dst_wall_time_that_does_not_exist_moves_forward_by_the_gap():
    v = when("tomorrow at 2:30 am", now=SPRING_EVE).value
    assert A.SKIPPED_WALL_TIME in v.ambiguities
    assert (v.instant.hour, v.instant.minute) == (3, 30)
    assert v.instant.utcoffset() == timedelta(hours=-4)
    assert st.say_when(v) == "tomorrow at three thirty in the morning"
    assert_round_trip(v)


def test_dst_wall_time_that_occurs_twice_takes_the_first():
    v = when("tomorrow at 1:30 am", now=FALL_BACK_EVE).value
    assert A.REPEATED_WALL_TIME in v.ambiguities
    assert v.instant.utcoffset() == timedelta(hours=-4)                 # daylight time
    assert_round_trip(v)


# (spoken text, now, wall-clock result)
EDGES = [
    ("tomorrow at 9 am", at("2026-11-30 22:00"), "2026-12-01 09:00"),            # month end
    ("in 20 minutes", at("2026-12-31 23:50"), "2027-01-01 00:10"),               # year end
    ("at midnight", at("2026-12-31 23:50"), "2027-01-01 00:00"),
    ("on the 1st at 9 am", at("2026-12-31 23:50"), "2027-01-01 09:00"),
    ("on February 29th at noon", at("2027-11-19 10:00"), "2028-02-29 12:00"),     # leap day
    ("on the 30th at noon", at("2027-01-30 10:00"), "2027-01-30 12:00"),
    ("at midnight", at("2026-11-19 23:59:30"), "2026-11-20 00:00"),
    ("at midnight", at("2026-11-19 00:00"), "2026-11-20 00:00"),
    ("at 12:05 am", NOW, "2026-11-20 00:05"),
    ("tonight at midnight", NOW, "2026-11-20 00:00"),
    ("today at midnight", NOW, "2026-11-20 00:00"),
]


@pytest.mark.parametrize("text,now,expected", EDGES, ids=[f"{e[0]} @ {e[1]:%m-%d %H:%M}"
                                                          for e in EDGES])
def test_calendar_edges(text, now, expected):
    p = when(text, now=now)
    assert p.ok, (p.refusal, p.question)
    assert p.value.instant == at(expected)
    assert_round_trip(p.value)


def test_a_day_of_the_month_the_next_month_lacks_is_refused_not_moved():
    # said on 31 January: this month's 30th has passed and February has none
    p = when("on the 30th at noon", now=at("2027-01-31 10:00"))
    assert p.refusal is Refusal.INVALID_DATE


def test_midnight_named_tonight_is_not_ambiguous():
    assert when("tonight at midnight").value.ambiguities == ()


# ------------------------------------------------------------------ refusals and bounds

# (spoken text, parser, refusal, question)
REFUSALS = [
    ("", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("remind me to call the dentist", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("in a few minutes", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("in a couple of hours", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("at around 8", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("by 8", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("at 25", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("at 8:75", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("at 8 pm in the morning", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("every day at 8 am", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("daily at noon", when, Refusal.NOT_UNDERSTOOD, st.QUESTION_WHEN),
    ("remind me at 5 or 6", when, Refusal.CONFLICTING, st.QUESTION_CONFLICTING),
    ("remind me at 5 to call mom at 6", when, Refusal.CONFLICTING, st.QUESTION_CONFLICTING),
    ("tomorrow at 9 on Friday", when, Refusal.CONFLICTING, st.QUESTION_CONFLICTING),
    ("today at 9 am", when, Refusal.IN_THE_PAST, st.QUESTION_IN_THE_PAST),
    ("at 3 this afternoon", when, Refusal.IN_THE_PAST, st.QUESTION_IN_THE_PAST),
    ("this Thursday at 9 am", when, Refusal.IN_THE_PAST, st.QUESTION_IN_THE_PAST),
    ("tomorrow", when, Refusal.NO_TIME_OF_DAY, "What time tomorrow?"),
    ("tomorrow morning", when, Refusal.NO_TIME_OF_DAY, "What time tomorrow?"),
    ("in three days", when, Refusal.NO_TIME_OF_DAY, "What time on Sunday the twenty-second?"),
    ("in the morning", when, Refusal.NO_TIME_OF_DAY, st.QUESTION_WHEN),
    ("on February 30 at noon", when, Refusal.INVALID_DATE, st.QUESTION_INVALID_DATE),
    ("on February 29th at noon", when, Refusal.INVALID_DATE, st.QUESTION_INVALID_DATE),
    ("Friday the 21st at 10 am", when, Refusal.INVALID_DATE, st.QUESTION_INVALID_DATE),
    ("in 400 days at 9 am", when, Refusal.TOO_FAR_AHEAD, st.QUESTION_TOO_FAR),
    ("in 9000 hours", when, Refusal.TOO_FAR_AHEAD, st.QUESTION_TOO_FAR),
    ("in 0 minutes", when, Refusal.ZERO_DURATION, st.QUESTION_ZERO),
    ("for 0 minutes", length, Refusal.ZERO_DURATION, st.QUESTION_ZERO),
    ("for 25 hours", length, Refusal.DURATION_TOO_LONG,
     "The longest timer I can set is twenty-four hours. How long should it be?"),
    ("timer for 8 pm", length, Refusal.NOT_UNDERSTOOD, st.QUESTION_DURATION),
    ("set a pasta timer", length, Refusal.NOT_UNDERSTOOD, st.QUESTION_DURATION),
    ("for a couple of minutes", length, Refusal.NOT_UNDERSTOOD, st.QUESTION_DURATION),
]


@pytest.mark.parametrize("text,parser,refusal,question", REFUSALS,
                         ids=[f"{r[0] or 'empty'} {r[2].value}" for r in REFUSALS])
def test_refusals_carry_a_reason_and_one_scripted_question(text, parser, refusal, question):
    p = parser(text)
    assert p.value is None and not p.ok
    assert p.refusal is refusal
    assert p.question == question


def test_stricter_limits_can_be_passed():
    assert when("in 3 days at 9 am", max_ahead=timedelta(days=1)).refusal is \
        Refusal.TOO_FAR_AHEAD
    assert length("for 2 hours", max_duration=timedelta(hours=1)).refusal is \
        Refusal.DURATION_TOO_LONG


def test_limits_are_inclusive_at_the_edge():
    assert when("in 8760 hours").ok                                     # exactly 365 days
    assert length("for 24 hours").ok


# ------------------------------------------------------------------ span, remainder, subject

# (spoken text, the time expression's text, remainder, subject, connector)
SUBJECTS = [
    ("remind me tomorrow at 8 to call the dentist", ["tomorrow at 8"],
     "remind me to call the dentist", "call the dentist", "to"),
    ("remind me in two minutes, no, three minutes, to stretch",
     ["in two minutes, no, three minutes"], "remind me to stretch", "stretch", "to"),
    ("remind me tomorrow to call the dentist at 9", ["tomorrow", "at 9"],
     "remind me to call the dentist", "call the dentist", "to"),
    ("remind me about the meeting at 10 am", ["at 10 am"], "remind me about the meeting",
     "the meeting", "about"),
    ("can you remind me that the plumber comes tomorrow at 9 am please",
     ["tomorrow at 9 am"], "can you remind me that the plumber comes please",
     "the plumber comes", "that"),
    ("At 5, remind me to call mom", ["At 5"], "remind me to call mom", "call mom", "to"),
    ("hey vox, remind me in ten minutes to stretch", ["in ten minutes"],
     "hey vox, remind me to stretch", "stretch", "to"),
    ("remind me to take 2 pills at 8 pm", ["at 8 pm"], "remind me to take 2 pills",
     "take 2 pills", "to"),
    ("set a reminder for 6:30 am", ["for 6:30 am"], "set a reminder", "", "to"),
    ("remind me at noon to plan Friday", ["at noon"], "remind me to plan Friday",
     "plan Friday", "to"),
]


@pytest.mark.parametrize("text,exprs,remainder,subject,connector", SUBJECTS,
                         ids=[s[0] for s in SUBJECTS])
def test_span_and_remainder_leave_the_subject(text, exprs, remainder, subject, connector):
    p = when(text)
    assert p.ok
    assert [text[s:e] for s, e in p.spans] == exprs
    assert p.span == (p.spans[0][0], p.spans[-1][1])
    assert p.remainder == remainder
    assert st.reminder_subject(p.remainder) == st.Subject(subject, connector)


def test_rebuilt_sentence_keeps_the_connector():
    # D-3 in the design: today's parser says "I'll remind you call the dentist"
    s = st.reminder_subject(when("remind me tomorrow at 8 to call the dentist").remainder)
    assert f"I'll remind you {s.connector} {s.text}" == "I'll remind you to call the dentist"


# ------------------------------------------------------------------ read-back words

READ_BACKS = [
    ("in 3 minutes", NOW, "in three minutes"),
    ("tomorrow at 8am", NOW, "tomorrow at eight in the morning"),
    ("at 8", NOW, "today at eight in the evening"),
    ("on Friday at noon", at("2026-08-17 09:00"), "on Friday the twenty-first at noon"),
    ("next Monday at 9 pm", NOW, "on Monday the twenty-third at nine at night"),
    ("on the 30th at 9:45 am", NOW,
     "on Monday the thirtieth of November at nine forty-five in the morning"),
    ("at 12:30 pm", NOW, "tomorrow at twelve thirty in the afternoon"),
    ("tomorrow at 12:05 am", NOW, "tomorrow at twelve oh five in the morning"),
    ("tomorrow at midnight", NOW, "tomorrow at midnight, at the end of the day"),
    ("at midnight", NOW, "tonight at midnight"),
    ("tomorrow at 8 am", at("2026-11-20 00:30"), "on Saturday the twenty-first at eight in the "
                                                 "morning"),
    ("at 8 am", at("2026-11-20 00:30"), "today at eight in the morning"),
    ("in an hour and a half", NOW, "in an hour and a half"),
    ("in 90 seconds", NOW, "in one minute and thirty seconds"),
    ("at 5:15 pm", NOW, "today at five fifteen in the afternoon"),
    ("at 6 pm", NOW, "today at six in the evening"),
    ("at 11 pm", NOW, "today at eleven at night"),
]


@pytest.mark.parametrize("text,now,words", READ_BACKS, ids=[r[0] for r in READ_BACKS])
def test_read_back_words(text, now, words):
    v = when(text, now=now).value
    assert st.say_when(v) == words
    assert_round_trip(v)


@pytest.mark.parametrize("seconds,words,announce", [
    (600, "ten minutes", "Ten minute timer, starting now."),
    (60, "one minute", "One minute timer, starting now."),
    (3600, "an hour", "One hour timer, starting now."),
    (30, "thirty seconds", "Thirty second timer, starting now."),
    (90, "one minute and thirty seconds",
     "Timer for one minute and thirty seconds, starting now."),
    (5400, "an hour and a half", "Ninety minute timer, starting now."),
    (9000, "two and a half hours", "One hundred and fifty minute timer, starting now."),
    (3661, "an hour, one minute and one second",
     "Timer for an hour, one minute and one second, starting now."),
])
def test_duration_words_and_timer_announcement(seconds, words, announce):
    d = timedelta(seconds=seconds)
    assert st.say_duration(d) == words
    v = length(words).value
    assert v.duration == d
    assert st.announce_timer(v) == announce
    assert length(announce).value.duration == d


def test_every_clock_read_back_names_its_part_of_day():
    period = re.compile(r"(in the morning|in the afternoon|in the evening|at night|noon|"
                        r"midnight)$")
    for text, *_ in FORMS + [(c[0],) for c in CORRECTIONS]:
        v = when(text).value
        if not v.relative:
            words = st.say_when(v).removesuffix(", at the end of the day")
            assert period.search(words), words


def test_read_back_is_built_from_the_value_not_the_spelling():
    spellings = ["at 8 pm", "at eight in the evening", "at 20:00", "at 8:00 p.m.",
                 "at eight o'clock tonight", "at 7 pm, no, 8 pm"]
    lines = {st.read_back_reminder(when(s).value, st.Subject("call the dentist"))
             for s in spellings}
    assert lines == {"Today at eight in the evening, call the dentist. Shall I set it?"}


def test_number_words():
    assert [st.number_words(n) for n in (0, 7, 13, 45, 100, 120, 1005, 8760)] == [
        "zero", "seven", "thirteen", "forty-five", "one hundred", "one hundred and twenty",
        "one thousand and five", "eight thousand seven hundred and sixty"]


# ------------------------------------------------------------------ validated arguments only


def test_read_back_only_from_validated_arguments_refuses_other_objects():
    good = when("in three minutes").value
    forged = st.SpokenTime(st.Kind.INSTANT, NOW + timedelta(minutes=3), timedelta(minutes=3),
                           True, (), (), False, (), NOW, TZ, st.MAX_REMINDER_AHEAD,
                           good._expr, object())
    for bad in (forged, "in three minutes", NOW + timedelta(minutes=3), None):
        with pytest.raises(TypeError):
            st.read_back_reminder(bad, st.Subject("stretch"))
        with pytest.raises(TypeError):
            st.say_when(bad)
    with pytest.raises(TypeError):
        st.read_back_reminder(good, "stretch")                       # a raw string subject


def test_read_back_only_from_validated_arguments_checks_bounds_again():
    good = when("in three minutes").value
    for instant in (NOW - timedelta(minutes=1), NOW + timedelta(days=400)):
        with pytest.raises(ValueError):
            st.read_back_reminder(replace(good, instant=instant, relative=False),
                                  st.Subject("stretch"))
    timer = length("for ten minutes").value
    with pytest.raises(ValueError):
        st.announce_timer(replace(timer, duration=timedelta(hours=30)))
    with pytest.raises(TypeError):
        st.announce_timer(good)
    with pytest.raises(TypeError):
        st.read_back_reminder(timer, None)
    with pytest.raises(ValueError):
        st.read_back_reminder(good, st.Subject("x" * (st.MAX_SUBJECT_CHARS + 1)))


def test_now_must_be_aware_and_is_never_read_from_the_system_clock():
    with pytest.raises(ValueError):
        st.parse_when("in two minutes", now=datetime(2026, 11, 19, 15, 0), tz=TZ)
    source = inspect.getsource(st)
    for call in ("datetime.now(", "date.today(", "time.time(", "datetime.utcnow("):
        assert call not in source
    later = NOW + timedelta(days=3)
    assert when("in two minutes", now=later).value.instant == later + timedelta(minutes=2)


# ------------------------------------------------------------------ the read-back state machine


def test_validation_gate_sentence():
    """Design section 3: "remind me in two minutes, no, three minutes, to stretch"; hear
    "In three minutes, stretch. Shall I set it?"; say yes."""
    p = when("hey vox, remind me in two minutes, no, three minutes, to stretch")
    assert p.ok and p.value.corrected
    assert utc(p.value.instant) == utc(NOW) + timedelta(minutes=3)
    proposal = st.propose(p.value, st.reminder_subject(p.remainder), at=NOW)
    assert proposal.read_back == "In three minutes, stretch. Shall I set it?"
    assert proposal.state is ConfirmState.PENDING
    done = st.reply(proposal, "yes", at=NOW + timedelta(seconds=2))
    assert done.state is ConfirmState.CONFIRMED
    assert done.last_reply is ReplyKind.YES
    assert utc(done.when.instant) == utc(NOW) + timedelta(minutes=3)


def _stretch() -> st.Proposal:
    p = when("remind me in two minutes, no, three minutes, to stretch")
    return st.propose(p.value, st.reminder_subject(p.remainder), at=NOW)


@pytest.mark.parametrize("text", ["yes", "yes please", "yeah", "sure", "okay", "go ahead",
                                  "set it", "that's right", "yep, thanks"])
def test_reply_yes_confirms(text):
    out = st.reply(_stretch(), text, at=NOW + timedelta(seconds=2))
    assert out.state is ConfirmState.CONFIRMED and out.last_reply is ReplyKind.YES


@pytest.mark.parametrize("text,answer_normally", [
    ("no", False), ("nope", False), ("no thanks", False), ("cancel", False),
    ("never mind", False), ("don't", False), ("no, what's the weather", True),
])
def test_reply_no_declines(text, answer_normally):
    out = st.reply(_stretch(), text, at=NOW + timedelta(seconds=2))
    assert out.state is ConfirmState.DECLINED and out.last_reply is ReplyKind.NO
    assert out.answer_normally is answer_normally


@pytest.mark.parametrize("text,minutes", [
    ("no, eight", 8), ("no, five minutes", 5), ("make it ten", 10), ("eight", 8),
    ("no, in an hour", 60), ("actually, twenty minutes please", 20),
])
def test_reply_correction_reads_back_again(text, minutes):
    later = NOW + timedelta(seconds=4)
    out = st.reply(_stretch(), text, at=later)
    assert out.state is ConfirmState.PENDING
    assert out.last_reply is ReplyKind.CORRECTION
    assert out.asked_at == later
    assert utc(out.when.instant) == utc(later) + timedelta(minutes=minutes)
    assert out.read_back == f"In {st.say_duration(timedelta(minutes=minutes))}, stretch. " \
                            "Shall I set it?"
    assert out.when.corrected
    final = st.reply(out, "yes", at=later + timedelta(seconds=2))
    assert final.state is ConfirmState.CONFIRMED


def test_reply_correction_to_a_clock_time():
    p = when("remind me tomorrow at 8 am to call the dentist")
    proposal = st.propose(p.value, st.reminder_subject(p.remainder), at=NOW)
    out = st.reply(proposal, "no, nine", at=NOW + timedelta(seconds=2))
    assert out.when.instant == at("2026-11-20 09:00")                 # keeps tomorrow and am
    assert out.read_back == "Tomorrow at nine in the morning, call the dentist. Shall I set it?"
    out = st.reply(proposal, "no, Saturday at 7 pm", at=NOW + timedelta(seconds=2))
    assert out.when.instant == at("2026-11-21 19:00")


def test_reply_correction_that_is_out_of_bounds_declines_with_its_question():
    out = st.reply(_stretch(), "no, 400 days", at=NOW + timedelta(seconds=2))
    assert out.state is ConfirmState.DECLINED


@pytest.mark.parametrize("text", ["what's the weather", "okay, what's the weather tomorrow",
                                  "tell me a joke", "how are you"])
def test_reply_unrelated_turn_declines_and_is_answered_normally(text):
    out = st.reply(_stretch(), text, at=NOW + timedelta(seconds=2))
    assert out.state is ConfirmState.DECLINED
    assert out.last_reply is ReplyKind.UNRELATED
    assert out.answer_normally


def test_silence_inside_the_window_waits_and_past_it_declines():
    proposal = _stretch()
    assert st.silence(proposal, at=NOW + st.READBACK_WINDOW).state is ConfirmState.PENDING
    out = st.silence(proposal, at=NOW + st.READBACK_WINDOW + timedelta(seconds=1))
    assert out.state is ConfirmState.DECLINED and out.last_reply is ReplyKind.SILENCE


def test_a_yes_after_the_window_does_not_confirm():
    out = st.reply(_stretch(), "yes", at=NOW + st.READBACK_WINDOW + timedelta(seconds=1))
    assert out.state is ConfirmState.DECLINED and out.last_reply is ReplyKind.SILENCE
    assert out.answer_normally


def test_a_settled_proposal_takes_no_more_replies():
    done = st.reply(_stretch(), "yes", at=NOW)
    with pytest.raises(ValueError):
        st.reply(done, "no", at=NOW)


def test_correct_returns_none_when_the_reply_has_no_time():
    v = when("in three minutes").value
    assert st.correct(v, "yes please", now=NOW) is None
    assert st.correct(v, "what time is it in London", now=NOW) is None


# ------------------------------------------------------------------ day ranges

# (text, start, end, words)
RANGES = [
    ("what's on my calendar today", "2026-11-19 00:00", "2026-11-20 00:00", "today"),
    ("what's on my calendar tomorrow", "2026-11-20 00:00", "2026-11-21 00:00", "tomorrow"),
    ("tomorrow afternoon", "2026-11-20 12:00", "2026-11-20 18:00", "tomorrow afternoon"),
    ("this afternoon", "2026-11-19 12:00", "2026-11-19 18:00", "this afternoon"),
    ("tonight", "2026-11-19 18:00", "2026-11-20 05:00", "tonight"),
    ("on Saturday morning", "2026-11-21 05:00", "2026-11-21 12:00",
     "on Saturday the twenty-first, in the morning"),
    ("the day after tomorrow", "2026-11-21 00:00", "2026-11-22 00:00",
     "on Saturday the twenty-first"),
    ("this week", "2026-11-19 00:00", "2026-11-23 00:00", "this week"),
    ("next week", "2026-11-23 00:00", "2026-11-30 00:00", "next week"),
    ("this weekend", "2026-11-21 00:00", "2026-11-23 00:00", "this weekend"),
    ("on the 30th", "2026-11-30 00:00", "2026-12-01 00:00",
     "on Monday the thirtieth of November"),
]


@pytest.mark.parametrize("text,start,end,words", RANGES, ids=[r[0] for r in RANGES])
def test_day_ranges(text, start, end, words):
    r = st.parse_day_range(text, now=NOW, tz=TZ)
    assert r is not None
    assert (r.start, r.end, r.words) == (at(start), at(end), words)


def test_day_range_absent_and_across_dst():
    assert st.parse_day_range("am I free at three", now=NOW, tz=TZ) is None
    r = st.parse_day_range("tomorrow", now=FALL_BACK_EVE, tz=TZ)
    assert utc(r.end) - utc(r.start) == timedelta(hours=25)            # the long day
    r = st.parse_day_range("next Thursday", now=NOW, tz=TZ)
    assert r.ambiguities == (A.SAME_WEEKDAY,)
