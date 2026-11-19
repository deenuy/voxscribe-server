"""Reminders and timers in a voice turn (PL45, TOOLS-CONNECTORS.md sections 2.5 and 2.11).

Deterministic from end to end: an intent matcher picks the turn up, voice/spoken_time.py reads
the time, the tool runtime (app/local/tools/runtime.py) checks and records every step, and the
store (voice/reminder_store.py) holds the rows. No model is asked, and nothing here is on the
delegate lane: these are fast-path tools with a timeout in their records.

    intent            words                                  tool                 confirm
    set a reminder    "remind me tomorrow at 8 to ..."       reminder.create      ask
    list reminders    "what reminders do I have?"            reminder.list        none
    cancel reminder   "cancel my reminder to ..."            reminder.cancel      ask
    start a timer     "set a timer for ten minutes"          timer.start          announce
    timer status      "how long is left on my timer?"        timer.status         none
    cancel a timer    "cancel the timer"                     timer.cancel         announce

A reminder, step by step, and the audit row each step leaves (tool_calls, by status):

    1. The registry is asked first: in classroom (timers only, TC12) the turn gets the
       timers-only line and a `refused` row, and nothing is parsed or stored.
    2. The time is parsed with the turn's self-corrections applied ("in two minutes, no, three").
       No time, a time refused by its bounds, or a time with two readings ("at 8") asks its one
       scripted question first and leaves an `invalid_args` row with the parser's reason; the
       next turn's answer is read as the time (or the choice) and the flow goes on at 3. A
       reminder with no words asks "What should I remind you about?" the same way.
    3. The read-back is built from the validated arguments only ("Tomorrow at eight in the
       morning, call the dentist. Shall I set it?"). Before it is offered it is put to the
       session's gate with the turn's words, as the reply stage will: a read-back the gate would
       refuse (senior care's dosing rule, TC6) is never offered, nothing is proposed, and a
       `refused` row says gate_refused. Otherwise a `proposed` row is written and the read-back
       is spoken; when the time falls inside quiet hours a warning is spoken before it (TC5).
    4. The next turn answers it (spoken_time.reply). The read-back counts only if that turn's
       history shows it was heard whole (an interrupted or replaced read-back confirms nothing).
         yes                 reminder.create runs with confirm_turn_id = this turn: the row is
                             written, an `executed` row is the receipt, and "Reminder set." is
                             spoken (the gate passes the claim only with this turn's receipt).
                             A store failure speaks the failure line and leaves a `failed` row
         no                  `declined` (no); "Okay, I won't set it.", or the rest of the turn
                             is answered normally when it said more than no
         "no, nine"          `declined` (corrected) for the old one, a new `proposed` row and a
                             new read-back
         silence             past the read-back window (measured by the host from the end of the
                             read-back to the start of the reply), or the session ends with no
                             reply: `declined` (silence or session_ended), nothing is set, and a
                             late turn is answered normally
         anything else       `declined` (unrelated), and the turn is answered normally

A timer is started at once and announced ("Ten minute timer, starting now.") from its receipt;
"cancel the timer" undoes it. Cancelling a reminder reads it back and waits for a yes, the same
way. Lists and the time left on a timer are read from the rows.

A reminder's words are the tenant's data: they go into the row, the read-back and the receipt's
success line, never into a log line; the audit row keeps their digest and length.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.local.tools import audit, builtin
from app.local.tools import runtime as tool_runtime
from app.local.tools.registry import canonical_mode
from app.local.tools.runtime import CallContext, ToolFailed, ToolResult
from app.local.voice import reminder_store as rs
from app.local.voice import spoken_time as st

log = logging.getLogger("app.local.voice.reminder_dialog")

# ── scripted lines (fixed text; each passes the gate like any other sentence) ──────────────

DECLINED_LINE = "Okay, I won't set it."
CANCEL_KEPT_LINE = "Okay, I'll keep it."
RECURRING_LINE = "I can't set repeating reminders yet."
QUIET_HOURS_WARNING = ("That's during quiet hours, so it will show as a notification without"
                       " sound.")
PASSED_LINE = "That time has already passed, so I didn't set it."
NEUTRAL_REFUSED_LINE = "I can't set a reminder with those words."
NO_REMINDERS_LINE = "You have no reminders coming up."
REMINDERS_INTRO = "You have {count} coming up."
REMINDER_ITEM = "{when}, {text}."
WHICH_REMINDER_LINE = ("You have {count}. Say cancel, and what the reminder is about.")
NO_SUCH_REMINDER_LINE = "I couldn't find that reminder."
NO_TIMERS_LINE = "You have no timers running."
TIMER_LEFT = "{remaining} left on your {length} timer."
TIMER_DONE = "Your {length} timer is done."
TIMERS_INTRO = "You have {count} running."

# Quiet hours and due items (TC5): what a due item does inside the window, per product mode.
# Core's Profile has no field for it yet; a profile that grows `quiet_hours_due` wins.
QUIET_DUE_NOTIFY = "notify"          # no sound: the notification only, spoken at the next session
QUIET_DUE_SPEAK = "speak"
QUIET_DUE_BY_MODE = {"personal": QUIET_DUE_NOTIFY, "ideascribe": QUIET_DUE_NOTIFY,
                     "senior_care": QUIET_DUE_NOTIFY, "classroom": QUIET_DUE_NOTIFY}

# ── the intent matcher ─────────────────────────────────────────────────────────────────────

SET_REMINDER = "set_reminder"
LIST_REMINDERS = "list_reminders"
CANCEL_REMINDER = "cancel_reminder"
SET_TIMER = "set_timer"
TIMER_STATUS = "timer_status"
CANCEL_TIMER = "cancel_timer"

_I = re.IGNORECASE
_LEAD = (r"^\s*(?:(?:hey|hi|ok|okay)\s+vox\b[\s,]*)?(?:(?:please|can you|could you|would you"
         r"|will you|i want you to|i'd like you to|i need you to)[\s,]+)*")
_LEADS_WITH_REMINDER = re.compile(
    _LEAD + r"(?:remind\s+me|(?:set|make|add|create)\s+(?:me\s+)?(?:a\s+)?reminder)\b", _I)
_CANCEL_TIMER = re.compile(
    r"\b(?:cancel|stop|end|delete|clear|turn\s+off|kill|reset)\b[^.?!]*\btimers?\b", _I)
_CANCEL_REMINDER = re.compile(
    r"\b(?:cancel|delete|remove|clear|forget|drop)\b[^.?!]*\breminders?\b"
    r"|\bdon'?t\s+remind\s+me\b", _I)
_TIMER_STATUS = re.compile(
    r"\b(?:how\s+(?:long|much\s+time|many\s+(?:minutes|seconds))|time)\b[^.?!]*"
    r"\b(?:left|remaining)\b[^.?!]*\btimers?\b"
    r"|\btimers?\b[^.?!]*\b(?:left|remaining)\b"
    r"|\b(?:what|which|any)\s+timers?\b|\bmy\s+timers\b", _I)
_LIST_REMINDERS = re.compile(
    r"\b(?:what(?:'s|\s+are|\s+do\s+i\s+have)?\s+(?:(?:my|upcoming|the)\s+)?reminders"
    r"|(?:list|show|read|tell)\s+(?:me\s+)?(?:my\s+|all\s+)?reminders"
    r"|upcoming\s+reminders|any\s+reminders|my\s+reminders)\b", _I)
_SET_TIMER = re.compile(
    r"\b(?:set|start|make|put\s+on|begin|run)\b[^.?!]*\btimer\b|\btimer\s+for\b"
    r"|\b[\w-]+[\s-](?:second|minute|hour)s?\s+timer\b", _I)
_SET_REMINDER = re.compile(
    r"\bremind\s+me\b|\b(?:set|make|add|create)\s+(?:me\s+)?(?:a\s+)?reminder\b"
    r"|\breminder\s+(?:to|for|about|that)\b", _I)


def match_intent(text: str) -> str | None:
    """Which reminder or timer intent the turn is, or None. Pure, deterministic."""
    if not text or not text.strip():
        return None
    if _LEADS_WITH_REMINDER.search(text):
        return SET_REMINDER
    if _CANCEL_TIMER.search(text):
        return CANCEL_TIMER
    if _CANCEL_REMINDER.search(text):
        return CANCEL_REMINDER
    if _TIMER_STATUS.search(text):
        return TIMER_STATUS
    if _LIST_REMINDERS.search(text):
        return LIST_REMINDERS
    if _SET_TIMER.search(text):
        return SET_TIMER
    if _SET_REMINDER.search(text):
        return SET_REMINDER
    return None


# ── helpers ───────────────────────────────────────────────────────────────────────────────

_PERIODS = {"morning": "in the morning", "afternoon": "in the afternoon",
            "evening": "in the evening", "night": "at night", "tonight": "tonight"}
_FIRST = frozenset({"first", "former"})
_SECOND = frozenset({"second", "latter", "other", "last"})
_STOP = frozenset({"the", "a", "an", "my", "to", "for", "about", "of", "and", "reminder",
                   "reminders", "cancel", "delete", "remove", "clear", "forget", "drop", "one",
                   "that", "this", "please", "me", "remind", "don't", "dont", "i", "it", "at",
                   "on", "in"})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (text or "").lower())).strip()


def _words(text: str) -> set[str]:
    return {w for w in _norm(text).split() if w not in _STOP and len(w) > 2}


def _capital(s: str) -> str:
    return s[:1].upper() + s[1:]


def _count(n: int, noun: str) -> str:
    return f"{st.number_words(n)} {noun}{'' if n == 1 else 's'}"


def _iso_utc(when: datetime) -> str:
    return when.astimezone(UTC).isoformat()


def _heard(history: Sequence[dict] | None, line: str) -> bool:
    """True when the last thing the assistant said, as recorded (what was heard after an
    interruption, the gate's line after a replacement), contains `line` whole."""
    if not history:
        return False
    last = history[-1]
    if last.get("role") != "assistant":
        return False
    return _norm(line) in _norm(str(last.get("content", "")))


def _remaining_words(left: timedelta) -> str:
    secs = max(1, math.ceil(left.total_seconds()))
    if secs >= 120:
        secs = int(round(secs / 60)) * 60
    return st.say_duration(timedelta(seconds=secs))


def quiet_hours_policy(app):
    """(the quiet-hours window or None, what a due item does inside it). The window is the one
    the wake listener uses (the product profile, or the operator's settings)."""
    from app.local.voice.host import VoiceConfig  # lazy: the host imports adapters lazily too

    state = getattr(app, "state", None)
    profile = getattr(state, "profile", None)
    settings = getattr(state, "settings", None)
    try:
        config = (VoiceConfig.from_profile(profile, settings=settings) if profile is not None
                  else VoiceConfig.from_settings(settings))
        window = config.quiet_hours
    except Exception:  # noqa: BLE001 - no window that can be read is no window
        window = None
    mode = canonical_mode(getattr(profile, "mode", None) or "personal")
    due = getattr(profile, "quiet_hours_due", None) or QUIET_DUE_BY_MODE.get(mode,
                                                                            QUIET_DUE_NOTIFY)
    return window, due


def in_quiet_hours(window, at: datetime) -> bool:
    if window is None:
        return False
    try:
        return bool(window.active(at))
    except TypeError:          # a window that only knows "now"
        return False


@dataclass(frozen=True)
class DialogReply:
    """What the dialog says this turn. With `answer_normally` the turn is also answered as any
    other turn would be (the sentences, if any, come first)."""

    sentences: tuple[str, ...]
    answer_normally: bool = False
    route: str = "reminder"


@dataclass
class _Readback:
    """reminder.create read back, waiting for the next turn."""

    proposal: st.Proposal
    args: dict
    ctx: CallContext
    line: str


@dataclass
class _CancelAsk:
    item: rs.Item
    args: dict
    ctx: CallContext
    line: str
    asked_at: datetime


@dataclass
class _Clarify:
    """A question asked before a read-back: the time (value None), the choice between two
    readings (value with alternatives), or what the reminder is about (subject None)."""

    kind: str                          # "reminder" or "timer"
    subject: st.Subject | None
    value: st.SpokenTime | None
    asked_at: datetime
    need: str = "time"                 # time | choice | subject


Pending = _Readback | _CancelAsk | _Clarify


@dataclass
class ReminderDialog:
    """One per delegate. Holds at most one open question per session, in memory: a restart or a
    session's end forgets it (declined, never set)."""

    app: object
    store: rs.ReminderStore
    clock: Callable[[], datetime] | None = None     # aware UTC now; the system clock if None
    gate: object | None = None                      # the session's gate if None
    notify: Callable[[], None] | None = None        # told when a row changes (the scheduler)
    _pending: dict[str, Pending] = field(default_factory=dict)

    # ── time and place ──

    def now(self) -> datetime:
        if self.clock is not None:
            return self.clock()
        injected = getattr(getattr(self.app, "state", None), "clock", None)
        return injected() if injected is not None else datetime.now(UTC)

    def zone_name(self) -> str:
        return rs.device_zone(getattr(getattr(self.app, "state", None), "settings", None))

    def _changed(self) -> None:
        notify = self.notify or getattr(getattr(getattr(self.app, "state", None),
                                                "due_scheduler", None), "notify", None)
        if notify is not None:
            try:
                notify()
            except Exception:  # noqa: BLE001 - the scheduler re-plans on its own next wake
                log.warning("reminder dialog: scheduler not told of a change")

    def _gate(self):
        if self.gate is None:
            from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy

            self.gate = CorePolicyGate(session_policy(self.app))
        return self.gate

    def speakable(self, line: str, user_text: str = "") -> bool:
        """Would the session's gate pass `line` in a turn whose words were `user_text`? A gate
        that fails says no (fail closed)."""
        from voxscribe.session.ports import GATE_PASS

        try:
            verdict = self._gate().check(line, (), user_text=user_text)
        except Exception:  # noqa: BLE001 - fail closed
            return False
        return verdict.decision == GATE_PASS

    # ── session lifecycle ──

    def has_pending(self, session_id: str) -> bool:
        return session_id in self._pending

    async def session_closed(self, session_id: str) -> None:
        """The session ended with a read-back unanswered: it is declined, and nothing is set."""
        pending = self._pending.pop(session_id, None)
        if isinstance(pending, _Readback):
            await tool_runtime.decline(self.app, builtin.REMINDER_CREATE, pending.args,
                                       pending.ctx, "session_ended")
        elif isinstance(pending, _CancelAsk):
            await tool_runtime.decline(self.app, builtin.REMINDER_CANCEL, pending.args,
                                       pending.ctx, "session_ended")

    def forget(self) -> None:
        """The device now works for someone else: drop every open question unanswered."""
        self._pending.clear()

    # ── one turn ──

    async def handle(self, text: str, *, session_id: str, turn_id: str | None,
                     history: Sequence[dict] | None = None,
                     waited: timedelta | None = None) -> DialogReply | None:
        """The dialog's answer to this turn, or None when the turn is not one of its own."""
        ctx = CallContext(session_id=session_id, turn_id=turn_id)
        pending = self._pending.pop(session_id, None)
        if pending is not None:
            out = await self._resume(pending, text, ctx, history, waited)
            if out is not None and not out.answer_normally:
                return out
            prefix = out.sentences if out is not None else ()
            fresh = await self._fresh(text, ctx)
            if fresh is not None:
                return DialogReply(prefix + fresh.sentences, fresh.answer_normally, fresh.route)
            return DialogReply(prefix, answer_normally=True) if prefix else None
        return await self._fresh(text, ctx)

    async def _fresh(self, text: str, ctx: CallContext) -> DialogReply | None:
        intent = match_intent(text)
        if intent is None:
            return None
        handler = {SET_REMINDER: self._set_reminder, LIST_REMINDERS: self._list_reminders,
                   CANCEL_REMINDER: self._cancel_reminder, SET_TIMER: self._set_timer,
                   TIMER_STATUS: self._timer_status, CANCEL_TIMER: self._cancel_timer}[intent]
        reply = await handler(text, ctx)
        log.info("reminder dialog: intent=%s session_id=%s", intent, ctx.session_id)
        return reply

    # ── reminders ──

    async def _set_reminder(self, text: str, ctx: CallContext) -> DialogReply:
        refused = await tool_runtime.preflight(self.app, builtin.REMINDER_CREATE, ctx)
        if refused is not None:
            return DialogReply((refused.line,))
        now = self.now()
        tz = rs.zone(self.zone_name())
        if st.is_recurring(text):
            await tool_runtime.record(self.app, builtin.REMINDER_CREATE, ctx,
                                      status=audit.INVALID_ARGS, reason="time:recurring")
            return DialogReply((RECURRING_LINE,))
        parsed = st.parse_when(text, now=now, tz=tz)
        subject = st.reminder_subject(parsed.remainder)
        return await self._after_parse("reminder", parsed, subject, text, ctx, now)

    async def _after_parse(self, kind: str, parsed: st.TimeParse, subject: st.Subject | None,
                           text: str, ctx: CallContext, now: datetime) -> DialogReply:
        tool = builtin.REMINDER_CREATE if kind == "reminder" else builtin.TIMER_START
        if not parsed.ok:
            await tool_runtime.record(self.app, tool, ctx, status=audit.INVALID_ARGS,
                                      reason=f"time:{parsed.refusal.value}")
            self._pending[ctx.session_id] = _Clarify(kind, subject, None, now)
            return DialogReply((parsed.question,))
        if parsed.value.alternatives and kind == "reminder":
            await tool_runtime.record(self.app, tool, ctx, status=audit.INVALID_ARGS,
                                      reason=f"time:ambiguous_{parsed.value.ambiguities[0].value}"
                                      if parsed.value.ambiguities else "time:ambiguous")
            self._pending[ctx.session_id] = _Clarify(kind, subject, parsed.value, now,
                                                     need="choice")
            return DialogReply((parsed.question,))
        if kind == "timer":
            return await self._start_timer(parsed.value, ctx, now)
        return await self._offer(parsed.value, subject, text, ctx, now)

    async def _offer(self, value: st.SpokenTime, subject: st.Subject | None, user_text: str,
                     ctx: CallContext, now: datetime) -> DialogReply:
        if subject is None or not subject.text:
            await tool_runtime.record(self.app, builtin.REMINDER_CREATE, ctx,
                                      status=audit.INVALID_ARGS, reason="subject:missing")
            self._pending[ctx.session_id] = _Clarify("reminder", None, value, now,
                                                     need="subject")
            return DialogReply((st.QUESTION_SUBJECT,))
        proposal = st.propose(value, subject, at=now)
        args = {"text": subject.text, "fire_at": _iso_utc(value.instant),
                "when": _capital(st.say_when(value)), "time_zone": self.zone_name()}
        # Put to the gate as the reply stage will put it: with the turn's words. A read-back
        # the gate would refuse is never offered (TC6: fail closed), so no yes can set it.
        if not self.speakable(proposal.read_back, user_text):
            await tool_runtime.record(self.app, builtin.REMINDER_CREATE, ctx,
                                      status=audit.REFUSED, reason="gate_refused", args=args)
            return DialogReply((NEUTRAL_REFUSED_LINE,))
        outcome = await tool_runtime.propose(self.app, builtin.REMINDER_CREATE, args, ctx)
        if outcome.status != audit.PROPOSED:
            return DialogReply((outcome.line,))
        line = outcome.line
        if line != proposal.read_back:      # one wording, from one validated value
            log.warning("reminder dialog: record and parser read-backs differ; record's used")
        self._pending[ctx.session_id] = _Readback(proposal, args, ctx, line)
        window, due = quiet_hours_policy(self.app)
        if due == QUIET_DUE_NOTIFY and in_quiet_hours(window, value.instant.astimezone(value.tz)):
            return DialogReply((QUIET_HOURS_WARNING, line))
        return DialogReply((line,))

    async def _resume(self, pending: Pending, text: str, ctx: CallContext,
                      history: Sequence[dict] | None,
                      waited: timedelta | None) -> DialogReply | None:
        now = self.now()
        if isinstance(pending, _Readback):
            return await self._answer_readback(pending, text, ctx, history, waited, now)
        if isinstance(pending, _CancelAsk):
            return await self._answer_cancel(pending, text, ctx, history, waited, now)
        return await self._answer_question(pending, text, ctx, waited, now)

    async def _answer_readback(self, pending: _Readback, text: str, ctx: CallContext,
                               history, waited, now: datetime) -> DialogReply | None:
        name = builtin.REMINDER_CREATE
        if not _heard(history, pending.line):
            await tool_runtime.decline(self.app, name, pending.args, pending.ctx,
                                       "readback_unheard")
            return None
        out = st.reply(pending.proposal, text, at=now, waited=waited)
        if out.state is st.ConfirmState.CONFIRMED:
            return await self._create(pending, ctx, now)
        if out.state is st.ConfirmState.PENDING:          # "no, nine": read back again
            await tool_runtime.decline(self.app, name, pending.args, pending.ctx, "corrected")
            return await self._offer(out.when, out.subject, text, ctx, now)
        reason = {st.ReplyKind.NO: "no", st.ReplyKind.SILENCE: "silence",
                  st.ReplyKind.UNRELATED: "unrelated",
                  st.ReplyKind.CORRECTION: "correction_refused"}[out.last_reply]
        await tool_runtime.decline(self.app, name, pending.args, pending.ctx, reason)
        if out.last_reply is st.ReplyKind.CORRECTION and out.question:
            self._pending[ctx.session_id] = _Clarify("reminder", pending.proposal.subject, None,
                                                     now)
            return DialogReply((out.question,))
        if out.last_reply is st.ReplyKind.NO and not out.answer_normally:
            return DialogReply((DECLINED_LINE,))
        return None

    async def _create(self, pending: _Readback, ctx: CallContext,
                      now: datetime) -> DialogReply:
        args = pending.args
        yes = CallContext(session_id=ctx.session_id, turn_id=ctx.turn_id,
                          confirm_turn_id=ctx.turn_id)
        if datetime.fromisoformat(args["fire_at"]) <= now:
            await tool_runtime.decline(self.app, builtin.REMINDER_CREATE, args, pending.ctx,
                                       "passed")
            return DialogReply((PASSED_LINE,))
        store = self.store

        async def create(validated: dict) -> ToolResult:
            row = await store.add(kind=rs.KIND_REMINDER, text=validated["text"],
                                  fire_at=datetime.fromisoformat(validated["fire_at"]),
                                  time_zone=validated["time_zone"], created_at=now,
                                  session_id=ctx.session_id, turn_id=ctx.turn_id)
            return ToolResult(fields={"reminder_id": row.id}, ref=row.id)

        outcome = await tool_runtime.call(self.app, builtin.REMINDER_CREATE, args, yes, create)
        await self._receipted(outcome)
        return DialogReply((outcome.line,))

    async def _receipted(self, outcome) -> None:
        if outcome.receipt is not None and outcome.receipt.result_ref:
            try:
                await self.store.set_receipt(outcome.receipt.result_ref, outcome.receipt.call_id)
            except Exception:  # noqa: BLE001 - the row stands; the audit row is the receipt
                log.warning("reminder dialog: receipt id not written on the row")
        if outcome.executed:
            self._changed()

    async def _answer_question(self, pending: _Clarify, text: str, ctx: CallContext, waited,
                               now: datetime) -> DialogReply | None:
        silent = (waited if waited is not None else now - pending.asked_at) > st.READBACK_WINDOW
        if silent:
            return None
        kind, more = st.classify_reply(text)
        tz = rs.zone(self.zone_name())
        if pending.need == "subject":
            if kind is st.ReplyKind.NO:
                return DialogReply((DECLINED_LINE,)) if not more else None
            # "to check the oven", "about the oven": read as the rest of "remind me ...".
            said = text if _SET_REMINDER.search(text) else f"remind me {text.strip()}"
            subject = st.reminder_subject(said)
            if not subject.text:
                return None
            return await self._offer(pending.value, subject, text, ctx, now)
        if pending.need == "choice":
            chosen = _choose(pending.value, text, now)
            if chosen is None:
                if kind is st.ReplyKind.NO and not more:
                    return DialogReply((DECLINED_LINE,))
                return None
            return await self._offer(chosen, pending.subject, text, ctx, now)
        # The time: the answer alone, or with "at" for a bare hour ("nine").
        if pending.kind == "timer":
            parsed = st.parse_duration(text, now=now, tz=tz)
            if not parsed.ok:
                parsed = st.parse_duration(f"for {text}", now=now, tz=tz)
            if not parsed.ok:
                return DialogReply((DECLINED_LINE,)) if kind is st.ReplyKind.NO and not more \
                    else None
            return await self._start_timer(parsed.value, ctx, now)
        parsed = st.parse_when(text, now=now, tz=tz)
        if not parsed.ok and len(_norm(text).split()) <= 4:
            parsed = st.parse_when(f"at {text}", now=now, tz=tz)
        if not parsed.ok:
            if kind is st.ReplyKind.NO and not more:
                return DialogReply((DECLINED_LINE,))
            return None
        subject = pending.subject
        if subject is None or not subject.text:
            subject = st.reminder_subject(parsed.remainder)
        return await self._offer(parsed.value, subject, text, ctx, now)

    async def _list_reminders(self, text: str, ctx: CallContext) -> DialogReply:
        store, now, tz = self.store, self.now(), rs.zone(self.zone_name())

        async def read(_args: dict) -> ToolResult:
            items = await store.list_scheduled(rs.KIND_REMINDER)
            return ToolResult(fields={"count": len(items)}, items=tuple(items))

        outcome = await tool_runtime.call(self.app, builtin.REMINDER_LIST, {}, ctx, read)
        if not outcome.executed:
            return DialogReply((outcome.line,))
        items = list(outcome.items)
        if not items:
            return DialogReply((NO_REMINDERS_LINE,))
        lines = [REMINDERS_INTRO.format(count=_count(len(items), "reminder"))]
        lines += [REMINDER_ITEM.format(when=_capital(st.say_moment(i.fire_at, now=now, tz=tz)),
                                       text=i.text) for i in items]
        return DialogReply(tuple(lines))

    async def _cancel_reminder(self, text: str, ctx: CallContext) -> DialogReply:
        refused = await tool_runtime.preflight(self.app, builtin.REMINDER_CANCEL, ctx)
        if refused is not None:
            return DialogReply((refused.line,))
        now = self.now()
        items = await self.store.list_scheduled(rs.KIND_REMINDER)
        if not items:
            return DialogReply((NO_REMINDERS_LINE,))
        item = _pick(items, text)
        if item is None:
            if len(items) > 1 and not _words(text):
                return DialogReply((WHICH_REMINDER_LINE.format(
                    count=_count(len(items), "reminder")),))
            return DialogReply((NO_SUCH_REMINDER_LINE,))
        args = {"reminder_id": item.id, "text": item.text or "a reminder",
                "when": st.say_moment(item.fire_at, now=now, tz=item.tz)}
        outcome = await tool_runtime.propose(self.app, builtin.REMINDER_CANCEL, args, ctx)
        if outcome.status != audit.PROPOSED:
            return DialogReply((outcome.line,))
        self._pending[ctx.session_id] = _CancelAsk(item, args, ctx, outcome.line, now)
        return DialogReply((outcome.line,))

    async def _answer_cancel(self, pending: _CancelAsk, text: str, ctx: CallContext, history,
                             waited, now: datetime) -> DialogReply | None:
        name = builtin.REMINDER_CANCEL
        if not _heard(history, pending.line):
            await tool_runtime.decline(self.app, name, pending.args, pending.ctx,
                                       "readback_unheard")
            return None
        if (waited if waited is not None else now - pending.asked_at) > st.READBACK_WINDOW:
            await tool_runtime.decline(self.app, name, pending.args, pending.ctx, "silence")
            return None
        kind, more = st.classify_reply(text)
        if kind is st.ReplyKind.YES:
            store = self.store
            yes = CallContext(session_id=ctx.session_id, turn_id=ctx.turn_id,
                              confirm_turn_id=ctx.turn_id)

            async def cancel(validated: dict) -> ToolResult:
                row = await store.cancel(validated["reminder_id"], at=now)
                if row is None:
                    raise ToolFailed("not_scheduled")
                return ToolResult(fields={"reminder_id": row.id}, ref=row.id)

            outcome = await tool_runtime.call(self.app, name, pending.args, yes, cancel)
            if outcome.executed:
                self._changed()
            return DialogReply((outcome.line,))
        await tool_runtime.decline(self.app, name, pending.args, pending.ctx,
                                   "no" if kind is st.ReplyKind.NO else "unrelated")
        if kind is st.ReplyKind.NO and not more:
            return DialogReply((CANCEL_KEPT_LINE,))
        return None

    # ── timers ──

    async def _set_timer(self, text: str, ctx: CallContext) -> DialogReply:
        refused = await tool_runtime.preflight(self.app, builtin.TIMER_START, ctx)
        if refused is not None:
            return DialogReply((refused.line,))
        now = self.now()
        parsed = st.parse_duration(text, now=now, tz=rs.zone(self.zone_name()))
        return await self._after_parse("timer", parsed, None, text, ctx, now)

    async def _start_timer(self, value: st.SpokenTime, ctx: CallContext,
                           now: datetime) -> DialogReply:
        duration = value.duration
        args = {"duration_s": int(duration.total_seconds()), "fire_at": _iso_utc(now + duration),
                "length": _capital(st.timer_length(duration)), "time_zone": self.zone_name()}
        store = self.store

        async def start(validated: dict) -> ToolResult:
            row = await store.add(kind=rs.KIND_TIMER, text="",
                                  label=validated["length"].lower(),
                                  fire_at=datetime.fromisoformat(validated["fire_at"]),
                                  time_zone=validated["time_zone"], created_at=now,
                                  session_id=ctx.session_id, turn_id=ctx.turn_id)
            return ToolResult(fields={"timer_id": row.id}, ref=row.id)

        outcome = await tool_runtime.call(self.app, builtin.TIMER_START, args, ctx, start)
        await self._receipted(outcome)
        return DialogReply((outcome.line,))

    async def _timer_status(self, text: str, ctx: CallContext) -> DialogReply:
        store, now = self.store, self.now()

        async def read(_args: dict) -> ToolResult:
            items = await store.list_scheduled(rs.KIND_TIMER)
            return ToolResult(fields={"count": len(items)}, items=tuple(items))

        outcome = await tool_runtime.call(self.app, builtin.TIMER_STATUS, {}, ctx, read)
        if not outcome.executed:
            return DialogReply((outcome.line,))
        timers = list(outcome.items)
        if not timers:
            return DialogReply((NO_TIMERS_LINE,))

        def left(t: rs.Item) -> str:
            length = t.label or "timer"
            if t.fire_at <= now:
                return TIMER_DONE.format(length=length)
            return TIMER_LEFT.format(remaining=_capital(_remaining_words(t.fire_at - now)),
                                     length=length)

        if len(timers) == 1:
            return DialogReply((left(timers[0]),))
        return DialogReply((TIMERS_INTRO.format(count=_count(len(timers), "timer")),
                            *[left(t) for t in timers]))

    async def _cancel_timer(self, text: str, ctx: CallContext) -> DialogReply:
        refused = await tool_runtime.preflight(self.app, builtin.TIMER_CANCEL, ctx)
        if refused is not None:
            return DialogReply((refused.line,))
        now = self.now()
        timers = await self.store.list_scheduled(rs.KIND_TIMER)
        if not timers:
            return DialogReply((NO_TIMERS_LINE,))
        timer = timers[0]
        said = st.parse_duration(text, now=now, tz=rs.zone(self.zone_name()))
        if said.ok:     # "cancel the ten minute timer"
            want = st.timer_length(said.value.duration)
            timer = next((t for t in timers if t.label == want), timer)
        args = {"timer_id": timer.id, "length": _capital(timer.label or "the")}
        store = self.store

        async def cancel(validated: dict) -> ToolResult:
            row = await store.cancel(validated["timer_id"], at=now)
            if row is None:
                raise ToolFailed("not_scheduled")
            return ToolResult(fields={"timer_id": row.id}, ref=row.id)

        outcome = await tool_runtime.call(self.app, builtin.TIMER_CANCEL, args, ctx, cancel)
        if outcome.executed:
            self._changed()
        return DialogReply((outcome.line,))


def _choose(value: st.SpokenTime, text: str, now: datetime) -> st.SpokenTime | None:
    """The reading the answer to "Do you mean A, or B?" picks, or None."""
    words = set(_norm(text).split())
    options = (value, *value.alternatives)
    if words & _FIRST:
        return options[0]
    if words & _SECOND and len(options) > 1:
        return options[1]
    tries = [text] + [f"{_PERIODS[w]}" for w in _PERIODS if w in words]
    for attempt in tries:
        corrected = st.correct(value, attempt, now=now)
        if corrected is not None and corrected.ok and not corrected.value.alternatives:
            return corrected.value
    kind, _ = st.classify_reply(text)
    return options[0] if kind is st.ReplyKind.YES else None


def _pick(items: list[rs.Item], text: str) -> rs.Item | None:
    """The reminder the words name: the one whose words overlap most with them, when exactly one
    does; the only one when the words name none and there is only one."""
    said = _words(text)
    if said:
        scored = [(len(said & _words(i.text)), i) for i in items]
        best = max(s for s, _ in scored)
        if best > 0:
            top = [i for s, i in scored if s == best]
            return top[0] if len(top) == 1 else None
        return None
    return items[0] if len(items) == 1 else None
