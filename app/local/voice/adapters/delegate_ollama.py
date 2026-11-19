"""Delegate over the reply chain: a talker that answers at once and a thinker on the delegate lane.

`speculate` starts the reply model on the turn's text before the final recognizer flush, so
token generation overlaps it; the host keeps or cancels it with its drift rule. `reply` routes
the turn and returns its sentences; the reply stage gates and speaks them. After a barge-in the
host passes `system_note` (voice/barge_in.py), which reaches the reply model's system prompt.

Routing, in order, all before the reply model is called (T14/T16):

  reminder reminders and timers (voice/reminder_dialog.py, PL45): "remind me ..." is read back
           and set only on the next turn's yes; "set a timer for ..." starts at once; lists,
           the time left and cancelling. A turn that answers a read-back is the dialog's too.
           Every step goes through the tool runtime and leaves its audit row
  rerun    a yes to last turn's stale-result offer submits the same request again
  cancel   "cancel that", "never mind": the newest open task is cancelled, "Okay, cancelled."
  change   "actually, X", "search for X instead": the newest open task is cancelled and X
           is submitted in its place
  results  "what did you find?": the waiting results, or that the task is still running
  clock    the pre-router's answer from the OS clock, one sentence, no model and no tool
  thinker  a delegation signal or a pre-router search route: the request is submitted to the
           lane and the talker says it is looking it up. When the network decision refuses
           web search (senior care, classroom, the profile, privacy offline: PL41) nothing is
           submitted, one `refused` tool row is written and the scripted refusal is spoken
  talker  the reply model, which has no tools; if it emits [DELEGATE: <request>] the request
           goes to the lane and the marker is never spoken

The session's expressiveness (voice/expression.py, PL66) chooses the talker's prompt. In
natural the talker's own sentences, and the thinker's summary, go through the register's
deterministic rules before the gate; scripted lines (refusals, read-backs, confirmations, the
lane's phrases) never do. In natural the reply has no immediate thinking cue: the host says one
only when the reply is late (PL67).

The lane is polled at the start of every turn: a finished result is spoken after the turn's own
reply, through the same gate; a stale one is offered instead of read out. Nothing here runs on
another thread, and the thinker is an asyncio task on the session's event loop.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import timedelta

from voxscribe.session.ports import ReplyStream

from app.local import reply_chain
from app.local.tools import builtin
from app.local.tools import runtime as tool_runtime
from app.local.tools.runtime import CallContext
from app.local.voice import delegate_lane as dl
from app.local.voice import expression
from app.local.voice import reminder_store as rs
from app.local.voice.adapters.events_app import publish as publish_event
from app.local.voice.adapters.store_postgres import PostgresDelegateTaskStore
from app.local.voice.delegate_lane import DelegateLane, DelegateTask, IntentKind
from app.local.voice.host import current_session
from app.local.voice.reminder_dialog import ReminderDialog

log = logging.getLogger("app.local.voice.delegate")

TASK_EVENT = "wake.delegate"     # status only: a result reaches a person through the gate

# Hands a request to the thinker. Returns None when it was submitted, or the scripted refusal
# line when the network decision (PL41) refuses the search it needs: then nothing is submitted.
Submit = Callable[[str], Awaitable[str | None]]


def _persona():
    """The session's assistant settings (name, persona label, system prompt), read once when it
    opened; None outside a session. The thinker's task inherits the session it was started in."""
    session = current_session()
    return session.settings if session is not None else None


def _expressiveness() -> str:
    """The session's register; plain outside a session."""
    session = current_session()
    return getattr(session, "expressiveness", expression.PLAIN) if session is not None \
        else expression.PLAIN


def _natural() -> bool:
    return _expressiveness() == expression.NATURAL


def _cue(text: str) -> str | None:
    """The thinking cue said at once before a reply: plain only (in natural the host says one
    only when the reply is late)."""
    return reply_chain.THINKING_PHRASE if text and not _natural() else None


async def _shaped(sentences: AsyncIterator[str], *, reaction_allowed: bool = True
                  ) -> AsyncIterator[str]:
    """The reply model's sentences under the natural register's rules, and the opener kept for
    the session's next reply. Names of what was removed are logged, never words."""
    session = current_session()
    shape = expression.ReplyShape()
    previous = getattr(session, "last_opener", None) if session is not None else None
    async for sentence in expression.shape_reply(sentences, previous_opener=previous,
                                                 shape=shape,
                                                 reaction_allowed=reaction_allowed):
        yield sentence
    if reaction_allowed and session is not None:
        session.last_opener = shape.opener
    if shape.reaction or shape.removed:
        log.info("voice delegate: register reaction=%s removed=%s", shape.reaction or "none",
                 ",".join(shape.removed or ()) or "none")


def _waited() -> timedelta | None:
    """The silence the host measured between the end of the last reply and the start of this
    turn's speech; None when it does not know."""
    session = current_session()
    ended = getattr(session, "reply_ended_mono", None)
    onset = getattr(session, "speech_onset_mono", None)
    if ended is None or onset is None:
        return None
    return timedelta(seconds=max(0.0, onset - ended))


class OllamaDelegate:
    def __init__(self, app, *, lane: DelegateLane | None = None,
                 reminder_store=None, clock=None, reminder_gate=None) -> None:
        self._app = app
        self._task_store = PostgresDelegateTaskStore(app)
        self.lane = lane or DelegateLane(runner=self._think, on_change=self._task_changed)
        self._contexts: dict[str, list[dict]] = {}      # task id -> history when submitted
        self._offered: DelegateTask | None = None       # stale result offered last turn
        self._writes: set[asyncio.Task] = set()
        self._reminder_store = (reminder_store if reminder_store is not None
                                else rs.ReminderStore.for_voice(app))
        self.reminders = ReminderDialog(app, self._reminder_store, clock=clock,
                                        gate=reminder_gate)

    @property
    def _settings(self):
        return getattr(self._app.state, "settings", None)

    @property
    def model(self) -> str:
        return reply_chain.reply_settings(self._settings)[0]

    async def warm(self) -> None:
        """One-token request so Ollama loads the reply model and caches the prompt prefix."""
        from app.local.llm_adapter import stream_reply

        try:
            log.info("voice delegate: pre-warming reply model=%s", self.model)
            async for _ in stream_reply("hi", model=self.model):
                break
            log.info("voice delegate: reply model warmed model=%s", self.model)
        except Exception:
            log.debug("voice delegate: prewarm failed (non-critical) model=%s", self.model)

    def speculate(self, partial_text: str, history: Sequence[dict] | None) -> _Speculation:
        from app.local.llm_adapter import stream_reply

        return _Speculation(stream_reply(
            partial_text, model=self.model, history=list(history) if history else None,
            system_prompt=reply_chain.talker_system_prompt(self._settings, _persona(),
                                                           _expressiveness())))

    async def reply(self, turn_text: str, session_id: str, *, turn_id: str | None = None,
                    history: Sequence[dict] | None = None, speculation=None,
                    system_note: str | None = None,
                    system_prefix: str | None = None) -> ReplyStream:
        text = turn_text.strip()
        hist = list(history) if history else None

        # Reminders and timers: deterministic, no model, every step through the tool runtime
        # (PL45). The read-back is spoken on one turn and acted on at the next.
        handled = await self.reminders.handle(text, session_id=session_id, turn_id=turn_id,
                                              history=hist, waited=_waited())
        prefix: list[str] = list(handled.sentences) if handled is not None else []
        if handled is not None:
            log.info("voice delegate: route=%s answer_normally=%s session_id=%s",
                     handled.route, handled.answer_normally, session_id)
            if not handled.answer_normally:
                cancel = getattr(speculation, "cancel", None)
                if cancel is not None:
                    cancel()
                return ReplyStream(
                    sentences=_each(prefix),
                    source="reminder",
                    cue=_cue(text),
                    fallback=reply_chain.FALLBACK_PHRASE,
                )

        async def submit(request: str) -> str | None:
            # The thinker's one tool is web search: it is refused here, before the lane, so the
            # person hears why at once and no task is started (PL41).
            refused = await tool_runtime.preflight(
                self._app, builtin.WEB_SEARCH, CallContext(session_id=session_id,
                                                           turn_id=turn_id))
            if refused is not None:
                log.info("voice delegate: search refused session_id=%s", session_id)
                return refused.line
            task = self.lane.submit(request, session_id=session_id, turn_id=turn_id)
            if hist and task.status in dl.ACTIVE_STATUSES:
                self._contexts[task.task_id] = hist
            return None

        offered, self._offered = self._offered, None
        fixed, source, deliver = await self._route(text, offered, submit)
        if fixed is None:
            src = reply_chain.talker_sentences(self._app, session_id, text, history=hist,
                                               pre_stream=speculation, system_note=system_note,
                                               system_prefix=system_prefix, persona=_persona(),
                                               expressiveness=_expressiveness())
            talker = _shaped(src.sentences) if _natural() else src.sentences
            own, source = _without_marker(talker, submit), src.source
        else:
            cancel = getattr(speculation, "cancel", None)
            if cancel is not None:
                cancel()
            own = _each(fixed)
        log.info("voice delegate: route=%s session_id=%s", source, session_id)
        return ReplyStream(sentences=self._with_delivery(_prepend_then(prefix, own),
                                                         deliver),
                           source=source,
                           cue=_cue(text),
                           fallback=reply_chain.FALLBACK_PHRASE)

    # ------------------------------------------------------------------ routing

    async def _route(self, text: str, offered: DelegateTask | None,
                     submit: Submit) -> tuple[list[str] | None, str, bool]:
        """(scripted sentences or None for the talker, source, deliver waiting results)."""
        lane = self.lane
        if offered is not None and dl.is_affirmative(text):
            refusal = await submit(offered.request)
            return [refusal or dl.RERUN_PHRASE], "delegate", True

        intent = dl.detect_intent(text)
        kind = intent.kind if intent is not None else None
        route = reply_chain.route_turn(self._settings, text)
        is_clock = route is not None and route[0] == "clock"
        open_task = lane.latest_open()

        if kind is IntentKind.CANCEL and open_task is not None:
            lane.cancel(open_task.task_id)
            return [dl.CANCELLED_PHRASE], "cancel", False
        if kind is IntentKind.CHANGE and open_task is not None and not is_clock:
            lane.cancel(open_task.task_id)
            refusal = await submit(intent.request)
            return [refusal or dl.CHANGED_PHRASE], "delegate", True
        if kind is IntentKind.RESULTS:
            if lane.get_pending():
                return [], "delivery", True
            working = lane.active()
            if working:
                topic = dl.request_topic(working[-1].request)
                return [dl.STILL_WORKING_PHRASE.format(topic=topic)], "delegate", False
            return [dl.NOTHING_WAITING_PHRASE], "delegate", False

        if is_clock:
            return [route[1]], "clock", True
        if route is not None or dl.delegation_signal(text):
            explicit = kind is IntentKind.CHANGE and intent.explicit
            refusal = await submit(intent.request if explicit else text)
            return [refusal or dl.DELEGATING_PHRASE], "delegate", True
        return None, "llm", True

    async def _with_delivery(self, own: AsyncIterator[str], deliver: bool) -> AsyncIterator[str]:
        """The turn's own sentences, then what the lane has waiting.

        A task is marked delivered as its first sentence is pulled, so a reply the gate stops
        early leaves later results waiting for the next turn. At most one stale result is
        offered per turn, last, because the offer ends in a question."""
        async for sentence in own:
            yield sentence
        if not deliver:
            return
        lane = self.lane
        stale: DelegateTask | None = None
        for task in lane.get_pending():
            if lane.is_stale(task):
                stale = stale or task
                continue
            lane.mark_delivered(task.task_id)
            if task.error is None:
                yield dl.DELIVERY_LEAD.format(topic=dl.request_topic(task.request))
            found = _each(reply_chain.split_sentences(task.result or ""))
            if task.error is None and _natural():
                # The thinker's summary is the reply model's words: the register's rules, with
                # no reaction, since it follows the scripted lead.
                found = _shaped(found, reaction_allowed=False)
            async for sentence in found:
                yield sentence
        if stale is not None:
            lane.mark_delivered(stale.task_id)
            self._offered = stale
            yield dl.STALE_OFFER.format(topic=dl.request_topic(stale.request))

    # ------------------------------------------------------------------ thinker

    async def _think(self, task: DelegateTask) -> str:
        """The thinker: the only place a tool is called. Today its one tool is web search."""
        history = self._contexts.pop(task.task_id, None)
        return await reply_chain.search_answer(
            self._app, task.session_id, task.request, dl.request_topic(task.request),
            task_id=task.task_id, turn_id=task.turn_id, history=history, persona=_persona(),
            expressiveness=_expressiveness())

    def _task_changed(self, task: DelegateTask) -> None:
        """Lane hook: keep the task's row current and tell the app its status."""
        if task.status not in dl.ACTIVE_STATUSES:
            self._contexts.pop(task.task_id, None)
        publish_event(TASK_EVENT, {"taskId": task.task_id, "meetingId": task.session_id,
                                       "status": task.status.value})
        try:
            write = asyncio.get_running_loop().create_task(self._task_store.save(task),
                                                           name="delegate-task-row")
        except RuntimeError:
            log.warning("voice delegate: no running loop, task row not written task=%s",
                        task.task_id)
            return
        self._writes.add(write)
        write.add_done_callback(self._writes.discard)

    def forget(self) -> None:
        """The device now listens for someone else: nothing held here for the person before may
        reach them. Open tasks are cancelled, waiting results and the stale offer are dropped.
        The host calls this inside the previous person's scope, so the rows it updates are theirs."""
        lane = self.lane
        for task in [*lane.active(), *lane.get_pending()]:
            lane.cancel(task.task_id)
        self._contexts.clear()
        self._offered = None
        self.reminders.forget()

    async def drain(self) -> None:
        """Wait for task rows still being written (shutdown and tests)."""
        while self._writes:
            # Gathering writes that have all finished returns without yielding to the loop, so
            # their done callbacks never run: drop what was awaited here, or this never ends.
            pending = list(self._writes)
            await asyncio.gather(*pending, return_exceptions=True)
            self._writes.difference_update(pending)

    def awaiting_answer(self, session_id: str) -> bool:
        """True while a read-back or a question of the reminder dialog waits for this session's
        next turn, so the host lets a one-word "yes" or "no" through its speech guard."""
        return self.reminders.has_pending(session_id)

    async def session_closed(self, session_id: str) -> None:
        """The host closed the session: a read-back still waiting for its yes is declined."""
        await self.reminders.session_closed(session_id)

async def _prepend_then(prefix: list[str],
                        gen: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the prefix sentences first, then everything from gen."""
    for sentence in prefix:
        yield sentence
    async for sentence in gen:
        yield sentence


async def _each(sentences: Sequence[str]) -> AsyncIterator[str]:
    for sentence in sentences:
        yield sentence


async def _without_marker(sentences: AsyncIterator[str], submit: Submit) -> AsyncIterator[str]:
    """The talker's sentences, unless it asks to delegate: then the request goes to the lane,
    the talker says it is looking it up, and nothing else it produced is spoken."""
    held = ""
    async for sentence in sentences:
        text = f"{held} {sentence}" if held else sentence
        held = ""
        request = dl.find_delegate_marker(text)
        if request is None and dl.has_open_marker(text):
            held = text        # the marker was split at punctuation inside the request
            continue
        if request is None:
            yield text
            continue
        if request:
            yield await submit(request) or dl.DELEGATING_PHRASE
        return
    # A marker the model never closed: what follows it is the request.
    if held:
        start = held.lower().find(dl.DELEGATE_MARKER_OPEN.lower()) + len(dl.DELEGATE_MARKER_OPEN)
        request = held[start:].strip()
        if request:
            yield await submit(request) or dl.DELEGATING_PHRASE


class _Speculation:
    """Reads a token stream into a buffer in the background; iterate to replay, or cancel."""

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._q: asyncio.Queue[str | None] = asyncio.Queue()
        self._task = asyncio.create_task(self._drain(source), name="reply-speculation")

    async def _drain(self, source: AsyncIterator[str]) -> None:
        with contextlib.suppress(Exception):
            async for token in source:
                await self._q.put(token)
        await self._q.put(None)

    def cancel(self) -> None:
        self._task.cancel()
        while not self._q.empty():
            self._q.get_nowait()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        token = await self._q.get()
        if token is None:
            raise StopAsyncIteration
        return token
