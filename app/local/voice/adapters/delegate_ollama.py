"""Delegate over the reply chain: a talker that answers at once and a thinker on the delegate lane.

`speculate` starts the reply model on the turn's text before the final recognizer flush, so
token generation overlaps it; the host keeps or cancels it with its drift rule. `reply` routes
the turn and returns its sentences; the reply stage gates and speaks them. After a barge-in the
host passes `system_note` (voice/barge_in.py), which reaches the reply model's system prompt.

Routing, in order, all before the reply model is called (T14):

  rerun    a yes to last turn's stale-result offer submits the same request again
  cancel   "cancel that", "never mind": the newest open task is cancelled, "Okay, cancelled."
  change   "actually, X", "search for X instead": the newest open task is cancelled and X
           is submitted in its place
  results  "what did you find?": the waiting results, or that the task is still running
  clock    the pre-router's answer from the OS clock, one sentence, no model and no tool
  thinker  a delegation signal or a pre-router search route: the request is submitted to the
           lane and the talker says it is looking it up
  talker   the reply model, which has no tools; if it emits [DELEGATE: <request>] the request
           goes to the lane and the marker is never spoken

The lane is polled at the start of every turn: a finished result is spoken after the turn's own
reply, through the same gate; a stale one is offered instead of read out. Nothing here runs on
another thread, and the thinker is an asyncio task on the session's event loop.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Sequence

from voxscribe.session.ports import ReplyStream

from app.local import reply_chain
from app.local.voice import delegate_lane as dl
from app.local.voice.adapters.events_app import publish_broadcast
from app.local.voice.adapters.store_postgres import PostgresDelegateTaskStore
from app.local.voice.delegate_lane import DelegateLane, DelegateTask, IntentKind

log = logging.getLogger("app.local.voice.delegate")

TASK_EVENT = "wake.delegate"     # status only: a result reaches a person through the gate

Submit = Callable[[str], DelegateTask]


class OllamaDelegate:
    def __init__(self, app, *, lane: DelegateLane | None = None) -> None:
        self._app = app
        self._task_store = PostgresDelegateTaskStore(app)
        self.lane = lane or DelegateLane(runner=self._think, on_change=self._task_changed)
        self._contexts: dict[str, list[dict]] = {}      # task id -> history when submitted
        self._offered: DelegateTask | None = None       # stale result offered last turn
        self._writes: set[asyncio.Task] = set()

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
            system_prompt=reply_chain.talker_system_prompt(self._settings)))

    async def reply(self, turn_text: str, session_id: str, *, turn_id: str | None = None,
                    history: Sequence[dict] | None = None, speculation=None,
                    system_note: str | None = None) -> ReplyStream:
        text = turn_text.strip()
        hist = list(history) if history else None

        def submit(request: str) -> DelegateTask:
            task = self.lane.submit(request, session_id=session_id, turn_id=turn_id)
            if hist and task.status in dl.ACTIVE_STATUSES:
                self._contexts[task.task_id] = hist
            return task

        offered, self._offered = self._offered, None
        fixed, source, deliver = self._route(text, offered, submit)
        if fixed is None:
            src = reply_chain.talker_sentences(self._app, session_id, text, history=hist,
                                               pre_stream=speculation, system_note=system_note)
            own, source = _without_marker(src.sentences, submit), src.source
        else:
            cancel = getattr(speculation, "cancel", None)
            if cancel is not None:
                cancel()
            own = _each(fixed)
        log.info("voice delegate: route=%s session_id=%s", source, session_id)
        return ReplyStream(sentences=self._with_delivery(own, deliver), source=source,
                           cue=reply_chain.THINKING_PHRASE if text else None,
                           fallback=reply_chain.FALLBACK_PHRASE)

    # ------------------------------------------------------------------ routing

    def _route(self, text: str, offered: DelegateTask | None,
               submit: Submit) -> tuple[list[str] | None, str, bool]:
        """(scripted sentences or None for the talker, source, deliver waiting results)."""
        lane = self.lane
        if offered is not None and dl.is_affirmative(text):
            submit(offered.request)
            return [dl.RERUN_PHRASE], "delegate", True

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
            submit(intent.request)
            return [dl.CHANGED_PHRASE], "delegate", True
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
            submit(intent.request if explicit else text)
            return [dl.DELEGATING_PHRASE], "delegate", True
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
            for sentence in reply_chain.split_sentences(task.result or ""):
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
            task_id=task.task_id, turn_id=task.turn_id, history=history)

    def _task_changed(self, task: DelegateTask) -> None:
        """Lane hook: keep the task's row current and tell the app its status."""
        if task.status not in dl.ACTIVE_STATUSES:
            self._contexts.pop(task.task_id, None)
        publish_broadcast(TASK_EVENT, {"taskId": task.task_id, "meetingId": task.session_id,
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

    async def drain(self) -> None:
        """Wait for task rows still being written (shutdown and tests)."""
        while self._writes:
            await asyncio.gather(*list(self._writes), return_exceptions=True)


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
            submit(request)
            yield dl.DELEGATING_PHRASE
        return
    # A marker the model never closed: what follows it is the request.
    if held:
        start = held.lower().find(dl.DELEGATE_MARKER_OPEN.lower()) + len(dl.DELEGATE_MARKER_OPEN)
        request = held[start:].strip()
        if request:
            submit(request)
            yield dl.DELEGATING_PHRASE


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
