"""Delegate lane: the background task registry behind the talker and thinker split (T14).

The talker (the reply model) answers short questions at once and has no tools. Anything that
needs a tool is submitted here as a task; a thinker runs it as an asyncio task on the session's
event loop while the conversation carries on. A finished result waits in the lane until the
delegate adapter picks it up at the start of a later turn.

    QUEUED -> RUNNING -> DONE -> DELIVERED
    any of QUEUED, RUNNING, DONE -> CANCELLED

A thinker that fails or runs past THINKER_TIMEOUT_S is marked DONE with THINKER_FAILED_PHRASE
as its result and `error` set, so the person hears that it did not work. A result that has been
DONE for longer than STALE_RESULT_S is stale: it is offered, never read out.

Also here, because they are pure text rules with no I/O: the spoken intents handled before the
reply model (cancel, change, asking for results), the delegation signals, and the scripted
phrases. No transcript text or result text is logged.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import Enum

log = logging.getLogger("app.local.voice.lane")

STALE_RESULT_S = 300             # seconds before a result is considered stale
THINKER_TIMEOUT_S = 30           # max seconds for a background task before marking failed
DELEGATION_SIGNALS = frozenset({"search", "find", "look up", "latest", "current", "news"})
FINISHED_TASKS_KEPT = 32         # delivered and cancelled tasks held in memory; rows keep all

# Spoken intents, matched on the normalised turn text before the reply model is called.
CANCEL_PHRASES = ("cancel that", "never mind", "nevermind", "forget it", "stop searching")
CHANGE_PREFIX = "actually"
CHANGE_SUFFIX = "instead"
CHANGE_VERBS = ("search", "look up", "find")
RESULTS_PHRASES = ("what did you find", "what have you found", "what did you get",
                   "any results", "still working on that", "are you still searching")
AFFIRM_WORDS = frozenset({"yes", "yeah", "yep", "sure", "okay", "ok", "please"})
AFFIRM_PHRASES = ("run it again", "go ahead", "do it")
AFFIRM_MAX_WORDS = 5             # "yes please" is a yes; "okay, what time is it now" is not

# Scripted phrases. The delegate adapter yields them as sentences, so each one passes the gate.
DELEGATING_PHRASE = "I'm looking that up, it'll take a moment."
CANCELLED_PHRASE = "Okay, cancelled."
CHANGED_PHRASE = "Okay, I'll look that up instead."
RERUN_PHRASE = "Okay, I'll run it again."
DELIVERY_LEAD = "Here's what I found about {topic}."
STALE_OFFER = "I have an older result for {topic}. Want me to run it again?"
STILL_WORKING_PHRASE = "I'm still working on {topic}."
NOTHING_WAITING_PHRASE = "I don't have any results waiting."
THINKER_FAILED_PHRASE = "I couldn't finish looking that up."

# What the talker emits when it decides the turn needs a tool it does not have.
DELEGATE_MARKER_OPEN = "[DELEGATE:"
TALKER_DELEGATE_INSTRUCTION = (
    "A helper can search the web for you. When a question is about recent events, live "
    "scores, prices, weather, or who holds an office today, output only "
    f"{DELEGATE_MARKER_OPEN} <the question>] and the helper will handle it. For everything "
    "else just answer. Never mention the helper."
)

_PUNCT_RE = re.compile(r"[^\w\s']")
_SPACE_RE = re.compile(r"\s+")
_SIGNAL_RE = re.compile(
    "|".join(rf"\b{re.escape(s)}\b" for s in sorted(DELEGATION_SIGNALS, key=len, reverse=True)))
_CHANGE_PREFIX_RE = re.compile(rf"^\s*{CHANGE_PREFIX}\b[\s,.:;!-]*", re.IGNORECASE)
_CHANGE_SUFFIX_RE = re.compile(rf"[\s,]*\b{CHANGE_SUFFIX}\b[\s.!?]*$", re.IGNORECASE)
_CHANGE_VERB_RE = re.compile("|".join(rf"\b{re.escape(v)}\b" for v in CHANGE_VERBS))
_MARKER_RE = re.compile(rf"{re.escape(DELEGATE_MARKER_OPEN)}\s*(.*?)\s*\]",
                        re.IGNORECASE | re.DOTALL)
# "search the web for X", "please look up X", "find X": what is left is the topic.
_TOPIC_LEAD_RE = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you|hey|no|okay|ok)[\s,]+)*"
    r"(?:search|look\s+up|find)(?:\s+(?:the\s+web|the\s+internet|online))?"
    r"(?:\s+(?:for|about|up))?\s+",
    re.IGNORECASE,
)


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    DELIVERED = "delivered"


OPEN_STATUSES = frozenset({TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.DONE})
FINISHED_STATUSES = frozenset({TaskStatus.CANCELLED, TaskStatus.DELIVERED})
ACTIVE_STATUSES = frozenset({TaskStatus.QUEUED, TaskStatus.RUNNING})


@dataclass
class DelegateTask:
    task_id: str                          # uuid; tool_calls.task_id carries the same value
    request: str                          # the original user request
    status: TaskStatus = TaskStatus.QUEUED
    result: str | None = None             # set when DONE
    created_at: float = 0.0               # lane clock (monotonic seconds)
    delivered_at: float | None = None
    done_at: float | None = None          # lane clock when the result arrived
    session_id: str | None = None         # the session whose turn submitted it
    turn_id: str | None = None            # that turn; tool rows are written against it
    error: str | None = None              # why a DONE task carries the failure phrase


Runner = Callable[[DelegateTask], Awaitable[str]]


class DelegateLane:
    """Background task registry. Thread-safe.

    State lives behind a lock, so any thread may read or change it. The thinker itself is an
    asyncio task: `submit` starts it when called on a running event loop and a `runner` is
    set; without either the task stays QUEUED until `start` and `complete` are called.
    `on_change` is called with a copy of the task after every state change, outside the lock.
    The tasks handed out are the registry's own: read them, change them only through the lane.
    """

    def __init__(self, runner: Runner | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 on_change: Callable[[DelegateTask], None] | None = None,
                 timeout_s: float = THINKER_TIMEOUT_S) -> None:
        self._runner = runner
        self._clock = clock
        self._on_change = on_change
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._tasks: dict[str, DelegateTask] = {}       # insertion order is submit order
        self._jobs: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ registry

    def submit(self, request: str, *, session_id: str | None = None,
               turn_id: str | None = None) -> DelegateTask:
        task = DelegateTask(task_id=str(uuid.uuid4()), request=request.strip(),
                            created_at=self._clock(), session_id=session_id, turn_id=turn_id)
        with self._lock:
            self._tasks[task.task_id] = task
            finished = [t.task_id for t in self._tasks.values()
                        if t.status in FINISHED_STATUSES]
            for old in finished[:max(0, len(finished) - FINISHED_TASKS_KEPT)]:
                del self._tasks[old]
        log.info("lane: submitted task=%s words=%d", task.task_id, len(task.request.split()))
        self._changed(task)
        if self._runner is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                log.warning("lane: no running loop, task=%s stays queued", task.task_id)
            else:
                job = loop.create_task(self._run(task.task_id), name="delegate-thinker")
                with self._lock:
                    self._jobs[task.task_id] = job
                job.add_done_callback(lambda _j, tid=task.task_id: self._forget_job(tid))
        return task

    def start(self, task_id: str) -> bool:
        """QUEUED to RUNNING. False when the task is unknown or no longer queued."""
        return self._move(task_id, {TaskStatus.QUEUED}, TaskStatus.RUNNING)

    def complete(self, task_id: str, result: str, *, error: str | None = None) -> bool:
        """The result arrived. False when the task was cancelled or delivered meanwhile."""
        return self._move(task_id, ACTIVE_STATUSES, TaskStatus.DONE, result=result,
                          error=error)

    def cancel(self, task_id: str) -> bool:
        """Stop a task, or drop a finished result nobody has heard. False when there is
        nothing left to cancel."""
        if not self._move(task_id, OPEN_STATUSES, TaskStatus.CANCELLED):
            return False
        with self._lock:
            job = self._jobs.get(task_id)
        if job is not None and not job.done():
            job.get_loop().call_soon_threadsafe(job.cancel)
        return True

    def mark_delivered(self, task_id: str) -> None:
        self._move(task_id, {TaskStatus.DONE}, TaskStatus.DELIVERED)

    def get(self, task_id: str) -> DelegateTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_pending(self) -> list[DelegateTask]:
        """Finished results waiting to be delivered, oldest first."""
        with self._lock:
            return [t for t in self._tasks.values() if t.status is TaskStatus.DONE]

    def active(self) -> list[DelegateTask]:
        """Tasks still queued or running, oldest first."""
        with self._lock:
            return [t for t in self._tasks.values() if t.status in ACTIVE_STATUSES]

    def latest_open(self) -> DelegateTask | None:
        """The newest task that is queued, running, or done and not yet delivered."""
        with self._lock:
            for task in reversed(self._tasks.values()):
                if task.status in OPEN_STATUSES:
                    return task
        return None

    def is_stale(self, task: DelegateTask) -> bool:
        """True for a result that has waited longer than STALE_RESULT_S to be delivered."""
        return (task.status is TaskStatus.DONE and task.done_at is not None
                and self._clock() - task.done_at > STALE_RESULT_S)

    async def wait(self, task_id: str) -> None:
        """Wait for a task's thinker to finish (tests and shutdown)."""
        with self._lock:
            job = self._jobs.get(task_id)
        if job is not None:
            await asyncio.wait({job})

    # ------------------------------------------------------------------ internals

    def _move(self, task_id: str, allowed, to: TaskStatus, *, result: str | None = None,
              error: str | None = None) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status not in allowed:
                return False
            task.status = to
            now = self._clock()
            if to is TaskStatus.DONE:
                task.result, task.error, task.done_at = result, error, now
            elif to is TaskStatus.DELIVERED:
                task.delivered_at = now
        log.info("lane: task=%s status=%s", task_id, to.value)
        self._changed(task)
        return True

    def _changed(self, task: DelegateTask) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(replace(task))
        except Exception:
            log.exception("lane: change hook failed task=%s", task.task_id)

    def _forget_job(self, task_id: str) -> None:
        with self._lock:
            self._jobs.pop(task_id, None)

    async def _run(self, task_id: str) -> None:
        task = self.get(task_id)
        if task is None or not self.start(task_id):
            return
        try:
            result = await asyncio.wait_for(self._runner(task), self._timeout_s)
        except asyncio.TimeoutError:
            log.warning("lane: task=%s timed out after %s s", task_id, self._timeout_s)
            self.complete(task_id, THINKER_FAILED_PHRASE, error="timeout")
        except Exception as exc:
            log.warning("lane: task=%s failed: %s", task_id, type(exc).__name__)
            self.complete(task_id, THINKER_FAILED_PHRASE, error=type(exc).__name__)
        else:
            text = (result or "").strip()
            self.complete(task_id, text or THINKER_FAILED_PHRASE,
                          error=None if text else "empty")


# ---------------------------------------------------------------------- spoken intents


class IntentKind(Enum):
    CANCEL = "cancel"
    CHANGE = "change"
    RESULTS = "results"


@dataclass(frozen=True)
class SpokenIntent:
    kind: IntentKind
    request: str = ""        # CHANGE: the revised request, as spoken
    explicit: bool = False   # CHANGE: "search for X instead", a request on its own


def _norm(text: str) -> str:
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def detect_intent(text: str) -> SpokenIntent | None:
    """Cancel, change or a request for results, by keyword. None for anything else.

    "actually, never mind" is a cancel. "actually, <request>" and "<search verb> ... instead"
    are changes; the caller decides whether there is a task to change.
    """
    body, prefixed = _CHANGE_PREFIX_RE.subn("", text.strip(), count=1)
    norm = _norm(body)
    if not norm:
        return None
    if any(p in norm for p in CANCEL_PHRASES):
        return SpokenIntent(IntentKind.CANCEL)
    if norm.split()[-1] == CHANGE_SUFFIX and _CHANGE_VERB_RE.search(norm):
        request = _CHANGE_SUFFIX_RE.sub("", body).strip()
        return SpokenIntent(IntentKind.CHANGE, request, explicit=True)
    if any(p in norm for p in RESULTS_PHRASES):
        return SpokenIntent(IntentKind.RESULTS)
    if prefixed:
        return SpokenIntent(IntentKind.CHANGE, body.strip())
    return None


def is_affirmative(text: str) -> bool:
    """A yes to "want me to run it again?": a short turn opening with a yes word, or a turn
    that says to run it."""
    norm = _norm(text)
    words = norm.split()
    if not words:
        return False
    short_yes = words[0] in AFFIRM_WORDS and len(words) <= AFFIRM_MAX_WORDS
    return short_yes or any(p in norm for p in AFFIRM_PHRASES)


def delegation_signal(text: str) -> str | None:
    """The first delegation signal in the turn text, or None."""
    m = _SIGNAL_RE.search(_norm(text))
    return m.group(0) if m else None


def request_topic(request: str) -> str:
    """The request without its leading search verb: what to search for, and what to call the
    task when speaking about it."""
    topic = _TOPIC_LEAD_RE.sub("", request.strip(), count=1).strip().rstrip(".?!").strip()
    return topic or request.strip().rstrip(".?!")


def find_delegate_marker(text: str) -> str | None:
    """The request inside a talker's [DELEGATE: <request>] marker, or None."""
    m = _MARKER_RE.search(text)
    return m.group(1) if m else None


def has_open_marker(text: str) -> bool:
    return DELEGATE_MARKER_OPEN.lower() in text.lower()
