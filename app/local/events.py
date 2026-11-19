"""In-process event bus for the /v1/events stream.

Routers and the voice host publish small JSON events (event_schema.STREAM_EVENTS); the SSE
endpoint subscribes and forwards them. One process, no broker, nothing on disk.

Streams. Every tenant has one stream (with isolation off there is one shared stream, tenant
None). A stream numbers its events 1, 2, 3, ... and keeps the most recent ones in a bounded
buffer, so a client that reconnects with the id of the last event it received gets every event
it missed, once and in order. When the buffer no longer holds them, or the id is not one this
stream issued (the engine restarted, or the stream was let go while nobody listened), the client
gets a `stream.reset` event instead of a silent gap. An id is "<stream epoch>-<n>"; clients treat
it as opaque.

Who receives what:
  a tenant's events          the subscribers of that tenant's stream only. An event published
                             with no tenant in scope goes to the shared stream, which with
                             isolation on has no subscribers: nobody's
  a voice session's events   its tenant's stream (they are published in the session's scope),
                             with the session's id and its sequence number
  device events              wake_state and capture_health are facts about a device, and the
                             device works for one tenant at a time: they go to the tenant the
                             device is bound to (VoiceHost publishes them in that scope). With
                             isolation on and nobody bound they go to nobody. They are never
                             broadcast: a wake state carries the meeting a person is in
  machine facts              broadcast=True (a model download's progress): every stream

Bounds (configure(), from config: VOXSCRIBE_EVENTS_REPLAY_BUFFER, VOXSCRIBE_EVENTS_MAX_STREAMS):
the replay buffer per stream, and the number of streams kept. A stream nobody subscribes to is
let go, oldest first, past that number; it is made again on the next subscription with a new
epoch, so ids from before are answered with a reset. Events published to a tenant that has no
stream are not kept: nobody could ever ask for them back. A subscriber's own queue is bounded too
(_MAX_QUEUE) and drops its oldest event instead of blocking a publisher; Subscription.next()
notices the hole and fills it from the buffer, or sends a reset.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any

from . import tenancy
from .event_schema import RESET, SCHEMA_VERSION, STREAM_EVENTS

log = logging.getLogger(__name__)

_MAX_QUEUE = 256                    # per subscriber, between the publisher and the stream writer
DEFAULT_REPLAY_BUFFER = 1024        # per stream (VOXSCRIBE_EVENTS_REPLAY_BUFFER)
DEFAULT_MAX_STREAMS = 256           # streams kept (VOXSCRIBE_EVENTS_MAX_STREAMS)

_replay_buffer = DEFAULT_REPLAY_BUFFER
_max_streams = DEFAULT_MAX_STREAMS
_unknown_kinds: set[str] = set()    # kinds outside the schema already warned about


class Event(dict):
    """One event on one stream.

    As a mapping it is the event as subscribers have always read it, {"type", "data"}. The
    envelope fields are attributes: `id` (unique and ordered within its stream), `n` (its
    position in the stream), and `session_id` and `seq` for an event a voice session published."""

    __slots__ = ("id", "n", "session_id", "seq")

    def __init__(self, event_type: str, data: Any, *, id: str, n: int,
                 session_id: str | None = None, seq: int | None = None) -> None:
        super().__init__(type=event_type, data=data)
        self.id = id
        self.n = n
        self.session_id = session_id
        self.seq = seq

    @property
    def type(self) -> str:
        return self["type"]

    @property
    def data(self) -> Any:
        return self["data"]

    def envelope(self) -> dict:
        """The event as event_schema.Envelope (schema version SCHEMA_VERSION)."""
        out = {"v": SCHEMA_VERSION, "id": self.id, "type": self.type, "data": self.data}
        if self.session_id is not None:
            out["sessionId"] = self.session_id
        if self.seq is not None:
            out["seq"] = self.seq
        return out


class _Stream:
    """One tenant's events: its numbering, its replay buffer and its subscribers."""

    def __init__(self, capacity: int) -> None:
        self.epoch = secrets.token_hex(4)
        self.head = 0                                   # n of the last event published
        self.buffer: deque[Event] = deque(maxlen=capacity)
        self.subscribers: set[Subscription] = set()

    def id_for(self, n: int) -> str:
        return f"{self.epoch}-{n}"

    def append(self, event_type: str, data: Any, session_id: str | None, seq: int | None) -> Event:
        self.head += 1
        event = Event(event_type, data, id=self.id_for(self.head), n=self.head,
                      session_id=session_id, seq=seq)
        self.buffer.append(event)
        for sub in list(self.subscribers):
            sub.offer(event)
        return event

    def after(self, n: int) -> list[Event] | None:
        """Every event after position n, in order, or None when the buffer no longer holds them all."""
        if n >= self.head:
            return []
        oldest = self.buffer[0].n if self.buffer else self.head + 1
        if n + 1 < oldest:
            return None
        return list(self.buffer)[n + 1 - oldest:]

    def position(self, last_event_id: str) -> int | None:
        """The position an id names in this stream, or None when this stream did not issue it."""
        epoch, _, n = last_event_id.strip().rpartition("-")
        if epoch != self.epoch or not n.isdigit() or int(n) > self.head:
            return None
        return int(n)

    def reset(self, reason: str, last_event_id: str, at: int | None = None) -> Event:
        """The reset event, positioned at `at` (the head by default): the client carries on from
        there. It is not kept in the buffer."""
        n = self.head if at is None else at
        return Event(RESET, {"reason": reason, "lastEventId": last_event_id}, id=self.id_for(n), n=n)


# tenant (None: the single shared scope) -> its stream, least recently used first
_streams: OrderedDict[str | None, _Stream] = OrderedDict()


def configure(*, replay_buffer: int | None = None, max_streams: int | None = None) -> None:
    """Set the bounds (at startup, from config). Streams made before keep their buffer size."""
    global _replay_buffer, _max_streams
    if replay_buffer is not None:
        if replay_buffer < 1:
            raise ValueError("the replay buffer holds at least one event")
        _replay_buffer = replay_buffer
    if max_streams is not None:
        if max_streams < 1:
            raise ValueError("at least one stream is kept")
        _max_streams = max_streams


def _stream(tenant: str | None, *, create: bool) -> _Stream | None:
    stream = _streams.get(tenant)
    if stream is None:
        if not create:
            return None
        stream = _streams[tenant] = _Stream(_replay_buffer)
        _let_go_idle()
    _streams.move_to_end(tenant)
    return stream


def _let_go_idle() -> None:
    """Past the bound, forget the least recently used streams that nobody is subscribed to."""
    excess = len(_streams) - _max_streams
    if excess <= 0:
        return
    for tenant in [t for t, s in _streams.items() if not s.subscribers][:excess]:
        del _streams[tenant]


def publish(event_type: str, data: Any, *, broadcast: bool = False,
            session_id: str | None = None, seq: int | None = None) -> None:
    """Publish an event to the stream of the tenant in scope. Call it on the event loop.

    Events carry whole records (a source, an idea, a meeting), so they only ever go to the tenant
    they belong to. `broadcast=True` is for facts about the machine, not a person (a model
    download's progress): every stream gets it. `session_id` and `seq` are a voice session's
    (voice/adapters/events_app.py fills them in)."""
    if event_type not in STREAM_EVENTS and event_type not in _unknown_kinds:
        _unknown_kinds.add(event_type)
        log.warning("events: kind %r is not in event_schema.STREAM_EVENTS", event_type)
    if broadcast:
        for stream in list(_streams.values()):
            stream.append(event_type, data, session_id, seq)
        return
    stream = _stream(tenancy.current_tenant(), create=False)
    if stream is not None:
        stream.append(event_type, data, session_id, seq)


class Subscription(asyncio.Queue):
    """A subscriber's view of its tenant's stream.

    next() is the reader: events in stream order, each once, starting with the replay (or a
    reset) when the subscription resumed from an id. As a queue (get(), qsize()) it is the raw
    feed, which drops its oldest event when full."""

    def __init__(self, stream: _Stream) -> None:
        super().__init__(maxsize=_MAX_QUEUE)
        self.stream = stream
        self.last = stream.head                 # position of the last event handed out by next()
        self.pending: deque[Event] = deque()    # replay to hand out before the queue

    def offer(self, event: Event) -> None:
        if self.full():
            try:
                self.get_nowait()               # drop the oldest; next() fills the hole
            except asyncio.QueueEmpty:
                pass
        self.put_nowait(event)

    def resume(self, last_event_id: str) -> None:
        """Start from the event after `last_event_id`: replay what was missed, or a reset."""
        n = self.stream.position(last_event_id)
        missed = None if n is None else self.stream.after(n)
        if missed is None:
            self.pending.append(self.stream.reset("unknown_id" if n is None else "gap", last_event_id))
        else:
            self.pending.extend(missed)

    async def next(self) -> Event:
        while True:
            if self.pending:
                event = self.pending.popleft()
                self.last = event.n
                return event
            event = await self.get()
            if event.n <= self.last:
                continue                        # already handed out from the replay
            if event.n > self.last + 1:         # the queue dropped some: take them from the buffer
                missed = self.stream.after(self.last)
                if missed is None:
                    log.warning("events: a subscriber fell behind the replay buffer; sending a reset")
                    self.pending.append(self.stream.reset("gap", self.stream.id_for(self.last), at=event.n - 1))
                    self.pending.append(event)
                else:
                    self.pending.extend(e for e in missed if e.n <= event.n)
                continue
            self.last = event.n
            return event


@asynccontextmanager
async def subscribe(last_event_id: str | None = None):
    """Subscribe to the stream of the tenant in scope for the lifetime of the context.

    With `last_event_id` (the client's Last-Event-ID), next() first hands out every event after
    it, or a reset when they are no longer held."""
    stream = _stream(tenancy.current_tenant(), create=True)
    sub = Subscription(stream)
    if last_event_id:
        sub.resume(last_event_id)
    stream.subscribers.add(sub)
    try:
        yield sub
    finally:
        stream.subscribers.discard(sub)


def drop_stream(tenant: str | None) -> bool:
    """Forget a tenant's stream and its replay buffer (erase). A later subscription starts a new stream with a new
    epoch, so an id from before resumes with a reset and no old event. Subscribers still attached to the old stream
    receive nothing more from it. True when there was a stream."""
    stream = _streams.pop(tenant, None)
    if stream is None:
        return False
    stream.buffer.clear()
    return True


def subscriber_count() -> int:
    return sum(len(s.subscribers) for s in _streams.values())
