"""The engine's event schema: every event kind the engine emits, typed, under one version.

This module is the one description of engine events, for the engine's own tests and for the
client generator (json_schema() is its input). Two families:

  STREAM_EVENTS    what GET /v1/events carries. Each event goes out in an envelope (Envelope):
                   the schema version, an identifier unique within the subscriber's stream, the
                   kind, and, for an event a voice session published, the session id and its
                   per-session sequence number. `data` is the payload model for the kind.
  MEETING_EVENTS   what a meeting's bus carries to its watchers (the live meeting socket). The
                   socket shapes these into its own frames (routers/live.py _to_frame); they have
                   no envelope yet.

Wire forms of the event stream:
  GET /v1/events             the form clients already read: `id:` is the envelope id, `event:` the
                             kind, `data:` the payload exactly as before (no field added or removed)
  GET /v1/events?schema=1    `data:` is the whole envelope (this module's Envelope, version 1)

A change that removes or renames a field, or changes a field's meaning, is a new SCHEMA_VERSION.
Adding an optional field or a new kind is not.

Payload models allow fields they do not name, so a payload that grows a field still validates;
the fields they name are the ones a client may rely on.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import models_json_schema

from .models import EnhancedBlock, Meeting, Model, Segment, Source

SCHEMA_VERSION = 1

RESET = "stream.reset"      # the kind a subscriber gets when its missed events are no longer held


class _Payload(BaseModel):
    model_config = ConfigDict(extra="allow")


# ── the envelope ──────────────────────────────────────────────────────────────


class Envelope(BaseModel):
    """One event on the event stream.

    id         opaque. Unique and ordered within the subscriber's stream (one stream per tenant,
               or one shared stream with isolation off). A client sends the last one it received
               back as Last-Event-ID when it reconnects; it must not parse it
    sessionId  the voice session that published the event, when one did
    seq        that session's sequence number: 1 for its first event, one more for each after,
               with no gap. Absent on events no session published"""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = SCHEMA_VERSION
    id: str
    type: str
    sessionId: str | None = None
    seq: int | None = Field(default=None, ge=1)
    data: Any


# ── event stream payloads ─────────────────────────────────────────────────────


class ConnectorTokenNeeded(_Payload):
    connector: str


class JobUpdate(_Payload):
    jobId: str
    kind: str
    state: Literal["queued", "running", "done", "error"]
    detail: dict[str, Any] = {}


class IdeaUpdate(_Payload):
    idea: dict[str, Any]


class IdeaDelete(_Payload):
    id: str


class ProjectUpdate(_Payload):
    """The project record, as GET /ideas/projects/{id} returns it."""

    id: str


class WakeStatePayload(_Payload):
    """The device's wake state. A fact about a device, not a session: it goes to the tenant the
    device is bound to (see events.py, "Who receives what")."""

    state: Literal["off", "armed", "listening", "error"]
    model: str | None = None
    phrase: str | None = None
    trigger: str | None = None
    meeting_id: str | None = None


class WakeSegment(_Payload):
    id: str
    text: str
    state: str
    startMs: int
    endMs: int


class WakeTranscript(_Payload):
    """What the recognizer heard in a voice session's turn."""

    meetingId: str
    turn: int
    segment: WakeSegment


class WakeReply(_Payload):
    """A reply in a voice session: `thinking` when the turn has closed, `done` with what was spoken."""

    meetingId: str
    turn: int
    phase: Literal["thinking", "done"]
    userText: str
    text: str
    interrupted: bool | None = None


class WakeDelegate(_Payload):
    """A delegated task's status. Status only: a result reaches a person through the gate."""

    taskId: str
    meetingId: str | None = None
    status: str


class CaptureHealthPayload(_Payload):
    """The device's capture health (a stall, a reopen). Device level, like wake_state."""

    issue: str
    action: str
    value: float | None = None


class MeetingPipeline(_Payload):
    """A step after Stop changed state (app/local/pipeline.py), published to the meeting's tenant. `reason` is a
    short code, never model or transcript text; null when there is nothing to add. GET
    /v1/meetings/{id}/pipeline answers the same for every step at once."""

    meetingId: str
    step: Literal["transcript", "summary", "tasks"]
    state: Literal["working", "ready", "failed", "skipped"]
    reason: str | None


class ReminderDue(_Payload):
    """A reminder or timer came due (PL46), to the item's tenant, for the app to show a
    notification. `text` is the reminder's words as the person reads them in it, after the
    response gate: null for a timer, and null when the gate refused the words (the app then shows
    a neutral line naming only the time). `dueAt` is ISO 8601 with the offset of the zone it was
    set in. `missed`: it came due while the engine was not running. `spoken`: its line was said
    on the device (false inside quiet hours, on another person's device, or with no voice)."""

    id: str
    kind: Literal["reminder", "timer"]
    text: str | None
    dueAt: str
    missed: bool
    spoken: bool


class StreamReset(_Payload):
    """Sent instead of a replay when the events a reconnecting client missed are no longer held
    (or its last id is not from this stream: the engine restarted). The client reloads the state
    it shows; the stream carries on live from this event's id."""

    reason: Literal["gap", "unknown_id"]
    lastEventId: str


STREAM_EVENTS: dict[str, type[BaseModel]] = {
    "source.update": Source,
    "model.progress": Model,
    "connector.token_needed": ConnectorTokenNeeded,
    "job.update": JobUpdate,
    "idea.update": IdeaUpdate,
    "idea.delete": IdeaDelete,
    "project.update": ProjectUpdate,
    "meeting.update": Meeting,
    "meeting.pipeline": MeetingPipeline,
    "wake_state": WakeStatePayload,
    "wake.transcript": WakeTranscript,
    "wake.reply": WakeReply,
    "wake.delegate": WakeDelegate,
    "capture_health": CaptureHealthPayload,
    "reminder.due": ReminderDue,
    RESET: StreamReset,
}

# Kinds published by a voice session: they carry sessionId and seq.
SESSION_EVENTS = frozenset({"wake.transcript", "wake.reply", "wake.delegate"})
# Kinds about a device rather than a session: they go to the tenant the device is bound to.
DEVICE_EVENTS = frozenset({"wake_state", "capture_health"})
# Kinds about the machine, not a person: every subscriber gets them (publish(broadcast=True)).
MACHINE_EVENTS = frozenset({"model.progress"})


# ── meeting bus payloads ──────────────────────────────────────────────────────


class MeetingStatus(_Payload):
    state: Literal["recording", "processing", "done"]


class SummaryReady(_Payload):
    pass


class NoteUpdated(_Payload):
    blockId: str


class AudioWarning(_Payload):
    droppedMs: float


class AssistSuggestion(_Payload):
    id: str
    offsetMs: int
    question: str
    answer: str
    citations: list[dict[str, Any]]


class ReplyPayload(_Payload):
    id: str
    text: str
    turn_n: int
    gate: Any = None
    category: str | None = None
    bargeIn: bool | None = None


MEETING_EVENTS: dict[str, type[BaseModel]] = {
    "transcript.partial": Segment,
    "transcript.stable": Segment,
    "enhanced.update": EnhancedBlock,
    "assist.suggestion": AssistSuggestion,
    "status": MeetingStatus,
    "summary.ready": SummaryReady,
    "note.updated": NoteUpdated,
    "audio.warning": AudioWarning,
    "reply": ReplyPayload,
}


def validate(kind: str, data: Any, *, family: dict[str, type[BaseModel]] = STREAM_EVENTS) -> BaseModel:
    """The payload of `kind` as its model; raises KeyError for an unknown kind and
    pydantic.ValidationError for a payload that does not fit."""
    return family[kind].model_validate(data)


def json_schema() -> dict[str, Any]:
    """The schema a client generator reads: the version, the envelope, and each kind's payload."""
    models = [Envelope, *STREAM_EVENTS.values(), *MEETING_EVENTS.values()]
    refs, top = models_json_schema([(m, "serialization") for m in dict.fromkeys(models)],
                                   ref_template="#/$defs/{model}")
    ref = {model: refs[(model, "serialization")]["$ref"] for model in dict.fromkeys(models)}
    return {
        "schemaVersion": SCHEMA_VERSION,
        "envelope": ref[Envelope],
        "streamEvents": {kind: ref[m] for kind, m in STREAM_EVENTS.items()},
        "sessionEvents": sorted(SESSION_EVENTS),
        "deviceEvents": sorted(DEVICE_EVENTS),
        "machineEvents": sorted(MACHINE_EVENTS),
        "meetingEvents": {kind: ref[m] for kind, m in MEETING_EVENTS.items()},
        "$defs": top.get("$defs", {}),
    }
