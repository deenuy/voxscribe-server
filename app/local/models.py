"""Pydantic models for the engine contract."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Health and Status
# ============================================================================


class ProviderStatus(BaseModel):
    ok: bool
    name: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    profile: str
    capabilities: list[str]
    offline: bool
    database: ProviderStatus | None = None
    providers: dict[str, ProviderStatus]


# ============================================================================
# Meetings
# ============================================================================


class MeetingStatus(BaseModel):
    status: Literal["scheduled", "recording", "processing", "done"]


class Meeting(BaseModel):
    id: str
    title: str
    context: str | None = None
    status: Literal["scheduled", "recording", "processing", "done"]
    scheduledStart: str | None = None
    scheduledEnd: str | None = None
    startedAt: str | None = None
    endedAt: str | None = None
    calendarEventId: str | None = None
    reminderEnabled: bool = False
    enhanceStyle: Literal["standard", "concise"] = "standard"
    autoStopMinutes: int | None = None
    attachmentCount: int = 0
    hasTranscript: bool = False
    hasSummary: bool = False
    projectId: str | None = None
    origin: Literal["recorded", "text", "file", "youtube"] = "recorded"
    language: str | None = None
    segmentCount: int = 0
    wordCount: int = 0
    durationMs: int | None = None
    publishedAt: str | None = None
    transcriptRevision: int = 0
    summaryStale: bool = False
    revision: int = 0  # changes whenever the title, context or project changes: send it back as baseRevision to edit
    createdAt: str
    updatedAt: str


class MeetingListResponse(BaseModel):
    items: list[Meeting]
    nextCursor: str | None = None


def _clean(value: str | None, *, blank_to: str | None = None, field: str = "text") -> str | None:
    """Trim a text field. A blank one becomes `blank_to` (a default) or is refused when there is no default."""
    if value is None:
        return None
    value = value.strip()
    if value:
        return value
    if blank_to is not None:
        return blank_to
    raise ValueError(f"{field} must not be blank")


IDEA_TYPE_KEYS = (
    "new_product", "new_feature", "feature_enhancement", "ai_capability", "workflow_automation", "data_product",
    "platform_capability", "governance_control", "experiment", "technical_enabler", "risk_compliance",
)
IdeaType = Literal[IDEA_TYPE_KEYS]  # type: ignore[valid-type]


class CreateMeetingRequest(BaseModel):
    title: str | None = Field("New meeting", max_length=300)
    context: str | None = Field(None, max_length=20000)

    @field_validator("title")
    @classmethod
    def _title(cls, v):
        return _clean(v, blank_to="New meeting")  # nothing is ever saved with an empty title
    projectId: str | None = None
    calendarEventId: str | None = None
    scheduledStart: str | None = None
    scheduledEnd: str | None = None
    clientRequestId: str | None = Field(None, max_length=200)


class UpdateMeetingRequest(BaseModel):
    title: str | None = Field(None, max_length=300)
    context: str | None = Field(None, max_length=20000)
    reminderEnabled: bool | None = None
    enhanceStyle: Literal["standard", "concise"] | None = None
    autoStopMinutes: int | None = Field(None, ge=1, le=1440)
    baseRevision: int | None = Field(None, ge=0)  # optional: when sent, a stale one is a 409 `revision_conflict`
    # Send null to take the scribe out of its project; leave it out to keep it where it is.
    projectId: str | None = Field(None, max_length=200)

    @field_validator("title")
    @classmethod
    def _title(cls, v):
        return _clean(v, field="title")  # a rename to nothing is refused, not saved


class StartMeetingRequest(BaseModel):
    consentAcknowledged: bool = False


class StartMeetingResponse(BaseModel):
    meeting: Meeting
    liveUrl: str


class ImportMeetingRequest(BaseModel):
    kind: Literal["text", "file", "audio", "youtube"]
    title: str | None = Field(default=None, max_length=300)
    text: str | None = Field(default=None, max_length=5_000_000)  # the transcript itself: pasted, or a file the browser read
    filename: str | None = Field(default=None, max_length=255)  # tells .srt / .vtt / .txt / .md apart when `text` is a file
    path: str | None = Field(default=None, max_length=4096)  # a file on this computer (the desktop app)
    url: str | None = Field(default=None, max_length=2000)
    projectId: str | None = Field(default=None, max_length=200)
    extractIdeas: bool = False


class ImportMeetingResponse(BaseModel):
    meetingId: str
    jobId: str


# ============================================================================
# Transcript
# ============================================================================


class Segment(BaseModel):
    id: str
    startMs: int
    endMs: int
    source: Literal["mic", "system"]
    speaker: str | None = None
    text: str
    state: Literal["provisional", "stable"]
    revision: int = 0
    edited: bool = False


class TranscriptResponse(BaseModel):
    segments: list[Segment]
    # True once the meeting is published (status "done"): only then may segments be edited.
    editable: bool = False
    revision: int = 0
    nextCursor: int | None = None


class EditSegmentRequest(BaseModel):
    """Change one segment's text and/or speaker. baseRevision is the revision the client last saw."""

    text: str | None = None
    speaker: str | None = None
    baseRevision: int


# ============================================================================
# Notes (Shorthand blocks)
# ============================================================================


class Block(BaseModel):
    blockId: str = Field(min_length=1, max_length=200)
    text: str = Field(max_length=50000)
    offsetMs: int = Field(ge=0)


class NotesRequest(BaseModel):
    blocks: list[Block] = Field(max_length=5000)


class NotesResponse(BaseModel):
    blocks: list[Block]


# ============================================================================
# Enhanced Notes
# ============================================================================


class Citation(BaseModel):
    kind: Literal["transcript", "source"]
    segmentId: str | None = None
    startMs: int | None = None
    endMs: int | None = None
    sourceId: str | None = None
    chunkId: str | None = None
    title: str | None = None
    locator: str | None = None


class EnhancedBlock(BaseModel):
    blockId: str
    text: str
    status: Literal["pending", "stable", "edited"]
    citations: list[Citation] = []
    updatedAt: str


class EnhancedResponse(BaseModel):
    style: Literal["standard", "concise"]
    blocks: list[EnhancedBlock]


class EnhanceRequest(BaseModel):
    style: Literal["standard", "concise"] | None = None


class EnhanceJobResponse(BaseModel):
    jobId: str


class EditEnhancedRequest(BaseModel):
    text: str


# ============================================================================
# Summary
# ============================================================================


class SummaryDetail(BaseModel):
    title: str
    body: str


class Summary(BaseModel):
    status: Literal["none", "pending", "ready"]
    overview: str | None = None
    details: list[SummaryDetail] = []
    generatedAt: str | None = None
    revision: int = 0
    edited: bool = False  # written or changed by a person: regenerating it needs an explicit overwrite


# ============================================================================
# Sources (Knowledge)
# ============================================================================


class Label(BaseModel):
    id: str
    name: str
    color: str | None = None


class Source(BaseModel):
    id: str
    kind: Literal["file", "folder", "url", "drive"]
    uri: str
    name: str
    mime: str | None = None
    parentId: str | None = None
    status: Literal["queued", "indexing", "ready", "error"]
    error: str | None = None
    importedAt: str | None = None
    chunkCount: int = 0
    labels: list[Label] = []


class SourceListResponse(BaseModel):
    items: list[Source]
    nextCursor: str | None = None
    total: int
    countsByKind: dict[str, int]


class CreateSourceRequest(BaseModel):
    kind: Literal["file", "folder", "url", "drive"]
    uri: str
    labelIds: list[str] | None = None


class UpdateSourceRequest(BaseModel):
    labelIds: list[str]


class SourceChunk(BaseModel):
    id: str
    ordinal: int
    text: str
    locator: str | None = None


class SourceChunkListResponse(BaseModel):
    items: list[SourceChunk]
    nextCursor: str | None = None


class GraphNode(BaseModel):
    id: str
    type: Literal["source", "label", "meeting"]
    name: str


class GraphEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    kind: Literal["labelled", "attached", "child"]


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    class Config:
        populate_by_name = True


# ============================================================================
# Chat
# ============================================================================


class ChatScope(BaseModel):
    labelIds: list[str] | None = None
    meetingId: str | None = None
    sourceIds: list[str] | None = None


class Thread(BaseModel):
    id: str
    scope: ChatScope
    createdAt: str


class Message(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    text: str
    citations: list[Citation] = []
    createdAt: str


class ThreadResponse(BaseModel):
    thread: Thread
    messages: list[Message] = []


class CreateThreadRequest(BaseModel):
    scope: ChatScope | None = None


class SendMessageRequest(BaseModel):
    text: str = Field(max_length=20000)

    @field_validator("text")
    @classmethod
    def _text(cls, v):
        return _clean(v, field="text")


# ============================================================================
# Models (for selection and download)
# ============================================================================


class Model(BaseModel):
    id: str
    kind: Literal["expert", "generative"]
    name: str
    description: str
    sizeBytes: int | None = None
    required: bool
    selected: bool
    selectedRoles: list[Literal["main", "live"]] = []
    status: Literal["available", "downloading", "ready", "failed"]
    downloadedBytes: int = 0
    error: str | None = None


class ModelsListResponse(BaseModel):
    models: list[Model]


class ModelProgressEvent(BaseModel):
    event: str = "model.progress"
    model: Model


# ============================================================================
# Stats
# ============================================================================


class WeeklyStat(BaseModel):
    weekStart: str
    tokens: int


class Stats(BaseModel):
    tokensSaved: int
    weekly: list[WeeklyStat]
    meetingsTranscribed: int
    sourcesAdded: int
    assistsProvided: int
    cloudCalls: int = 0


# ============================================================================
# Calendar
# ============================================================================


class CalendarEvent(BaseModel):
    id: str
    title: str
    start: str
    end: str
    attendees: list[str] = []


class CalendarEventsResponse(BaseModel):
    items: list[CalendarEvent]


class CalendarImportRequest(BaseModel):
    kind: Literal["ics", "google"]
    path: str | None = None
    calendarId: str | None = None


class CalendarImportResponse(BaseModel):
    imported: int
    lastImportedAt: str


class CalendarStatusResponse(BaseModel):
    lastImportedAt: str | None = None
    connectedAccounts: list[str] = []


# ============================================================================
# Settings
# ============================================================================


class HotkeysSettings(BaseModel):
    toggleTranscribing: str | None = None
    showApp: str | None = None


class Settings(BaseModel):
    openAtLogin: bool = False
    reminders: bool = False
    enhanceStyle: Literal["standard", "concise"] = "standard"
    autoStopMinutes: int | None = None
    generativeModelId: str | None = None
    privacyMode: Literal["offline", "connectors"] = "offline"
    hotkeys: HotkeysSettings = HotkeysSettings()


# ============================================================================
# Live WebSocket events
# ============================================================================


class LiveEventStatus(BaseModel):
    type: str = "status"
    state: Literal["recording", "processing", "done"]


class LiveEventTranscriptPartial(BaseModel):
    type: str = "transcript.partial"
    segment: Segment


class LiveEventTranscriptStable(BaseModel):
    type: str = "transcript.stable"
    segment: Segment


class LiveEventEnhancedUpdate(BaseModel):
    type: str = "enhanced.update"
    block: EnhancedBlock


class LiveEventAssistSuggestion(BaseModel):
    type: str = "assist.suggestion"
    id: str
    offsetMs: int
    question: str
    answer: str
    citations: list[Citation]


class LiveEventSummaryReady(BaseModel):
    type: str = "summary.ready"


class LiveEventError(BaseModel):
    type: str = "error"
    code: str
    detail: str


class LiveEventPong(BaseModel):
    type: str = "pong"


# ============================================================================
# Events stream (SSE)
# ============================================================================


class SourceUpdateEvent(BaseModel):
    event: str = "source.update"
    source: Source


# ============================================================================
# Google Connector
# ============================================================================


class GoogleAccount(BaseModel):
    sub: str
    email: str
    name: str
    picture: str | None = None


class GoogleConnector(BaseModel):
    connected: bool
    account: GoogleAccount | None = None
    scopes: list[str] = []
    tokenExpiresAt: str | None = None
    lastSyncAt: str | None = None


class GoogleConnectorPutRequest(BaseModel):
    account: GoogleAccount
    accessToken: str
    expiresAt: str
    scopes: list[str]


class ConnectorTokenNeededEvent(BaseModel):
    event: str = "connector.token_needed"
    connector: str


# ============================================================================
# Auth Events
# ============================================================================


class AuthEvent(BaseModel):
    id: str
    at: str
    kind: str
    detail: dict = {}


# ============================================================================
# Ideas
# ============================================================================


class CreateIdeaRequest(BaseModel):
    title: str = Field(max_length=300)
    summary: str | None = Field(None, max_length=20000)
    type: IdeaType | None = None
    projectId: str | None = None


class UpdateIdeaRequest(BaseModel):
    title: str | None = Field(None, max_length=300)
    summary: str | None = Field(None, max_length=20000)
    type: IdeaType | None = None
    projectId: str | None = None
    featureId: str | None = None
    owner: str | None = None
    rubric: dict | None = None


class CreateProjectRequest(BaseModel):
    name: str = Field(max_length=120)
    description: str | None = Field(None, max_length=5000)
    parentId: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _clean(v, field="name")


class UpdateProjectRequest(BaseModel):
    name: str | None = Field(None, max_length=120)
    description: str | None = Field(None, max_length=5000)

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _clean(v, field="name")


class CreateFeatureRequest(BaseModel):
    name: str = Field(max_length=120)

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _clean(v, field="name")


class ExtractIdeasRequest(BaseModel):
    projectId: str | None = None


class ResscoreIdeaRequest(BaseModel):
    pass


class HandoffIdeaRequest(BaseModel):
    target: str
