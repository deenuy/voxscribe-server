"""Pydantic models for the engine contract."""

from typing import Literal

from pydantic import BaseModel, Field

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
    createdAt: str
    updatedAt: str


class MeetingListResponse(BaseModel):
    items: list[Meeting]
    nextCursor: str | None = None


class CreateMeetingRequest(BaseModel):
    title: str | None = "New meeting"
    context: str | None = None
    projectId: str | None = None
    calendarEventId: str | None = None
    scheduledStart: str | None = None
    scheduledEnd: str | None = None


class UpdateMeetingRequest(BaseModel):
    title: str | None = None
    context: str | None = None
    reminderEnabled: bool | None = None
    enhanceStyle: Literal["standard", "concise"] | None = None
    autoStopMinutes: int | None = None


class StartMeetingRequest(BaseModel):
    consentAcknowledged: bool = False


class StartMeetingResponse(BaseModel):
    meeting: Meeting
    liveUrl: str


class ImportMeetingRequest(BaseModel):
    kind: Literal["text", "file", "audio", "youtube"]
    title: str | None = None
    text: str | None = None
    path: str | None = None
    url: str | None = None
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
    blockId: str
    text: str
    offsetMs: int


class NotesRequest(BaseModel):
    blocks: list[Block]


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
    text: str


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
    title: str
    summary: str | None = None
    type: str | None = None
    projectId: str | None = None


class UpdateIdeaRequest(BaseModel):
    title: str | None = None
    summary: str | None = None
    type: str | None = None
    projectId: str | None = None
    featureId: str | None = None
    owner: str | None = None
    rubric: dict | None = None


class CreateProjectRequest(BaseModel):
    name: str
    description: str | None = None
    parentId: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class CreateFeatureRequest(BaseModel):
    name: str


class ExtractIdeasRequest(BaseModel):
    projectId: str | None = None


class ResscoreIdeaRequest(BaseModel):
    pass


class HandoffIdeaRequest(BaseModel):
    target: str
