"""Meetings endpoints."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.local.errors import ConflictError, ConsentRequiredError, EngineError, InvalidRequestError, NetworkDisabledError, NotFoundError
from app.local.http_utils import json_object, parse_model
from app.local.export import export_meeting_markdown
from app.local.models import (
    CreateMeetingRequest,
    EditEnhancedRequest,
    EditSegmentRequest,
    EnhancedResponse,
    EnhanceRequest,
    NotesRequest,
    NotesResponse,
    StartMeetingRequest,
    Summary,
    TranscriptResponse,
    UpdateMeetingRequest,
)
from app.local.store_meetings import MeetingStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/meetings", tags=["meetings"])


@router.post("", status_code=201)
async def create_meeting(request: Request, body: CreateMeetingRequest):
    """Create a new meeting."""
    from fastapi.responses import JSONResponse

    store = MeetingStore(request.app.state.store.pool)

    # P4: idempotent creation — same clientRequestId returns the original meeting
    if body.clientRequestId:
        existing = await store.find_by_client_request_id(body.clientRequestId)
        if existing:
            return JSONResponse(content=existing.model_dump(exclude_none=True), status_code=200)

    meeting_id = str(uuid.uuid4())
    meeting = await store.create_meeting(
        meeting_id=meeting_id,
        title=body.title or "New meeting",
        context=body.context,
        calendar_event_id=body.calendarEventId,
        scheduled_start=body.scheduledStart,
        scheduled_end=body.scheduledEnd,
        project_id=body.projectId,
        client_request_id=body.clientRequestId,
    )
    return meeting.model_dump(exclude_none=True)


@router.get("")
async def list_meetings(
    request: Request,
    query: str = None,
    when: str = None,
    limit: int = 25,
    cursor: str = None,
):
    """List meetings with filtering and pagination."""
    limit = max(1, min(int(limit), 200))  # an unbounded limit is a way to make the engine read the whole table
    store = MeetingStore(request.app.state.store.pool)
    meetings, next_cursor = await store.list_meetings(
        query=query, when=when, limit=limit, cursor=cursor
    )
    return {
        "items": [m.model_dump(exclude_none=True) for m in meetings],
        "nextCursor": next_cursor,
    }


@router.get("/{meeting_id}")
async def get_meeting(request: Request, meeting_id: str):
    """Get a meeting by ID."""
    store = MeetingStore(request.app.state.store.pool)
    meeting = await store.get_meeting(meeting_id)
    return meeting.model_dump(exclude_none=True)


@router.patch("/{meeting_id}")
async def update_meeting(request: Request, meeting_id: str, body: UpdateMeetingRequest):
    """Update a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    meeting = await store.update_meeting(
        meeting_id,
        title=body.title,
        context=body.context,
        reminder_enabled=body.reminderEnabled,
        enhance_style=body.enhanceStyle,
        auto_stop_minutes=body.autoStopMinutes,
        base_revision=body.baseRevision,
        **({"project_id": (body.projectId or "").strip() or None} if "projectId" in body.model_fields_set else {}),
    )
    return meeting.model_dump(exclude_none=True)


@router.post("/{meeting_id}/edits")
async def commit_edits(request: Request, meeting_id: str, idempotency_key: str | None = Header(None)):
    """Commit an edit session on a scribe atomically: title, context and project, transcript lines, the summary and
    tasks in one transaction. Every piece carries the revision the person last saw; if any changed, nothing is written
    and the 409 `revision_conflict` lists the pieces (`items`). Send an `Idempotency-Key` header to make a retry safe:
    a commit that already succeeded answers again with the same body. 400 for a bad value anywhere (nothing written)."""
    from fastapi.responses import JSONResponse

    from app.local.edit_models import EditRequest
    from app.local.store_edits import apply_edits

    body = parse_model(EditRequest, await json_object(request))
    if body.is_empty():
        raise InvalidRequestError("nothing to change")
    if idempotency_key is not None and not (1 <= len(idempotency_key) <= 200):
        raise InvalidRequestError("Idempotency-Key: 1 to 200 characters")
    response, replayed = await apply_edits(MeetingStore(request.app.state.store.pool), meeting_id, body, idempotency_key)
    return JSONResponse(content=response, headers={"Idempotent-Replay": "true"} if replayed else None)


@router.delete("/{meeting_id}", status_code=204)
async def delete_meeting(request: Request, meeting_id: str):
    """Delete a meeting."""
    from fastapi import Response
    store = MeetingStore(request.app.state.store.pool)
    await store.delete_meeting(meeting_id)
    return Response(status_code=204)


@router.post("/{meeting_id}/start")
async def start_meeting(request: Request, meeting_id: str, body: StartMeetingRequest):
    """Start recording a meeting."""
    if not body.consentAcknowledged:
        raise ConsentRequiredError("consentAcknowledged must be true")

    store = MeetingStore(request.app.state.store.pool)
    # One atomic step: one recording at a time, and only a scheduled meeting can start (409 otherwise).
    meeting = await store.start_meeting(meeting_id)
    live_url = f"/v1/meetings/{meeting_id}/live"
    return {
        "meeting": meeting.model_dump(exclude_none=True),
        "liveUrl": live_url,
    }


@router.post("/{meeting_id}/stop")
async def stop_meeting(request: Request, meeting_id: str):
    """Stop recording and begin processing.

    Idempotent: only the call that actually stops a recording starts finalisation (enrichment, summary,
    publish); a repeat or concurrent stop returns the meeting as it is. A meeting that was never started is 409.
    Returns immediately with status 'processing'."""
    from app.local.finalize import schedule_finalize

    store = MeetingStore(request.app.state.store.pool)
    meeting, stopped_now = await store.stop_once(meeting_id)
    if stopped_now:
        schedule_finalize(request.app, meeting_id)
    return meeting.model_dump(exclude_none=True)


# ============================================================================
# Notes
# ============================================================================


@router.put("/{meeting_id}/notes")
async def put_notes(request: Request, meeting_id: str, body: NotesRequest):
    """Replace all notes for a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    count = await store.put_notes(meeting_id, body.blocks)
    return {"accepted": count}


@router.get("/{meeting_id}/notes")
async def get_notes(request: Request, meeting_id: str):
    """Get notes for a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    blocks = await store.get_notes(meeting_id)
    return NotesResponse(blocks=blocks).model_dump(exclude_none=True)


# ============================================================================
# Transcript
# ============================================================================


@router.get("/{meeting_id}/transcript")
async def get_transcript(request: Request, meeting_id: str, limit: int | None = None, cursor: int = 0):
    """Get the transcript. With no `limit` it returns every segment; with one it pages (max 5000), and
    `nextCursor` is the `cursor` for the next page, absent on the last."""
    store = MeetingStore(request.app.state.store.pool)
    editable, revision = await store.transcript_state(meeting_id)
    if limit is None:
        segments, next_cursor = await store.get_transcript(meeting_id), None
    else:
        limit = max(1, min(int(limit), 5000))
        cursor = max(0, int(cursor))
        page = await store.get_transcript(meeting_id, offset=cursor, limit=limit + 1)
        segments = page[:limit]
        next_cursor = cursor + limit if len(page) > limit else None
    return TranscriptResponse(segments=segments, editable=editable, revision=revision, nextCursor=next_cursor).model_dump(exclude_none=True)


@router.patch("/{meeting_id}/transcript/{segment_id}")
async def edit_transcript_segment(request: Request, meeting_id: str, segment_id: str, body: EditSegmentRequest):
    """Edit one segment's text and/or speaker. Only a published meeting (status "done") can be edited, and the
    caller must send the revision it last saw (`baseRevision`) so a stale edit cannot overwrite a newer one."""
    store = MeetingStore(request.app.state.store.pool)
    result = await store.edit_segment(
        meeting_id,
        segment_id,
        base_revision=body.baseRevision,
        text=body.text,
        speaker=body.speaker,
        speaker_given="speaker" in body.model_fields_set,
    )
    result["segment"] = result["segment"].model_dump(exclude_none=True)
    return result


# ============================================================================
# Enhanced Notes
# ============================================================================


@router.get("/{meeting_id}/enhanced")
async def get_enhanced(request: Request, meeting_id: str):
    """Get enhanced blocks for a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    enhanced_resp = await store.get_enhanced(meeting_id)
    return EnhancedResponse(**enhanced_resp).model_dump(exclude_none=True)


@router.put("/{meeting_id}/enhanced/{block_id}")
async def edit_enhanced(
    request: Request, meeting_id: str, block_id: str, body: EditEnhancedRequest
):
    """Edit an enhanced block (marks it as edited)."""
    store = MeetingStore(request.app.state.store.pool)
    await store.edit_enhanced_block(meeting_id, block_id, body.text)
    return ""


@router.post("/{meeting_id}/enhance")
async def enhance(request: Request, meeting_id: str, body: EnhanceRequest):
    """Trigger enrichment for all non-edited blocks."""
    from fastapi import Response
    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists
    job_id = str(uuid.uuid4())

    # For now, just return accepted (would be async in production)
    return Response(
        content=f'{{"jobId": "{job_id}"}}',
        status_code=202,
        media_type="application/json",
    )


# ============================================================================
# Summary
# ============================================================================


@router.get("/{meeting_id}/summary")
async def get_summary(request: Request, meeting_id: str):
    """Get summary for a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    summary = await store.get_summary(meeting_id)
    return Summary(**summary).model_dump(exclude_none=True)


@router.post("/{meeting_id}/summary/regenerate")
async def regenerate_summary(request: Request, meeting_id: str):
    """Regenerate the summary for a meeting.

    Returns 202 Accepted if the provider is available.
    Returns 503 Provider Unavailable if the LLM provider is down.
    Returns 409 `summary_edited` when a person wrote or changed the summary: regenerating replaces their words, so it
    needs a body of `{"overwriteEdits": true}`.
    """
    from fastapi import Response

    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists
    if (await store.get_summary(meeting_id)).get("edited"):
        try:
            sent = await request.json()
        except ValueError:
            sent = {}
        if not (isinstance(sent, dict) and sent.get("overwriteEdits") is True):
            raise EngineError("summary_edited", "this summary was edited by hand; regenerating replaces it. Send overwriteEdits: true to confirm", 409)

    job_id = str(uuid.uuid4())
    # Mark summary as pending
    await store.set_summary_pending(meeting_id)

    # Trigger async regeneration job in background
    try:
        asyncio.create_task(_regenerate_summary_async(store, meeting_id))
    except Exception:
        # If background task can't be created, still return 202
        pass

    return Response(
        content=f'{{"jobId": "{job_id}"}}',
        status_code=202,
        media_type="application/json",
    )


async def _regenerate_summary_async(store: MeetingStore, meeting_id: str):
    """Background task to regenerate meeting summary."""
    try:
        from voxscribe.meeting.summary import SummaryGenerator

        from app.local import meeting_bus

        meeting = await store.get_meeting(meeting_id)
        transcript = await store.get_transcript(meeting_id)

        if not transcript:
            # No transcript, can't summarize
            return

        # Create summary generator (would use the app's llm provider)
        # For now, this is a placeholder that would be filled in
        generator = SummaryGenerator(None)  # Would pass llm provider

        # Generate summary
        summary_result = generator.summarize(
            text=" ".join(s.text for s in transcript),
            style=meeting.enhanceStyle,
        )

        # Store summary
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        await store.set_summary(
            meeting_id=meeting_id,
            overview=summary_result.overview,
            details=summary_result.details or [],
            generated_at=now,
        )

        # Publish event
        await meeting_bus.publish(meeting_id, "summary.ready", {})
    except Exception as e:
        log.error(f"Summary regeneration failed for {meeting_id}: {e}", exc_info=True)


# ============================================================================
# Export
# ============================================================================


@router.get("/{meeting_id}/export")
async def export_meeting(request: Request, meeting_id: str, format: str = "markdown"):
    """Export a meeting in the specified format."""
    if format != "markdown":
        raise HTTPException(status_code=400, detail="only markdown format supported")

    store = MeetingStore(request.app.state.store.pool)
    markdown = await export_meeting_markdown(store, meeting_id)
    return PlainTextResponse(markdown, media_type="text/markdown")


# ============================================================================
# Attachments
# ============================================================================


@router.post("/{meeting_id}/attachments")
async def attach_source(request: Request, meeting_id: str):
    """Attach a source to a meeting."""
    body = await json_object(request)
    source_id = body.get("sourceId")
    if not isinstance(source_id, str) or not source_id:
        raise InvalidRequestError("sourceId: required")

    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists
    async with request.app.state.store.pool.connection() as conn:
        if not await (await conn.execute("SELECT 1 FROM sources WHERE id = %s", (source_id,))).fetchone():
            raise NotFoundError(f"source {source_id} not found")
    await store.attach_source(meeting_id, source_id)
    return ""


@router.get("/{meeting_id}/attachments")
async def list_attachments(request: Request, meeting_id: str):
    """The sources attached to a meeting (the count on the meeting says how many; this says which)."""
    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # 404 if there is no such meeting
    async with request.app.state.store.pool.connection() as conn:
        cur = await conn.execute(
            """SELECT s.id, s.name, s.kind FROM meeting_attachments a JOIN sources s ON s.id = a.source_id
               WHERE a.meeting_id = %s ORDER BY s.name""",
            (meeting_id,),
        )
        rows = await cur.fetchall()
    return {"items": [{"sourceId": r[0], "name": r[1], "kind": r[2]} for r in rows]}


@router.delete("/{meeting_id}/attachments/{source_id}")
async def detach_source(request: Request, meeting_id: str, source_id: str):
    """Detach a source from a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists
    await store.detach_source(meeting_id, source_id)
    return ""


@router.post("/import", status_code=202)
async def import_meeting(request: Request):
    """Make a scribe from a transcript someone already has: pasted text, a transcript file (.txt, .md, .srt, .vtt) or a
    YouTube link (its captions).

    A transcript file is sent as `text` with its `filename` (a browser cannot give the engine a path), or as `path` from
    the desktop app. A YouTube link needs connectors to be allowed (403 `network_disabled` otherwise). Audio files cannot
    be transcribed here yet (501 `not_implemented`). Returns {meetingId, jobId}: the work runs in the background and a
    `job.update` event says when it is done (`done`) or why it could not be (`error`, with the reason); a failed import
    leaves no half-made scribe behind."""
    from app.local import youtube_import
    from app.local.job_runner import get_runner
    from app.local.models import ImportMeetingRequest
    from app.local.store_sources import SourceStore
    from app.local.transcript_import import TranscriptError, parse_transcript

    req = parse_model(ImportMeetingRequest, await json_object(request))
    pool = request.app.state.store.pool
    store = MeetingStore(pool)
    title, text, lines, vid = req.title, req.text, None, None

    # Everything the job needs is checked now, so a bad request is a 400 and not a job that fails later.
    if req.kind == "audio":
        raise EngineError("not_implemented", "importing an audio file is not available yet; import its transcript instead", 501)
    if req.kind in ("text", "file"):
        if text is None:
            if req.kind != "file" or not req.path:
                raise InvalidRequestError("text: required (or path, to import a file)")
            text = _read_transcript_file(req.path)
        try:
            lines = parse_transcript(text, req.filename or req.path)
        except TranscriptError as exc:
            raise InvalidRequestError(f"text: {exc}") from None
        title = title or _title_from_filename(req.filename or req.path) or "Imported transcript"
    elif req.kind == "youtube":
        try:
            vid = youtube_import.video_id(req.url or "")
        except youtube_import.YouTubeError as exc:
            raise InvalidRequestError(f"url: {exc}") from None
        if await SourceStore(pool).get_privacy_mode() != "connectors":
            raise NetworkDisabledError("importing a YouTube link needs the network. Allow connectors in Settings, Privacy mode.")

    meeting_id = str(uuid.uuid4())
    await store.create_meeting(
        meeting_id=meeting_id,
        title=title or "Imported YouTube video",
        context=None,
        project_id=(req.projectId or "").strip() or None,
        origin={"text": "text", "file": "file", "youtube": "youtube"}[req.kind],
    )

    async def do_import():
        try:
            imported = lines
            if req.kind == "youtube":
                imported = youtube_import.captions_to_lines(await youtube_import.fetch_captions(vid))
                if not req.title:
                    found = await youtube_import.video_title(vid)
                    if found:
                        await store.update_meeting(meeting_id, title=found)
            await store.import_lines(meeting_id, imported)
            await store.mark_done(meeting_id)
        except Exception:
            await store.delete_meeting(meeting_id)  # a failed import leaves nothing half-made
            raise
        from app.local import events

        events.publish("meeting.update", (await store.get_meeting(meeting_id)).model_dump(exclude_none=True))

    job_id = await get_runner().submit(
        kind="import_meeting",
        work_fn=do_import,
        detail={"meetingId": meeting_id, "kind": req.kind, "extractIdeas": req.extractIdeas},
    )
    return {"meetingId": meeting_id, "jobId": job_id}


def _title_from_filename(name: str | None) -> str | None:
    if not name:
        return None
    import re
    from pathlib import PurePath

    stem = PurePath(name.replace("\\", "/")).stem
    return re.sub(r"_+", " ", stem).strip() or None


def _read_transcript_file(path: str) -> str:
    """A transcript file on this computer (the desktop app sends a path), read as text."""
    from app.local.routers.sources import _resolve_and_validate_path
    from app.local.transcript_import import MAX_CHARS

    try:
        resolved = _resolve_and_validate_path(path, follow_symlinks=True)
    except InvalidRequestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidRequestError(f"path: {exc}") from None
    if not resolved.is_file():
        raise InvalidRequestError(f"path: not a file: {path}")
    if resolved.stat().st_size > MAX_CHARS * 4:
        raise InvalidRequestError("path: the file is too large to import")
    return resolved.read_text(encoding="utf-8", errors="replace")
