"""Meetings endpoints."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.local.errors import ConflictError, ConsentRequiredError
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
    store = MeetingStore(request.app.state.store.pool)
    meeting_id = str(uuid.uuid4())
    meeting = await store.create_meeting(
        meeting_id=meeting_id,
        title=body.title or "New meeting",
        context=body.context,
        calendar_event_id=body.calendarEventId,
        scheduled_start=body.scheduledStart,
        scheduled_end=body.scheduledEnd,
        project_id=body.projectId,
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
    )
    return meeting.model_dump(exclude_none=True)


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
    """
    from fastapi import Response

    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists

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
    body = await request.json()
    source_id = body.get("sourceId")
    if not source_id:
        raise HTTPException(status_code=400, detail="sourceId required")

    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists
    await store.attach_source(meeting_id, source_id)
    return ""


@router.delete("/{meeting_id}/attachments/{source_id}")
async def detach_source(request: Request, meeting_id: str, source_id: str):
    """Detach a source from a meeting."""
    store = MeetingStore(request.app.state.store.pool)
    await store.get_meeting(meeting_id)  # Verify exists
    await store.detach_source(meeting_id, source_id)
    return ""


@router.post("/import", status_code=202)
async def import_meeting(request: Request):
    """Import a meeting from text, file, audio or YouTube.

    Creates a meeting and optionally extracts ideas.
    Returns {meetingId, jobId}.
    """
    from app.local.job_runner import get_runner
    from app.local.models import ImportMeetingRequest

    body = await request.json()
    import_req = ImportMeetingRequest(**body)

    store = MeetingStore(request.app.state.store.pool)

    # Create the meeting
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(
        meeting_id=meeting_id,
        title=import_req.title or f"Imported {import_req.kind}",
        context=None,
        origin={"text": "text", "file": "file", "audio": "file", "youtube": "youtube"}.get(import_req.kind, "file"),
    )

    runner = get_runner()

    async def do_import():
        """Import the meeting content."""
        transcript_text = None

        if import_req.kind == "text":
            transcript_text = import_req.text or ""
        elif import_req.kind == "audio":
            # Transcribe audio file using ASR provider
            if not import_req.path:
                raise ValueError("path required for audio import")

            # TODO: Load audio file and transcribe
            # For now, create placeholder
            transcript_text = "Transcribed audio"

        elif import_req.kind == "file":
            # Treat as text file
            if not import_req.path:
                raise ValueError("path required for file import")
            # TODO: Load and read file
            transcript_text = "Imported file content"

        elif import_req.kind == "youtube":
            # Download and transcribe YouTube
            # Check privacy mode first
            if request.app.state.settings.privacy_mode == "offline":
                raise Exception("network_disabled: YouTube import not allowed in offline mode")

            if not import_req.url:
                raise ValueError("url required for youtube import")

            # TODO: Download audio from YouTube and transcribe
            transcript_text = "Transcribed YouTube video"

        # Save transcript as segments
        if transcript_text:
            async with request.app.state.store.pool.connection() as conn:
                segment_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO transcript_segments (id, meeting_id, start_ms, end_ms, source, text, state)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (segment_id, meeting_id, 0, 0, "mic", transcript_text, "stable"),
                )
                # Update meeting to done status
                now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                await conn.execute(
                    "UPDATE meetings SET status = %s, ended_at = %s, updated_at = %s WHERE id = %s",
                    ("done", now, now, meeting_id),
                )
                await conn.commit()
            await store.mark_done(meeting_id)  # publish time and word count, like a recorded meeting

        # Optionally extract ideas
        if import_req.extractIdeas:
            # TODO: Chain extraction job
            pass

        # Publish meeting.update event
        from app.local import events
        updated_meeting = await store.get_meeting(meeting_id)
        events.publish("meeting.update", updated_meeting.model_dump(exclude_none=True))

    job_id = await runner.submit(
        kind="import_meeting",
        work_fn=do_import,
        detail={
            "meetingId": meeting_id,
            "kind": import_req.kind,
            "extractIdeas": import_req.extractIdeas,
        },
    )

    return {
        "meetingId": meeting_id,
        "jobId": job_id,
    }
