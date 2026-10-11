"""Sessions endpoints with lifecycle logging.

Device id comes from X-Device-Id header. Session context is set for each log
line within a request so the session trace is continuous from submission through
processing.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..errors import NotFoundError, QueueUnavailable
from ..ids import next_idea_id
from ..jobs import Job
from ..logging_config import get_logger, set_session_context, clear_session_context
from ..models import SessionFromUrl, SessionStart
from ..sessions import DATA_DIR, hub
from ..store import store
from ..vocab import now_iso

log = get_logger(__name__)
router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


def _ctx(request: Request) -> tuple[str, str, str]:
    return request.state.tenant_id, request.state.workspace_id, request.state.device_id


def _new_session_record(tenant, workspace, idea_id, title, source_type,
                        project_id, context, participants, device_id):
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    sess = {
        "sessionId": session_id,
        "ideaId": idea_id,
        "tenantId": tenant,
        "workspaceId": workspace,
        "projectId": project_id,
        "title": title,
        "sourceType": source_type,
        "participants": participants,
        "context": context,
        "deviceId": device_id,
        "status": "processing",
        "createdAt": now_iso(),
    }
    store.put_session(sess)
    return sess


def _save_upload(idea_id: str, upload: UploadFile) -> str:
    folder = os.path.join(DATA_DIR, "sessions", idea_id)
    os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(upload.filename or "audio")[1] or ".bin"
    path = os.path.join(folder, f"{idea_id}-source{ext}")
    with open(path, "wb") as fh:
        fh.write(upload.file.read())
    log.debug("session.audio_saved", extra={"path": path, "idea_id": idea_id})
    return path


def _job_from_session(sess: dict, audio_path: Optional[str]) -> Job:
    return Job(
        session_id=sess["sessionId"],
        idea_id=sess["ideaId"],
        tenant_id=sess["tenantId"],
        workspace_id=sess["workspaceId"],
        project_id=sess.get("projectId"),
        session_title=sess["title"],
        source_type=sess["sourceType"],
        source_ref=os.path.basename(audio_path) if audio_path else "",
        participants=sess.get("participants", 1),
        context=sess.get("context", ""),
        audio_path=audio_path,
    )


async def _enqueue(request: Request, sess: dict, audio_path: Optional[str]) -> None:
    queue = getattr(request.app.state, "queue", None)
    if queue is None:
        raise QueueUnavailable("processing queue is not running")
    await queue.enqueue(_job_from_session(sess, audio_path))


@router.post("/start")
def start_session(request: Request, body: SessionStart):
    """Claim the device, pre-allocate the idea id.
    Raises 409 if this device already has an active recording."""
    tenant, workspace, device_id = _ctx(request)
    idea_id = next_idea_id()
    sess = _new_session_record(tenant, workspace, idea_id, body.title,
                               body.source_type, body.project_id, body.context,
                               body.participants, device_id)
    sess["status"] = "recording"
    store.put_session(sess)
    store.claim_device(tenant, device_id, sess["sessionId"])  # may raise DeviceBusy
    token = set_session_context(sess["sessionId"], tenant, device_id)
    try:
        log.info(
            "session.started",
            extra={"idea_id": idea_id, "title": body.title,
                   "participants": body.participants, "project": body.project_id},
        )
    finally:
        clear_session_context(token)
    return {"sessionId": sess["sessionId"], "ideaId": idea_id, "status": "recording"}


@router.post("/from-file")
async def from_file(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("Untitled session"),
    participants: int = Form(1),
    context: str = Form(""),
    projectId: Optional[str] = Form(None),
    qTitle: Optional[str] = Query(None, alias="title"),
    qParticipants: Optional[int] = Query(None, alias="participants"),
    qContext: Optional[str] = Query(None, alias="context"),
    qProjectId: Optional[str] = Query(None, alias="projectId"),
):
    """One-shot upload: recording already finished client-side, no device claim.
    Enqueues immediately and returns."""
    tenant, workspace, device_id = _ctx(request)
    title = qTitle or title
    participants = qParticipants if qParticipants is not None else participants
    context = qContext or context
    project_id = qProjectId or projectId

    idea_id = next_idea_id()
    sess = _new_session_record(tenant, workspace, idea_id, title, "file_upload",
                               project_id, context, participants, device_id)
    audio_path = _save_upload(idea_id, file)
    await _enqueue(request, sess, audio_path)
    token = set_session_context(sess["sessionId"], tenant, device_id)
    try:
        log.info(
            "session.submitted",
            extra={"idea_id": idea_id, "title": title, "participants": participants,
                   "file": file.filename, "path": "from-file"},
        )
    finally:
        clear_session_context(token)
    return {"sessionId": sess["sessionId"], "ideaId": idea_id, "status": "processing"}


@router.post("/{session_id}/from-file")
async def attach_file(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    participants: Optional[int] = Form(None),
    context: Optional[str] = Form(None),
    projectId: Optional[str] = Form(None),
):
    """Finalize a started session: upload the recording, send metadata as query
    parameters or form fields, release the device, enqueue processing."""
    tenant, workspace, device_id = _ctx(request)
    sess = store.get_session(tenant, workspace, session_id)
    if not sess:
        raise NotFoundError("session not found")
    if title:
        sess["title"] = title
    if participants is not None:
        sess["participants"] = participants
    if context:
        sess["context"] = context
    if projectId:
        sess["projectId"] = projectId
    sess["status"] = "processing"
    store.put_session(sess)
    store.release_device(tenant, sess.get("deviceId", device_id))
    audio_path = _save_upload(sess["ideaId"], file)
    await _enqueue(request, sess, audio_path)
    token = set_session_context(session_id, tenant, device_id)
    try:
        log.info(
            "session.finalized",
            extra={"idea_id": sess["ideaId"], "title": sess["title"],
                   "file": file.filename, "device_released": sess.get("deviceId", device_id)},
        )
    finally:
        clear_session_context(token)
    return {"sessionId": session_id, "ideaId": sess["ideaId"], "status": "processing"}


@router.post("/from-url")
async def from_url(request: Request, body: SessionFromUrl):
    tenant, workspace, device_id = _ctx(request)
    idea_id = next_idea_id()
    sess = _new_session_record(tenant, workspace, idea_id, body.title, "youtube",
                               body.project_id, body.context, body.participants, device_id)
    sess["sourceUrl"] = body.url
    store.put_session(sess)
    await _enqueue(request, sess, None)
    token = set_session_context(sess["sessionId"], tenant, device_id)
    try:
        log.info("session.from_url", extra={"idea_id": idea_id, "url": body.url})
    finally:
        clear_session_context(token)
    return {"sessionId": sess["sessionId"], "ideaId": idea_id, "status": "processing"}


@router.get("")
def list_sessions(request: Request):
    tenant, workspace, _ = _ctx(request)
    return store.list_sessions(tenant, workspace)


@router.get("/{session_id}")
def get_session(request: Request, session_id: str):
    tenant, workspace, _ = _ctx(request)
    sess = store.get_session(tenant, workspace, session_id)
    if not sess:
        raise NotFoundError("session not found")
    idea = store.get_idea(tenant, workspace, sess.get("ideaId", "")) if sess.get("ideaId") else None
    return {
        "session": sess,
        "idea": idea,
        "processingStatus": sess.get("status"),
        "jobStatus": store.job_status_for_session(session_id),
    }


@router.get("/{session_id}/progress")
async def progress(request: Request, session_id: str):
    tenant, workspace, device_id = _ctx(request)
    log.info("session.sse_subscribed", extra={"session_id": session_id, "device": device_id})

    async def event_stream():
        async for event in hub.subscribe(session_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
