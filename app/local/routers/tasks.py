"""Tasks: a to-do list per scribe, by hand or drafted by the AI from the transcript."""

import logging
from datetime import date
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from app.local.errors import ConflictError, InvalidRequestError
from app.local.http_utils import json_object, parse_model
from app.local.store_meetings import MeetingStore
from app.local.store_tasks import TaskStore
from app.local.task_service import draft_tasks

router = APIRouter(prefix="/v1", tags=["tasks"])
log = logging.getLogger(__name__)


Status = Literal["todo", "in_progress", "done"]


class CreateTaskRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    detail: str | None = Field(default=None, max_length=2000)
    owner: str | None = Field(default=None, max_length=120)
    dueDate: date | None = None
    projectId: str | None = Field(default=None, max_length=200)
    status: Status = "todo"


class UpdateTaskRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    detail: str | None = Field(default=None, max_length=2000)
    owner: str | None = Field(default=None, max_length=120)
    dueDate: date | None = None
    projectId: str | None = Field(default=None, max_length=200)
    status: Status | None = None
    baseRevision: int | None = Field(default=None, ge=0)  # optional: when sent, a stale one is a 409 `revision_conflict`


def _blank_title(title: str | None) -> None:
    if title is not None and not title.strip():
        raise InvalidRequestError("title: cannot be blank")


def _text(value: str | None) -> str | None:
    """Trimmed text, or None for blank (how an optional text field is cleared)."""
    return (value or "").strip() or None


@router.get("/meetings/{meeting_id}/tasks")
async def list_scribe_tasks(request: Request, meeting_id: str) -> dict:
    await MeetingStore(request.app.state.store.pool).get_meeting(meeting_id)  # 404 for an unknown scribe
    return {"items": await TaskStore(request.app.state.store.pool).list_for_meeting(meeting_id)}


@router.post("/meetings/{meeting_id}/tasks", status_code=201)
async def add_task(request: Request, meeting_id: str):
    """Add a task by hand. The same title twice in one scribe is a 409."""
    body = parse_model(CreateTaskRequest, await json_object(request))
    _blank_title(body.title)
    await MeetingStore(request.app.state.store.pool).get_meeting(meeting_id)
    return await TaskStore(request.app.state.store.pool).create(
        meeting_id, title=body.title, detail=_text(body.detail), owner=_text(body.owner), due_date=body.dueDate, project_id=_text(body.projectId), status=body.status,
    )


@router.post("/meetings/{meeting_id}/tasks/generate")
async def generate_tasks(request: Request, meeting_id: str) -> dict:
    """Draft tasks from the transcript (and the scribe's project, when it has one).

    Adds only titles the scribe does not have yet, so it is safe to run again: tasks that are done, edited or written
    by hand are never touched or repeated. Answers `{created, skipped}`. 409 until the scribe has finished and has a
    transcript; 503 when the AI is not running; 502 when its answer cannot be used (try again)."""
    pool = request.app.state.store.pool
    meetings = MeetingStore(pool)
    meeting = await meetings.get_meeting(meeting_id)
    if meeting.status == "recording":
        raise ConflictError("tasks can be drafted once the scribe has finished")
    segments = await meetings.get_transcript(meeting_id)
    spoken = [s for s in segments if s.text and s.text.strip()]
    lines = [s.text.strip() for s in spoken]
    labelled = [f"{s.speaker}: {s.text.strip()}" if s.speaker else s.text.strip() for s in spoken]  # who said it, for the AI
    if not lines:
        raise ConflictError("this scribe has no transcript to draft tasks from")
    project = None
    if getattr(meeting, "projectId", None):
        from app.local.store_ideas import ProjectStore

        try:
            project = await ProjectStore(pool).get_project(meeting.projectId)
        except Exception:  # a project that cannot be read only costs the AI some context
            project = None
    providers = getattr(request.app.state, "providers", {})
    held = next((str(v)[:10] for v in (getattr(meeting, "startedAt", None), getattr(meeting, "scheduledStart", None), getattr(meeting, "createdAt", None)) if v), None)
    drafts = await draft_tasks(providers.get("llm"), lines=lines, project=project, held_on=held, labelled=labelled, fake=request.app.state.settings.providers == "fake")
    created, skipped = await TaskStore(pool).add_generated(meeting_id, drafts)
    return {"created": created, "skipped": skipped}


@router.get("/tasks")
async def list_tasks(request: Request, status: str | None = None, projectId: str | None = None, meetingId: str | None = None, limit: int = 50) -> dict:
    """Every task, newest first: the to-do list across scribes. Filter by status, project or scribe."""
    if status is not None and status not in ("todo", "in_progress", "done"):
        raise InvalidRequestError("status: one of todo, in_progress, done")
    if not 1 <= limit <= 200:
        raise InvalidRequestError("limit: 1 to 200")
    items, total = await TaskStore(request.app.state.store.pool).list_all(status=status, project_id=projectId, meeting_id=meetingId, limit=limit)
    return {"items": items, "total": total}


@router.patch("/tasks/{task_id}")
async def update_task(request: Request, task_id: str):
    """Change any of: title, detail, owner, dueDate, projectId, status (todo, in_progress, done). Send only what changes.
    Sending null (or empty text) for detail, owner, dueDate or projectId clears it; a cleared projectId falls back to
    the scribe's project. title and status cannot be null."""
    raw = await json_object(request)
    body = parse_model(UpdateTaskRequest, raw)
    fields = [k for k in ("title", "detail", "owner", "dueDate", "projectId", "status") if k in raw]
    if not fields:
        raise InvalidRequestError("nothing to change: send title, detail, owner, dueDate, projectId and/or status")
    for required in ("title", "status"):
        if required in raw and raw[required] is None:
            raise InvalidRequestError(f"{required}: cannot be null")
    _blank_title(body.title)
    values = {"title": body.title, "detail": _text(body.detail), "owner": _text(body.owner), "dueDate": body.dueDate, "projectId": _text(body.projectId), "status": body.status}
    return await TaskStore(request.app.state.store.pool).update(task_id, {k: values[k] for k in fields}, body.baseRevision)


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(request: Request, task_id: str):
    """Delete a task. Deleting one that is already gone is also 204."""
    await TaskStore(request.app.state.store.pool).delete(task_id)
    return Response(status_code=204)
