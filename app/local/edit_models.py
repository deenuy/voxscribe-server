"""The request for one atomic edit commit on a scribe (see store_edits.py)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .models import SummaryDetail

Status = Literal["todo", "in_progress", "done"]


class MetaEdit(BaseModel):
    """Title, context and project. Only fields that are sent change; `projectId: null` takes the scribe out of its project."""

    baseRevision: int = Field(ge=0)
    title: str | None = Field(default=None, max_length=300)
    context: str | None = Field(default=None, max_length=20000)
    projectId: str | None = Field(default=None, max_length=200)


class SegmentEdit(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    baseRevision: int = Field(ge=0)
    text: str | None = Field(default=None, max_length=20000)
    speaker: str | None = Field(default=None, max_length=120)  # "" or null clears it


class SummaryEdit(BaseModel):
    baseRevision: int = Field(ge=0)
    overview: str = Field(min_length=1, max_length=20000)
    details: list[SummaryDetail] = Field(default_factory=list, max_length=50)


class TaskCreate(BaseModel):
    tempId: str = Field(min_length=1, max_length=64)  # the caller's name for it until it has an id
    title: str = Field(min_length=1, max_length=300)
    detail: str | None = Field(default=None, max_length=2000)
    owner: str | None = Field(default=None, max_length=120)
    dueDate: date | None = None
    projectId: str | None = Field(default=None, max_length=200)
    status: Status = "todo"


class TaskUpdate(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    baseRevision: int = Field(ge=0)
    title: str | None = Field(default=None, min_length=1, max_length=300)
    detail: str | None = Field(default=None, max_length=2000)
    owner: str | None = Field(default=None, max_length=120)
    dueDate: date | None = None
    projectId: str | None = Field(default=None, max_length=200)
    status: Status | None = None


class TaskDelete(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    baseRevision: int | None = Field(default=None, ge=0)  # when sent, deleting a task that changed since is a conflict


class TasksEdit(BaseModel):
    create: list[TaskCreate] = Field(default_factory=list, max_length=100)
    update: list[TaskUpdate] = Field(default_factory=list, max_length=200)
    delete: list[TaskDelete] = Field(default_factory=list, max_length=200)


class EditRequest(BaseModel):
    meeting: MetaEdit | None = None
    segments: list[SegmentEdit] = Field(default_factory=list, max_length=1000)
    summary: SummaryEdit | None = None
    tasks: TasksEdit | None = None

    def is_empty(self) -> bool:
        t = self.tasks
        return not (self.meeting or self.segments or self.summary or (t and (t.create or t.update or t.delete)))
