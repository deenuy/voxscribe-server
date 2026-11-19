"""Reminders and timers set by voice (PL45), for the app's list and its cancel button.

    GET    /v1/reminders        the caller's reminders and timers, soonest due first
    DELETE /v1/reminders/{id}   cancel one still to come: 204. Unknown: 404 not_found. One that
                                has fired, was missed or is being delivered: 409 conflict. One
                                already cancelled: 204 again (idempotent)

Rows are the caller's tenant's (the pool selects the tenant in scope). `text` is the reminder's
words as the person said them; a timer has none. `dueAt` is ISO 8601 with the offset of the zone
the time was said in. A cancel here tells the due-item scheduler, which re-plans its wake.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app.local.errors import ConflictError, NotFoundError
from app.local.voice import reminder_store as rs

router = APIRouter(prefix="/v1", tags=["reminders"])
log = logging.getLogger(__name__)


class ReminderItem(BaseModel):
    id: str
    kind: Literal["reminder", "timer"]
    text: str | None
    dueAt: str
    state: Literal["scheduled", "fired", "missed", "cancelled"]


class ReminderList(BaseModel):
    items: list[ReminderItem]


def _now(request: Request) -> datetime:
    clock = getattr(request.app.state, "clock", None)
    return clock() if clock is not None else datetime.now(UTC)


def _view(item: rs.Item) -> ReminderItem:
    return ReminderItem(id=item.id, kind=item.kind, text=item.subject, dueAt=item.due_at,
                        state=item.state)


@router.get("/reminders", response_model=ReminderList)
async def list_reminders(request: Request) -> ReminderList:
    store = rs.ReminderStore.for_pool(request.app.state.store.pool)
    return ReminderList(items=[_view(i) for i in await store.list_all()])


@router.delete("/reminders/{reminder_id}", status_code=204)
async def cancel_reminder(request: Request, reminder_id: str) -> Response:
    store = rs.ReminderStore.for_pool(request.app.state.store.pool)
    if await store.cancel(reminder_id, at=_now(request)) is None:
        item = await store.get(reminder_id)
        if item is None:
            raise NotFoundError(f"reminder {reminder_id} not found")
        if item.status != rs.CANCELLED:
            raise ConflictError(f"reminder {reminder_id} is {item.state}, not scheduled")
    else:
        scheduler = getattr(request.app.state, "due_scheduler", None)
        if scheduler is not None:
            scheduler.notify()
    return Response(status_code=204)
