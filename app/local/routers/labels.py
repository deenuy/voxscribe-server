"""Labels endpoints."""

import uuid

from fastapi import APIRouter, Response, Request

from app.local.errors import InvalidRequestError, NotFoundError
from app.local.store_sources import SourceStore

router = APIRouter(prefix="/v1/labels", tags=["labels"])


@router.get("")
async def list_labels(request: Request):
    """List all labels."""
    store = SourceStore(request.app.state.store.pool)
    labels = await store.list_labels()
    return [label.model_dump(exclude_none=True) for label in labels]


@router.post("", status_code=201)
async def create_label(request: Request, body: dict):
    """Create a new label."""
    store = SourceStore(request.app.state.store.pool)
    label_id = str(uuid.uuid4())
    name = body.get("name")
    color = body.get("color")

    if not name:
        raise InvalidRequestError("label name is required")

    label = await store.create_label(label_id, name, color)
    return label.model_dump(exclude_none=True)


@router.delete("/{label_id}", status_code=204)
async def delete_label(request: Request, label_id: str):
    """Delete a label."""
    store = SourceStore(request.app.state.store.pool)
    labels = await store.list_labels()
    if not any(label.id == label_id for label in labels):
        raise NotFoundError(f"label {label_id} not found")

    await store.delete_label(label_id)
    return Response(status_code=204)
