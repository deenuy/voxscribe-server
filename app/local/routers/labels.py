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
    name, color = _label_fields(body, required=True)
    label = await store.create_label(label_id, name, color)
    return label.model_dump(exclude_none=True)


def _label_fields(body: dict, *, required: bool) -> tuple[str | None, str | None]:
    name, color = body.get("name"), body.get("color")
    if name is not None and (not isinstance(name, str) or not name.strip() or len(name.strip()) > 80):
        raise InvalidRequestError("name: a label name is 1 to 80 characters")
    if required and name is None:
        raise InvalidRequestError("label name is required")
    if color is not None and (not isinstance(color, str) or len(color) > 32):
        raise InvalidRequestError("color: text of at most 32 characters")
    return (name.strip() if name else None), color


@router.patch("/{label_id}")
async def update_label(request: Request, label_id: str, body: dict):
    """Rename a label or change its colour. A name already in use is a 409."""
    name, color = _label_fields(body, required=False)
    if name is None and color is None:
        raise InvalidRequestError("nothing to change: send name and/or color")
    store = SourceStore(request.app.state.store.pool)
    label = await store.update_label(label_id, name, color)
    if label is None:
        raise NotFoundError(f"label {label_id} not found")
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
