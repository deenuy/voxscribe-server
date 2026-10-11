"""Sources endpoints."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Response, Query, Request

from app.local import events
from app.local.errors import (
    ConnectorNotConnectedError,
    ConnectorTokenExpiredError,
    InvalidRequestError,
    NetworkDisabledError,
    NotFoundError,
)
from app.local.models import (
    CreateSourceRequest,
    UpdateSourceRequest,
)
from app.local.store_sources import SourceStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/sources", tags=["sources"])


def _resolve_and_validate_path(base_path: str, follow_symlinks: bool = True) -> Path:
    """Resolve path with symlink safety.

    Ensures the resolved path stays within the original folder.
    Raises InvalidRequestError if symlink escapes the folder.
    """
    base = Path(base_path).resolve()

    # If follow_symlinks is False and path is a symlink, reject it
    if not follow_symlinks and Path(base_path).is_symlink():
        raise InvalidRequestError(f"symlinks not allowed: {base_path}")

    # For folders, verify no symlinks escape the boundary
    if base.is_dir():
        for item in base.rglob("*"):
            if item.is_symlink():
                try:
                    resolved = item.resolve()
                    # Check if resolved path is under base
                    resolved.relative_to(base)
                except ValueError as e:
                    raise InvalidRequestError(
                        f"symlink escapes folder boundary: {item}"
                    ) from e

    return base


@router.post("", status_code=201)
async def create_source(request: Request, body: CreateSourceRequest):
    """Create a new source (file, folder, url, or drive)."""
    store = SourceStore(request.app.state.store.pool)

    source_id = str(uuid.uuid4())
    uri = body.uri
    kind = body.kind

    # Validate kind
    if kind not in ["file", "folder", "url", "drive"]:
        raise InvalidRequestError(f"unsupported source kind: {kind}")

    # For file and folder, validate path exists and is readable with symlink safety
    if kind in ["file", "folder"]:
        try:
            path = _resolve_and_validate_path(uri, follow_symlinks=True)
        except InvalidRequestError:
            raise
        except Exception as e:
            raise InvalidRequestError(f"invalid path: {str(e)}") from e

        if not path.exists():
            raise InvalidRequestError(f"{kind} does not exist: {uri}")
        if not path.is_file() and kind == "file":
            raise InvalidRequestError(f"path is not a file: {uri}")
        if not path.is_dir() and kind == "folder":
            raise InvalidRequestError(f"path is not a directory: {uri}")

        name = path.name
        mime = None

    # For URL and drive, check privacy mode
    elif kind == "url":
        privacy_mode = await store.get_privacy_mode()
        if privacy_mode != "connectors":
            raise NetworkDisabledError("network sources disabled in privacy mode")
        name = uri
        mime = None

    elif kind == "drive":
        privacy_mode = await store.get_privacy_mode()
        if privacy_mode != "connectors":
            raise NetworkDisabledError("network sources disabled in privacy mode")

        # Check if Google account is connected and token is fresh
        async with request.app.state.store.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT token_expires_at FROM connector_accounts WHERE connector = %s",
                ("google",),
            )
            account = await cur.fetchone()

        if not account:
            events.publish("connector.token_needed", {"connector": "google"})
            raise ConnectorNotConnectedError("no connected Google account")

        # Check token expiry
        from datetime import datetime
        expires_at_str = account[0]
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                if expires_at <= datetime.now(expires_at.tzinfo):
                    events.publish("connector.token_needed", {"connector": "google"})
                    raise ConnectorTokenExpiredError("Google token expired, please re-authorize")
            except (ValueError, AttributeError):
                pass

        # For now, just use the Drive file/folder ID as name
        name = uri

    # Create source with status queued
    source = await store.create_source(
        source_id=source_id,
        kind=kind,
        uri=uri,
        name=name,
        mime=mime,
        parent_id=None,
    )

    # Add labels if provided
    if body.labelIds:
        await store.update_source_labels(source_id, body.labelIds)
        source = await store.get_source(source_id)

    # Publish event
    events.publish("source.update", source.model_dump(exclude_none=True))

    # Queue for ingestion if ingestion worker is available
    if hasattr(request.app.state, "ingestion_worker"):
        await request.app.state.ingestion_worker.queue_source(source_id)

    # For folders, also create child sources upfront
    if kind == "folder":
        from voxscribe.knowledge import scan_folder
        try:
            files = scan_folder(Path(uri))
            for file_path in files:
                child_id = str(uuid.uuid4())
                await store.create_source(
                    source_id=child_id,
                    kind="file",
                    uri=str(file_path),
                    name=file_path.name,
                    mime=None,
                    parent_id=source_id,
                )
                if body.labelIds:
                    await store.update_source_labels(child_id, body.labelIds)

                # Publish child event
                child_source = await store.get_source(child_id)
                events.publish("source.update", child_source.model_dump(exclude_none=True))

                # Queue child for ingestion
                if hasattr(request.app.state, "ingestion_worker"):
                    await request.app.state.ingestion_worker.queue_source(child_id)
        except Exception as e:
            # Log but don't fail - child creation is nice-to-have
            log.warning(f"Failed to scan folder children: {e}")

    return source.model_dump(exclude_none=True)


@router.get("")
async def list_sources(
    request: Request,
    query: str | None = Query(None),
    labelId: str | None = Query(None),
    kind: str | None = Query(None),
    sort: str = Query("imported", enum=["imported", "name"]),
    limit: int = Query(25, ge=1, le=100),
    cursor: str | None = Query(None),
):
    """List sources with filtering and pagination."""
    store = SourceStore(request.app.state.store.pool)
    sources, next_cursor, total, counts = await store.list_sources(
        query=query,
        label_id=labelId,
        kind=kind,
        sort=sort,
        limit=limit,
        cursor=cursor,
    )

    return {
        "items": [s.model_dump(exclude_none=True) for s in sources],
        "nextCursor": next_cursor,
        "total": total,
        "countsByKind": counts,
    }


@router.get("/graph")
async def get_sources_graph(request: Request):
    """Get knowledge graph of sources, labels, and meetings."""
    store = SourceStore(request.app.state.store.pool)
    main_store = request.app.state.store

    nodes = []
    edges = []

    # Add source nodes
    async with main_store.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, name FROM sources WHERE parent_id IS NULL"
        )
        sources = await cur.fetchall()
        for source_id, name in sources:
            nodes.append({"id": source_id, "type": "source", "name": name})

    # Add label nodes
    labels = await store.list_labels()
    for label in labels:
        nodes.append({"id": label.id, "type": "label", "name": label.name})

    # Add label edges (source -> label)
    async with main_store.pool.connection() as conn:
        cur = await conn.execute(
            """SELECT source_id, label_id FROM source_labels"""
        )
        label_edges = await cur.fetchall()
        for source_id, label_id in label_edges:
            edges.append({"from": source_id, "to": label_id, "kind": "labelled"})

    # Add attachment edges (meeting -> source)
    async with main_store.pool.connection() as conn:
        cur = await conn.execute(
            """SELECT meeting_id, source_id FROM meeting_attachments"""
        )
        attachments = await cur.fetchall()
        for meeting_id, source_id in attachments:
            # Add meeting node if not already there
            if not any(n["id"] == meeting_id for n in nodes):
                nodes.append({"id": meeting_id, "type": "meeting", "name": meeting_id})
            edges.append({"from": meeting_id, "to": source_id, "kind": "attached"})

    # Add child edges (folder -> file)
    async with main_store.pool.connection() as conn:
        cur = await conn.execute(
            """SELECT parent_id, id, name FROM sources WHERE parent_id IS NOT NULL"""
        )
        children = await cur.fetchall()
        for parent_id, child_id, name in children:
            nodes.append({"id": child_id, "type": "source", "name": name})
            edges.append({"from": parent_id, "to": child_id, "kind": "child"})

    return {
        "nodes": nodes,
        "edges": [{"from": e["from"], "to": e["to"], "kind": e["kind"]} for e in edges],
    }


# Labels are handled in a separate labels router at /v1/labels


@router.get("/{source_id}")
async def get_source(request: Request, source_id: str):
    """Get a source by ID."""
    store = SourceStore(request.app.state.store.pool)
    source = await store.get_source(source_id)
    if not source:
        raise NotFoundError(f"source {source_id} not found")
    return source.model_dump(exclude_none=True)


@router.patch("/{source_id}")
async def update_source(request: Request, source_id: str, body: UpdateSourceRequest):
    """Update source labels."""
    store = SourceStore(request.app.state.store.pool)
    source = await store.get_source(source_id)
    if not source:
        raise NotFoundError(f"source {source_id} not found")

    await store.update_source_labels(source_id, body.labelIds)
    source = await store.get_source(source_id)

    events.publish("source.update", source.model_dump(exclude_none=True))
    return source.model_dump(exclude_none=True)


@router.delete("/{source_id}", status_code=204)
async def delete_source(request: Request, source_id: str):
    """Delete a source (cascade deletes chunks and vectors)."""
    store = SourceStore(request.app.state.store.pool)
    source = await store.get_source(source_id)
    if not source:
        raise NotFoundError(f"source {source_id} not found")

    await store.delete_source(source_id)
    return Response(status_code=204)


@router.post("/{source_id}/reindex", status_code=202)
async def reindex_source(request: Request, source_id: str):
    """Queue a source for re-indexing."""
    store = SourceStore(request.app.state.store.pool)
    source = await store.get_source(source_id)
    if not source:
        raise NotFoundError(f"source {source_id} not found")

    await store.update_source_status(source_id, "queued", error=None)
    source = await store.get_source(source_id)

    events.publish("source.update", source.model_dump(exclude_none=True))
    return Response(status_code=202)


@router.get("/{source_id}/chunks")
async def get_source_chunks(
    request: Request,
    source_id: str,
    limit: int = Query(25, ge=1, le=100),
    cursor: str | None = Query(None),
):
    """Get chunks for a source."""
    store = SourceStore(request.app.state.store.pool)
    source = await store.get_source(source_id)
    if not source:
        raise NotFoundError(f"source {source_id} not found")

    chunks, next_cursor = await store.get_chunks(source_id, limit=limit, cursor=cursor)
    return {
        "items": chunks,
        "nextCursor": next_cursor,
    }
