"""Data endpoints: tenant data erase and export.

DELETE /v1/data   -- erase all of this tenant's data
GET    /v1/data/export -- stream an archive of this tenant's data

Both routes require the X-Confirm: delete-all header (erase) or X-Confirm: export (export).
The contract (engine-v1.md) specifies the erase header as 'delete-all'.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import io
import json
import logging
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg import sql as psql

from app.local import events, finalize, meeting_bus, tenancy
from app.local.errors import InvalidRequestError

router = APIRouter(prefix="/v1", tags=["data"])
log = logging.getLogger(__name__)

# Tables that belong to the migration bookkeeping and must never be touched by erase or export.
_META_TABLES = frozenset({"schema_migrations", "meta"})


async def _tenant_tables(conn) -> list[str]:
    """Every base table in the current schema, excluding migration bookkeeping.

    Uses information_schema so a table added by a future migration is automatically included.
    The result is ordered so that parent rows come before child rows (by FK depth), which
    matters for TRUNCATE without CASCADE or for a manual DELETE loop.
    """
    cur = await conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_type = 'BASE TABLE'
          AND table_name NOT IN ('schema_migrations', 'meta')
        ORDER BY table_name
        """,
    )
    return [row[0] for row in await cur.fetchall()]


# The append-only gate ledger (gate_records, migration 021) refuses DELETE and TRUNCATE unless this setting is 'on'
# in the current transaction. Erase is the only code that sets it, and only with SET LOCAL inside the one
# transaction that empties the tenant's tables, so it ends with that transaction.
_ERASE_SETTING = "voxscribe.erase_tenant_data"


async def _truncate_tenant_tables(conn) -> int:
    """Empty every tenant-owned table in the current schema in one transaction, the gate ledger included."""
    async with conn.transaction():
        await conn.execute(f"SET LOCAL {_ERASE_SETTING} = 'on'")
        tables = await _tenant_tables(conn)
        if tables:
            # One TRUNCATE ... CASCADE for all tables. psycopg cannot bind a list of identifiers to one placeholder,
            # so the statement is composed from quoted names read from information_schema.
            await conn.execute(psql.SQL("TRUNCATE TABLE {} CASCADE").format(
                psql.SQL(", ").join(psql.Identifier(t) for t in tables)))
    await conn.commit()
    log.info("erase: truncated %d tables (shared schema)", len(tables))
    return len(tables)


# Where a person's files live under the data directory: data_dir/tenants/<key>/ with isolation on, and
# data_dir/tenants/shared/ for the single shared scope with isolation off. Erase and export touch this directory and
# nothing else in data_dir, which may hold files the engine does not own (the desktop app picks the directory).
# Today no engine code writes here; the directory is the one place such files go when it does.
_TENANT_FILES = "tenants"
_SHARED_FILES = "shared"            # never a tenant key: those are t_<24 hex>

_SETTLE_S = 10                      # how long erase waits for a tenant's cancelled work to wind down
_EXPORT_PAGE_ROWS = 500             # rows per fetch from the server-side cursor
_EXPORT_SPOOL_BYTES = 8 * 1024 * 1024   # a table's rows past this go to a temporary file, not memory


def tenant_files_dir(data_dir: Path, tenant_key: str | None) -> Path:
    """The directory that holds a tenant's files (see _TENANT_FILES)."""
    return data_dir / _TENANT_FILES / (tenant_key or _SHARED_FILES)


def _voice_session_running(voice_host) -> bool:
    from app.local.voice.host import WakeState

    return voice_host.state == WakeState.LISTENING or bool(voice_host.snapshot().get("meeting_id"))


def _release_voice_host(voice_host, tenant_key: str | None) -> bool:
    """Take the device off the erasing tenant. False (refuse the erase) when that tenant's session is running.

    The host is touched only when it works for this tenant: another tenant's binding, armed device and running
    session are theirs. Unbinding replaces only who the device is for (tenant, organization, role); product and
    device stay, so a later bind with a device claim still matches."""
    from app.local.voice.host import SessionInProgressError

    binding = voice_host.binding
    if binding.tenant != tenant_key:
        return True
    if tenant_key is None:                  # isolation off: one scope, nothing to unbind
        return not _voice_session_running(voice_host)
    try:
        voice_host.bind(dataclasses.replace(binding, tenant=None, organization=None, role=None))
    except SessionInProgressError:          # raised only while a session runs, which is this tenant's
        return False
    return True


async def _settle_voice_work(voice_host, tenant_key: str | None) -> None:
    """The host's forget step for this tenant (the delegate's open tasks are cancelled, waiting results dropped,
    inside the tenant's scope as the host does it), then wait for the cancelled tasks and their row writes."""
    delegate = getattr(getattr(voice_host, "ports", None), "delegate", None)
    forget = getattr(delegate, "forget", None)
    if forget is None:
        return
    lane = getattr(delegate, "lane", None)
    open_ids = [t.task_id for t in [*lane.active(), *lane.get_pending()]] if lane is not None else []
    with tenancy.tenant_scope(tenant_key) if tenant_key else contextlib.nullcontext():
        forget()

    async def wind_down():
        for task_id in open_ids:
            await lane.wait(task_id)
        drain = getattr(delegate, "drain", None)
        if drain is not None:
            await drain()

    try:
        await asyncio.wait_for(wind_down(), _SETTLE_S)
    except asyncio.TimeoutError:
        log.warning("erase: the delegate's cancelled work did not finish within %d s", _SETTLE_S)


async def _forget_in_memory(app, tenant_key: str | None) -> None:
    """Drop what the engine holds in memory for this tenant: running finalisations, the event stream and its replay
    buffer (which holds transcript and reply text), the meeting buses and their held events, the profile cache."""
    cancelled = await finalize.cancel_for_tenant(tenant_key)
    events.drop_stream(tenant_key)
    buses = meeting_bus.drop_tenant(tenant_key)
    profile_store = getattr(app.state, "profile_store", None)
    if profile_store is not None:
        profile_store.reset(tenant=tenant_key)
    log.info("erase: cancelled %d finalisations, dropped %d meeting buses", cancelled, buses)


@router.delete("/data")
async def delete_all_data(
    request: Request,
    x_confirm: str | None = Header(None),
    models: str | None = None,
) -> JSONResponse:
    """Erase all data for this tenant.

    Refused with 409 session_in_progress while this tenant's voice session runs (another tenant's session does not
    block it). Then, in this order: the voice host is taken off this tenant if it works for them, and the delegate's
    work for them is cancelled and waited out; their running finalisations are cancelled; their event stream, replay
    buffer, meeting buses and profile cache entry are dropped; then their rows go and their files.

    With tenant isolation on the tenant's schema is dropped, under the provisioner's lock and after the tenant
    leaves its ready set, so the next request starts from a new, empty schema. With isolation off every
    tenant-owned table is emptied in one transaction (the gate ledger through its erase door); the shared schema
    and schema_migrations stay.

    Files: only the tenant's directory under the data directory (tenant_files_dir). Downloaded models only with
    ?models=true.

    Requires header: X-Confirm: delete-all
    Returns 204 on success. The contract specifies no body on 204.
    """
    if x_confirm != "delete-all":
        raise InvalidRequestError("X-Confirm: delete-all header required")

    app = request.app
    store = app.state.store
    settings = app.state.settings
    data_dir: Path = app.state.data_dir
    pool = store.pool
    tenant_key = tenancy.current_tenant()  # None when isolation is off

    voice_host = getattr(app.state, "voice_host", None)
    if voice_host is not None:
        served = voice_host.binding.tenant == tenant_key
        if not _release_voice_host(voice_host, tenant_key):
            return JSONResponse(
                status_code=409,
                content={
                    "error": "session_in_progress",
                    "detail": "a voice session is running; stop it before erasing",
                },
            )
        if served:
            await _settle_voice_work(voice_host, tenant_key)

    await _forget_in_memory(app, tenant_key)

    if pool.enforced and tenant_key:
        await pool.provisioner.drop(tenant_key)
        log.info("erase: dropped schema tenant=%s", tenant_key)
    else:
        async with pool.connection() as conn:
            await _truncate_tenant_tables(conn)

    tenant_dir = tenant_files_dir(data_dir, tenant_key)
    if tenant_dir.exists():
        try:
            await asyncio.to_thread(shutil.rmtree, tenant_dir)
            log.info("erase: removed the tenant's files")
        except Exception:
            log.warning("erase: could not remove the tenant's files", exc_info=True)

    # Delete Ollama models if requested (independent of tenant data).
    if models == "true" and settings.providers == "local":
        import httpx

        ollama_host = settings.ollama_host
        model_ids = ["gemma4:12b", "gemma4:e4b", "gemma4:e2b"]
        for model_id in model_ids:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.delete(f"{ollama_host}/api/delete", json={"name": model_id})
            except Exception:
                pass

    return JSONResponse(status_code=204, content=None)


class _ArchiveWriter:
    """Builds the export archive. Every method runs in a worker thread (asyncio.to_thread), one call at a time, so
    compression never runs on the event loop. take() hands back the compressed bytes written so far."""

    def __init__(self) -> None:
        self._out = io.BytesIO()
        self._tar = tarfile.open(fileobj=self._out, mode="w:gz")
        self._spool = None

    def take(self) -> bytes:
        chunk = self._out.getvalue()
        self._out.seek(0)
        self._out.truncate(0)
        return chunk

    def begin_table(self) -> None:
        self._spool = tempfile.SpooledTemporaryFile(max_size=_EXPORT_SPOOL_BYTES)

    def rows(self, lines: list[str]) -> None:
        self._spool.write(("\n".join(lines) + "\n").encode())

    def end_table(self, table: str) -> None:
        spool, self._spool = self._spool, None
        try:
            info = tarfile.TarInfo(name=f"tables/{table}.jsonl")
            info.size = spool.tell()
            info.mtime = int(time.time())
            spool.seek(0)
            self._tar.addfile(info, spool)
        finally:
            spool.close()

    def drop_table(self) -> None:
        if self._spool is not None:
            self._spool.close()
            self._spool = None

    def add_files(self, tenant_dir: Path) -> list[str]:
        added: list[str] = []
        if not tenant_dir.exists():
            return added
        for fpath in sorted(tenant_dir.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(tenant_dir)
                try:
                    self._tar.add(str(fpath), arcname=f"files/{rel}")
                    added.append(str(rel))
                except Exception:
                    log.warning("export: could not add a file", exc_info=True)
        return added

    def finish(self, manifest: dict) -> None:
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        self._tar.addfile(info, io.BytesIO(manifest_bytes))
        self._tar.close()


@router.get("/data/export")
async def export_data(
    request: Request,
    x_confirm: str | None = Header(None),
) -> StreamingResponse:
    """Export this tenant's data as a streaming gzip-compressed tar archive.

    Layout (stable across engine versions):
      manifest.json          -- {schema_version, tenant, exported_at, counts: {table: n}, files}
      tables/<table>.jsonl   -- one JSON object per row per line, all columns
      files/...              -- the tenant's files (tenant_files_dir), paths preserved

    Rows are read in pages from a server-side cursor; the archive is compressed in a worker thread, so a large
    tenant neither fills memory nor holds up the event loop the wake path runs on.

    Requires header: X-Confirm: export
    Returns 200 with Content-Type: application/gzip.
    """
    if x_confirm != "export":
        raise InvalidRequestError("X-Confirm: export header required")

    store = request.app.state.store
    data_dir: Path = request.app.state.data_dir
    pool = store.pool
    tenant_key = tenancy.current_tenant()

    async def _generate():
        writer = await asyncio.to_thread(_ArchiveWriter)
        counts: dict[str, int] = {}
        schema_version = 0
        try:
            async with pool.connection() as conn:
                try:
                    async with conn.transaction():
                        cur = await conn.execute("SELECT MAX(version) FROM schema_migrations")
                        row = await cur.fetchone()
                        schema_version = int(row[0]) if row and row[0] else 0
                except Exception:
                    schema_version = 0

                for table in await _tenant_tables(conn):
                    await asyncio.to_thread(writer.begin_table)
                    count = 0
                    try:
                        async with conn.transaction():   # a server-side cursor lives in a transaction
                            async with conn.cursor(name="export_rows") as cur:
                                await cur.execute(psql.SQL("SELECT row_to_json(t)::text FROM {} t").format(
                                    psql.Identifier(table)))
                                while page := await cur.fetchmany(_EXPORT_PAGE_ROWS):
                                    await asyncio.to_thread(writer.rows, [r[0] for r in page])
                                    count += len(page)
                    except Exception:
                        log.warning("export: failed reading table=%s", table, exc_info=True)
                        await asyncio.to_thread(writer.drop_table)
                        continue
                    await asyncio.to_thread(writer.end_table, table)
                    counts[table] = count
                    chunk = writer.take()
                    if chunk:
                        yield chunk

            files_added = await asyncio.to_thread(writer.add_files, tenant_files_dir(data_dir, tenant_key))
            await asyncio.to_thread(writer.finish, {
                "schema_version": schema_version,
                "tenant": tenant_key,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "counts": counts,
                "files": files_added,
            })
            yield writer.take()
        finally:
            await asyncio.to_thread(writer.drop_table)

    return StreamingResponse(
        _generate(),
        media_type="application/gzip",
        headers={"Content-Disposition": 'attachment; filename="voxscribe-export.tar.gz"'},
    )
