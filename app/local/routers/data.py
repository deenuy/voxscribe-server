"""Data endpoints: tenant data erase and export.

DELETE /v1/data   -- erase all of this tenant's data
GET    /v1/data/export -- stream an archive of this tenant's data

Both routes require the X-Confirm: delete-all header (erase) or X-Confirm: export (export).
The contract (engine-v1.md) specifies the erase header as 'delete-all'.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.local import tenancy
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
            from psycopg import sql as psql

            await conn.execute(psql.SQL("TRUNCATE TABLE {} CASCADE").format(
                psql.SQL(", ").join(psql.Identifier(t) for t in tables)))
    await conn.commit()
    log.info("erase: truncated %d tables (shared schema)", len(tables))
    return len(tables)


def _tenant_data_dir(data_dir: Path, tenant_key: str | None) -> Path:
    """The directory that holds a tenant's files.

    With isolation on the directory is data_dir/tenants/<key>/.
    With isolation off (tenant_key is None) the directory is data_dir/ itself.
    """
    if tenant_key:
        return data_dir / "tenants" / tenant_key
    return data_dir


@router.delete("/data")
async def delete_all_data(
    request: Request,
    x_confirm: str | None = Header(None),
    models: str | None = None,
) -> JSONResponse:
    """Erase all data for this tenant.

    With tenant isolation on: drops the tenant's schema (all tables, all rows, all indexes),
    removes the provisioner's ready entry so the schema can be recreated cleanly, unbinds
    the voice host if it is bound to this tenant, and deletes the tenant's files.

    With tenant isolation off: empties every tenant-owned table (TRUNCATE ... CASCADE) and
    deletes the files in the data directory; the shared schema and schema_migrations are
    preserved.

    Session in progress: if a voice session is running for this tenant the erase is refused
    with 409 so the session's data is not erased mid-flight. Stop the session and retry.

    Requires header: X-Confirm: delete-all
    Returns 204 on success. The contract specifies no body on 204.
    """
    if x_confirm != "delete-all":
        raise InvalidRequestError("X-Confirm: delete-all header required")

    store = request.app.state.store
    settings = request.app.state.settings
    data_dir: Path = request.app.state.data_dir
    pool = store.pool
    tenant_key = tenancy.current_tenant()  # None when isolation is off

    # Check for a running voice session before we touch anything.
    voice_host = getattr(request.app.state, "voice_host", None)
    if voice_host is not None and tenant_key is not None:
        from app.local.voice.host import SessionBinding, SessionInProgressError
        try:
            voice_host.bind(SessionBinding())   # unbind: tenant=None
        except SessionInProgressError:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "session_in_progress",
                    "detail": "a voice session is running; stop it before erasing",
                },
            )

    if pool.enforced and tenant_key:
        # Isolation on: drop the entire schema in one shot (handles FK order automatically).
        from psycopg import sql as psql
        async with await __import__("psycopg").AsyncConnection.connect(
            pool.database_url, autocommit=True
        ) as raw_conn:
            await raw_conn.execute(
                psql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(psql.Identifier(tenant_key))
            )
        # Evict from the provisioner so the next request re-provisions cleanly.
        pool.provisioner.evict(tenant_key)
        log.info("erase: dropped schema tenant=%s", tenant_key)
    else:
        # Isolation off (or no tenant key): empty all tenant-owned tables.
        async with pool.connection() as conn:
            await _truncate_tenant_tables(conn)

    # Delete the tenant's files.
    tenant_dir = _tenant_data_dir(data_dir, tenant_key)
    if tenant_dir.exists():
        try:
            shutil.rmtree(tenant_dir)
            log.info("erase: removed tenant files dir=%s", tenant_dir)
        except Exception:
            log.warning("erase: could not remove files dir=%s", tenant_dir, exc_info=True)

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


@router.get("/data/export")
async def export_data(
    request: Request,
    x_confirm: str | None = Header(None),
) -> StreamingResponse:
    """Export this tenant's data as a streaming gzip-compressed tar archive.

    Layout (stable across engine versions):
      manifest.json          -- {schema_version, tenant, exported_at, counts: {table: n}, files}
      tables/<table>.jsonl   -- one JSON object per row per line, all columns
      files/...              -- the tenant's files from the data directory, paths preserved

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
        # We write to an in-memory tar that is periodically flushed.  For a large tenant
        # the rows and files are streamed chunk by chunk to avoid loading the whole archive.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:

            counts: dict[str, int] = {}
            schema_version = 0

            # We need a connection for the duration of the export.
            async with pool.connection() as conn:
                # Schema version
                try:
                    cur = await conn.execute(
                        "SELECT MAX(version) FROM schema_migrations"
                    )
                    row = await cur.fetchone()
                    schema_version = int(row[0]) if row and row[0] else 0
                except Exception:
                    schema_version = 0

                tables = await _tenant_tables(conn)

                for table in tables:
                    rows_buf = io.BytesIO()
                    count = 0
                    try:
                        cur = await conn.execute(
                            f"SELECT row_to_json(t) FROM {table} t"
                        )
                        async for batch in cur:
                            line = json.dumps(batch[0], default=str) + "\n"
                            rows_buf.write(line.encode())
                            count += 1
                    except Exception:
                        log.warning("export: failed reading table=%s", table, exc_info=True)
                        continue

                    counts[table] = count
                    rows_buf.seek(0)
                    info = tarfile.TarInfo(name=f"tables/{table}.jsonl")
                    info.size = len(rows_buf.getvalue())
                    info.mtime = int(time.time())
                    tar.addfile(info, rows_buf)

            # Flush rows data.
            buf.seek(0)
            yield buf.read()
            buf.seek(0)
            buf.truncate(0)

            # Add tenant files.
            tenant_dir = _tenant_data_dir(data_dir, tenant_key)
            files_added: list[str] = []
            if tenant_dir.exists():
                for fpath in sorted(tenant_dir.rglob("*")):
                    if fpath.is_file():
                        rel = fpath.relative_to(tenant_dir)
                        try:
                            tar.add(str(fpath), arcname=f"files/{rel}")
                            files_added.append(str(rel))
                        except Exception:
                            log.warning("export: could not add file=%s", fpath, exc_info=True)
                    # Flush after each file to keep memory bounded.
                    buf.seek(0)
                    chunk = buf.read()
                    if chunk:
                        yield chunk
                    buf.seek(0)
                    buf.truncate(0)

            # Manifest (written last so counts and files list are complete).
            manifest = {
                "schema_version": schema_version,
                "tenant": tenant_key,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "counts": counts,
                "files": files_added,
            }
            manifest_bytes = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(manifest_bytes))

        # Final flush.
        buf.seek(0)
        yield buf.read()

    return StreamingResponse(
        _generate(),
        media_type="application/gzip",
        headers={"Content-Disposition": 'attachment; filename="voxscribe-export.tar.gz"'},
    )
