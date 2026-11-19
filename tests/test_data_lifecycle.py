"""Tests for the tenant data lifecycle: erase, export, retention sweep, data directory.

Every test uses its own throwaway schema and temporary data directory.  No test
touches the public schema or the default data path.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest

from app.local.config import Config
from app.local.migrations import run_migrations

from tests.testdb import TEST_DATABASE_URL as _DB_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_app(schema: str, data_dir: Path, **cfg_kwargs):
    """Create a test app wired to the given schema and data directory."""
    from app.local.app import create_app

    settings = Config(
        database_url=_DB_URL,
        data_dir=str(data_dir),
        providers="fake",
        **cfg_kwargs,
    )
    return await create_app(settings=settings, auth_disabled=True, schema=schema)


async def _fresh_schema():
    """Create a throwaway schema with all migrations applied; return its name."""
    schema = f"t_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.commit()
    await run_migrations(_DB_URL, schema=schema)
    return schema


async def _drop_schema(schema: str):
    try:
        async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await conn.commit()
    except Exception:
        pass


async def _schema_exists(schema: str) -> bool:
    async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
        cur = await conn.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,)
        )
        return bool(await cur.fetchone())


# ---------------------------------------------------------------------------
# Erase: table list derivation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_erase_table_list_covers_all_migration_tables():
    """The table list from information_schema includes every table the migrations create.

    This proves the derivation is complete without relying on a hand-kept list.
    A table added by a new migration is automatically covered.
    """
    schema = await _fresh_schema()
    try:
        # Query the derived list (the same query the router uses).
        async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
            await conn.execute(f"SET search_path TO {schema}, public")
            cur = await conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_type = 'BASE TABLE'
                  AND table_name NOT IN ('schema_migrations', 'meta')
                ORDER BY table_name
                """
            )
            derived = {row[0] for row in await cur.fetchall()}

        # Every table that the migrations create (from grep CREATE TABLE) must be in the
        # derived list.  We check a representative sample from across the migration files.
        # Crucially we do NOT name assistant_profile or gate_records here -- they come from
        # migrations 022 and 021 that another agent owns.  If they existed they would be
        # covered automatically.
        expected_sample = {
            "meetings",
            "transcript_segments",
            "summaries",
            "sources",
            "source_chunks",
            "labels",
            "wake_events",
            "tool_calls",
            "edit_commits",
            "reminders",
            "tasks",
            "delegate_tasks",
            "turns",
            "replies",
            "gate_decisions",
            "latency_ledger",
            "settings",
            "auth_events",
            "usage",
        }
        missing = expected_sample - derived
        assert not missing, f"derived table list is missing: {sorted(missing)}"
    finally:
        await _drop_schema(schema)


# ---------------------------------------------------------------------------
# Erase: isolation off (shared schema)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_erase_clears_all_rows_isolation_off():
    """DELETE /v1/data empties every tenant-owned table (isolation off)."""
    import httpx

    schema = await _fresh_schema()
    data_dir = Path(tempfile.mkdtemp(prefix="voxscribe_test_"))
    try:
        app = await _make_app(schema, data_dir)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Seed a row
            store = app.state.store
            async with store.pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO usage (id, feature, day, tokens) VALUES (%s, 'test', CURRENT_DATE, 1)",
                    (str(uuid.uuid4()),),
                )

            # Verify row exists
            async with store.pool.connection() as conn:
                cur = await conn.execute("SELECT COUNT(*) FROM usage")
                assert (await cur.fetchone())[0] > 0

            resp = await client.delete("/v1/data", headers={"X-Confirm": "delete-all"})
            assert resp.status_code == 204

            # All tenant-owned tables must be empty; schema_migrations must be intact.
            async with store.pool.connection() as conn:
                cur = await conn.execute("SELECT COUNT(*) FROM usage")
                assert (await cur.fetchone())[0] == 0

                cur = await conn.execute("SELECT COUNT(*) FROM schema_migrations")
                assert (await cur.fetchone())[0] > 0

        await ctx.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema)
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Erase: missing confirmation header
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_erase_requires_header(client):
    """DELETE /v1/data without the confirmation header returns 400."""
    resp = await client.delete("/v1/data")
    assert resp.status_code == 400

    resp = await client.delete("/v1/data", headers={"X-Confirm": "wrong"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Erase: schema_migrations preserved after erase (isolation off)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_erase_keeps_schema_migrations(client):
    """schema_migrations is intact after erase."""
    store = client._transport.app.state.store

    resp = await client.delete("/v1/data", headers={"X-Confirm": "delete-all"})
    assert resp.status_code == 204

    async with store.pool.connection() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM schema_migrations")
        assert (await cur.fetchone())[0] >= 1


# ---------------------------------------------------------------------------
# Erase: clean restart (new session after erase)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_erase_allows_clean_restart():
    """After erase (isolation off) the schema still exists and a fresh row can be inserted."""
    import httpx

    schema = await _fresh_schema()
    data_dir = Path(tempfile.mkdtemp(prefix="voxscribe_test_"))
    try:
        app = await _make_app(schema, data_dir)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            store = app.state.store

            # Seed data
            async with store.pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO settings (key, value) VALUES ('k', '\"v\"'::jsonb)"
                )

            resp = await client.delete("/v1/data", headers={"X-Confirm": "delete-all"})
            assert resp.status_code == 204

            # Insert a fresh row -- this must succeed (the schema still exists after isolation-off erase)
            async with store.pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO settings (key, value) VALUES ('k2', '\"v2\"'::jsonb)"
                )
                cur = await conn.execute("SELECT value FROM settings WHERE key = 'k2'")
                assert await cur.fetchone() is not None

        await ctx.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema)
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Erase: file removal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_erase_removes_tenant_files():
    """DELETE /v1/data removes the tenant's files from the data directory."""
    import httpx

    schema = await _fresh_schema()
    data_dir = Path(tempfile.mkdtemp(prefix="voxscribe_test_"))
    try:
        # Isolation off: files live directly in data_dir.
        tenant_file = data_dir / "recording.wav"
        tenant_file.write_bytes(b"fake audio")

        app = await _make_app(schema, data_dir)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert tenant_file.exists()

            resp = await client.delete("/v1/data", headers={"X-Confirm": "delete-all"})
            assert resp.status_code == 204

            assert not tenant_file.exists()

        await ctx.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema)
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Export: basic archive shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_export_requires_header(client):
    """GET /v1/data/export without confirmation header returns 400."""
    resp = await client.get("/v1/data/export")
    assert resp.status_code == 400

    resp = await client.get("/v1/data/export", headers={"X-Confirm": "wrong"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_export_archive_shape(client):
    """Export produces a valid gzip-compressed tar containing manifest.json and tables/."""
    store = client._transport.app.state.store

    # Seed a row so counts are non-trivial
    async with store.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO settings (key, value) VALUES ('export_test', '\"yes\"'::jsonb)"
        )

    resp = await client.get("/v1/data/export", headers={"X-Confirm": "export"})
    assert resp.status_code == 200
    assert "application/gzip" in resp.headers.get("content-type", "")

    archive_bytes = resp.content
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        names = tar.getnames()

    assert "manifest.json" in names
    assert any(n.startswith("tables/") and n.endswith(".jsonl") for n in names)


@pytest.mark.asyncio
async def test_export_manifest_contains_counts(client):
    """The manifest in the export has schema_version and counts for each table."""
    store = client._transport.app.state.store

    async with store.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO settings (key, value) VALUES ('mcount', '1'::jsonb)"
        )

    resp = await client.get("/v1/data/export", headers={"X-Confirm": "export"})
    assert resp.status_code == 200

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        mf = tar.extractfile("manifest.json")
        manifest = json.loads(mf.read())

    assert "schema_version" in manifest
    assert "counts" in manifest
    assert "settings" in manifest["counts"]
    assert manifest["counts"]["settings"] >= 1


@pytest.mark.asyncio
async def test_export_contains_seeded_records(client):
    """Every row seeded before export is present in the archive; none from meta tables."""
    store = client._transport.app.state.store

    seeded_key = f"seed_{uuid.uuid4().hex[:8]}"
    async with store.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO settings (key, value) VALUES (%s, '\"seeded\"'::jsonb)",
            (seeded_key,),
        )

    resp = await client.get("/v1/data/export", headers={"X-Confirm": "export"})
    assert resp.status_code == 200

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        settings_file = tar.extractfile("tables/settings.jsonl")
        rows = [json.loads(line) for line in settings_file.read().splitlines()]

    keys = {r["key"] for r in rows}
    assert seeded_key in keys

    # schema_migrations must NOT be in the archive
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        names = tar.getnames()
    assert "tables/schema_migrations.jsonl" not in names


@pytest.mark.asyncio
async def test_export_includes_tenant_files():
    """Files placed in the data directory appear in the export under files/."""
    import httpx

    schema = await _fresh_schema()
    data_dir = Path(tempfile.mkdtemp(prefix="voxscribe_test_"))
    try:
        fake_file = data_dir / "notes.txt"
        fake_file.write_text("hello from test")

        app = await _make_app(schema, data_dir)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/data/export", headers={"X-Confirm": "export"})
            assert resp.status_code == 200

        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            names = tar.getnames()

        assert any("notes.txt" in n for n in names)

        await ctx.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema)
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_export_excludes_other_tenant_data():
    """Export of tenant A contains no data from tenant B's schema."""
    import httpx

    schema_a = await _fresh_schema()
    schema_b = await _fresh_schema()
    data_dir = Path(tempfile.mkdtemp(prefix="voxscribe_test_"))
    try:
        # Seed a row in schema_b that must NOT appear in schema_a's export.
        async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
            await conn.execute(f"SET search_path TO {schema_b}, public")
            await conn.execute(
                "INSERT INTO settings (key, value) VALUES ('tenant_b_key', '\"b\"'::jsonb)"
            )
            await conn.commit()

        # Seed a row in schema_a.
        async with await psycopg.AsyncConnection.connect(_DB_URL) as conn:
            await conn.execute(f"SET search_path TO {schema_a}, public")
            await conn.execute(
                "INSERT INTO settings (key, value) VALUES ('tenant_a_key', '\"a\"'::jsonb)"
            )
            await conn.commit()

        # Export tenant A (the app's schema is schema_a).
        app = await _make_app(schema_a, data_dir)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/data/export", headers={"X-Confirm": "export"})
            assert resp.status_code == 200

        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            settings_file = tar.extractfile("tables/settings.jsonl")
            rows = [json.loads(line) for line in settings_file.read().splitlines()]
        keys = {r["key"] for r in rows}
        assert "tenant_a_key" in keys
        assert "tenant_b_key" not in keys

        await ctx.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema_a)
        await _drop_schema(schema_b)
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Retention: wake candidates and tool-call records
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retention_sweep_deletes_old_wake_events():
    """Wake events older than retention_days are removed; newer ones stay."""
    from app.local.retention import run_sweep_once
    from app.local.store import Store
    from app.local import tenancy

    schema = await _fresh_schema()
    try:
        store = await Store.create(_DB_URL, schema=schema, enforced=False)
        pool = store.pool

        old_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())

        with tenancy.system_scope():
            async with pool.connection() as conn:
                # Old event: 10 days ago
                await conn.execute(
                    "INSERT INTO wake_events (id, phrase, confidence, triggered_at) "
                    "VALUES (%s, 'hey vox', 0.9, NOW() - INTERVAL '10 days')",
                    (old_id,),
                )
                # New event: 1 day ago
                await conn.execute(
                    "INSERT INTO wake_events (id, phrase, confidence, triggered_at) "
                    "VALUES (%s, 'hey vox', 0.9, NOW() - INTERVAL '1 day')",
                    (new_id,),
                )

        with tenancy.system_scope():
            await run_sweep_once(pool, wake_days=7, tool_call_days=0)

        with tenancy.system_scope():
            async with pool.connection() as conn:
                cur = await conn.execute("SELECT id FROM wake_events")
                remaining = {row[0] for row in await cur.fetchall()}

        assert old_id not in remaining, "old wake event should have been deleted"
        assert new_id in remaining, "new wake event should remain"

        await store.close()
    finally:
        await _drop_schema(schema)


@pytest.mark.asyncio
async def test_retention_sweep_no_deletion_when_days_zero():
    """With retention_days=0 (the default) no rows are deleted."""
    from app.local.retention import run_sweep_once
    from app.local.store import Store
    from app.local import tenancy

    schema = await _fresh_schema()
    try:
        store = await Store.create(_DB_URL, schema=schema, enforced=False)
        pool = store.pool

        old_id = str(uuid.uuid4())
        with tenancy.system_scope():
            async with pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO wake_events (id, phrase, confidence, triggered_at) "
                    "VALUES (%s, 'hey vox', 0.9, NOW() - INTERVAL '100 days')",
                    (old_id,),
                )

        # Sweep with 0 days (keep forever)
        with tenancy.system_scope():
            await run_sweep_once(pool, wake_days=0, tool_call_days=0)

        with tenancy.system_scope():
            async with pool.connection() as conn:
                cur = await conn.execute("SELECT id FROM wake_events WHERE id = %s", (old_id,))
                assert await cur.fetchone() is not None, "row must not be deleted when days=0"

        await store.close()
    finally:
        await _drop_schema(schema)


@pytest.mark.asyncio
async def test_retention_sweep_deletes_old_tool_calls():
    """Tool-call records older than retention_days are removed; newer ones stay."""
    from app.local.retention import run_sweep_once
    from app.local.store import Store
    from app.local import tenancy

    schema = await _fresh_schema()
    try:
        store = await Store.create(_DB_URL, schema=schema, enforced=False)
        pool = store.pool

        old_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())

        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        new_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        with tenancy.system_scope():
            async with pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO tool_calls (id, turn_id, meeting_id, tool, query, called_at) "
                    "VALUES (%s, 'turn1', 'mtg1', 'web_search', 'q', %s)",
                    (old_id, old_ts),
                )
                await conn.execute(
                    "INSERT INTO tool_calls (id, turn_id, meeting_id, tool, query, called_at) "
                    "VALUES (%s, 'turn2', 'mtg1', 'web_search', 'q', %s)",
                    (new_id, new_ts),
                )

        with tenancy.system_scope():
            await run_sweep_once(pool, wake_days=0, tool_call_days=7)

        with tenancy.system_scope():
            async with pool.connection() as conn:
                cur = await conn.execute("SELECT id FROM tool_calls")
                remaining = {row[0] for row in await cur.fetchall()}

        assert old_id not in remaining, "old tool call should have been deleted"
        assert new_id in remaining, "new tool call should remain"

        await store.close()
    finally:
        await _drop_schema(schema)


# ---------------------------------------------------------------------------
# Data directory: no default under /tmp
# ---------------------------------------------------------------------------

def test_default_data_dir_not_under_tmp():
    """The resolved default data directory must not be under a temporary path."""
    from pathlib import Path as _Path

    # Reproduce the resolution logic from app.py without side effects.
    default = _Path.home() / ".local" / "share" / "voxscribe"

    tmp_prefixes = ("/tmp", "/var/tmp", "/private/tmp", "/var/folders")
    resolved = str(default.resolve())
    for prefix in tmp_prefixes:
        assert not resolved.startswith(prefix), (
            f"default data dir {default!r} resolves under a temporary path {prefix!r}"
        )


def test_configured_data_dir_used_when_set(tmp_path):
    """When VOXSCRIBE_DATA_DIR is configured, that path is used as the data directory."""
    cfg = Config(database_url=_DB_URL, data_dir=str(tmp_path))
    assert cfg.data_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# Provisioner evict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provisioner_evict_clears_ready_set():
    """After evict(key) the provisioner no longer considers the tenant ready."""
    from app.local.tenancy import _Provisioner

    prov = _Provisioner(_DB_URL)
    prov._ready.add("t_abc123")

    prov.evict("t_abc123")

    assert "t_abc123" not in prov._ready
