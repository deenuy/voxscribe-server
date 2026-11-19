"""The migration runner applies every unrecorded file, in order, once (PL32 part D).

It used to skip every version at or below the highest recorded one, so a lower-numbered migration that arrived after a
higher one (another branch merged later) never ran, silently. It now works from the set of recorded versions.
"""

import asyncio
import logging
import uuid

import psycopg
import pytest

from app.local.migrations import migration_files, run_migrations
from tests.testdb import TEST_DATABASE_URL


@pytest.fixture
async def schema():
    name = f"t_mig_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {name}")
    yield name
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")


async def _query(schema: str, query: str) -> list[tuple]:
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        await conn.execute(f"SET search_path TO {schema}")
        return await (await conn.execute(query)).fetchall()


async def test_a_lower_numbered_migration_that_arrives_late_is_applied_with_a_warning(schema, tmp_path, caplog):
    (tmp_path / "001_a.sql").write_text("CREATE TABLE a (n int);")
    (tmp_path / "003_c.sql").write_text("CREATE TABLE c (n int);")
    assert await run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path) == 3

    (tmp_path / "002_b.sql").write_text("CREATE TABLE b (n int); INSERT INTO b VALUES (2);")
    with caplog.at_level(logging.WARNING, logger="app.local.migrations"):
        assert await run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path) == 3

    assert await _query(schema, "SELECT n FROM b") == [(2,)]
    assert await _query(schema, "SELECT version FROM schema_migrations ORDER BY version") == [(1,), (2,), (3,)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["migration 002 applied out of order: 003 was applied before it"]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.local.migrations"):
        assert await run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path) == 3
    assert caplog.records == []          # nothing left to apply, nothing to warn about


async def test_the_shipped_files_at_version_n_with_n_minus_1_unapplied_get_it(schema, tmp_path):
    """The real migrations, merged out of order: N lands before N-1 (two branches, the later number first)."""
    import shutil

    files = migration_files()
    (n, _), (n_minus_1, late) = files[-1], files[-2]
    for version, path in files:
        if version != n_minus_1:
            shutil.copy(path, tmp_path / path.name)
    assert await run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path) == n
    assert n_minus_1 not in {v for (v,) in await _query(schema, "SELECT version FROM schema_migrations")}

    shutil.copy(late, tmp_path / late.name)
    assert await run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path) == n
    recorded = {v for (v,) in await _query(schema, "SELECT version FROM schema_migrations")}
    assert recorded == {v for v, _ in files}


async def test_concurrent_migration_of_one_schema_applies_each_file_once(schema, tmp_path):
    (tmp_path / "001_runs.sql").write_text("CREATE TABLE runs (version int); INSERT INTO runs VALUES (1);")
    (tmp_path / "002_slow.sql").write_text("SELECT pg_sleep(0.2); INSERT INTO runs VALUES (2);")
    (tmp_path / "003_more.sql").write_text("INSERT INTO runs VALUES (3);")

    results = await asyncio.gather(*[
        run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path) for _ in range(6)])

    assert results == [3] * 6
    assert await _query(schema, "SELECT version FROM runs ORDER BY version") == [(1,), (2,), (3,)]
    assert await _query(schema, "SELECT version FROM schema_migrations ORDER BY version") == [(1,), (2,), (3,)]


async def test_a_failing_migration_records_nothing_and_releases_the_lock(schema, tmp_path):
    (tmp_path / "001_ok.sql").write_text("CREATE TABLE ok (n int);")
    (tmp_path / "002_bad.sql").write_text("CREATE TABLE half (n int); SELECT no_such_function();")
    with pytest.raises(psycopg.errors.UndefinedFunction):
        await run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path)
    assert await _query(schema, "SELECT version FROM schema_migrations") == [(1,)]
    assert await _query(schema, "SELECT to_regclass('half') IS NULL") == [(True,)]
    (tmp_path / "002_bad.sql").write_text("CREATE TABLE half (n int);")
    # The lock was released: a second run is not blocked behind the failed one.
    assert await asyncio.wait_for(
        run_migrations(TEST_DATABASE_URL, schema=schema, migrations_dir=tmp_path), 10) == 2
