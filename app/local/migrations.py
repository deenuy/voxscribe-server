"""Database migration system.

Every numbered file in migrations/ is applied once per schema, in ascending order, each in its own transaction, and
recorded in that schema's schema_migrations table. The runner works from the set of recorded versions, not the
highest one: a file whose number is not recorded is applied even when a higher number already is (a migration that
arrived late from another branch), and that is logged as a warning naming both versions.

Two engines, or two requests, migrating one schema at once exclude each other with the same PostgreSQL advisory lock
the runner has always taken (key 123), so an engine on older code and one on this code also exclude each other. The
lock is held from reading the recorded versions until the last file is recorded, so each file runs exactly once.
"""

import logging
import sys
from pathlib import Path

import psycopg
from psycopg import sql

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_KEY = 123  # unchanged from earlier releases: older engines take the same lock


def migration_files(migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """(version, path) for every numbered .sql file, in ascending order of version."""
    found = []
    for path in migrations_dir.glob("*.sql"):
        try:
            found.append((int(path.stem.split("_")[0]), path))
        except ValueError:
            continue
    return sorted(found)


async def run_migrations(database_url: str, schema: str = None, *, migrations_dir: Path = MIGRATIONS_DIR) -> int:
    """Apply every migration not yet recorded in the schema; return the highest recorded version.

    schema: the schema to migrate (tenant schemas, tests); defaults to the connection's own search path (public).
    """
    files = migration_files(migrations_dir)
    if not files:
        return 0

    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        if schema:
            await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        # Session-level lock, held for the whole run; released explicitly, and by the server if the connection drops.
        await conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            await conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            cur = await conn.execute("SELECT version FROM schema_migrations")
            applied = {row[0] for row in await cur.fetchall()}
            highest = max(applied, default=0)

            for version, path in files:
                if version in applied:
                    continue
                if version < highest:
                    log.warning("migration %03d applied out of order: %03d was applied before it", version, highest)
                try:
                    async with conn.transaction():
                        await conn.execute(path.read_text())
                        await conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                except psycopg.DatabaseError as e:
                    print(f"Error running migration {version}: {e}", file=sys.stderr, flush=True)
                    raise
                applied.add(version)
                highest = max(highest, version)
        finally:
            if not conn.closed:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))

    return highest
