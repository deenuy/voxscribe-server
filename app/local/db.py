"""Shared database utilities for the engine.

Provides pool creation, transaction management, row helpers, and database availability detection.
"""

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool

from .errors import DatabaseUnavailableError


async def reset_connection(conn) -> None:
    """Runs whenever a connection goes back to the pool.

    Pooled connections are shared. Many store methods set `conn.row_factory = dict_row` on the
    connection they borrowed and never undo it, so the next borrower silently received dicts
    instead of tuples: the health probe's `row[0]` raised (reported as "database unavailable")
    and the ideas list zipped column names with dict keys. Resetting here makes that class of
    leak impossible, whatever a caller does.
    """
    conn.row_factory = tuple_row


async def create_pool(database_url: str, min_size: int = 2, max_size: int = 10, schema: str | None = None) -> AsyncConnectionPool:
    """Create and open a connection pool.

    Args:
        database_url: PostgreSQL connection URL
        min_size: Minimum pool size
        max_size: Maximum pool size
        schema: Optional schema name to set search_path (for tests)
    """
    async def configure_conn(conn):
        """Configure each connection with the test schema if provided."""
        if schema:
            await conn.execute(f"SET search_path TO {schema}, public")
            # The pool rejects connections left inside a transaction.
            await conn.commit()

    pool = AsyncConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        configure=configure_conn if schema else None,
        reset=reset_connection,
    )
    await pool.open()
    return pool


@asynccontextmanager
async def transaction(pool: AsyncConnectionPool) -> AsyncIterator[psycopg.AsyncConnection]:
    """Async context manager for database transactions.

    Usage:
        async with transaction(pool) as conn:
            await conn.execute("INSERT INTO ...")
            await conn.commit()
    """
    async with pool.connection() as conn:
        try:
            yield conn
        except Exception:
            await conn.rollback()
            raise


def rows_to_dicts(rows: list[tuple]) -> list[dict]:
    """Convert a list of rows (tuples) to dicts with column names.

    Note: This is a simple helper. For better results, use dict_row factory with conn.
    """
    return [dict(row) for row in rows]


def camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def redact_password(url: str) -> str:
    """Redact password from PostgreSQL connection URL."""
    # Match postgresql://user:password@host/db
    pattern = r"(postgresql://[^:]+:)[^@]+(@)"
    return re.sub(pattern, r"\1[redacted]\2", url)


async def check_database_available(database_url: str) -> bool:
    """Check if database is reachable and has the vector extension.

    Returns True if database is available and has vector extension, False otherwise.
    """
    try:
        async with await psycopg.AsyncConnection.connect(database_url) as conn:
            # Test connectivity
            await conn.execute("SELECT 1")

            # Check for vector extension
            cursor = await conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            result = await cursor.fetchone()
            return result is not None
    except Exception:
        return False


async def get_database_info(pool: AsyncConnectionPool) -> tuple[bool, str]:
    """Get database name and version.

    Returns (ok, db_name) tuple.
    """
    try:
        async with pool.connection() as conn:
            # Get database name
            cursor = await conn.execute("SELECT current_database()")
            db_row = await cursor.fetchone()
            db_name = db_row[0] if db_row else "unknown"
        return True, db_name
    except Exception as e:
        raise DatabaseUnavailableError(f"failed to get database info: {e}") from e
