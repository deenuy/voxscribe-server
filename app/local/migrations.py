"""Database migration system."""

import sys
from pathlib import Path

import psycopg


async def run_migrations(database_url: str, schema: str = None) -> int:
    """Run pending migrations and return the final schema version.

    schema: optional schema name (for tests); defaults to public
    """
    migrations_dir = Path(__file__).parent / "migrations"

    # Get all migration files
    migration_files = sorted(migrations_dir.glob("*.sql"))
    if not migration_files:
        return 0

    # Connect and run migrations
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        # Set search_path if schema is specified (for test isolation)
        if schema:
            await conn.execute(f"SET search_path TO {schema}, public")
            await conn.commit()

        # Ensure the schema_migrations table exists (inside transaction with lock)
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_lock(123)")

            try:
                await conn.execute(
                    """CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
            except psycopg.ProgrammingError as e:
                # If table exists, that's OK
                if "already exists" not in str(e):
                    raise

            # Get the current schema version (now that table definitely exists)
            cursor = await conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            current_version = row[0] if row else 0

            await conn.execute("SELECT pg_advisory_unlock(123)")

        # Apply pending migrations
        final_version = current_version
        for mig_file in migration_files:
            # Extract version number from filename
            version_str = mig_file.stem.split("_")[0]
            try:
                version = int(version_str)
            except ValueError:
                continue

            if version <= current_version:
                continue

            # Read and execute the migration
            migration_sql = mig_file.read_text()
            try:
                async with conn.transaction():
                    # Use advisory lock to prevent concurrent migrations
                    await conn.execute("SELECT pg_advisory_lock(123)")

                    # Check again after acquiring lock
                    cursor = await conn.execute(
                        "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
                    )
                    row = await cursor.fetchone()
                    latest_version = row[0] if row else 0

                    if version > latest_version:
                        # Execute migration
                        await conn.execute(migration_sql)
                        # Record it
                        await conn.execute(
                            "INSERT INTO schema_migrations (version) VALUES (%s)",
                            (version,),
                        )

                    await conn.execute("SELECT pg_advisory_unlock(123)")

                final_version = version
            except psycopg.DatabaseError as e:
                print(f"Error running migration {version}: {e}", file=sys.stderr, flush=True)
                raise

    return final_version
