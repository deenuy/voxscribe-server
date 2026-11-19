"""The one definition of the database the test suite uses, and the check that it is not anyone's live data.

Tests run against VOXSCRIBE_TEST_DATABASE_URL, by default a dedicated local database (`ideascribe_test`). Create it
once with `make test-db`. Inside it every test still works in a throwaway schema, so tests do not see each other.

The guard (`unsafe_reason`) runs once per pytest session from conftest. It refuses the run when the test database is
the database the engine itself would use (the default `Config` database, or VOXSCRIBE_DATABASE_URL in a `.env` at the
repository root), or when the test database's public schema holds rows, which means something other than the suite
writes there. Messages name a database by its name, host and port only, never by its URL, so no credential is printed.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict

ENV_VAR = "VOXSCRIBE_TEST_DATABASE_URL"
DEFAULT_TEST_DATABASE_URL = "postgresql://localhost:5432/ideascribe_test"
TEST_DATABASE_URL = os.environ.get(ENV_VAR, "").strip() or DEFAULT_TEST_DATABASE_URL

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def schema_url(schema: str, base: str = TEST_DATABASE_URL) -> str:
    """`base` with its search path set to `schema` first, for an engine process started by a test."""
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}options=-c%20search_path%3D{schema},public"


def describe(url: str) -> str:
    """A database named by name, host and port: safe to print, never carries a user or password."""
    parts = _parts(url)
    return f"database {parts['dbname']!r} on {parts['host'] or 'localhost'}:{parts['port']}"


def _parts(url: str) -> dict:
    raw = conninfo_to_dict(url)
    host = str(raw.get("host") or "")
    return {
        "dbname": str(raw.get("dbname") or raw.get("user") or os.environ.get("USER", "")),
        "host": "" if host in _LOCAL_HOSTS or host.startswith("/") else host,
        "port": str(raw.get("port") or "5432"),
    }


def same_database(a: str, b: str) -> bool:
    """True when two URLs name one database on one server (localhost, loopback and a local socket count as one)."""
    pa, pb = _parts(a), _parts(b)
    return pa == pb


def engine_database_urls(repo_root: Path = REPO_ROOT) -> list[tuple[str, str]]:
    """Every database the engine's own configuration resolves to: (where it came from, url)."""
    from app.local.config import Config, _parse_env_file

    found = [("the default Config database", Config().database_url)]
    for env_file in _env_files(repo_root):
        url = _parse_env_file(str(env_file)).get("VOXSCRIBE_DATABASE_URL", "").strip()
        if url:
            found.append((f"VOXSCRIBE_DATABASE_URL in {env_file}", url))
    return found


def _env_files(repo_root: Path) -> list[Path]:
    """The .env at the repository root and, when this checkout is a linked git worktree, the main checkout's .env."""
    files = [repo_root / ".env"]
    git = repo_root / ".git"
    if git.is_file():  # a linked worktree: ".git" holds "gitdir: <main>/.git/worktrees/<name>"
        text = git.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            if gitdir.parent.name == "worktrees":
                files.append(gitdir.parent.parent.parent / ".env")
    return [f for f in files if f.is_file()]


def public_tables_with_rows(conn) -> list[str]:
    """Tables in the public schema that hold at least one row (the vector extension adds none)."""
    names = [r[0] for r in conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    ).fetchall()]
    return [n for n in names if conn.execute(f'SELECT EXISTS (SELECT 1 FROM public."{n}")').fetchone()[0]]


def unsafe_reason(test_url: str = TEST_DATABASE_URL, repo_root: Path = REPO_ROOT, connect=psycopg.connect) -> str | None:
    """Why the suite must not run against `test_url`, or None when it may. A database that cannot be reached is not
    refused here: the tests that need it fail on their own, and the tests that do not still run."""
    for source, url in engine_database_urls(repo_root):
        if same_database(test_url, url):
            return (
                f"the test database is {describe(test_url)}, which is the engine's own database ({source}). "
                f"Tests would write next to live data. Point {ENV_VAR} at a dedicated database "
                f"(default {describe(DEFAULT_TEST_DATABASE_URL)}) and create it with `make test-db`."
            )
    try:
        with connect(test_url, autocommit=True, connect_timeout=3) as conn:
            busy = public_tables_with_rows(conn)
    except psycopg.OperationalError:
        return None
    if busy:
        shown = ", ".join(busy[:5]) + (" and more" if len(busy) > 5 else "")
        return (
            f"the public schema of the test {describe(test_url)} holds rows (tables: {shown}). "
            "Tests only ever write to throwaway schemas, so these rows belong to someone else, or a test escaped its "
            "schema. Nothing was changed. If this database is yours to discard, recreate it with "
            "`make test-db-reset CONFIRM=yes`; otherwise point "
            f"{ENV_VAR} at a dedicated database."
        )
    return None


def _maintenance_url(url: str) -> str:
    """The same server's `postgres` database, from which another database can be created or dropped."""
    from psycopg.conninfo import make_conninfo

    return make_conninfo(url, dbname="postgres")


def create(test_url: str = TEST_DATABASE_URL) -> str:
    """Create the test database if it is missing and install, in its public schema, the extensions the migrations
    need. Safe to run again. Refuses a URL that names the engine's own database."""
    for source, url in engine_database_urls():
        if same_database(test_url, url):
            raise SystemExit(f"refusing: {describe(test_url)} is the engine's own database ({source})")
    dbname = _parts(test_url)["dbname"]
    with psycopg.connect(_maintenance_url(test_url), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')
    with psycopg.connect(test_url, autocommit=True) as conn:
        # 001_init.sql creates this with IF NOT EXISTS; installed here first, it lives in public and is shared by
        # every throwaway schema instead of being created in (and dropped with) the first one.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    return f"{'created' if not exists else 'kept'} {describe(test_url)}, extension vector in public"


def drop(test_url: str = TEST_DATABASE_URL) -> str:
    """Drop the test database. Refuses a URL that names the engine's own database."""
    for source, url in engine_database_urls():
        if same_database(test_url, url):
            raise SystemExit(f"refusing: {describe(test_url)} is the engine's own database ({source})")
    dbname = _parts(test_url)["dbname"]
    with psycopg.connect(_maintenance_url(test_url), autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
    return f"dropped {describe(test_url)}"


if __name__ == "__main__":
    import sys

    action = sys.argv[1] if len(sys.argv) > 1 else "create"
    if action == "create":
        print(create())
    elif action == "reset":
        print(drop())
        print(create())
    else:
        raise SystemExit("usage: python -m tests.testdb [create|reset]")
