"""The suite cannot reach live data: one test database, never the engine's own, refused when its public schema is used.

The incident behind this: a boot test started an engine on the default database, which migrated the developer's live
public schema with two unmerged migrations. Tests now run on a dedicated database (tests/testdb.py) and a session
guard refuses the run when that database is the engine's, or when its public schema holds rows.
"""

import ast
import uuid
from pathlib import Path

import psycopg
import pytest

from app.local.config import Config
from tests import testdb

TESTS = Path(__file__).parent


def test_the_test_database_is_not_the_engine_default_database():
    assert not testdb.same_database(testdb.TEST_DATABASE_URL, Config().database_url)
    assert testdb.DEFAULT_TEST_DATABASE_URL.endswith("/ideascribe_test")


def test_no_test_file_names_a_database_of_its_own():
    """Every test reaches Postgres through tests/testdb.py; a literal database URL elsewhere is a second definition."""
    allowed = {"testdb.py", "test_test_database.py", "test_config.py"}  # test_config checks the engine's default
    offenders = []
    for path in TESTS.rglob("*.py"):
        if path.name in allowed or "room" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("postgresql://"):
                if "localhost:59999" in node.value:  # the unreachable server the degraded-health test points at
                    continue
                offenders.append(f"{path.relative_to(TESTS)}:{node.lineno}")
    assert not offenders, f"database URL defined outside tests/testdb.py: {offenders}"


@pytest.mark.parametrize("a,b,same", [
    ("postgresql://localhost:5432/ideascribe_dev", "postgresql://127.0.0.1/ideascribe_dev", True),
    ("postgresql://localhost/ideascribe_dev", "postgresql:///ideascribe_dev?host=/tmp", True),
    ("postgresql://me:secret@localhost:5432/ideascribe_dev", "postgresql://localhost/ideascribe_dev", True),
    ("postgresql://localhost/ideascribe_dev", "postgresql://localhost/ideascribe_test", False),
    ("postgresql://localhost:5433/ideascribe_dev", "postgresql://localhost:5432/ideascribe_dev", False),
    ("postgresql://db.internal/ideascribe_dev", "postgresql://localhost/ideascribe_dev", False),
])
def test_same_database_compares_name_server_and_port(a, b, same):
    assert testdb.same_database(a, b) is same


def test_the_guard_refuses_the_engine_default_database_without_connecting(tmp_path):
    def must_not_connect(*_a, **_k):
        raise AssertionError("the guard connected to the engine's database")

    reason = testdb.unsafe_reason(Config().database_url, repo_root=tmp_path, connect=must_not_connect)
    assert reason and "engine's own database" in reason and "default Config" in reason


def test_the_guard_refuses_the_database_a_repo_env_file_names(tmp_path):
    (tmp_path / ".env").write_text("VOXSCRIBE_DATABASE_URL=postgresql://owner:hunter2@localhost:5432/someones_live_db\n")
    reason = testdb.unsafe_reason("postgresql://localhost/someones_live_db", repo_root=tmp_path,
                                  connect=lambda *a, **k: pytest.fail("connected"))
    assert reason and ".env" in reason
    assert "hunter2" not in reason and "owner" not in reason  # no credential in the message


def test_the_guard_reads_the_main_checkout_env_file_from_a_linked_worktree(tmp_path):
    main, wt = tmp_path / "main", tmp_path / "wt"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    (main / ".env").write_text("VOXSCRIBE_DATABASE_URL=postgresql://localhost/main_checkout_db\n")
    reason = testdb.unsafe_reason("postgresql://localhost/main_checkout_db", repo_root=wt,
                                  connect=lambda *a, **k: pytest.fail("connected"))
    assert reason and str(main / ".env") in reason


def test_the_guard_refuses_a_test_database_whose_public_schema_holds_rows(tmp_path):
    """A database whose public schema holds rows is in use by something other than the suite."""
    name = f"ideascribe_guard_{uuid.uuid4().hex[:8]}"
    admin = psycopg.conninfo.make_conninfo(testdb.TEST_DATABASE_URL, dbname="postgres")
    url = psycopg.conninfo.make_conninfo(testdb.TEST_DATABASE_URL, dbname=name)
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    try:
        assert testdb.unsafe_reason(url, repo_root=tmp_path) is None  # empty: allowed
        with psycopg.connect(url, autocommit=True) as c:
            c.execute("CREATE TABLE public.meetings (id text)")
            assert testdb.unsafe_reason(url, repo_root=tmp_path) is None  # a table with no rows: allowed
            c.execute("INSERT INTO public.meetings VALUES ('m1')")
        reason = testdb.unsafe_reason(url, repo_root=tmp_path)
        assert reason and "meetings" in reason and "public schema" in reason
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_the_guard_lets_an_unreachable_test_database_through(tmp_path):
    """Unit tests run without Postgres; the database tests then fail on their own."""
    assert testdb.unsafe_reason("postgresql://localhost:59999/nowhere", repo_root=tmp_path) is None


def test_the_suite_database_is_safe_right_now():
    assert testdb.unsafe_reason() is None


def test_schema_url_keeps_an_existing_query_string():
    assert testdb.schema_url("t_x", "postgresql://localhost/db").endswith("/db?options=-c%20search_path%3Dt_x,public")
    assert testdb.schema_url("t_x", "postgresql://localhost/db?sslmode=disable").endswith(
        "?sslmode=disable&options=-c%20search_path%3Dt_x,public")


def test_creating_the_test_database_refuses_the_engine_database():
    with pytest.raises(SystemExit, match="engine's own database"):
        testdb.create(Config().database_url)
