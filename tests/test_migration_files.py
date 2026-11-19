"""Checks over the migration files themselves (README, "Migrations only add: expand now, contract later").

A migration only adds. A statement that removes or renames something (DROP TABLE, DROP COLUMN, DROP VIEW, RENAME) is
a contract step: it is allowed only in a file that says so with a marker naming the earlier migration that started
the expand, `-- contract: completes NNN (why nothing still uses it)`.
"""

import re
from pathlib import Path

import pytest

MIGRATIONS = Path(__file__).resolve().parent.parent / "app" / "local" / "migrations"

_DESTRUCTIVE = re.compile(r"\bDROP\s+(TABLE|COLUMN|VIEW)\b|\bRENAME\b", re.IGNORECASE)
_MARKER = re.compile(r"^--[ \t]*contract:[ \t]*completes[ \t]+(\d{3})\b[ \t]+\S", re.IGNORECASE | re.MULTILINE)


def _version(path: Path) -> int:
    return int(path.stem.split("_")[0])


def _strip_comments(sql: str) -> str:
    """The SQL without line comments, block comments or string literals, so prose cannot trip the check."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)  # first, so an apostrophe in a comment does not open a string
    sql = re.sub(r"\$\$.*?\$\$", "$$ $$", sql, flags=re.DOTALL)  # function bodies
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def contract_violations(name: str, sql: str, earlier: set[int]) -> list[str]:
    """Why `sql` breaks the expand-only rule; empty when it does not. `earlier` is every lower migration number."""
    found = sorted({m.group(0).upper() for m in _DESTRUCTIVE.finditer(_strip_comments(sql))})
    if not found:
        return []
    marker = _MARKER.search(sql)
    if marker is None:
        return [f"{name}: {', '.join(found)} without a '-- contract: completes NNN (why)' marker"]
    started = int(marker.group(1))
    if started not in earlier:
        return [f"{name}: contract marker names {started:03d}, which is not an earlier migration"]
    return []


def test_no_migration_removes_or_renames_without_a_contract_marker():
    files = sorted(MIGRATIONS.glob("*.sql"))
    versions = {_version(p) for p in files}
    problems = []
    for path in files:
        earlier = {v for v in versions if v < _version(path)}
        problems += contract_violations(path.name, path.read_text(), earlier)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("sql", [
    "DROP TABLE gate_decisions;",
    "drop table if exists gate_decisions cascade;",
    "ALTER TABLE turns DROP COLUMN user_text;",
    "ALTER TABLE turns RENAME COLUMN user_text TO heard;",
    "ALTER TABLE turns RENAME TO spoken_turns;",
    "DROP VIEW gate_outcomes;",
])
def test_the_check_refuses_a_removal_without_a_marker(sql):
    assert contract_violations("030_x.sql", sql, {21, 29})


def test_the_check_accepts_a_removal_whose_marker_names_an_earlier_migration():
    sql = "-- contract: completes 021 (no writer since 021's release)\nDROP TABLE gate_decisions;"
    assert contract_violations("030_x.sql", sql, {21, 29}) == []
    assert contract_violations("030_x.sql", sql.replace("021", "031"), {21, 29})   # not earlier
    assert contract_violations("030_x.sql", "-- contract: completes 021\nDROP TABLE g;", {21})  # no reason given


def test_the_check_ignores_words_in_comments_strings_and_function_bodies():
    sql = (
        "-- Dropping it is the contract step; DROP TABLE gate_decisions in a later release's migration.\n"
        "/* RENAME nothing here */\n"
        "COMMENT ON TABLE t IS 'do not DROP TABLE this';\n"
        "CREATE FUNCTION f() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE 'no RENAME'; END; $$;\n"
        "ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_status_check;\n"
    )
    assert contract_violations("030_x.sql", sql, {21}) == []
