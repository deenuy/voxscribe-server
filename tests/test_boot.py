"""Test that the engine boots correctly.

Every engine started here gets a throwaway schema. An engine started on the default database
with no schema migrates the public schema, which is the developer's live data: a branch that
adds a migration would change that database just by running this file.
"""

import asyncio
import contextlib
import json
import subprocess
import time
import uuid

import httpx
import psycopg

BASE_DB = "postgresql://localhost:5432/ideascribe_dev"


@contextlib.contextmanager
def _throwaway_database_url():
    """A database URL whose search path is a fresh, migrated schema; the schema is dropped afterwards."""
    from app.local.migrations import run_migrations

    schema = f"t_boot_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(BASE_DB, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    try:
        asyncio.run(run_migrations(BASE_DB, schema=schema))
        yield f"{BASE_DB}?options=-c%20search_path%3D{schema},public"
    finally:
        with psycopg.connect(BASE_DB, autocommit=True) as c:
            c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_subprocess_boot_and_health():
    """Test engine boots via subprocess, prints ready JSON, and responds to health."""
    with _throwaway_database_url() as url:
        _boot_and_health(url)


def _boot_and_health(url):
    proc = subprocess.Popen(
        ["uv", "run", "voxscribe-engine", "--port", "0", "--providers", "fake", "--no-auth",
         "--database-url", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    port = None
    ready_line = None

    # Read stdout for ready line
    for _ in range(100):
        line = proc.stdout.readline()
        if line:
            line = line.strip()
            try:
                data = json.loads(line)
                if data.get("event") == "ready":
                    ready_line = line
                    port = data.get("port")
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)

    try:
        # Verify we got the JSON ready line
        assert port is not None, "No ready port found"
        assert ready_line is not None, "No ready JSON line printed"
        ready_data = json.loads(ready_line)
        assert ready_data["event"] == "ready"
        from importlib.metadata import version as _pkg_version
        assert ready_data["version"] == _pkg_version("voxscribe-server")
        assert isinstance(ready_data["port"], int)
        assert ready_data["port"] > 0

        # Test health endpoint
        time.sleep(0.5)
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=5)
        assert resp.status_code == 200
        health = resp.json()
        assert health["database"]["ok"] is True
        assert "schemaVersion" in health["database"]
        assert health["providers"]["asr"]["ok"] is True
        assert health["providers"]["llm"]["ok"] is True
        assert "liveLlm" in health["providers"]

        # Test SIGTERM exit
        proc.terminate()
        exit_code = proc.wait(timeout=5)
        # SIGTERM causes exit code 143 (or -15)
        assert exit_code in (0, 143, -15), f"Exit code {exit_code} not clean"
        # Process should be dead (poll returns exit code, not None)
        assert proc.poll() is not None, "Process still running after wait()"

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_subprocess_without_token_fails_with_auth():
    """Test that requests without token fail when auth is enabled."""
    with _throwaway_database_url() as url:
        _without_token_fails(url)


def _without_token_fails(url):
    proc = subprocess.Popen(
        ["uv", "run", "voxscribe-engine", "--port", "0", "--providers", "fake", "--database-url", url],
        # NOTE: no --no-auth, so token is required
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    port = None
    for _ in range(100):
        line = proc.stdout.readline()
        if line:
            try:
                data = json.loads(line.strip())
                if data.get("event") == "ready":
                    port = data.get("port")
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)

    try:
        if port:
            time.sleep(0.5)
            # /v1/health is public (commit 8acc390); use a protected route to verify auth is enforced
            resp = httpx.get(
                f"http://127.0.0.1:{port}/v1/meetings",
                timeout=5,
            )
            assert resp.status_code == 401
            data = resp.json()
            assert data["error"] == "unauthorized"
            assert "detail" in data

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_database_down_returns_degraded():
    """Test that health returns degraded when database is unreachable."""
    proc = subprocess.Popen(
        ["uv", "run", "voxscribe-engine", "--port", "0", "--providers", "fake",
         "--no-auth", "--database-url", "postgresql://localhost:59999/nonexistent"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    port = None
    for _ in range(100):
        line = proc.stdout.readline()
        if line:
            try:
                data = json.loads(line.strip())
                if data.get("event") == "ready":
                    port = data.get("port")
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)

    try:
        if port:
            time.sleep(0.5)
            resp = httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=5)
            assert resp.status_code == 200
            health = resp.json()
            assert health["database"]["ok"] is False
            assert health["status"] == "degraded"

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_no_test_boots_an_engine_on_the_default_database():
    """Every test that starts an engine process names its database, so none can migrate the public schema."""
    import ast
    from pathlib import Path

    offenders = []
    for path in Path(__file__).parent.rglob("*.py"):
        if "room" in path.parts:
            continue  # room tests run by hand against a database the person running them chooses
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.List):
                continue
            words = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if "voxscribe-engine" in words and "--database-url" not in words:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"engine started without --database-url: {offenders}"
