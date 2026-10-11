"""Test that the engine boots correctly."""

import json
import subprocess
import time

import httpx


def test_subprocess_boot_and_health():
    """Test engine boots via subprocess, prints ready JSON, and responds to health."""
    proc = subprocess.Popen(
        ["uv", "run", "voxscribe-engine", "--port", "0", "--providers", "fake", "--no-auth"],
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
        assert ready_data["version"] == "0.1.0"
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
    proc = subprocess.Popen(
        ["uv", "run", "voxscribe-engine", "--port", "0", "--providers", "fake"],
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
            # Request without token should fail
            resp = httpx.get(
                f"http://127.0.0.1:{port}/v1/health",
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
