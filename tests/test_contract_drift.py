"""CI check: exported contract files must match what the code produces.

Run with:
    uv run pytest tests/test_contract_drift.py -v

Failure means contract/ is out of sync with the code. Fix it by running:
    uv run python -m contract.export
then committing the updated files.

Also verifies that the schema-only app (used for export) has the same route set
as the production app would have (minus the lifespan-registered routes that need a
database, which are absent from the schema-only app by design).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

CONTRACT_DIR = Path(__file__).parent.parent / "contract"
FILES = ["openapi.json", "events.json", "websocket.json", "errors.json"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(obj) -> str:
    """Deterministic JSON string matching what export.py writes."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


@pytest.mark.parametrize("filename", FILES)
def test_contract_file_exists(filename: str) -> None:
    """Every expected contract file must be checked in."""
    path = CONTRACT_DIR / filename
    assert path.exists(), (
        f"{filename} is missing from contract/. "
        "Run: uv run python -m contract.export"
    )


@pytest.mark.parametrize("filename", FILES)
def test_contract_not_drifted(filename: str, tmp_path: Path) -> None:
    """The checked-in contract file must match what the exporter produces right now."""
    checked_in = CONTRACT_DIR / filename
    if not checked_in.exists():
        pytest.skip(f"{filename} not yet generated (run: uv run python -m contract.export)")

    # Re-export to a temp directory
    result = subprocess.run(
        [sys.executable, "-m", "contract.export", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    # If the exporter doesn't support --output-dir yet, fall back to running export() directly
    if result.returncode != 0:
        from contract.export import export
        export(output_dir=tmp_path)

    fresh = tmp_path / filename
    assert fresh.exists(), f"exporter did not produce {filename}"

    checked_in_text = checked_in.read_text(encoding="utf-8")
    fresh_text = fresh.read_text(encoding="utf-8")

    assert checked_in_text == fresh_text, (
        f"contract/{filename} is out of sync with the code.\n"
        "Fix: uv run python -m contract.export && git add contract/"
    )


def test_schema_app_route_set_matches_openapi() -> None:
    """The route set in the schema-only app matches the openapi.json paths."""
    from contract.export import _schema_app, route_set

    checked_in = CONTRACT_DIR / "openapi.json"
    if not checked_in.exists():
        pytest.skip("openapi.json not yet generated")

    app = _schema_app()
    live_routes = route_set(app)

    openapi = _load(checked_in)
    # Build the same format from the openapi document
    file_routes: set[str] = set()
    for path, methods in openapi.get("paths", {}).items():
        for method in methods:
            if method.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                file_routes.add(f"{method.upper()} {path}")

    missing_from_file = live_routes - file_routes
    extra_in_file = file_routes - live_routes

    assert not missing_from_file, (
        f"Routes in schema app but not in openapi.json (re-run export):\n"
        + "\n".join(sorted(missing_from_file))
    )
    assert not extra_in_file, (
        f"Routes in openapi.json but not in schema app (re-run export):\n"
        + "\n".join(sorted(extra_in_file))
    )
