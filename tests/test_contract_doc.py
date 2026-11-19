"""CI check: every non-planned route in the hand-written contract doc must exist in the schema.

The inverse is not enforced here (a route in the schema not in the doc is acceptable for
internal or infrastructure routes). The failing direction is a route a client depends on
going missing from the schema without anyone noticing.

Skips with a clear message when the doc file is not found.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

CONTRACT_DIR = Path(__file__).parent.parent / "contract"
# The hand-written contract document lives in the ideascribe-app repo.
# Try the worktree path and the normal project layout.
_CANDIDATES = [
    Path(__file__).parent.parent.parent.parent / "ideascribe-app" / "docs" / "contract" / "engine-v1.md",
    Path(__file__).parent.parent.parent / "ideascribe-app" / "docs" / "contract" / "engine-v1.md",
]
_DOC_PATH: Path | None = next((p for p in _CANDIDATES if p.exists()), None)


# Matches inline backtick route references: `METHOD /path`
# e.g. `GET /v1/health`, `POST /meetings/{id}/start`, `DELETE /data`
_ROUTE_RE = re.compile(
    r"`(GET|POST|PUT|PATCH|DELETE|HEAD)\s+(/v?1?/?[a-zA-Z0-9_/{}.?|\\*=-]+)`"
)

# A route is "planned" when the same line or the following line contains the word "planned"
# (case-insensitive) or the phrase "not implemented" or "501".
_PLANNED_MARKERS = re.compile(r"planned|not.implemented|501\b", re.IGNORECASE)


def _extract_documented_routes(text: str) -> list[tuple[str, str, bool]]:
    """Return (method, path, is_planned) for every route in the document.

    A route is marked planned when the word 'planned' (or '501'/'not implemented')
    appears on the same table row or the following non-blank line.
    """
    lines = text.splitlines()
    results: list[tuple[str, str, bool]] = []

    for i, line in enumerate(lines):
        for match in _ROUTE_RE.finditer(line):
            method, path = match.group(1), match.group(2)
            # Strip trailing punctuation that isn't part of the path
            path = path.rstrip(").,;")
            # Check this line and the next non-blank line for planned markers
            context = line
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if nxt:
                    context += " " + nxt
                    break
            is_planned = bool(_PLANNED_MARKERS.search(context))
            results.append((method, path, is_planned))

    return results


def _normalise_path(path: str) -> str:
    """Normalise a doc path to the FastAPI path parameter style.

    The doc uses {id}, {meeting_id} etc. interchangeably; FastAPI uses whatever
    the route declares. Normalise both to {param} placeholders for comparison.
    """
    # Remove leading /v1 since the doc and the schema both include it; keep as-is
    # Replace any {something} with a canonical {p}
    return re.sub(r"\{[^}]+\}", "{p}", path)


def _schema_paths(openapi: dict) -> set[str]:
    """Return the set of 'METHOD /path' (with {p} placeholders) from openapi.json."""
    result: set[str] = set()
    for path, methods in openapi.get("paths", {}).items():
        norm = _normalise_path(path)
        for method in methods:
            if method.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
                result.add(f"{method.upper()} {norm}")
    return result


@pytest.mark.skipif(_DOC_PATH is None, reason="engine-v1.md not found alongside this repo (acceptable in CI without the app repo)")
def test_documented_non_planned_routes_exist_in_schema() -> None:
    """Every non-planned route in engine-v1.md must appear in openapi.json."""
    openapi_path = CONTRACT_DIR / "openapi.json"
    if not openapi_path.exists():
        pytest.skip("openapi.json not yet generated (run: uv run python -m contract.export)")

    openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
    schema_routes = _schema_paths(openapi)

    doc_text = _DOC_PATH.read_text(encoding="utf-8")
    documented = _extract_documented_routes(doc_text)

    missing: list[str] = []
    for method, path, is_planned in documented:
        if is_planned:
            continue
        norm = f"{method} {_normalise_path(path)}"
        if norm not in schema_routes:
            missing.append(f"{method} {path}")

    assert not missing, (
        "These routes are in engine-v1.md (non-planned) but missing from contract/openapi.json.\n"
        "Either add the route to the schema-only app in contract/export.py, or mark it '(planned)' in the doc:\n"
        + "\n".join(f"  {r}" for r in sorted(missing))
    )
