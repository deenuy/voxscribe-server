"""Export the engine contract to checked-in JSON files.

Usage:
    uv run python -m contract.export

Writes four files to the contract/ directory:
    openapi.json      OpenAPI 3.1 document (all HTTP routes, with response models)
    events.json       event schema from event_schema.json_schema()
    websocket.json    machine-readable description of the live meeting WebSocket
    errors.json       named error codes with their HTTP statuses

Output is deterministic: sorted keys, no timestamps. A diff signals a real change.
The route set in the schema-only app is verified to match the running app's routes; the
test suite (tests/test_contract_drift.py) catches divergence at CI time.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import sys
from pathlib import Path

from fastapi import FastAPI

# The contract/ directory itself
_CONTRACT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, obj) -> None:
    """Write a JSON file with sorted keys and no trailing newline variance."""
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _schema_app() -> FastAPI:
    """Build a FastAPI app that registers all routes without connecting to a database.

    The app has no lifespan, no middleware, and no database pool: it exists only so
    app.openapi() produces the full route list with response models. The test suite
    verifies that every route in this schema-only app also appears in the production
    app (and vice versa).
    """
    from app.local.version import engine_version

    app = FastAPI(
        title="VoxScribe Engine",
        version=engine_version(),
        # No lifespan: nothing to connect
    )

    # Health route (defined inline in create_app; replicate it here for schema coverage)
    from app.local.models import HealthResponse

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():  # pragma: no cover - schema only
        pass  # noqa: D401

    # Auto-discover and register all routers (same logic as create_app)
    from app.local import routers as _routers_pkg

    for _importer, modname, _ispkg in pkgutil.iter_modules(_routers_pkg.__path__):
        if modname.startswith("_"):
            continue
        full_name = f"app.local.routers.{modname}"
        module = importlib.import_module(full_name)
        if hasattr(module, "router"):
            app.include_router(module.router)

    return app


# ---------------------------------------------------------------------------
# Named error codes
# ---------------------------------------------------------------------------

# Every named code the engine sends, with its HTTP status and a one-line description.
# Update this list when a new code is added to errors.py or any router.
_ERRORS: list[dict] = [
    # generic
    {"code": "not_found",            "status": 404, "detail": "the addressed resource does not exist"},
    {"code": "invalid_request",      "status": 400, "detail": "the request body or a parameter failed validation"},
    {"code": "conflict",             "status": 409, "detail": "the operation conflicts with the current state"},
    {"code": "provider_unavailable", "status": 503, "detail": "the requested AI provider is not reachable"},
    {"code": "consent_required",     "status": 400, "detail": "the caller has not given the required consent"},
    {"code": "network_disabled",     "status": 403, "detail": "the engine is in offline mode; network calls are blocked"},
    {"code": "database_unavailable", "status": 503, "detail": "the database is not reachable"},
    {"code": "connector_not_connected", "status": 409, "detail": "the connector has no active token"},
    {"code": "connector_token_expired", "status": 409, "detail": "the connector token has expired"},
    {"code": "service_unavailable",  "status": 503, "detail": "a required internal service is not ready"},
    {"code": "internal_error",       "status": 500, "detail": "an unexpected error occurred"},
    # auth
    {"code": "unauthorized",         "status": 401, "detail": "no or invalid engine token"},
    {"code": "identity_required",    "status": 401, "detail": "no identity token was provided"},
    {"code": "identity_invalid",     "status": 401, "detail": "the identity token failed verification"},
    {"code": "origin_not_allowed",   "status": 403, "detail": "the request Origin is not in the allowed list"},
    # tickets
    {"code": "ticket_invalid",       "status": 401, "detail": "the ticket is not valid, was already used, or has expired"},
    {"code": "too_many_tickets",     "status": 429, "detail": "the caller already has the maximum number of live tickets"},
    # wake / voice
    {"code": "role_not_permitted",   "status": 403, "detail": "the caller's role may not perform this device operation"},
    {"code": "device_mismatch",      "status": 403, "detail": "the caller's token is bound to a different device"},
    {"code": "product_mismatch",     "status": 409, "detail": "the meeting was started by a different product"},
    {"code": "voice_tenant_unbound", "status": 409, "detail": "no tenant has armed the device yet"},
    {"code": "voice_session_in_progress", "status": 409, "detail": "a voice session is already running"},
    {"code": "voice_unavailable", "status": 409, "detail": "this engine was started without the voice assistant"},
    {"code": "session_in_progress",  "status": 409, "detail": "a session is already in progress"},
    # summary and tasks (the steps after Stop)
    {"code": "summary_edited",       "status": 409, "detail": "the summary was edited by hand; regenerating needs overwriteEdits: true"},
    {"code": "model_timeout",        "status": 504, "detail": "the on-device AI did not answer in time; try again"},
    # assistant profile
    {"code": "assistant_field_fixed", "status": 400, "detail": "the assistant setting is fixed for this engine or product mode and cannot be changed"},
    # uploads
    {"code": "payload_too_large",    "status": 413, "detail": "the request body exceeds the configured size limit"},
    # HTTP infrastructure
    {"code": "not_found",            "status": 404, "detail": "no such address"},
    {"code": "method_not_allowed",   "status": 405, "detail": "this method is not allowed here"},
]

# Deduplicate by (code, status) while preserving order
def _dedup_errors(errors: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result = []
    for e in errors:
        key = (e["code"], e["status"])
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


# ---------------------------------------------------------------------------
# WebSocket description
# ---------------------------------------------------------------------------

_WEBSOCKET: dict = {
    "description": (
        "The live meeting socket. Connect once a meeting is recording. The server sends "
        "typed frames; the client may send control frames."
    ),
    "route": "/v1/meetings/{meeting_id}/live",
    "auth": "ticket query parameter (from POST /v1/tickets with path=/v1/meetings/{id}/live)",
    "serverFrames": {
        "transcript.partial": {
            "description": "A transcript segment that may still change.",
            "schema": {"$ref": "#/$defs/Segment"},
        },
        "transcript.stable": {
            "description": "A transcript segment that will not change.",
            "schema": {"$ref": "#/$defs/Segment"},
        },
        "enhanced.update": {
            "description": "An enhanced block that was updated.",
            "schema": {"$ref": "#/$defs/EnhancedBlock"},
        },
        "assist.suggestion": {
            "description": "A live assistance suggestion.",
            "schema": {"$ref": "#/$defs/AssistSuggestion"},
        },
        "status": {
            "description": "The meeting state changed.",
            "schema": {"$ref": "#/$defs/MeetingStatus"},
        },
        "summary.ready": {
            "description": "The meeting summary is ready.",
            "schema": {"$ref": "#/$defs/SummaryReady"},
        },
        "note.updated": {
            "description": "A note block was updated.",
            "schema": {"$ref": "#/$defs/NoteUpdated"},
        },
        "audio.warning": {
            "description": "Audio was dropped during recording.",
            "schema": {"$ref": "#/$defs/AudioWarning"},
        },
        "reply": {
            "description": "A spoken reply from the assistant.",
            "schema": {"$ref": "#/$defs/ReplyPayload"},
        },
        "ping": {
            "description": "Keepalive ping. The client responds with a pong frame.",
            "schema": {"type": "object", "properties": {"t": {"type": "integer"}}},
        },
    },
    "clientFrames": {
        "note": {
            "description": "The client sends a note to attach to the meeting.",
            "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        "assist": {
            "description": "Request a live assistance suggestion at the current offset.",
            "schema": {"type": "object"},
        },
        "enhance": {
            "description": "Request enhanced enrichment for the current segment.",
            "schema": {"type": "object"},
        },
        "pong": {
            "description": "Response to a server ping.",
            "schema": {"type": "object", "properties": {"t": {"type": "integer"}}},
        },
    },
    "frameEnvelope": {
        "description": "Every frame (server and client) is a JSON object with 'type' and 'data'.",
        "schema": {
            "type": "object",
            "required": ["type", "data"],
            "properties": {
                "type": {"type": "string"},
                "data": {"type": "object"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def export(output_dir: Path | None = None) -> None:
    """Write all four contract files to *output_dir* (default: contract/)."""
    out = output_dir or _CONTRACT_DIR

    # 1. OpenAPI schema
    app = _schema_app()
    openapi = app.openapi()
    _write(out / "openapi.json", openapi)
    print(f"wrote {out / 'openapi.json'}", file=sys.stderr)

    # 2. Event schema
    from app.local.event_schema import json_schema
    events = json_schema()
    _write(out / "events.json", events)
    print(f"wrote {out / 'events.json'}", file=sys.stderr)

    # 3. WebSocket description
    _write(out / "websocket.json", _WEBSOCKET)
    print(f"wrote {out / 'websocket.json'}", file=sys.stderr)

    # 4. Error codes
    errors_doc = {
        "description": "Named error codes the engine returns in {error, detail} JSON bodies.",
        "errors": _dedup_errors(_ERRORS),
    }
    _write(out / "errors.json", errors_doc)
    print(f"wrote {out / 'errors.json'}", file=sys.stderr)


def route_set(app: FastAPI) -> set[str]:
    """Return the set of 'METHOD /path' strings for all API routes in *app*.

    Uses app.openapi()["paths"] as the canonical source because FastAPI stores
    included routers in a lazy form in app.routes; they are only flattened when
    the OpenAPI schema is generated.
    """
    openapi = app.openapi()
    result: set[str] = set()
    for path, methods in openapi.get("paths", {}).items():
        for method in methods:
            if method.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                result.add(f"{method.upper()} {path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the engine contract to JSON files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write contract files (default: contract/)",
    )
    args = parser.parse_args()
    export(Path(args.output_dir) if args.output_dir else None)
