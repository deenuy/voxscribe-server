# voxscribe-server

The LLM processing backend for IdeaScribe, packaged to run in Docker on a Mac.

## What runs where

Recording happens in the Electron React Native client, which holds microphone
consent. Docker hosts this backend for processing only: it never touches the
mic. The client uploads the recorded session; this service queues it and a
worker pool runs ASR and LLM extraction, then serves the IdeaScribe API
contract (ideas, sessions, SSE).

```
Electron / iPhone client (mic + consent)        Docker container (this service)
   record locally  -- submit audio + metadata -->  enqueue -> worker -> ASR -> extract
   subscribe SSE   <----- progress events --------  serve /v1 contract
```

## Why a queue

Submission is decoupled from processing. When the user ends a capture, the
client uploads it, the backend enqueues a job and returns at once, and the user
can start the next capture immediately without waiting for the LLM. Workers
drain the queue in the background, so submit lag is tiny.

Day one this is an in-process worker pool, durable through the SQLite jobs
table. The `JobQueue` protocol in `app/queue.py` is the seam: a Kafka or
Temporal adapter implements the same three methods and the routers and the
processing handler do not change. Flip `QUEUE_BACKEND` to switch.

Production behavior built in:

- durability: jobs are persisted before they run and re-enqueued on startup if
  the process died mid-flight
- retries: transient failures retry with exponential backoff up to
  `QUEUE_MAX_ATTEMPTS`
- dead-letter: after the max attempts the job is marked dead and the failure is
  pushed to the client over SSE rather than hanging
- idempotency: one job per session, so duplicate submits are no-ops

## One recording per device, independent across devices

A device may hold one active recording at a time, enforced by a per-device
lock keyed on tenant and `X-Device-Id`. Starting a second recording on the same
device returns 409 `device_busy`. The same user on another device (iPhone while
the Mac records) is fully independent. Finalizing or cancelling releases the
device, and a stale claim past `RECORDING_TTL_SECONDS` is reclaimed so a crashed
client never blocks a device permanently. The one-shot upload path does not
claim a device, since the recording already finished client-side.

## voxscribe-core dependency

This server depends on voxscribe-core, the shared intelligence kernel. The
dependency is pinned to an exact version and resolved from a wheel that is
tracked in this repository at `wheels/voxscribe_core-0.1.1-py3-none-any.whl`.
The path is relative to `pyproject.toml` and `uv.lock` records the wheel's
sha256, so any checkout builds the same way with no sibling repository needed.
The `knowledge` extra (pypdf, openpyxl) is requested, so PDF and spreadsheet
ingestion work.

To rebuild the wheel from a core release tag (after a core update):

```bash
make core-wheel                       # builds from ../voxscribe-core at tag v0.1.1
make core-wheel CORE_TAG=v0.1.2       # or another tag; CORE_DIR=... for another checkout
# if the version changed, update the pin and the file name in pyproject.toml, then:
uv lock --upgrade-package voxscribe-core && git add wheels pyproject.toml uv.lock
```

Rebuilding from the same tag gives the same bytes (the build uses the tag's commit
time), so the hash only changes when core's content changes. If you swap the wheel
without refreshing the lock, `uv sync --frozen` refuses it because the hash no longer
matches. That is the pin doing its job.

### Development: switch between the pinned wheel and the sibling core

To work on voxscribe-core while running this server:

```bash
make core-dev            # installs ../voxscribe-core in editable mode
# ... change ../voxscribe-core; changes are picked up on the next import ...
make core-pinned         # back to the pinned wheel
```

Switching back uses `uv sync --frozen --reinstall-package voxscribe-core`. A plain
`uv sync` does not undo the editable install, because the version number is the same.

## Run it

```bash
make up        # builds the image and starts the service on :8000
make health    # shows queue, worker count, extractor, asr
make smoke     # upload a transcript and watch the progress stream
make logs
make down
```

Local without Docker: `make dev`. OpenAPI docs at http://localhost:8000/docs.

## Processing backends

Default to the lightest option so the image boots with no model and no key.

| Var | Default | Options |
|---|---|---|
| `QUEUE_BACKEND` | `memory` | `kafka`, `temporal` (future adapters) |
| `WORKER_CONCURRENCY` | `2` | sessions processed in parallel |
| `EXTRACTOR` | `simulated` | `ollama` (local), `claude` (cloud) |
| `ASR_BACKEND` | `stub` | `faster_whisper` (local), `openai` (cloud) |

## Integration flow

Headers on every request: `X-Tenant-Id`, `X-Workspace-Id`, `X-Device-Id`
(defaults applied in dev until JWT lands).

Pre-allocated (lets the UI show progress while recording):

```
POST /v1/sessions/start  { title, participants, context, projectId }
  -> { sessionId, ideaId, status: "recording" }     # 409 if this device is busy
# client records locally, then on end sends the query parameters with the file:
POST /v1/sessions/{sessionId}/from-file?title=...&participants=...&projectId=...
  multipart: file=<recording>
  -> { sessionId, ideaId, status: "processing" }     # returns immediately
GET  /v1/sessions/{sessionId}/progress               # SSE: {step, pct, done, error?}
GET  /v1/sessions/{sessionId}                         # session, idea, processingStatus, jobStatus
```

One-shot (upload at the end, no device claim):

```
POST /v1/sessions/from-file?title=...&participants=...&projectId=...
  multipart: file=<recording>
  -> { sessionId, ideaId, status: "processing" }
```

Metadata is accepted as both multipart form fields and query parameters.

## Endpoints

Ideas: `GET /v1/ideas` (status, type, projectId, scoreMin, scoreMax, ownerId,
hasLineage, search, sortBy, sortDir, page, pageSize), `GET /v1/ideas/{id}`,
`POST /v1/ideas`, `PATCH /v1/ideas/{id}`, `GET /v1/ideas/next-id`.

Sessions: `POST /v1/sessions/start`, `POST /v1/sessions/from-file`,
`POST /v1/sessions/{id}/from-file`, `POST /v1/sessions/from-url`,
`GET /v1/sessions`, `GET /v1/sessions/{id}`, `GET /v1/sessions/{id}/progress`.

Errors return clean JSON `{ error, detail }` with the right status: 409
`device_busy`, 404 `not_found`, 422 `validation_failed`, 503 `queue_unavailable`,
500 `internal_error` (no stack trace leaked).

## Storage and limits

SQLite plus uploaded files under `./data` (mounted into the container). Ideas,
sessions, jobs, and device claims are tenant scoped through a single store
helper. Backend ids start at `idea-0053` so they do not collide with the client
seed range.

Single uvicorn worker keeps the in-memory queue and SSE hub correct. Scaling to
multiple processes is exactly what the Kafka or Temporal adapter is for: move
the queue and the progress hub onto a shared bus, then run many workers.
