# Sprint 0: Recommendations & Task Breakdown

Date: 2026-07-09. Derived from the PRD `voxscribe-server-implementation-prd.md`
Sprint 0 goal: the repo reflects the sync-and-projection design, the contracts
package exists, and the kernel/product boundary is enforced by structure.

---

## 1. Gap Analysis Summary

| PRD Requirement | Current State | Severity |
|---|---|---|
| voxscribe-contracts package | Does not exist. `app/models.py` has only 2 request models (`SessionStart`, `SessionFromUrl`). No event, session, segment, usage, or blob-metadata Pydantic models. | **Blocker** |
| Resolve `app/asr.py` and `app/extraction.py` | Both exist and are actively imported by `app/sessions.py` as the core processing pipeline. They run ASR and LLM extraction server-side — exactly what the PRD forbids. | **Blocker** |
| Restructure routers into kernel/product planes | Current `app/routers/` is flat: `sessions.py` (mixed — session CRUD + upload/processing/SSE) and `ideas.py` (idea CRUD). No `app/kernel/` or `app/products/ideascribe/`. | **Blocker** |
| Boundary enforcement (import-linter or test) | No enforcement exists. Product and kernel concerns are freely intermixed. | **Required** |
| Config: Postgres DSN, GCS/blob, auth mode | Config has only SQLite path, no Postgres, no GCS/blob, and auth is default tenant/workspace headers only. `docker-compose.yml` has no Postgres service. | **Required** |
| `make test` and `make lint` green | No `tests/` directory exists. `make test` and `make lint` targets are missing from the Makefile. | **Required** |
| docker-compose with Postgres + app | `docker-compose.yml` only defines the `api` service. No Postgres. | **Required** |
| GET /health returns ok | Exists and works. | **Done** |
| EVENT-CONTRACT.md source of truth | Not present in `docs/`. Referenced by PRD as the schema source for voxscribe-contracts. | **Dependency** |

---

## 2. Recommended Sprint 0 Tasks (in order)

### Task 0.0 — Locate or Author the Event Contract

**What:** Find the Phase 0 `EVENT-CONTRACT.md` (expected at `docs/EVENT-CONTRACT.md`
in voxscribe-core) or reconstruct it from available sources. The event vocabulary
is the source of truth for the contracts package.

**Why:** Without it, the contracts package has no schema to encode. Every model in
voxscribe-contracts must match the Phase 0 event vocabulary exactly.

**Actions:**
- [ ] Search voxscribe-core repo for `docs/EVENT-CONTRACT.md` or equivalent
- [ ] If missing, extract the schema from kernel source (likely Python dataclasses
  or Pydantic models in voxscribe-core) and draft the contract document
- [ ] Place a copy in `docs/EVENT-CONTRACT.md` for reference

**Owner:** Deenu (domain knowledge of Phase 0 kernel)

---

### Task 0.1 — Scaffold the voxscribe-contracts Package

**What:** Create `voxscribe-contracts/` as a sibling directory (or separate repo)
with a `pyproject.toml`, logic-free Pydantic models, and a version tied to the
event-contract schema version.

**Models required (minimum):**
- `Session` — id, tenant_id, created_at, status, metadata
- `Segment` — id, session_id, speaker, start_ms, end_ms, text, clean_text
- `Event` — id, session_id, segment_id, event_type, payload, timestamp, content_hash
- `PolicyEvent` — extends Event for safety/retention decisions (heard, decided,
  policy_version, rationale)
- `UsageRecord` — finalized_word_count, action_type, tenant_id, period
- `BlobMetadata` — id, tenant_id, path, content_hash, retention_window_days, policy_permitted

**Actions:**
- [ ] Create `voxscribe-contracts/` directory beside `voxscribe-server/`
- [ ] Add `pyproject.toml` (name: `voxscribe-contracts`, version: matches
  event-contract schema version, pure Pydantic dependency only)
- [ ] Implement Pydantic models per the event contract, with zero logic methods
  (no business rules, no scoring, no validation beyond type/shape)
- [ ] Add `from __future__ import annotations` and `ConfigDict` for camelCase alias
  generation (matching current `CamelModel` pattern)
- [ ] Verify: `pip install -e ../voxscribe-contracts` succeeds from voxscribe-server
- [ ] Add `voxscribe-contracts` to voxscribe-server's `pyproject.toml` dependencies
  and `requirements.txt`

**Package structure:**
```
voxscribe-contracts/
  pyproject.toml
  voxscribe_contracts/
    __init__.py          # re-exports all public models
    sessions.py          # Session model
    segments.py          # Segment model
    events.py            # Event, PolicyEvent models
    usage.py             # UsageRecord model
    blobs.py             # BlobMetadata model
```

**Model:** Sonnet. Mechanical Pydantic modeling from a known schema.

---

### Task 0.2 — Resolve app/asr.py and app/extraction.py

**What:** Per the PRD governing principle: the server must never re-run ASR or
re-derive kernel judgments from audio. These modules belong to a pre-kernel
conception of the server.

**Decision:** **Delete** them, not just demote. The governing principle is
unambiguous — keeping even a legacy fallback invites misuse and creates a second
source of truth. The kernel already computed these artifacts on-device.

**Actions:**
- [ ] Delete `app/asr.py`
- [ ] Delete `app/extraction.py`
- [ ] Remove the import of `asr` and `extraction` from `app/sessions.py`
- [ ] Remove the processing pipeline (`process_session`) that calls ASR → extraction
- [ ] Remove `transcribe` and `extract_idea` call sites
- [ ] Remove `app/vocab.py` or move the shared types (idea types, rubric constants)
  to `app/products/ideascribe/` where they belong
- [ ] Remove ASR/extraction-related config from:
  - `app/config.py` (`ProcessingSettings`, `OllamaSettings`, `ClaudeSettings`,
    `OpenAISettings`, `ModelsSettings`)
  - `config/config.yaml` (`processing`, `models` sections)
  - `requirements.txt` (no new ASR deps needed)
  - `pyproject.toml` optional-dependencies (`local-asr`, `cleanup`)
- [ ] Remove queue-based processing pipeline entirely (`app/queue.py`, `app/jobs.py`)
  — the server is now a sync-and-projection plane, not a processing worker
- [ ] Simplify `app/main.py` lifespan (no queue start/stop)
- [ ] Record the decision in a CHANGELOG entry or `docs/decisions/0001-remove-asr-extraction.md`

**Impact on other modules:**
- `app/sessions.py`: Remove `process_session`, `ProgressHub`, `hub`; this file
  becomes the SSE progress hub only, or the hub moves to `app/kernel/sync.py`
- `app/routers/sessions.py`: Remove upload/enqueue/SSE endpoints (upload and
  processing are kernel concerns; re-evaluate what session endpoints the server
  needs vs. what the kernel owns)
- `app/store.py`: Remove `jobs` table, `devices` table (device claims are
  kernel-side); simplify to sessions + ideas

**Model:** Sonnet. Straightforward deletion with import chain cleanup.

---

### Task 0.3 — Restructure Into Kernel and Product Planes

**What:** Reorganize `app/` to enforce the kernel/product boundary as described
in the PRD.

**Target structure:**
```
app/
  __init__.py
  main.py                    # FastAPI app, lifespan, middleware
  config.py                  # server config (Postgres DSN, GCS, auth)
  errors.py                  # shared error types
  logging_config.py          # structured logging (unchanged)
  kernel/                    # ← NEW: kernel plane (sync-and-projection)
    __init__.py               # public interface: re-export only what products may import
    router.py                 # includes kernel routers
    sync.py                   # sync ingest, outbox drain
    events.py                 # event storage/projection
    blobs.py                  # blob metadata, GCS/local-fs shim
    audit.py                  # audit query/projection
    usage.py                  # usage rollup
    tenancy.py                # tenant isolation, auth middleware
    store.py                  # Postgres store (replaces SQLite store)
  products/                   # ← NEW: product planes
    __init__.py
    ideascribe/               # IdeaScribe vertical
      __init__.py
      router.py               # ideascribe API router
      scoring.py              # rubric scoring
      prds.py                 # PRD generation
      projections.py          # idea projection from kernel events
    seniorcare/               # SeniorCare skeleton (Sprint 6, create dir only)
      __init__.py
```

**Actions:**
- [ ] Create `app/kernel/` package with `__init__.py` that defines the public
  interface (classes/functions products may import)
- [ ] Create `app/products/ideascribe/` package
- [ ] Create `app/products/seniorcare/` package (empty skeleton with TODO)
- [ ] Move `app/store.py` → `app/kernel/store.py` (adapt to Postgres later)
- [ ] Split `app/routers/sessions.py` → kernel portion (session sync/query) stays
  in `app/kernel/`; product portion (IdeaScribe-specific session views) moves to
  `app/products/ideascribe/`
- [ ] Move `app/routers/ideas.py` → `app/products/ideascribe/router.py`
- [ ] Move scoring/vocab logic (`app/vocab.py`) → `app/products/ideascribe/scoring.py`
- [ ] Remove `app/routers/` directory entirely
- [ ] Update `app/main.py` to include routers from new locations
- [ ] Run existing smoke test to verify nothing broke structurally

**Model:** Sonnet. Structural refactor with clear rules, no new algorithms.

---

### Task 0.4 — Enforce the Kernel/Product Boundary

**What:** Add a test that fails if any product module imports a kernel
implementation detail (private module, table, or ORM model) directly.

**Approach options:**
1. **Import-linter** (`pip install import-linter`): declarative contract file
   listing allowed/disallowed imports between packages. CI-friendly.
2. **Custom pytest fixture**: a test that walks `app/products/**/*.py` AST,
   collects imports, and asserts none reference `app.kernel` internals beyond
   the public `__init__.py` interface.

**Recommended:** Option 2 (custom test). Fewer dependencies, trivial to maintain,
fits the "simple test" described in the PRD acceptance criteria.

**Actions:**
- [ ] Define the kernel public interface in `app/kernel/__init__.py`:
  - Re-export only the classes, functions, and Pydantic models that product
    planes are permitted to use
  - Example: `from app.kernel.store import Store` is disallowed; products must
    import `from app.kernel import Store`
- [ ] Create `tests/` directory
- [ ] Write `tests/test_boundary.py`:
  - Test 1: Import every public symbol from `app.kernel` and verify the module
    path is `app.kernel` (not e.g. `app.kernel.store`)
  - Test 2: Walk `app/products/` AST, collect all `from app.kernel.* import`
    statements, assert none import from private submodules
  - Test 3 (negative): Verify a deliberate import of `app.kernel.store.Store`
    from a product module fails the test
- [ ] Add `import-linter` contract as optional CI check if desired

**Model:** Sonnet. AST walking + assertion logic.

---

### Task 0.5 — Update Config and Docker Compose

**What:** Add Postgres, GCS/local-fs blob shim, and auth config. Make
`docker-compose up` stand up Postgres + the app.

**Config additions needed (`app/config.py` + `config/config.yaml`):**
- [ ] `PostgresSettings`: `dsn` (from `DATABASE_URL` env var), `pool_size`, `pool_overflow`
- [ ] `BlobSettings`: `backend` (`gcs` | `local`), `gcs_bucket`, `local_path`
- [ ] `AuthSettings`: add `mode` (`token` for now, `idp` seam), `token_secret`
- [ ] Remove `ProcessingSettings`, `ModelsSettings`, `OllamaSettings`,
  `ClaudeSettings`, `OpenAISettings` (deleted with asr/extraction in Task 0.2)
- [ ] Remove `QueueSettings` (deleted with queue in Task 0.2)
- [ ] Remove `StorageSettings.data_dir`, `StorageSettings.db_path`,
  `StorageSettings.id_floor` (SQLite-specific, replaced by Postgres DSN)

**Docker compose updates:**
- [ ] Add `postgres` service with official `postgres:16-alpine` image
- [ ] Add healthcheck on Postgres
- [ ] Add `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` env vars
- [ ] Wire `DATABASE_URL` into the `api` service
- [ ] Add volume for Postgres data persistence
- [ ] Remove `extra_hosts` for Ollama (no longer needed)

**Actions:**
- [ ] Update `app/config.py` with new settings dataclasses
- [ ] Update `config/config.yaml` with default Postgres/blob/auth values
- [ ] Update `docker-compose.yml` with Postgres service
- [ ] Verify: `docker-compose up` → Postgres healthy → app starts → `GET /health` returns ok
- [ ] Add `docker-compose.override.yml` to `.gitignore` for local overrides

**Model:** Sonnet. Configuration and Docker plumbing.

---

### Task 0.6 — Add Tests, Lint, and Make Targets

**What:** Create the `tests/` directory, add initial tests, wire `make test`
and `make lint` targets.

**Actions:**
- [ ] Create `tests/` directory with `__init__.py`
- [ ] Create `tests/conftest.py` with FastAPI `TestClient` fixture and test database
- [ ] Write `tests/test_health.py`: `GET /health` returns 200 with expected keys
- [ ] Write `tests/test_boundary.py` (from Task 0.4)
- [ ] Add `make test` target: `pytest -v`
- [ ] Add `make lint` target: `ruff check app/ tests/`
- [ ] Add `make format` target: `ruff format app/ tests/`
- [ ] Run lint and fix any existing violations
- [ ] Verify: `make test && make lint` both green

**Model:** Sonnet. Boilerplate test setup.

---

### Task 0.7 — Update Documentation

**What:** Update `README.md` to reflect the new sync-and-projection architecture.
Record all Sprint 0 decisions.

**Actions:**
- [ ] Update `README.md` architecture section:
  - Replace "LLM processing backend" narrative with "sync-and-projection plane"
  - Document the kernel/product boundary
  - Update directory structure diagram
  - Remove references to ASR backends, extraction, queue
  - Add voxscribe-contracts dependency
- [ ] Create `CHANGELOG.md` (or `docs/changelog.md`) with Sprint 0 entry:
  - Date and scope
  - Decision: app/asr.py and app/extraction.py deleted per governing principle
  - Decision: contracts package extracted
  - Decision: kernel/product boundary established
- [ ] Create `docs/decisions/` directory (ADR-lite):
  - `0001-remove-asr-extraction.md` — why server must not re-derive from audio
  - `0002-kernel-product-boundary.md` — the split and what each plane owns
- [ ] Remove `claude-session.md` from repo root (61KB session transcript; add to
  `.gitignore` if needed)

**Model:** Sonnet. Documentation writing.

---

## 3. Dependency Order

```
Task 0.0 (Event Contract) ──┐
                             ├──> Task 0.1 (Contracts Package) ──┐
                             │                                     │
Task 0.2 (Delete ASR/Extract) ────────────────────────────────────┤
                             │                                     │
                             └──> Task 0.3 (Restructure) ─────────┤
                                                                   │
Task 0.5 (Config/Docker) ─────────────────────────────────────────┤
                                                                   │
                             ┌─────────────────────────────────────┘
                             ▼
                      Task 0.4 (Boundary Test)
                             │
                             ▼
                      Task 0.6 (Tests/Lint/Make)
                             │
                             ▼
                      Task 0.7 (Documentation)
```

Tasks 0.0, 0.2, and 0.5 can run in parallel. Task 0.1 depends on 0.0.
Task 0.3 depends on 0.1 (needs contracts) and 0.2 (needs deleted files).
Task 0.4 depends on 0.3 (the boundary to enforce must exist).
Task 0.6 depends on 0.4 (needs the boundary test).
Task 0.7 spans the sprint but finalizes last.

---

## 4. Files to Delete

| File | Reason |
|---|---|
| `app/asr.py` | Server must never re-run ASR (governing principle) |
| `app/extraction.py` | Server must never re-derive kernel judgments from audio |
| `app/queue.py` | Processing queue — no processing pipeline remains |
| `app/jobs.py` | Job model — no queue/jobs remain |
| `app/ids.py` | Sequential idea ID allocator — client/kernel owns ID generation |
| `app/routers/__init__.py` | Routers directory being replaced |
| `app/routers/sessions.py` | Split into kernel + product (not deleted, restructured) |
| `app/routers/ideas.py` | Moved to products/ideascribe/ (not deleted, restructured) |
| `claude-session.md` | 61KB session transcript, not source code |

## 5. Files to Create

| File | Purpose |
|---|---|
| `voxscribe-contracts/pyproject.toml` | Contracts package metadata |
| `voxscribe-contracts/voxscribe_contracts/__init__.py` | Public re-exports |
| `voxscribe-contracts/voxscribe_contracts/sessions.py` | Session model |
| `voxscribe-contracts/voxscribe_contracts/segments.py` | Segment model |
| `voxscribe-contracts/voxscribe_contracts/events.py` | Event + PolicyEvent models |
| `voxscribe-contracts/voxscribe_contracts/usage.py` | UsageRecord model |
| `voxscribe-contracts/voxscribe_contracts/blobs.py` | BlobMetadata model |
| `app/kernel/__init__.py` | Kernel public interface |
| `app/kernel/router.py` | Kernel FastAPI router |
| `app/kernel/store.py` | Postgres store (migrated from SQLite store) |
| `app/kernel/tenancy.py` | Auth middleware, tenant resolution |
| `app/kernel/sync.py` | Sync ingest endpoint (stub for Sprint 0) |
| `app/kernel/events.py` | Event storage/projection (stub for Sprint 0) |
| `app/kernel/blobs.py` | Blob metadata, local-fs shim |
| `app/kernel/audit.py` | Audit query (stub for Sprint 0) |
| `app/kernel/usage.py` | Usage rollup (stub for Sprint 0) |
| `app/products/__init__.py` | Empty |
| `app/products/ideascribe/__init__.py` | Public interface |
| `app/products/ideascribe/router.py` | IdeaScribe API (from app/routers/ideas.py) |
| `app/products/ideascribe/scoring.py` | Rubric scoring (from app/vocab.py) |
| `app/products/ideascribe/projections.py` | Idea projection from events (stub) |
| `app/products/ideascribe/prds.py` | PRD generation (stub) |
| `app/products/seniorcare/__init__.py` | Skeleton for Sprint 6 |
| `tests/__init__.py` | Empty |
| `tests/conftest.py` | Test fixtures |
| `tests/test_health.py` | Health endpoint test |
| `tests/test_boundary.py` | Kernel/product boundary enforcement |
| `docs/decisions/0001-remove-asr-extraction.md` | Decision record |
| `docs/decisions/0002-kernel-product-boundary.md` | Decision record |
| `docs/EVENT-CONTRACT.md` | Event schema source of truth |

## 6. Files to Modify

| File | Change |
|---|---|
| `app/main.py` | Remove queue, update router includes, update docstring |
| `app/config.py` | Add Postgres/blob/auth settings; remove ASR/extraction/queue settings |
| `app/sessions.py` | Remove ASR/extraction pipeline; keep only SSE hub |
| `app/vocab.py` | Move scoring logic to products/ideascribe/; delete remainder |
| `app/store.py` | Simplify (remove jobs, devices tables); prepare for Postgres migration |
| `app/models.py` | Replace with imports from voxscribe-contracts |
| `config/config.yaml` | Add Postgres/blob sections; remove processing/models/queue |
| `docker-compose.yml` | Add Postgres service |
| `Dockerfile` | Update to include contracts package; remove ffmpeg |
| `Makefile` | Add `test`, `lint`, `format` targets; update `smoke` |
| `pyproject.toml` | Add voxscribe-contracts dependency; remove optional-deps |
| `requirements.txt` | Add voxscribe-contracts, asyncpg/psycopg; remove unused deps |
| `.gitignore` | Add `claude-session.md` |
| `README.md` | Rewrite architecture section |

---

## 7. Sprint 0 Acceptance Checklist

Per the PRD, Sprint 0 is done when:

- [ ] **voxscribe-contracts installs and imports** in voxscribe-server
- [ ] **Boundary test fails** on a deliberate kernel-table import from a product
  module, passes otherwise
- [ ] **make test and make lint green**
- [ ] **docker-compose up** stands up Postgres and the app
- [ ] **GET /health returns ok** with the new topology
- [ ] **app/asr.py and app/extraction.py fate is recorded** in CHANGELOG /
  decision records

---

## 8. Model Recommendation

**Sonnet** throughout Sprint 0 (matches PRD recommendation). Structural refactor
with clear rules, no deep algorithm design, no novel architecture. Every task is
well-specified with explicit before/after states.

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| No EVENT-CONTRACT.md found; contracts package blocked | Extract schema from voxscribe-core kernel source; draft contract from existing types in `app/vocab.py` and `app/models.py` |
| Deletion of queue/jobs breaks existing integration tests | No integration tests exist yet; the `make smoke` target uses the `/from-file` endpoint which will be restructured — rewrite smoke test post-refactor |
| Existing client depends on current API shape | Sprint 0 is structural only; the IdeaScribe API contract (`/v1/ideas`, `/v1/sessions`) should be preserved at the HTTP level even as internals move. If the client is tightly coupled, coordinate API surface changes |
| Postgres dependency adds local dev complexity | Keep the local-fs blob shim and SQLite as dev fallback via config; Postgres is the docker-compose default |