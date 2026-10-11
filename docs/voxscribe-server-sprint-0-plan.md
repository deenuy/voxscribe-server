# VoxScribe Server Sprint 0: Implementation Plan

Date 2026-07-09. Scope voxscribe-server only. Prepares the repo for Sprint 1
(multi-tenant Postgres). Split into Sprint 0a (safe restructure) and Sprint 0b
(contracts package plus Postgres service seam). No em dashes.

Guiding constraints for this sprint:
- Archive, never delete. Superseded files move to archive/ as a backup copy.
- voxscribe-core is NOT modified in this sprint. The kernel is green; it adopts
  the shared contracts package later, as its own deliberate task.
- Postgres service is stood up, but the store migration to partitioned,
  RLS-enforced Postgres is Sprint 1, not here. Sprint 0 keeps the existing
  store working behind a seam.
- Test skeletons only. Write test files with descriptive names, docstrings, and
  TODO-marked stubs. Do not implement assertions this sprint; a later holistic
  test pass fills them in. Exception: keep the two smoke checks that already
  pass (health, app boot) real, so the suite is green.
- Google SWE naming: snake_case modules and functions, PascalCase classes,
  descriptive not abbreviated (config not cfg, extraction_legacy not extr).
  One clear responsibility per module. Package public surface via __init__.py.

Model for all of Sprint 0: Sonnet. Structural work with clear rules.

---

## Sprint 0a: Reconcile and restructure (server-only, reversible)

Goal: the repo reflects the sync-and-projection design and the kernel/product
boundary, with all superseded code archived, not deleted, and the suite green.

### Step 1: Capture the current HTTP surface before touching anything

Rationale: the restructure must provably preserve the existing API. Capture it
first so the refactor is verifiable.

- Create `tests/contract/test_http_surface.py` (skeleton): list the current
  routes and their shapes as TODO-documented expectations. Real assertions are
  deferred, but the file records the surface (`/v1/sessions`, `/v1/ideas`,
  health) so a later pass can lock it.
- Run the app, record the OpenAPI JSON to `docs/api/openapi_baseline.json` as a
  reference snapshot of the pre-refactor surface.

### Step 2: Create the archive area

- Create `archive/` at repo root with a `README.md` explaining: this holds
  superseded modules kept as backup during the Phase 1 restructure; nothing here
  is imported by the running app; safe to remove after Phase 1 ships.
- Add `archive/` to the ruff and pytest exclude lists so archived code is not
  linted or collected.

### Step 3: Archive the server-side processing pipeline

Rationale: the server must never re-run ASR or re-derive kernel judgments. These
modules belong to the pre-kernel design. Archive, do not delete.

Move to archive (git mv, preserving history):
- `app/asr.py` to `archive/app/asr.py`
- `app/extraction.py` to `archive/app/extraction.py`
- `app/queue.py` to `archive/app/queue.py`
- `app/jobs.py` to `archive/app/jobs.py`
- `app/ids.py` to `archive/app/ids.py` (quarantine; the sync contract in
  Sprint 2 confirms the kernel owns ID generation before final removal)

Then, in the active tree:
- Remove imports of the archived modules from `app/sessions.py` and
  `app/main.py`.
- Remove the `process_session` pipeline and the queue lifecycle from
  `app/main.py` lifespan.
- Keep an upload endpoint STUB. Do not remove the upload concept. IdeaScribe
  (Sprint 5) needs an upload path that forwards to the kernel for extraction.
  Replace the server-side processing body with a `NotImplementedError` stub
  documented as "forwards to kernel extraction in Phase 1 Sprint 5."

### Step 4: Restructure into kernel and product planes

Target layout (Google-style: descriptive snake_case, one responsibility each):

```
app/
  __init__.py
  main.py
  config.py
  errors.py
  logging_config.py
  kernel/
    __init__.py          # public interface: re-export only product-safe symbols
    router.py            # kernel FastAPI router aggregation
    sync_ingest.py       # sync ingest endpoint (stub this sprint)
    event_store.py       # event storage and projection (stub)
    blob_store.py        # blob metadata plus local-fs shim (stub)
    audit_query.py       # audit projection (stub)
    usage_rollup.py      # usage aggregation (stub)
    tenancy.py           # tenant resolution and auth middleware (stub)
    session_store.py     # session persistence (from app/store.py, seam kept)
  products/
    __init__.py
    ideascribe/
      __init__.py        # product public interface
      router.py          # from app/routers/ideas.py
      readiness_scoring.py   # from app/vocab.py scoring logic
      idea_projection.py     # project ideas from kernel events (stub)
      prd_generation.py      # PRD assembly (stub)
    seniorcare/
      __init__.py        # Sprint 6 skeleton, empty with TODO
```

Actions:
- Create the `kernel/` and `products/` packages with the modules above. Stubbed
  modules contain typed function signatures, docstrings, and `raise
  NotImplementedError("Phase 1 Sprint N")` bodies. No logic this sprint.
- `git mv app/store.py app/kernel/session_store.py`. Keep its current SQLite
  behavior working behind the seam; do not migrate to Postgres here.
- `git mv app/routers/ideas.py app/products/ideascribe/router.py`.
- Move scoring/rubric logic from `app/vocab.py` to
  `app/products/ideascribe/readiness_scoring.py`. Archive the remainder of
  `app/vocab.py` if anything is left.
- Split `app/routers/sessions.py`: the sync/query half becomes kernel surface
  (thin stubs this sprint); the IdeaScribe-specific session views move to the
  ideascribe router. Archive the original `app/routers/sessions.py`.
- Archive the now-empty `app/routers/` package.
- Update `app/main.py` to include routers from the new locations.
- Define the kernel public interface in `app/kernel/__init__.py`: re-export only
  the symbols product planes may use. Products import `from app.kernel import X`,
  never `from app.kernel.session_store import X`.

### Step 5: Enforce the boundary with a test skeleton

- Create `tests/architecture/test_plane_boundary.py` (skeleton with real intent
  documented, assertions TODO): walk `app/products/**/*.py`, collect
  `from app.kernel.*` imports, assert none reach past the public `__init__`.
  Leave the AST walk as pseudocode this sprint; wire it real in the later test
  pass. Include the negative case in the docstring so intent is preserved.

### Step 6: Tests, lint, make targets (skeletons plus the two real smokes)

- Create `tests/__init__.py`, `tests/conftest.py` (TestClient fixture, real).
- `tests/smoke/test_health.py`: REAL, asserts GET /health returns ok.
- `tests/smoke/test_app_boot.py`: REAL, asserts the app imports and mounts
  routers with the new topology.
- All other test files: skeletons with docstrings and TODO stubs.
- Makefile: add `test` (pytest), `lint` (ruff check app tests), `format`
  (ruff format app tests). Exclude `archive/`.
- Run `make lint` and fix violations in the active tree only.

### Sprint 0a acceptance

- `make test` green (the two real smokes pass; skeletons are collected but
  marked skip/TODO, not failing).
- `make lint` clean on the active tree; `archive/` excluded.
- App boots with the new kernel/products topology; GET /health returns ok.
- The OpenAPI surface still contains the preserved routes (manual check against
  `openapi_baseline.json` this sprint; automated in the later test pass).
- Every archived file is under `archive/` with history preserved; nothing in
  `archive/` is imported by the running app.

---

## Sprint 0b: Contracts package and Postgres service seam

Goal: the shared contracts package exists and voxscribe-server imports it, and
the Postgres service is available for Sprint 1, without migrating the store yet.

### Step 7: Locate or reconstruct the event contract

- Copy `docs/EVENT-CONTRACT.md` from voxscribe-core into this repo's `docs/` as
  the schema reference. If absent, reconstruct it from the kernel's event
  dataclasses. This is a read from the kernel, not a modification of it.

### Step 8: Scaffold voxscribe-contracts (server-only dependency)

Create `voxscribe-contracts/` as a sibling package. voxscribe-core is NOT
modified this sprint; only voxscribe-server depends on the package.

```
voxscribe-contracts/
  pyproject.toml            # name voxscribe-contracts, version = event schema version
  voxscribe_contracts/
    __init__.py             # re-export all public models
    sessions.py             # Session
    segments.py             # Segment
    events.py               # Event, PolicyEvent
    usage.py                # UsageRecord
    blobs.py                # BlobMetadata
```

- Logic-free Pydantic models matching the event contract exactly. No business
  rules, no scoring, no validation beyond type and shape.
- Match the existing camelCase alias convention (ConfigDict alias generator).
- `pip install -e ../voxscribe-contracts` from voxscribe-server succeeds.
- Add the dependency to voxscribe-server `pyproject.toml` and
  `requirements.txt`.
- Replace `app/models.py` internals with re-exports from voxscribe-contracts,
  keeping any server-only request models (`SessionStart`, `SessionFromUrl`)
  local. Archive the replaced model definitions.

### Step 9: Postgres service seam (service up, store not migrated)

- Add config: `PostgresSettings` (dsn from `DATABASE_URL`, pool_size,
  pool_overflow), `BlobSettings` (backend gcs or local, bucket, local_path),
  `AuthSettings` (mode token now, idp seam, token_secret). Archive the removed
  processing/models/queue settings blocks.
- `docker-compose.yml`: add a `postgres:16-alpine` service with healthcheck,
  env (`POSTGRES_DB/USER/PASSWORD`), a data volume, and `DATABASE_URL` wired to
  the api service. Remove the Ollama `extra_hosts`.
- Do NOT migrate `session_store.py` to Postgres. It keeps its current behavior
  behind the seam. Sprint 1 implements the partitioned, RLS-enforced Postgres
  schema. Sprint 0b only makes the service available.
- `Dockerfile`: include the contracts package, remove ffmpeg and ASR system
  deps.

### Step 10: Documentation and decision records

- `docs/decisions/0001_archive_server_side_asr_extraction.md`: why the server
  must not re-derive from audio; what was archived and where.
- `docs/decisions/0002_kernel_product_plane_boundary.md`: the split and what
  each plane owns.
- `docs/decisions/0003_contracts_package_and_postgres_seam.md`: why contracts is
  server-only for now and why Postgres is a seam this sprint.
- Update `README.md`: sync-and-projection architecture, the plane boundary, the
  new directory layout, the contracts dependency. Remove ASR/extraction/queue
  narrative.
- Add `CHANGELOG.md` with the Sprint 0a and 0b entries.
- Move `claude-session.md` to `archive/` (do not delete; confirm it is not the
  only copy of useful history) and add it to `.gitignore`.

### Sprint 0b acceptance

- voxscribe-contracts installs and imports in voxscribe-server; `app/models.py`
  re-exports from it.
- `docker-compose up` stands up Postgres (healthy) and the app; GET /health
  returns ok. The store still works on its existing backend behind the seam.
- voxscribe-core is unchanged (verify: no commits to that repo this sprint).
- Decision records and CHANGELOG present; README reflects the new architecture.

---

## Google SWE naming conventions applied

- Modules: snake_case, descriptive, one responsibility. `readiness_scoring.py`
  not `scoring.py` or `vocab.py`; `sync_ingest.py` not `sync.py`.
- Decision records: zero-padded ordinal plus snake_case slug,
  `0001_archive_server_side_asr_extraction.md`.
- Test layout by category: `tests/smoke/`, `tests/architecture/`,
  `tests/contract/`, mirroring the concern under test.
- Classes PascalCase, functions and variables snake_case, constants
  UPPER_SNAKE_CASE. No abbreviations that lose meaning.
- Public package surface is explicit in `__init__.py`; private modules are not
  imported across plane boundaries.

## Dependency order

```
Sprint 0a:
  Step 1 (capture surface) ─> Step 2 (archive area) ─> Step 3 (archive pipeline)
  ─> Step 4 (restructure) ─> Step 5 (boundary skeleton) ─> Step 6 (tests/make)

Sprint 0b (after 0a green):
  Step 7 (event contract) ─> Step 8 (contracts package)
  Step 9 (Postgres seam) can run parallel to Step 8
  Step 10 (docs) finalizes last
```

## What Sprint 0 deliberately does not do

- Does not migrate the store to Postgres (Sprint 1).
- Does not implement sync, events, audit, usage, or projections (stubs only).
- Does not modify voxscribe-core.
- Does not hard-delete anything (archive only; final deletion after Phase 1).
- Does not implement test assertions beyond the two real smokes (skeletons with
  TODOs for the later holistic test pass).

---

# Holistic view: Sprints 1 through 6 (unchanged, for context)

Sprint 0 prepares for these. Full detail lives in
`voxscribe-server-implementation-prd.md`. Summary so the boundary set in
Sprint 0 stays aligned with what comes next:

| Sprint | Goal | Difficulty | Model |
|---|---|---|---|
| 1 Multi-tenant persistence and auth | Postgres partitioned by tenant_id, RLS, tenant-scoped auth, idempotency keys | Medium | Sonnet (Opus for RLS design review) |
| 2 Lossless outbox sync | Device outbox drains to server, zero loss, idempotent, hash-verified, blob upload to tenant-prefixed GCS | Complex | Opus |
| 3 Audit export and usage rollup | Central tenant-scoped audit-replay parity with kernel; usage aggregated to billing-ready rollup; server-side retention purge | Medium | Sonnet |
| 4 IdeaScribe product plane backend | Idea projection, 10-dimension readiness scoring (config weights), gaps/contradictions, PRD generation, credit-metered | Medium | Sonnet |
| 5 IdeaScribe frontend and demo loop | Mockup UI on the Sprint 4 APIs; capture-or-upload path (upload forwards to kernel, never re-extracts); readiness pipeline and pulse metrics from real data | Medium | Sonnet |
| 6 SeniorCare plane skeleton and self-host | Wellbeing-timeline and escalation-consumer skeletons; single docker-compose self-host; compliance seams inert and flagged | Medium | Sonnet |

Boundary implications Sprint 0 must respect so later sprints drop in cleanly:
- Sprint 2 confirms the kernel owns ID generation; only then is archived
  `ids.py` removed for good.
- Sprint 4 product-plane LLM calls (gap detection) are credit-metered and
  product-side, never on a kernel path; the boundary test must allow product
  planes to call an LLM while forbidding kernel-table imports.
- Sprint 5 upload path forwards to the kernel; the upload stub created in
  Sprint 0a Step 3 is where that lands.
- Sprint 1 migrates `session_store.py` to partitioned Postgres; the seam kept in
  Sprint 0a Step 4 is where that migration attaches.

## Phase 1 exit (unchanged)

Sync lossless and idempotent (hash-verified); tenancy isolation holds under
adversarial cross-tenant tests; audit export matches kernel local audit-replay;
IdeaScribe runs end to end as the proof client; SeniorCare skeleton and
self-host path exist; every counsel-gated compliance seam (therapist add-on,
biometric features) inert and flagged.
