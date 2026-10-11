# VoxScribe Server Implementation PRD (Phase 1)

Date 2026-07-09. Scope voxscribe-server only. Phase 0 (native kernel) is
complete; this is the additive server tier. No em dashes. Every sprint states
goal, spec, what it closes, pass/fail acceptance, and its model recommendation.

## Governing principle (read before Sprint 0)

The server is a sync-and-projection plane, not a second brain. The kernel
already computed events, clean text, extractions, usage, and safety decisions
on-device, with provenance. The server ingests those artifacts, stores them
multi-tenant, and projects them into product surfaces. The server must never
re-run ASR or re-derive kernel judgments from audio; doing so creates a second
source of truth and breaks the kernel's provenance and audit guarantees at the
sync boundary.

Consequence for the existing boilerplate: app/asr.py and app/extraction.py
belong to a pre-kernel conception of the server and are the wrong layer now.
They are either deleted or demoted to a clearly-marked legacy/fallback path
that is off by default. Sprint 0 resolves this explicitly.

Boundary discipline (the kernel/product split, lifted to the server):
- Kernel plane: sync, event storage, blob storage, audit export, usage,
  auth/tenancy. Product-neutral. Knows sessions, segments, events, blobs.
- Product plane: per-vertical projections and policy (IdeaScribe scoring and
  PRD generation; later SeniorCare wellbeing and escalation). Product routers
  may only import the kernel plane's public interfaces, never its tables.
- One FastAPI modular monolith now; a product plane extracts into its own
  service later only when a vertical earns it. Extraction is a deployment
  change, not a redesign.

## The contracts package (do this in Sprint 0, it is now earned)

With a second consumer of the event schema (kernel + server + clients),
extract the Pydantic event/session/segment models into voxscribe-contracts, a
logic-free package that voxscribe-core, voxscribe-server, and every client
import. The event contract from Phase 0 Sprint 1 (docs/EVENT-CONTRACT.md) is
the source of truth; the package is its typed form. This package, not the
database, is what products depend on.

## How to read each sprint

Goal, one sentence. Spec, named against the contracts. Closes, the capability
it establishes. Acceptance, pass/fail commands or assertions. Model, the
recommended Claude model and why. Every sprint ends with the test suite green
and does not start the next sprint.

---

## Sprint 0: Reconcile the boilerplate and freeze the boundary  [Easy | Regular]

Goal: the repo reflects the sync-and-projection design, the contracts package
exists, and the kernel/product boundary is enforced by structure.

Spec:
- Extract voxscribe-contracts: the Pydantic models for sessions, segments,
  events (the full Phase 0 event vocabulary including policy events), usage
  records, and blob metadata. Logic-free. Version it to the EVENT-CONTRACT
  schema version. voxscribe-server imports it.
- Resolve app/asr.py and app/extraction.py: delete them, or move them under
  app/legacy/ with a module docstring stating they are a pre-kernel fallback,
  off by default, not part of the sync design. Record the decision.
- Restructure app/routers into app/kernel/ (sync, events, blobs, audit, usage,
  tenancy) and app/products/ideascribe/ (scoring, PRDs, projections). Product
  routers import only from app/kernel public interfaces. Add an import-linter
  or a simple test that fails if a product module imports a kernel table
  directly.
- Establish config: Postgres DSN, GCS bucket (or local-fs blob shim for dev),
  auth mode. Keep a single docker-compose that stands up Postgres and the app.

Closes: the boundary, the contracts package, the boilerplate reconciliation.

Acceptance:
- voxscribe-contracts installs and imports in both repos.
- The boundary test fails on a deliberate kernel-table import from a product
  module, passes otherwise.
- make test and make lint green; docker-compose up stands up Postgres and the
  app; GET /health returns ok.
- app/asr.py and app/extraction.py fate is recorded in the changelog.

Model: Sonnet. Structural refactor with clear rules, no deep algorithm design.

---

## Sprint 1: Multi-tenant persistence and auth  [Medium | Regular + IP-adjacent]

Goal: Postgres holds sessions, events, and usage partitioned by tenant with
row-level isolation, and every request is authenticated to a tenant.

Spec:
- Postgres schema mirroring the kernel's event stream: sessions, events,
  segments, usage_records, blob_metadata. PARTITION BY HASH (tenant_id) from
  day one; row-level security scoping every query to the caller's tenant.
- Tenant model: tenant = account (household for SeniorCare, workspace for
  IdeaScribe). Users belong to tenants; a session belongs to one tenant.
- Auth: token-based, tenant-scoped. A request cannot read or write another
  tenant's rows even with a valid token. Start with a simple issued-token
  scheme; leave an IdP seam, do not build it.
- Idempotency keys on write endpoints (sync will retry).

Closes: the tenancy and isolation foundation every later sprint relies on.

Acceptance:
- Cross-tenant read test: tenant A's token cannot read tenant B's session
  (returns not-found, not forbidden, to avoid leaking existence).
- RLS holds even for a raw query missing an explicit tenant filter.
- Idempotent write test: the same write with the same key twice yields one row.
- Schema is hash-partitioned by tenant_id (verified by catalog query).

Model: Sonnet, with Opus for the RLS + partitioning design review only.
Isolation correctness is security-sensitive; the schema decision is worth one
Opus pass, the CRUD is Sonnet.

---

## Sprint 2: Lossless outbox sync (the heart of Phase 1)  [Complex | IP]

Goal: a device drains its outbox to the server with zero loss and full
idempotency, verified by hash, so nothing spoken at home is lost in the cloud.

Spec:
- Sync ingest endpoint: accepts a batch of kernel events, usage records, and
  blob-metadata rows from a device outbox, ordered, with per-item content
  hashes. Server persists in one transaction per batch, returns per-item ack.
- Resumable and idempotent: a device that crashes mid-sync resumes from the
  last acked item; re-sending an acked item is a no-op. The device outbox
  (Phase 0 SQLite) is the source of truth until acked.
- Hash verification: server recomputes each item's hash and rejects a batch
  whose hashes do not match, so corruption is caught, not stored.
- Blob upload: audio blobs (policy-permitted only) upload to GCS under
  tenant-prefixed paths, referenced by the metadata rows. Retention windows
  from the kernel's policy travel with the metadata and are honored server-side.
- Offline-to-online: a device offline for days syncs cleanly when it returns;
  ordering and causal consistency of the event stream are preserved.

Closes: DoD lines "lossless outbox sync verified by hash comparison" and
"idempotent resumable uploads verified." The one place "we test later" converts
directly into silent data loss, so this sprint is fully tested now.

Acceptance:
- Lossless test: generate N events on a fake device, kill mid-sync, resume,
  assert server has exactly N with matching hashes, zero duplicates, zero loss.
- Corruption test: a tampered item's hash mismatch rejects its batch.
- Ordering test: events arrive causally ordered per session after an
  out-of-order, multi-batch, resumed sync.
- Blob test: a permitted blob lands in the tenant-prefixed path; a
  policy-forbidden blob (seniorcare raw audio past retention) is never
  uploaded.

Model: Opus. This is the IP and correctness core of the server; loss or
duplication here is a product-fatal bug in a care setting.

---

## Sprint 3: Audit export and usage/billing rollup  [Medium | Regular + compliance seam]

Goal: the kernel's audit ledger is queryable server-side per tenant, and usage
aggregates into a billing-ready rollup.

Spec:
- Audit projection: the synced safety-ledger and retention/access records
  become a tenant-scoped, queryable audit view supporting the same
  audit-replay shape as the kernel (heard, decided, policy version, rationale)
  for any session moment, now centrally for a compliance reviewer.
- Usage rollup: aggregate the kernel's usage records (finalized words as the
  billing unit, plus cloud-costed actions) into per-tenant, per-period totals.
  This is the metering input to billing; do not integrate a payment processor
  yet, expose the rollup.
- Retention enforcement server-side: honor the per-mode retention windows that
  arrived with the metadata; a scheduled purge marks expired blobs purged
  (row kept, content gone), mirroring the kernel's retention semantics.

Closes: the compliance-audit story assembled centrally; the billing metering
input; server-side retention.

Acceptance:
- Audit-replay parity: server audit-replay for a session matches the kernel's
  local audit-replay for the same session (same decisions, same order).
- Usage rollup test: N finalized-word events aggregate to the expected
  per-tenant total for a period.
- Retention purge test: a blob past its window shows purged server-side, row
  present, GCS object gone.

Model: Sonnet. Projection and aggregation over a known schema; no novel
algorithm.

---

## Sprint 4: IdeaScribe product plane, backend  [Medium | Regular, product-layer]

Goal: IdeaScribe's server-side logic (idea projection, 10-dimension readiness
scoring, gaps and contradictions, PRD generation) runs as a product plane on
the kernel event stream, proving the server end to end with a real vertical.

Spec (all in app/products/ideascribe/, importing only kernel public
interfaces):
- Idea projection: from a session's synced events and clean text, project the
  ideas list (the kernel already extracted structured content on-device; the
  server organizes, scores, and relates it, it does not re-derive from audio).
- Readiness scoring: the 10-dimension rubric, scoring each idea 0 to 100, with
  per-dimension breakdown. Rubric weights live in config, not code, so the
  rubric can evolve without redeploy.
- Gaps, contradictions, orphans: surface unresolved intents and cross-idea
  conflicts from the kernel's event metadata plus a server-side pass. A
  server-side LLM call is permitted here (adult user, non-regulated vertical);
  it is a product-plane call, gated by the tenant's usage/credits, never on the
  kernel hot path.
- PRD generation: assemble a PRD from a sufficiently-ready idea, returning
  structured sections. Credit-metered.
- Sessions, ideas, PRDs, insights read APIs backing the mockup's Home, Ideas,
  Sessions, PRDs, Insights views and the Workspace Pulse metrics.

Closes: the first product plane; end-to-end proof of the server; the
IdeaScribe bridge.

Acceptance:
- Projection test: a synced session yields the expected ideas with provenance
  back to kernel events.
- Scoring test: a fixture idea scores deterministically on the rubric given
  fixed inputs (the rubric math is deterministic; only the LLM-assisted gap
  detection is not, and that is rubric-verified, not byte-diffed).
- Credit test: a PRD generation call debits the tenant's usage rollup; an
  out-of-credit tenant is refused cleanly.
- Boundary test still green: no ideascribe module imports a kernel table.

Model: Sonnet. Product logic over the event stream; the rubric is
deterministic config, the LLM calls are standard.

---

## Sprint 5: IdeaScribe frontend and the demo loop  [Medium | Regular, product-layer]

Goal: the IdeaScribe UI from the mockup runs against the server, closing the
full loop: capture on a device (or upload), sync, project, score, view, PRD.

Spec:
- Frontend implementing the mockup: Home (welcome, start session, quick start,
  recent ideas, workspace pulse), Ideas, Sessions, PRDs, Insights, Governance,
  Settings. Read from the Sprint 4 APIs.
- Capture path: either a device running voxscribe-core in ideascribe mode
  syncing to the server, or a transcript/audio upload path for users without a
  device. Upload path still routes through the kernel for extraction, it does
  not re-extract on the server.
- The readiness pipeline visualization (Raw, Needs discovery, Promising, PRD
  candidate, Delivery-ready) and Workspace Pulse metrics from real data.
- Governance view backed by the audit projection; Settings backed by tenant
  config.

Closes: the demonstrable IdeaScribe product; the server's first shipped client.

Acceptance:
- End-to-end test: capture or upload a session, see ideas scored, advance one
  to PRD, all against the live server.
- The five pipeline stages and the pulse metrics reflect real tenant data.
- Multi-tenant check: two workspaces see only their own ideas and sessions.

Model: Sonnet, with the frontend-design skill for the UI. Standard full-stack
product work.

---

## Sprint 6: SeniorCare server plane (skeleton) and self-host path  [Medium | Regular + compliance]

Goal: the SeniorCare product plane exists as a thin skeleton on the same
kernel plane, and the whole server is self-host deployable for facilities that
require isolation.

Spec:
- SeniorCare plane skeleton in app/products/seniorcare/: wellbeing-timeline
  projection from synced events, and an escalation consumer that turns
  escalation_emitted events into caregiver-facing records. No therapist add-on
  yet (it is gated behind legal review from the manifesto); build the seam,
  not the integration.
- Caregiver read APIs: timeline, alerts, device status, per the deployment
  diagram's caregiver app. No UI required this sprint (SeniorCare UI is
  deferred, you noted you will bridge via IdeaScribe first).
- Self-host packaging: the single docker-compose stands up the full server
  (app plus Postgres plus blob shim) for a facility that demands on-premise,
  matching the manifesto's self-host escape hatch.
- Compliance flags carried, not resolved: PHIPA/HIPAA seams for the future
  therapist add-on, biometric flags for any future voiceprint feature, all
  raised in code comments and the changelog, none activated.

Closes: the SeniorCare seam on the server; the self-host deployment path; the
compliance seams recorded.

Acceptance:
- Timeline test: synced seniorcare events project to a caregiver timeline for
  the tenant.
- Escalation test: an escalation_emitted event becomes a caregiver alert
  record; delivery is a product-layer stub (no real paging yet).
- Self-host test: docker-compose up on a clean machine yields a working server
  with tenant isolation intact.
- No therapist/PHI data path is active; the seam is present and inert.

Model: Sonnet. Skeleton projection plus packaging; no novel algorithm.

---

## Phase 1 exit

Phase 1 is done when: the sync tier is lossless and idempotent (hash-verified);
tenancy isolation holds under adversarial cross-tenant tests; the audit export
matches the kernel's local audit-replay; IdeaScribe runs end to end against the
server as the proof client; the SeniorCare plane skeleton and self-host path
exist; and every compliance seam that requires counsel (therapist add-on, any
biometric feature) is inert and flagged, not shipped.

## Model summary

| Sprint | Model | Why |
|---|---|---|
| 0 Reconcile + contracts | Sonnet | Structural refactor, clear rules |
| 1 Tenancy + auth | Sonnet (Opus for RLS design review) | Isolation is security-sensitive |
| 2 Outbox sync | Opus | Loss/duplication here is product-fatal |
| 3 Audit + usage rollup | Sonnet | Projection over known schema |
| 4 IdeaScribe backend | Sonnet | Product logic, deterministic rubric |
| 5 IdeaScribe frontend | Sonnet | Standard full-stack + design skill |
| 6 SeniorCare skeleton + self-host | Sonnet | Skeleton + packaging |

## Deliberately excluded from Phase 1

- No payment-processor integration (expose the usage rollup; wire Stripe later).
- No therapist add-on (legal-gated; build the seam only).
- No SeniorCare UI (bridge via IdeaScribe first, per founder decision).
- No IdP (leave the auth seam; do not build SSO yet).
- No re-running ASR or extraction on the server (the kernel owns that; the
  server projects, never re-derives).
- No dedicated vector database, no second message broker beyond what sync needs.
