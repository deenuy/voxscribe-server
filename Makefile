# VoxScribe processing backend. `make up` builds the image and starts the service.

COMPOSE := docker compose
BASE := http://localhost:8000
TENANT := tnt-acme
WORKSPACE := ws-discovery

ENGINE_PORT          ?= 65364
ENGINE_DB            ?= postgresql://localhost:5432/ideascribe_dev
ENGINE_LOG           ?= /tmp/engine.log
ENGINE_READY_TIMEOUT ?= 30

WAKE_LOG             ?= /tmp/engine_wake.log

# Load .env if present so VOXSCRIBE_TOKEN does not need to be passed manually.
-include .env
export

.DEFAULT_GOAL := help

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

.PHONY: env
env: ## Create .env from the template if missing
	@test -f .env || cp .env.example .env
	@echo ".env ready"

.PHONY: build
build: env ## Build the Docker image
	$(COMPOSE) build

.PHONY: up
up: env ## Build and start the service in the background (the main target)
	$(COMPOSE) up -d --build
	@echo "API on $(BASE)  (docs at $(BASE)/docs)"

.PHONY: down
down: ## Stop and remove the container
	$(COMPOSE) down

.PHONY: restart
restart: down up ## Restart the service

.PHONY: logs
logs: ## Tail service logs
	$(COMPOSE) logs -f

.PHONY: health
health: ## Check the health endpoint
	@curl -fsS $(BASE)/health && echo

.PHONY: dev
dev: ## Run locally with hot reload, no Docker
	pip install -r requirements.txt
	uvicorn app.main:app --reload --port 8000

.PHONY: smoke
smoke: ## End to end check: upload a transcript and follow progress
	@echo "we should build voice capture. actually the real problem is idea readiness." > /tmp/vs_smoke.txt
	@SID=$$(curl -fsS -X POST "$(BASE)/v1/sessions/from-file?title=Smoke&participants=2" \
	  -H "X-Tenant-Id: $(TENANT)" -H "X-Workspace-Id: $(WORKSPACE)" \
	  -F file=@/tmp/vs_smoke.txt | python3 -c "import sys,json;print(json.load(sys.stdin)['sessionId'])"); \
	  echo "session $$SID"; \
	  curl -fsS -N "$(BASE)/v1/sessions/$$SID/progress" \
	  -H "X-Tenant-Id: $(TENANT)" -H "X-Workspace-Id: $(WORKSPACE)"

CONSOLE := scripts/engine-console.sh

.PHONY: engine
engine: ## Start the engine; waits for health then tails the log
	@chmod +x $(CONSOLE)
	@ENGINE_PORT=$(ENGINE_PORT) ENGINE_DB=$(ENGINE_DB) ENGINE_LOG=$(ENGINE_LOG) \
	  ENGINE_READY_TIMEOUT=$(ENGINE_READY_TIMEOUT) \
	  bash $(CONSOLE) start

.PHONY: engine-restart
engine-restart: ## Kill, restart the engine, then tail the log
	@chmod +x $(CONSOLE)
	@ENGINE_PORT=$(ENGINE_PORT) ENGINE_DB=$(ENGINE_DB) ENGINE_LOG=$(ENGINE_LOG) \
	  ENGINE_READY_TIMEOUT=$(ENGINE_READY_TIMEOUT) \
	  bash $(CONSOLE) restart

.PHONY: engine-stop
engine-stop: ## Stop the local engine
	@chmod +x $(CONSOLE)
	@ENGINE_PORT=$(ENGINE_PORT) ENGINE_DB=$(ENGINE_DB) ENGINE_LOG=$(ENGINE_LOG) \
	  bash $(CONSOLE) stop

.PHONY: engine-status
engine-status: ## Show engine running state and health summary
	@chmod +x $(CONSOLE)
	@ENGINE_PORT=$(ENGINE_PORT) ENGINE_DB=$(ENGINE_DB) ENGINE_LOG=$(ENGINE_LOG) \
	  bash $(CONSOLE) status

.PHONY: engine-health
engine-health: ## GET /v1/health and render the result
	@chmod +x $(CONSOLE)
	@ENGINE_PORT=$(ENGINE_PORT) ENGINE_DB=$(ENGINE_DB) ENGINE_LOG=$(ENGINE_LOG) \
	  bash $(CONSOLE) health

.PHONY: engine-logs
engine-logs: ## Tail the engine log
	@chmod +x $(CONSOLE)
	@ENGINE_PORT=$(ENGINE_PORT) ENGINE_LOG=$(ENGINE_LOG) \
	  bash $(CONSOLE) logs

.PHONY: wake
wake: ## Kill any engine, start in wake+default mode, wait for "reply model warmed"
	@echo "Stopping any running wake engine..."
	@pkill -f "app.local.engine.*--wake" 2>/dev/null || true
	@sleep 1
	@echo "Starting wake engine (default mode, no-auth)..."
	@uv run python3 -m app.local.engine --wake --no-auth --port 0 \
	  > $(WAKE_LOG) 2>&1 &
	@echo "Waiting for reply model warm..."
	@for i in $$(seq 1 30); do \
	  sleep 2; \
	  grep -q "reply model warmed" $(WAKE_LOG) 2>/dev/null && break; \
	  grep -q "Traceback\|startup failed\|Address already in use" $(WAKE_LOG) 2>/dev/null && \
	    { echo "Engine failed — check $(WAKE_LOG)"; exit 1; }; \
	done
	@PORT=$$(grep '"port":' $(WAKE_LOG) 2>/dev/null | python3 -c \
	  "import sys,json,re; m=re.search(r'\"port\": ?(\d+)', sys.stdin.read()); print(m.group(1) if m else '?')"); \
	  echo ""; \
	  echo "  Engine ready on port $$PORT  (log: $(WAKE_LOG))"; \
	  echo "  Trigger: curl -s -X POST http://127.0.0.1:$$PORT/v1/wake/trigger"; \
	  echo ""

.PHONY: wake-trigger
wake-trigger: ## Trigger one session on the running wake engine (speak immediately after)
	@PORT=$$(grep '"port":' $(WAKE_LOG) 2>/dev/null | python3 -c \
	  "import sys,json,re; m=re.search(r'\"port\": ?(\d+)', sys.stdin.read()); print(m.group(1) if m else '')"); \
	  [ -z "$$PORT" ] && { echo "No wake engine found — run: make wake"; exit 1; }; \
	  echo "Triggering session on port $$PORT — speak now"; \
	  curl -s -X POST http://127.0.0.1:$$PORT/v1/wake/trigger

.PHONY: wake-test6
wake-test6: ## Trigger 6 sessions sequentially — speak after each TRIGGERING line
	@PORT=$$(grep '"port":' $(WAKE_LOG) 2>/dev/null | python3 -c \
	  "import sys,json,re; m=re.search(r'\"port\": ?(\d+)', sys.stdin.read()); print(m.group(1) if m else '')"); \
	  [ -z "$$PORT" ] && { echo "No wake engine found — run: make wake"; exit 1; }; \
	  ENGINE=http://127.0.0.1:$$PORT; \
	  for i in 1 2 3 4 5 6; do \
	    echo "$$(date +%H:%M:%S) session $$i: waiting for ARMED..."; \
	    for _ in $$(seq 1 30); do \
	      STATE=$$(curl -s $$ENGINE/v1/wake/state | python3 -c \
	        "import sys,json; print(json.load(sys.stdin).get('state',''))" 2>/dev/null); \
	      [ "$$STATE" = "armed" ] && break; sleep 2; \
	    done; \
	    echo "$$(date +%H:%M:%S) session $$i: TRIGGERING — speak now"; \
	    curl -s -X POST $$ENGINE/v1/wake/trigger > /dev/null; \
	    sleep 3; \
	    for _ in $$(seq 1 60); do \
	      STATE=$$(curl -s $$ENGINE/v1/wake/state | python3 -c \
	        "import sys,json; print(json.load(sys.stdin).get('state',''))" 2>/dev/null); \
	      [ "$$STATE" = "armed" ] && break; sleep 2; \
	    done; \
	    echo "$$(date +%H:%M:%S) session $$i: done"; \
	    sleep 1; \
	  done; \
	  echo "All 6 done."

.PHONY: wake-logs
wake-logs: ## Tail the wake engine log
	@tail -f $(WAKE_LOG)

# ── SearXNG ────────────────────────────────────────────────────────────────────
SEARXNG_DIR     ?= $(HOME)/services/searxng
SEARXNG_PLIST   ?= $(HOME)/Library/LaunchAgents/com.voxscribe.searxng.plist
SEARXNG_LABEL   ?= com.voxscribe.searxng
SEARXNG_LOG     ?= /tmp/searxng.log
SEARXNG_SETTINGS ?= $(SEARXNG_DIR)/etc/settings.yml

.PHONY: searxng-install
searxng-install: ## Clone and install SearXNG into ~/services/searxng with its own venv
	@if [ -d "$(SEARXNG_DIR)/.git" ]; then \
	  echo "SearXNG already cloned at $(SEARXNG_DIR)"; \
	else \
	  mkdir -p $(HOME)/services && git clone https://github.com/searxng/searxng.git $(SEARXNG_DIR); \
	fi
	@if [ ! -d "$(SEARXNG_DIR)/.venv" ]; then \
	  echo "Creating venv..."; \
	  python3 -m venv $(SEARXNG_DIR)/.venv; \
	fi
	@$(SEARXNG_DIR)/.venv/bin/pip install -r $(SEARXNG_DIR)/requirements.txt --quiet
	@$(SEARXNG_DIR)/.venv/bin/pip install granian --quiet
	@mkdir -p $(SEARXNG_DIR)/etc
	@echo "SearXNG installed. Run: make searxng-start"

.PHONY: searxng-start
searxng-start: ## Load the SearXNG launchd agent (starts now and on login)
	@launchctl load -w $(SEARXNG_PLIST) 2>/dev/null || launchctl load $(SEARXNG_PLIST)
	@sleep 2
	@curl -s http://127.0.0.1:8080/healthz > /dev/null && echo "SearXNG: reachable" || echo "SearXNG: not yet up (check $(SEARXNG_LOG))"

.PHONY: searxng-stop
searxng-stop: ## Unload the SearXNG launchd agent
	@launchctl unload $(SEARXNG_PLIST) 2>/dev/null && echo "SearXNG: stopped" || echo "SearXNG: was not loaded"

.PHONY: searxng-status
searxng-status: ## Check if SearXNG is reachable
	@curl -s http://127.0.0.1:8080/healthz > /dev/null && echo "SearXNG: reachable" || echo "SearXNG: down"
	@launchctl list | grep $(SEARXNG_LABEL) || echo "(launchd job not loaded)"

.PHONY: local
local: ## Run the local engine with voxscribe-engine
	@echo "Use: uv run voxscribe-engine --port 0 --providers fake"

.PHONY: test
test: ## Run tests with pytest
	uv run pytest tests/ -v

.PHONY: lint
lint: ## Lint with ruff
	uv run ruff check app/

.PHONY: db-migrate
db-migrate: ## Run pending database migrations
	@echo "Migrations run at startup automatically"
	@echo "To test: VOXSCRIBE_DATABASE_URL=postgresql://localhost/ideascribe_dev uv run python -c 'from app.local.migrations import run_migrations; import asyncio; asyncio.run(run_migrations(\"postgresql://localhost/ideascribe_dev\"))'"

.PHONY: db-status
db-status: ## Show database status and migration version
	@psql $${VOXSCRIBE_DATABASE_URL:-postgresql://localhost/ideascribe_dev} -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;" || echo "No migrations applied"

.PHONY: db-reset
db-reset: ## Drop and recreate the engine schema (requires CONFIRM=yes)
	@test "$$CONFIRM" = "yes" || (echo "DANGER: This will delete all engine data."; echo "Run with: make db-reset CONFIRM=yes"; exit 1)
	@echo "Dropping engine schema..."
	@psql $${VOXSCRIBE_DATABASE_URL:-postgresql://localhost/ideascribe_dev} -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"

.PHONY: clean
clean: ## Remove the local data directory (sessions + sqlite db)
	rm -rf ./data

# ---------------------------------------------------------------------------
# Core wheel management (rebuild from source, switch between pinned/dev)
# ---------------------------------------------------------------------------

CORE_DIR ?= ../core-pkg
CORE_TAG ?= v0.1.1

.PHONY: core-wheel
core-wheel: ## Rebuild the pinned wheel from core repo git tag
	@echo "Building voxscribe-core from tag $(CORE_TAG)..."
	@TMPDIR=$$(mktemp -d); \
	  git -C $(CORE_DIR) archive $(CORE_TAG) | tar -x -C $$TMPDIR; \
	  cd $$TMPDIR && \
	  python3.13 -m venv .venv && \
	  .venv/bin/python -m pip install -q build && \
	  .venv/bin/python -m build --wheel && \
	  WHL=$$(ls -1 dist/voxscribe_core-*.whl | tail -1); \
	  WHL_BASE=$$(basename $$WHL); \
	  cd - > /dev/null; \
	  rm -f wheels/voxscribe_core-*.whl; \
	  cp $$TMPDIR/dist/$$WHL_BASE wheels/; \
	  WHL_HASH=$$(shasum -a 256 wheels/$$WHL_BASE | cut -d' ' -f1); \
	  echo "Wheel ready: wheels/$$WHL_BASE"; \
	  echo "SHA256: $$WHL_HASH"; \
	  echo "Content changed: config version, CHANGELOG style, resource path docs."; \
	  echo "Next: git add wheels/ && git commit && uv lock"

.PHONY: core-dev
core-dev: ## Switch to editable development source of voxscribe-core (sibling repo)
	uv pip install --python .venv/bin/python -e $(CORE_DIR)
	@echo "Switched to editable voxscribe-core from $(CORE_DIR)"
	@echo "Run: uv sync to update dependencies"

.PHONY: core-pinned
core-pinned: ## Switch back to the pinned wheel version
	uv sync --frozen --reinstall-package voxscribe-core
	@echo "Switched back to pinned wheel from wheels/"
