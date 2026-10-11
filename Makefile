# VoxScribe processing backend. `make up` builds the image and starts the service.

COMPOSE := docker compose
BASE := http://localhost:8000
TENANT := tnt-acme
WORKSPACE := ws-discovery

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
