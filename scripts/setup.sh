#!/usr/bin/env bash
# setup.sh — scaffold the VoxScribe server git repo and development environment.
#
# Usage (from the project root):
#   ./scripts/setup.sh              standard setup
#   ./scripts/setup.sh --commit     setup and make the initial git commit
#   ./scripts/setup.sh --no-venv    skip Python venv creation
#
# Run it as many times as you like. It is idempotent: it creates what is
# missing and skips what already exists.

set -euo pipefail

# ---- colours ----------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[info]${RESET}  $*"; }
ok()      { echo -e "${GREEN}${BOLD}[ ok ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[warn]${RESET}  $*"; }
err()     { echo -e "${RED}${BOLD}[err ]${RESET}  $*" >&2; }
section() { echo -e "\n${BOLD}=== $* ===${RESET}"; }

die() { err "$*"; exit 1; }

# ---- argument parsing -------------------------------------------------------
DO_COMMIT=false
DO_VENV=true
for arg in "$@"; do
  case "$arg" in
    --commit)   DO_COMMIT=true ;;
    --no-venv)  DO_VENV=false ;;
    -h|--help)
      echo "Usage: $0 [--commit] [--no-venv]"
      echo "  --commit    create the initial git commit after setup"
      echo "  --no-venv   skip Python virtualenv creation"
      exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---- must be run from the project root --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "app/main.py" ]]; then
  die "Run this script from the voxscribe-server project root (app/main.py not found)."
fi

echo -e "${BOLD}VoxScribe server — project setup${RESET}"
echo    "  project root: $PROJECT_ROOT"

# ---- prerequisites ----------------------------------------------------------
section "Checking prerequisites"

check_cmd() {
  if command -v "$1" &>/dev/null; then ok "$1 found: $(command -v "$1")"; else warn "$1 not found — $2"; fi
}

check_cmd git   "install Xcode CLT: xcode-select --install"
check_cmd python3 "install Python 3.10+: https://python.org"
check_cmd docker  "install Docker Desktop: https://docker.com"

PYTHON_OK=false
if command -v python3 &>/dev/null; then
  PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
  PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
  if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
    ok "Python $PY_VER"
    PYTHON_OK=true
  else
    warn "Python $PY_VER found but 3.10+ is required"
  fi
fi

# ---- directory structure ----------------------------------------------------
section "Creating directory structure"

DIRS=(config scripts logs data data/sessions app app/routers)
for d in "${DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    mkdir -p "$d"
    ok "created $d/"
  else
    info "$d/ already exists"
  fi
done

# ---- .gitignore -------------------------------------------------------------
section "Writing .gitignore"

GITIGNORE=".gitignore"
if [[ ! -f "$GITIGNORE" ]]; then
  cat > "$GITIGNORE" << 'IGNORE'
# secrets and local environment — never commit
.env

# runtime data
data/
logs/
*.log
*.db
*.sqlite

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
env/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# macOS
.DS_Store
.AppleDouble

# editors
.idea/
.vscode/
*.swp
*.swo
*~

# Docker
.docker/
IGNORE
  ok ".gitignore created"
else
  info ".gitignore already exists, skipping"
fi

# ---- git initialisation -----------------------------------------------------
section "Initialising git repository"

if [[ ! -d ".git" ]]; then
  git init -b main
  ok "git repository initialised (branch: main)"
else
  info "git repository already exists"
fi

# ---- .env -------------------------------------------------------------------
section "Creating .env"

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    ok ".env copied from .env.example"
  else
    # Write a minimal .env inline if .env.example is also missing.
    cat > ".env" << 'DOTENV'
QUEUE_BACKEND=memory
WORKER_CONCURRENCY=2
QUEUE_MAX_ATTEMPTS=3
QUEUE_BACKOFF=2.0
RECORDING_TTL_SECONDS=3600
EXTRACTOR=simulated
ASR_BACKEND=stub
CLEANUP=off
OLLAMA_HOST=http://host.docker.internal:11434
VOXSCRIBE_OLLAMA_MODEL=gemma3:4b
CLAUDE_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
IDEASCRIBE_TENANT_ID=tnt-acme
IDEASCRIBE_WORKSPACE_ID=ws-discovery
IDEASCRIBE_ID_FLOOR=53
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_FILE=./logs/voxscribe.log
DOTENV
    ok ".env created from inline template"
  fi
  echo ""
  warn "IMPORTANT: open .env and fill in secrets before running the service."
  warn "  ANTHROPIC_API_KEY  — required for EXTRACTOR=claude"
  warn "  OPENAI_API_KEY     — required for ASR_BACKEND=openai"
else
  info ".env already exists, skipping"
fi

# ---- config/config.yaml -----------------------------------------------------
section "Creating config/config.yaml"

if [[ ! -f "config/config.yaml" ]]; then
  cat > "config/config.yaml" << 'YAML'
server:
  host: "0.0.0.0"
  port: 8000

queue:
  backend: "memory"
  worker_concurrency: 2
  max_attempts: 3
  base_backoff: 2.0
  recording_ttl_seconds: 3600

processing:
  extractor: "simulated"
  asr_backend: "stub"
  cleanup: false

models:
  ollama:
    host: "http://host.docker.internal:11434"
    model: "gemma3:4b"
  claude:
    model: "claude-sonnet-4-6"
    api_key: ""
  openai:
    api_key: ""
    whisper_model: "base"

storage:
  data_dir: "./data"
  db_path: "./data/ideascribe.db"
  id_floor: 53

auth:
  default_tenant: "tnt-acme"
  default_workspace: "ws-discovery"

cors:
  origins:
    - "http://localhost:3000"
    - "http://localhost:8080"

logging:
  level: "INFO"
  format: "text"
  log_file: "./logs/voxscribe.log"
  max_bytes: 10485760
  backup_count: 5
YAML
  ok "config/config.yaml created"
else
  info "config/config.yaml already exists, skipping"
fi

# ---- Python virtual environment ---------------------------------------------
section "Python virtual environment"

if [[ "$DO_VENV" == true && "$PYTHON_OK" == true ]]; then
  if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    ok ".venv created"
  else
    info ".venv already exists"
  fi

  info "Installing requirements..."
  .venv/bin/pip install --quiet --upgrade pip
  if [[ -f "requirements.txt" ]]; then
    .venv/bin/pip install --quiet -r requirements.txt
    ok "requirements installed"
  else
    warn "requirements.txt not found, skipping install"
  fi

  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
  if [[ "$DO_VENV" == false ]]; then info "virtualenv skipped (--no-venv)"; fi
fi

# ---- validate imports -------------------------------------------------------
section "Validating application"

if [[ "$PYTHON_OK" == true ]]; then
  VALIDATE_OK=true
  # Load env for the validation run.
  set -a
  # shellcheck disable=SC1091
  [[ -f ".env" ]] && source .env
  set +a

  # Temp data dir for the import check so it does not touch ./data.
  export DATA_DIR=/tmp/voxscribe-setup-check
  export DB_PATH=/tmp/voxscribe-setup-check/check.db
  export LOG_FILE=/tmp/voxscribe-setup-check/check.log
  mkdir -p /tmp/voxscribe-setup-check

  if $PYTHON_BIN -c "from app.main import app; print('app import ok')" 2>&1 | grep -q "app import ok"; then
    ok "app.main imports successfully"
  else
    warn "app import failed — check your Python environment and dependencies"
    VALIDATE_OK=false
  fi

  if $PYTHON_BIN -c "from app.config import settings; assert settings.queue.backend" 2>/dev/null; then
    ok "config loaded: extractor=${EXTRACTOR:-simulated} asr=${ASR_BACKEND:-stub}"
  else
    warn "config validation failed"
    VALIDATE_OK=false
  fi

  rm -rf /tmp/voxscribe-setup-check
fi

# ---- initial git commit -----------------------------------------------------
if [[ "$DO_COMMIT" == true ]]; then
  section "Initial git commit"
  git add -A
  # .env is gitignored, so it is not staged. Confirm.
  if git diff --cached --name-only | grep -q "\.env$"; then
    die ".env was staged for commit — check .gitignore"
  fi
  git commit -m "chore: initial voxscribe-server scaffold

  - FastAPI backend with queue-based session processing
  - InProcessJobQueue with retry, dead-letter, and crash recovery
  - Per-device recording lock (one active capture per device)
  - Structured logging with session ContextVar trace
  - config/config.yaml + .env for configuration and secrets
  - Docker + Makefile for local dev and deployment"
  ok "initial commit created"
fi

# ---- summary ----------------------------------------------------------------
section "Setup complete"

echo ""
echo -e "  ${BOLD}Start the service:${RESET}"
if command -v docker &>/dev/null; then
  echo -e "    ${CYAN}make up${RESET}         build Docker image and start on :8000"
fi
echo -e "    ${CYAN}make dev${RESET}        local uvicorn with hot reload (no Docker)"
echo ""
echo -e "  ${BOLD}Check it:${RESET}"
echo -e "    ${CYAN}make health${RESET}     GET /health"
echo -e "    ${CYAN}make smoke${RESET}      upload a transcript and follow SSE progress"
echo ""
echo -e "  ${BOLD}Config:${RESET}"
echo -e "    Edit ${BOLD}.env${RESET} to set secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY)"
echo -e "    Edit ${BOLD}config/config.yaml${RESET} for non-secret settings"
echo -e "    Set ${BOLD}EXTRACTOR=claude${RESET} for real LLM extraction"
echo -e "    Set ${BOLD}ASR_BACKEND=faster_whisper${RESET} for local real ASR"
echo ""
echo -e "  ${BOLD}Logs:${RESET}  tail -f logs/voxscribe.log"
echo ""
