#!/usr/bin/env bash
# VoxScribe Engine console -- retro terminal dashboard with production-grade logging
# Usage: engine-console.sh <start|restart|stop|status|health|logs|help>
set -euo pipefail

# ── config (overridden by env / Makefile) ─────────────────────────────────────
PORT="${ENGINE_PORT:-65364}"
DB="${ENGINE_DB:-postgresql://localhost:5432/ideascribe_dev}"
LOG="${ENGINE_LOG:-/tmp/engine.log}"
TIMEOUT="${ENGINE_READY_TIMEOUT:-30}"
CMD="${1:-help}"
COMPONENT="voxscribe-engine"

# ── ANSI palette (ideascribe brand: #0066CC blue) ─────────────────────────────
# Use $'...' so bash stores the actual ESC character (octal 033), not a literal backslash.
BLUE=$'\033[38;2;0;102;204m'      # #0066CC primary blue -- borders, accents
BLUE_LT=$'\033[38;2;10;132;255m'  # #0A84FF lighter blue -- headings
GREEN=$'\033[38;2;0;122;47m'      # #007A2F success
AMBER=$'\033[38;2;163;71;0m'      # #A34700 warning
RED=$'\033[38;2;204;0;0m'         # #CC0000 error
DIM=$'\033[2m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

# ── Production-grade structured logger ───────────────────────────────────────
# Format: YYYY-MM-DDTHH:MM:SSZ  LEVEL  component  message
_ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

log_info()  { printf "${BLUE}$(_ts)${RESET}  ${BOLD}INFO ${RESET}  ${DIM}${COMPONENT}${RESET}  $*\n"; }
log_warn()  { printf "${AMBER}$(_ts)${RESET}  ${AMBER}${BOLD}WARN ${RESET}  ${DIM}${COMPONENT}${RESET}  $*\n"; }
log_error() { printf "${RED}$(_ts)  ERROR  ${COMPONENT}  $*${RESET}\n" >&2; }
log_fatal() { log_error "$*"; exit 1; }

# ── Box drawing (full terminal width, min 60) ────────────────────────────────
W=$(( $(tput cols 2>/dev/null || echo 80) - 2 ))
[ "$W" -lt 58 ] && W=58

_box_top()    { printf "${BLUE}╭"; printf '─%.0s' $(seq 1 $W); printf "╮${RESET}\n"; }
_box_mid()    { printf "${BLUE}├"; printf '─%.0s' $(seq 1 $W); printf "┤${RESET}\n"; }
_box_bot()    { printf "${BLUE}╰"; printf '─%.0s' $(seq 1 $W); printf "╯${RESET}\n"; }
_box_sep()    { printf "${BLUE}"; printf '─%.0s' $(seq 1 $(( W + 2 ))); printf "${RESET}\n"; }
_box_row() {
  # Print a box row, stripping ANSI escapes for width calculation
  local raw="$1"
  local visible; visible=$(printf "%s" "$raw" | sed 's/\x1b\[[0-9;]*m//g')
  local pad=$(( W - 2 - ${#visible} ))
  [ $pad -lt 0 ] && pad=0
  printf "${BLUE}│${RESET} %s%${pad}s ${BLUE}│${RESET}\n" "$raw" ""
}
_box_blank()  { _box_row ""; }

# ── ASCII art microphone (7 rows, 10 chars wide) ──────────────────────────────
MIC=(
  "    _    "
  "  _| |_  "
  " |     | "
  " |     | "
  "  -| |-  "
  "   | |   "
  "  =====  "
)

# ── Banner ────────────────────────────────────────────────────────────────────
_banner() {
  local subtitle="${1:-}"
  local version
  version=$(cd "$(dirname "$0")/.." && python3 -c \
    "import tomllib,pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(d['project']['version'])" \
    2>/dev/null || echo "")
  local ver_str; [ -n "$version" ] && ver_str=" v${version}" || ver_str=""

  local db_short; db_short=$(printf "%s" "$DB" \
    | sed 's|postgresql://[^@]*@||' | cut -c1-28)

  local -a info=(
    "port    ${BLUE_LT}${PORT}${RESET}"
    "db      ${DIM}${db_short}${RESET}"
    "log     ${DIM}${LOG}${RESET}"
    ""
    "${DIM}${subtitle}${RESET}"
  )

  echo ""
  _box_top
  _box_row "${BOLD}${BLUE_LT}  VoxScribe Engine${ver_str}${RESET}"
  _box_mid

  local col1=11
  local nrows=${#MIC[@]}
  for i in $(seq 0 $(( nrows - 1 ))); do
    local mic="${MIC[$i]}"
    local inf="${info[$i]:-}"
    local inf_vis; inf_vis=$(printf "%s" "$inf" | sed 's/\x1b\[[0-9;]*m//g')
    local pad=$(( W - col1 - 3 - ${#inf_vis} ))
    [ $pad -lt 0 ] && pad=0
    printf "${BLUE}│${RESET} ${BLUE}%s${RESET}  %s%${pad}s ${BLUE}│${RESET}\n" \
      "$mic" "$inf" ""
  done

  _box_blank
  _box_bot
  echo ""
}

# ── Health helpers ────────────────────────────────────────────────────────────
_fetch_health() {
  curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/health" 2>/dev/null
}

_parse_health() {
  local json="$1" key="$2"
  python3 -c "import sys,json; d=json.loads('''${json}'''); print($key)" 2>/dev/null || echo "?"
}

_print_health() {
  local json="$1"
  local status asr_ok asr_name llm_ok db_ok schema

  status=$(echo "$json"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")
  asr_ok=$(echo "$json"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('providers',{}).get('asr',{}).get('ok',False))" 2>/dev/null || echo "False")
  asr_name=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('providers',{}).get('asr',{}).get('name','?'))" 2>/dev/null || echo "?")
  llm_ok=$(echo "$json"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('providers',{}).get('llm',{}).get('ok',False))" 2>/dev/null || echo "False")
  db_ok=$(echo "$json"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('database',{}).get('ok',False))" 2>/dev/null || echo "False")
  schema=$(echo "$json"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('database',{}).get('schemaVersion','?'))" 2>/dev/null || echo "?")

  local status_col; [ "$status" = "ok" ] && status_col="${GREEN}" || status_col="${RED}"
  local asr_col;    [ "$asr_ok"  = "True" ] && asr_col="${GREEN}"  || asr_col="${RED}"
  local llm_col;    [ "$llm_ok"  = "True" ] && llm_col="${GREEN}"  || llm_col="${RED}"
  local db_col;     [ "$db_ok"   = "True" ] && db_col="${GREEN}"   || db_col="${RED}"

  _box_mid
  _box_row "  ${BOLD}Health${RESET}"
  _box_row "  status    ${status_col}${status}${RESET}"
  _box_row "  asr       ${asr_col}$([ "$asr_ok" = "True" ] && echo ok || echo down)${RESET}  ${DIM}${asr_name}${RESET}"
  _box_row "  llm       ${llm_col}$([ "$llm_ok" = "True" ] && echo ok || echo down)${RESET}"
  _box_row "  database  ${db_col}$([ "$db_ok" = "True" ] && echo ok || echo down)${RESET}  ${DIM}schema=${schema}${RESET}"
  _box_bot
  echo ""
}

# ── PID helpers ───────────────────────────────────────────────────────────────
_engine_pid() { lsof -ti ":${PORT}" 2>/dev/null | head -1 || true; }

_kill_engine() {
  local pid; pid=$(_engine_pid)
  if [ -n "$pid" ]; then
    kill -9 "$pid" 2>/dev/null || true
    log_info "stopped pid=${pid}"
  fi
}

# ── Wait for healthy ──────────────────────────────────────────────────────────
_wait_healthy() {
  log_info "waiting for health endpoint on :${PORT} (timeout=${TIMEOUT}s)"
  for i in $(seq 1 "$TIMEOUT"); do
    if _fetch_health >/dev/null 2>&1; then
      log_info "health check passed after ${i}s"
      return 0
    fi
    sleep 1
  done
  log_error "health check timed out after ${TIMEOUT}s"
  log_warn  "inspect log: tail -f ${LOG}"
  return 1
}

# ── Tail with detach message ──────────────────────────────────────────────────
_tail_log() {
  log_info "tailing ${LOG}  (Ctrl-C detaches, engine keeps running)"
  _box_sep
  trap 'printf "\n"; log_info "detached -- engine still running on :${PORT}\n"' INT
  tail -f "$LOG"
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_start() {
  _banner "starting"
  if [ -z "${VOXSCRIBE_TOKEN:-}" ]; then
    log_fatal "VOXSCRIBE_TOKEN is not set -- add it to .env"
  fi
  log_info "killing any existing process on :${PORT}"
  _kill_engine
  log_info "launching engine on :${PORT} db=${DB}"
  VOXSCRIBE_TOKEN="$VOXSCRIBE_TOKEN" \
    uv run python3 -m app.local.engine \
      --port "$PORT" \
      --database-url "$DB" \
    > "$LOG" 2>&1 &
  _wait_healthy || exit 1
  local json; json=$(_fetch_health)
  _box_top
  _print_health "$json"
  _tail_log
}

cmd_restart() {
  _banner "restarting"
  if [ -z "${VOXSCRIBE_TOKEN:-}" ]; then
    log_fatal "VOXSCRIBE_TOKEN is not set -- add it to .env"
  fi
  log_info "stopping current engine"
  _kill_engine
  sleep 1
  log_info "launching engine on :${PORT}"
  VOXSCRIBE_TOKEN="$VOXSCRIBE_TOKEN" \
    uv run python3 -m app.local.engine \
      --port "$PORT" \
      --database-url "$DB" \
    > "$LOG" 2>&1 &
  _wait_healthy || exit 1
  local json; json=$(_fetch_health)
  _box_top
  _print_health "$json"
  _tail_log
}

cmd_stop() {
  echo ""
  log_info "stopping engine on :${PORT}"
  _kill_engine
  log_info "done"
  echo ""
}

cmd_status() {
  echo ""
  local pid; pid=$(_engine_pid)
  _box_top
  if [ -n "$pid" ]; then
    _box_row "  ${GREEN}running${RESET}  pid=${BLUE_LT}${pid}${RESET}  port=${BLUE_LT}${PORT}${RESET}"
    local json; json=$(_fetch_health 2>/dev/null || true)
    if [ -n "$json" ]; then
      _print_health "$json"
    else
      _box_row "  ${RED}health endpoint not responding${RESET}"
      _box_bot
      echo ""
    fi
  else
    _box_row "  ${RED}not running${RESET}  port=${PORT}"
    _box_bot
    echo ""
  fi
}

cmd_health() {
  echo ""
  local json; json=$(_fetch_health 2>/dev/null || true)
  if [ -z "$json" ]; then
    _box_top
    _box_row "  ${RED}engine not responding on :${PORT}${RESET}"
    _box_bot
    echo ""
    exit 1
  fi
  _box_top
  _print_health "$json"
}

cmd_logs() {
  _tail_log
}

cmd_help() {
  echo ""
  _box_top
  _box_row "  ${BOLD}${BLUE_LT}VoxScribe Engine${RESET}  management commands"
  _box_mid
  _box_row "  ${BLUE}make engine${RESET}          start engine, then tail log"
  _box_row "  ${BLUE}make engine-restart${RESET}  kill, restart, then tail log"
  _box_row "  ${BLUE}make engine-stop${RESET}     stop the engine"
  _box_row "  ${BLUE}make engine-status${RESET}   running state and health summary"
  _box_row "  ${BLUE}make engine-health${RESET}   GET /v1/health and render result"
  _box_row "  ${BLUE}make engine-logs${RESET}     tail the engine log file"
  _box_blank
  _box_row "  ${DIM}ENGINE_PORT=${PORT}   ENGINE_LOG=${LOG}${RESET}"
  _box_row "  ${DIM}VOXSCRIBE_TOKEN loaded from .env automatically${RESET}"
  _box_bot
  echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$CMD" in
  start)   cmd_start   ;;
  restart) cmd_restart ;;
  stop)    cmd_stop    ;;
  status)  cmd_status  ;;
  health)  cmd_health  ;;
  logs)    cmd_logs    ;;
  help|*)  cmd_help    ;;
esac
