#!/usr/bin/env bash
# Margaret SC baseline — P1 exit gate (C10).
# Plays sessions 1-8 through the real speaker/mic in seniorcare mode.
# Sessions 9-10 are held out and never run here.
#
# Usage: bash tools/margaret_baseline.sh
# Requires: Ollama running, engine not already running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${TMPDIR:-/tmp}/margaret_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== MARGARET SC BASELINE ==="
echo "log dir: $LOG_DIR"
echo ""

# ── start engine in seniorcare mode ──────────────────────────────────────────
echo "[0] starting engine --wake --no-auth seniorcare mode..."
cd "$ROOT"
VOXSCRIBE_CAPTURE_DEVICE="MacBook Pro Microphone" \
VOXSCRIBE_PLAYBACK_DEVICE="MacBook Pro Speakers" \
VOXSCRIBE_VOICE_MODE="seniorcare" \
VOXSCRIBE_VOICE_REPLY_MODEL="llama3.2" \
  uv run python -m app.local.engine --wake --no-auth --port 0 \
  > "$LOG_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
trap "echo 'stopping engine...'; kill $ENGINE_PID 2>/dev/null; wait $ENGINE_PID 2>/dev/null || true" EXIT

WAITED=0
ENGINE_PORT=""
until [ -n "$ENGINE_PORT" ]; do
  ENGINE_PORT=$(grep -o '"port": [0-9]*' "$LOG_DIR/engine.log" 2>/dev/null | grep -o '[0-9]*' || true)
  sleep 0.5
  WAITED=$((WAITED + 1))
  if [ $WAITED -gt 60 ]; then
    echo "ERROR: engine did not start within 30s"
    cat "$LOG_DIR/engine.log"
    exit 1
  fi
done
ENGINE_URL="http://127.0.0.1:${ENGINE_PORT}"
echo "    engine ready on $ENGINE_URL"
echo ""

# ── wait for pre-warm ─────────────────────────────────────────────────────────
echo "[1] waiting for LLM + Kokoro pre-warm (up to 30s)..."
sleep 5
if grep -q "reply model warmed\|Kokoro TTS warmed" "$LOG_DIR/engine.log" 2>/dev/null; then
  echo "    pre-warm confirmed"
else
  echo "    pre-warm not confirmed in log, continuing..."
fi
echo ""

# ── run baseline harness ─────────────────────────────────────────────────────
echo "[2] running Margaret baseline (sessions 1-8)..."
REPORT="/tmp/sc_round_01.json"
uv run tools/margaret_baseline.py \
  --engine-url "$ENGINE_URL" \
  --sessions 1-8 \
  --out "$REPORT" 2>&1 | tee "$LOG_DIR/baseline.log"

echo ""
echo "=== DONE ==="
echo "report: $REPORT"
echo "engine log: $LOG_DIR/engine.log"
echo "baseline log: $LOG_DIR/baseline.log"
echo ""

# Print summary
if [ -f "$REPORT" ]; then
  python3 -c "
import json, sys
with open('$REPORT') as f:
    r = json.load(f)
print(f'C10 baseline: {r[\"c10_score\"]}')
print(f'Sessions run: {r[\"sessions_run\"]}')
print()
for sid, res in r['session_results'].items():
    passed = [k for k,v in res.items() if v=='pass' and k.startswith('SC-')]
    failed = [k for k,v in res.items() if v=='fail' and k.startswith('SC-')]
    turns = res.get('_turns','?')
    print(f'  session_{sid}: turns={turns} pass={passed} fail={failed}')
"
fi
