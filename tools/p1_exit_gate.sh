#!/usr/bin/env bash
# P1 exit gate: automated roomplay runs via speaker→mic.
# Near-field self-playback; all scores tagged 70% (synthetic weight).
#
# Usage:
#   bash tools/p1_exit_gate.sh
#
# Requires the engine to NOT be running (this script starts it).
# Ctrl-C kills everything when done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="${TMPDIR:-/tmp}/p1_gate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

echo "=== P1 EXIT GATE ==="
echo "report dir: $REPORT_DIR"
echo ""

# -- start engine in wake mode in background --------------------------------
echo "[1/4] starting engine --wake ..."
cd "$ROOT"
uv run python -m app.local.engine --wake --no-auth \
  > "$REPORT_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
trap "echo 'stopping engine...'; kill $ENGINE_PID 2>/dev/null; wait $ENGINE_PID 2>/dev/null; echo done" EXIT

# wait for engine ready signal
WAITED=0
until grep -q '"event": "ready"' "$REPORT_DIR/engine.log" 2>/dev/null; do
  sleep 0.5
  WAITED=$((WAITED + 1))
  if [ $WAITED -gt 40 ]; then
    echo "ERROR: engine did not start within 20s"
    cat "$REPORT_DIR/engine.log"
    exit 1
  fi
done
echo "    engine ready ($(( WAITED / 2 ))s)"
echo ""

# -- brief warm-up: give wake loop time to arm --------------------------------
sleep 2.0

# -- run 1: five pause lengths (C3 check, no stop) --------------------------
echo "[2/4] RUN 1 -- five pause lengths, no stop ..."
echo "      utterance: 'hey jarvis [PAUSE:N] tell me something interesting'"
echo "      pauses: 1.5 3 5 8 11 seconds"
uv run tools/roomplay.py \
  --utterance "hey jarvis [PAUSE:N] tell me something interesting" \
  --pause-sweep 1.5,3,5,8,11 \
  --label c3_sweep \
  --report "$REPORT_DIR/run1_c3_sweep.json" || echo "WARNING: run1 had errors"

echo ""
sleep 3.0

# -- run 2: with mid-sentence pause + "um" disfluency (C3 hard case) --------
echo "[3/4] RUN 2 -- disfluency pair, no stop ..."
for LABEL in "um_3s" "um_8s"; do
  if [ "$LABEL" = "um_3s" ]; then
    UTT="hey jarvis [PAUSE:1] what should I, um, [PAUSE:3] remember about Thursday"
  else
    UTT="hey jarvis [PAUSE:1] what should I, um, [PAUSE:8] remember about Thursday"
  fi
  echo "      label=$LABEL utterance='$UTT'"
  uv run tools/roomplay.py \
    --utterance "$UTT" \
    --label "$LABEL" \
    --report "$REPORT_DIR/run2_$LABEL.json" || echo "WARNING: $LABEL had errors"
  sleep 3.0
done

echo ""

# -- run 3: stop injection (main P1 check) -----------------------------------
echo "[4/4] RUN 3 -- stop injection 1s into reply ..."
echo "      utterance: 'hey jarvis [PAUSE:1] what should I remember about Thursday's meeting with Susan'"
echo "      stop-at: 1.0 s after first TTS audio"
uv run tools/roomplay.py \
  --utterance "hey jarvis [PAUSE:1] what should I remember about Thursday's meeting with Susan" \
  --stop-at 1.0 \
  --label p1_stop \
  --report "$REPORT_DIR/run3_p1_stop.json" \
  --keep-wav || echo "WARNING: run3 had errors"

echo ""
echo "=== ALL RUNS COMPLETE ==="
echo "Reports in: $REPORT_DIR"
echo ""
echo "--- CHECKPOINT 1 SUMMARY ---"
for f in "$REPORT_DIR"/*.json; do
  echo "  $f"
  python3 -c "
import json, sys
data = json.load(open('$f'))
for r in data.get('runs', []):
    v = r.get('verdicts', {})
    err = r.get('error')
    fa = v.get('first_audio_ms', 'N/A')
    gate = v.get('gate_decision', 'N/A')
    ok = v.get('latency_ok', False)
    print(f\"    {r['label']:30s}  first_audio={fa}ms  gate={gate}  ok={ok}\")
    if err: print(f\"    ERROR: {err}\")
    L = r.get('ledger')
    if L:
        print(f\"    ledger: turn_end={L['turn_end_ms']} llm_first={L['llm_first_token_ms']} tts_first={L['tts_first_audio_ms']} pb_end={L['playback_end_ms']}\")
" 2>/dev/null || true
done

echo ""
echo "OWNER: listen to /tmp/p1_stop.wav (the utterance played), then say 'P1 accepted' or name the defect."
echo "Engine log: $REPORT_DIR/engine.log"
