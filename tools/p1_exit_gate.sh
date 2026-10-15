#!/usr/bin/env bash
# P1 exit gate: automated roomplay runs via speaker→mic.
# Wake detection bypassed via /v1/wake/trigger [SIM:wake]; all other
# stages are near-field real-path (speaker→mic). 70% weight applies.
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

# Poll /v1/wake/state until not 'listening' (session closed and engine armed).
# Usage: wait_for_armed <engine_url> [timeout_s]
wait_for_armed() {
  local url="$1/v1/wake/state"
  local timeout="${2:-90}"
  local elapsed=0
  while [ $elapsed -lt $timeout ]; do
    local state
    state=$(curl -sf "$url" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('state',''))" 2>/dev/null || true)
    if [ "$state" != "listening" ] && [ -n "$state" ]; then
      echo "    engine state: $state (ready)"
      return 0
    fi
    sleep 0.5
    elapsed=$((elapsed + 1))
  done
  echo "    WARNING: wait_for_armed timed out after ${timeout}s"
  return 1
}

echo "=== P1 EXIT GATE ==="
echo "report dir: $REPORT_DIR"
echo ""

# -- start engine in wake mode in background --------------------------------
echo "[0/4] starting engine --wake --no-auth ..."
cd "$ROOT"
VOXSCRIBE_CAPTURE_DEVICE="MacBook Pro Microphone" \
VOXSCRIBE_PLAYBACK_DEVICE="MacBook Pro Speakers" \
  uv run python -m app.local.engine --wake --no-auth --port 0 \
  > "$REPORT_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
trap "echo 'stopping engine...'; kill $ENGINE_PID 2>/dev/null; wait $ENGINE_PID 2>/dev/null || true; echo done" EXIT

# wait for engine ready signal and capture port
WAITED=0
ENGINE_PORT=""
until [ -n "$ENGINE_PORT" ]; do
  ENGINE_PORT=$(grep -o '"port": [0-9]*' "$REPORT_DIR/engine.log" 2>/dev/null | grep -o '[0-9]*' || true)
  sleep 0.5
  WAITED=$((WAITED + 1))
  if [ $WAITED -gt 40 ]; then
    echo "ERROR: engine did not start within 20s"
    cat "$REPORT_DIR/engine.log"
    exit 1
  fi
done
ENGINE_URL="http://127.0.0.1:${ENGINE_PORT}"
echo "    engine ready on $ENGINE_URL ($(( WAITED / 2 ))s)"
echo ""

# -- brief warm-up: wake loop needs ~1s to arm after engine ready -----------
sleep 2.0

# -- run 1: five pause lengths (C3 check, no stop) --------------------------
echo "[1/4] RUN 1 -- five pause lengths (1.5/3/5/8/11 s), no stop"
echo "      utterance: 'tell me [PAUSE:N] something interesting'"
echo "      NOTE: wake phase simulated via trigger endpoint [SIM:wake]"
uv run tools/roomplay.py \
  --utterance "tell me [PAUSE:N] something interesting" \
  --pause-sweep 1.5,3,5,8,11 \
  --engine-url "$ENGINE_URL" \
  --label c3_sweep \
  --report "$REPORT_DIR/run1_c3_sweep.json" || echo "WARNING: run1 had errors"

echo ""
wait_for_armed "$ENGINE_URL"

# -- run 2: disfluency pair (C3 hard case) ----------------------------------
echo "[2/4] RUN 2 -- two disfluency utterances (um inside pause)"
for LABEL in "um_3s" "um_8s"; do
  if [ "$LABEL" = "um_3s" ]; then
    UTT="what should I, um, [PAUSE:3] remember about Thursday"
  else
    UTT="what should I, um, [PAUSE:8] remember about Thursday"
  fi
  echo "      label=$LABEL utterance='$UTT'"
  uv run tools/roomplay.py \
    --utterance "$UTT" \
    --engine-url "$ENGINE_URL" \
    --label "$LABEL" \
    --report "$REPORT_DIR/run2_$LABEL.json" || echo "WARNING: $LABEL had errors"
  wait_for_armed "$ENGINE_URL"
done

echo ""

# -- run 3: stop injection (main P1 check) -----------------------------------
echo "[3/4] RUN 3 -- stop injection 1s into reply (main P1 check)"
echo "      utterance: 'what should I remember about Thursday's meeting with Susan'"
echo "      stop-at: 1.0 s after first TTS audio"
uv run tools/roomplay.py \
  --utterance "what should I remember about Thursday's meeting with Susan" \
  --stop-at 1.0 \
  --engine-url "$ENGINE_URL" \
  --label p1_stop \
  --report "$REPORT_DIR/run3_p1_stop.json" \
  --keep-wav || echo "WARNING: run3 had errors"

echo ""
echo "=== ALL RUNS COMPLETE ==="
echo "Reports in: $REPORT_DIR"
echo ""
echo "--- CHECKPOINT 1 SUMMARY ---"
python3 -c "
import json, pathlib, sys
rdir = pathlib.Path('$REPORT_DIR')
for f in sorted(rdir.glob('*.json')):
    data = json.loads(f.read_text())
    for r in data.get('runs', []):
        v = r.get('verdicts', {})
        err = r.get('error')
        fa = v.get('first_audio_ms', 'N/A')
        gate = v.get('gate_decision', 'N/A')
        ok = v.get('latency_ok', False)
        mode = r.get('mode', '')
        print(f\"  {r['label']:30s}  first_audio={fa}ms  gate={gate}  ok={ok}\")
        if err: print(f\"    ERROR: {err}\")
        L = r.get('ledger')
        if L:
            print(f\"    ledger: turn_end={L['turn_end_ms']} llm_first={L['llm_first_token_ms']} tts_first={L['tts_first_audio_ms']} pb_end={L['playback_end_ms']}\")
        G = r.get('gate')
        if G:
            print(f\"    gate:   decision={G['decision']}  category={G['category']}  mode={G['mode']}\")
        M = r.get('meeting')
        if M:
            print(f\"    meeting: id={M['id'][:8]}...  status={M['status']}  origin={M['origin']}\")
" 2>/dev/null || true

echo ""
echo "Utterance WAV saved to: /tmp/p1_stop.wav"
echo "Engine log: $REPORT_DIR/engine.log"
echo ""
echo "SCORING NOTE: All results are near-field acoustic + [SIM:wake]."
echo "  Exercise level: synthetic (70% weight). Real-path (100%) requires owner at UM04."
echo ""
echo "OWNER CHECKPOINT 1: Play /tmp/p1_stop.wav to hear what was spoken."
echo "  Then say 'P1 accepted' or name the defect."
