#!/usr/bin/env bash
# Watches scratch/wake/positive/ for 40 wav files, then launches OWW training detached.
# Usage: bash tools/watch_wake_clips.sh [clip_dir] [target_count]
# Defaults: clip_dir=scratch/wake/positive, target_count=40

set -euo pipefail

CLIP_DIR="${1:-/Users/deenuy/Documents/Workspace/Projects/voxscribe/scratch/wake/positive}"
TARGET="${2:-40}"
LOG="/tmp/wake_watcher.log"
TRAIN_SCRIPT="$(dirname "$0")/train_wake_model.py"
VENV="/Users/deenuy/Documents/Workspace/Projects/voxscribe/voxscribe-server/.venv/bin/python3"

mkdir -p "$CLIP_DIR"
echo "[$(date)] watcher: monitoring $CLIP_DIR for $TARGET wav files" | tee -a "$LOG"

while true; do
    count=$(find "$CLIP_DIR" -maxdepth 1 -name "*.wav" | wc -l | tr -d ' ')
    echo "[$(date)] watcher: $count/$TARGET clips present" | tee -a "$LOG"
    if [ "$count" -ge "$TARGET" ]; then
        echo "[$(date)] watcher: $count clips ready — launching training detached" | tee -a "$LOG"
        nohup "$VENV" "$TRAIN_SCRIPT" \
            --clips "$CLIP_DIR" \
            --output "/Users/deenuy/Documents/Workspace/Projects/voxscribe/scratch/wake/model" \
            >> "$LOG" 2>&1 &
        echo "[$(date)] watcher: training PID=$! — log at $LOG" | tee -a "$LOG"
        exit 0
    fi
    sleep 10
done
