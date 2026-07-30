#!/bin/bash
# Usage: handoff_watch.sh <pid_to_wait_for> <gpu_idx> <seed> <log_file>
# Waits for another local_gpu_driver.sh instance (by PID) to exit, then
# launches local_gpu_driver.sh on the given GPU for bengali_only/<seed>.
# 2026-07-29: bengali_only seed0/seed1 have fisher done but no GPU assigned
# (only seed2 is currently running, on GPU1) -- without this they'd sit idle
# forever once GPU0 (swahili_only_b) and GPU1 (bengali_only seed2) finish,
# since local_gpu_driver.sh doesn't auto-continue to unrelated conditions/seeds.
set -euo pipefail

WAIT_PID="$1"; GPU="$2"; SEED="$3"; LOG="$4"
SCRIPT_DIR="/home/minjeong/project/FERRET/otter/scripts"

log() { echo "[handoff $(date -u +%H:%M:%S)] $*"; }

log "watching PID $WAIT_PID (GPU$GPU's current job) -- will start bengali_only/seed$SEED on GPU$GPU once it exits"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 20
done
log "PID $WAIT_PID gone -- starting bengali_only/seed$SEED on GPU$GPU"

cd "$SCRIPT_DIR"
SEEDS="$SEED" nohup bash local_gpu_driver.sh "$GPU" bengali_only >> "$LOG" 2>&1 &
log "launched bengali_only/seed$SEED on GPU$GPU (PID $!), logging to $LOG"
