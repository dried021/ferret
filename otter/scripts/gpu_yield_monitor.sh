#!/bin/bash
# Periodically checks whether any of OUR (minjeong) GPU compute processes
# share a GPU with another user's process. If so, kills only our process on
# that GPU (never touches other users' processes) so their job isn't starved
# for memory/compute by us landing there (see safe_gpus.sh's GPU1 exclusion
# note -- this is the general, ongoing version of that same incident type).
#
# Does NOT try to add parallel GPU workers for merge_eval-family jobs
# (phase1_belebele_eval.py, phase1_merge_eval.py, run_phase1_belebele_grid.py,
# etc.) -- those are deliberately serialized by /tmp/phase1_merge_eval.lock
# because two such jobs loading fisher_processed.pt+svd_scale_processed.pt
# onto CPU RAM at once OOM-killed this host before (125GB RAM, 2026-07-29).
# Only self-yield is safe to automate; adding more parallel GPU load is not.
set -uo pipefail

LOG="/home/minjeong/project/FERRET/otter/logs/gpu_yield_monitor.log"
INTERVAL="${INTERVAL:-180}"
ME=$(whoami)

log() { echo "[gpu-yield] $(date '+%F %T') $1" | tee -a "$LOG"; }

log "monitor started (pid $$, interval ${INTERVAL}s)"

while true; do
  # pid,gpu_uuid for every compute process on the host
  while IFS=, read -r pid gpu_uuid; do
    pid=$(echo "$pid" | xargs); gpu_uuid=$(echo "$gpu_uuid" | xargs)
    [ -z "$pid" ] && continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
    [ "$owner" != "$ME" ] && continue

    gpu_idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
      | grep "$gpu_uuid" | cut -d',' -f1 | xargs)
    [ -z "$gpu_idx" ] && continue

    # is any OTHER user also compute-active on this same GPU right now?
    other_on_same_gpu=""
    while IFS=, read -r pid2 uuid2; do
      pid2=$(echo "$pid2" | xargs); uuid2=$(echo "$uuid2" | xargs)
      [ "$uuid2" != "$gpu_uuid" ] && continue
      owner2=$(ps -o user= -p "$pid2" 2>/dev/null | xargs || true)
      [ -z "$owner2" ] && continue
      if [ "$owner2" != "$ME" ]; then
        other_on_same_gpu="$owner2"
      fi
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)

    if [ -n "$other_on_same_gpu" ]; then
      cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || echo "unknown")
      log "COLLISION: our pid $pid on GPU $gpu_idx shares it with '$other_on_same_gpu' -- killing our pid (cmd: $cmd)"
      kill "$pid" 2>/dev/null
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)

  sleep "$INTERVAL"
done
