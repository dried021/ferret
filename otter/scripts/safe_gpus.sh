#!/bin/bash
# Same GPU-ownership safety check as 00_config.py::_safe_gpu_indices(), but for
# shelling out to the vendored D2MoE/ scripts directly (they have no such check
# built in). Source this before any D2MoE python invocation:
#   source ../scripts/safe_gpus.sh
#   CUDA_VISIBLE_DEVICES=$SAFE_GPUS python D2-deepseek.py ...
#
# Excludes any GPU with a compute process owned by someone other than the
# current user, regardless of how much free memory it appears to have.
#
# Also hard-caps at MAX_GPUS (user request, 2026-07-24: "GPU 계속 2개만 써줘")
# -- even if 3+ GPUs are free of other users, only the first MAX_GPUS
# (lowest index) are used, so this project's footprint on the shared host
# stays predictable regardless of how many GPUs happen to be idle.
set -euo pipefail

MAX_GPUS="${MAX_GPUS:-2}"
# 2026-07-30: GPU1 is in persistent use by another user (kahyeon, RelPrune) --
# the other-user-process check below only catches whoever is *currently*
# resident at query time, which raced against a belebele-grid seed's GPU pick
# (english_only/seed2 landed on GPU1 right as kahyeon's job grew into it,
# OOM'd). User-requested: keep GPU1 out of the pool outright rather than
# relying on that per-call race. Remove/adjust once GPU1 is reliably free.
MANUAL_EXCLUDE_GPUS="${MANUAL_EXCLUDE_GPUS:-1}"

ME=$(whoami)
ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
BUSY_OTHER=""

while IFS=, read -r gpu_uuid pid; do
  gpu_uuid=$(echo "$gpu_uuid" | xargs)
  pid=$(echo "$pid" | xargs)
  [ -z "$pid" ] && continue
  owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
  [ -z "$owner" ] && continue
  if [ "$owner" != "$ME" ]; then
    gpu_idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | xargs)
    echo "[safe_gpus] GPU $gpu_idx: process owned by '$owner' (pid $pid), not '$ME' -- excluding" >&2
    BUSY_OTHER="$BUSY_OTHER $gpu_idx"
  fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)

for g in $MANUAL_EXCLUDE_GPUS; do
  echo "[safe_gpus] GPU $g: manually excluded (MANUAL_EXCLUDE_GPUS)" >&2
  BUSY_OTHER="$BUSY_OTHER $g"
done

SAFE_GPUS=""
for g in $ALL_GPUS; do
  if ! echo "$BUSY_OTHER" | grep -qw "$g"; then
    SAFE_GPUS="${SAFE_GPUS:+$SAFE_GPUS,}$g"
  fi
done

if [ -z "$SAFE_GPUS" ]; then
  echo "[safe_gpus] no GPU is free of other users' processes -- refusing to proceed" >&2
  exit 1
fi

# Cap to MAX_GPUS (lowest indices first)
SAFE_GPUS=$(echo "$SAFE_GPUS" | tr ',' '\n' | head -n "$MAX_GPUS" | paste -sd, -)

echo "[safe_gpus] safe GPUs (capped at $MAX_GPUS): $SAFE_GPUS" >&2
export SAFE_GPUS
