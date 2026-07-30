#!/bin/bash
# §6 whitening-targeted robust mixing pilot -- unattended orchestrator chaining
# STEP1's gate result -> STEP2 (calib text + token-length report) -> STEP3
# (GPU pipeline for the new "robust_mix" condition, seed0 only) the moment
# GPU2 is actually idle. Meant to be launched once and left running in the
# background; every stage below polls/waits rather than assuming its
# precondition is already true when this script starts.
set -o pipefail
cd /home/minjeong/project/FERRET/otter/scripts || exit 1

log() { echo "[robust-mix-orch $(date -u +%H:%M:%S)] $*"; }

WEIGHTS_JSON=../results/robust_mix_weights.json
TARGET_GPU=2

# ---- Stage 0: wait for STEP1 (phase1_6_robust_mix_weights.py) to finish ----
log "waiting for STEP1 output ($WEIGHTS_JSON)..."
while [ ! -f "$WEIGHTS_JSON" ]; do
  sleep 30
done
log "STEP1 output found."

GATE_PASSED=$(python3 -c "import json; print(json.load(open('$WEIGHTS_JSON'))['gate_passed'])")
if [ "$GATE_PASSED" != "True" ]; then
  log "GATE FAILED (w* does not improve max relFrob over balanced) -- per task spec, STOPPING here. No GPU work will start."
  exit 0
fi
log "GATE PASSED -- proceeding to STEP2/STEP3."

# ---- Stage 1: STEP2 -- verify calib text builds + measure realized tokens (CPU, fast) ----
log "STEP2: building robust_mix seed0 calibration text (sanity check) + measuring realized token length..."
conda run -n d2moe_env python3 -c "
import sys; sys.path.insert(0, '.')
import phase1_calib_data as c
w = c.load_robust_mix_weights()
print('[robust-mix-orch] loaded w* =', w)
sents = c.build_condition_sentences('robust_mix', seed=0)
print(f'[robust-mix-orch] robust_mix seed0: {len(sents)} sentences built OK')
" 2>&1 | tee ../logs/robust_mix_step2_sanity.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log "STEP2 sanity check FAILED -- aborting, not touching GPU."
  exit 1
fi

conda run -n d2moe_env env HF_HOME=/mnt/HDD/minjeong/hf_cache python3 -u phase1_6_robust_mix_calib_tokens.py \
  --condition robust_mix --n-samples 64 --seqlen 512 --seeds 0 1 2 \
  2>&1 | tee ../logs/robust_mix_step2_tokens.log
log "STEP2 done (see ../results/calib_token_length_robust_mix.json for the Appendix A1 row)."

# ---- Stage 2: wait for GPU2 to be genuinely idle (no other user, low usage) ----
gpu_busy_or_owned_by_other() {
  local target="$1" me; me=$(whoami)
  local used_mib
  used_mib=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v t="$target" '{gsub(/ /,"",$1); if ($1==t) print $2}')
  if [ -n "$used_mib" ] && [ "$used_mib" -ge 2000 ]; then
    return 0
  fi
  while IFS=, read -r gpu_uuid pid; do
    gpu_uuid=$(echo "$gpu_uuid" | xargs); pid=$(echo "$pid" | xargs)
    [ -z "$pid" ] && continue
    local owner; owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
    [ -z "$owner" ] || [ "$owner" = "$me" ] && continue
    local idx; idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | xargs)
    [ "$idx" = "$target" ] && return 0
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  return 1
}

log "STEP3: waiting for GPU$TARGET_GPU to be idle (<2000MiB used, no other user's process)..."
waited=0
while gpu_busy_or_owned_by_other "$TARGET_GPU"; do
  if [ $((waited % 300)) -eq 0 ]; then
    log "GPU$TARGET_GPU still busy/owned by another user, waiting (${waited}s so far)..."
  fi
  sleep 15
  waited=$((waited + 15))
done
log "GPU$TARGET_GPU is idle -- launching STEP3 pipeline (freq->fisher->svd_scale->merge_eval, seed0 only, helper GPU3 for merge_eval)."

# ---- Stage 3: run the actual GPU pipeline (reuses local_gpu_driver.sh verbatim) ----
SEEDS="0" bash local_gpu_driver.sh "$TARGET_GPU" robust_mix 2>&1 | tee -a ../logs/robust_mix_seed0_gpu2.log
log "STEP3 driver exited (see ../logs/robust_mix_seed0_gpu2.log for detail)."

OUT_JSON=/mnt/HDD/minjeong/d2moe_results/phase1/robust_mix/seed0/scale_robust_mix_seed0/eval_ppl.json
if [ -f "$OUT_JSON" ]; then
  log "eval_ppl.json produced: $OUT_JSON"
  cat "$OUT_JSON"
else
  log "WARNING: expected output $OUT_JSON not found -- check ../logs/robust_mix_seed0_gpu2.log for the failure."
fi
log "ORCHESTRATOR DONE."
