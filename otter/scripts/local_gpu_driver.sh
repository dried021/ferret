#!/bin/bash
# Usage: local_gpu_driver.sh <gpu_idx> <condition1> [condition2 ...]
# Local adaptation of pod2_gpu_driver.sh: runs freq -> fisher -> svd_scale -> merge_eval
# for seeds 0,1,2 of each condition, on a single pinned local GPU, skipping stages
# whose output already exists. For the conditions queued here, fisher already
# exists locally for all 3 seeds (verified before launch), so this effectively
# starts at svd_scale for most seeds.
set -o pipefail

GPU="$1"; shift
CONDITIONS="$@"
# SEEDS (optional env var, default "0 1 2"): lets a caller split one
# condition's 3 seeds across two GPU drivers for load balancing, since
# merge_eval now needs a second (borrowed) GPU -- see MERGE_EVAL_HELPER_GPU
# below -- and GPU3 no longer runs its own independent condition queue.
SEEDS="${SEEDS:-0 1 2}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/mnt/HDD/minjeong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Thread oversubscription fix (2026-07-28, same issue as pod2_gpu_driver.sh):
# 20 physical cores, 2 concurrent GPU jobs each defaulting to ~20 threads = 2x
# oversubscription, observed to cut svd_scale throughput roughly in half.
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

ROOT=/mnt/HDD/minjeong/d2moe_results/phase1
cd /home/minjeong/project/FERRET/otter/scripts || exit 1

log() { echo "[localgpu$GPU $(date -u +%H:%M:%S)] $*"; }

# 2026-07-28 (overnight-unattended hardening): the old pattern was "try
# twice, then `continue` and abandon this seed forever" -- fine for a
# supervised session, but under heavy concurrent host load (3 local GPU jobs
# at once) we've now seen svd_scale/merge_eval fail transiently (OS OOM
# killer, CUDA OOM, or -- merge_eval specifically -- silent disk-offload
# producing all-NaN output, see phase1_merge_eval.py's own NaN guard) for
# reasons that go away once load eases, not because the seed is actually
# broken. Retrying a large-but-bounded number of times with a real sleep
# between attempts (instead of immediately hammering the same contended
# resource) means an unattended overnight run self-heals instead of quietly
# giving up on a seed. MAX_STAGE_RETRIES=40 at RETRY_SLEEP=90s is up to 1
# hour of persistence per stage before we truly give up -- generous relative
# to how fast host load has been observed to swing here.
MAX_STAGE_RETRIES=40
RETRY_SLEEP=90

# on_retry (optional 2nd arg): a command run once between a failed attempt
# and the next retry -- used by fisher/svd_scale to purge any corrupt
# layer_*.pt before trying again, same as the old two-attempt logic did.
run_stage() {
  local desc="$1"; local on_retry="$2"; shift 2
  local attempt=1
  while true; do
    "$@" && return 0
    if [ "$attempt" -ge "$MAX_STAGE_RETRIES" ]; then
      log "$desc: FAILED after $attempt attempts, giving up on this seed"
      return 1
    fi
    log "$desc: attempt $attempt failed, retrying in ${RETRY_SLEEP}s (${attempt}/${MAX_STAGE_RETRIES})"
    if [ -n "$on_retry" ]; then eval "$on_retry"; fi
    sleep "$RETRY_SLEEP"
    attempt=$((attempt + 1))
  done
}

# 2026-07-28 (actual root cause, found after the memory-headroom theory
# below was disproven by a clean run with 117GB free STILL producing NaN):
# it's not about host RAM at all. The model is ~31GB in bf16; a single 3090
# (24GB) can never hold it, so merge_eval on ONE GPU always forces
# accelerate's cpu/disk-offload path -- and THAT path is what's silently
# producing all-NaN output (confirmed: identical merge_condition() call
# under CUDA_VISIBLE_DEVICES=1,3 -- two real GPUs, no offload needed -- came
# back completely clean; the same call pinned to one GPU failed every time).
# Fix: merge_eval always runs across the caller's own GPU + a fixed helper
# GPU, so the full model fits in VRAM with no offload. GPU3 is that fixed
# helper and therefore does not get its own condition queue anymore (see
# README note in the launch commands). If this script IS started with
# GPU=3, it has no helper of its own to pair with, so it borrows GPU0.
MERGE_EVAL_HELPER_GPU=3
if [ "$GPU" = "3" ]; then MERGE_EVAL_HELPER_GPU=0; fi
MERGE_EVAL_DEVICES="$GPU,$MERGE_EVAL_HELPER_GPU"

# 2026-07-29: the helper GPU is shared with other users on this host (unlike
# the caller's own $GPU, which safe_gpus.sh already keeps clear for
# fisher/svd_scale). merge_eval doesn't go through that check at all, so
# without this it would happily grab MERGE_EVAL_HELPER_GPU mid-use by
# someone else. Same ownership check as safe_gpus.sh (ps owner of every
# compute-app pid on that GPU), just scoped to one GPU and looped until
# free instead of excluding-and-proceeding -- user instruction: don't touch
# it while someone else is on it, just wait and grab it the moment it's free.
helper_gpu_owned_by_other() {
  local target="$1" me; me=$(whoami)
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


# 2026-07-29 (same-day follow-up): the other-user check above missed a
# same-user conflict -- a separate ad-hoc backfill loop (this user's own
# `phase1_merge_eval.py`, re-running old conditions on CUDA_VISIBLE_DEVICES=
# 2,3 to add the Bengali eval column) held the helper GPU for hours and
# bengali_only's own merge_eval OOM'd through 13 of its 40 retries before
# anyone noticed (ownership check saw "owned by me" and let it proceed
# straight into a full GPU). Free-memory headroom doesn't care who owns the
# other process, so check it directly as a second gate alongside ownership.
# 20GiB is conservative headroom for this GPU's half of the ~31GB bf16 model.
HELPER_GPU_MIN_FREE_MIB=20000

helper_gpu_low_memory() {
  local target="$1"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' -v t="$target" '{gsub(/ /,"",$1); if ($1==t) print $2}')
  [ -z "$free_mib" ] && return 1
  [ "$free_mib" -lt "$HELPER_GPU_MIN_FREE_MIB" ]
}

wait_for_helper_gpu_free() {
  local waited=0
  while helper_gpu_owned_by_other "$MERGE_EVAL_HELPER_GPU" || helper_gpu_low_memory "$MERGE_EVAL_HELPER_GPU"; do
    if [ $((waited % 300)) -eq 0 ]; then
      log "helper GPU $MERGE_EVAL_HELPER_GPU busy or low on free memory (<${HELPER_GPU_MIN_FREE_MIB}MiB free), waiting (${waited}s so far)"
    fi
    sleep 15
    waited=$((waited + 15))
  done
}

# Old memory-headroom theory, kept as a cheap extra safety net (costs
# nothing when RAM is already fine, which it should be now that the real
# fix above is in place) -- but it is NOT the fix, the two-GPU pairing is.
MERGE_EVAL_MIN_AVAIL_GB=40
MERGE_EVAL_MEM_WAIT_MAX=20   # 20 * 30s = up to 10min waiting, then proceed regardless

wait_for_free_memory() {
  local waited=0
  while true; do
    avail=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$avail" -ge "$MERGE_EVAL_MIN_AVAIL_GB" ]; then
      return 0
    fi
    if [ "$waited" -ge "$MERGE_EVAL_MEM_WAIT_MAX" ]; then
      log "only ${avail}GB available after a short wait -- proceeding anyway (two-GPU pairing is the real fix, not this)"
      return 0
    fi
    sleep 30
    waited=$((waited + 1))
  done
}

scan_and_purge_corrupt() {
  python3 - "$1" <<'PYEOF'
import sys, glob, torch
d = sys.argv[1]
n = 0
for f in glob.glob(d + "/layer_*.pt"):
    try:
        torch.load(f, map_location="cpu")
    except Exception:
        import os
        os.remove(f)
        n += 1
        print(f"[corrupt-scan] purged {f}")
print(f"[corrupt-scan] purged {n} file(s)")
PYEOF
}

for cond in $CONDITIONS; do
  for seed in $SEEDS; do
    seed_dir="$ROOT/$cond/seed$seed"
    mkdir -p "$seed_dir"

    if ! ls "$seed_dir"/deepseek_wikitext_*_expert_frequencies.json >/dev/null 2>&1; then
      log "$cond/seed$seed: freq starting"
      run_stage "$cond/seed$seed: freq" "" \
        conda run -n d2moe_env python -u phase1_run_freq_and_scale.py --condition "$cond" --seed "$seed" \
        || continue
    else
      log "$cond/seed$seed: freq already present, skipping"
    fi

    if [ ! -f "$seed_dir/fisher_processed.pt" ]; then
      log "$cond/seed$seed: fisher starting"
      run_stage "$cond/seed$seed: fisher" "scan_and_purge_corrupt '$seed_dir/fisher_layers'" \
        conda run -n d2moe_env python -u phase1_fisher.py --condition "$cond" --seed "$seed" --n-samples 64 --seqlen 512 \
        || continue
      rm -rf "$seed_dir/fisher_layers"
      log "$cond/seed$seed: fisher done, fisher_layers cleaned"
    else
      log "$cond/seed$seed: fisher_processed.pt already present, skipping"
    fi

    if [ ! -f "$seed_dir/svd_scale_processed.pt" ]; then
      log "$cond/seed$seed: svd_scale starting"
      run_stage "$cond/seed$seed: svd_scale" "scan_and_purge_corrupt '$seed_dir/svd_scale_layers'" \
        conda run -n d2moe_env python -u phase1_svd_scale.py --condition "$cond" --seed "$seed" --n-samples 64 --seqlen 512 \
        || continue
      rm -rf "$seed_dir/svd_scale_layers"
      log "$cond/seed$seed: svd_scale done, svd_scale_layers cleaned"
    else
      log "$cond/seed$seed: svd_scale_processed.pt already present, skipping"
    fi

    out_json="$seed_dir/scale_${cond}_seed${seed}/eval_ppl.json"
    if [ ! -f "$out_json" ]; then
      # merge_eval needs 2 real GPUs to fit the model without the
      # NaN-producing offload path (see MERGE_EVAL_HELPER_GPU above) --
      # borrows the helper GPU, so flock still serializes across all
      # local_gpu_driver.sh instances (only one merge_eval, and thus one
      # user of the shared helper GPU, at a time). svd_scale/fisher on the
      # OTHER (non-helper) GPU keep running unblocked.
      log "$cond/seed$seed: waiting for helper GPU $MERGE_EVAL_HELPER_GPU to be free of other users' jobs"
      wait_for_helper_gpu_free
      log "$cond/seed$seed: merge_eval starting on GPU $MERGE_EVAL_DEVICES (waiting for merge_eval lock if held)"
      run_stage "$cond/seed$seed: merge_eval" "wait_for_free_memory" \
        flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES="$MERGE_EVAL_DEVICES" conda run -n d2moe_env python -u phase1_merge_eval.py --condition "$cond" --seed "$seed" --scale-condition "$cond" --scale-seed "$seed" \
        || continue
      log "$cond/seed$seed: merge_eval done"
    else
      log "$cond/seed$seed: merge+eval already done, skipping"
    fi
  done
done
log "ALL DONE for conditions: $CONDITIONS"
