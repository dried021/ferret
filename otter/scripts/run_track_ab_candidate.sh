#!/bin/bash
# 2026-07-30: §6 Track A/B 후보 조건(vulnerability_targeted, _abs, _random,
# interference_minimax, interference_minimax_random) 전용 실행 스크립트.
#
# ★★★ 이 fisher_processed.pt 자동삭제 로직은 아래 5개 조건에만 적용할 것 ★★★
#   vulnerability_targeted / vulnerability_targeted_abs /
#   vulnerability_targeted_random / interference_minimax /
#   interference_minimax_random
# 이 조건들의 fisher는 merge_eval 한 번 먹이는 용도로만 쓰이고 다른 어떤
# 분석도 재참조하지 않는다(§6 budget_gate_v2.py는 eval_ppl.json만 읽음,
# interference_model.py의 LOO도 eval_ppl.json만 읽음 -- fisher_processed.pt
# 자체를 읽는 코드는 phase1_merge_eval.py 뿐). english_only/korean_only/
# chinese_only/swahili_only/bengali_only/mixed_5lang/disagreement_targeted*
# 등 기존 조건에는 절대 쓰지 말 것 -- Table 3 2x2, 향후 재분석이 그 fisher를
# 다시 읽을 수 있음. svd_scale_processed.pt(Track C용)도 삭제 대상 아님
# (Track C가 계속 재참조).
#
# 조건당 seed 0,1,2 순서로: freq_and_scale -> fisher -> merge_eval(plain,
# scale 없음) -> merge_eval 성공(rc=0, eval_ppl.json 존재) 확인되면 그
# seed의 fisher_processed.pt(27.8G)만 즉시 삭제, 다음 seed로.
# 실패하면 fisher는 남겨둠(재시도 시 fisher부터 다시 안 돌아도 되게).
#
# Usage: ./run_track_ab_candidate.sh <condition> [seed ...]
#   예: ./run_track_ab_candidate.sh interference_minimax 0 1 2
set -o pipefail

ALLOWED_CONDITIONS="vulnerability_targeted vulnerability_targeted_abs vulnerability_targeted_random interference_minimax interference_minimax_random"

COND="$1"; shift
if [ -z "$COND" ]; then
  echo "Usage: $0 <condition> [seed ...]" >&2
  exit 1
fi
if ! echo " $ALLOWED_CONDITIONS " | grep -q " $COND "; then
  echo "REFUSING: '$COND' is not one of the §6 Track A/B candidate conditions ($ALLOWED_CONDITIONS)." >&2
  echo "This script's fisher-delete-after-merge logic is only safe for those 5 -- see header comment." >&2
  exit 1
fi

SEEDS="${*:-0 1 2}"
RESULTS_ROOT=/mnt/HDD/minjeong/d2moe_results/phase1
export HF_HOME=/mnt/HDD/minjeong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/track_ab_${COND}.log"

log() { echo "[track-ab-$COND $(date -u +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

# 다른 유저(예: kahyeon)가 GPU0/1을 쓰고 있을 수 있으니 하드코딩 대신
# safe_gpus.sh로 매번 안전한 2장을 동적으로 고른다(project 관례와 동일).
pick_gpus() {
  local out
  out=$(bash -c 'source ./safe_gpus.sh >/dev/null 2>&1 && echo "$SAFE_GPUS"')
  echo "$out"
}

for SEED in $SEEDS; do
  SEED_DIR="$RESULTS_ROOT/$COND/seed$SEED"
  EVAL_PATH="$SEED_DIR/eval_ppl.json"
  FISHER_PATH="$SEED_DIR/fisher_processed.pt"

  if [ -f "$EVAL_PATH" ]; then
    log "seed$SEED: eval_ppl.json already exists, skipping entirely"
    continue
  fi

  if [ ! -f "$FISHER_PATH" ]; then
    log "seed$SEED: === STEP 1: phase1_run_freq_and_scale.py ==="
    gpus=$(pick_gpus)
    log "seed$SEED: using GPUs $gpus"
    env CUDA_VISIBLE_DEVICES="$gpus" conda run -n d2moe_env python -u phase1_run_freq_and_scale.py --condition "$COND" --seed "$SEED" >> "$LOGFILE" 2>&1
    rc=$?
    log "seed$SEED: freq_and_scale rc=$rc"
    if [ $rc -ne 0 ]; then
      log "seed$SEED: ABORT (freq_and_scale failed) -- moving to next seed"
      continue
    fi

    log "seed$SEED: === STEP 2: phase1_fisher.py ==="
    gpus=$(pick_gpus)
    log "seed$SEED: using GPUs $gpus"
    env CUDA_VISIBLE_DEVICES="$gpus" conda run -n d2moe_env python -u phase1_fisher.py --condition "$COND" --seed "$SEED" --n-samples 64 --seqlen 512 >> "$LOGFILE" 2>&1
    rc=$?
    log "seed$SEED: fisher rc=$rc"
    if [ $rc -ne 0 ]; then
      log "seed$SEED: ABORT (fisher failed) -- moving to next seed"
      continue
    fi
  else
    log "seed$SEED: fisher_processed.pt already exists, skipping freq/fisher steps"
  fi

  log "seed$SEED: === STEP 3: phase1_merge_eval.py (plain, no scale) ==="
  gpus=$(pick_gpus)
  log "seed$SEED: using GPUs $gpus"
  flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES="$gpus" conda run -n d2moe_env python -u phase1_merge_eval.py --condition "$COND" --seed "$SEED" >> "$LOGFILE" 2>&1
  rc=$?
  log "seed$SEED: merge_eval rc=$rc"

  if [ $rc -eq 0 ] && [ -f "$EVAL_PATH" ]; then
    if [ -f "$FISHER_PATH" ]; then
      sz=$(stat -c %s "$FISHER_PATH" 2>/dev/null)
      rm -f "$FISHER_PATH"
      log "seed$SEED: merge_eval succeeded, eval_ppl.json captured -- deleted $FISHER_PATH ($(numfmt --to=iec ${sz:-0}))"
    fi
  else
    log "seed$SEED: merge_eval did NOT succeed -- keeping fisher_processed.pt for retry"
  fi
done

log "=== $COND: ALL SEEDS DONE ==="
