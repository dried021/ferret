#!/bin/bash
# 2026-07-30: Track B(interference_minimax) 후보 조건의 실제 GPU 파이프라인 검증
# 파일럿 -- seed0 하나만 먼저 돌려서 실측 소요시간과 결과를 확인한다.
# freq_and_scale -> fisher -> merge_eval(plain, whitening scale 없음) 순서,
# GPU2,3 전용. merge_eval만 flock으로 보호(fisher_processed.pt+svd_scale_processed.pt
# 동시 로드 문제는 merge_eval 자체의 이슈이고, freq/fisher는 이 문제와 무관).
set -o pipefail
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/HDD/minjeong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/pilot_interference_minimax_seed0.log"

log() { echo "[pilot-interference-minimax $(date -u +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

COND=interference_minimax
SEED=0

log "=== STEP 1: phase1_run_freq_and_scale.py --condition $COND --seed $SEED ==="
conda run -n d2moe_env python -u phase1_run_freq_and_scale.py --condition "$COND" --seed "$SEED" >> "$LOGFILE" 2>&1
rc=$?
log "freq_and_scale rc=$rc"
if [ $rc -ne 0 ]; then
  log "ABORT: freq_and_scale failed, not proceeding to fisher/merge_eval"
  exit 1
fi

log "=== STEP 2: phase1_fisher.py --condition $COND --seed $SEED --n-samples 64 --seqlen 512 ==="
conda run -n d2moe_env python -u phase1_fisher.py --condition "$COND" --seed "$SEED" --n-samples 64 --seqlen 512 >> "$LOGFILE" 2>&1
rc=$?
log "fisher rc=$rc"
if [ $rc -ne 0 ]; then
  log "ABORT: fisher failed, not proceeding to merge_eval"
  exit 1
fi

log "=== STEP 3: phase1_merge_eval.py --condition $COND --seed $SEED (plain, no scale) ==="
flock /tmp/phase1_merge_eval.lock conda run -n d2moe_env python -u phase1_merge_eval.py --condition "$COND" --seed "$SEED" >> "$LOGFILE" 2>&1
rc=$?
log "merge_eval rc=$rc"

log "PILOT DONE (merge_eval rc=$rc) -- review $LOGFILE for eval_ppl.json bpb numbers, compare interference_minimax vs Balanced worst-language degradation (22.034 mean, ben_Beng) from phase1_6_budget_gate_v2.py's Table 3v2"
