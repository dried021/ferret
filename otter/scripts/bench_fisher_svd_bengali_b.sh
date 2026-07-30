#!/bin/bash
# 2026-07-30: fisher_processed.pt + svd_scale_processed.pt 한 조건/seed(~60G)
# 계산에 실제로 얼마나 걸리는지 실측하기 위한 벤치마크.
# 조건: bengali_only_b seed0 (phase1_calib_data.py에 이미 등록된 placebo
# 조건, 지금까지 한 번도 계산 안 됨 -- Chinese/Bengali 2x2 확장의 다음
# 단계인 chinese_only_b/bengali_only_b 준비 작업이기도 함).
# svd_scale은 다른 작업과 동시 실행 시 RAM 경합으로 죽은 전례가 있어(3기술노트
# §2.5), 같은 flock으로 직렬화해서 지금 도는 4.3 grid 뒤에 안전하게 줄 세운다.
set -o pipefail
export HF_HOME=/mnt/HDD/minjeong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
LOGFILE="$LOGDIR/bench_fisher_svd_bengali_b.log"

log() { echo "[bench-bengali_b $(date -u +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

pick_gpus() {
  bash -c 'source ./safe_gpus.sh >/dev/null 2>&1 && echo "$SAFE_GPUS"'
}

COND=bengali_only_b
SEED=0

log "=== waiting for merge_eval lock, then STEP1: phase1_fisher.py (timed) ==="
T0=$(date +%s)
gpus=$(pick_gpus)
log "using GPUs $gpus"
flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES="$gpus" conda run -n d2moe_env python -u phase1_fisher.py --condition "$COND" --seed "$SEED" --n-samples 64 --seqlen 512 >> "$LOGFILE" 2>&1
rc=$?
T1=$(date +%s)
FISHER_SEC=$((T1-T0))
log "fisher rc=$rc, elapsed=${FISHER_SEC}s ($((FISHER_SEC/60))min)"
if [ $rc -ne 0 ]; then
  log "ABORT: fisher failed"
  exit 1
fi

log "=== STEP2: phase1_svd_scale.py (timed) ==="
T2=$(date +%s)
gpus=$(pick_gpus)
log "using GPUs $gpus"
flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES="$gpus" conda run -n d2moe_env python -u phase1_svd_scale.py --condition "$COND" --seed "$SEED" --n-samples 64 --seqlen 512 >> "$LOGFILE" 2>&1
rc=$?
T3=$(date +%s)
SVD_SEC=$((T3-T2))
log "svd_scale rc=$rc, elapsed=${SVD_SEC}s ($((SVD_SEC/60))min)"

TOTAL_SEC=$((T3-T0))
log "=== BENCH DONE: fisher=${FISHER_SEC}s svd=${SVD_SEC}s total=${TOTAL_SEC}s ($((TOTAL_SEC/60))min) rc=$rc ==="
if [ $rc -eq 0 ]; then
  ls -la /mnt/HDD/minjeong/d2moe_results/phase1/$COND/seed$SEED/fisher_processed.pt \
         /mnt/HDD/minjeong/d2moe_results/phase1/$COND/seed$SEED/svd_scale_processed.pt >> "$LOGFILE" 2>&1
fi
