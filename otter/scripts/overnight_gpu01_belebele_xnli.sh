#!/bin/bash
# 2026-07-29 야간 자동화 (GPU0,1 큐): 우선순위 3(Belebele korean_only grid)
# -> 4(XNLI downstream smoke, 6 조건+baseline) -> 게이트/분석 -> 시간이
# 남으면 우선순위 6(4.3 Swahili 2x2 확장, best-effort).
#
# run_phase1_belebele_grid.py / run_phase1_2x2_lang.py는 이번 세션에 flock
# /tmp/phase1_merge_eval.lock을 내부 run()에 추가해뒀음(otter/scripts/logs의
# belebele_grid_korean_only.log에서 실제로 관측된 OOM 원인 -- GPU2,3 큐와
# CPU RAM을 나눠쓰다 exit 137로 죽음, 2026-07-29). XNLI는 전용 오케스트레이터가
#없어서 이 스크립트에서 직접 flock을 건다.
set -o pipefail
export HF_HOME=/mnt/HDD/minjeong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/overnight_gpu01_belebele_xnli.log"

log() { echo "[gpu01-queue $(date -u +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

safe_gpus() {
  ( source ./safe_gpus.sh >/dev/null 2>&1 && echo "$SAFE_GPUS" )
}

log "=========================================="
log "STEP 1: Belebele korean_only grid (seed 0/1/2, off/on pruning, 5-shot n=200)"
log "=========================================="
conda run -n d2moe_env python -u run_phase1_belebele_grid.py \
  --conditions korean_only --seeds 0 1 2 --pp-ratio 0.2 --num-fewshot 5 --limit 200 --batch-size 1 \
  >> "$LOGFILE" 2>&1
rc=$?
log "belebele grid rc=$rc"

log "=========================================="
log "STEP 2: Belebele gate (retention report)"
log "=========================================="
conda run -n d2moe_env python -u phase1_belebele_gate.py --condition korean_only --pp-ratio 0.2 \
  --scale-condition korean_only --num-fewshot 5 >> "$LOGFILE" 2>&1
log "belebele gate rc=$?"

log "=========================================="
log "STEP 3: XNLI smoke -- baseline + 6 conditions"
log "=========================================="
RESULTS_ROOT=/mnt/HDD/minjeong/d2moe_results/phase1
run_xnli() {
  local extra="$1" outdir="$2" desc="$3"
  local out_json="$RESULTS_ROOT/$outdir/eval_xnli_smoke.json"
  if [ -f "$out_json" ]; then
    log "xnli $desc: already done ($out_json exists), skipping"
    return 0
  fi
  local logfile="$LOGDIR/overnight_xnli_${desc}.log"
  log "xnli $desc: starting -> $logfile"
  gpus=$(safe_gpus)
  flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES="$gpus" \
    conda run -n d2moe_env python -u phase1_xnli_eval.py $extra --smoke > "$logfile" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    log "xnli $desc: DONE"
  else
    log "xnli $desc: FAILED (rc=$rc) -- see $logfile -- continuing"
  fi
}

run_xnli "--baseline" "baseline" "baseline"
for cond in english_only korean_only chinese_only swahili_only bengali_only mixed_5lang; do
  run_xnli "--condition $cond --scale-condition $cond --scale-seed 0 --seed 0" "$cond/seed0/scale_${cond}_seed0" "$cond"
done

log "=========================================="
log "STEP 4: XNLI analyze"
log "=========================================="
conda run -n d2moe_env python -u phase1_xnli_analyze.py --seed 0 --smoke --scale-diagonal >> "$LOGFILE" 2>&1
log "xnli analyze rc=$?"

log "=========================================="
log "STEP 5 (best-effort, lowest priority): Swahili 2x2 extension (svd_scale swahili_only_b seed2 + 12 merge_evals)"
log "=========================================="
conda run -n d2moe_env python -u run_phase1_2x2_lang.py \
  --lang-cond swahili_only --lang-cond-b swahili_only_b --lang-code swh_Latn >> "$LOGFILE" 2>&1
rc=$?
log "2x2-lang swahili rc=$rc"
if [ $rc -eq 0 ]; then
  conda run -n d2moe_env python -u phase1_2x2_gate_lang.py \
    --lang-cond swahili_only --lang-cond-b swahili_only_b --lang-code swh_Latn >> "$LOGFILE" 2>&1
  log "2x2-gate swahili rc=$?"
fi

log "GPU0,1 QUEUE FULLY DONE"
