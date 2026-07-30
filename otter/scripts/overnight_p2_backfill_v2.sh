#!/bin/bash
# 2026-07-29 야간 자동화: track_ab_prompt.txt "P2 backfill"의 잔여분.
# 원래 백필 프로세스(otter/scripts/logs/phase1_6_p2_backfill.log를 쓰던 것)가
# 세션 종료와 함께 죽어 korean_only seed2부터 미완료로 남음(ps aux로 확인,
# 2026-07-29 16:xx UTC). /tmp/p2_remaining.txt + eval_ppl.json의 ben_Beng 키
# 유무로 실제 남은 작업만 다시 계산해서 이어서 돈다(멱등, --scale-condition
# 없는 plain merge_eval -- Table 1 grid의 scale_* 버전과는 다른 산출물).
set -o pipefail
export HF_HOME=/mnt/HDD/minjeong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
mkdir -p "$LOGDIR"
MARKER="$LOGDIR/overnight_p2_backfill_v2.done"
DRIVERLOG="$LOGDIR/overnight_p2_backfill_v2.log"
rm -f "$MARKER"

log() { echo "[p2-backfill-v2 $(date -u +%H:%M:%S)] $*" | tee -a "$DRIVERLOG"; }

has_bengali() {
  # $1 = eval_ppl.json path
  [ -f "$1" ] && python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
except Exception:
    sys.exit(1)
sys.exit(0 if any('ben' in str(k).lower() or 'bengali' in str(k).lower() for k in d.keys()) else 1)
" 2>/dev/null
}

run_one() {
  local cond="$1" seed="$2"
  local f="/mnt/HDD/minjeong/d2moe_results/phase1/$cond/seed$seed/eval_ppl.json"
  if has_bengali "$f"; then
    log "$cond seed$seed: already has Bengali, skipping"
    return 0
  fi
  local logfile="$LOGDIR/p2_backfill_v2_${cond}_seed${seed}.log"
  log "$cond seed$seed: starting plain merge_eval -> $logfile"
  # GPU2,3 pinned to match this queue's established pairing tonight (GPU0,1
  # is the belebele/xnli queue's pair) -- CUDA_VISIBLE_DEVICES set explicitly
  # here (not safe_gpus.sh) because this script predates that helper's use in
  # this family and must not accidentally grab GPU0,1 mid-belebele-eval.
  flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES=2,3 \
    conda run -n d2moe_env python -u phase1_merge_eval.py \
    --condition "$cond" --seed "$seed" > "$logfile" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    log "$cond seed$seed: DONE (rc=0)"
  else
    log "$cond seed$seed: FAILED (rc=$rc) -- see $logfile -- continuing to next job"
  fi
}

for pair in "english_only 2" "korean_only 2" "chinese_only 0" "chinese_only 1" "chinese_only 2" "swahili_only 0" "swahili_only 1" "swahili_only 2"; do
  set -- $pair
  run_one "$1" "$2"
done

log "ALL DONE"
touch "$MARKER"
