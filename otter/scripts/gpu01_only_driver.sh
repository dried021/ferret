#!/bin/bash
# 2026-07-30: 사용자 지시 -- GPU2,3은 무조건 비워두고 GPU0,1만 계속 돌릴 것.
# run_phase1_2x2_lang.py/pilot_interference_minimax_seed0.sh 둘 다 안에서
# safe_gpus.sh(가장 낮은 인덱스 2장, 다른 유저가 안 쓰는 GPU만)로 GPU를 고르는데
# GPU2,3에 아무것도 안 띄워두는 한 항상 0,1이 선택된다 -- 파일럿 스크립트는
# CUDA_VISIBLE_DEVICES를 아예 0,1로 하드코딩해서 이중으로 보장.
set -o pipefail
cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
LOGFILE="$LOGDIR/gpu01_only_driver.log"

log() { echo "[gpu01-only $(date -u +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

log "=== STEP 1: 4.3 Swahili 2x2 grid 재개 (run_phase1_2x2_lang.py) ==="
conda run -n d2moe_env python -u run_phase1_2x2_lang.py \
  --lang-cond swahili_only --lang-cond-b swahili_only_b --lang-code swh_Latn \
  >> "$LOGFILE" 2>&1
rc=$?
log "2x2-lang swahili rc=$rc"
if [ $rc -eq 0 ]; then
  conda run -n d2moe_env python -u phase1_2x2_gate_lang.py \
    --lang-cond swahili_only --lang-cond-b swahili_only_b --lang-code swh_Latn >> "$LOGFILE" 2>&1
  log "2x2-gate swahili rc=$?"
fi

log "=== STEP 2: interference_minimax seed0 재개 (fisher 자동삭제 포함, run_track_ab_candidate.sh) ==="
./run_track_ab_candidate.sh interference_minimax 0
log "track-ab interference_minimax seed0 rc=$?"

log "GPU0,1 ONLY DRIVER DONE"
