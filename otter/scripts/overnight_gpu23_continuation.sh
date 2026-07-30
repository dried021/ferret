#!/bin/bash
# 2026-07-29 야간 자동화: GPU2,3 큐(메인 결과 뱅골어 백필 + P2 backfill)가
# 둘 다 끝나면 자동으로 이어서: §4.1 headline gate 최종본 -> Track B
# (interference_model) -> Track A/B 게이트(budget_gate_v2). 새 조건의 GPU
# 파이프라인(freq+fisher+merge_eval 새로 도는 것)은 여기서 자동 실행하지
# 않음 -- 어느 조건(vulnerability_targeted/abs/random, interference_minimax
# 등)을 실제로 돌릴지는 이 세 스크립트의 결과를 봐야 판단 가능한 사용자
# 판단 영역이라 pending 상태로 남겨둠 (0730_0057_track_ab_prompt.txt §3 참고).
set -o pipefail
cd /home/minjeong/project/FERRET/otter/scripts || exit 1
LOGDIR=/home/minjeong/project/FERRET/otter/logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/overnight_gpu23_continuation.log"

log() { echo "[gpu23-continuation $(date -u +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

log "waiting for pending_merge_eval driver (6 Bengali cells) AND p2_backfill_v2 (7 cells) to both finish..."
while true; do
  driver_done=0
  grep -q "ALL DONE" "$LOGDIR/pending_merge_eval_driver.log" 2>/dev/null && driver_done=1
  backfill_done=0
  [ -f "$LOGDIR/overnight_p2_backfill_v2.done" ] && backfill_done=1
  if [ "$driver_done" -eq 1 ] && [ "$backfill_done" -eq 1 ]; then
    break
  fi
  sleep 120
done
log "both queues drained -- proceeding"

log "=== phase1_41_headline_gate.py ==="
conda run -n d2moe_env python -u phase1_41_headline_gate.py >> "$LOGFILE" 2>&1
log "headline_gate rc=$?"

log "=== phase1_6_interference_model.py --seed 0 ==="
conda run -n d2moe_env python -u phase1_6_interference_model.py --seed 0 >> "$LOGFILE" 2>&1
log "interference_model rc=$?"

log "=== phase1_6_budget_gate_v2.py ==="
conda run -n d2moe_env python -u phase1_6_budget_gate_v2.py >> "$LOGFILE" 2>&1
log "budget_gate_v2 rc=$?"

log "GPU2,3 QUEUE FULLY DONE -- review $LOGFILE for headline/Track A/B verdicts. Next step (which condition's GPU pipeline to run, if any) needs a human/Claude read of this output, not automated."
