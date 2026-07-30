#!/bin/bash
# §4.2 grid expansion (2026-07-30 evening prompt): korean_only-only Belebele
# result -> at least 3 conditions (english_only, mixed_5lang, +chinese_only
# best-effort), eng/kor/zho only (swh/ben pre-excluded, at-chance in
# baseline), pp_ratio OFF only (ON is §4.4 ablation scope, not this grid).
#
# Gated on the whitening_geometry priority-1 job (phase1_whitening_geometry.py
# -> _analysis.py -> make_figure_whitening_geometry.py, see
# wait_and_run_whitening_geometry.sh) finishing first -- same OOM precedent
# (belebele_grid_korean_only.log, exit 137) that motivated
# run_phase1_belebele_grid.py's per-job flock. NOTE: the ".pt 정리"
# (svd_scale_processed.pt cleanup) half of that precondition has no script/
# manifest anywhere in this repo as of 2026-07-30 -- this wrapper gates ONLY
# on the whitening_geometry process chain exiting, logs a disk snapshot for
# visibility, and does NOT wait on or perform any cleanup. See the milestone
# log for this caveat spelled out at runtime.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FULL_LOG="/home/minjeong/project/FERRET/otter/logs/belebele_extended_0730.log"
MILESTONE_LOG="/home/minjeong/project/FERRET/otter/scripts/logs/belebele_extended_milestones.log"
WAIT_PATTERN='phase1_whitening_geometry\.py|phase1_whitening_geometry_analysis\.py|make_figure_whitening_geometry\.py'
MAX_WAIT_ITERS=480   # 480 * 60s = 8h safety cap, same as wait_and_run_whitening_geometry.sh

mkdir -p "$(dirname "$MILESTONE_LOG")"

milestone() {
  local line="[belebele-ext] $(date '+%F %T') $1"
  echo "$line" | tee -a "$MILESTONE_LOG"
}

milestone "queued -- waiting for whitening_geometry chain to finish"

i=0
while true; do
  n_running=$(pgrep -f "$WAIT_PATTERN" | wc -l)
  if [ "$n_running" -eq 0 ]; then
    break
  fi
  i=$((i+1))
  if [ "$i" -ge "$MAX_WAIT_ITERS" ]; then
    milestone "GAVE UP after $MAX_WAIT_ITERS checks (whitening_geometry chain still running, n_running=$n_running) -- NOT starting Belebele grid. Re-run this script manually once it's actually done."
    exit 1
  fi
  sleep 60
done

milestone "whitening_geometry chain finished (n_running=0)"
milestone "disk snapshot: $(df -h / /mnt/HDD 2>/dev/null | tr '\n' ' ')"
milestone "NOTE: .pt cleanup status NOT verified (no cleanup script found in repo) -- proceeding on whitening_geometry completion alone; english_only/mixed_5lang/chinese_only merges are in-memory only (no new large .pt writes), so this should not need cleanup headroom regardless"

source /home/minjeong/anaconda3/etc/profile.d/conda.sh
conda activate d2moe_env

run_condition() {
  local cond="$1"
  local priority="$2"
  milestone "=== starting $cond (seeds 0,1,2; eng_Latn/kor_Hang/zho_Hans; pp_ratio OFF only) [$priority] ==="
  local start_line
  start_line=$(wc -l < "$FULL_LOG" 2>/dev/null || echo 0)
  conda run -n d2moe_env python -u run_phase1_belebele_grid.py \
    --conditions "$cond" --seeds 0 1 2 --limit 200 --batch-size 1 --num-fewshot 5 \
    --langs eng_Latn kor_Hang zho_Hans --off-only \
    >> "$FULL_LOG" 2>&1
  rc=$?
  # run_phase1_belebele_grid.py catches per-seed CalledProcessError internally and
  # keeps going (rc=0 even if a seed OOMed) -- grep this condition's own log slice
  # for its per-seed FAILED markers so a swallowed OOM doesn't read as a clean DONE.
  local seed_failures
  seed_failures=$(tail -n +"$((start_line + 1))" "$FULL_LOG" | grep -c "FAILED (rc=" || true)
  if [ "$rc" -ne 0 ]; then
    milestone "$cond FAILED (rc=$rc) -- see $FULL_LOG -- continuing to next condition"
  elif [ "${seed_failures:-0}" -gt 0 ]; then
    milestone "$cond DONE (rc=0) but $seed_failures seed-level FAILED marker(s) inside it -- see $FULL_LOG, results for this condition are PARTIAL"
  else
    milestone "$cond DONE (rc=0, all seeds clean)"
  fi
}

run_condition english_only "priority-1, own-language diagonal control"
run_condition mixed_5lang "priority-2"
run_condition chinese_only "best-effort, low priority -- completes the 3x3 grid"

milestone "ALL QUEUED CONDITIONS DONE (english_only, mixed_5lang, chinese_only[best-effort]) -- run phase1_belebele_gate.py per condition to summarize, then the own-language-diagonal check"
