#!/bin/bash
set -uo pipefail
PATTERN='phase1_merge_eval\.py|phase1_belebele_eval\.py|run_phase1_belebele_grid\.py|phase1_6_p2_backfill'
MIN_AVAIL_MB=15000
MAX_WAIT_ITERS=480   # 480 * 60s = 8 hours safety cap
i=0
echo "[whitening-geom-wait] $(date '+%F %T') waiting for phase1_merge_eval/phase1_belebele_eval/backfill queue to clear..."
while true; do
  n_running=$(pgrep -f "$PATTERN" | wc -l)
  avail_mb=$(free -m | awk '/^Mem:/{print $7}')
  if [ "$n_running" -eq 0 ] && [ "$avail_mb" -gt "$MIN_AVAIL_MB" ]; then
    break
  fi
  i=$((i+1))
  if [ "$i" -ge "$MAX_WAIT_ITERS" ]; then
    echo "[whitening-geom-wait] $(date '+%F %T') gave up after $MAX_WAIT_ITERS checks (queue still running or memory still tight: n_running=$n_running avail_mb=$avail_mb) -- NOT running the extraction automatically. Re-run this script or phase1_whitening_geometry.py manually."
    exit 1
  fi
  sleep 60
done
echo "[whitening-geom-wait] $(date '+%F %T') queue clear (n_running=0, avail_mb=$avail_mb) -- starting whitening geometry extraction"
cd /home/minjeong/project/FERRET/otter/scripts
source /home/minjeong/anaconda3/etc/profile.d/conda.sh
conda run -n d2moe_env python -u phase1_whitening_geometry.py --num-threads 6
echo "[whitening-geom-wait] $(date '+%F %T') extraction done, running analysis + figures"
conda run -n d2moe_env python -u phase1_whitening_geometry_analysis.py
conda run -n d2moe_env python -u make_figure_whitening_geometry.py
echo "[whitening-geom-wait] $(date '+%F %T') ALL DONE"
