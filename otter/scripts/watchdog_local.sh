#!/bin/bash
LOCK=/tmp/watchdog_local.lock
[ -f "$LOCK" ] && [ $(($(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0))) -lt 300 ] && exit 0
touch "$LOCK"

AVAIL_GB=$(df --output=avail -BG /mnt/HDD | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 15 ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog: /mnt/HDD only ${AVAIL_GB}G free, NOT relaunching (would likely crash again), needs manual cleanup" >> /home/minjeong/project/FERRET/otter/watchdog_local.log
  rm -f "$LOCK"
  exit 0
fi

if ! pgrep -f "run_phase1_41_diagonal.py --conditions mixed_5lang" >/dev/null; then
  cd /home/minjeong/project/FERRET/otter/scripts || exit 1
  nohup conda run -n d2moe_env python run_phase1_41_diagonal.py --conditions mixed_5lang --skip-wait >> /home/minjeong/project/FERRET/otter/logs_41_local_gpu01.log 2>&1 &
  disown
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog: mixed_5lang was dead, relaunched (avail=${AVAIL_GB}G)" >> /home/minjeong/project/FERRET/otter/watchdog_local.log
fi
rm -f "$LOCK"
