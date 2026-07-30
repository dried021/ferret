#!/bin/bash
# 1Hz live status view: timestamp + gpustat + currently running phase1_*.py task
# (its log tail + elapsed time). Meant to be run interactively over an SSH
# session to the GPU node itself (e.g. `ssh theta5090` then run this), not
# through the Claude Code tool loop -- it's an unbounded terminal watch.
#
# Usage on the node:
#   bash watch_theta.sh                 # watches /root/*.log by default
#   bash watch_theta.sh /path/to/*.log  # watch a specific log glob instead

LOG_GLOB="${1:-/root/*.log}"

while true; do
  clear
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo
  /opt/conda/bin/gpustat --no-header 2>/dev/null || nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
  echo
  echo "--- running phase1_*.py / lm-eval processes ---"
  ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | grep -E "phase1_|belebele" | grep -v grep | head -5
  echo
  latest_log=$(ls -t $LOG_GLOB 2>/dev/null | head -1)
  if [ -n "$latest_log" ]; then
    echo "--- tail: $latest_log ---"
    tail -c 400 "$latest_log" | tr '\r' '\n' | tail -5
  else
    echo "(no log matching $LOG_GLOB found)"
  fi
  sleep 1
done
