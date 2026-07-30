#!/bin/bash
# STEP 3 driver: 6 compression conditions x seed0, EN/ZH/SW n=200, own-language
# whitened SVD (--scale-condition == --condition, --scale-seed 0), matching the
# existing v1 result-directory convention (scale_<condition>_seed0/). Each
# condition acquires/releases /tmp/phase1_merge_eval.lock separately (not held
# across the whole loop) so other concurrent sessions get a fair turn between
# conditions, same convention as every other *.sh driver in this directory.
cd /home/minjeong/project/FERRET/otter/scripts
CONDITIONS="english_only korean_only chinese_only swahili_only bengali_only mixed_5lang"
for cond in $CONDITIONS; do
  echo "[step3] === starting $cond $(date -u +%FT%TZ) ==="
  flock /tmp/phase1_merge_eval.lock env CUDA_VISIBLE_DEVICES=1,2,3 conda run -n d2moe_env python -u \
    phase1_xnli_eval_v2.py --condition "$cond" --seed 0 --scale-condition "$cond" --scale-seed 0 --limit 200 \
    > "logs/xnli_v2_step3_${cond}.log" 2>&1
  rc=$?
  echo "[step3] === finished $cond rc=$rc $(date -u +%FT%TZ) ==="
done
echo "[step3] ALL DONE"
