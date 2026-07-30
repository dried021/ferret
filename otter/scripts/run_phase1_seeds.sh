#!/usr/bin/env bash
# Phase 1 budget-normalized + seed-replicated run (2026-07-24).
# Requested by user after reviewing the first pilot's seed=1, 16-sample/
# seqlen-256 result: normalize calibration budget to the scripts' own
# defaults (n_samples=128, seqlen=512 -- already above the "minimum 64x512"
# ask) and repeat across 3 seeds so condition-vs-noise can be told apart.
#
# Also picks up two code fixes made alongside this budget change:
#   - calibration/evaluation FLORES sentence overlap removed (phase1_calib_data.py)
#   - eval_flores_ppl now reports bits-per-byte natively, not just per-token PPL
#     (00_docs/03_기술노트.md "Phase 1 pilot" -- Korean's per-token PPL was
#     mechanically deflated by tokenizer byte-fallback on Hangul)
#
# GPU safety: safe_gpus.sh is re-sourced before every single invocation (not
# just once at the top), matching this project's standing policy of never
# trusting a GPU-ownership check from more than one command ago.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(0 1 2)
CONDITIONS=(english_only korean_only chinese_only balanced)

RESULTS_ROOT=/mnt/HDD/minjeong/d2moe_results/phase1

run() {
    source ./safe_gpus.sh
    echo "[run_phase1_seeds] + $*" >&2
    CUDA_VISIBLE_DEVICES="$SAFE_GPUS" HF_HOME=/mnt/HDD/minjeong/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n d2moe_env python "$@"
    # 2026-07-25: a prior run OOM'd when the next `conda run` started before
    # the previous process's CUDA context had fully released (transient --
    # confirmed via nvidia-smi that no process, ours or anyone else's, was
    # actually still resident). A short pause here is cheap insurance against
    # that race recurring across 12 sequential model loads.
    sleep 10
}

# Baseline (uncompressed) doesn't depend on calibration seed -- run once.
if [ ! -f "$RESULTS_ROOT/baseline/eval_ppl.json" ]; then
    run phase1_merge_eval.py --baseline
else
    echo "[run_phase1_seeds] baseline already done, skipping" >&2
fi

for seed in "${SEEDS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        out_json="$RESULTS_ROOT/$cond/seed$seed/eval_ppl.json"
        if [ -f "$out_json" ]; then
            echo "=== seed=$seed condition=$cond: already done ($out_json), skipping ===" >&2
            continue
        fi
        echo "=== seed=$seed condition=$cond: freq ===" >&2
        run phase1_run_freq_and_scale.py --condition "$cond" --seed "$seed"
        echo "=== seed=$seed condition=$cond: fisher (n_samples=64, seqlen=512 -- user-specified minimum budget) ===" >&2
        run phase1_fisher.py --condition "$cond" --seed "$seed" --n-samples 64 --seqlen 512
        echo "=== seed=$seed condition=$cond: merge+eval ===" >&2
        run phase1_merge_eval.py --condition "$cond" --seed "$seed"
    done
done

echo "[run_phase1_seeds] all seeds/conditions done" >&2
