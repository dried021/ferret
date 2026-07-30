#!/usr/bin/env bash
# mixed_5lang (EN/KO/ZH/Swahili/Bengali, equal-character-budget mix) full
# grid: freq + real Fisher + merge+eval, seed 0/1/2 (2026-07-27).
#
# Why this exists: phase1_calib_data.py already defines "mixed_5lang" (added
# 2026-07-27 code review, see its docstring/CHAR_BALANCED_MIXES) as a
# SEPARATE condition from "balanced" (the old EN/KO/ZH-only mix, kept
# untouched for reproducibility of its already-published 3.7/3.8 results).
# claude_plan.md's "확정된 실험 조건" section requires §4.1/§6 comparisons
# that include the low-resource languages (Swahili/Bengali) to use
# mixed_5lang, not the old 3-language "balanced" -- but mixed_5lang itself
# had never been run (no results directory existed as of the 2026-07-27
# review). This script is the runner that was missing, not a new experiment
# design -- it follows run_phase1_seeds.sh's exact freq->fisher->merge+eval
# sequence, just for one condition across 3 seeds instead of a
# CONDITIONS array.
#
# Deliberately NOT using whitening (svd_scale) or pp_ratio pruning here --
# every other §4.1 single/mixed-language condition this compares against
# (english_only, korean_only, ..., balanced) was also run with plain SVD, no
# pruning (see phase1_merge_eval.py's module docstring, "two deliberate
# scope cuts"). Running mixed_5lang any differently would make it an unfair
# comparison. If/when §4.1 moves to the full Fisher+whitening+pruning
# pipeline (claude_plan.md D-7), every condition -- including this one --
# needs to be re-run together under that pipeline, not just this one.
#
# GPU safety: safe_gpus.sh re-sourced before every invocation (see
# run_phase1_seeds.sh for why -- never trust a GPU-ownership check from more
# than one command ago).
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(0 1 2)
COND=mixed_5lang

RESULTS_ROOT=/mnt/HDD/minjeong/d2moe_results/phase1

run() {
    source ./safe_gpus.sh
    echo "[run_phase1_mixed5lang] + $*" >&2
    CUDA_VISIBLE_DEVICES="$SAFE_GPUS" HF_HOME=/mnt/HDD/minjeong/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n d2moe_env python "$@"
    sleep 10
}

if [ ! -f "$RESULTS_ROOT/baseline/eval_ppl.json" ]; then
    run phase1_merge_eval.py --baseline
else
    echo "[run_phase1_mixed5lang] baseline already done, skipping" >&2
fi

for seed in "${SEEDS[@]}"; do
    out_json="$RESULTS_ROOT/$COND/seed$seed/eval_ppl.json"
    if [ -f "$out_json" ]; then
        echo "=== seed=$seed condition=$COND: already done ($out_json), skipping ===" >&2
        continue
    fi
    echo "=== seed=$seed condition=$COND: freq ===" >&2
    run phase1_run_freq_and_scale.py --condition "$COND" --seed "$seed"
    echo "=== seed=$seed condition=$COND: fisher (n_samples=64, seqlen=512) ===" >&2
    run phase1_fisher.py --condition "$COND" --seed "$seed" --n-samples 64 --seqlen 512
    echo "=== seed=$seed condition=$COND: merge+eval ===" >&2
    run phase1_merge_eval.py --condition "$COND" --seed "$seed"
done

echo "[run_phase1_mixed5lang] all seeds done" >&2
