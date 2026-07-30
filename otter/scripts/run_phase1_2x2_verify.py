"""Seed-replication + placebo verification for the Fisher x Scale 2x2
(2026-07-26, see 00_docs/03_기술노트.md "2) whitening 복원" -- the initial
2x2 was seed=1, no placebo, flagged PRELIMINARY. This script brings it to
the same rigor as the main Phase 1 result: 3 seeds + a real placebo).

Tasks (24 total, resumable -- skips anything already on disk):
  A) svd_scale for english_only/korean_only, seeds 1,2 (seed 0 already done)
  B) svd_scale for english_only_b/korean_only_b (placebo arm), seeds 0,1,2
  C) merge+eval for all 4 Fisher x Scale cells, seeds 1,2 (seed 0 already done)
  D) merge+eval for Fisher=korean_only x Scale in {english_only_b, korean_only_b},
     seeds 0,1,2 -- the placebo comparison, mirroring phase1_placebo_gate.py's
     noise-floor definition but for the SCALE axis instead of the Fisher axis.
     Fisher is fixed at korean_only (not also varied) since that's the
     confirmed/robust Fisher condition and the core claim under test is
     "does own_scale_gain(KO | Fisher=KO) survive a real placebo."

GPU policy: standard MAX_GPUS=2 (safe_gpus.sh default) -- GPU 3 has another
user's job right now, and 3 free GPUs (0,1,2) don't split evenly into two
2-GPU streams, so this runs single-stream/sequential rather than the 4-GPU
parallel trick used for the Swahili run.

Usage: conda run -n d2moe_env python run_phase1_2x2_verify.py
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]

SCALE_TASKS = [
    ("english_only", 1), ("english_only", 2),
    ("korean_only", 1), ("korean_only", 2),
    ("english_only_b", 0), ("english_only_b", 1), ("english_only_b", 2),
    ("korean_only_b", 0), ("korean_only_b", 1), ("korean_only_b", 2),
]

MERGE_EVAL_MAIN = [  # (fisher_cond, scale_cond, seed) -- seed 0 already done for these 4 combos
    (f, s, seed)
    for f in ("english_only", "korean_only")
    for s in ("english_only", "korean_only")
    for seed in (1, 2)
]
MERGE_EVAL_PLACEBO = [  # Fisher fixed at korean_only, Scale = placebo arm, all 3 seeds
    ("korean_only", "english_only_b", seed) for seed in SEEDS
] + [
    ("korean_only", "korean_only_b", seed) for seed in SEEDS
]


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[2x2-verify] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def main():
    print(f"[2x2-verify] plan: {len(SCALE_TASKS)} scale computations, "
          f"{len(MERGE_EVAL_MAIN)} main-grid merge+eval, {len(MERGE_EVAL_PLACEBO)} placebo merge+eval")

    for cond, seed in SCALE_TASKS:
        out_dir = RESULTS_ROOT / cond / f"seed{seed}"
        out_path = out_dir / "svd_scale_processed.pt"
        layer_dir = out_dir / "svd_scale_layers"
        if out_path.exists():
            print(f"[2x2-verify] scale {cond}/seed{seed} already done, skipping")
            if layer_dir.exists():
                print(f"[2x2-verify] cleaning up stale {layer_dir}")
                shutil.rmtree(layer_dir)
            continue
        run(["phase1_svd_scale.py", "--condition", cond, "--seed", str(seed),
             "--n-samples", "64", "--seqlen", "512"])
        # svd_scale_layers/ is a disposable per-layer checkpoint dir (~34GB) --
        # the disk-full crash on 2026-07-26 was caused by these never being
        # cleaned up across many conditions/seeds. Remove it immediately once
        # the final assembled file is confirmed present.
        if out_path.exists() and layer_dir.exists():
            print(f"[2x2-verify] scale {cond}/seed{seed} done, cleaning up {layer_dir}")
            shutil.rmtree(layer_dir)

    for fisher_cond, scale_cond, seed in MERGE_EVAL_MAIN + MERGE_EVAL_PLACEBO:
        out_json = (RESULTS_ROOT / fisher_cond / f"seed{seed}" /
                    f"scale_{scale_cond}_seed{seed}" / "eval_ppl.json")
        if out_json.exists():
            print(f"[2x2-verify] merge+eval Fisher={fisher_cond} Scale={scale_cond} seed={seed} "
                  f"already done, skipping")
            continue
        run(["phase1_merge_eval.py", "--condition", fisher_cond, "--seed", str(seed),
             "--scale-condition", scale_cond, "--scale-seed", str(seed)])

    print("[2x2-verify] all tasks done")


if __name__ == "__main__":
    main()
