"""Fisher x Scale 2x2 (2026-07-26, see 00_docs/03_기술노트.md "2) whitening
복원"). Reuses ALREADY-COMPUTED english_only/korean_only Fisher+freq (seed0)
-- only computes the two NEW get_scale.py SVD_scale artifacts (now that its
memory leak is fixed), then runs the 4 merge+eval cells.

Cells: (Fisher, Scale) in {EN, KO} x {EN, KO}. Fisher=EN reuses english_only
seed0's data; Fisher=KO reuses korean_only seed0's. Scale=EN/KO are computed
fresh here via --include-scale (uses the SAME calibration text as the
matching Fisher condition/seed, since both draw from
phase1_calib_data.build_condition_sentences(condition, seed)).

Usage: conda run -n d2moe_env python run_phase1_2x2.py
"""
import subprocess
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEED = 0
FISHER_CONDITIONS = ["english_only", "korean_only"]  # already have freq+fisher from the main Phase 1 run
CELLS = [
    ("english_only", "english_only"),
    ("english_only", "korean_only"),
    ("korean_only", "english_only"),
    ("korean_only", "korean_only"),
]


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[2x2] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def main():
    # Step 1: compute svd_scale for EN and KO (64 samples/seqlen 512, same as Fisher's
    # budget) via phase1_svd_scale.py -- NOT the vendored get_scale.py, which has a
    # third, unrelated bug beyond the two memory leaks (see 00_docs/03_기술노트.md
    # "2) whitening 복원").
    for cond in FISHER_CONDITIONS:
        scale_path = RESULTS_ROOT / cond / f"seed{SEED}" / "svd_scale_processed.pt"
        if scale_path.exists():
            print(f"[2x2] {cond}/seed{SEED} already has svd_scale_processed.pt, skipping")
            continue
        run(["phase1_svd_scale.py", "--condition", cond, "--seed", str(SEED),
             "--n-samples", "64", "--seqlen", "512"])

    # Step 2: 4 merge+eval cells.
    for fisher_cond, scale_cond in CELLS:
        out_json = (RESULTS_ROOT / fisher_cond / f"seed{SEED}" /
                    f"scale_{scale_cond}_seed{SEED}" / "eval_ppl.json")
        if out_json.exists():
            print(f"[2x2] Fisher={fisher_cond} Scale={scale_cond} already done, skipping")
            continue
        run(["phase1_merge_eval.py", "--condition", fisher_cond, "--seed", str(SEED),
             "--scale-condition", scale_cond, "--scale-seed", str(SEED)])

    print("[2x2] all cells done")


if __name__ == "__main__":
    main()
