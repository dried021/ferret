"""Placebo run: english_only_b + korean_only_b, 3 seeds each -- reuses
phase1_run_freq_and_scale.py / phase1_fisher.py / phase1_merge_eval.py
unmodified (their --condition argparse choices already include the new
"_b" placebo conditions via phase1_calib_data.CONDITIONS).

This is a thin Python wrapper (not another shell script) so it can be
resumed the same way run_phase1_seeds.sh was: skip any (condition, seed)
whose eval_ppl.json already exists.
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
CONDITIONS = ["english_only_b", "korean_only_b"]
SEEDS = [0, 1, 2]


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[placebo] + {' '.join(args)} (SAFE_GPUS={gpus})", flush=True)
    env = {"CUDA_VISIBLE_DEVICES": gpus, "HF_HOME": "/mnt/HDD/minjeong/hf_cache",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    import os
    full_env = dict(os.environ, **env)
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=full_env, check=True)
    time.sleep(10)  # let the previous process's CUDA context fully release (2026-07-25 incident)


def main():
    for cond in CONDITIONS:
        for seed in SEEDS:
            out_json = RESULTS_ROOT / cond / f"seed{seed}" / "eval_ppl.json"
            if out_json.exists():
                print(f"[placebo] {cond}/seed{seed} already done, skipping")
                continue
            print(f"=== {cond} seed={seed}: freq ===")
            run(["phase1_run_freq_and_scale.py", "--condition", cond, "--seed", str(seed)])
            print(f"=== {cond} seed={seed}: fisher (n_samples=64, seqlen=512) ===")
            run(["phase1_fisher.py", "--condition", cond, "--seed", str(seed),
                 "--n-samples", "64", "--seqlen", "512"])
            print(f"=== {cond} seed={seed}: merge+eval ===")
            run(["phase1_merge_eval.py", "--condition", cond, "--seed", str(seed)])
    print("[placebo] all placebo runs done")


if __name__ == "__main__":
    main()
