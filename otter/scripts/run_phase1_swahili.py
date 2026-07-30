"""Swahili low-resource-language experiment, parallelized across two GPU
pairs (2026-07-25, user asked to use all 4 GPUs -- temporary override of the
usual MAX_GPUS=2 policy, confirmed all 4 GPUs free via nvidia-smi first).

Tasks (10 total), split into two streams that run concurrently on disjoint
GPU pairs so they never contend for the same device:

  - swahili_only x3 seeds (full: freq+fisher+merge+eval) -- arm A
  - swahili_only_b x3 seeds (full) -- arm B placebo, same disjoint-half-pool
    design as english_only_b/korean_only_b
  - baseline re-eval (quick: eval_flores_ppl() now includes swh_Latn in
    EVAL_LANGS, needs re-running once to add that field -- EN/KO/ZH values
    are deterministic and will come out identical)
  - english_only x3 seeds, "eval-only" rerun (quick: merge_condition() reuses
    the ALREADY-COMPUTED fisher_processed.pt/expert_freq files from the
    original Phase 1 run, so this just redoes the merge (deterministic, same
    numbers) + eval, now scoring Swahili too -- this is the "distant/default
    calibration" reference point for computing Swahili's own-language gain,
    without paying for a full freq+fisher recompute)

own-language gain(Swahili) will be computed by phase1_swahili_gate.py as:
    bpb_increase(english_only, seed, swh) - bpb_increase(swahili_only, seed, swh)
noise floor: max seed |swahili_only - swahili_only_b| (same definition as
phase1_placebo_gate.py).

Usage:
    conda run -n d2moe_env python run_phase1_swahili.py --stream a
    conda run -n d2moe_env python run_phase1_swahili.py --stream b
(run both, in background, at the same time -- see the launch commands used)
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")

# (task_type, condition, seed) -- task_type in {"full", "eval_only", "baseline"}
STREAM_A = [
    ("baseline", None, None),
    ("eval_only", "english_only", 0),
    ("eval_only", "english_only", 1),
    ("full", "swahili_only", 0),
    ("full", "swahili_only", 1),
    ("full", "swahili_only_b", 0),
]
STREAM_B = [
    ("eval_only", "english_only", 2),
    ("full", "swahili_only", 2),
    ("full", "swahili_only_b", 1),
    ("full", "swahili_only_b", 2),
]


def check_gpus_safe(gpu_csv):
    """Re-checks (every call, not just once at startup) that none of the
    requested GPU indices are running another user's process -- same
    ownership check as safe_gpus.sh, applied to this run's specific pair
    instead of auto-selecting the lowest-N. Raises if any requested GPU is
    not safe, rather than silently substituting a different one (silently
    picking a different GPU than what the two parallel streams agreed on
    could make them collide on the same device)."""
    me = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout
    uuid_to_idx = {}
    idx_out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout
    for line in idx_out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        uuid_to_idx[uuid] = idx

    requested = set(gpu_csv.split(","))
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [x.strip() for x in line.split(",")]
        gpu_idx = uuid_to_idx.get(uuid)
        if gpu_idx not in requested:
            continue
        owner = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if owner and owner != me:
            raise RuntimeError(f"GPU {gpu_idx} has a process owned by '{owner}' (pid {pid}), not '{me}' -- "
                                f"refusing to use it. Requested pair was {gpu_csv}.")


def run(args, gpus):
    check_gpus_safe(gpus)
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[swahili] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)  # let the previous process's CUDA context fully release


def already_has_swahili(path):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return "swh_Latn" in data
    except Exception:
        return False


def do_task(task_type, condition, seed, gpus):
    if task_type == "baseline":
        out = RESULTS_ROOT / "baseline" / "eval_ppl.json"
        if already_has_swahili(out):
            print(f"[swahili] baseline already has swh_Latn, skipping")
            return
        run(["phase1_merge_eval.py", "--baseline"], gpus)

    elif task_type == "eval_only":
        out = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
        if already_has_swahili(out):
            print(f"[swahili] {condition}/seed{seed} already has swh_Latn, skipping")
            return
        run(["phase1_merge_eval.py", "--condition", condition, "--seed", str(seed)], gpus)

    elif task_type == "full":
        out = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
        if already_has_swahili(out):
            print(f"[swahili] {condition}/seed{seed} already done, skipping")
            return
        run(["phase1_run_freq_and_scale.py", "--condition", condition, "--seed", str(seed)], gpus)
        run(["phase1_fisher.py", "--condition", condition, "--seed", str(seed),
             "--n-samples", "64", "--seqlen", "512"], gpus)
        run(["phase1_merge_eval.py", "--condition", condition, "--seed", str(seed)], gpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", choices=["a", "b"], required=True)
    parser.add_argument("--gpus-a", default="0,1")
    parser.add_argument("--gpus-b", default="2,3")
    args = parser.parse_args()

    tasks = STREAM_A if args.stream == "a" else STREAM_B
    gpus = args.gpus_a if args.stream == "a" else args.gpus_b

    for task_type, condition, seed in tasks:
        label = condition or "baseline"
        print(f"=== stream {args.stream} ({gpus}): {label} seed={seed} type={task_type} ===", flush=True)
        do_task(task_type, condition, seed, gpus)

    print(f"[swahili] stream {args.stream} done")


if __name__ == "__main__":
    main()
