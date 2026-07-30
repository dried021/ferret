"""Bengali full grid -- the remaining work after the 2026-07-27 smoke test
(see claude_plan.md D-8: fragmentation check 1.453 bytes/token ~ Korean's
1.433, bengali_only/seed0 bpb increase +17.68% ~ English's +16.95%, well
below Swahili's +25.71% -- "판단: 스왑 불필요, Bengali 유지"). The smoke test
only ran bengali_only/seed0; own-language gain needs the same three things
Swahili's own_gain check needed (see phase1_swahili_gate.py's own docstring
and run_phase1_swahili.py, whose task-list structure this mirrors exactly,
including the two-GPU-pair parallel split that worked cleanly for Swahili):

  - bengali_only x3 seeds (seed0 already done by the smoke test; seed1,2 new)
  - bengali_only_b x3 seeds (placebo arm, same disjoint-half-pool design as
    english_only_b/korean_only_b/swahili_only_b) -- entirely new
  - english_only x3 seeds, "eval-only" rerun -- english_only/seed{0,1,2}'s
    eval_ppl.json predates ben_Beng being added to EVAL_LANGS (confirmed by
    inspection 2026-07-27: those 3 files have keys {eng_Latn,kor_Hang,
    zho_Hans,swh_Latn}, no ben_Beng), so english_only's Bengali score doesn't
    exist yet even though its Fisher/expert_freq artifacts do -- this reuses
    them (merge_condition() is deterministic given the same artifacts) and
    just re-runs merge+eval to add the ben_Beng field.
  - baseline: already has ben_Beng (confirmed 2026-07-27, the smoke test's
    baseline re-eval already covered this) -- included here anyway with the
    same already-has-language skip check run_phase1_swahili.py uses, so this
    script is correct to run standalone even if that smoke-test artifact
    somehow went missing.

own-language gain(Bengali) will be computed by phase1_bengali_gate.py as:
    bpb_increase(english_only, seed, ben) - bpb_increase(bengali_only, seed, ben)
noise floor: max seed |bengali_only - bengali_only_b| (same definition as
phase1_placebo_gate.py / phase1_swahili_gate.py).

Usage:
    conda run -n d2moe_env python run_phase1_bengali_grid.py --stream a
    conda run -n d2moe_env python run_phase1_bengali_grid.py --stream b
(run both, in background, at the same time, on disjoint GPU pairs -- see
--gpus-a/--gpus-b; defaults match run_phase1_swahili.py's 0,1 / 2,3 split.
Only safe if all 4 GPUs are actually free -- check nvidia-smi first, same as
the Swahili run did; otherwise run stream a and b sequentially with the
default MAX_GPUS=2 policy instead of in parallel.)
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
# bengali_only/seed0 is deliberately omitted -- the 2026-07-27 smoke test
# already produced it (see phase1_bengali_fragmentation_check.py's sibling
# run_phase1_bengali_smoke.py); ensure_done() below would skip it anyway,
# but leaving it out of the plan makes the already-done state visible here
# rather than only inside a skip-check at runtime.
STREAM_A = [
    ("baseline", None, None),
    ("eval_only", "english_only", 0),
    ("eval_only", "english_only", 1),
    ("full", "bengali_only", 1),
    ("full", "bengali_only_b", 0),
]
STREAM_B = [
    ("eval_only", "english_only", 2),
    ("full", "bengali_only", 2),
    ("full", "bengali_only_b", 1),
    ("full", "bengali_only_b", 2),
]


def check_gpus_safe(gpu_csv):
    """Same ownership re-check as run_phase1_swahili.py -- never trust a
    GPU-ownership read from more than one command ago, and never silently
    substitute a different GPU than what the two parallel streams agreed on."""
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
    print(f"[bengali-grid] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def already_has_bengali(path):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return "ben_Beng" in data
    except Exception:
        return False


def do_task(task_type, condition, seed, gpus):
    if task_type == "baseline":
        out = RESULTS_ROOT / "baseline" / "eval_ppl.json"
        if already_has_bengali(out):
            print(f"[bengali-grid] baseline already has ben_Beng, skipping")
            return
        run(["phase1_merge_eval.py", "--baseline"], gpus)

    elif task_type == "eval_only":
        out = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
        if already_has_bengali(out):
            print(f"[bengali-grid] {condition}/seed{seed} already has ben_Beng, skipping")
            return
        run(["phase1_merge_eval.py", "--condition", condition, "--seed", str(seed)], gpus)

    elif task_type == "full":
        out = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
        if already_has_bengali(out):
            print(f"[bengali-grid] {condition}/seed{seed} already done, skipping")
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

    print(f"[bengali-grid] stream {args.stream} done")


if __name__ == "__main__":
    main()
