"""Bengali smoke test (D-8, 01_plans/claude_plan.md D-8 item: "Bengali 스모크
테스트 -- 아직 미착수"). Single-seed go/no-go run, meant to finish BEFORE the
D-7 main grid launches, to decide whether ben_Beng's bpb behaves abnormally
(tokenizer fragmentation -- see phase1_bengali_fragmentation_check.py for a
CPU-only, non-GPU pre-check of that same question) and needs swapping for
Vietnamese/Indonesian per the plan's stated fallback.

This is NOT the full 3-seed grid run_phase1_swahili.py did for Swahili --
just seed 0, since the point here is a fast keep/swap decision, not a
paper-reportable result. If Bengali is kept, the D-7 grid script covers the
remaining seeds (and bengali_only_b, needed for its own noise floor / gate,
mirroring phase1_swahili_gate.py) itself.

Tasks (2 total):
  - baseline re-eval (quick: eval_flores_ppl() already includes ben_Beng in
    EVAL_LANGS -- see phase1_merge_eval.py -- but baseline/eval_ppl.json
    hasn't been re-run since that field was added, same gap swh_Latn had
    before run_phase1_swahili.py). Gives baseline (uncompressed-model)
    ben_Beng bpb -- on its own already informative: if this is far higher
    than baseline eng/kor/zh/swh bpb, that's fragmentation showing up
    independent of any D2MoE compression.
  - bengali_only seed 0 (full: freq+scale, fisher, merge+eval). Its ben_Beng
    bpb increase over baseline (using the same baseline denominator as every
    other condition) is comparable to english_only/korean_only/etc.'s own
    bpb-increase numbers even though this smoke run doesn't recompute those.

own-language gain and a formal noise floor are NOT computed here (that needs
bengali_only_b and >=1 more seed, i.e. the full grid) -- this script only
produces the numbers needed to eyeball whether Bengali's bpb is sane before
committing D-7's budget to it.

GPU note (as of the 2026-07-27 investigation that flagged this task
unstarted): GPU 3 was idle, GPU 2 had another user's (non-FERRET) job running
at ~82% util -- check_gpus_safe() below will refuse a busy GPU rather than
silently substituting one, but re-verify with nvidia-smi before launching
since that snapshot will be stale by the time this runs.

Usage:
    conda run -n d2moe_env python run_phase1_bengali_smoke.py --gpus 2,3
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")


def check_gpus_safe(gpu_csv):
    """Re-checks (every call, not just once at startup) that none of the
    requested GPU indices are running another user's process -- same
    ownership check as run_phase1_swahili.py's. Raises rather than silently
    substituting a different GPU."""
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
                                f"refusing to use it. Requested was {gpu_csv}.")


def run(args, gpus):
    check_gpus_safe(gpus)
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[bengali-smoke] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)  # let the previous process's CUDA context fully release


def already_has_bengali(path):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return "ben_Beng" in data
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3")
    args = parser.parse_args()

    baseline_out = RESULTS_ROOT / "baseline" / "eval_ppl.json"
    if already_has_bengali(baseline_out):
        print("[bengali-smoke] baseline already has ben_Beng, skipping")
    else:
        run(["phase1_merge_eval.py", "--baseline"], args.gpus)

    seed0_out = RESULTS_ROOT / "bengali_only" / "seed0" / "eval_ppl.json"
    if already_has_bengali(seed0_out):
        print("[bengali-smoke] bengali_only/seed0 already done, skipping")
    else:
        run(["phase1_run_freq_and_scale.py", "--condition", "bengali_only", "--seed", "0"], args.gpus)
        run(["phase1_fisher.py", "--condition", "bengali_only", "--seed", "0",
             "--n-samples", "64", "--seqlen", "512"], args.gpus)
        run(["phase1_merge_eval.py", "--condition", "bengali_only", "--seed", "0"], args.gpus)

    print("\n[bengali-smoke] done -- inspect bengali_only/seed0/eval_ppl.json's ben_Beng "
          "bits_per_byte against baseline/eval_ppl.json's ben_Beng and its eng_Latn/"
          "kor_Hang/zho_Hans/swh_Latn entries (baseline bpb spread across languages is "
          "already informative on its own, independent of the calibration run), and "
          "cross-check against phase1_bengali_fragmentation_check.py's bytes/token ratio "
          "to judge swap-or-keep per 01_plans/claude_plan.md.")


if __name__ == "__main__":
    main()
