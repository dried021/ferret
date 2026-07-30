"""Orchestrates the "full pipeline + downstream task" reconfirmation that
00_docs/04_전체요약.md flags as top priority and 01_plans/claude_plan.md gates
the paper's core claim on (see phase1_belebele_eval.py's module docstring).

For each --condition x --seed, runs phase1_belebele_eval.py twice:
  1. pp_ratio OFF -- Fisher-merge + (by default) own-language whitened/
     truncation-aware SVD, no pruning
  2. pp_ratio ON  -- same, plus structured pruning -- the full 3-stage
     pipeline (Fisher merge + truncation-aware SVD + semi-dynamical pruning)

Defaults to korean_only and swahili_only: the two conditions whose
own-language protective effect is already placebo-verified against a noise
floor on FLORES bpb (00_docs/04_전체요약.md 핵심 발견 6,7번) -- the cleanest
test of whether that effect survives (a) a real downstream task metric and
(b) the addition of pruning on top of Fisher+whitening.

Uses --scale-condition=<condition> (own-language whitened SVD) by default,
NOT phase1_merge_eval.py's plain-SVD default -- pass --plain-svd to match
that default instead. Requires fisher_processed.pt (+ svd_scale_processed.pt
if not --plain-svd) to already exist per condition/seed; this script only
runs merge+prune+eval, it does not compute Fisher or whitening statistics.

IMPORTANT (2026-07-29, user-flagged methodological risk): run
`phase1_belebele_eval.py --baseline` (no --limit, or a large one) and
`phase1_belebele_floor_check.py` BEFORE this grid. DeepSeek-MoE-16B-base is
not instruction-tuned; if a language's baseline Belebele accuracy is already
at chance, no compressed condition can show a retention effect for it and
running the grid for that language wastes compute -- phase1_belebele_gate.py
reads the floor-check result and excludes flagged languages from its
macro/worst-language summary, but running fewer languages here saves time.

Usage:
    conda run -n d2moe_env python run_phase1_belebele_grid.py --limit 200
    conda run -n d2moe_env python run_phase1_belebele_grid.py --conditions korean_only --seeds 0 --limit 100
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
DEFAULT_CONDITIONS = ["korean_only", "swahili_only"]
DEFAULT_SEEDS = [0, 1, 2]
PP_RATIO_DEFAULT = 0.2


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    # flock same lock phase1_merge_eval.py's callers use (see run_pending_merge_eval.sh /
    # phase1_6 track_ab notes): merge_condition() loads fisher_processed.pt (28GB) +
    # svd_scale_processed.pt (35GB) onto CPU RAM, and a concurrent second such load (from
    # another merge_eval-family job on a different GPU pair) has actually OOM-killed this
    # host before (125GB total RAM, 2026-07-29). Serializing across ALL merge_eval-family
    # callers, not just phase1_merge_eval.py itself, is required to run this grid safely
    # alongside other GPU-pair queues.
    cmd = ["flock", "/tmp/phase1_merge_eval.lock", "conda", "run", "-n", "d2moe_env", "python", "-u"] + args
    print(f"[belebele-grid] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def out_dir_for(cond, seed, own_scale, pp_ratio=None, num_fewshot=0):
    d = RESULTS_ROOT / cond / f"seed{seed}"
    if own_scale:
        d = d / f"scale_{cond}_seed{seed}"
    if pp_ratio is not None:
        d = d / f"pp_ratio_{pp_ratio}"
    if num_fewshot:
        d = d / f"fewshot_{num_fewshot}"
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--pp-ratio", type=float, default=PP_RATIO_DEFAULT)
    parser.add_argument("--plain-svd", action="store_true",
                         help="skip whitened SVD (plain SVD only, matches phase1_merge_eval.py's own default "
                              "scope cut); default is own-language whitened SVD")
    parser.add_argument("--limit", type=int, default=None, help="lm-eval --limit, passed through to every run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-offload-gib", type=int, default=0)
    parser.add_argument("--num-fewshot", type=int, default=0,
                         help="2026-07-29: 0-shot Belebele accuracy sat too close to chance across the board "
                              "(even English only +10pp) to reliably detect a compression effect -- use a "
                              "few-shot value (e.g. 5) matching whatever gave the baseline enough headroom in "
                              "phase1_belebele_floor_check.py. Must match the --num-fewshot used for the "
                              "baseline eval_belebele.json this grid's retention claim is relative to.")
    parser.add_argument("--langs", nargs="+", default=None,
                         help="subset of languages to evaluate per condition/seed (passed through to "
                              "phase1_belebele_eval.py --langs); default None = all 5. Use to skip languages "
                              "already flagged at-chance by phase1_belebele_floor_check.py and save time.")
    parser.add_argument("--off-only", action="store_true",
                         help="skip the pp_ratio ON (+pruning) pass entirely -- only run Fisher+SVD-merge-only "
                              "eval. Use when the pruning ablation is out of scope for this grid (e.g. it's "
                              "covered separately, see §4.4).")
    args = parser.parse_args()
    own_scale = not args.plain_svd

    baseline_dir = RESULTS_ROOT / "baseline"
    if args.num_fewshot:
        baseline_dir = baseline_dir / f"fewshot_{args.num_fewshot}"
    baseline_json = baseline_dir / "eval_belebele.json"
    if not baseline_json.exists():
        raise SystemExit(f"{baseline_json} missing -- run `phase1_belebele_eval.py --baseline "
                          f"--num-fewshot {args.num_fewshot}` (and phase1_belebele_floor_check.py) BEFORE this "
                          "grid, see this script's module docstring.")

    for cond in args.conditions:
        for seed in args.seeds:
            seed_dir = RESULTS_ROOT / cond / f"seed{seed}"
            fisher_prereq = seed_dir / "fisher_processed.pt"
            scale_prereq = seed_dir / "svd_scale_processed.pt"
            if not fisher_prereq.exists():
                raise SystemExit(f"{fisher_prereq} missing -- run phase1_fisher.py --condition {cond} --seed {seed} first")
            if own_scale and not scale_prereq.exists():
                raise SystemExit(f"{scale_prereq} missing -- run phase1_svd_scale.py --condition {cond} --seed {seed} "
                                  f"first, or pass --plain-svd to skip whitening")

            base_args = ["phase1_belebele_eval.py", "--condition", cond, "--seed", str(seed),
                         "--batch-size", str(args.batch_size)]
            if args.limit is not None:
                base_args += ["--limit", str(args.limit)]
            if args.cpu_offload_gib:
                base_args += ["--cpu-offload-gib", str(args.cpu_offload_gib)]
            if args.num_fewshot:
                base_args += ["--num-fewshot", str(args.num_fewshot)]
            if args.langs:
                base_args += ["--langs"] + args.langs
            if own_scale:
                base_args += ["--scale-condition", cond, "--scale-seed", str(seed)]

            off_json = out_dir_for(cond, seed, own_scale, num_fewshot=args.num_fewshot) / "eval_belebele.json"
            if off_json.exists():
                print(f"[belebele-grid] OFF {cond}/seed{seed} already done, skipping")
            else:
                print(f"[belebele-grid] === OFF (Fisher+SVD, no pruning) {cond}/seed{seed} ===")
                try:
                    run(base_args)
                except subprocess.CalledProcessError as e:
                    # 2026-07-29 (overnight-unattended hardening): this loop used to abort
                    # entirely on the first failed seed (see otter/logs/
                    # belebele_grid_korean_only.log, an OOM mid-merge killed the whole
                    # script before seed1/seed2 ever ran). One seed's transient failure
                    # (OOM, disk contention) shouldn't cost the other seeds a shot -- log
                    # and move on, same as run_pending_merge_eval.sh's per-job pattern.
                    print(f"[belebele-grid] OFF {cond}/seed{seed} FAILED (rc={e.returncode}) -- continuing")

            if args.off_only:
                print(f"[belebele-grid] --off-only set, skipping ON (+pruning) pass for {cond}/seed{seed}")
                continue

            on_json = out_dir_for(cond, seed, own_scale, args.pp_ratio, args.num_fewshot) / "eval_belebele.json"
            if on_json.exists():
                print(f"[belebele-grid] ON {cond}/seed{seed} pp_ratio={args.pp_ratio} already done, skipping")
            else:
                print(f"[belebele-grid] === ON (+pruning pp_ratio={args.pp_ratio}) {cond}/seed{seed} ===")
                try:
                    run(base_args + ["--pp-ratio", str(args.pp_ratio)])
                except subprocess.CalledProcessError as e:
                    print(f"[belebele-grid] ON {cond}/seed{seed} FAILED (rc={e.returncode}) -- continuing")

    print("[belebele-grid] all conditions/seeds done -- run phase1_belebele_gate.py per condition to summarize")


if __name__ == "__main__":
    main()
