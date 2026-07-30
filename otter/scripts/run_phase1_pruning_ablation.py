"""§4.3(iii) pruning on/off ablation (06_논문_구성.md: "pruning은 시간 제약상
2×2로 확장하지 않고, 같은 언어 조건에서 pruning 적용/미적용을 비교하는
on/off ablation으로 축소해 추가"). 2026-07-27, written alongside
phase1_structured_prune.py (see its docstring for why this isn't the
vendored flap_ratio() path) and phase1_merge_eval.py's new --pp-ratio flag.

Design: for ONE calibration condition -- default korean_only, since that's
this project's most thoroughly placebo-verified condition (§3.8/3.11/2.5 --
both the Fisher-merge and whitening stages are already confirmed SUPPORTED
for KO, making it the cleanest base to ask "does adding pruning change
anything further?" against) -- run merge+eval with --pp-ratio unset (off,
already-existing behavior/artifacts) vs --pp-ratio=PP_RATIO (on), at each of
3 seeds. This is deliberately NOT a language cross (that would be a second
2x2); it's a single on/off toggle, matching the paper's own scope cut for
this axis.

PP_RATIO default 0.2 matches D2-deepseek.py's own --pp_ratio default
(D2-deepseek.py argparse), so the ablation uses the same magnitude the
D²-MoE paper itself treats as a normal operating point, not an extreme case.

Usage:
    conda run -n d2moe_env python run_phase1_pruning_ablation.py
    conda run -n d2moe_env python run_phase1_pruning_ablation.py --condition swahili_only --pp-ratio 0.3
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
PP_RATIO_DEFAULT = 0.2


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[pruning-ablation] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="korean_only",
                         help="calibration condition, must already have expert_freq + fisher_processed.pt "
                              "(true for english_only/korean_only/chinese_only/balanced/swahili_only/... at "
                              "seed 0-2 already)")
    parser.add_argument("--pp-ratio", type=float, default=PP_RATIO_DEFAULT)
    args = parser.parse_args()
    cond, pp_ratio = args.condition, args.pp_ratio

    for seed in SEEDS:
        seed_dir = RESULTS_ROOT / cond / f"seed{seed}"
        fisher_prereq = seed_dir / "fisher_processed.pt"
        freq_prereq = list(seed_dir.glob("deepseek_wikitext_*_expert_frequencies.json")) if seed_dir.exists() else []
        if not fisher_prereq.exists() or not freq_prereq:
            raise SystemExit(f"{seed_dir} missing fisher_processed.pt and/or expert_frequencies.json -- run "
                              f"phase1_run_freq_and_scale.py + phase1_fisher.py for --condition {cond} "
                              f"--seed {seed} first (this ablation reuses those artifacts, same as every "
                              f"other merge+eval call).")

        off_json = RESULTS_ROOT / cond / f"seed{seed}" / "eval_ppl.json"
        if off_json.exists():
            print(f"[pruning-ablation] OFF {cond}/seed{seed} already done (pre-existing baseline result), skipping")
        else:
            print(f"[pruning-ablation] === OFF (pp_ratio unset) {cond}/seed{seed} ===")
            run(["phase1_merge_eval.py", "--condition", cond, "--seed", str(seed)])

        on_json = RESULTS_ROOT / cond / f"seed{seed}" / f"pp_ratio_{pp_ratio}" / "eval_ppl.json"
        if on_json.exists():
            print(f"[pruning-ablation] ON {cond}/seed{seed} pp_ratio={pp_ratio} already done, skipping")
        else:
            print(f"[pruning-ablation] === ON (pp_ratio={pp_ratio}) {cond}/seed{seed} ===")
            run(["phase1_merge_eval.py", "--condition", cond, "--seed", str(seed), "--pp-ratio", str(pp_ratio)])

    print(f"[pruning-ablation] all seeds done for condition={cond} pp_ratio={pp_ratio}")


if __name__ == "__main__":
    main()
