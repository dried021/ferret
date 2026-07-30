"""Generalized Fisher x Scale 2x2 (seed 0-2 + placebo), parametrized by target
language -- lets §4.3's stage-separation experiment be repeated for any
calibration language pair, not just {english_only, korean_only}.

Why this exists (2026-07-27 review): run_phase1_2x2.py (seed=0 preliminary)
and run_phase1_2x2_verify.py (seed 1-2 + placebo, gated SUPPORTED for KO --
see 00_docs/03_기술노트.md §2.5) hardcode korean_only/kor_Hang throughout.
The paper's own pre-registered design (06_논문_구성.md §4.3) says the 2x2
should be "centered on the most sensitive language from §4.1" -- but §4.1's
actual most-sensitive language turned out to be Swahili (own-language gain
37.14% relative vs KO's 24.23%, see phase1_swahili_gate_result.json), which
was already known (2026-07-25) before the KO 2x2 started (2026-07-26). This
script lets the 2x2 be re-run centered on Swahili (or any future language)
without duplicating ~150 lines of merge/scale orchestration per language.

Unlike run_phase1_2x2_verify.py's two-stage history (seed0 prelim, then a
separate verify script for seed1-2+placebo), this script does the full
3-seed + placebo design in one pass -- there's no reason to re-live the
staged discovery process for a language we're deliberately choosing based on
already knowing it matters.

Tasks (resumable -- skips anything already on disk):
  A) expert_freq + real Fisher for {english_only, <lang>} x seed 0,1,2 --
     NOT english_only_b/<lang>_b: those are only ever used below as
     --scale-condition (which reads svd_scale_processed.pt only), never as
     the Fisher-merge --condition, so computing Fisher for them would waste
     GPU time on artifacts nothing reads (Fisher is the most expensive step
     in this pipeline, a full gradient backward per layer). english_only
     already exists from prior runs (3.7 budget-normalization grid); <lang>
     may or may not, depending which language this is invoked for (already
     present for swahili_only via run_phase1_swahili.py, so this step is a
     no-op skip-fest for Swahili specifically).
  B) svd_scale for the same 4 conditions x seed 0,1,2 (12 total, minus
     whatever english_only(_b) cells the KO 2x2 already computed).
  C) merge+eval, main grid: Fisher x Scale in {english_only, <lang>}^2,
     seed 0,1,2 (8 cells, minus (english_only,english_only) which is
     language-pair-independent and already exists from the KO run).
  D) merge+eval, placebo: Fisher=<lang> fixed, Scale in
     {english_only_b, <lang>_b}, seed 0,1,2 (6 cells) -- mirrors
     run_phase1_2x2_verify.py's placebo design, still Fisher-side fixed at
     the target language since that's the confirmed/robust Fisher condition
     and the claim under test is "does own_scale_gain(<lang> | Fisher=<lang>)
     survive a real placebo."

Usage:
    conda run -n d2moe_env python run_phase1_2x2_lang.py \\
        --lang-cond swahili_only --lang-cond-b swahili_only_b --lang-code swh_Latn
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
EN, EN_B = "english_only", "english_only_b"


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    # See run_phase1_belebele_grid.py's run() for why: serialize every merge_eval-family
    # caller (fisher/svd_scale CPU RAM load) across the whole host, not just
    # phase1_merge_eval.py itself, to avoid the OOM-kill seen 2026-07-29.
    cmd = ["flock", "/tmp/phase1_merge_eval.lock", "conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[2x2-lang] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def ensure_freq_and_fisher(cond, seed):
    """expert_freq + fisher_processed.pt are prerequisites for BOTH the
    merge (Fisher axis) and, indirectly, for this condition to be usable as
    a Fisher source in the grid. svd_scale (step B) is independent of these
    -- it only needs the model + calibration text -- so it's handled
    separately in main()."""
    out_dir = RESULTS_ROOT / cond / f"seed{seed}"
    freq_paths = list(out_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
    if not freq_paths:
        run(["phase1_run_freq_and_scale.py", "--condition", cond, "--seed", str(seed)])
    else:
        print(f"[2x2-lang] expert_freq {cond}/seed{seed} already done, skipping")
    fisher_path = out_dir / "fisher_processed.pt"
    if not fisher_path.exists():
        run(["phase1_fisher.py", "--condition", cond, "--seed", str(seed),
             "--n-samples", "64", "--seqlen", "512"])
    else:
        print(f"[2x2-lang] fisher {cond}/seed{seed} already done, skipping")


def ensure_scale(cond, seed):
    out_dir = RESULTS_ROOT / cond / f"seed{seed}"
    out_path = out_dir / "svd_scale_processed.pt"
    layer_dir = out_dir / "svd_scale_layers"
    if out_path.exists():
        print(f"[2x2-lang] scale {cond}/seed{seed} already done, skipping")
        if layer_dir.exists():
            print(f"[2x2-lang] cleaning up stale {layer_dir}")
            shutil.rmtree(layer_dir)
        return
    run(["phase1_svd_scale.py", "--condition", cond, "--seed", str(seed),
         "--n-samples", "64", "--seqlen", "512"])
    # svd_scale_layers/ is a disposable per-layer checkpoint dir (~34GB/condition)
    # -- the 2026-07-26 disk-full crash (00_docs/03_기술노트.md §2.5) was caused by
    # these accumulating across many conditions/seeds uncleaned. Remove immediately.
    if out_path.exists() and layer_dir.exists():
        print(f"[2x2-lang] scale {cond}/seed{seed} done, cleaning up {layer_dir}")
        shutil.rmtree(layer_dir)


def ensure_merge_eval(fisher_cond, scale_cond, seed):
    out_json = (RESULTS_ROOT / fisher_cond / f"seed{seed}" /
                f"scale_{scale_cond}_seed{seed}" / "eval_ppl.json")
    if out_json.exists():
        print(f"[2x2-lang] merge+eval Fisher={fisher_cond} Scale={scale_cond} seed={seed} "
              f"already done, skipping")
        return
    run(["phase1_merge_eval.py", "--condition", fisher_cond, "--seed", str(seed),
         "--scale-condition", scale_cond, "--scale-seed", str(seed)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-cond", required=True, help="e.g. swahili_only")
    parser.add_argument("--lang-cond-b", required=True, help="e.g. swahili_only_b")
    parser.add_argument("--lang-code", required=True, help="FLORES code, e.g. swh_Latn (for logging only)")
    args = parser.parse_args()
    lang, lang_b = args.lang_cond, args.lang_cond_b

    print(f"[2x2-lang] target language: {lang} / {lang_b} ({args.lang_code})")

    print("[2x2-lang] --- step A: expert_freq + Fisher prerequisites ---")
    # EN_B/lang_b are ONLY ever used as --scale-condition below (step B/D reads
    # svd_scale_processed.pt for them), never as the Fisher-merge --condition
    # (step C/D's fisher_cond only ranges over (EN, lang), matching
    # run_phase1_2x2_verify.py's own MERGE_EVAL_MAIN/PLACEBO). merge_condition()
    # only reads expert_freq/fisher_processed.pt for the Fisher axis, not the
    # scale axis (svd_scale is loaded straight from svd_scale_processed.pt,
    # see phase1_merge_eval.py's merge_condition()) -- so computing Fisher for
    # EN_B/lang_b here would be wasted GPU time (Fisher is the most expensive
    # step, a full gradient backward per layer) for artifacts nothing reads.
    for cond in (EN, lang):
        for seed in SEEDS:
            ensure_freq_and_fisher(cond, seed)

    print("[2x2-lang] --- step B: svd_scale ---")
    for cond in (EN, EN_B, lang, lang_b):
        for seed in SEEDS:
            ensure_scale(cond, seed)

    print("[2x2-lang] --- step C: merge+eval main grid ---")
    for fisher_cond in (EN, lang):
        for scale_cond in (EN, lang):
            for seed in SEEDS:
                ensure_merge_eval(fisher_cond, scale_cond, seed)

    print("[2x2-lang] --- step D: merge+eval placebo (Fisher=lang fixed) ---")
    for scale_cond in (EN_B, lang_b):
        for seed in SEEDS:
            ensure_merge_eval(lang, scale_cond, seed)

    print("[2x2-lang] all tasks done")


if __name__ == "__main__":
    main()
