"""§4.1 headline grid completion (2026-07-27): whitening-ON diagonal
(Fisher-language == Scale-language, same seed) for the conditions the
main grid still needs -- chinese_only, swahili_only, bengali_only,
mixed_5lang -- plus swahili_only_b/english_only_b/korean_only_b (needed so
each language's already-published placebo-verified own-gain claim, see
phase1_swahili_gate.py/phase1_placebo_gate.py, still has a same-pipeline
noise floor once EN/KO/ZH/Sw/Bn/Balanced all move to whitening-ON for
§4.1's headline table; see 06_논문_구성.md §4.1).

english_only/korean_only's whitening-ON diagonal already exists (run_phase1_2x2.py
+ run_phase1_2x2_verify.py, seeds 0-2) and is NOT touched here.

2026-07-27 code review fix: english_only_b/korean_only_b's own whitening-ON
diagonal (Fisher=Scale=that same B-arm condition) does NOT already exist --
the 2x2 verify work only ran Fisher=korean_only(fixed) x Scale in
{english_only_b, korean_only_b}, never Fisher=english_only_b x
Scale=english_only_b (or the korean_only_b equivalent). phase1_41_headline_gate.py's
noise_floor() needs exactly those diagonal cells for eng_Latn/kor_Hang.
Fisher and svd_scale for both conditions already exist on disk (from the
2x2 placebo work), so only their merge+eval diagonal step (6 cells, no fresh
Fisher/Scale compute) was missing -- now included below.

The existing whitening-OFF (svd_scale=None) 6-language grid is NOT deleted or
overwritten by this script -- those eval_ppl.json files stay right where they
are and are repurposed as §4.3's "Fisher-path-alone" condition
(01_plans/claude_plan.md v2: "버릴 데이터 없음 ... §4.3용 데이터로 재배치").

"Balanced" here means mixed_5lang (EN/KO/ZH/Swahili/Bengali), NOT the older
3-language `balanced` condition -- 01_plans/claude_plan.md's 2026-07-27
review note is explicit that the 3-language `balanced` result must not be
reused for any comparison that includes the Swahili/Bengali axis, since it
never saw those languages during calibration. mixed_5lang has no prior
Fisher/freq artifacts on disk (unlike chinese_only/swahili_only/swahili_only_b,
which reuse their already-computed ones), so it's computed from scratch here,
same as bengali_only.

Per-condition steps (each independently skip-if-exists, so this script is
safe to interrupt and rerun):
  1. [bengali_only, mixed_5lang only] expert_freq (phase1_run_freq_and_scale.py)
  2. [bengali_only, mixed_5lang only] real Fisher (phase1_fisher.py, 64 samples/seqlen 512 --
     same budget as every other condition, see run_phase1_seeds.sh/run_phase1_swahili.py)
  3. [all 5] svd_scale (phase1_svd_scale.py, 64 samples/seqlen 512, same budget)
  4. [all 5] merge+eval, diagonal (phase1_merge_eval.py --scale-condition==--condition,
     --scale-seed==--seed)
  0. [once] baseline re-eval, if it doesn't have ben_Beng yet (EVAL_LANGS already
     includes ben_Beng -- see phase1_merge_eval.py -- but baseline/eval_ppl.json
     predates that and needs one no-compression re-run to backfill the field,
     same gap swh_Latn had before run_phase1_swahili.py)

GPU policy: waits for GPU 0/1 to be genuinely idle (not just "not owned by
someone else" -- safe_gpus.sh alone would happily hand this script GPU 0/1
while run_phase1_2x2_verify.py, launched 2026-07-26 and still finishing its
last few cells as of this writing, is mid-merge on them, causing two ~31GB
models to fight over the same 2x22GiB budget) before touching any GPU.
MAX_GPUS stays at the project default of 2 (safe_gpus.sh; GPU 2 has another
user's job as of 2026-07-27, GPU 3 is idle alone but a single 24GB card can't
hold this model, so 2-GPU sequential is what's actually available).

Usage:
    conda run -n d2moe_env python run_phase1_41_diagonal.py
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]

CONDITIONS_NEEDING_FISHER = ["bengali_only", "mixed_5lang"]  # no Fisher/freq on disk yet
# english_only_b/korean_only_b: Fisher+svd_scale already exist (2x2 placebo work),
# only their diagonal merge+eval was missing -- see module docstring's 2026-07-27 fix.
CONDITIONS_WITH_FISHER = ["chinese_only", "swahili_only", "swahili_only_b",
                          "english_only_b", "korean_only_b"]  # already have it
ALL_DIAGONAL_CONDITIONS = CONDITIONS_WITH_FISHER + CONDITIONS_NEEDING_FISHER

IDLE_MEM_THRESHOLD_MIB = 2000  # a GPU below this used-memory is considered "actually free"


def gpu_mem_used(idx):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        gi, mem = [x.strip() for x in line.split(",")]
        if gi == str(idx):
            return int(mem)
    raise RuntimeError(f"GPU {idx} not found in nvidia-smi output")


def wait_for_idle(indices, poll_seconds=60):
    printed = False
    while True:
        used = {i: gpu_mem_used(i) for i in indices}
        if all(v < IDLE_MEM_THRESHOLD_MIB for v in used.values()):
            if printed:
                print(f"[41-diagonal] GPUs {indices} now idle ({used}), proceeding", flush=True)
            return
        if not printed:
            print(f"[41-diagonal] waiting for GPUs {indices} to free up (current usage {used} MiB, "
                  f"threshold {IDLE_MEM_THRESHOLD_MIB} MiB) -- polling every {poll_seconds}s", flush=True)
            printed = True
        time.sleep(poll_seconds)


def safe_gpus():
    out = subprocess.run(["bash", "-c", "source ./safe_gpus.sh && echo $SAFE_GPUS"],
                          cwd=SCRIPT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run(args):
    gpus = safe_gpus()
    cmd = ["conda", "run", "-n", "d2moe_env", "python"] + args
    print(f"[41-diagonal] + {' '.join(args)} (GPUS={gpus})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, HF_HOME="/mnt/HDD/minjeong/hf_cache",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)
    time.sleep(10)


def has_lang(path, lang_code):
    if not path.exists():
        return False
    try:
        return lang_code in json.loads(path.read_text())
    except Exception:
        return False


def ensure_fisher(condition, seed):
    freq_glob = list((RESULTS_ROOT / condition / f"seed{seed}").glob("deepseek_wikitext_*_expert_frequencies.json"))
    if not freq_glob:
        run(["phase1_run_freq_and_scale.py", "--condition", condition, "--seed", str(seed)])
    else:
        print(f"[41-diagonal] {condition}/seed{seed}: expert_freq already present, skipping")

    fisher_path = RESULTS_ROOT / condition / f"seed{seed}" / "fisher_processed.pt"
    if not fisher_path.exists():
        run(["phase1_fisher.py", "--condition", condition, "--seed", str(seed),
             "--n-samples", "64", "--seqlen", "512"])
    else:
        print(f"[41-diagonal] {condition}/seed{seed}: fisher_processed.pt already present, skipping")


def ensure_scale(condition, seed):
    out_dir = RESULTS_ROOT / condition / f"seed{seed}"
    scale_path = out_dir / "svd_scale_processed.pt"
    layer_dir = out_dir / "svd_scale_layers"
    if scale_path.exists():
        print(f"[41-diagonal] {condition}/seed{seed}: svd_scale_processed.pt already present, skipping")
        if layer_dir.exists():
            shutil.rmtree(layer_dir)
        return
    run(["phase1_svd_scale.py", "--condition", condition, "--seed", str(seed),
         "--n-samples", "64", "--seqlen", "512"])
    if scale_path.exists() and layer_dir.exists():
        # ~34GB/condition-seed of disposable per-layer checkpoints -- the 2026-07-26
        # disk-full incident was caused by these piling up (see run_phase1_2x2_verify.py).
        shutil.rmtree(layer_dir)


def ensure_diagonal_merge_eval(condition, seed):
    out_json = RESULTS_ROOT / condition / f"seed{seed}" / f"scale_{condition}_seed{seed}" / "eval_ppl.json"
    if out_json.exists():
        print(f"[41-diagonal] merge+eval {condition}/seed{seed} (diagonal) already done, skipping")
        return
    run(["phase1_merge_eval.py", "--condition", condition, "--seed", str(seed),
         "--scale-condition", condition, "--scale-seed", str(seed)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", action="store_true",
                         help="skip the GPU-idle wait (only for manual reruns once you've confirmed "
                              "0/1 are free yourself)")
    parser.add_argument("--conditions", nargs="+", default=ALL_DIAGONAL_CONDITIONS,
                         choices=ALL_DIAGONAL_CONDITIONS, help="subset of conditions to run")
    args = parser.parse_args()

    if not args.skip_wait:
        wait_for_idle([0, 1])

    baseline_path = RESULTS_ROOT / "baseline" / "eval_ppl.json"
    if "bengali_only" in args.conditions and not has_lang(baseline_path, "ben_Beng"):
        print("[41-diagonal] baseline missing ben_Beng, re-running baseline eval")
        run(["phase1_merge_eval.py", "--baseline"])
    else:
        print("[41-diagonal] baseline already has ben_Beng (or bengali_only not requested), skipping")

    for condition in args.conditions:
        if condition in CONDITIONS_NEEDING_FISHER:
            for seed in SEEDS:
                ensure_fisher(condition, seed)

    for condition in args.conditions:
        for seed in SEEDS:
            ensure_scale(condition, seed)

    for condition in args.conditions:
        for seed in SEEDS:
            ensure_diagonal_merge_eval(condition, seed)

    print("[41-diagonal] all requested conditions done")


if __name__ == "__main__":
    main()
