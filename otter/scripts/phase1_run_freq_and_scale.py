"""Runs D2MoE's own preprocess/get_expert_freq.py and preprocess/get_scale.py
UNMODIFIED against one of Phase 1's four calibration conditions, by
monkeypatching datasets.load_dataset (before those scripts' own `from
datasets import load_dataset` line executes) to return our multilingual
FLORES text instead of wikitext. Both scripts are forward-only and already
process the model layer-by-layer internally, so they don't need Fisher Pilot
A's freeze-and-accumulate-on-CPU treatment -- get_expert_freq.py just needed
--batch_size lowered (see 00_docs/03_기술노트.md Phase 0 step 1).

Neither script has an `if __name__ == "__main__":` guard -- both execute
their whole pipeline (including model load and file save) as a side effect
of being imported, driven entirely by sys.argv. This wrapper sets sys.argv,
imports each via importlib with a fresh module name, then explicitly frees
that script's model from GPU memory before moving to the next one (otherwise
two ~31GB models would be resident at once).

Usage:
    conda run -n d2moe_env python phase1_run_freq_and_scale.py --condition english_only [--smoke]
"""
import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path

import datasets
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = SCRIPT_DIR.parent / "D2MoE"
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")

_real_load_dataset = datasets.load_dataset


def install_dataset_patch(condition, seed):
    fake_ds = phase1_calib_data.make_fake_wikitext_dataset(condition, seed)

    def _patched(name, *args, **kwargs):
        if isinstance(name, str) and name == "wikitext":
            return fake_ds
        return _real_load_dataset(name, *args, **kwargs)

    datasets.load_dataset = _patched


def restore_dataset_patch():
    datasets.load_dataset = _real_load_dataset


def run_script_with_argv(script_path, argv, module_name):
    old_argv = sys.argv
    old_cwd = Path.cwd()
    sys.argv = [str(script_path)] + argv
    try:
        os.chdir(script_path.parent.parent)  # D2MoE/ root, in case relative paths matter
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def free_module_model(module):
    if hasattr(module, "model"):
        del module.model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples-freq", type=int, default=2000)
    parser.add_argument("--max-samples-scale", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-scale", action="store_true",
                         help="also run get_scale.py -- OFF by default since most Phase 1 conditions "
                              "use svd_scale=None (plain SVD) intentionally. get_scale.py's two GPU "
                              "memory leaks were found and fixed 2026-07-26 (see 00_docs/03_기술노트.md "
                              "'2) whitening 복원'); pass this flag when a condition needs real "
                              "activation-aware whitening (the Fisher x Scale 2x2, run_phase1_2x2.py).")
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples_freq = 50 if args.smoke else args.max_samples_freq
    max_samples_scale = 8 if args.smoke else args.max_samples_scale

    print(f"[phase1] condition={args.condition} smoke={args.smoke} out_dir={out_dir} include_scale={args.include_scale}")
    install_dataset_patch(args.condition, args.seed)
    try:
        print(f"[phase1] running get_expert_freq.py (max_samples={max_samples_freq}) ...")
        freq_argv = [
            f"--base_model_path={MODEL_PATH}",
            f"--save_path={out_dir}",
            "--model_type=deepseek",
            "--dataset_name=wikitext",
            "--split=train",
            f"--seed={args.seed}",
            f"--max_samples={max_samples_freq}",
            "--batch_size=4",
        ]
        freq_mod = run_script_with_argv(D2MOE_DIR / "preprocess" / "get_expert_freq.py", freq_argv, "phase1_get_expert_freq")
        free_module_model(freq_mod)
        print("[phase1] get_expert_freq.py done")

        if not args.include_scale:
            print("[phase1] skipping get_scale.py (--include-scale not set, see --help)")
        else:
            print(f"[phase1] running get_scale.py (max_samples={max_samples_scale}, seqlen={args.seqlen}) ...")
            scale_argv = [
                f"--base_model_path={MODEL_PATH}",
                f"--save_path={out_dir}",
                "--model_type=deepseek",
                "--dataset_name=wikitext",
                "--split=train",
                f"--seed={args.seed}",
                f"--max_samples={max_samples_scale}",
                f"--seqlen={args.seqlen}",
                "--batch_size=1",
            ]
            scale_mod = run_script_with_argv(D2MOE_DIR / "preprocess" / "get_scale.py", scale_argv, "phase1_get_scale")
            free_module_model(scale_mod)
            print("[phase1] get_scale.py done")
    finally:
        restore_dataset_patch()

    print(f"[phase1] condition={args.condition}: wrote outputs to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
