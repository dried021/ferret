"""Bengali tokenizer-fragmentation smoke check (D-8 pre-registered decision
point, 01_plans/claude_plan.md D-8: "Bengali가 tokenizer fragmentation으로
bpb가 비정상적으로 튀면 Vietnamese/Indonesian으로 스왑"). CPU-only, no model
weights needed -- only the tokenizer -- so it can run before/independent of
the GPU smoke run in run_phase1_bengali_smoke.py, and is much cheaper.

Reports bytes-per-token for ben_Beng's full FLORES-200 devtest pool against
the four already-validated languages (eng_Latn/kor_Hang/zho_Hans/swh_Latn),
using the same tokenizer (deepseek-ai/deepseek-moe-16b-base, use_fast=False)
phase1_fisher.py/phase1_merge_eval.py load. A bytes/token ratio for Bengali
far below the others is the fragmentation symptom the plan's bpb-spike
concern depends on.

This script only reports the numbers -- it does NOT decide swap-or-keep. No
numeric threshold was pre-registered for this check (unlike
phase1_swahili_gate.py's fixed 2x-noise-floor criterion), so the call is
still manual, per the plan's "즉시 판단".

Usage:
    conda run -n d2moe_env python phase1_bengali_fragmentation_check.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import data_utils  # noqa: E402

SOURCE, SPLIT, COL_PREFIX = "israel/flores-parallel", "test", "sentence_"
MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]


def bytes_per_token(tokenizer, sentences):
    total_bytes = sum(len(s.encode("utf-8")) for s in sentences)
    total_tokens = sum(len(tokenizer(s)["input_ids"]) for s in sentences)
    return total_bytes / total_tokens, total_bytes, total_tokens


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    print("=== Bengali tokenizer fragmentation smoke check ===\n")
    ratios = {}
    for lang in LANGS:
        pool = data_utils.load_full_column(SOURCE, SPLIT, COL_PREFIX, lang)
        ratio, n_bytes, n_tokens = bytes_per_token(tokenizer, pool)
        ratios[lang] = ratio
        print(f"{lang:>10}: {ratio:.3f} bytes/token  ({n_bytes} bytes / {n_tokens} tokens, {len(pool)} sentences)")

    other_mean = sum(v for k, v in ratios.items() if k != "ben_Beng") / (len(ratios) - 1)
    drop_pct = 100 * (1 - ratios["ben_Beng"] / other_mean)
    print(f"\nben_Beng vs mean(other 4): {drop_pct:+.1f}% bytes/token "
          f"({'LOWER -- more fragmented' if drop_pct > 0 else 'not lower'})")
    print("No pre-registered swap threshold -- read this ratio together with the "
          "bengali_only bpb increase from run_phase1_bengali_smoke.py against the "
          "other four languages' known-good behavior, per 01_plans/claude_plan.md.")


if __name__ == "__main__":
    main()
