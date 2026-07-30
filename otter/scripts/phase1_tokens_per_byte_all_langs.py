"""Tokens-per-byte for all 5 Phase 1 eval languages (EN/KO/ZH/SW/BN), using
the exact same FLORES sentences (first 60, offset=0) and tokenizer as
phase1_merge_eval.py / phase1_bpb_recompute.py.

This is a pure tokenization statistic -- it depends only on the eval
sentences and the tokenizer, not on which merge condition (baseline,
english_only, ..., balanced) produced a given eval_ppl.json. So a single
measurement per language is definitive for every condition; there is no
separate "balanced" value to compute.

No model forward pass / GPU needed -- tokenizer only.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import data_utils  # noqa: E402

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX = "israel/flores-parallel", "test", "sentence_"
EVAL_LANGS = {"eng_Latn": "English", "kor_Hang": "Korean", "zho_Hans": "Chinese",
              "swh_Latn": "Swahili", "ben_Beng": "Bengali"}
EVAL_N_SENTENCES = 60


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    results = {}
    for lang_code, lang_name in EVAL_LANGS.items():
        sentences = data_utils.load_flores_sentences(EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX, lang_code, EVAL_N_SENTENCES)
        n_bytes = sum(len(s.encode("utf-8")) for s in sentences)
        n_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences)
        results[lang_code] = {
            "lang_name": lang_name,
            "n_sentences": len(sentences),
            "n_tokens": n_tokens,
            "n_bytes": n_bytes,
            "tokens_per_byte": n_tokens / n_bytes,
        }

    out_path = SCRIPT_DIR.parent / "results" / "phase1_tokens_per_byte_all_langs.json" \
        if (SCRIPT_DIR.parent / "results").is_dir() else SCRIPT_DIR.parent / "phase1_tokens_per_byte_all_langs.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"{'lang':10s}{'n_sentences':>13s}{'n_tokens':>10s}{'n_bytes':>10s}{'tokens/byte':>13s}")
    for lang_code, r in results.items():
        print(f"{lang_code:10s}{r['n_sentences']:>13d}{r['n_tokens']:>10d}{r['n_bytes']:>10d}{r['tokens_per_byte']:>13.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
