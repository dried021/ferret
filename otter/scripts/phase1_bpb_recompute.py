"""Recompute Phase 1's per-language PPL as bits-per-byte (bpb), from the
already-saved eval_ppl.json aggregates -- no model rerun needed.

Motivation (2026-07-24): baseline Korean PPL (3.87) is far lower than English
(19.93) or Chinese (22.97), a gap too large to be explained by language
difficulty alone. n_tokens for the same 60 parallel FLORES sentences is
~7174 for Korean vs ~1834 (English) / ~1944 (Chinese) -- DeepSeek's tokenizer
is over-segmenting Hangul (likely byte-level fallback), so each Korean token
carries much less information and per-token PPL is mechanically deflated.
bits-per-byte normalizes by the language-agnostic byte count of the *same*
underlying text instead of the tokenizer's arbitrary segmentation, so it is
not distorted by this mismatch (see 00_docs/03_기술노트.md for the exact
argument this addresses).

total_nll (nats) is recovered exactly from ppl = exp(total_nll / n_tokens),
i.e. total_nll = ln(ppl) * n_tokens -- this is an exact inverse (corpus_ppl in
data_utils.py computes ppl this same way), not an approximation.
"""
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import data_utils  # noqa: E402

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
CONDITIONS = ["baseline", "english_only", "korean_only", "chinese_only", "balanced"]
EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX = "israel/flores-parallel", "test", "sentence_"
EVAL_LANGS = {"eng_Latn": "English", "kor_Hang": "Korean", "zho_Hans": "Chinese"}
EVAL_N_SENTENCES = 60


def total_bytes_for_lang(lang_code):
    sentences = data_utils.load_flores_sentences(EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX, lang_code, EVAL_N_SENTENCES)
    return sum(len(s.encode("utf-8")) for s in sentences)


def main():
    byte_counts = {lang: total_bytes_for_lang(lang) for lang in EVAL_LANGS}
    print("[bpb] byte counts (60 FLORES sentences, UTF-8):", byte_counts)

    table = {}
    for cond in CONDITIONS:
        path = RESULTS_ROOT / cond / "eval_ppl.json"
        data = json.loads(path.read_text())
        table[cond] = {}
        for lang in EVAL_LANGS:
            ppl = data[lang]["ppl"]
            n_tokens = data[lang]["n_tokens"]
            total_nll_nats = math.log(ppl) * n_tokens
            n_bytes = byte_counts[lang]
            bpb = (total_nll_nats / math.log(2)) / n_bytes
            table[cond][lang] = {
                "ppl_per_token": ppl,
                "n_tokens": n_tokens,
                "n_bytes": n_bytes,
                "tokens_per_byte": n_tokens / n_bytes,
                "bits_per_byte": bpb,
                "ppl_per_byte": 2 ** bpb,  # bpb re-expressed as an exp-space "PPL" for readability
            }

    out_path = RESULTS_ROOT / "bpb_recompute.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(f"[bpb] wrote {out_path}")

    baseline = table["baseline"]
    print("\n=== bits-per-byte table ===")
    header = f"{'condition':14s}" + "".join(f"{EVAL_LANGS[l]:>12s}" for l in EVAL_LANGS)
    print(header)
    for cond in CONDITIONS:
        row = f"{cond:14s}"
        for lang in EVAL_LANGS:
            row += f"{table[cond][lang]['bits_per_byte']:>12.4f}"
        print(row)

    print("\n=== relative increase vs baseline (bits-per-byte, %) ===")
    print(header)
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        row = f"{cond:14s}"
        for lang in EVAL_LANGS:
            incr = 100 * (table[cond][lang]["bits_per_byte"] / baseline[lang]["bits_per_byte"] - 1)
            row += f"{incr:>11.1f}%"
        print(row)

    print("\n=== tokens-per-byte (segmentation check) ===")
    print(header)
    row = f"{'baseline':14s}"
    for lang in EVAL_LANGS:
        row += f"{baseline[lang]['tokens_per_byte']:>12.4f}"
    print(row)


if __name__ == "__main__":
    main()
