"""§5.2 집필용 companion to phase1_52_deficiency_layer_distribution.py.

That script answers "how many deficient (layer, expert, language) cells per
layer" but only keeps the aggregate counts -- it discards which specific
(layer, expert) triggered each flag. This script reuses the EXACT SAME
predicate (same COPIED-not-reimplemented source as
phase1_6_targeted_budget.allocate_budget(), see that function's docstring
for the target_hit_count/reference_budget_tokens rule) and instead emits the
individual list, so "where did the 36-pair list come from" has a
reproducible answer file rather than only a paper-draft table.

Pure post-processing of scan_disagreement_experts.py's already-computed
scan_results.json (452KB) -- no GPU, no multi-GB file I/O, seconds to run.

Usage:
    conda run -n d2moe_env python phase1_52b_deficiency_pairs_list.py
        [--target-hit-count 20] [--n-samples 64] [--seqlen 512]
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from disagreement_common import conditions_for, load_scan_results  # noqa: E402
from phase1_6_targeted_budget import (  # noqa: E402 -- reused so this can't drift from the 36/810 headline number
    LANG_OF_COND, MOE_LAYERS, SCAN_RESULTS_DEFAULT, collect_disagreement_pairs,
)

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")


def list_deficient_pairs(scan, n_tokens_by_cond, pairs, target_hit_count, reference_budget_tokens):
    """Same predicate as phase1_52_deficiency_layer_distribution.per_layer_deficiency():
    a (layer, expert, language) cell is deficient if that language's expected
    hit count, at an equal 1/5 split of reference_budget_tokens, falls short
    of target_hit_count. Returns a flat list of dicts, one per flagged cell."""
    langs = list(LANG_OF_COND.values())
    baseline_tokens = reference_budget_tokens / len(langs)

    flagged = []
    for layer, expert in pairs:
        entry = scan[layer]
        for cond, lang in LANG_OF_COND.items():
            hits = entry["hit_count"][cond][expert]
            n_tok = n_tokens_by_cond[cond]
            rate = hits / n_tok if n_tok > 0 else 0.0
            expected = rate * baseline_tokens
            if expected < target_hit_count:
                flagged.append({
                    "layer": int(layer), "expert": int(expert), "lang": lang,
                    "hits": int(hits), "n_tok": int(n_tok), "rate": float(rate),
                    "expected_hits_at_baseline": float(expected),
                })
    return flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-results", type=Path, default=None)
    parser.add_argument("--balanced-condition", default="balanced", choices=["balanced", "mixed_5lang"])
    parser.add_argument("--target-hit-count", type=int, default=20)
    # Matches phase1_52_deficiency_layer_distribution.py's default -- see
    # that script's comment for why 64 (not its own CLI default of 128)
    # reproduces the paper's headline 36/810 (Swahili=31, Bengali=5).
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    conditions = conditions_for(args.balanced_condition)
    scan_path = args.scan_results or SCAN_RESULTS_DEFAULT
    if not scan_path.exists():
        raise FileNotFoundError(f"{scan_path} missing -- run scan_disagreement_experts.py first")
    print(f"[deficiency-pairs] loading {scan_path}")
    scan, meta = load_scan_results(scan_path, MOE_LAYERS, conditions, log_prefix="[deficiency-pairs]")
    n_tokens_by_cond = meta.get("n_tokens")
    if not n_tokens_by_cond:
        raise KeyError(f"{scan_path} has no _meta.n_tokens")

    pairs = collect_disagreement_pairs(scan)
    reference_budget_tokens = args.n_samples * args.seqlen
    flagged = list_deficient_pairs(scan, n_tokens_by_cond, pairs, args.target_hit_count, reference_budget_tokens)

    by_lang = {}
    for f in flagged:
        by_lang[f["lang"]] = by_lang.get(f["lang"], 0) + 1

    out_path = RESULTS_ROOT / "phase1_52b_deficiency_pairs_list_result.json"
    out_path.write_text(json.dumps({
        "target_hit_count": args.target_hit_count,
        "reference_budget_tokens": reference_budget_tokens,
        "total_pairs_scanned": len(pairs),
        "total_flagged": len(flagged),
        "by_lang": by_lang,
        "flagged": sorted(flagged, key=lambda f: (f["layer"], f["lang"], f["expert"])),
    }, indent=2))
    print(f"[deficiency-pairs] wrote {out_path}")
    print(f"[deficiency-pairs] total flagged: {len(flagged)}/{len(pairs)}, by_lang={by_lang}")
    for f in sorted(flagged, key=lambda x: (x["layer"], x["lang"], x["expert"])):
        print(f"  layer={f['layer']:>2} expert={f['expert']:>3} lang={f['lang']:<10} "
              f"hits={f['hits']:>4} n_tok={f['n_tok']:>6} rate={f['rate']:.6f}")


if __name__ == "__main__":
    main()
