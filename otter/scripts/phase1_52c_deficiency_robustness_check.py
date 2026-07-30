"""§5.2 lock-before-write robustness check (2026-07-30 evening pass) for the
36-pair deficiency flag that phase1_52_deficiency_layer_distribution.py /
phase1_52b_deficiency_pairs_list.py compute. Reuses the exact same predicate
(same COPIED-not-reimplemented source as phase1_6_targeted_budget.
allocate_budget()) but asks how FRAGILE the "36/810, Swahili=31, Bengali=5"
headline is, rather than just restating it:

1. Threshold sweep (15/20/25/30) per language -- if the count near 20 moves
   a lot, "36" is an artifact of the threshold choice, not a stable finding.
2. Borderline non-flagged cells (expected_hits_at_baseline in [20,25)) --
   how many near-misses exist and whether they'd change the layer-locality
   pattern (mid-network layers empty) if they flipped.
3. Denominator sanity: confirms 810 = 27 layers x top-k 30 disagreement
   candidates (by cross-language routing-proxy variance, restricted to
   experts with >=min_hit_count=5 routed tokens in every single-language
   condition), and separately confirms the OTHER 918 (=27*64-810) experts
   per layer DO have hit_count recorded in scan_results.json but were never
   run through the deficiency predicate at all (excluded from the
   disagreement-candidate pool for low cross-language variance and/or
   ineligibility, not because they weren't scanned) -- this changes how
   "36/810" should be read: it is 36 flagged among the pre-filtered
   disagreement candidates, not among all (layer, expert) pairs.

Does NOT attempt a window-level bootstrap (item 5 of the 2026-07-30 request)
-- scan_disagreement_experts.py's output schema only stores the final
per-condition, per-expert hit_count TOTAL (see its scan_condition()
docstring), accumulated over the whole forward pass across every
calibration text/window for that condition. No per-window or per-text
breakdown is persisted anywhere (checked otter/logs/ and the results dir),
so a genuine window-resampling bootstrap is not computable from existing
artifacts -- it would require re-instrumenting and re-running the GPU scan,
which is out of scope for a CPU-only lock-before-write pass.

Pure post-processing of scan_disagreement_experts.py's already-computed
scan_results.json (452KB) -- no GPU, seconds to run.

Usage:
    conda run -n d2moe_env python phase1_52c_deficiency_robustness_check.py
        [--n-samples 64] [--seqlen 512]
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from disagreement_common import conditions_for, load_scan_results  # noqa: E402
from phase1_6_targeted_budget import (  # noqa: E402
    LANG_OF_COND, MOE_LAYERS, SCAN_RESULTS_DEFAULT, collect_disagreement_pairs,
)

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
THRESHOLDS = [15, 20, 25, 30]
BORDERLINE_LO, BORDERLINE_HI = 20, 25


def build_cells(scan, n_tokens_by_cond, pairs, reference_budget_tokens):
    """One row per (layer, expert, language) candidate cell -- the same
    predicate inputs phase1_52b's list_deficient_pairs() computes, but kept
    for EVERY cell (not just flagged ones) so threshold sweeps/borderline
    checks don't need to re-scan."""
    langs = list(LANG_OF_COND.values())
    baseline_tokens = reference_budget_tokens / len(langs)
    cells = []
    for layer, expert in pairs:
        entry = scan[layer]
        for cond, lang in LANG_OF_COND.items():
            hits = entry["hit_count"][cond][expert]
            n_tok = n_tokens_by_cond[cond]
            rate = hits / n_tok if n_tok > 0 else 0.0
            expected = rate * baseline_tokens
            cells.append({"layer": int(layer), "expert": int(expert), "lang": lang,
                          "hits": int(hits), "expected_hits_at_baseline": float(expected)})
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-results", type=Path, default=None)
    parser.add_argument("--balanced-condition", default="balanced", choices=["balanced", "mixed_5lang"])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    conditions = conditions_for(args.balanced_condition)
    scan_path = args.scan_results or SCAN_RESULTS_DEFAULT
    if not scan_path.exists():
        raise FileNotFoundError(f"{scan_path} missing -- run scan_disagreement_experts.py first")
    print(f"[robustness] loading {scan_path}")
    scan, meta = load_scan_results(scan_path, MOE_LAYERS, conditions, log_prefix="[robustness]")
    n_tokens_by_cond = meta["n_tokens"]

    pairs = collect_disagreement_pairs(scan)
    reference_budget_tokens = args.n_samples * args.seqlen
    cells = build_cells(scan, n_tokens_by_cond, pairs, reference_budget_tokens)

    lang_order = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]

    # item 1: threshold sweep
    sweep = {t: {l: 0 for l in lang_order} for t in THRESHOLDS}
    for c in cells:
        for t in THRESHOLDS:
            if c["expected_hits_at_baseline"] < t:
                sweep[t][c["lang"]] += 1

    # item 2: borderline non-flagged cells
    borderline = [c for c in cells
                  if BORDERLINE_LO <= c["expected_hits_at_baseline"] < BORDERLINE_HI]
    front, mid, back = range(1, 10), range(10, 19), range(19, 28)

    def band_of(layer):
        if layer in front:
            return "front(1-9)"
        if layer in mid:
            return "mid(10-18)"
        return "back(19-27)"

    borderline_by_band = {"front(1-9)": 0, "mid(10-18)": 0, "back(19-27)": 0}
    for c in borderline:
        borderline_by_band[band_of(c["layer"])] += 1

    # item 3: denominator sanity
    per_layer_n_candidates = {l: len(scan[l]["disagreement_experts"]) for l in scan}
    n_experts_total = len(scan[MOE_LAYERS[0]]["hit_count"]["swahili_only"])
    n_layers = len(MOE_LAYERS)

    result = {
        "reference_budget_tokens": reference_budget_tokens,
        "n_candidate_pairs": len(pairs), "n_candidate_cells": len(cells),
        "threshold_sweep": {str(t): sweep[t] for t in THRESHOLDS},
        "borderline_20_to_25": sorted(borderline, key=lambda c: (c["layer"], c["lang"], c["expert"])),
        "borderline_count": len(borderline),
        "borderline_by_band": borderline_by_band,
        "denominator_check": {
            "n_layers": n_layers, "top_k_per_layer": 30,
            "expected_total": n_layers * 30,
            "actual_total_candidates": sum(per_layer_n_candidates.values()),
            "layers_with_nonstandard_count": {l: n for l, n in per_layer_n_candidates.items() if n != 30},
            "n_experts_per_layer_total": n_experts_total,
            "n_experts_scanned_total": n_layers * n_experts_total,
            "n_experts_excluded_from_candidacy_but_hit_count_recorded":
                n_layers * n_experts_total - n_layers * 30,
        },
        "hit_count_semantics": (
            "hit_count[cond][expert] is the raw count of ROUTED TOKENS "
            "(top-k gate assignments) from that condition's calibration "
            "corpus that were dispatched to that expert, accumulated over "
            "the whole single-seed forward pass over all calibration texts "
            "for that condition -- not a count of calibration windows/texts, "
            "and not normalized by corpus size. See scan_disagreement_experts.py "
            "make_capturing_forward()/scan_condition()."
        ),
        "window_level_bootstrap": (
            "NOT COMPUTABLE from existing artifacts -- scan_results.json only "
            "stores the final per-condition/per-expert hit_count TOTAL, no "
            "per-window or per-text breakdown is persisted anywhere in the "
            "repo. A genuine window-resampling bootstrap would require "
            "re-instrumenting and re-running scan_disagreement_experts.py on "
            "GPU, which this CPU-only lock-before-write pass does not do."
        ),
    }
    out_path = RESULTS_ROOT / "phase1_52c_deficiency_robustness_check_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[robustness] wrote {out_path}")

    print("\n[robustness] item 1: threshold sweep")
    print("threshold | " + " | ".join(lang_order) + " | total")
    for t in THRESHOLDS:
        row = sweep[t]
        print(f"{t:>9} | " + " | ".join(f"{row[l]:>3}" for l in lang_order) + f" | {sum(row.values()):>5}")

    print(f"\n[robustness] item 2: borderline non-flagged (expected in [{BORDERLINE_LO},{BORDERLINE_HI})): "
          f"{len(borderline)} cells, by band {borderline_by_band}")
    for c in sorted(borderline, key=lambda c: (c["layer"], c["lang"], c["expert"])):
        print(f"  layer={c['layer']:>2} expert={c['expert']:>3} lang={c['lang']:<10} "
              f"expected={c['expected_hits_at_baseline']:.2f}")

    dc = result["denominator_check"]
    print(f"\n[robustness] item 3: {dc['n_layers']} layers x top_k {dc['top_k_per_layer']} "
          f"= {dc['expected_total']} (actual {dc['actual_total_candidates']}); "
          f"nonstandard layers: {dc['layers_with_nonstandard_count'] or 'none'}")
    print(f"  {dc['n_experts_scanned_total']} experts scanned total, {dc['actual_total_candidates']} "
          f"selected as disagreement candidates, "
          f"{dc['n_experts_excluded_from_candidacy_but_hit_count_recorded']} scanned-but-excluded")


if __name__ == "__main__":
    main()
