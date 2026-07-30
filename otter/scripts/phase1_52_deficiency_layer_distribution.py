"""§5.2 "불일치 expert의 정량화" (논문 초안, 2026-07-30 draft) TODO: "검출된
deficient 쌍의 레이어 분포". The aggregate deficiency counts already quoted in
the draft (English/Korean/Chinese=0, Bengali=5, Swahili=31, 36/810 total) come
from phase1_6_targeted_budget.py's allocate_budget() (01_plans/06_불일치표적배분_
실측결과.md, 2026-07-28) -- but that function only accumulates deficiency PER
LANGUAGE, summed across all 27 layers x top-30 disagreement-expert pairs. This
script reuses the exact same per-(layer, expert, language) deficiency
predicate (COPIED, not reimplemented, from allocate_budget() -- see its
docstring for the target_hit_count/reference_budget_tokens rule) but keeps the
per-layer breakdown allocate_budget() throws away, to answer where in the
network the 36 deficient pairs sit -- the paper draft's layer-locality claim
(§3 Figure 1, also §5.1) predicts they should concentrate in the back ~20% of
layers, mirroring the prior-work "U-shaped" language-specialization pattern
the draft cites for the mid-layer/peripheral-layer split.

Pure post-processing of scan_disagreement_experts.py's already-computed
scan_results.json (452KB) -- no GPU, no multi-GB file I/O, seconds to run.

Usage:
    conda run -n d2moe_env python phase1_52_deficiency_layer_distribution.py
        [--target-hit-count 20] [--n-samples 128] [--seqlen 512]
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fisher_pilot_a import LAYER_ROLE  # noqa: E402
from disagreement_common import LANG_CONDITIONS, conditions_for, load_scan_results  # noqa: E402
from phase1_6_targeted_budget import (  # noqa: E402 -- reused so this can't drift from the 36/810 headline number
    LANG_OF_COND, MOE_LAYERS, SCAN_RESULTS_DEFAULT, collect_disagreement_pairs,
)

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")


def per_layer_deficiency(scan, n_tokens_by_cond, pairs, target_hit_count, reference_budget_tokens):
    """Same predicate as phase1_6_targeted_budget.allocate_budget(): a
    (layer, expert, language) cell is deficient if that language's expected
    hit count, at an equal 1/5 split of reference_budget_tokens, falls short
    of target_hit_count. Returns {layer: {"role": str, "n_pairs": int,
    "deficient_by_lang": {lang: int}, "any_deficient": int}}."""
    langs = list(LANG_OF_COND.values())
    baseline_tokens = reference_budget_tokens / len(langs)

    by_layer = {}
    for layer, expert in pairs:
        entry = scan[layer]
        row = by_layer.setdefault(layer, {
            "role": entry.get("role", LAYER_ROLE.get(int(layer), "other")),
            "n_pairs": 0, "deficient_by_lang": {lang: 0 for lang in langs}, "any_deficient": 0,
        })
        row["n_pairs"] += 1
        any_def = False
        for cond, lang in LANG_OF_COND.items():
            hits = entry["hit_count"][cond][expert]
            n_tok = n_tokens_by_cond[cond]
            rate = hits / n_tok if n_tok > 0 else 0.0
            if rate * baseline_tokens < target_hit_count:
                row["deficient_by_lang"][lang] += 1
                any_def = True
        if any_def:
            row["any_deficient"] += 1
    return by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-results", type=Path, default=None)
    parser.add_argument("--balanced-condition", default="balanced", choices=["balanced", "mixed_5lang"])
    parser.add_argument("--target-hit-count", type=int, default=20)
    # NOT phase1_6_targeted_budget.py's own default (128) -- the paper's
    # headline 36/810 (Swahili=31, Bengali=5) was produced with
    # --n-samples 64, matching phase1_fisher.py's actual real-Fisher run
    # budget (01_plans/06_불일치표적배분_실측결과.md §1). Verified 2026-07-30
    # by re-running phase1_6_targeted_budget.py --n-samples 64 --seqlen 512
    # and reproducing swh=31/ben=5/weights={0.732,0.118} exactly; the script's
    # own CLI default of 128 instead gives swh=16/ben=2 -- a DIFFERENT,
    # smaller reference budget in tokens, so pass --n-samples explicitly if
    # you want to match a specific real run rather than this default.
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    conditions = conditions_for(args.balanced_condition)
    scan_path = args.scan_results or SCAN_RESULTS_DEFAULT
    if not scan_path.exists():
        raise FileNotFoundError(f"{scan_path} missing -- run scan_disagreement_experts.py first")
    print(f"[deficiency-layers] loading {scan_path}")
    scan, meta = load_scan_results(scan_path, MOE_LAYERS, conditions, layer_role=LAYER_ROLE, log_prefix="[deficiency-layers]")
    n_tokens_by_cond = meta.get("n_tokens")
    if not n_tokens_by_cond:
        raise KeyError(f"{scan_path} has no _meta.n_tokens")

    pairs = collect_disagreement_pairs(scan)
    reference_budget_tokens = args.n_samples * args.seqlen
    by_layer = per_layer_deficiency(scan, n_tokens_by_cond, pairs, args.target_hit_count, reference_budget_tokens)

    total_pairs = sum(r["n_pairs"] for r in by_layer.values())
    total_any_deficient = sum(r["any_deficient"] for r in by_layer.values())
    total_by_lang = {lang: sum(r["deficient_by_lang"][lang] for r in by_layer.values()) for lang in LANG_OF_COND.values()}

    layers_sorted = sorted(by_layer, key=lambda l: int(l))
    n = len(layers_sorted)
    front = layers_sorted[: n // 2]
    back = layers_sorted[n // 2:]

    def band_total(band, lang=None):
        if lang is None:
            return sum(by_layer[l]["any_deficient"] for l in band)
        return sum(by_layer[l]["deficient_by_lang"][lang] for l in band)

    result = {
        "target_hit_count": args.target_hit_count, "reference_budget_tokens": reference_budget_tokens,
        "total_pairs": total_pairs, "total_any_deficient": total_any_deficient,
        "total_by_lang": total_by_lang,
        "per_layer": {str(l): by_layer[l] for l in layers_sorted},
        "front_half_layers": front, "back_half_layers": back,
        "front_half_any_deficient": band_total(front), "back_half_any_deficient": band_total(back),
        "front_half_by_lang": {lang: band_total(front, lang) for lang in LANG_OF_COND.values()},
        "back_half_by_lang": {lang: band_total(back, lang) for lang in LANG_OF_COND.values()},
    }
    out_path = RESULTS_ROOT / "phase1_52_deficiency_layer_distribution_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[deficiency-layers] wrote {out_path}")

    print(f"\n[deficiency-layers] total: {total_any_deficient}/{total_pairs} pairs deficient for >=1 language "
          f"(by language, pair-cells: {total_by_lang})")
    print(f"[deficiency-layers] front half (layers {front[0]}-{front[-1]}): {result['front_half_any_deficient']} "
          f"deficient pairs, by lang: {result['front_half_by_lang']}")
    print(f"[deficiency-layers] back half (layers {back[0]}-{back[-1]}): {result['back_half_any_deficient']} "
          f"deficient pairs, by lang: {result['back_half_by_lang']}")
    print("\n[deficiency-layers] per-layer any_deficient (role):")
    for l in layers_sorted:
        r = by_layer[l]
        if r["any_deficient"] > 0:
            print(f"  layer {l} ({r['role']}): any_deficient={r['any_deficient']}/{r['n_pairs']} "
                  f"by_lang={ {k: v for k, v in r['deficient_by_lang'].items() if v > 0} }")


if __name__ == "__main__":
    main()
