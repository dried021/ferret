"""Stage 2 of the disagreement-aware calibration method (06_논문_구성.md §6
"제안 방법: 불일치-표적 calibration budget 배분"): turns Stage 1's scan
(scan_disagreement_experts.py's disagreement_experts + hit_count) into a
per-language calibration budget allocation, then hands it to
phase1_calib_data.py's "disagreement_targeted" condition so phase1_fisher.py/
phase1_run_freq_and_scale.py/phase1_merge_eval.py can run on it completely
unmodified, exactly like any other condition.

Core logic (matches the doc's framing: "실제로 언어 간 이견이 있는 expert는
소수이므로 그 소수에 예산을 집중하면 같은 비용으로 최악 언어의 손실을 더
줄일 수 있다"):

1. Take the UNION of every layer's disagreement_experts (layer, expert)
   pairs from Stage 1's scan -- the small set of experts whose importance
   actually disagrees across languages, the only ones a smarter allocation
   can help.
2. For each such pair, estimate each language's per-token routing RATE onto
   that expert (hit_count / tokens-processed-in-the-scan, both from Stage
   1 -- see scan_disagreement_experts.py's `_meta.n_tokens`).
3. Ask: "if the real-Fisher run's total token budget were split EQUALLY
   across the 5 languages (what Balanced/mixed_5lang effectively does),
   would this pair get enough hits (>= --target-hit-count) from THIS
   language alone to have a numerically stable per-language Fisher
   contribution?" Sum the shortfall (deficiency) over every disagreement
   pair, per language.
4. Languages whose disagreement-relevant experts are most under-covered at
   an equal split get proportionally MORE of the (same total) budget;
   languages whose disagreement-relevant experts are already well covered
   at an equal split get less (floored, never zero -- see --floor-share).

This is deliberately a simple, auditable proportional-deficiency rule, not a
constrained optimizer -- the same "explicit formula, print every
intermediate, no black box" style as phase1_42_vulnerability_reanalysis.py.
Ablation (i) in 06_논문_구성.md §6 (targeting random experts instead) is
--random-experts; ablation (ii) (proxy replaced by real Fisher for Stage 1)
is out of scope for this script, since it would require a real-Fisher scan
result in the same schema, which run_phase1_2x2.py-style scripts already
produce per-layer for the language conditions.

Usage:
    conda run -n d2moe_env python phase1_6_targeted_budget.py [--smoke]
        [--scan-results PATH] [--balanced-condition balanced]
        [--target-hit-count 20] [--floor-share 0.05]
        [--n-samples 128] [--seqlen 512] [--seed 0]
        [--budget-scale 1.0] [--random-experts]

    (GPU not needed -- pure post-processing of scan_disagreement_experts.py's
    output; only imports torch/transformers indirectly via phase1_fisher for
    its build_samples()-compatible token-budget math, no forward pass here.)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402
from fisher_pilot_a import LAYER_ROLE  # noqa: E402 -- reused, not redefined
from disagreement_common import (  # noqa: E402
    LANG_CONDITIONS, conditions_for, make_synthetic_scan, load_scan_results,
)

RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/scan_disagreement_experts")
SCAN_RESULTS_DEFAULT = RESULTS_DIR / "scan_results.json"

NUM_HIDDEN_LAYERS = 28
MOE_LAYERS = list(range(1, NUM_HIDDEN_LAYERS))

# phase1_calib_data.py's LANG_OF_SIMPLE keys these conditions by lang code;
# reuse that mapping instead of redefining condition->lang here.
LANG_OF_COND = {c: phase1_calib_data.LANG_OF_SIMPLE[c] for c in LANG_CONDITIONS}


def collect_disagreement_pairs(scan, random_experts=False, n_random=30, seed=0):
    """Union of every layer's (layer, expert) disagreement pairs. Ablation
    (i) (06_논문_구성.md §6): --random-experts picks the SAME total COUNT of
    (layer, expert) pairs uniformly at random instead of using Stage 1's
    variance ranking, to isolate "does TARGETING matter" from "does more
    budget anywhere matter"."""
    pairs = []
    for layer, entry in scan.items():
        for e in (entry.get("disagreement_experts") or []):
            pairs.append((layer, e))
    if random_experts:
        rng = np.random.default_rng(seed)
        n_experts = len(next(iter(scan.values()))["proxy"][LANG_CONDITIONS[0]])
        n_total = len(pairs) if pairs else n_random
        chosen_layers = rng.choice(MOE_LAYERS, size=n_total, replace=True)
        chosen_experts = rng.integers(0, n_experts, size=n_total)
        pairs = list(zip((int(l) for l in chosen_layers), (int(e) for e in chosen_experts)))
    return pairs


def project_to_capped_simplex(raw_weights, floor_share, max_share):
    """Projects raw (non-negative, not-necessarily-normalized) weights onto
    the simplex {w >= 0, sum(w) = 1} subject to floor_share <= w[lang] <=
    max_share for every language. Without this, allocate_budget()'s pure
    proportional-to-deficiency rule is an unconstrained max-min: if one
    language is the sole bottleneck, the math happily assigns it the entire
    budget (2026-07-27 code review point 2 -- "이 objective 자체가 편향
    자체가 아니라, 편향을 만드는 최적화 구조"). max_share=None or >= 1.0
    disables the cap (floor-only).

    Water-filling: at each step, compute what every still-free language
    WOULD get if the remaining budget were split proportionally to raw
    weight among only the free languages; fix the single worst floor/cap
    violator (if any) at its bound and recompute from scratch. One language
    at a time, not a batch -- fixing multiple languages per pass off a
    single stale snapshot (an earlier version of this function did that)
    silently drops budget whenever 2+ languages hit a bound in the same
    pass (e.g. floor=0.05 x 3 + max_share=0.40 x 2 = 0.95, not 1.0 -- caught
    by the --max-share 0.40 sensitivity sweep, see 00_docs notes). Doing it
    one at a time and always re-normalizing the free set from the CURRENT
    remaining budget guarantees the result always sums to exactly 1."""
    langs = list(raw_weights)
    n = len(langs)
    total = sum(raw_weights.values())
    norm = {l: (raw_weights[l] / total if total > 0 else 1.0 / n) for l in langs}
    if max_share is None:
        max_share = 1.0
    if floor_share * n > 1.0 + 1e-9:
        raise ValueError(f"floor_share={floor_share} x {n} languages > 1.0 -- infeasible")
    if max_share < floor_share:
        raise ValueError(f"max_share={max_share} < floor_share={floor_share} -- infeasible")

    fixed = {}
    free = set(langs)
    while free:
        remaining = 1.0 - sum(fixed.values())
        free_total = sum(norm[l] for l in free)
        shares = ({l: remaining * (norm[l] / free_total) for l in free} if free_total > 0
                  else {l: remaining / len(free) for l in free})

        violator, bound, worst = None, None, 1e-9
        for l in free:
            if shares[l] - max_share > worst:
                violator, bound, worst = l, max_share, shares[l] - max_share
            elif floor_share - shares[l] > worst:
                violator, bound, worst = l, floor_share, floor_share - shares[l]
        if violator is None:
            fixed.update(shares)
            break
        fixed[violator] = bound
        free.discard(violator)
    return fixed


def allocate_budget(scan, n_tokens_by_cond, pairs, target_hit_count, floor_share, reference_budget_tokens,
                     max_share=None):
    """The proportional-deficiency rule described in the module docstring.

    deficiency(lang) = COUNT of disagreement pairs where lang's expected hit
    count, at an equal 1/5 split of `reference_budget_tokens`, falls short
    of target_hit_count. This is a bounded count (0..n_pairs), not a raw
    token-cost sum -- an earlier version summed target_hit_count/rate
    directly, which let one or two near-zero-rate pairs (a language that
    almost never routes to some expert) dominate the whole sum and starve
    languages that were actually fine everywhere else. Counting "how many
    pairs are under-covered" instead of "how many tokens the worst pair
    would cost" keeps one outlier pair from hijacking the whole allocation.

    reference_budget_tokens is fixed at a canonical scale (args.n_samples *
    args.seqlen, i.e. budget_scale=1.0) regardless of the ACTUAL
    --budget-scale being evaluated -- the allocation is a property of which
    language needs help, evaluated once at a reference point, then applied
    at whatever total budget the real run uses. This also avoids the same
    budget-scale-dependence bug (see git history): computing deficiency at
    the scaled budget would flatten every language toward equal deficiency
    as the budget shrinks, exactly backwards from 06_논문_구성.md §6 Figure
    5's prediction that targeting's advantage over Balanced should GROW as
    the budget tightens. Realized coverage AT the actual --budget-scale
    (whether targeted weights actually clear target_hit_count, vs Balanced,
    at that specific budget) is computed separately by realized_coverage().

    max_share (2026-07-27 code review): the raw deficiency-proportional rule
    below is an unconstrained max-min -- if one language is the sole
    bottleneck, it mathematically "deserves" the entire budget. None (or
    >= 1.0) leaves that behavior unchanged (floor-only, the original
    version); a finite cap (e.g. 0.40-0.50) restricts any single language to
    at most that share via project_to_capped_simplex(), trading allocation
    "purity" for a defensible bound on how extreme the redistribution can
    get -- compare realized_coverage() with and without a cap to see
    whether this actually costs coverage or not.

    Returns (weights: {lang: float, sums to 1}, diagnostics: dict)."""
    langs = list(LANG_OF_COND.values())
    baseline_share = 1.0 / len(langs)
    baseline_tokens = baseline_share * reference_budget_tokens

    deficiency = {lang: 0 for lang in langs}
    for layer, expert in pairs:
        entry = scan[layer]
        if entry["hit_count"] is None:
            raise ValueError(f"layer {layer} has no hit_count for at least one condition -- Stage 2 needs it "
                              f"for every disagreement pair (re-run scan_disagreement_experts.py)")
        for cond, lang in LANG_OF_COND.items():
            hits = entry["hit_count"][cond][expert]
            n_tok = n_tokens_by_cond[cond]
            rate = hits / n_tok if n_tok > 0 else 0.0
            if rate * baseline_tokens < target_hit_count:
                deficiency[lang] += 1

    total_deficiency = sum(deficiency.values())
    if total_deficiency <= 0:
        # Every disagreement pair already clears target_hit_count for every
        # language at an equal split -- nothing to target, fall back to
        # Balanced's equal weights rather than dividing by zero.
        weights = {lang: baseline_share for lang in langs}
    else:
        weights = project_to_capped_simplex(deficiency, floor_share, max_share)

    diagnostics = {
        "deficiency": deficiency, "total_deficiency": total_deficiency,
        "baseline_share": baseline_share, "reference_budget_tokens": reference_budget_tokens,
        "n_pairs": len(pairs), "target_hit_count": target_hit_count,
    }
    return weights, diagnostics


def realized_coverage(scan, n_tokens_by_cond, pairs, weights, total_budget_tokens, target_hit_count):
    """Diagnostic (not used to compute weights): given a specific total
    token budget split according to `weights`, how many disagreement pairs
    would each language actually clear target_hit_count on -- compared
    against what an equal Balanced-style split would achieve at the SAME
    total budget. This is the number that should show targeting's advantage
    widening as total_budget_tokens shrinks (06_논문_구성.md §6 Figure 5)."""
    langs = list(LANG_OF_COND.values())
    baseline_share = 1.0 / len(langs)
    targeted_cleared = {lang: 0 for lang in langs}
    balanced_cleared = {lang: 0 for lang in langs}
    for layer, expert in pairs:
        entry = scan[layer]
        for cond, lang in LANG_OF_COND.items():
            hits = entry["hit_count"][cond][expert]
            n_tok = n_tokens_by_cond[cond]
            rate = hits / n_tok if n_tok > 0 else 0.0
            if rate * weights[lang] * total_budget_tokens >= target_hit_count:
                targeted_cleared[lang] += 1
            if rate * baseline_share * total_budget_tokens >= target_hit_count:
                balanced_cleared[lang] += 1
    n_pairs = len(pairs)
    return {
        "n_pairs": n_pairs, "targeted_cleared": targeted_cleared, "balanced_cleared": balanced_cleared,
        "targeted_cleared_total": sum(targeted_cleared.values()),
        "balanced_cleared_total": sum(balanced_cleared.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scan-results", type=Path, default=None)
    parser.add_argument("--balanced-condition", default="balanced", choices=["balanced", "mixed_5lang"])
    parser.add_argument("--target-hit-count", type=int, default=20,
                         help="per-(layer,expert,language) hit count considered 'stably estimated'")
    parser.add_argument("--floor-share", type=float, default=0.05, help="minimum weight share per language")
    parser.add_argument("--max-share", type=float, default=None,
                         help="cap on any single language's weight share (e.g. 0.4-0.5); default None = uncapped "
                              "max-min (2026-07-27 code review point 2 -- see project_to_capped_simplex())")
    parser.add_argument("--n-samples", type=int, default=128, help="matches phase1_fisher.py's default real-Fisher budget")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-scale", type=float, default=1.0,
                         help="scales total_budget_tokens for the §6 budget-curve sweep (Figure 5): "
                              "0.25-2.0 x the n_samples*seqlen default")
    parser.add_argument("--random-experts", action="store_true",
                         help="ablation (i): target random (layer,expert) pairs instead of Stage 1's ranking")
    parser.add_argument("--out-tag", default=None,
                         help="write to budget_allocation_<tag>.json instead of the canonical budget_allocation.json "
                              "-- use this for throwaway sensitivity sweeps (2026-07-27 code review) so they don't "
                              "clobber the file a real condition actually reads")
    parser.add_argument("--out-condition", default=None, choices=phase1_calib_data.TARGETED_CONDITIONS,
                         help="which TARGETED_CONDITIONS entry to write the canonical (non-tagged) "
                              "budget_allocation.json for -- e.g. disagreement_targeted_cap50 when using "
                              "--max-share 0.5 as a real (not throwaway) variant carried through the full "
                              "pipeline. Defaults to phase1_calib_data.TARGETED_CONDITION (the uncapped one).")
    args = parser.parse_args()

    conditions = conditions_for(args.balanced_condition)
    total_budget_tokens = args.budget_scale * args.n_samples * args.seqlen

    if args.smoke:
        print("[budget] --smoke: using a fabricated synthetic scan, NOT real data")
        scan = make_synthetic_scan(MOE_LAYERS, conditions, layer_role=LAYER_ROLE, seed=args.seed)
        n_tokens_by_cond = {c: 20000 for c in conditions}
    else:
        scan_path = args.scan_results or SCAN_RESULTS_DEFAULT
        if not scan_path.exists():
            raise FileNotFoundError(f"{scan_path} missing -- run scan_disagreement_experts.py first (or pass --smoke)")
        print(f"[budget] loading {scan_path}")
        scan, meta = load_scan_results(scan_path, MOE_LAYERS, conditions, layer_role=LAYER_ROLE, log_prefix="[budget]")
        n_tokens_by_cond = meta.get("n_tokens")
        if not n_tokens_by_cond:
            raise KeyError(f"{scan_path} has no _meta.n_tokens -- re-run scan_disagreement_experts.py "
                            f"(this field was added 2026-07-27; an older scan file predates it).")

    pairs = collect_disagreement_pairs(scan, random_experts=args.random_experts, seed=args.seed)
    print(f"[budget] {len(pairs)} disagreement (layer, expert) pairs "
          f"{'(RANDOM ablation)' if args.random_experts else '(Stage 1 ranking)'}, "
          f"total_budget_tokens={total_budget_tokens:.0f} (n_samples={args.n_samples} x seqlen={args.seqlen} "
          f"x budget_scale={args.budget_scale})")

    reference_budget_tokens = args.n_samples * args.seqlen
    weights, diagnostics = allocate_budget(
        scan, n_tokens_by_cond, pairs, args.target_hit_count, args.floor_share, reference_budget_tokens,
        max_share=args.max_share,
    )

    print(f"\n[budget] per-language deficiency (of {diagnostics['n_pairs']} pairs, count under-covered at an "
          f"equal split of the {reference_budget_tokens}-token reference budget, target={args.target_hit_count} hits/pair):")
    for lang, d in sorted(diagnostics["deficiency"].items(), key=lambda kv: -kv[1]):
        print(f"  {lang}: deficiency={d} pairs under-covered -> allocated weight={weights[lang]:.3f} "
              f"({'above' if weights[lang] > diagnostics['baseline_share'] else 'below' if weights[lang] < diagnostics['baseline_share'] else '='} "
              f"equal-share {diagnostics['baseline_share']:.3f})")
    if diagnostics["total_deficiency"] <= 0:
        print("  (total deficiency was 0 -- no language showed any observed routing to a disagreement pair in "
              "this scan; fell back to Balanced's equal weights, not an optimized allocation)")

    coverage = realized_coverage(scan, n_tokens_by_cond, pairs, weights, total_budget_tokens, args.target_hit_count)
    print(f"\n[budget] realized coverage at total_budget_tokens={total_budget_tokens:.0f} "
          f"(of {coverage['n_pairs']} disagreement pairs x 5 languages = {coverage['n_pairs'] * 5} pair-language cells) "
          f"-- a rough SANITY CHECK, not 06_논문_구성.md §6's actual pre-registered metric (worst-language bpb "
          f"degradation, computed downstream by phase1_6_budget_gate.py from a real merge+eval run):")
    print(f"  targeted: {coverage['targeted_cleared_total']} cells clear >= {args.target_hit_count} hits "
          f"vs balanced: {coverage['balanced_cleared_total']} cells -- "
          f"{'targeting wins' if coverage['targeted_cleared_total'] > coverage['balanced_cleared_total'] else 'NO advantage by this rough proxy -- a total-cells-cleared count is not the same objective as worst-language degradation, and some disagreement pairs may be near-unreachable for a language regardless of budget (see docstring); this does not by itself invalidate the allocation.'}")

    out_condition = args.out_condition or phase1_calib_data.TARGETED_CONDITION
    out_dir = RESULTS_DIR.parent / "phase1" / out_condition / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Smoke output is suffixed (matches phase1_fisher.py's convention) so it
    # can NEVER be picked up at phase1_calib_data.budget_allocation_path()
    # (the canonical, non-smoke path the real disagreement_targeted*
    # conditions read) -- a smoke run must not silently supply fake weights
    # to a real Fisher/merge run.
    if args.out_tag:
        out_path = out_dir / f"budget_allocation_{args.out_tag}.json"
    else:
        out_path = out_dir / ("budget_allocation_smoke.json" if args.smoke else "budget_allocation.json")
    payload = {
        "weights": weights, "diagnostics": diagnostics, "realized_coverage": coverage,
        "balanced_condition": args.balanced_condition, "random_experts_ablation": args.random_experts,
        "smoke": args.smoke, "seed": args.seed, "target_hit_count": args.target_hit_count,
        "floor_share": args.floor_share, "max_share": args.max_share, "out_condition": out_condition,
        "total_budget_tokens": total_budget_tokens, "budget_scale": args.budget_scale,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[budget] wrote {out_path}")
    print(f"[budget] next: conda run -n d2moe_env python phase1_run_freq_and_scale.py "
          f"--condition {out_condition} --seed {args.seed}, then phase1_fisher.py, "
          f"then phase1_merge_eval.py -- same as any other condition.")


if __name__ == "__main__":
    main()
