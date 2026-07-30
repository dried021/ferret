"""§5.1 variant: does the headline "F_e = sum of squared-gradient elements,
combined across gate/up/down" ranking used by phase1_51_fisher_rank_correlation.py
survive if F_e is redefined to match the ACTUAL merge weight
Merge_deepseekMoE.merge_experts() computes for merge_method="fisher"
(merge_deepseek.py:265-281), instead of the plain unweighted Fisher sum?

merge_experts()'s real quantity, per expert j and projection proj in
{gate_proj, up_proj, down_proj} (kept separate -- see below), before the
per-weight-position normalization at merge_deepseek.py:275-277:

    fisher_scale_w_proj[j] = hessian[proj_j] (elementwise) * expert_freq[j]

expert_freq[j] is a single scalar per (condition, seed, layer, expert) --
loaded from deepseek_wikitext_2000_expert_frequencies.json (2000 = the real
pipeline's max_samples_freq default; phase1_run_freq_and_scale.py only writes
a "_50" file for --smoke runs, which some conditions still have on disk
alongside the real one -- picking the wrong one silently would understate
routing-frequency spread, so this script requires the literal "_2000" filename
per condition instead of glob()[0]'s first match). Since expert_freq[j] is a
per-expert CONSTANT (not a per-weight-position tensor), summing
fisher_scale_w_proj[j] over every element of that projection's weight matrix
factors the constant out:

    F_e_merge[proj][j] = sum_elements(hessian[proj_j]) * expert_freq[j]
                        = F_e_raw[proj][j] * expert_freq[j]

i.e. the merge-relevant scalar is just the existing raw per-projection Fisher
sum, rescaled per expert by how often the router actually picked it. The
post-normalization divide-by-sum-across-experts step (line 275-277) is NOT
applied here: that division is by a per-WEIGHT-POSITION tensor (not a
per-expert scalar), so after already collapsing to one scalar per expert it
would just rescale every expert by an unrelated constant baked from summing
across positions -- it cannot change the CROSS-EXPERT ranking this script
compares, and the task this script answers only asked for
"element x expert_freq[j] -> summed to a per-expert scalar", not the full
normalized merge coefficient.

Per the same task, gate/up/down are kept SEPARATE (not combined into one
scalar the way phase1_51_fisher_rank_correlation.py's F_e does) for the
per-definition rank-agreement check (part a below); the headline reproduction
(part b) sums all three back together, matching how phase1_51's F_e was
defined, so its overall mean_lang_rho/mean_placebo_rho is directly comparable
to that script's seed0 result (0.3139 / 0.9519).

Reuses phase1_51_fisher_rank_correlation.py's fisher_processed.pt reduction
approach (mmap=True load, CPU + disk only, no GPU, no forward/backward pass)
but caches per-projection arrays separately (fisher_expert_scalar_cache/ only
has the three projections pre-combined, from that script's reduce_condition).

Usage:
    conda run -n d2moe_env python phase1_51b_merge_weight_rank_correlation.py [--seed 0] [--smoke]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
CACHE_DIR = RESULTS_ROOT / "fisher_expert_scalar_by_proj_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_MOE_LAYERS = list(range(1, 28))
LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERT_FREQ_FILENAME = "deepseek_wikitext_2000_expert_frequencies.json"  # real pipeline default; "_50" is --smoke only


def _placebo_of():
    """Same exclusion logic as phase1_51_fisher_rank_correlation.py's
    _placebo_of(): chinese_only has no _b arm defined at all, and
    bengali_only_b is defined but was never actually run (empty dir, no
    fisher_processed.pt) -- confirmed still true 2026-07-30."""
    out = {}
    for c in LANG_CONDITIONS:
        b = f"{c}_b"
        if b not in phase1_calib_data.PLACEBO_CONDITIONS:
            continue
        if not (RESULTS_ROOT / b / "seed0" / "fisher_processed.pt").exists():
            print(f"[rank-corr-b] WARNING: {b} has no real Fisher on disk -- excluding {c} from placebo comparison")
            continue
        out[c] = b
    return out


PLACEBO_OF = _placebo_of()


def reduce_condition_by_proj(condition, seed, layers):
    """Returns {layer_idx: {proj: np.ndarray[n_experts]}} of raw per-projection
    Fisher scalars (sum over elements of that expert's gate/up/down tensor
    SEPARATELY, not combined) -- from cache if available, else by mmap-loading
    fisher_processed.pt (~30GB/condition, CPU + disk only)."""
    cache_path = CACHE_DIR / f"{condition}_seed{seed}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if all(str(l) in cached for l in layers):
            print(f"[rank-corr-b] {condition} seed{seed}: cache hit ({cache_path.name})")
            return {
                l: {proj: np.asarray(cached[str(l)][proj], dtype=np.float64) for proj in PROJECTIONS}
                for l in layers
            }

    pt_path = RESULTS_ROOT / condition / f"seed{seed}" / "fisher_processed.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} missing -- run phase1_fisher.py --condition {condition} --seed {seed} first")
    import torch  # deferred: only needed on an actual cache miss

    print(f"[rank-corr-b] {condition} seed{seed}: cache miss, loading {pt_path} (mmap=True) ...")
    assembled = torch.load(pt_path, map_location="cpu", mmap=True)
    out = {}
    for layer_idx in layers:
        layer_dict = assembled[layer_idx]
        n_experts = max(int(k.split(".")[2]) for k in layer_dict) + 1
        scalars = {proj: np.zeros(n_experts, dtype=np.float64) for proj in PROJECTIONS}
        for key, tensor in layer_dict.items():
            parts = key.split(".")
            e_idx, proj = int(parts[2]), parts[3]
            scalars[proj][e_idx] += tensor.float().sum().item()
        out[layer_idx] = scalars
        print(f"[rank-corr-b] {condition} seed{seed} layer {layer_idx}: reduced ({n_experts} experts)")
    del assembled

    existing = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    existing.update({str(k): {proj: v[proj].tolist() for proj in PROJECTIONS} for k, v in out.items()})
    cache_path.write_text(json.dumps(existing))
    print(f"[rank-corr-b] {condition} seed{seed}: cached -> {cache_path}")
    return out


def load_expert_freq(condition, seed, layers):
    """Returns {layer_idx: np.ndarray[n_experts]} of routing-frequency counts,
    from the real (max_samples_freq=2000) expert_frequencies.json -- required
    to exist by that exact filename, not picked via glob()[0] (some conditions
    also have a leftover "_50" --smoke file on disk)."""
    freq_path = RESULTS_ROOT / condition / f"seed{seed}" / EXPERT_FREQ_FILENAME
    if not freq_path.exists():
        raise FileNotFoundError(
            f"{freq_path} missing -- expert_freq not found for {condition}/seed{seed}; "
            "recomputing it is a GPU step (phase1_run_freq_and_scale.py), not run here per instructions"
        )
    raw = json.loads(freq_path.read_text())
    return {l: np.asarray(raw[str(l)], dtype=np.float64) for l in layers}


def merge_weighted(raw_by_proj, freq_by_layer, layers):
    """F_e_merge[proj][e] = F_e_raw[proj][e] * expert_freq[e] (elementwise across
    experts, per layer, per projection -- see module docstring for why this
    factoring is exact, not an approximation)."""
    out = {}
    for l in layers:
        freq = freq_by_layer[l]
        out[l] = {proj: raw_by_proj[l][proj] * freq for proj in PROJECTIONS}
    return out


def part_a_definition_agreement(raw_all, merge_all, conditions, layers):
    """Same (condition, layer): Spearman rank-corr of raw-F_e vs merge-weighted
    F_e, per projection, plus a "combined" (sum of the 3 projections) line for
    reference against phase1_51's single-scalar F_e definition."""
    rows = []
    for cond in conditions:
        for l in layers:
            row = {"condition": cond, "layer": l}
            raw_combined = np.zeros_like(raw_all[cond][l]["gate_proj"])
            merge_combined = np.zeros_like(raw_combined)
            for proj in PROJECTIONS:
                rho, p = spearmanr(raw_all[cond][l][proj], merge_all[cond][l][proj])
                row[proj] = {"rho": float(rho), "p": float(p)}
                raw_combined += raw_all[cond][l][proj]
                merge_combined += merge_all[cond][l][proj]
            rho, p = spearmanr(raw_combined, merge_combined)
            row["combined"] = {"rho": float(rho), "p": float(p)}
            rows.append(row)

    summary = {}
    for key in PROJECTIONS + ("combined",):
        rhos = [r[key]["rho"] for r in rows]
        summary[key] = {"mean_rho": float(np.mean(rhos)), "min_rho": float(np.min(rhos)), "n": len(rhos)}
    summary["overall_mean_rho_across_all_keys"] = float(
        np.mean([summary[k]["mean_rho"] for k in PROJECTIONS + ("combined",)])
    )
    return rows, summary


def combined_merge_scalar(merge_all, conditions, layers):
    return {
        cond: {l: sum(merge_all[cond][l][proj] for proj in PROJECTIONS) for l in layers}
        for cond in conditions
    }


def part_b_headline(combined_scalars, layers):
    """Re-runs phase1_51_fisher_rank_correlation.py's compute_correlations/
    summarize logic (lang-pair vs placebo-pair Spearman, per layer, averaged),
    but on the merge-weighted combined scalar instead of the raw one."""
    lang_pairs = list(itertools.combinations(LANG_CONDITIONS, 2))
    placebo_pairs = list(PLACEBO_OF.items())
    per_layer = {}
    for layer in layers:
        lang_rhos, placebo_rhos = [], []
        for a, b in lang_pairs:
            rho, _ = spearmanr(combined_scalars[a][layer], combined_scalars[b][layer])
            lang_rhos.append(rho)
        for a, b in placebo_pairs:
            rho, _ = spearmanr(combined_scalars[a][layer], combined_scalars[b][layer])
            placebo_rhos.append(rho)
        per_layer[layer] = {
            "mean_lang_rho": float(np.mean(lang_rhos)),
            "mean_placebo_rho": float(np.mean(placebo_rhos)) if placebo_rhos else None,
        }
    overall_lang = float(np.mean([v["mean_lang_rho"] for v in per_layer.values()]))
    placebo_vals = [v["mean_placebo_rho"] for v in per_layer.values() if v["mean_placebo_rho"] is not None]
    overall_placebo = float(np.mean(placebo_vals)) if placebo_vals else None
    return per_layer, {"mean_lang_rho": overall_lang, "mean_placebo_rho": overall_placebo}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="2 layers, english_only(+_b) only -- pipeline sanity check")
    args = parser.parse_args()

    layers = [1, 27] if args.smoke else ALL_MOE_LAYERS
    conditions = ["english_only", "english_only_b"] if args.smoke else (
        LANG_CONDITIONS + list(PLACEBO_OF.values())
    )

    print(f"[rank-corr-b] {'SMOKE' if args.smoke else 'FULL'}: seed={args.seed} layers={layers} conditions={conditions}")

    raw_all, freq_all = {}, {}
    for cond in conditions:
        raw_all[cond] = reduce_condition_by_proj(cond, args.seed, layers)
        freq_all[cond] = load_expert_freq(cond, args.seed, layers)
    merge_all = {cond: merge_weighted(raw_all[cond], freq_all[cond], layers) for cond in conditions}

    if args.smoke:
        print("[rank-corr-b] smoke reduction OK, skipping correlation (needs >=2 languages for part b)")
        return

    rows, def_summary = part_a_definition_agreement(raw_all, merge_all, conditions, layers)
    combined = combined_merge_scalar(merge_all, conditions, layers)
    per_layer_headline, headline_summary = part_b_headline(combined, layers)

    RAW_SEED0_BASELINE = {"mean_lang_rho": 0.3138953330619997, "mean_placebo_rho": 0.9518914213358657}

    out_path = RESULTS_ROOT / f"phase1_51b_merge_weight_rank_correlation_result_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "seed": args.seed, "layers": layers, "conditions": conditions, "placebo_of": PLACEBO_OF,
        "part_a_definition_agreement": {"rows": rows, "summary": def_summary},
        "part_b_headline_reproduction": {
            "per_layer": {str(k): v for k, v in per_layer_headline.items()},
            "merge_weighted": headline_summary,
            "raw_seed0_baseline": RAW_SEED0_BASELINE if args.seed == 0 else None,
        },
    }, indent=2))
    print(f"[rank-corr-b] wrote {out_path}")

    print("\n[rank-corr-b] === part (a) definition agreement: raw-F_e rank vs merge-weight rank ===")
    for key in PROJECTIONS + ("combined",):
        s = def_summary[key]
        flag = " >0.9" if s["mean_rho"] > 0.9 else ""
        print(f"  {key}: mean_rho={s['mean_rho']:.4f} min_rho={s['min_rho']:.4f} n={s['n']}{flag}")
    print(f"  overall mean across all keys: {def_summary['overall_mean_rho_across_all_keys']:.4f}")

    print("\n[rank-corr-b] === part (b) headline reproduction (merge-weighted combined scalar) ===")
    print(f"  merge-weighted: mean_lang_rho={headline_summary['mean_lang_rho']:.4f} "
          f"mean_placebo_rho={headline_summary['mean_placebo_rho']:.4f}")
    if args.seed == 0:
        print(f"  raw seed0 baseline: mean_lang_rho={RAW_SEED0_BASELINE['mean_lang_rho']:.4f} "
              f"mean_placebo_rho={RAW_SEED0_BASELINE['mean_placebo_rho']:.4f}")


if __name__ == "__main__":
    main()
