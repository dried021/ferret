"""Phase 1 whitening-geometry ANALYSIS (consumes phase1_whitening_geometry.py's
whitening_geometry_pairs.csv): turns the raw pairwise table into the
09_부족한_실험_정리.md item 3 / RQ-W1 headline claim -- "calibration-language
whitening subspaces differ MORE than same-language seed-resampling noise" --
plus the layer-curve and language-pair-heatmap data the two figures need.

Design-note section 12's core test: within-language distance (same
condition, different calibration seed -- the seed/sampling-noise floor)
vs. between-language distance (different condition, any seed pair). The
headline claim needs between > within by more than seed noise alone; this
script reports the ratio R = mean(between)/mean(within) and a permutation
test (shuffle the 18 condition x seed nodes' CONDITION LABELS 10,000 times,
holding the already-computed 153 pairwise values fixed, and recompute the
within/between split each time -- cheap, no GPU/recompute needed, just
relabeling arithmetic over the small CSV already on disk).

"Late layers" = the last 25% of MoE layers (design note §11's predefined
subset, avoiding a post-hoc cutoff pick) -- for DeepSeek-MoE-16B's 27 MoE
layers (1..27), that's the last 7 layers (21..27).

Usage:
    conda run -n d2moe_env python phase1_whitening_geometry_analysis.py \
        [--pairs-csv PATH] [--metric mean_angle_deg_k64] [--n-permutations 10000] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from disagreement_common import conditions_for  # noqa: E402
from phase1_fisher import MOE_LAYERS  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
CONDITIONS = conditions_for("mixed_5lang")
LATE_LAYER_FRACTION = 0.25
LATE_LAYERS = MOE_LAYERS[-max(1, round(len(MOE_LAYERS) * LATE_LAYER_FRACTION)):]

PRIMARY_METRICS = [
    "mean_angle_deg_k64", "proj_dist_k64", "whitening_reldist_k64",
]
SECONDARY_METRICS = [
    "max_angle_deg_k64", "cov_reldist_k64",
    "mean_angle_deg_k32", "proj_dist_k32", "whitening_reldist_k32",
    "mean_angle_deg_k16", "proj_dist_k16", "whitening_reldist_k16",
]


def load_pairs(csv_path):
    df = pd.read_csv(csv_path)
    df["within"] = df["cond_a"] == df["cond_b"]
    return df


def within_between_summary(df, metric, layers=None):
    """One row: within-language vs between-language mean/std/n, ratio,
    permutation p-value (H1: between > within). `layers=None` pools all
    layers; pass a layer subset (e.g. LATE_LAYERS) to restrict."""
    sub = df if layers is None else df[df["layer"].isin(layers)]
    within_vals = sub.loc[sub["within"], metric].to_numpy()
    between_vals = sub.loc[~sub["within"], metric].to_numpy()
    mean_w, mean_b = within_vals.mean(), between_vals.mean()
    return {
        "metric": metric,
        "n_within": int(len(within_vals)), "n_between": int(len(between_vals)),
        "mean_within": float(mean_w), "std_within": float(within_vals.std(ddof=1)),
        "mean_between": float(mean_b), "std_between": float(between_vals.std(ddof=1)),
        "ratio_between_over_within": float(mean_b / mean_w) if mean_w > 0 else float("nan"),
    }


def permutation_test(df, metric, layers=None, n_permutations=10000, seed=0):
    """Relabels the 6 condition names across the 18 (condition, seed) nodes
    (a permutation of WHICH condition each node belongs to, holding node
    identity -- and every already-computed pairwise metric value -- fixed),
    recomputes the within/between split under each relabeling, and reports
    the fraction of permutations whose ratio_between_over_within >= the
    observed ratio (one-sided: H1 is "language differs more than sampling
    noise", i.e. observed ratio should sit in the upper tail of the null).

    Implementation note: conditions have 3 seeds each in this design (see
    phase1_whitening_geometry.py's SEEDS), so relabeling permutes the 6
    condition NAMES across 6 groups of 3 nodes each (not a free permutation
    of all 18 nodes individually) -- this preserves the within-group seed
    structure the null should respect (same seed-count-per-language as the
    real data), matching the design note §12 "permutation: language label을
    seed-level whitening object 사이에서 섞음" prescription."""
    sub = df if layers is None else df[df["layer"].isin(layers)]
    rng = np.random.default_rng(seed)

    nodes = sorted(set(sub["cond_a"]) | set(sub["cond_b"]))
    seeds_by_cond = {c: sorted(set(sub.loc[sub["cond_a"] == c, "seed_a"]) | set(sub.loc[sub["cond_b"] == c, "seed_b"]))
                     for c in nodes}
    node_ids = [(c, s) for c in nodes for s in seeds_by_cond[c]]
    node_index = {ns: i for i, ns in enumerate(node_ids)}

    pair_a = np.fromiter((node_index[t] for t in zip(sub["cond_a"], sub["seed_a"])), dtype=int, count=len(sub))
    pair_b = np.fromiter((node_index[t] for t in zip(sub["cond_b"], sub["seed_b"])), dtype=int, count=len(sub))
    values = sub[metric].to_numpy()
    orig_cond_of_node = np.array([nodes.index(c) for c, s in node_ids])

    observed = within_between_summary(sub, metric)["ratio_between_over_within"]

    null_ratios = np.empty(n_permutations)
    for p in range(n_permutations):
        # Shuffle WHICH node holds which condition label (breaks the real
        # seed-triplet grouping) -- relabeling the 6 group IDENTITIES with a
        # bijection instead (permutation of range(len(nodes)) composed with
        # orig_cond_of_node) would preserve every group's membership and
        # give an identical within/between split every single draw, which
        # is what an earlier version of this loop did by mistake.
        perm_cond_labels = rng.permutation(orig_cond_of_node)
        is_within = perm_cond_labels[pair_a] == perm_cond_labels[pair_b]
        mean_w = values[is_within].mean() if is_within.any() else np.nan
        mean_b = values[~is_within].mean() if (~is_within).any() else np.nan
        null_ratios[p] = mean_b / mean_w if mean_w and mean_w > 0 else np.nan

    valid = ~np.isnan(null_ratios)
    p_value = float((null_ratios[valid] >= observed).sum() / valid.sum())
    return {
        "observed_ratio": float(observed), "n_permutations": int(valid.sum()),
        "p_value_between_ge_within": p_value,
        "null_ratio_mean": float(np.nanmean(null_ratios)), "null_ratio_std": float(np.nanstd(null_ratios)),
    }


def language_pair_heatmap(df, metric, layers=None):
    """6x6 matrix (CONDITIONS order), cell = mean(metric) over seed-pairs and
    the given layer subset, for cond_a/cond_b in both orders (off-diagonal
    averaged over both directions since the metrics are symmetric already,
    but cond_a/cond_b storage order in the CSV is itertools.combinations'
    arbitrary node order, not necessarily alphabetical). Diagonal = mean
    within-language (cross-seed) distance for that condition, NOT zero --
    the design note's recommended "diagonal annotation: within-language seed
    distance" convention (§13 Figure B)."""
    sub = df if layers is None else df[df["layer"].isin(layers)]
    mat = np.full((len(CONDITIONS), len(CONDITIONS)), np.nan)
    idx = {c: i for i, c in enumerate(CONDITIONS)}
    for _, r in sub.groupby(["cond_a", "cond_b"])[metric].mean().reset_index().iterrows():
        i, j = idx[r["cond_a"]], idx[r["cond_b"]]
        mat[i, j] = r[metric]
        mat[j, i] = r[metric]
    return mat


def layer_curve(df, metric):
    """Per-layer within-mean and between-mean (pooled over matrix_type and
    all pairs), for Figure A -- design note §13 "layer-wise between vs
    within, 95% bootstrap CI shaded"."""
    rows = []
    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer]
        w = sub.loc[sub["within"], metric].to_numpy()
        b = sub.loc[~sub["within"], metric].to_numpy()
        rows.append({
            "layer": int(layer),
            "within_mean": float(w.mean()), "within_lo": float(np.percentile(w, 2.5)), "within_hi": float(np.percentile(w, 97.5)),
            "between_mean": float(b.mean()), "between_lo": float(np.percentile(b, 2.5)), "between_hi": float(np.percentile(b, 97.5)),
            "n_within": int(len(w)), "n_between": int(len(b)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", type=Path, default=OUT_DIR / "whitening_geometry_pairs.csv")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-stem", default="whitening_geometry_analysis")
    args = parser.parse_args()

    if not args.pairs_csv.exists():
        raise FileNotFoundError(f"{args.pairs_csv} missing -- run phase1_whitening_geometry.py first")
    df = load_pairs(args.pairs_csv)
    print(f"[whitening-analysis] loaded {len(df)} rows from {args.pairs_csv} "
          f"({df['layer'].nunique()} layers, {df['matrix_type'].nunique()} matrix types)")

    summary = {"late_layers": LATE_LAYERS, "conditions_order": CONDITIONS, "metrics": {}}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        all_layer = within_between_summary(df, metric)
        late_layer = within_between_summary(df, metric, layers=LATE_LAYERS)
        perm_all = permutation_test(df, metric, n_permutations=args.n_permutations, seed=args.seed)
        perm_late = permutation_test(df, metric, layers=LATE_LAYERS, n_permutations=args.n_permutations, seed=args.seed)
        summary["metrics"][metric] = {
            "all_layers": {**all_layer, "permutation": perm_all},
            "late_layers": {**late_layer, "permutation": perm_late},
        }
        print(f"[whitening-analysis] {metric}: all-layer ratio={all_layer['ratio_between_over_within']:.3f} "
              f"(p={perm_all['p_value_between_ge_within']:.4f}); "
              f"late-layer ratio={late_layer['ratio_between_over_within']:.3f} "
              f"(p={perm_late['p_value_between_ge_within']:.4f})")

    out_json = OUT_DIR / f"{args.out_stem}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[whitening-analysis] wrote {out_json}")

    heatmap_metric = "mean_angle_deg_k64"
    heatmap_all = language_pair_heatmap(df, heatmap_metric)
    heatmap_late = language_pair_heatmap(df, heatmap_metric, layers=LATE_LAYERS)
    np.savez(OUT_DIR / f"{args.out_stem}_heatmap.npz",
             conditions=np.array(CONDITIONS), all_layers=heatmap_all, late_layers=heatmap_late,
             metric=np.array(heatmap_metric))
    print(f"[whitening-analysis] wrote {OUT_DIR / f'{args.out_stem}_heatmap.npz'}")

    curve = layer_curve(df, heatmap_metric)
    out_curve = OUT_DIR / f"{args.out_stem}_layer_curve.json"
    out_curve.write_text(json.dumps({"metric": heatmap_metric, "layers": curve}, indent=2))
    print(f"[whitening-analysis] wrote {out_curve}")


if __name__ == "__main__":
    main()
