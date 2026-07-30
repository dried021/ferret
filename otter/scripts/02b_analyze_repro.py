"""Phase 0.5 gate: does Toy0's pattern (layer-8 FAIL, layer-22/38 PASS)
reproduce across seeds and a token-budget-controlled sample, using the
pre-registered decision rule fixed *before* this data was collected (see
data/phase0_5_config.yaml `gate:` block and README.md "Phase 0.5 게이트").

Five criteria, >=3 met -> proceed to Phase 0/1 (본실험):
  1. median ratio (pooled over seeds) >= gate.ratio_ok_threshold at BOTH
     layer 22 and layer 38
  2. layer 38 is per-seed PASS (ratio >= GATE_MARGIN) in at least
     gate.layer38_pass_min_seeds seeds
  3. ratio(layer 38) > ratio(layer 8) in a majority of seeds
  4. within-English Fisher-proxy correlation > EN-KO and > EN-ZH correlation,
     at layer 38, in a majority of seeds
  5. bootstrap CI (resampling seeds) on the layer-38 ratio has a lower bound
     > 1.0

If 1-2 criteria hold AND criterion 2 alone is stable -> CONDITIONAL (narrow
the paper's claim to late-layer calibration-language sensitivity). Otherwise
-> STOP_OR_NARROW.

Usage:
    conda run -n torch_env python 02b_analyze_repro.py [--smoke]
"""
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent

GATE_MARGIN = 1.5          # same per-(seed,layer) PASS/FAIL threshold as Toy0
TOP_K_FRACTION = 0.25
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
REFERENCE_EARLY_LAYER = 8  # the "layer 8" side of criterion 3
REFERENCE_LATE_LAYER = 38  # the "layer 38" side of criteria 2/3/4/5
MID_LAYER_FOR_CRIT1 = 22


def load_config():
    spec = importlib.util.spec_from_file_location("d2moe_ml_config", SCRIPT_DIR / "00_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_dist(hit_count, eps=1e-12):
    arr = np.array(hit_count, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.full_like(arr, 1.0 / len(arr))
    return (arr + eps) / (total + eps * len(arr))


def analyze_seed(seed_data, layers):
    """Returns {layer: {"noise_floor":, "cross_lang_mean":, "ratio":,
    "pass":, "within_en_rho":, "en_ko_rho":, "en_zh_rho":}}."""
    conditions = list(seed_data.keys())
    per_layer = {}
    for layer in layers:
        layer_key = str(layer)
        dists = {c: normalized_dist(seed_data[c]["hit_count"][layer_key]) for c in conditions}
        fisher = {c: np.array(seed_data[c]["fisher_proxy"][layer_key], dtype=np.float64) for c in conditions}

        pair_js = {}
        for c1, c2 in itertools.combinations(conditions, 2):
            pair_js[frozenset((c1, c2))] = float(jensenshannon(dists[c1], dists[c2], base=2))

        noise_floor = pair_js[frozenset(("english_a", "english_b"))]
        cross_pairs = [k for k in pair_js if k != frozenset(("english_a", "english_b"))]
        cross_mean = float(np.mean([pair_js[k] for k in cross_pairs]))
        ratio = cross_mean / noise_floor if noise_floor else float("nan")

        within_en_rho, _ = spearmanr(fisher["english_a"], fisher["english_b"])
        en_ko_rho, _ = spearmanr(fisher["english_a"], fisher["korean"])
        en_zh_rho, _ = spearmanr(fisher["english_a"], fisher["chinese"])

        per_layer[layer] = {
            "noise_floor": noise_floor,
            "cross_lang_mean": cross_mean,
            "ratio": ratio,
            "pass": bool(ratio >= GATE_MARGIN),
            "within_en_rho": float(within_en_rho),
            "en_ko_rho": float(en_ko_rho),
            "en_zh_rho": float(en_zh_rho),
        }
    return per_layer


def bootstrap_ci(values, n_resamples, seed, alpha=0.05):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=np.float64)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(values.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    spec = cfg.load_phase0_5_spec()
    layers = spec["layer_indices"]
    seeds = [spec["seeds"][0]] if args.smoke else spec["seeds"]
    gate_cfg = spec["gate"]

    per_seed = {}
    for seed in seeds:
        path = cfg.phase0_5_stats_json(seed)
        if args.smoke:
            path = path.with_name(path.stem + "_smoke" + path.suffix)
        seed_data = json.loads(path.read_text())
        per_seed[seed] = analyze_seed(seed_data, layers)

    # --- seed x layer table ---
    table_rows = []
    for seed in seeds:
        for layer in layers:
            row = {"seed": seed, "layer": layer, **per_seed[seed][layer]}
            table_rows.append(row)
    table_csv = cfg.PHASE0_5_SEED_LAYER_CSV
    if args.smoke:
        table_csv = table_csv.with_name(table_csv.stem + "_smoke" + table_csv.suffix)
    with open(table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"[02b] wrote {table_csv}")

    # --- per-layer aggregation across seeds ---
    median_ratio = {l: float(np.median([per_seed[s][l]["ratio"] for s in seeds])) for l in layers}
    for l in layers:
        print(f"[02b] layer {l}: median ratio (n={len(seeds)} seeds) = {median_ratio[l]:.3f}")

    # --- criterion 1: median ratio >= threshold at both mid and late layer ---
    crit1 = (
        MID_LAYER_FOR_CRIT1 in median_ratio and REFERENCE_LATE_LAYER in median_ratio and
        median_ratio[MID_LAYER_FOR_CRIT1] >= gate_cfg["ratio_ok_threshold"] and
        median_ratio[REFERENCE_LATE_LAYER] >= gate_cfg["ratio_ok_threshold"]
    )

    # --- criterion 2: layer 38 PASS in >= layer38_pass_min_seeds seeds ---
    layer38_pass_count = sum(1 for s in seeds if per_seed[s][REFERENCE_LATE_LAYER]["pass"])
    crit2 = layer38_pass_count >= gate_cfg["layer38_pass_min_seeds"]

    # --- criterion 3: ratio(38) > ratio(8) in a majority of seeds ---
    seeds_38_gt_8 = sum(
        1 for s in seeds
        if per_seed[s][REFERENCE_LATE_LAYER]["ratio"] > per_seed[s][REFERENCE_EARLY_LAYER]["ratio"]
    )
    crit3 = seeds_38_gt_8 > len(seeds) / 2

    # --- criterion 4: within-EN rho > EN-KO and > EN-ZH at layer 38, majority of seeds ---
    seeds_fisher_gap = sum(
        1 for s in seeds
        if per_seed[s][REFERENCE_LATE_LAYER]["within_en_rho"] > per_seed[s][REFERENCE_LATE_LAYER]["en_ko_rho"]
        and per_seed[s][REFERENCE_LATE_LAYER]["within_en_rho"] > per_seed[s][REFERENCE_LATE_LAYER]["en_zh_rho"]
    )
    crit4 = seeds_fisher_gap > len(seeds) / 2

    # --- criterion 5: bootstrap CI on layer-38 ratio (resampling seeds), lower bound > 1.0 ---
    layer38_ratios = [per_seed[s][REFERENCE_LATE_LAYER]["ratio"] for s in seeds]
    boot_lo, boot_hi, boot_mean = bootstrap_ci(layer38_ratios, N_BOOTSTRAP, BOOTSTRAP_SEED)
    crit5 = boot_lo > 1.0

    criteria = {
        "1_median_ratio_mid_and_late_layer": crit1,
        "2_layer38_pass_min_seeds": crit2,
        "3_layer38_ratio_gt_layer8_majority_seeds": crit3,
        "4_fisher_within_en_gt_cross_lang_majority_seeds": crit4,
        "5_bootstrap_ci_layer38_excludes_1.0": crit5,
    }
    n_criteria_met = sum(criteria.values())

    if n_criteria_met >= 3:
        verdict = "PROGRESS_PASS"
        verdict_msg = ("PROGRESS_PASS -- >=3/5 pre-registered criteria met. Proceed to "
                       "README.md Phase 0 (official D^2-MoE repo) and Phase 1 pilot.")
    elif crit2:
        verdict = "CONDITIONAL"
        verdict_msg = ("CONDITIONAL -- layer 38 alone is stable across seeds but fewer than "
                       "3/5 criteria hold overall. Narrow the paper's claim to late-layer "
                       "calibration-language sensitivity rather than a general MoE-wide claim.")
    else:
        verdict = "STOP_OR_NARROW"
        verdict_msg = ("STOP_OR_NARROW -- neither the overall criteria count nor layer 38 alone "
                       "is stable across seeds. Check whether ratios are converging toward 1.0 "
                       "as seeds/tokens increase (null result) before any further scale-up.")

    # --- layer depth vs ratio: per-seed Spearman + slope, plus pooled Spearman ---
    per_seed_trend = {}
    for s in seeds:
        ratios = [per_seed[s][l]["ratio"] for l in layers]
        rho, pval = spearmanr(layers, ratios)
        slope = float(np.polyfit(layers, ratios, 1)[0])
        per_seed_trend[s] = {"spearman_rho": float(rho), "spearman_p": float(pval), "slope": slope}

    pooled_layers = [l for s in seeds for l in layers]
    pooled_ratios = [per_seed[s][l]["ratio"] for s in seeds for l in layers]
    pooled_rho, pooled_p = spearmanr(pooled_layers, pooled_ratios)

    gate_summary = {
        "seeds": seeds,
        "layers": layers,
        "gate_config": gate_cfg,
        "median_ratio_by_layer": median_ratio,
        "layer38_pass_count": layer38_pass_count,
        "seeds_with_layer38_gt_layer8": seeds_38_gt_8,
        "seeds_with_fisher_gap_at_layer38": seeds_fisher_gap,
        "bootstrap_layer38_ratio": {"lo": boot_lo, "hi": boot_hi, "mean": boot_mean, "n_resamples": N_BOOTSTRAP},
        "criteria": criteria,
        "n_criteria_met": n_criteria_met,
        "verdict": verdict,
        "verdict_message": verdict_msg,
        "per_seed_layer_depth_trend": per_seed_trend,
        "pooled_layer_depth_spearman": {"rho": float(pooled_rho), "p": float(pooled_p)},
    }

    gate_path = cfg.PHASE0_5_GATE_JSON
    if args.smoke:
        gate_path = gate_path.with_name(gate_path.stem + "_smoke" + gate_path.suffix)
    gate_path.write_text(json.dumps(gate_summary, indent=2))
    print(f"[02b] wrote {gate_path}")

    print(f"[02b] criteria met: {n_criteria_met}/5 -> {criteria}")
    print(f"[02b] bootstrap layer-38 ratio 95% CI: [{boot_lo:.3f}, {boot_hi:.3f}] (mean {boot_mean:.3f})")
    print(f"[02b] VERDICT: {verdict_msg}")


if __name__ == "__main__":
    main()
