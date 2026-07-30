"""0) Layer-38 locality check (00_docs/02_Toy_실험.md): is the Fisher-proxy
EN-vs-non-EN gap Phase 0.5 found at layer 38 confined to that one layer, or
spread across the back half of the network (layers 38-47)?

Reuses 02b_analyze_repro.py's per-(seed,layer) analysis (noise floor, cross-
language JS ratio, Fisher-proxy Spearman correlations) against the wider
layer sweep from data/layer_locality_config.yaml (4/8/14/22/30/34/38/42/45/47
-- Qwen3-30B-A3B has 48 layers total, 0-47, so 42/45/47 are new and reach the
true final layer). Adds one new quantity: fisher_gap = within_en_rho -
mean(en_ko_rho, en_zh_rho), the size of the EN-vs-non-EN Fisher-correlation
gap Phase 0.5 found concentrated at layer 38.

This does NOT reuse Phase 0.5's 5-criteria PROGRESS_PASS/CONDITIONAL/
STOP_OR_NARROW gate -- that gate answers "should we proceed to Phase 0/1",
which Phase 0.5 already answered. This script answers a narrower question:
which layers, specifically, show the large gap.

Usage:
    conda run -n torch_env python 02c_analyze_layer_locality.py [--smoke]
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
PREFIX = "layer_locality"
GATE_MARGIN = 1.5  # same per-(seed,layer) PASS/FAIL threshold as Toy0/Phase 0.5


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
        fisher_gap = float(within_en_rho - np.mean([en_ko_rho, en_zh_rho]))

        per_layer[layer] = {
            "noise_floor": noise_floor,
            "cross_lang_mean": cross_mean,
            "ratio": ratio,
            "pass": bool(ratio >= GATE_MARGIN),
            "within_en_rho": float(within_en_rho),
            "en_ko_rho": float(en_ko_rho),
            "en_zh_rho": float(en_zh_rho),
            "fisher_gap": fisher_gap,
        }
    return per_layer


def describe_locality(layers, mean_gap_by_layer, threshold):
    """Classifies which contiguous run(s) of layers exceed `threshold`, to
    turn "which layers passed" into the locality/spread verdict text."""
    above = [l for l in layers if mean_gap_by_layer[l] >= threshold]
    if not above:
        return "NONE", f"no layer's mean fisher_gap reaches the {threshold} threshold -- Phase 0.5's layer-38 gap may not replicate at this threshold."

    # contiguous runs among the *measured* layers (not literal layer-index adjacency)
    idx_of = {l: i for i, l in enumerate(layers)}
    above_idx = sorted(idx_of[l] for l in above)
    runs = []
    run = [above_idx[0]]
    for i in above_idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    run_layers = [[layers[i] for i in run] for run in runs]

    if len(run_layers) == 1 and len(run_layers[0]) <= 2 and run_layers[0][-1] == layers[-1]:
        verdict = "LOCALIZED_FINAL"
    elif len(run_layers) == 1 and len(run_layers[0]) >= 4:
        verdict = "SPREAD_BACK_HALF"
    else:
        verdict = "SCATTERED"
    detail = f"layers exceeding threshold (contiguous runs among measured layers): {run_layers}"
    return verdict, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    spec = cfg.load_spec(cfg.LAYER_LOCALITY_YAML)
    layers = spec["layer_indices"]
    seeds = [spec["seeds"][0]] if args.smoke else spec["seeds"]
    gap_threshold = spec["gate"]["fisher_gap_threshold"]

    per_seed = {}
    for seed in seeds:
        path = cfg.stats_json(PREFIX, seed)
        if args.smoke:
            path = path.with_name(path.stem + "_smoke" + path.suffix)
        seed_data = json.loads(path.read_text())
        per_seed[seed] = analyze_seed(seed_data, layers)

    table_rows = []
    for seed in seeds:
        for layer in layers:
            table_rows.append({"seed": seed, "layer": layer, **per_seed[seed][layer]})
    table_csv = cfg.seed_layer_csv(PREFIX)
    if args.smoke:
        table_csv = table_csv.with_name(table_csv.stem + "_smoke" + table_csv.suffix)
    with open(table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"[02c] wrote {table_csv}")

    mean_gap_by_layer = {l: float(np.mean([per_seed[s][l]["fisher_gap"] for s in seeds])) for l in layers}
    std_gap_by_layer = {l: float(np.std([per_seed[s][l]["fisher_gap"] for s in seeds])) for l in layers}
    mean_ratio_by_layer = {l: float(np.mean([per_seed[s][l]["ratio"] for s in seeds])) for l in layers}

    for l in layers:
        print(f"[02c] layer {l}: mean fisher_gap = {mean_gap_by_layer[l]:.3f} "
              f"(+/- {std_gap_by_layer[l]:.3f}), mean ratio = {mean_ratio_by_layer[l]:.2f}")

    verdict, detail = describe_locality(layers, mean_gap_by_layer, gap_threshold)
    print(f"[02c] {detail}")
    print(f"[02c] VERDICT: {verdict}")

    # Spearman(layer, fisher_gap) restricted to the back half (layer >= 22) --
    # is there a monotonic trend *within* the back half specifically, distinct
    # from Phase 0.5's whole-network non-monotonic (W-shaped) ratio trend?
    back_half_layers = [l for l in layers if l >= 22]
    pooled_layers = [l for s in seeds for l in back_half_layers]
    pooled_gaps = [per_seed[s][l]["fisher_gap"] for s in seeds for l in back_half_layers]
    back_half_rho, back_half_p = spearmanr(pooled_layers, pooled_gaps)

    summary = {
        "seeds": seeds,
        "layers": layers,
        "gap_threshold": gap_threshold,
        "mean_fisher_gap_by_layer": mean_gap_by_layer,
        "std_fisher_gap_by_layer": std_gap_by_layer,
        "mean_ratio_by_layer": mean_ratio_by_layer,
        "verdict": verdict,
        "verdict_detail": detail,
        "back_half_layer_vs_gap_spearman": {"rho": float(back_half_rho), "p": float(back_half_p), "layers_used": back_half_layers},
    }
    summary_path = cfg.gate_json(PREFIX)
    if args.smoke:
        summary_path = summary_path.with_name(summary_path.stem + "_smoke" + summary_path.suffix)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[02c] wrote {summary_path}")


if __name__ == "__main__":
    main()
