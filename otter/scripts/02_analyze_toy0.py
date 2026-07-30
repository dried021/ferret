"""Toy0 gate: does calibration language move routing/importance statistics
by more than the noise floor of just re-sampling the *same* language?

Reads results/toy0_routing_fisher_stats.json (from 01_calibration_stats.py)
and, per MoE layer, computes:

  - routing distribution Jensen-Shannon divergence between every condition
    pair, plus top-k expert-overlap (Jaccard)
  - Spearman rank correlation of the Fisher-proxy importance score between
    every condition pair
  - the same-language noise floor: divergence(en_only, en_only_control) --
    two disjoint English sentence sets, so any language *label* difference
    is entirely explained away
  - a gate verdict: cross-language JS divergence must exceed
    GATE_MARGIN x noise_floor on a majority of layers for Toy0 to PASS

This does not touch compression at all -- it only asks whether the two
statistics D^2-MoE's compression procedure would consume (routing coverage,
importance ranking) are even sensitive to calibration language on this
model, before installing the official D^2-MoE repo (README.md Phase 0).

Usage:
    python 02_analyze_toy0.py [--smoke]
"""
import argparse
import csv
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent

GATE_MARGIN = 1.5   # cross-language divergence must be >= this x noise floor
TOP_K_FRACTION = 0.25  # top-25%-by-hit-count experts, for overlap


def load_config():
    spec = importlib.util.spec_from_file_location("d2moe_ml_config", SCRIPT_DIR / "00_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_path(p, smoke):
    return p if not smoke else p.with_name(p.stem + "_smoke" + p.suffix)


def normalized_dist(hit_count, eps=1e-12):
    arr = np.array(hit_count, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.full_like(arr, 1.0 / len(arr))
    return (arr + eps) / (total + eps * len(arr))


def top_k_set(hit_count, k):
    arr = np.array(hit_count, dtype=np.float64)
    order = np.argsort(-arr)
    return set(order[:k].tolist())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    stats_path = smoke_path(cfg.ROUTING_STATS_JSON, args.smoke)
    data = json.loads(stats_path.read_text())

    conditions = list(data.keys())
    layers = list(data[conditions[0]]["hit_count"].keys())
    num_experts = len(data[conditions[0]]["hit_count"][layers[0]])
    top_k = max(1, int(round(num_experts * TOP_K_FRACTION)))

    div_rows = []
    corr_rows = []
    per_layer_gate = {}

    for layer in layers:
        dists = {c: normalized_dist(data[c]["hit_count"][layer]) for c in conditions}
        top_sets = {c: top_k_set(data[c]["hit_count"][layer], top_k) for c in conditions}
        fisher = {c: np.array(data[c]["fisher_proxy"][layer], dtype=np.float64) for c in conditions}

        pair_div = {}
        for c1, c2 in itertools.combinations(conditions, 2):
            js = float(jensenshannon(dists[c1], dists[c2], base=2))
            overlap = jaccard(top_sets[c1], top_sets[c2])
            rho, pval = spearmanr(fisher[c1], fisher[c2])
            pair_div[(c1, c2)] = js
            div_rows.append({
                "layer": layer, "condition_a": c1, "condition_b": c2,
                "js_divergence": js, "top_k_jaccard": overlap,
            })
            corr_rows.append({
                "layer": layer, "condition_a": c1, "condition_b": c2,
                "fisher_spearman_rho": rho, "fisher_spearman_p": pval,
            })

        noise_floor = pair_div.get(("en_only", "en_only_control"))
        if noise_floor is None:
            noise_floor = pair_div.get(("en_only_control", "en_only"))

        cross_lang_pairs = [
            k for k in pair_div
            if {"en_only", "en_only_control"} != set(k)
        ]
        cross_lang_mean = float(np.mean([pair_div[k] for k in cross_lang_pairs])) if cross_lang_pairs else float("nan")

        passed = noise_floor is not None and cross_lang_mean >= GATE_MARGIN * noise_floor
        per_layer_gate[layer] = {
            "noise_floor_js": noise_floor,
            "cross_language_mean_js": cross_lang_mean,
            "ratio": (cross_lang_mean / noise_floor) if noise_floor else None,
            "pass": bool(passed),
        }
        print(f"[02] layer {layer}: noise_floor={noise_floor:.4f} "
              f"cross_lang_mean={cross_lang_mean:.4f} "
              f"ratio={per_layer_gate[layer]['ratio']:.2f} "
              f"-> {'PASS' if passed else 'FAIL'}")

    n_pass = sum(1 for v in per_layer_gate.values() if v["pass"])
    overall_pass = n_pass > len(layers) / 2

    div_csv = smoke_path(cfg.DIVERGENCE_CSV, args.smoke)
    with open(div_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "condition_a", "condition_b", "js_divergence", "top_k_jaccard"])
        writer.writeheader()
        writer.writerows(div_rows)
    print(f"[02] wrote {div_csv}")

    corr_csv = smoke_path(cfg.FISHER_CORR_CSV, args.smoke)
    with open(corr_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "condition_a", "condition_b", "fisher_spearman_rho", "fisher_spearman_p"])
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"[02] wrote {corr_csv}")

    gate_summary = {
        "gate_margin": GATE_MARGIN,
        "top_k_fraction": TOP_K_FRACTION,
        "per_layer": per_layer_gate,
        "n_layers_pass": n_pass,
        "n_layers_total": len(layers),
        "overall_pass": bool(overall_pass),
        "verdict": (
            "PASS -- calibration language moves routing/importance statistics "
            "beyond same-language resampling noise; proceed to README.md Phase 0 "
            "(install official D^2-MoE repo, real Fisher, Phase 1 pilot)."
            if overall_pass else
            "FAIL -- cross-language divergence is not clearly larger than the "
            "same-language noise floor on this model/layer subset/sample size. "
            "Do not proceed to Phase 0/1 yet -- first check whether this is a "
            "sample-size issue (raise N_SENTENCES) or a genuine null result on "
            "Qwen3-30B-A3B, per README.md risks."
        ),
    }
    gate_path = smoke_path(cfg.GATE_SUMMARY_JSON, args.smoke)
    gate_path.write_text(json.dumps(gate_summary, indent=2))
    print(f"[02] wrote {gate_path}")
    print(f"[02] overall: {gate_summary['verdict']}")


if __name__ == "__main__":
    main()
