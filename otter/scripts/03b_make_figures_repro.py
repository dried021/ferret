"""Phase 0.5 figures -- does the Toy0 ratio/Fisher pattern hold up across
seeds? Renders 02b_analyze_repro.py's output into figures/*.png.

Usage:
    conda run -n torch_env python 03b_make_figures_repro.py [--smoke]
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# Fixed hue assignment, Okabe-Ito colorblind-safe -- seeds get their own
# identity color; within-EN reuses the "noise floor" gray from Toy0's
# figures, EN-KO/EN-ZH reuse ko_only/zh_only's colors for visual continuity.
SEED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
FISHER_COLORS = {"within_en_rho": MUTED, "en_ko_rho": "#D55E00", "en_zh_rho": "#009E73"}
FISHER_LABELS = {
    "within_en_rho": "within-English (english_a vs english_b, placebo)",
    "en_ko_rho": "English vs Korean",
    "en_zh_rho": "English vs Chinese",
}


def load_config():
    spec = importlib.util.spec_from_file_location("d2moe_ml_config", SCRIPT_DIR / "00_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_ratio_trend(table_rows, gate_summary, out_path):
    seeds = sorted({int(r["seed"]) for r in table_rows})
    layers = sorted({int(r["layer"]) for r in table_rows})

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)

    for i, seed in enumerate(seeds):
        ratios = [float(r["ratio"]) for r in table_rows if int(r["seed"]) == seed]
        ratios = [r for _, r in sorted(zip([int(rr["layer"]) for rr in table_rows if int(rr["seed"]) == seed], ratios))]
        ax.plot(layers, ratios, color=SEED_COLORS[i % len(SEED_COLORS)], marker="o", markersize=5,
                linewidth=1.6, alpha=0.85, label=f"seed {seed}", zorder=3)

    median_ratio = [gate_summary["median_ratio_by_layer"][str(l)] for l in layers]
    ax.plot(layers, median_ratio, color=INK, linewidth=2.4, linestyle="-", marker="s", markersize=6,
            label="median across seeds", zorder=4)

    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax.text(layers[0], 1.0, " ratio = 1.0 (no language effect)", fontsize=8, color=MUTED, va="bottom")
    threshold = gate_summary["gate_config"]["ratio_ok_threshold"]
    ax.axhline(threshold, color="#D55E00", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(layers[0], threshold, f" gate threshold = {threshold}", fontsize=8, color="#D55E00", va="bottom")

    ax.set_xlabel("layer index")
    ax.set_ylabel("cross-language / same-language JS divergence ratio")
    ax.set_title("Phase 0.5: does the language-divergence ratio reproduce across seeds?",
                 fontsize=11, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03b] wrote {out_path}")


def fig_fisher_trend(table_rows, out_path):
    seeds = sorted({int(r["seed"]) for r in table_rows})
    layers = sorted({int(r["layer"]) for r in table_rows})

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)

    for key in ("within_en_rho", "en_ko_rho", "en_zh_rho"):
        per_layer_vals = []
        for l in layers:
            vals = [float(r[key]) for r in table_rows if int(r["layer"]) == l]
            per_layer_vals.append(vals)
        means = [np.mean(v) for v in per_layer_vals]
        lo = [np.min(v) for v in per_layer_vals]
        hi = [np.max(v) for v in per_layer_vals]
        color = FISHER_COLORS[key]
        ax.plot(layers, means, color=color, linewidth=2.0, marker="o", markersize=5,
                label=FISHER_LABELS[key], zorder=3)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.15, zorder=1, linewidth=0)

    ax.set_ylim(-1.0, 1.05)
    ax.set_xlabel("layer index")
    ax.set_ylabel("Fisher-proxy Spearman correlation")
    ax.set_title(f"Fisher-proxy rank correlation by layer (mean, min-max band over {len(seeds)} seeds)",
                 fontsize=11, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03b] wrote {out_path}")


def fig_bootstrap_ci(gate_summary, out_path):
    boot = gate_summary["bootstrap_layer38_ratio"]
    late_layer = [l for l in gate_summary["layers"] if str(l) == str(max(gate_summary["layers"]))][0]

    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=160)
    ax.errorbar([0], [boot["mean"]], yerr=[[boot["mean"] - boot["lo"]], [boot["hi"] - boot["mean"]]],
                fmt="o", markersize=10, color="#0072B2", ecolor="#0072B2", elinewidth=2.2, capsize=6, zorder=3)
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=":", zorder=2)
    ax.text(0.15, 1.0, "ratio = 1.0", fontsize=8.5, color=MUTED, va="center")

    ax.set_xlim(-0.6, 0.9)
    ax.set_xticks([0])
    ax.set_xticklabels([f"layer {late_layer}"])
    ax.set_ylabel("cross-language / same-language JS divergence ratio")
    ax.set_title(f"Bootstrap 95% CI\n(resampling {len(gate_summary['seeds'])} seeds, "
                 f"n={boot['n_resamples']})", fontsize=10.5, color=INK)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03b] wrote {out_path}")


def fig_gate_scorecard(gate_summary, out_path):
    criteria = gate_summary["criteria"]
    labels = [
        "1. median ratio >= threshold\n   at mid AND late layer",
        "2. late layer PASS in\n   >= min seeds",
        "3. late-layer ratio > early-layer\n   ratio, majority of seeds",
        "4. within-EN Fisher corr >\n   cross-lingual, majority of seeds",
        "5. bootstrap CI excludes 1.0",
    ]
    values = list(criteria.values())

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=160)
    y = np.arange(len(labels))[::-1]
    colors = ["#009E73" if v else "#D55E00" for v in values]
    ax.barh(y, [1] * len(values), color=colors, height=0.6, zorder=3)
    for yi, v in zip(y, values):
        ax.text(0.5, yi, "PASS" if v else "FAIL", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_met = gate_summary["n_criteria_met"]
    verdict = gate_summary["verdict"]
    fig.suptitle(f"Phase 0.5 pre-registered gate: {n_met}/5 criteria met -> {verdict}",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03b] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    table_csv = cfg.PHASE0_5_SEED_LAYER_CSV
    gate_json = cfg.PHASE0_5_GATE_JSON
    if args.smoke:
        table_csv = table_csv.with_name(table_csv.stem + "_smoke" + table_csv.suffix)
        gate_json = gate_json.with_name(gate_json.stem + "_smoke" + gate_json.suffix)

    table_rows = list(csv.DictReader(open(table_csv)))
    gate_summary = json.loads(gate_json.read_text())

    suffix = "_smoke" if args.smoke else ""
    fig_ratio_trend(table_rows, gate_summary, cfg.FIGURES_DIR / f"phase0_5_ratio_trend{suffix}.png")
    fig_fisher_trend(table_rows, cfg.FIGURES_DIR / f"phase0_5_fisher_trend{suffix}.png")
    fig_bootstrap_ci(gate_summary, cfg.FIGURES_DIR / f"phase0_5_bootstrap_ci{suffix}.png")
    fig_gate_scorecard(gate_summary, cfg.FIGURES_DIR / f"phase0_5_gate_scorecard{suffix}.png")


if __name__ == "__main__":
    main()
