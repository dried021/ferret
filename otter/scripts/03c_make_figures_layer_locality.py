"""0) Layer-38 locality figures -- renders 02c_analyze_layer_locality.py's
output: does the Fisher-proxy EN-vs-non-EN gap stay confined to layer 38, or
spread across the back half (up to Qwen3-30B-A3B's true final layer, 47)?

Usage:
    conda run -n torch_env python 03c_make_figures_layer_locality.py [--smoke]
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
PREFIX = "layer_locality"

SEED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"


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


def fig_ratio_trend(table_rows, out_path):
    seeds = sorted({int(r["seed"]) for r in table_rows})
    layers = sorted({int(r["layer"]) for r in table_rows})

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    for i, seed in enumerate(seeds):
        rows = sorted([r for r in table_rows if int(r["seed"]) == seed], key=lambda r: int(r["layer"]))
        ax.plot(layers, [float(r["ratio"]) for r in rows], color=SEED_COLORS[i % len(SEED_COLORS)],
                marker="o", markersize=5, linewidth=1.6, alpha=0.85, label=f"seed {seed}", zorder=3)

    median_ratio = [np.median([float(r["ratio"]) for r in table_rows if int(r["layer"]) == l]) for l in layers]
    ax.plot(layers, median_ratio, color=INK, linewidth=2.4, marker="s", markersize=6,
            label="median across seeds", zorder=4)

    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax.axvspan(38, max(layers), color="#D55E00", alpha=0.06, zorder=1)
    ax.text(38, max(median_ratio) * 1.02, "layers 38+ new in this run -->",
            fontsize=7.5, color=MUTED, ha="left", va="bottom")

    ax.set_xlabel("layer index (Qwen3-30B-A3B has 48 layers total, 0-47)")
    ax.set_ylabel("cross-language / same-language JS divergence ratio")
    ax.set_title("Layer-locality check: routing-divergence ratio across the full back half",
                 fontsize=11, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03c] wrote {out_path}")


def fig_gap_trend(table_rows, summary, out_path):
    seeds = sorted({int(r["seed"]) for r in table_rows})
    layers = sorted({int(r["layer"]) for r in table_rows})
    threshold = summary["gap_threshold"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    for i, seed in enumerate(seeds):
        rows = sorted([r for r in table_rows if int(r["seed"]) == seed], key=lambda r: int(r["layer"]))
        ax.plot(layers, [float(r["fisher_gap"]) for r in rows], color=SEED_COLORS[i % len(SEED_COLORS)],
                marker="o", markersize=5, linewidth=1.6, alpha=0.85, label=f"seed {seed}", zorder=3)

    mean_gap = [summary["mean_fisher_gap_by_layer"][str(l)] for l in layers]
    ax.plot(layers, mean_gap, color=INK, linewidth=2.4, marker="s", markersize=6,
            label="mean across seeds", zorder=4)

    ax.axhline(threshold, color="#D55E00", linewidth=1.2, linestyle="--", zorder=2)
    ax.text(layers[0], threshold, f" locality threshold = {threshold}", fontsize=8, color="#D55E00", va="bottom")
    ax.axhline(0.0, color=MUTED, linewidth=1.0, linestyle=":", zorder=2)

    ax.set_xlabel("layer index (Qwen3-30B-A3B has 48 layers total, 0-47)")
    ax.set_ylabel("Fisher-proxy gap: within-English rho - mean(EN-KO, EN-ZH rho)")
    ax.set_title(f"Where does the EN-vs-non-EN Fisher gap actually live? -> {summary['verdict']}",
                 fontsize=11, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03c] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    table_csv = cfg.seed_layer_csv(PREFIX)
    summary_path = cfg.gate_json(PREFIX)
    if args.smoke:
        table_csv = table_csv.with_name(table_csv.stem + "_smoke" + table_csv.suffix)
        summary_path = summary_path.with_name(summary_path.stem + "_smoke" + summary_path.suffix)

    table_rows = list(csv.DictReader(open(table_csv)))
    summary = json.loads(summary_path.read_text())

    suffix = "_smoke" if args.smoke else ""
    fig_ratio_trend(table_rows, cfg.FIGURES_DIR / f"layer_locality_ratio_trend{suffix}.png")
    fig_gap_trend(table_rows, summary, cfg.FIGURES_DIR / f"layer_locality_fisher_gap{suffix}.png")


if __name__ == "__main__":
    main()
