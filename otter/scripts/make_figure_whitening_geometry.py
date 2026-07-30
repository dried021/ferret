"""Whitening-geometry paper figures (09_부족한_실험_정리.md item 3 / RQ-W1-W2):
consumes phase1_whitening_geometry_analysis.py's outputs (layer-curve JSON +
heatmap NPZ), never touches the raw svd_scale_processed.pt files or GPU --
pure post-processing/plotting, matches make_figure_expert_language_heatmap.py's
INK/MUTED/GRID palette convention so every whitening-mechanism figure in the
paper reads as one figure family.

Figure A (layer-wise): between-language vs within-language (same condition,
different calibration seed -- the sampling-noise floor) principal angle per
MoE layer, shaded 95% band from language-pair-level percentiles already
computed by the analysis script (design note §13 Figure A).

Figure B (heatmap): 6x6 language-pair matrix (design note §13 Figure B) --
off-diagonal = mean between-language distance (upper triangle shown, matrix
is symmetric by construction), diagonal = mean WITHIN-language cross-seed
distance for that condition (annotated, not left as zero -- the design
note's explicit recommendation, since a zero diagonal would visually imply
"same language calibration is perfectly reproducible across seeds", which is
not what's measured or true).

Usage:
    conda run -n d2moe_env python make_figure_whitening_geometry.py \
        [--analysis-stem whitening_geometry_analysis] [--layer-subset all|late]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase1_whitening_geometry_analysis import LATE_LAYERS  # noqa: E402

RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
WITHIN_COLOR = "#0072B2"
BETWEEN_COLOR = "#D55E00"
CMAP = "viridis"

COND_LABEL = {
    "english_only": "en", "chinese_only": "zh", "korean_only": "ko",
    "swahili_only": "sw", "bengali_only": "bn", "mixed_5lang": "mixed5",
}


def plot_layer_curve(curve, metric, out_stem):
    layers = [r["layer"] for r in curve]
    within_mean = np.array([r["within_mean"] for r in curve])
    within_lo = np.array([r["within_lo"] for r in curve])
    within_hi = np.array([r["within_hi"] for r in curve])
    between_mean = np.array([r["between_mean"] for r in curve])
    between_lo = np.array([r["between_lo"] for r in curve])
    between_hi = np.array([r["between_hi"] for r in curve])

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax.fill_between(layers, within_lo, within_hi, color=WITHIN_COLOR, alpha=0.18, linewidth=0, zorder=2)
    ax.fill_between(layers, between_lo, between_hi, color=BETWEEN_COLOR, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(layers, within_mean, color=WITHIN_COLOR, linewidth=1.8, marker="o", markersize=3,
            label="within-language (cross-seed noise floor)", zorder=3)
    ax.plot(layers, between_mean, color=BETWEEN_COLOR, linewidth=1.8, marker="o", markersize=3,
            label="between-language", zorder=3)
    ax.axvspan(min(LATE_LAYERS), max(LATE_LAYERS), color=GRID, alpha=0.5, zorder=1,
               label=f"late layers ({min(LATE_LAYERS)}-{max(LATE_LAYERS)})")

    ax.set_xlabel("MoE layer index", fontsize=10, color=INK)
    ax.set_ylabel(metric.replace("_", " "), fontsize=10, color=INK)
    ax.set_title("Whitening-subspace principal angle: between- vs within-language calibration",
                 fontsize=11, color=INK, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="best")

    caption = (
        "Each point = mean over all language-pair x seed-pair comparisons at that layer, pooled across "
        "gate/up/down projection matrices and all routed (non-dead, non-low-hit) experts; shaded band = "
        "[2.5, 97.5] percentile across those comparisons, not a parametric CI. \"Within-language\" pairs "
        "calibrate on the SAME language with a different calibration-sample seed (the sampling-noise floor); "
        "\"between-language\" pairs calibrate on different languages. If between-language sits above the "
        "within-language band, calibration-language identity moves the whitening subspace by more than "
        "seed-resampling noise alone -- see the accompanying analysis JSON for the permutation-test p-value."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.2, color=MUTED, wrap=True)

    out_path = FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    print(f"[whitening-fig] wrote {out_path}")
    plt.close(fig)


def plot_heatmap(conditions, matrix, metric, layer_note, out_stem):
    n = len(conditions)
    labels = [COND_LABEL.get(c, c) for c in conditions]

    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=300)
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, cmap=CMAP, vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("calibration condition b", fontsize=9.5, color=INK)
    ax.set_ylabel("calibration condition a", fontsize=9.5, color=INK)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val > 0.55 * vmax else INK
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=txt_color, fontweight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Language-pair whitening-subspace distance ({metric.replace('_',' ')}, {layer_note})",
                 fontsize=10.5, color=INK, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    caption = (
        "Off-diagonal = mean between-language distance over all seed pairs. Diagonal (bold) = mean "
        "WITHIN-language distance across calibration seeds for that same condition -- shown, not zeroed, "
        "since it is the sampling-noise floor the off-diagonal values should be compared against, not a "
        "trivial self-distance. 'mixed5' = mixed_5lang (EN/KO/ZH/SW/BN balanced mix)."
    )
    fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=7.2, color=MUTED, wrap=True)

    out_path = FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    print(f"[whitening-fig] wrote {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-stem", default="whitening_geometry_analysis")
    parser.add_argument("--layer-subset", choices=["all", "late"], default="all",
                         help="which heatmap layer-subset to plot (both are always in the NPZ)")
    args = parser.parse_args()

    curve_path = RESULTS_DIR / f"{args.analysis_stem}_layer_curve.json"
    heatmap_path = RESULTS_DIR / f"{args.analysis_stem}_heatmap.npz"
    if not curve_path.exists() or not heatmap_path.exists():
        raise FileNotFoundError(
            f"{curve_path} / {heatmap_path} missing -- run phase1_whitening_geometry_analysis.py first"
        )

    curve_data = json.loads(curve_path.read_text())
    plot_layer_curve(curve_data["layers"], curve_data["metric"], "figureG1_whitening_layer_curve_raw")

    npz = np.load(heatmap_path, allow_pickle=True)
    conditions = [str(c) for c in npz["conditions"]]
    metric = str(npz["metric"])
    matrix = npz["all_layers"] if args.layer_subset == "all" else npz["late_layers"]
    layer_note = "all MoE layers" if args.layer_subset == "all" else f"late layers ({min(LATE_LAYERS)}-{max(LATE_LAYERS)})"
    heatmap_stem = "figureG2_whitening_heatmap" if args.layer_subset == "all" else f"figureG2_whitening_heatmap_{args.layer_subset}"
    plot_heatmap(conditions, matrix, metric, layer_note, heatmap_stem)


if __name__ == "__main__":
    main()
