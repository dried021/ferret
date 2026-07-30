"""Figure 5.3-distribution: between-language vs within-language distribution
comparison for the two §5.3 headline metrics (mean principal angle, k=64;
covariance relative Frobenius distance, k=64), as violin plots with the
n_within/n_between counts and permutation-test significance annotated.
max_angle_deg_k64 intentionally excluded (saturates near 90 deg for both
within and between -- see the pre-lock review, not a usable discriminator).

Source: whitening_geometry_pairs.csv (raw, for the distributions) +
whitening_geometry_analysis.json (for n / p-value annotation, so the
annotated numbers are exactly what's already been locked, not recomputed).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
WITHIN_COLOR = "#0072B2"
BETWEEN_COLOR = "#D55E00"

METRICS = [
    ("mean_angle_deg_k64", "mean principal angle (deg, k=64)"),
    ("cov_reldist_k64", "covariance rel. Frobenius distance (k=64)"),
]


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(RESULTS_DIR / "whitening_geometry_pairs.csv")
    df["within"] = df["cond_a"] == df["cond_b"]
    analysis = json.loads((RESULTS_DIR / "whitening_geometry_analysis.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), dpi=160)
    for ax, (metric, label) in zip(axes, METRICS):
        w = df.loc[df["within"], metric].to_numpy()
        b = df.loc[~df["within"], metric].to_numpy()
        parts = ax.violinplot([w, b], positions=[0, 1], showmeans=True, showextrema=True, widths=0.75)
        for pc, color in zip(parts["bodies"], (WITHIN_COLOR, BETWEEN_COLOR)):
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(MUTED)
            parts[key].set_linewidth(1.0)

        m = analysis["metrics"][metric]["all_layers"]
        n_w, n_b = m["n_within"], m["n_between"]
        p = m["permutation"]["p_value_between_ge_within"]
        p_str = "p < 0.0001" if p == 0.0 else f"p = {p:.4f}"

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"within\n(n={n_w})", f"between\n(n={n_b})"])
        ax.set_ylabel(label)
        ax.set_title(f"{label}\nratio={m['ratio_between_over_within']:.2f}, {p_str} (permutation, one-sided)",
                     fontsize=9.5, color=INK)
        style_axis(ax)

    fig.tight_layout()

    out_path = FIGURES_DIR / "figure4_whitening_distributions.png"
    fig.savefig(out_path, facecolor="white")
    print(f"[figure4] wrote {out_path}")


if __name__ == "__main__":
    main()
