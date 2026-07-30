"""Fisher Pilot A figure: proxy-vs-real Spearman rho, and the within-EN vs
EN-vs-non-EN gap for real Fisher, both plotted by layer role.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/fisher_pilot_a")
FIGURES_DIR = Path("/home/minjeong/project/FERRET/otter/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
COND_COLORS = {"english_a": "#0072B2", "english_b": "#56B4E9", "korean": "#D55E00", "chinese": "#009E73"}


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    rows = json.loads((RESULTS_DIR / "pilot_a_analysis.json").read_text())
    layers = [r["layer"] for r in rows]
    roles = [r["role"] for r in rows]
    labels = [f"L{l}\n({r})" for l, r in zip(layers, roles)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)

    ax = axes[0]
    x = np.arange(len(layers))
    conds = ["english_a", "english_b", "korean", "chinese"]
    w = 0.2
    for i, cond in enumerate(conds):
        vals = [r["proxy_vs_real"][cond]["rho"] for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=COND_COLORS[cond], label=cond, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman rho (proxy vs real gradient Fisher)")
    ax.set_title("Q1: does proxy track real Fisher?", fontsize=10.5, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[1]
    real_gap = [r["real_gap"]["gap"] for r in rows]
    proxy_gap = [r["proxy_gap"]["gap"] for r in rows]
    ax.plot(x, real_gap, color="#0072B2", marker="o", markersize=7, linewidth=2, label="real gradient Fisher")
    ax.plot(x, proxy_gap, color="#D55E00", marker="s", markersize=7, linewidth=2, label="forward-only proxy")
    ax.axhline(0, color=MUTED, linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("gap = within-EN rho - mean(EN-KO, EN-ZH rho)")
    ax.set_title("Q2: EN-vs-non-EN gap by layer depth (DeepSeek-MoE-16B)", fontsize=10.5, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    out = FIGURES_DIR / "fisher_pilot_a_summary.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
