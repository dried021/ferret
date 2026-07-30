"""Figure D (00_docs/08_figure_정리.md): §5.1 "DeepSeek 실측 언어 간 rank 상관
vs placebo 상관 per-layer 곡선" -- replaces the Qwen3-30B-A3B diagnostic-model
placeholder (existing Figure 1) with the real cross-language Fisher-rank
correlation this script's companion, phase1_51_fisher_rank_correlation.py,
actually measures on DeepSeek-MoE-16B. Mean cross-language Spearman rho (solid)
vs mean same-language placebo rho (dashed, the noise-floor ceiling) per MoE
layer, with the individual language-pair rho spread shaded.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
FIGURES_DIR = Path("/home/minjeong/project/FERRET/otter/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
LANG_COLOR = "#0072B2"
PLACEBO_COLOR = "#D55E00"


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main(seed=0):
    result = json.loads((RESULTS_DIR / f"phase1_51_fisher_rank_correlation_result_seed{seed}.json").read_text())
    per_layer = result["per_layer"]
    layers = sorted((int(l) for l in per_layer), key=int)

    mean_lang = np.array([per_layer[str(l)]["mean_lang_rho"] for l in layers])
    mean_placebo = np.array([per_layer[str(l)]["mean_placebo_rho"] for l in layers])
    lang_lo = np.array([min(v["rho"] for v in per_layer[str(l)]["lang_pairs"].values()) for l in layers])
    lang_hi = np.array([max(v["rho"] for v in per_layer[str(l)]["lang_pairs"].values()) for l in layers])

    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=160)
    x = np.arange(len(layers))
    ax.fill_between(x, lang_lo, lang_hi, color=LANG_COLOR, alpha=0.15, zorder=1, label="cross-language rho range")
    ax.plot(x, mean_lang, color=LANG_COLOR, marker="o", markersize=4.5, linewidth=2, zorder=3,
            label="mean cross-language rho (10 lang pairs)")
    ax.plot(x, mean_placebo, color=PLACEBO_COLOR, marker="s", markersize=4.5, linewidth=2, linestyle="--", zorder=3,
            label="mean placebo rho (noise floor)")

    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), 2)], fontsize=7.5)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Spearman rho (per-expert Fisher importance)")
    ax.set_ylim(min(0, lang_lo.min()) - 0.05, 1.02)
    ax.set_title("§5.1 real-Fisher expert-rank correlation: cross-language vs placebo, DeepSeek-MoE-16B",
                 fontsize=10.5, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.tight_layout()
    out_path = FIGURES_DIR / "figure_d_fisher_rank_correlation.png"
    fig.savefig(out_path)
    print(f"[figure-d] wrote {out_path}")


if __name__ == "__main__":
    main()
