"""Toy0 figures -- renders the gate decision and the underlying routing/
Fisher-proxy statistics from 01/02's outputs into figures/*.png.

Usage:
    conda run -n torch_env python 03_make_figures.py [--smoke]
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

# Fixed categorical order + Okabe-Ito colorblind-safe hues, assigned by
# identity (not cycled) -- en_only/en_only_control share a blue family on
# purpose since one is the noise-floor control for the other.
CONDITION_ORDER = ["en_only", "en_only_control", "ko_only", "zh_only", "balanced"]
CONDITION_COLOR = {
    "en_only": "#0072B2",
    "en_only_control": "#56B4E9",
    "ko_only": "#D55E00",
    "zh_only": "#009E73",
    "balanced": "#CC79A7",
}
CONDITION_LABEL = {
    "en_only": "EN-only",
    "en_only_control": "EN-only (control)",
    "ko_only": "KO-only",
    "zh_only": "ZH-only",
    "balanced": "Balanced EN+KO+ZH",
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"


def load_config():
    spec = importlib.util.spec_from_file_location("d2moe_ml_config", SCRIPT_DIR / "00_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_path(p, smoke):
    return p if not smoke else p.with_name(p.stem + "_smoke" + p.suffix)


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_gate_bars(gate_summary, out_path):
    layers = sorted(gate_summary["per_layer"].keys(), key=int)
    noise = [gate_summary["per_layer"][l]["noise_floor_js"] for l in layers]
    cross = [gate_summary["per_layer"][l]["cross_language_mean_js"] for l in layers]
    passed = [gate_summary["per_layer"][l]["pass"] for l in layers]
    margin = gate_summary["gate_margin"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    x = np.arange(len(layers))
    w = 0.32

    ax.bar(x - w / 2, noise, width=w, color=MUTED, label="Same-language noise floor\n(en_only vs en_only_control)", zorder=3)
    bar_colors = ["#009E73" if p else "#D55E00" for p in passed]
    ax.bar(x + w / 2, cross, width=w, color=bar_colors, label="Cross-language mean JS divergence", zorder=3)

    for i, (n, c, p) in enumerate(zip(noise, cross, passed)):
        ax.plot([i - w / 2 - 0.18, i + w / 2 + 0.18], [n * margin, n * margin],
                color=INK, linewidth=1.2, linestyle="--", zorder=4)
        ax.text(i, max(n, c) + 0.02, "PASS" if p else "FAIL",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#009E73" if p else "#D55E00")

    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("Jensen-Shannon divergence (routing distribution)")
    ax.set_title("Toy0 gate: does calibration language beat the same-language noise floor?",
                 fontsize=11, color=INK, loc="left", pad=14)
    ax.text(0, -0.22, "dashed line = 1.5x noise floor (gate threshold)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03] wrote {out_path}")


def fig_divergence_heatmaps(div_rows, out_path):
    layers = sorted({r["layer"] for r in div_rows}, key=int)
    conds = CONDITION_ORDER

    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.0), dpi=160)
    if len(layers) == 1:
        axes = [axes]

    mat_by_layer = {}
    for layer in layers:
        mat = np.full((len(conds), len(conds)), np.nan)
        for r in div_rows:
            if r["layer"] != layer:
                continue
            i, j = conds.index(r["condition_a"]), conds.index(r["condition_b"])
            mat[i, j] = mat[j, i] = float(r["js_divergence"])
        np.fill_diagonal(mat, 0.0)
        mat_by_layer[layer] = mat

    vmax = max(np.nanmax(m) for m in mat_by_layer.values())
    for k, (ax, layer) in enumerate(zip(axes, layers)):
        mat = mat_by_layer[layer]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(conds)))
        ax.set_yticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], rotation=45, ha="right", fontsize=7.5)
        # y tick labels only on the leftmost panel -- repeating them on every
        # panel collided with the next panel's own left edge (see 2026-07-23
        # review of the first render).
        ax.set_yticklabels([CONDITION_LABEL[c] for c in conds] if k == 0 else [], fontsize=7.5)
        for i in range(len(conds)):
            for j in range(len(conds)):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > vmax * 0.6 else INK)
        ax.set_title(f"layer {layer}", fontsize=10, color=INK)

    fig.suptitle("Routing-distribution JS divergence between calibration conditions", fontsize=11, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.15)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[03] wrote {out_path}")


def fig_expert_routing(stats, out_path):
    layers = sorted(stats[CONDITION_ORDER[0]]["hit_count"].keys(), key=int)
    num_experts = len(stats[CONDITION_ORDER[0]]["hit_count"][layers[0]])

    fig, axes = plt.subplots(len(layers), 1, figsize=(9, 2.1 * len(layers)), dpi=160, sharex=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        for cond in CONDITION_ORDER:
            hc = np.array(stats[cond]["hit_count"][layer], dtype=np.float64)
            dist = hc / max(hc.sum(), 1.0)
            ax.plot(np.arange(num_experts), dist, color=CONDITION_COLOR[cond],
                    linewidth=1.4, alpha=0.9, label=CONDITION_LABEL[cond], zorder=3)
        ax.set_ylabel(f"layer {layer}\nfraction of hits", fontsize=8.5)
        style_axis(ax)

    axes[-1].set_xlabel("expert index")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.5))
    fig.suptitle("Expert routing coverage by calibration condition", fontsize=11, color=INK, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[03] wrote {out_path}")


def fig_fisher_correlation(corr_rows, out_path):
    layers = sorted({r["layer"] for r in corr_rows}, key=int)
    conds = CONDITION_ORDER

    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.0), dpi=160)
    if len(layers) == 1:
        axes = [axes]

    mat_by_layer = {}
    for layer in layers:
        mat = np.full((len(conds), len(conds)), np.nan)
        for r in corr_rows:
            if r["layer"] != layer:
                continue
            i, j = conds.index(r["condition_a"]), conds.index(r["condition_b"])
            mat[i, j] = mat[j, i] = float(r["fisher_spearman_rho"])
        np.fill_diagonal(mat, 1.0)
        mat_by_layer[layer] = mat

    for k, (ax, layer) in enumerate(zip(axes, layers)):
        mat = mat_by_layer[layer]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(conds)))
        ax.set_yticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels([CONDITION_LABEL[c] for c in conds] if k == 0 else [], fontsize=7.5)
        for i in range(len(conds)):
            for j in range(len(conds)):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.6 else INK)
        ax.set_title(f"layer {layer}", fontsize=10, color=INK)

    fig.suptitle("Fisher-proxy importance rank correlation (Spearman) between conditions", fontsize=11, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.15, bottom=0.32)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.02, 0.03, "Fisher-proxy = (router_weight x ||expert output||)^2, a forward-only stand-in -- see README.md", fontsize=8, color=MUTED)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[03] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    stats = json.loads(smoke_path(cfg.ROUTING_STATS_JSON, args.smoke).read_text())
    gate_summary = json.loads(smoke_path(cfg.GATE_SUMMARY_JSON, args.smoke).read_text())
    div_rows = list(csv.DictReader(open(smoke_path(cfg.DIVERGENCE_CSV, args.smoke))))
    corr_rows = list(csv.DictReader(open(smoke_path(cfg.FISHER_CORR_CSV, args.smoke))))

    suffix = "_smoke" if args.smoke else ""
    fig_gate_bars(gate_summary, cfg.FIGURES_DIR / f"toy0_gate_bars{suffix}.png")
    fig_divergence_heatmaps(div_rows, cfg.FIGURES_DIR / f"toy0_divergence_heatmap{suffix}.png")
    fig_expert_routing(stats, cfg.FIGURES_DIR / f"toy0_expert_routing{suffix}.png")
    fig_fisher_correlation(corr_rows, cfg.FIGURES_DIR / f"toy0_fisher_correlation{suffix}.png")


if __name__ == "__main__":
    main()
