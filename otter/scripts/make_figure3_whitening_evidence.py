"""Figure 5 (combined): §5.1 + §5.2 + §5.3 stacked on a shared MoE-layer
x-axis (1-27), so the "mid-layer gap/trough" alignment across all three
mechanisms is visible in one figure. Pure post-processing of already-computed
result JSONs / the whitening-geometry pairs CSV -- no GPU, no recompute of
the underlying statistics.

Panel A (§5.1): mean cross-language Spearman rho (10 language pairs, shaded
  min-max range) vs mean placebo rho (3 pairs, noise floor) per layer.
  Source: phase1_51_fisher_rank_correlation_result_seed0.json
Panel B (§5.2): stacked bar of deficient (layer, expert) pair counts per
  layer, colored by language (Swahili/Bengali -- English/Korean/Chinese are
  always 0). Source: phase1_52b_deficiency_pairs_list_result.json (raw
  flagged list, aggregated here by layer x lang -- NOT the same file
  make_figure_deficiency_layer_histogram.py uses, per this figure's brief).
Panel C (§5.3): between/within ratio per layer for mean_angle_deg_k64 (left
  axis) and cov_reldist_k64 (right axis, different scale), ratio=1 dashed
  reference line. Source: whitening_geometry_pairs.csv (raw pairs, ratio
  computed here per layer -- the saved layer_curve.json only has
  mean_angle_deg_k64, not cov_reldist_k64).

All three panels share x-ticks/x-range and a shared light shading over
layers 10-18 (the mid-layer trough region flagged in the pre-lock review).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
OTTER_RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
LANG_COLOR = "#0072B2"
PLACEBO_COLOR = "#D55E00"
LANG_COLORS_52 = {"swh_Latn": "#D55E00", "ben_Beng": "#0072B2"}
LANG_LABELS_52 = {"swh_Latn": "Swahili", "ben_Beng": "Bengali"}
ANGLE_COLOR = "#0072B2"
COV_COLOR = "#D55E00"
MID_SHADE = "#f0f0f0"
MID_LAYERS = (10, 18)  # inclusive, shared shading band across all 3 panels

LAYERS = list(range(1, 28))


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def shade_mid(ax, x):
    lo, hi = MID_LAYERS
    ax.axvspan(x[LAYERS.index(lo)] - 0.5, x[LAYERS.index(hi)] + 0.5, color=MID_SHADE, zorder=0)


def load_panel_a(seed=0):
    result = json.loads((RESULTS_DIR / f"phase1_51_fisher_rank_correlation_result_seed{seed}.json").read_text())
    per_layer = result["per_layer"]
    mean_lang = np.array([per_layer[str(l)]["mean_lang_rho"] for l in LAYERS])
    mean_placebo = np.array([per_layer[str(l)]["mean_placebo_rho"] for l in LAYERS])
    lang_lo = np.array([min(v["rho"] for v in per_layer[str(l)]["lang_pairs"].values()) for l in LAYERS])
    lang_hi = np.array([max(v["rho"] for v in per_layer[str(l)]["lang_pairs"].values()) for l in LAYERS])
    return mean_lang, mean_placebo, lang_lo, lang_hi


def load_panel_b():
    result = json.loads((RESULTS_DIR / "phase1_52b_deficiency_pairs_list_result.json").read_text())
    counts = {lang: defaultdict(int) for lang in LANG_COLORS_52}
    for row in result["flagged"]:
        lang = row["lang"]
        if lang in counts:
            counts[lang][row["layer"]] += 1
    return counts, result["total_flagged"], result["total_pairs_scanned"]


def load_panel_c():
    df = pd.read_csv(OTTER_RESULTS / "whitening_geometry_pairs.csv")
    df["within"] = df["cond_a"] == df["cond_b"]
    angle_ratio, cov_ratio = [], []
    for l in LAYERS:
        sub = df[df["layer"] == l]
        w_a = sub.loc[sub["within"], "mean_angle_deg_k64"].mean()
        b_a = sub.loc[~sub["within"], "mean_angle_deg_k64"].mean()
        w_c = sub.loc[sub["within"], "cov_reldist_k64"].mean()
        b_c = sub.loc[~sub["within"], "cov_reldist_k64"].mean()
        angle_ratio.append(b_a / w_a)
        cov_ratio.append(b_c / w_c)
    return np.array(angle_ratio), np.array(cov_ratio)


def main(seed_51=0):
    x = np.arange(len(LAYERS))

    mean_lang, mean_placebo, lang_lo, lang_hi = load_panel_a(seed_51)
    counts_b, total_flagged, total_pairs = load_panel_b()
    angle_ratio, cov_ratio = load_panel_c()

    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(10, 10.5), dpi=160, sharex=True)

    # --- Panel A ---
    shade_mid(axA, x)
    axA.fill_between(x, lang_lo, lang_hi, color=LANG_COLOR, alpha=0.15, zorder=1, label="cross-language rho range")
    axA.plot(x, mean_lang, color=LANG_COLOR, marker="o", markersize=4, linewidth=2, zorder=3,
              label="mean cross-language rho (10 pairs)")
    axA.plot(x, mean_placebo, color=PLACEBO_COLOR, marker="s", markersize=4, linewidth=2, linestyle="--", zorder=3,
              label="mean placebo rho (noise floor, 3 pairs)")
    axA.set_ylabel("Spearman rho")
    axA.set_ylim(min(0, lang_lo.min()) - 0.05, 1.02)
    style_axis(axA)
    axA.legend(frameon=False, fontsize=7.5, loc="lower left")

    # --- Panel B ---
    shade_mid(axB, x)
    bottom = np.zeros(len(LAYERS))
    for lang in ("swh_Latn", "ben_Beng"):
        vals = np.array([counts_b[lang].get(l, 0) for l in LAYERS])
        axB.bar(x, vals, bottom=bottom, color=LANG_COLORS_52[lang], label=LANG_LABELS_52[lang], zorder=3,
                edgecolor="white", linewidth=0.4)
        bottom += vals
    axB.set_ylabel("# deficient\n(layer,expert) pairs")
    style_axis(axB)
    axB.legend(frameon=False, fontsize=8, loc="upper left")

    # --- Panel C ---
    shade_mid(axC, x)
    axC.axhline(1.0, color=MUTED, linewidth=1, linestyle=":", zorder=2)
    l1, = axC.plot(x, angle_ratio, color=ANGLE_COLOR, marker="o", markersize=4, linewidth=2, zorder=3,
                     label="mean principal angle ratio (k=64)")
    axC2 = axC.twinx()
    l2, = axC2.plot(x, cov_ratio, color=COV_COLOR, marker="^", markersize=4, linewidth=2, zorder=3,
                      label="covariance Frobenius ratio (k=64)")
    axC.set_ylabel("angle ratio\n(between/within)", color=ANGLE_COLOR)
    axC2.set_ylabel("cov. Frobenius ratio\n(between/within)", color=COV_COLOR)
    axC.tick_params(axis="y", colors=ANGLE_COLOR)
    axC2.tick_params(axis="y", colors=COV_COLOR)
    axC2.spines["top"].set_visible(False)
    style_axis(axC)
    axC.legend(handles=[l1, l2], frameon=False, fontsize=8, loc="upper left")

    axC.set_xticks(x[::2])
    axC.set_xticklabels([str(LAYERS[i]) for i in range(0, len(LAYERS), 2)], fontsize=8)
    axC.set_xlabel("MoE layer")
    axC.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    fig.tight_layout()

    out_path = FIGURES_DIR / "figure3_whitening_layer_evidence.png"
    fig.savefig(out_path, facecolor="white")
    print(f"[figure3] wrote {out_path}")


if __name__ == "__main__":
    main()
