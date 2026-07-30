"""Figure E (00_docs/08_figure_정리.md): §5.2 "36개 deficient 쌍의 레이어별
히스토그램(언어 색 구분)" -- per-layer count of (layer, expert) disagreement
pairs deficient (hit_count under an equal 5-way calibration-budget split) for
each language, from phase1_52_deficiency_layer_distribution.py's output.
English/Korean/Chinese are always 0 (see that script's result json) so only
Swahili/Bengali get bars.
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
LANG_COLORS = {"swh_Latn": "#D55E00", "ben_Beng": "#0072B2"}
LANG_LABELS = {"swh_Latn": "Swahili", "ben_Beng": "Bengali"}


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    result = json.loads((RESULTS_DIR / "phase1_52_deficiency_layer_distribution_result.json").read_text())
    per_layer = result["per_layer"]
    layers = sorted((int(l) for l in per_layer), key=int)
    back_start = int(result["back_half_layers"][0])

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    x = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    for lang in ("swh_Latn", "ben_Beng"):
        vals = np.array([per_layer[str(l)]["deficient_by_lang"].get(lang, 0) for l in layers])
        ax.bar(x, vals, bottom=bottom, color=LANG_COLORS[lang], label=LANG_LABELS[lang], zorder=3,
               edgecolor="white", linewidth=0.4)
        bottom += vals

    # shade the back-half band the paper's layer-locality claim (§3 Figure 1,
    # §5.1) predicts deficiency should concentrate in
    back_idx = layers.index(back_start)
    ax.axvspan(back_idx - 0.5, len(layers) - 0.5, color="#f0f0f0", zorder=0)
    ax.text(back_idx + (len(layers) - back_idx) / 2 - 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 8,
            "back half", ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7.5)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("# deficient (layer, expert) pairs")
    total = result["total_any_deficient"]
    n_pairs = result["total_pairs"]
    ax.set_title(f"§5.2 deficiency by layer ({total}/{n_pairs} disagreement pairs deficient for >=1 language)",
                 fontsize=10.5, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    out_path = FIGURES_DIR / "figure_e_deficiency_layer_histogram.png"
    fig.savefig(out_path)
    print(f"[figure-e] wrote {out_path}")


if __name__ == "__main__":
    main()
