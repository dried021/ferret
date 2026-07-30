"""Figure A (paper §4.3) -- path decomposition of D2-MoE's Korean-protective
effect: Fisher-merge vs whitening vs pruning, own-language (KO) effect size,
each against the noise floor from the whitening placebo test.

English labels throughout -- no Korean-capable font is confirmed installed
on this host, and DejaVu Sans (matplotlib default) drops Hangul glyphs
silently otherwise (see scripts/make_figures_0726.py).

Data sources (real, already-computed gate results -- no new computation):
  - Fisher-merge alone:  /mnt/HDD/minjeong/d2moe_results/phase1/phase1_seed_gate_result.json
      per_language.kor_Hang.mean_gain   (Fisher-merge only, no whitening/pruning; §3.14/1.6)
  - Whitening (scale):   /mnt/HDD/minjeong/d2moe_results/phase1/phase1_2x2_verify_gate_result.json
      mean_own_scale_gain (Fisher=KO fixed, Scale EN vs KO), noise_floor_scale (3-seed max placebo |Delta|)
  - Pruning:             /mnt/HDD/minjeong/d2moe_results/phase1/phase1_pruning_gate_korean_only_result.json
      per_language.kor_Hang.mean_delta_pp (pruning on-off, Fisher/Scale calibration = KO, pp_ratio=0.2)

Note on the noise-floor line: noise_floor_scale is specific to the whitening
placebo test. Fisher-merge and pruning were checked against their own,
separate (and much smaller) floors -- see 00_docs/08_figure_정리.md "App:
placebo 설계 표". The dashed line here contextualizes the whitening bar; it
is not a shared threshold for the other two paths, and the figure says so
in a footnote so it can't be misread as one.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
FIGURES_DIR = Path("/home/minjeong/project/FERRET/otter/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
NEUTRAL_BAR = "#9aa0a6"
ACCENT_BAR = "#0072B2"
FLOOR_LINE = "#D55E00"


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    seed_gate = json.loads((RESULTS_DIR / "phase1_seed_gate_result.json").read_text())
    verify_gate = json.loads((RESULTS_DIR / "phase1_2x2_verify_gate_result.json").read_text())
    pruning_gate = json.loads((RESULTS_DIR / "phase1_pruning_gate_korean_only_result.json").read_text())

    fisher_gain = seed_gate["per_language"]["kor_Hang"]["mean_gain"]
    whitening_gain = verify_gate["mean_own_scale_gain"]
    pruning_gain = pruning_gate["per_language"]["kor_Hang"]["mean_delta_pp"]
    noise_floor = verify_gate["noise_floor_scale"]

    labels = ["Fisher merge\n(own_gain)", "Whitening\n(own_scale_gain)", "Pruning\n(mean Δpp)"]
    values = [fisher_gain, whitening_gain, pruning_gain]
    colors = [NEUTRAL_BAR, ACCENT_BAR, NEUTRAL_BAR]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    x = range(len(labels))
    bars = ax.bar(x, values, width=0.55, color=colors, zorder=3)

    label_bbox = dict(facecolor="white", edgecolor="none", pad=1.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.8, f"{v:.2f}%p", ha="center", fontsize=10, color=INK, fontweight="bold",
                 zorder=6, bbox=label_bbox)

    ax.axhline(noise_floor, color=FLOOR_LINE, linewidth=1.6, linestyle="--", zorder=4)
    ax.text(len(labels) - 0.42, noise_floor + 0.8, f"noise floor = {noise_floor:.2f}%p",
            ha="right", fontsize=9, color=FLOOR_LINE, zorder=6, bbox=label_bbox)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("KO own-language effect size (bpb, percentage points)", fontsize=10.5, color=INK)
    ax.set_title("Path decomposition of D2-MoE's Korean-protective effect (§4.3)\n"
                 "whitening accounts for ~7× the Fisher-merge effect alone",
                 fontsize=11.5, color=INK, loc="left")
    style_axis(ax)

    fig.text(0.01, 0.005,
              "Dashed line = noise floor from the whitening placebo test only (max |Δ| across 3 seeds, EN/KO_b).\n"
              "Fisher-merge and pruning were verified against their own, separate (smaller) floors, not this one.",
              fontsize=7.2, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIGURES_DIR / "figure_a_path_decomposition.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    print(f"Fisher={fisher_gain:.3f}%p  Whitening={whitening_gain:.3f}%p  "
          f"Pruning={pruning_gain:.3f}%p  noise_floor={noise_floor:.3f}%p")


if __name__ == "__main__":
    main()
