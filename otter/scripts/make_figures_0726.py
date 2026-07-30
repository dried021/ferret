"""Three publication-style PNG figures, one per proven/placebo-verified
result in the project (2026-07-26, user asked for a curated PNG set instead
of the HTML artifact gallery). Saved to otter/0726_results/.

English labels throughout (matching this project's existing matplotlib
figures, e.g. make_figure_phase1.py's "EN-only"/"KO-only") -- no Korean-
capable font is confirmed installed, and DejaVu Sans (matplotlib's default)
drops Hangul glyphs silently otherwise.

Data sources (real, already-computed results -- no new computation):
  1. results/layer_locality_gate_summary.json (layer locality, Qwen3, 3 seeds)
  2. /mnt/HDD/minjeong/d2moe_results/fisher_pilot_a/pilot_a_analysis.json (Fisher Pilot A, DeepSeek)
  3. /mnt/HDD/minjeong/d2moe_results/phase1/phase1_placebo_gate_result.json +
     phase1_swahili_gate_result.json (Phase 1 own-language placebo, DeepSeek)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/minjeong/project/FERRET/otter")
OUT_DIR = ROOT / "0726_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
ACCENT = "#b06a12"
ACCENT_SOFT = "#f3e2c8"
GOOD = "#1f7a4d"
GOOD_SOFT = "#cdead9"
EN, KO, ZH, SW = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
MUTED_BAR = "#c7cdc9"


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig1_layer_locality():
    data = json.loads((ROOT / "results" / "layer_locality_gate_summary.json").read_text())
    layers = data["layers"]
    ratios = [data["mean_ratio_by_layer"][str(l)] for l in layers]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    ax.axvspan(22, 47.6, color=ACCENT_SOFT, alpha=0.5, zorder=1, label="back ~20% of network (layers 22–47)")
    ax.axvspan(38, 47.6, color=GOOD_SOFT, alpha=0.8, zorder=2, label="exceeds gate threshold (38, 42, 45, 47)")
    ax.plot(layers, ratios, color=ZH, linewidth=2, zorder=3, marker="o", markersize=5,
            markerfacecolor=ZH, markeredgewidth=0)
    exceed_layers = [38, 42, 45, 47]
    exceed_ratios = [data["mean_ratio_by_layer"][str(l)] for l in exceed_layers]
    ax.scatter(exceed_layers, exceed_ratios, color=GOOD, s=70, zorder=4, edgecolor="white", linewidth=1.2)

    ax.set_xlim(0, 48)
    ax.set_ylim(0, 16)
    ax.set_xlabel("layer index (of 48)", fontsize=10, color=INK)
    ax.set_ylabel("routing / Fisher-proxy divergence ratio", fontsize=10, color=INK)
    ax.set_title("Calibration-language sensitivity concentrates in the back ~20% of the network\n"
                 "(Qwen3-30B-A3B, 3 seeds — back-half Spearman ρ=0.85, p=1.1e-6)",
                 fontsize=11.5, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "1_layer_locality.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


def fig2_fisher_pilot_a():
    rows = json.loads(Path("/mnt/HDD/minjeong/d2moe_results/fisher_pilot_a/pilot_a_analysis.json").read_text())
    roles = [r["role"] for r in rows]
    layer_labels = [f'{r["role"]}\n(L{r["layer"]})' for r in rows]
    avg_rho = [np.mean([r["proxy_vs_real"][c]["rho"] for c in ("english_a", "english_b", "korean", "chinese")])
               for r in rows]
    real_gap = [r["real_gap"]["gap"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=160)

    ax = axes[0]
    bar_colors = ["#1a8f82"] * len(avg_rho)
    ax.bar(layer_labels, avg_rho, color=bar_colors, width=0.55, zorder=3)
    for i, v in enumerate(avg_rho):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spearman ρ (proxy vs real Fisher)", fontsize=10, color=INK)
    ax.set_title("(a) proxy ↔ real Fisher correlation (avg. across conditions)", fontsize=10.5, color=INK, loc="left")
    style_axis(ax)

    ax2 = axes[1]
    bar_colors2 = ["#4a4a4a" if r in ("early", "transition") else GOOD for r in roles]
    ax2.bar(layer_labels, real_gap, color=bar_colors2, width=0.55, zorder=3)
    for i, v in enumerate(real_gap):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax2.set_ylim(0, 1.4)
    ax2.set_ylabel("within-EN vs EN-non-EN rank-corr. gap", fontsize=10, color=INK)
    ax2.set_title("(b) gap grows toward later layers", fontsize=10.5, color=INK, loc="left")
    style_axis(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.suptitle("Forward-only proxy is valid, and back-block sensitivity reproduces on a different model\n"
                 "(DeepSeek-MoE-16B, real gradient Fisher — Fisher Pilot A)",
                 fontsize=11.5, color=INK, x=0.02, ha="left", y=0.97)
    out = OUT_DIR / "2_fisher_pilot_a_validation.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


def fig3_phase1_placebo():
    placebo = json.loads(Path("/mnt/HDD/minjeong/d2moe_results/phase1/phase1_placebo_gate_result.json").read_text())
    swahili = json.loads(Path("/mnt/HDD/minjeong/d2moe_results/phase1/phase1_swahili_gate_result.json").read_text())
    baseline = json.loads(Path("/mnt/HDD/minjeong/d2moe_results/phase1/baseline/eval_ppl.json").read_text())

    langs = ["Korean", "Swahili", "English"]
    colors = [KO, SW, MUTED_BAR]
    gains = [placebo["kor_Hang"]["own_gain"], swahili["mean_own_gain"], placebo["eng_Latn"]["own_gain"]]
    floors = [placebo["kor_Hang"]["noise_floor"], swahili["noise_floor"], placebo["eng_Latn"]["noise_floor"]]
    thresholds = [f * 2 for f in floors]
    verdicts = ["SUPPORTED (5.6×)", "SUPPORTED (14.6×)", "below threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=160)

    ax = axes[0]
    y = np.arange(len(langs))
    ax.barh(y, gains, color=colors, height=0.55, zorder=3)
    for i, (g, t, v) in enumerate(zip(gains, thresholds, verdicts)):
        ax.plot([t, t], [i - 0.3, i + 0.3], color=INK, linewidth=1.5, zorder=4, alpha=0.6)
        ax.text(g + 0.15, i, f"{g:.3f}pp  {v}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(langs, fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("own-language gain (pp)  |  vertical tick = 2× noise-floor threshold", fontsize=9.5, color=INK)
    ax.set_title("(a) own-language gain vs placebo noise floor", fontsize=10.5, color=INK, loc="left")
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    ax2 = axes[1]
    vuln = [baseline["eng_Latn"]["bits_per_byte"], baseline["kor_Hang"]["bits_per_byte"], baseline["swh_Latn"]["bits_per_byte"]]
    gain_order = [placebo["eng_Latn"]["own_gain"], placebo["kor_Hang"]["own_gain"], swahili["mean_own_gain"]]
    labels2 = ["English", "Korean", "Swahili"]
    colors2 = [EN, KO, SW]
    ax2.scatter(vuln, gain_order, c=colors2, s=140, zorder=4, edgecolor="white", linewidth=1.2)
    for x, yv, lab in zip(vuln, gain_order, labels2):
        ax2.annotate(lab, (x, yv), textcoords="offset points", xytext=(8, 6), fontsize=10, color=INK, fontweight="bold")
    ax2.set_xlabel("pre-compression baseline bpb (vulnerability)", fontsize=10, color=INK)
    ax2.set_ylabel("own-language gain (pp)", fontsize=10, color=INK)
    ax2.set_title("(b) more vulnerable → bigger gain", fontsize=10.5, color=INK, loc="left")
    ax2.set_xlim(0.7, 2.5)
    ax2.set_ylim(-0.5, 9)
    style_axis(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.suptitle("Own-language calibration effect is real — and scales with language vulnerability\n"
                 "(confirmed on two independent languages: Korean, Swahili)",
                 fontsize=11.5, color=INK, x=0.02, ha="left", y=0.97)
    out = OUT_DIR / "3_phase1_placebo_verified.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


def fig4_2x2_whitening():
    """PRELIMINARY -- seed=1, no placebo yet (see 00_docs/03_기술노트.md '2)
    whitening 복원'). Included as a 4th figure per user request, but visually
    marked as unverified (dashed border, muted title) unlike figures 1-3."""
    baseline = json.loads(Path("/mnt/HDD/minjeong/d2moe_results/phase1/baseline/eval_ppl.json").read_text())
    langs = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn"]
    lang_labels = ["EN", "KO", "ZH", "SW"]
    cells = {
        ("EN", "EN"): "english_only/seed0/scale_english_only_seed0",
        ("EN", "KO"): "english_only/seed0/scale_korean_only_seed0",
        ("KO", "EN"): "korean_only/seed0/scale_english_only_seed0",
        ("KO", "KO"): "korean_only/seed0/scale_korean_only_seed0",
    }
    cell_incr = {}
    for key, path in cells.items():
        d = json.loads(Path(f"/mnt/HDD/minjeong/d2moe_results/phase1/{path}/eval_ppl.json").read_text())
        cell_incr[key] = [100 * (d[l]["bits_per_byte"] / baseline[l]["bits_per_byte"] - 1) for l in langs]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)

    # (a) Fisher x Scale heatmap of KO increase %
    ax = axes[0]
    grid = np.array([[cell_incr[("EN", "EN")][1], cell_incr[("EN", "KO")][1]],
                      [cell_incr[("KO", "EN")][1], cell_incr[("KO", "KO")][1]]])
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=95, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Scale=EN", "Scale=KO"], fontsize=10.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Fisher=EN", "Fisher=KO"], fontsize=10.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:.1f}%", ha="center", va="center", fontsize=13,
                    color="white" if grid[i, j] > 45 else INK, fontweight="bold")
    ax.set_title("(a) KO bpb increase % by Fisher × Scale language", fontsize=10.5, color=INK, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # (b) all 4 languages across the 4 cells -- sanity check (only KO should swing)
    ax2 = axes[1]
    x = np.arange(len(cells))
    w = 0.2
    colors = [EN, KO, ZH, SW]
    cell_order = [("EN", "EN"), ("EN", "KO"), ("KO", "EN"), ("KO", "KO")]
    for li, (lang, color) in enumerate(zip(lang_labels, colors)):
        vals = [cell_incr[c][li] for c in cell_order]
        ax2.bar(x + (li - 1.5) * w, vals, width=w, color=color, label=lang, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"F={f}\nS={s}" for f, s in cell_order], fontsize=9)
    ax2.set_ylabel("bpb increase vs baseline (%)", fontsize=10, color=INK)
    ax2.set_title("(b) only KO swings — EN/ZH/SW stay stable (sanity check)", fontsize=10.5, color=INK, loc="left")
    style_axis(ax2)
    ax2.legend(frameon=False, fontsize=9, loc="upper left", ncol=4)

    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_linestyle((0, (4, 3)))
            spine.set_edgecolor(MUTED)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.suptitle("PRELIMINARY — whitening (SVD scale) language may dominate over Fisher language for KO\n"
                 "seed=1, no placebo yet — needs the same verification as Figures 1–3 before it's a confirmed result",
                 fontsize=11, color=ACCENT, x=0.02, ha="left", y=0.97)
    out = OUT_DIR / "4_2x2_whitening_preliminary.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_layer_locality()
    fig2_fisher_pilot_a()
    fig3_phase1_placebo()
    fig4_2x2_whitening()
