"""Phase 1 seed-replicated figure: bits-per-byte relative increase (compressed
vs baseline) per language, per calibration condition, mean +/- range across
3 seeds (2026-07-24 rerun -- see 00_docs/03_기술노트.md "1) 예산 정상화").

Uses bits-per-byte (not per-token PPL) throughout -- per-token PPL is not
comparable across languages here because DeepSeek's tokenizer segments
Korean far more finely than English/Chinese (byte-fallback on Hangul),
which mechanically deflates Korean's per-token PPL regardless of true model
quality (see 00_docs/03_기술노트.md "0) bits-per-byte 재계산").
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
FIGURES_DIR = Path("/home/minjeong/project/FERRET/otter/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
CONDS = ["english_only", "korean_only", "chinese_only", "balanced"]
COND_LABELS = {"english_only": "EN-only", "korean_only": "KO-only", "chinese_only": "ZH-only", "balanced": "Balanced"}
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans"]
LANG_LABELS = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH"}
LANG_COLORS = {"eng_Latn": "#0072B2", "kor_Hang": "#D55E00", "zho_Hans": "#009E73"}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_bpb(condition, seed, lang):
    path = ROOT / condition / f"seed{seed}" / "eval_ppl.json"
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def main():
    baseline = json.loads((ROOT / "baseline" / "eval_ppl.json").read_text())

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    x = np.arange(len(CONDS))
    w = 0.25
    for i, lang in enumerate(LANGS):
        means, los, his = [], [], []
        for c in CONDS:
            incr = [100 * (load_bpb(c, s, lang) / baseline[lang]["bits_per_byte"] - 1) for s in SEEDS]
            means.append(np.mean(incr))
            los.append(np.mean(incr) - np.min(incr))
            his.append(np.max(incr) - np.mean(incr))
        xpos = x + (i - 1) * w
        ax.bar(xpos, means, width=w, color=LANG_COLORS[lang], label=LANG_LABELS[lang], zorder=3)
        ax.errorbar(xpos, means, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDS])
    ax.set_ylabel("bits-per-byte relative increase vs baseline (%)")
    ax.set_title("Phase 1: DeepSeek-MoE-16B compression retention by calibration language\n"
                  "(Fisher-weighted merge + plain SVD delta, ratio=0.8, no pp_ratio; "
                  "64 samples/seqlen 512, 3 seeds, bars = min-max)",
                  fontsize=10.5, color=INK, loc="left")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="eval language")
    fig.tight_layout()
    out = FIGURES_DIR / "phase1_pilot_summary.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
