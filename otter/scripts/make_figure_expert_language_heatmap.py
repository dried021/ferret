"""Expert x language Fisher-proxy importance heatmap (paper section 5 figure):
for 4 layer roles (early/transition/sensitive/final), shows how much a given
expert's forward-only Fisher-proxy importance varies across the 6 calibration-
language conditions phase1_calib_data.py defines (english_only, chinese_only,
korean_only, swahili_only, bengali_only, balanced/mixed_5lang) -- the
"disagreement expert" signal 01_연구설계.md Section 23's Stage-1 scan is meant
to surface, so Stage 2 can target calibration-sample budget at the experts
that actually disagree across languages instead of spreading it evenly
(which is what "balanced" already does -- see Section 23.3).

Layer roles/indices are fisher_pilot_a.py's LAYERS/LAYER_ROLE for DeepSeek-
MoE-16B (28 layers: 0 dense, 1-27 MoE), reused here as constants rather than
re-derived -- NOT 02c_analyze_layer_locality.py's layer set, which probes a
different model family (Qwen3-30B-A3B, 48 layers) and has no early/
transition/sensitive/final role labels of its own; that script's 10-layer
sweep answers a different question (how far the EN-vs-non-EN gap spreads
into the back half), not "pick 4 role layers".

Input: scan_disagreement_experts's output JSON, one file, in the same per-
layer/per-condition shape fisher_pilot_a.py's pilot_a_results.json already
uses (see load_scan_results docstring for the exact schema) -- this script
only changes which layers/conditions it reads, not the format itself. If
that file does not exist yet, --smoke fabricates a small synthetic scan (seed
0) so the plotting pipeline can be exercised standalone.

Also recomputes/reuses the proxy-vs-real Spearman rho analyze_fisher_pilot_a.py
already established per layer -- printed alongside the heatmap as the
validation context for trusting this scan's proxy values at all (same
proxy formula, same model, just re-run on more languages here).

2026-07-27 review fixes (all four addressed below, see the functions they
name):
  1. Normalization axis was unstated -- it is COLUMN-wise (per-language, over
     ALL experts in the layer, not just the shown top-k, not per-row, not
     global). Now spelled out in the figure caption -- see
     plot_heatmap_figure(). Only cross-language comparison *within a row* is
     valid; brightness must not be compared across rows or across panels.
  2. Raw min/max is sensitive to one or two outlier experts squashing the
     rest of the color range -- normalize_layer() now winsorizes each
     column at [--clip-percentile, 100-clip-percentile] before scaling.
  3. An extreme cell can be a genuine language-specific signal or a low-
     routing-sample artifact (see phase1_fisher.py's dead-expert check for
     the same underlying worry). load_scan_results()/make_synthetic_scan()
     now read/fabricate an optional "hit_count" field (same name Toy0's
     02c_analyze_layer_locality.py already uses for token-routing counts),
     and plot_heatmap_figure() hatches any cell below --min-hit-count so a
     reader (and the person writing the caption) can tell a spike from an
     artifact at a glance.
  4. "balanced" ambiguity -- phase1_calib_data.py's "balanced" condition
     interleaves only EN/KO/ZH (not SW/BN); "mixed_5lang" interleaves all 5.
     check_balanced_semantics() also asserts the chosen column is NOT just
     the post-hoc mean of the other 5 (which would mean it was computed by
     averaging instead of an actual balanced-corpus forward pass).

GPU not needed -- pure post-processing/visualization, runs on CPU.

Usage:
    conda run -n d2moe_env python make_figure_expert_language_heatmap.py \
        [--scan-results PATH] [--top-k 30] [--n-highlight 5] \
        [--normalize minmax|zscore] [--clip-percentile 2.0] \
        [--min-hit-count 5] [--balanced-condition balanced|mixed_5lang] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from disagreement_common import (  # noqa: E402 -- shared with scan_disagreement_experts.py, see its docstring
    LANG_CONDITIONS, BALANCED_CONDITIONS, conditions_for, normalize_layer,
    check_balanced_semantics, rank_by_disagreement, select_top_k,
    make_synthetic_scan as _make_synthetic_scan,
    load_scan_results as _load_scan_results,
)
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PILOT_A_DIR = Path("/mnt/HDD/minjeong/d2moe_results/fisher_pilot_a")
SCAN_RESULTS_DEFAULT = Path("/mnt/HDD/minjeong/d2moe_results/scan_disagreement_experts/scan_results.json")

# Same DeepSeek-MoE-16B layer indices/roles fisher_pilot_a.py established
# (28 layers: 0 dense, 1-27 MoE) -- reused as constants, not re-derived.
LAYERS = [4, 16, 24, 27]
LAYER_ROLE = {4: "early", 16: "transition", 24: "sensitive", 27: "final"}

# LANG_CONDITIONS/BALANCED_CONDITIONS/conditions_for/normalize_layer/
# check_balanced_semantics/rank_by_disagreement/select_top_k now live in
# disagreement_common.py (imported above) -- shared with
# scan_disagreement_experts.py (fix 4's balanced-vs-EN/KO/ZH-only note and
# the review-point 1/2 normalization notes below still apply, just to code
# that now lives there).
COND_LABEL = {
    "english_only": "en", "chinese_only": "zh", "korean_only": "ko",
    "swahili_only": "sw", "bengali_only": "bn",
    # "*" -- see caption footnote: "balanced" mixes only en/ko/zh (fix 4).
    # Kept short (not e.g. "balanced(en/ko/zh)") so the x-tick label doesn't
    # collide with the panel row below it -- see plot_heatmap_figure spacing.
    "balanced": "balanced*", "mixed_5lang": "mixed5*",
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
HIGHLIGHT = "#D55E00"
LOWSAMPLE = "#6b6b6b"
CMAP = "viridis"


def load_scan_results(path, layers=LAYERS, conditions=None, min_hit_count=5):
    """Thin wrapper around disagreement_common.load_scan_results (the shared
    parser, see its docstring for the schema) -- kept here under its
    original name/signature so existing call sites/imports don't change,
    and to drop the "_meta" second return value this script doesn't need."""
    if conditions is None:
        conditions = conditions_for("balanced")
    scan, _meta = _load_scan_results(path, layers, conditions, layer_role=LAYER_ROLE, log_prefix="[heatmap]")
    return scan


def make_synthetic_scan(layers=LAYERS, conditions=None, seed=0):
    if conditions is None:
        conditions = conditions_for("balanced")
    return _make_synthetic_scan(layers, conditions, layer_role=LAYER_ROLE, seed=seed)


def load_reference_correlation():
    """Proxy-vs-real Spearman rho per layer, from Fisher Pilot A -- the same
    proxy formula this scan reuses, validated once against real gradient
    Fisher there (analyze_fisher_pilot_a.py). Prefers the already-computed
    pilot_a_analysis.json; falls back to recomputing directly from
    pilot_a_results.json with the same per-condition spearmanr logic if the
    analysis file hasn't been generated yet. Returns {role: mean_rho} or {}
    if neither file is present (reference is optional context, not required
    to render the heatmap)."""
    analysis_path = PILOT_A_DIR / "pilot_a_analysis.json"
    if analysis_path.exists():
        rows = json.loads(analysis_path.read_text())
        return {r["role"]: float(np.mean([v["rho"] for v in r["proxy_vs_real"].values()])) for r in rows}

    results_path = PILOT_A_DIR / "pilot_a_results.json"
    if not results_path.exists():
        print(f"[heatmap] no Fisher Pilot A results found at {PILOT_A_DIR} -- skipping reference correlation")
        return {}
    data = json.loads(results_path.read_text())
    ref = {}
    for layer_str, conds in data.items():
        role = conds.get("role", "?")
        rhos = []
        for cond, v in conds.items():
            if cond == "role":
                continue
            rho, _ = spearmanr(v["real_fisher"], v["proxy"])
            rhos.append(rho)
        ref[role] = float(np.mean(rhos))
    return ref


def style_heatmap_axis(ax):
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_heatmap_figure(scan, layers, conditions, top_k, n_highlight, normalize_method, clip_percentile,
                         min_hit_count, reference_rho, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.5), dpi=300)
    axes = axes.flatten()

    panels = []
    global_vmax = 0.0
    n_low_sample_total = 0
    for layer in layers:
        entry = scan[layer]
        norm = normalize_layer(entry["proxy"], conditions, method=normalize_method, clip_percentile=clip_percentile)
        order, variance = rank_by_disagreement(norm)
        n_total = norm.shape[0]
        top_idx, top_var = select_top_k(order, variance, top_k)
        top_norm = norm[top_idx]
        global_vmax = max(global_vmax, float(top_norm.max()))

        candidates = entry["disagreement_experts"]
        if candidates is None:
            # Fallback: scan_disagreement_experts didn't flag its own
            # candidate list, so treat our own top-`n_highlight` (a subset
            # of top_idx, already the most language-divergent) as the
            # provisional targeted-allocation candidates.
            candidates = [int(e) for e in top_idx[:n_highlight]]
        highlight_rows = [i for i, e in enumerate(top_idx) if int(e) in set(candidates)]

        low_sample_cells = []  # (row, col) pairs below min_hit_count
        if entry["hit_count"] is not None:
            hit_matrix = np.stack([entry["hit_count"][c] for c in conditions], axis=1)[top_idx]
            rows, cols = np.where(hit_matrix < min_hit_count)
            low_sample_cells = list(zip(rows.tolist(), cols.tolist()))
            n_low_sample_total += len(low_sample_cells)

        panels.append({
            "layer": layer, "role": entry["role"], "top_idx": top_idx,
            "top_norm": top_norm, "n_total": n_total, "highlight_rows": highlight_rows,
            "low_sample_cells": low_sample_cells, "has_hit_count": entry["hit_count"] is not None,
        })

    for ax, panel in zip(axes, panels):
        sns.heatmap(
            panel["top_norm"], ax=ax, cmap=CMAP, vmin=0.0, vmax=global_vmax,
            cbar=False, linewidths=0.4, linecolor="white",
            yticklabels=[str(int(e)) for e in panel["top_idx"]] if top_k <= 40 else False,
            xticklabels=[COND_LABEL[c] for c in conditions],
        )
        for row in panel["highlight_rows"]:
            ax.add_patch(Rectangle((0, row), len(conditions), 1, fill=False,
                                    edgecolor=HIGHLIGHT, linewidth=2.2, zorder=5))
        for row, col in panel["low_sample_cells"]:
            ax.add_patch(Rectangle((col, row), 1, 1, fill=False, hatch="////",
                                    edgecolor=LOWSAMPLE, linewidth=0.0, zorder=4))
        ref_txt = f", proxy-vs-real rho={reference_rho[panel['role']]:.2f}" if panel["role"] in reference_rho else ""
        ax.set_title(f"layer {panel['layer']} ({panel['role']}{ref_txt})", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("calibration language", fontsize=9, color=INK)
        ax.set_ylabel(f"expert (top {top_k} by cross-language variance)" if top_k <= 40 else "expert (variance-sorted)",
                       fontsize=9, color=INK)
        style_heatmap_axis(ax)

    fig.subplots_adjust(right=0.90, hspace=0.45, wspace=0.55)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=global_vmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cbar_ax,
                 label=f"{normalize_method}-normalized Fisher-proxy importance")

    n_omitted = panels[0]["n_total"] - top_k
    hit_count_note = (f"Hatched cell = fewer than {min_hit_count} tokens routed to that expert under that "
                       f"language (hit_count) -- treat as an under-sampled estimate, not a confirmed "
                       f"language-specific signal." if any(p["has_hit_count"] for p in panels) else
                       "hit_count not available in this scan -- extreme single-language cells (e.g. Swahili/"
                       "Bengali) are NOT yet verified against routing-sample size; see script docstring point 3.")
    caption = (
        f"Color = per-language (column-wise) {normalize_method} normalization, computed over all "
        f"{panels[0]['n_total']} experts in the layer and independently per language (each language's own "
        f"scale) -- compare cross-language color WITHIN a row only; do not compare brightness across rows or "
        f"panels. Columns clipped at the [{clip_percentile:g}, {100 - clip_percentile:g}] percentile before "
        f"scaling so 1-2 outlier experts don't compress the rest of the range. "
        f"Bold outline = candidate disagreement expert for targeted calibration allocation (scan-provided if "
        f"available, else this figure's own top-{n_highlight}). {hit_count_note} "
        f"{n_omitted} of {panels[0]['n_total']} experts per layer omitted (bottom of the cross-language "
        f"variance ranking -- language-invariant importance; see expert_disagreement_variance_hist for the "
        f"full distribution). "
        f"* '{COND_LABEL[conditions[-1]]}' = '{conditions[-1]}' "
        f"({'interleaves only EN/KO/ZH, NOT SW/BN' if conditions[-1] == 'balanced' else 'interleaves all 5 shown languages'})."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.3, color=MUTED, wrap=True)

    out_path = FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    print(f"[heatmap] wrote {out_path}")
    plt.close(fig)
    if n_low_sample_total:
        print(f"[heatmap] {n_low_sample_total} shown cells across all layers are below min_hit_count="
              f"{min_hit_count} (hatched) -- see analysis JSON's low_sample_cells for the full list.")
    return panels


def plot_variance_histogram(scan, layers, conditions, top_k, normalize_method, clip_percentile, out_stem):
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.0), dpi=300, sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        entry = scan[layer]
        norm = normalize_layer(entry["proxy"], conditions, method=normalize_method, clip_percentile=clip_percentile)
        _, variance = rank_by_disagreement(norm)
        cutoff = np.sort(variance)[::-1][top_k - 1] if top_k <= len(variance) else variance.min()

        ax.hist(variance, bins=20, color="#0072B2", edgecolor="white", linewidth=0.6, zorder=3)
        ax.axvline(cutoff, color=HIGHLIGHT, linewidth=1.6, linestyle="--", zorder=4)
        ax.set_title(f"layer {layer} ({entry['role']})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("cross-language variance\n(normalized importance)", fontsize=8.5, color=INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("# experts", fontsize=9, color=INK)
    n_total = len(scan[layers[0]]["proxy"][conditions[0]])
    fig.suptitle(f"Cross-language variance across all {n_total} experts per layer "
                 f"(dashed line = top-{top_k} cutoff used in the heatmap)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path = FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    print(f"[heatmap] wrote {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-results", type=Path, default=SCAN_RESULTS_DEFAULT,
                         help="scan_disagreement_experts output JSON (see load_scan_results docstring for schema)")
    parser.add_argument("--top-k", type=int, default=30, help="experts kept per layer, sorted by cross-language variance")
    parser.add_argument("--n-highlight", type=int, default=5,
                         help="fallback candidate-expert count if the scan doesn't flag its own disagreement_experts")
    parser.add_argument("--normalize", choices=["minmax", "zscore"], default="minmax")
    parser.add_argument("--clip-percentile", type=float, default=2.0,
                         help="winsorize each language's column at [p, 100-p] before normalizing (0 disables)")
    parser.add_argument("--min-hit-count", type=int, default=5,
                         help="cells with fewer routed tokens than this are hatched as low-sample (needs scan hit_count)")
    parser.add_argument("--balanced-condition", choices=BALANCED_CONDITIONS, default="balanced",
                         help="balanced-type condition: 'balanced' (phase1_calib_data.py's EN/KO/ZH-only mix) or "
                              "'mixed_5lang' (all 5 shown languages)")
    parser.add_argument("--smoke", action="store_true", help="use a fabricated synthetic scan instead of --scan-results")
    args = parser.parse_args()

    sns.set_theme(style="white")
    conditions = conditions_for(args.balanced_condition)

    if args.smoke:
        print("[heatmap] --smoke: using a fabricated synthetic scan, NOT real data")
        scan = make_synthetic_scan(conditions=conditions)
    elif args.scan_results.exists():
        print(f"[heatmap] loading {args.scan_results}")
        scan = load_scan_results(args.scan_results, conditions=conditions, min_hit_count=args.min_hit_count)
    else:
        raise FileNotFoundError(
            f"{args.scan_results} not found. Run scan_disagreement_experts (or point --scan-results at its "
            "output) first, or pass --smoke to exercise this script's plotting pipeline on synthetic data. "
            "Expected schema: see load_scan_results()'s docstring."
        )

    reference_rho = load_reference_correlation()
    if reference_rho:
        print("[heatmap] reference: Fisher Pilot A proxy-vs-real Spearman rho by role "
              "(validation context for this scan's proxy formula):")
        for role, rho in reference_rho.items():
            print(f"[heatmap]   {role}: rho={rho:.3f}")

    print(f"[heatmap] balanced-type condition in use: {args.balanced_condition!r} "
          f"({'EN/KO/ZH only, NOT SW/BN' if args.balanced_condition == 'balanced' else 'all 5 shown languages'})")
    balanced_checks = {}
    for layer in LAYERS:
        r, max_diff, suspicious = check_balanced_semantics(scan[layer]["proxy"], args.balanced_condition)
        balanced_checks[layer] = {"pearson_r_vs_mean_of_5": r, "max_abs_diff_fraction": max_diff, "suspicious": suspicious}
        flag = " *** SUSPICIOUS -- looks like a post-hoc average, not a real balanced-corpus run ***" if suspicious else ""
        print(f"[heatmap]   layer {layer}: {args.balanced_condition} vs mean(5 languages) pearson_r={r:.4f}, "
              f"max_abs_diff_fraction={max_diff:.4f}{flag}")
    if any(v["suspicious"] for v in balanced_checks.values()):
        print(f"[heatmap] WARNING: the {args.balanced_condition!r} column is statistically indistinguishable from "
              f"the post-hoc mean of the other 5 languages at one or more layers -- verify scan_disagreement_experts "
              f"actually ran a separate balanced-corpus forward pass for this condition (2026-07-27 review point 4) "
              f"before using it in the paper.")

    heatmap_stem = "expert_language_heatmap_4layer"
    hist_stem = "expert_disagreement_variance_hist"
    panels = plot_heatmap_figure(scan, LAYERS, conditions, args.top_k, args.n_highlight, args.normalize,
                                  args.clip_percentile, args.min_hit_count, reference_rho, heatmap_stem)
    plot_variance_histogram(scan, LAYERS, conditions, args.top_k, args.normalize, args.clip_percentile, hist_stem)

    analysis = {
        "top_k": args.top_k, "n_highlight": args.n_highlight, "normalize": args.normalize,
        "clip_percentile": args.clip_percentile, "min_hit_count": args.min_hit_count,
        "balanced_condition": args.balanced_condition, "balanced_semantics_check_by_layer": balanced_checks,
        "reference_rho_by_role": reference_rho,
        "layers": [{
            "layer": p["layer"], "role": p["role"], "n_experts_total": p["n_total"],
            "n_experts_shown": len(p["top_idx"]),
            "top_expert_indices": [int(e) for e in p["top_idx"]],
            "low_sample_cells": [{"row_expert_idx": int(p["top_idx"][row]), "condition": conditions[col]}
                                  for row, col in p["low_sample_cells"]],
        } for p in panels],
    }
    out_json = RESULTS_DIR / "expert_language_heatmap_analysis.json"
    out_json.write_text(json.dumps(analysis, indent=2))
    print(f"[heatmap] wrote {out_json}")


if __name__ == "__main__":
    main()
