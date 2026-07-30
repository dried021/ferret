"""Shared, plotting-dependency-free math for the disagreement-aware
calibration pipeline (01_연구설계.md Section 23, 06_논문_구성.md §5-2/§6):
the condition list, column-wise normalization, and variance-based expert
ranking used by BOTH scan_disagreement_experts.py (Stage 1, runs inside
d2moe_env on GPU -- must NOT pull in matplotlib/seaborn) and
make_figure_expert_language_heatmap.py (the paper figure). Factored out
2026-07-27 after discovering the figure script's unconditional `import
seaborn` broke scan_disagreement_experts.py's import in d2moe_env (seaborn
is a plotting-only dependency never installed there) -- this module has no
plotting imports at all, only numpy.
"""
import json
from pathlib import Path

import numpy as np

# phase1_calib_data.py's CONDITIONS, restricted to the 5 single-language
# conditions the disagreement scan/figure compare (excludes the "_b" placebo
# arms, which are a noise-floor check, not a language condition) plus ONE
# balanced-type condition, chosen via --balanced-condition. "balanced" mixes
# only EN/KO/ZH; "mixed_5lang" mixes all 5 (see phase1_calib_data.py's
# MIXED_LANGS and check_balanced_semantics() below).
LANG_CONDITIONS = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only"]
BALANCED_CONDITIONS = ["balanced", "mixed_5lang"]


def conditions_for(balanced_condition):
    if balanced_condition not in BALANCED_CONDITIONS:
        raise ValueError(f"--balanced-condition must be one of {BALANCED_CONDITIONS}, got {balanced_condition!r}")
    return LANG_CONDITIONS + [balanced_condition]


def normalize_layer(proxy_by_cond, conditions, method="minmax", clip_percentile=2.0, eps=1e-12):
    """COLUMN-wise (per-language) normalization: each condition's importance
    vector is normalized independently, across ALL experts in this layer --
    so different languages' raw proxy scales don't distort the cross-
    language comparison. Only ROW-wise (within one expert, across languages)
    comparisons are meaningful in the output; brightness/magnitude must
    never be compared across rows or across layers/panels (2026-07-27 review
    point 1).

    Before scaling, each column is winsorized at
    [clip_percentile, 100-clip_percentile] (2026-07-27 review point 2): raw
    min/max normalization lets one or two outlier experts stretch the scale
    so far that every other expert's genuine cross-language variation gets
    compressed into a narrow, visually-uniform color band. Set
    clip_percentile=0 to disable and reproduce plain min/max.

    Returns (n_experts, n_conditions) array, columns in `conditions` order.
    """
    cols = []
    for cond in conditions:
        v = proxy_by_cond[cond]
        if clip_percentile > 0:
            lo_clip, hi_clip = np.percentile(v, [clip_percentile, 100 - clip_percentile])
            v = np.clip(v, lo_clip, hi_clip)
        if method == "minmax":
            lo, hi = v.min(), v.max()
            cols.append((v - lo) / (hi - lo + eps))
        elif method == "zscore":
            cols.append((v - v.mean()) / (v.std() + eps))
        else:
            raise ValueError(f"unknown normalize method {method!r}")
    return np.stack(cols, axis=1)


def check_balanced_semantics(proxy_by_cond, balanced_condition, lang_conditions=LANG_CONDITIONS):
    """2026-07-27 review point 4: verifies the "balanced" column is not
    secretly the post-hoc mean of the other 5 language columns (which would
    mean it was never computed from an actual balanced-corpus forward pass --
    see phase1_calib_data.py's real "balanced"/"mixed_5lang" conditions).
    Returns (pearson_r, max_abs_diff_fraction, suspicious: bool)."""
    balanced = proxy_by_cond[balanced_condition]
    mean_of_others = np.mean([proxy_by_cond[c] for c in lang_conditions], axis=0)
    if balanced.std() < 1e-12 or mean_of_others.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(balanced, mean_of_others)[0, 1])
    denom = np.maximum(np.abs(balanced), np.abs(mean_of_others)).max() + 1e-12
    max_abs_diff_fraction = float(np.max(np.abs(balanced - mean_of_others)) / denom)
    suspicious = (not np.isnan(r)) and r > 0.999 and max_abs_diff_fraction < 1e-3
    return r, max_abs_diff_fraction, suspicious


def rank_by_disagreement(normalized_matrix):
    """Row-wise (per-expert) variance across languages, on normalized values
    -- descending order is "most language-divergent expert first"."""
    variance = normalized_matrix.var(axis=1)
    order = np.argsort(-variance)
    return order, variance


def select_top_k(order, variance, top_k):
    return order[:top_k], variance[order[:top_k]]


def load_scan_results(path, layers, conditions, layer_role=None, min_hit_count=5, log_prefix="[scan]"):
    """Reads scan_disagreement_experts.py's output JSON. Shared by
    make_figure_expert_language_heatmap.py (4 role layers) and
    phase1_6_targeted_budget.py (all 27 MoE layers) -- this is the ONE
    parser for that file; both callers must transform the on-disk schema
    identically, so this used to be duplicated and silently drifted (Stage
    2's own ad hoc loader assumed `entry[cond]["hit_count"]`, the on-disk
    shape, when the rest of the pipeline actually consumes the transformed
    `entry["hit_count"][cond]` shape below -- caught by cross-testing the
    two scripts against the same file, 2026-07-27).

    On-disk schema (written by scan_disagreement_experts.py):
        {"<layer_idx>": {"role": "<early|transition|sensitive|final|other>",
                          "<condition>": {"proxy": [float, ...n_experts],
                                           "hit_count": [int, ...n_experts]},
                          ...for each condition...,
                          "disagreement_experts": [int, ...]},
         "_meta": {"n_tokens": {"<condition>": int, ...}}}
    hit_count is OPTIONAL per condition but strongly recommended -- without
    it, callers cannot tell a genuine language-specific spike from a low-
    routing-sample artifact (this module will print a loud warning).

    Returns {layer: {"role": str, "proxy": {cond: np.ndarray[n_experts]},
                      "hit_count": {cond: np.ndarray[n_experts]} | None,
                      "disagreement_experts": list[int] | None}}
    (note: "proxy"/"hit_count" are now keyed BY CONDITION, not the on-disk
    per-condition-then-field nesting -- this is the transform every
    downstream function in this module and in make_figure_expert_language_
    heatmap.py / phase1_6_targeted_budget.py actually consumes)."""
    if layer_role is None:
        layer_role = {}
    data = json.loads(Path(path).read_text())
    out = {}
    any_missing_hit_count = False
    for layer in layers:
        key = str(layer)
        if key not in data:
            raise KeyError(f"{path}: layer {layer} missing (need {layers})")
        entry = data[key]
        proxy, hit_count = {}, {}
        for cond in conditions:
            if cond not in entry:
                raise KeyError(f"{path}: layer {layer} missing condition {cond!r} (need {conditions})")
            proxy[cond] = np.asarray(entry[cond]["proxy"], dtype=np.float64)
            if "hit_count" in entry[cond]:
                hit_count[cond] = np.asarray(entry[cond]["hit_count"], dtype=np.int64)
            else:
                any_missing_hit_count = True
        n_experts = {len(v) for v in proxy.values()}
        if len(n_experts) != 1:
            raise ValueError(f"{path}: layer {layer} conditions disagree on n_experts: "
                              f"{ {c: len(v) for c, v in proxy.items()} }")
        out[layer] = {
            "role": entry.get("role", layer_role.get(layer, "other")),
            "proxy": proxy,
            "hit_count": hit_count if len(hit_count) == len(conditions) else None,
            "disagreement_experts": entry.get("disagreement_experts"),
        }
    if any_missing_hit_count:
        print(f"{log_prefix} WARNING: {path} has no (or incomplete) hit_count for at least one layer/condition -- "
              f"cannot verify whether extreme cells are genuine language-specific signal or low-routing-sample "
              f"artifacts. Add hit_count to scan_disagreement_experts's output before trusting any single-language "
              f"spike (e.g. a Swahili/Bengali-only 'important' expert) in the paper.")
    return out, data.get("_meta", {})


def make_synthetic_scan(layers, conditions, layer_role=None, n_experts=64, seed=0):
    """Fabricates a plausible-shaped scan (a handful of genuinely language-
    divergent experts among a majority of language-invariant ones, PLUS a
    few low-hit_count cells to exercise the artifact-flagging path) so
    --smoke can exercise a full loading/normalize/rank/(plot|allocate)
    pipeline without a real scan_disagreement_experts run. Never written to
    disk, clearly logged as synthetic -- not a substitute for real data.
    Shared by make_figure_expert_language_heatmap.py (4 role layers) and
    phase1_6_targeted_budget.py (all 27 MoE layers) -- `layer_role` lets each
    caller supply its own role labels (fisher_pilot_a.LAYER_ROLE) without
    this module depending on either."""
    if layer_role is None:
        layer_role = {}
    rng = np.random.default_rng(seed)
    n_divergent = max(3, n_experts // 12)
    divergent_experts = rng.choice(n_experts, size=n_divergent, replace=False)
    # A couple of the divergent experts get an artificially low hit_count
    # under one non-English language -- the exact ambiguous case
    # make_figure_expert_language_heatmap.py's review point 3 flagged (a
    # spike that might just be a starved sample).
    low_sample_experts = divergent_experts[:2]
    out = {}
    for layer in layers:
        base = rng.gamma(shape=2.0, scale=1.0, size=n_experts)
        proxy, hit_count = {}, {}
        for cond in conditions:
            vals = base.copy()
            vals[divergent_experts] *= rng.uniform(0.2, 3.0, size=n_divergent)
            vals += rng.normal(0, 0.05, size=n_experts).clip(min=0)
            proxy[cond] = vals
            counts = rng.integers(20, 200, size=n_experts)
            if cond in ("swahili_only", "bengali_only"):
                counts[low_sample_experts] = rng.integers(0, 4, size=len(low_sample_experts))
            hit_count[cond] = counts
        out[layer] = {
            "role": layer_role.get(layer, "other"),
            "proxy": proxy,
            "hit_count": hit_count,
            "disagreement_experts": sorted(int(e) for e in divergent_experts),
        }
    return out
