"""§5.1 "Fisher expert 순위의 언어 간 상관" (논문 초안, 2026-07-30 draft): does
calibration language reorder which experts D^2-MoE's Fisher-weighted merge
(W_base = sum_e (F_e / sum F) W_e) treats as important? For every MoE layer,
Spearman rank-correlates each calibration-language pair's per-expert Fisher
importance, and compares that against PLACEBO pairs (same language, disjoint
calibration text -- english_only vs english_only_b etc., phase1_calib_data.
PLACEBO_CONDITIONS) so a language-driven rank disagreement can be told apart
from ordinary finite-sample noise (placebo rho is the noise-floor ceiling: a
cross-language rho has to sit meaningfully BELOW it to blame language, not
sampling, for the disagreement).

Uses the REAL gradient Fisher phase1_fisher.py already computed for the actual
merge/eval pipeline (fisher_processed.pt), not the forward-only proxy
scan_disagreement_experts.py uses for the disagreement-targeted-calibration
method (06_논문_구성.md §6) -- that scan answers a different question
(routing-sample coverage) and 09_부족한_실험_정리.md / the draft both call for
the real-Fisher version here. Each fisher_processed.pt is a
{layer_idx: {"mlp.experts.<e>.<proj>": tensor}} dict, torch.save'd whole,
~30GB per condition/seed at bf16 (27 layers x 64 experts x 3 projections) --
loaded with mmap=True (torch>=2.1) so the multi-GB read is file-backed
(evictable page cache), not anonymous heap, and reduced to one scalar per
(layer, expert) immediately: F_e = sum over that expert's gate/up/down
projections of the (already sample-averaged, see phase1_fisher.py) per-element
squared-gradient tensor. Same reduction fisher_pilot_a.py's
real_fisher_for_condition used (sum of squared gradients per expert) -- just
applied to the on-disk real Fisher instead of a fresh backward pass, so this
script needs no GPU and no forward pass, only CPU + disk I/O. Reduced arrays
are cached to <RESULTS_ROOT>/fisher_expert_scalar_cache/ (tiny JSON, one
`--seed`/condition combination at a time) so a re-run (e.g. adding a language
or seed) never re-reads a multi-GB file for a condition already reduced.

Usage:
    conda run -n d2moe_env python phase1_51_fisher_rank_correlation.py [--seed 0] [--smoke]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402
from fisher_pilot_a import LAYER_ROLE as PILOT_LAYER_ROLE  # noqa: E402 -- only 4 layers, rest default to "other"

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
CACHE_DIR = RESULTS_ROOT / "fisher_expert_scalar_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_MOE_LAYERS = list(range(1, 28))  # 0 is dense, 1..27 MoE (phase1_fisher.py)
LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
# chinese_only has no "_b" placebo arm defined at all (phase1_calib_data.
# PLACEBO_CONDITIONS only lists english/korean/swahili/bengali). Separately,
# bengali_only_b IS defined there but its real Fisher was never actually run
# (dir exists, empty -- confirmed 2026-07-30, see 08_figure_정리.md's "어느
# 언어에 placebo가 있고(EN/KO/SW) 없는지(ZH/BN)" note) -- so this checks
# on-disk fisher_processed.pt existence, not just the name list, and drops
# whichever placebo(s) aren't actually there instead of crashing mid-run.
def _placebo_of():
    out = {}
    for c in LANG_CONDITIONS:
        b = f"{c}_b"
        if b not in phase1_calib_data.PLACEBO_CONDITIONS:
            continue
        if not (RESULTS_ROOT / b / "seed0" / "fisher_processed.pt").exists():
            print(f"[rank-corr] WARNING: {b} has no real Fisher on disk (fisher_processed.pt missing) -- "
                  f"excluding {c} from the placebo comparison, not just chinese_only")
            continue
        out[c] = b
    return out


PLACEBO_OF = _placebo_of()


def layer_role(layer_idx):
    return PILOT_LAYER_ROLE.get(layer_idx, "other")


def reduce_condition(condition, seed, layers):
    """Returns {layer_idx: np.ndarray[n_experts]} of per-expert Fisher scalars,
    from cache if a prior run already reduced every requested layer for this
    condition/seed, else by loading+reducing fisher_processed.pt (slow, ~30GB
    disk read) and caching the result."""
    cache_path = CACHE_DIR / f"{condition}_seed{seed}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if all(str(l) in cached for l in layers):
            print(f"[rank-corr] {condition} seed{seed}: cache hit ({cache_path.name})")
            return {l: np.asarray(cached[str(l)], dtype=np.float64) for l in layers}

    pt_path = RESULTS_ROOT / condition / f"seed{seed}" / "fisher_processed.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"{pt_path} missing -- run phase1_fisher.py --condition {condition} --seed {seed} first"
        )
    import torch  # deferred: only needed on an actual cache miss

    print(f"[rank-corr] {condition} seed{seed}: cache miss, loading {pt_path} (mmap=True) ...")
    assembled = torch.load(pt_path, map_location="cpu", mmap=True)
    out = {}
    for layer_idx in layers:
        layer_dict = assembled[layer_idx]
        n_experts = max(int(k.split(".")[2]) for k in layer_dict) + 1
        scalar = np.zeros(n_experts, dtype=np.float64)
        for key, tensor in layer_dict.items():
            e_idx = int(key.split(".")[2])
            scalar[e_idx] += tensor.float().sum().item()
        out[layer_idx] = scalar
        print(f"[rank-corr] {condition} seed{seed} layer {layer_idx}: reduced "
              f"({n_experts} experts, nonzero={int((scalar > 0).sum())})")
    del assembled

    # Merge with anything already cached for OTHER layers of this condition
    # (e.g. a prior --smoke run only reduced 2 layers) instead of clobbering it.
    existing = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    existing.update({str(k): v.tolist() for k, v in out.items()})
    cache_path.write_text(json.dumps(existing))
    print(f"[rank-corr] {condition} seed{seed}: cached -> {cache_path}")
    return out


def compute_correlations(scalars_by_cond, layers):
    lang_pairs = list(itertools.combinations(LANG_CONDITIONS, 2))
    placebo_pairs = list(PLACEBO_OF.items())
    per_layer = {}
    for layer in layers:
        row = {"role": layer_role(layer), "lang_pairs": {}, "placebo_pairs": {}}
        for a, b in lang_pairs:
            rho, p = spearmanr(scalars_by_cond[a][layer], scalars_by_cond[b][layer])
            row["lang_pairs"][f"{a}|{b}"] = {"rho": float(rho), "p": float(p)}
        for a, b in placebo_pairs:
            rho, p = spearmanr(scalars_by_cond[a][layer], scalars_by_cond[b][layer])
            row["placebo_pairs"][f"{a}|{b}"] = {"rho": float(rho), "p": float(p)}
        lang_rhos = [v["rho"] for v in row["lang_pairs"].values()]
        placebo_rhos = [v["rho"] for v in row["placebo_pairs"].values()]
        row["mean_lang_rho"] = float(np.mean(lang_rhos))
        row["mean_placebo_rho"] = float(np.mean(placebo_rhos)) if placebo_rhos else None
        per_layer[layer] = row
    return per_layer


def summarize(per_layer, layers):
    n = len(layers)
    front = layers[: n // 2]
    back = layers[n // 2:]

    def band_means(band):
        lang = [per_layer[l]["mean_lang_rho"] for l in band]
        placebo = [per_layer[l]["mean_placebo_rho"] for l in band if per_layer[l]["mean_placebo_rho"] is not None]
        return {
            "layers": band,
            "mean_lang_rho": float(np.mean(lang)) if lang else None,
            "mean_placebo_rho": float(np.mean(placebo)) if placebo else None,
        }

    return {
        "overall": band_means(layers),
        "front_half": band_means(front),
        "back_half": band_means(back),
        "min_lang_rho_layer": min(layers, key=lambda l: per_layer[l]["mean_lang_rho"]),
        "max_gap_layer": max(
            (l for l in layers if per_layer[l]["mean_placebo_rho"] is not None),
            key=lambda l: per_layer[l]["mean_placebo_rho"] - per_layer[l]["mean_lang_rho"],
            default=None,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="2 layers, english_only(+_b) only -- pipeline sanity check")
    args = parser.parse_args()

    layers = [1, 27] if args.smoke else ALL_MOE_LAYERS
    conditions = ["english_only", "english_only_b"] if args.smoke else (
        LANG_CONDITIONS + list(PLACEBO_OF.values())
    )

    print(f"[rank-corr] {'SMOKE' if args.smoke else 'FULL'}: seed={args.seed} layers={layers} conditions={conditions}")
    scalars_by_cond = {c: reduce_condition(c, args.seed, layers) for c in conditions}

    if args.smoke:
        print("[rank-corr] smoke reduction OK, skipping correlation (needs >=2 languages)")
        return

    per_layer = compute_correlations(scalars_by_cond, layers)
    summary = summarize(per_layer, layers)

    out_path = RESULTS_ROOT / f"phase1_51_fisher_rank_correlation_result_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "seed": args.seed, "layers": layers,
        "lang_conditions": LANG_CONDITIONS, "placebo_of": PLACEBO_OF,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "summary": summary,
    }, indent=2))
    print(f"[rank-corr] wrote {out_path}")

    print("\n[rank-corr] === summary ===")
    for band_name in ("overall", "front_half", "back_half"):
        b = summary[band_name]
        print(f"  {band_name} (layers {b['layers'][0]}-{b['layers'][-1]}): "
              f"mean_lang_rho={b['mean_lang_rho']:.4f} mean_placebo_rho={b['mean_placebo_rho']:.4f}")
    print(f"  lowest-lang-rho layer: {summary['min_lang_rho_layer']} "
          f"(rho={per_layer[summary['min_lang_rho_layer']]['mean_lang_rho']:.4f})")
    if summary["max_gap_layer"] is not None:
        l = summary["max_gap_layer"]
        print(f"  largest (placebo - lang) gap layer: {l} "
              f"(placebo={per_layer[l]['mean_placebo_rho']:.4f}, lang={per_layer[l]['mean_lang_rho']:.4f})")


if __name__ == "__main__":
    main()
