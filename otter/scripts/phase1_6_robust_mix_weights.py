"""§6 whitening-targeted robust mixing, STEP 1 (CPU, no new GPU calibration):
solves w* = argmin_w max_l relFrob(Sigma(w), Sigma_l) over the 5 single-
language conditions' activation covariances, reusing phase1_svd_scale.py's
ALREADY-COMPUTED svd_scale_processed.pt Cholesky factors (seed0) exactly the
way phase1_whitening_geometry.py does -- see that script's module docstring
for the derivation of why a truncated SVD of the Cholesky factor L (C = L L^T)
gives C's top-k eigenvectors/eigenvalues without ever forming the dense d x d
covariance.

Why this needs a NEW script instead of reading whitening_geometry_pairs.csv:
that CSV stores, per (layer, matrix_type, cond_a, cond_b), only the FINAL
relative-Frobenius ratio between a fixed PAIR (already normalized by a
denominator that mixes in each side's own norm) -- not the underlying
tr(Sigma_i Sigma_j) trace components. Solving for an arbitrary mixing weight
w over 5 languages needs the full 5x5 Gram matrix G_e[i,j] = tr(Sigma_i
Sigma_j) per (layer, matrix_type, expert) cell, from which ANY linear
combination Sigma(w) = sum_i w_i Sigma_i can be compared to any Sigma_l
without re-touching the Cholesky factors again ("linearity, no
re-extraction" per the task spec) -- so this script re-derives G_e once
(same extract_topk() cost phase1_whitening_geometry.py already paid, just
for 5 nodes instead of 18) and then treats w as a free variable purely
in terms of already-extracted (U, eigval) pairs.

Math (mirrors phase1_whitening_geometry.py's covariance_distance() trace
trick, generalized from a fixed pair to an arbitrary weighted sum):
  Sigma_i ~= U_i diag(eigval_i) U_i^T   (top-k truncated, k=K_PRIMARY=64)
  tr(Sigma_i Sigma_j) = sum_pq eigval_i[p] eigval_j[q] (U_i^T U_j)[p,q]^2
  Sigma(w) = sum_i w_i Sigma_i  (exact linear combination, no approximation
             beyond the top-k truncation already applied to each Sigma_i)
  ||Sigma(w)||_F^2   = w^T G w                      (G[i,j] = tr(Sigma_i Sigma_j))
  ||Sigma(w)-Sigma_l||_F^2 = w^T G w - 2 (G w)[l] + G[l,l]
  relFrob(Sigma(w), Sigma_l) = ||Sigma(w)-Sigma_l||_F / (0.5*(||Sigma(w)||_F + ||Sigma_l||_F))
(same "mean-of-both-norms" denominator convention as whitening_distance()/
covariance_distance() in phase1_whitening_geometry.py.)

Two solve granularities, both reported (task spec "레이어별로 풀지 전체
평균으로 풀지 둘 다 계산해서 w 비교"):
  --per-layer : one w* per MoE layer, pooling that layer's 3 matrix_types x
                common experts as the averaging batch for relFrob_l(w)
  --global    : one w* pooling ALL layers x matrix_types x common experts
                (this is the PRIMARY output -- the candidate used for STEP 2's
                actual robust_mix calibration text)
G_e is stored per (layer, matrix_type, expert) cell; relFrob_l(w) at any
pooling granularity is just the mean, over that granularity's batch of G_e
cells, of the per-cell relFrob_l(w) computed from THE SAME candidate w. This
mirrors phase1_whitening_geometry.py's own convention of averaging per-
expert relFrob values (not averaging the underlying Gram/eigen quantities
themselves, which would not commute through the nonlinear sqrt/ratio).

Dead/low-hit expert exclusion: identical policy to phase1_whitening_
geometry.py (routing_coverage_flags.json, kind in {dead_fallback_identity,
low_hit_flagged}, source="svd_scale") -- an expert must be eligible (not
excluded) in ALL 5 conditions to be included in a cell's batch, otherwise
its Sigma_i for the excluded condition(s) is a meaningless identity fallback.

w* solve: min_{w in simplex} max_l relFrob_l(w) is a smooth (away from w=0)
but non-convex ratio-of-quadratics minimax over only 5 variables -- solved
via SLSQP epigraph form (variables [w(5), t], minimize t s.t. relFrob_l(w)
<= t for all l, sum(w)=1, w>=0), the same LP-epigraph IDEA
phase1_6_interference_model.py's solve_minimax() uses for its (linear) a.s
objective, generalized to this nonlinear objective via scipy.optimize.minimize
(SLSQP handles nonlinear inequality constraints; multi-start from several
initial points guards against SLSQP's local-optimum risk on a non-convex
objective, since 5 variables is cheap to restart many times).

Usage:
    conda run -n d2moe_env python phase1_6_robust_mix_weights.py [--smoke]
        [--num-threads 6] [--n-restarts 12]
    (No GPU needed -- pure CPU post-processing over already-written
    svd_scale_processed.pt files; requires seed0 for all 5 single-language
    conditions in LANG_CONDITIONS below.)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase1_fisher import MOE_LAYERS  # noqa: E402 -- 1..27 for DeepSeek-MoE-16B
from phase1_whitening_geometry import (  # noqa: E402 -- reuse verbatim, see module docstring
    K_PRIMARY, LOW_HIT_THRESHOLD, N_EXPERTS, expert_key, extract_topk, load_excluded_experts,
)

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
OUT_DIR = SCRIPT_DIR.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
LANG_CONDITIONS = {
    "eng_Latn": "english_only", "kor_Hang": "korean_only", "zho_Hans": "chinese_only",
    "swh_Latn": "swahili_only", "ben_Beng": "bengali_only",
}
MATRIX_TYPES = ["gate_proj", "up_proj", "down_proj"]
BALANCED_W = np.full(5, 1.0 / 5)
EPS = 1e-12


def load_node(condition, seed=SEED):
    path = RESULTS_ROOT / condition / f"seed{seed}" / "svd_scale_processed.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run phase1_svd_scale.py --condition {condition} --seed {seed} first")
    handle = torch.load(path, map_location="cpu", mmap=True)
    excluded = load_excluded_experts(condition, seed)
    return handle, excluded


def gram_for_cell(handles, excluded_by_lang, layer, matrix_type, n_experts):
    """Returns list of 5x5 numpy Gram matrices, one per expert eligible
    (not excluded) in ALL 5 languages at this (layer, matrix_type)."""
    per_lang_UE = {}
    for lang in LANGS:
        layer_dict = handles[lang][layer]
        excl = excluded_by_lang[lang].get(layer, set())
        per_lang_UE[lang] = {}
        for e in range(n_experts):
            if e in excl:
                continue
            key = expert_key(e, matrix_type)
            if key not in layer_dict:
                continue
            per_lang_UE[lang][e] = extract_topk(layer_dict[key])

    common = set(per_lang_UE[LANGS[0]].keys())
    for lang in LANGS[1:]:
        common &= set(per_lang_UE[lang].keys())

    grams = []
    for e in sorted(common):
        U = {lang: per_lang_UE[lang][e][0] for lang in LANGS}
        Ev = {lang: per_lang_UE[lang][e][1] for lang in LANGS}
        G = np.zeros((5, 5), dtype=np.float64)
        for i, li in enumerate(LANGS):
            for j, lj in enumerate(LANGS):
                if j < i:
                    G[i, j] = G[j, i]
                    continue
                M = (U[li].T @ U[lj]).double().numpy()
                G[i, j] = float(np.sum((M ** 2) * np.outer(Ev[li].double().numpy(), Ev[lj].double().numpy())))
        grams.append(G)
    return grams, len(common)


def relfrob_all(w, G_batch):
    """G_batch: (n, 5, 5). Returns (n, 5) relFrob(Sigma(w), Sigma_l) for
    l=0..4, one row per Gram matrix in the batch."""
    Gw = np.einsum("nij,j->ni", G_batch, w)  # (n,5) = G_e @ w
    quad_w = np.einsum("i,ni->n", w, Gw)  # (n,) = w^T G_e w
    diag = np.diagonal(G_batch, axis1=1, axis2=2)  # (n,5) = tr(Sigma_l^2)
    d2 = np.clip(quad_w[:, None] - 2 * Gw + diag, 0, None)
    denom = 0.5 * (np.sqrt(np.clip(quad_w, 0, None))[:, None] + np.sqrt(np.clip(diag, 0, None))) + EPS
    return np.sqrt(d2) / denom


def objective_max_mean_relfrob(w, G_batch):
    return relfrob_all(w, G_batch).mean(axis=0).max()


def solve_minimax(G_batch, n_restarts=12, seed=0):
    """min_w max_l mean_e relFrob_l(w, G_batch[e]) over the probability
    simplex, via SLSQP epigraph form with multi-start (module docstring)."""
    n = 5
    cons = [{"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0}]
    for l in range(n):
        cons.append({
            "type": "ineq",
            "fun": (lambda x, l=l: x[n] - relfrob_all(x[:n], G_batch).mean(axis=0)[l]),
        })
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]

    rng = np.random.default_rng(seed)
    starts = [BALANCED_W] + [np.array([1.0, 0, 0, 0, 0])[np.argsort(rng.permutation(5))] for _ in range(n_restarts - 1)]
    best = None
    for w0 in starts:
        t0 = objective_max_mean_relfrob(w0, G_batch)
        x0 = np.concatenate([w0, [t0 + 1e-3]])
        res = minimize(lambda x: x[n], x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 300, "ftol": 1e-10})
        w_res = np.clip(res.x[:n], 0, None)
        w_res = w_res / w_res.sum()
        val = objective_max_mean_relfrob(w_res, G_batch)
        if best is None or val < best[1]:
            best = (w_res, val)
    return best  # (w*, max_l mean_e relFrob_l(w*))


def build_all_cells(handles, excluded_by_lang, layers, matrix_types, n_experts, log_prefix="[robust-mix]"):
    """{(layer, matrix_type): (G_batch [n,5,5], n_common_experts)} for every
    requested cell."""
    cells = {}
    total = len(layers) * len(matrix_types)
    i = 0
    for layer in layers:
        for matrix_type in matrix_types:
            i += 1
            t0 = time.time()
            grams, n_common = gram_for_cell(handles, excluded_by_lang, layer, matrix_type, n_experts)
            cells[(layer, matrix_type)] = (np.stack(grams, axis=0) if grams else np.zeros((0, 5, 5)), n_common)
            print(f"{log_prefix} [{i}/{total}] layer={layer} {matrix_type}: "
                  f"{n_common} common experts, {time.time()-t0:.1f}s")
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="2 layers, 8 experts, gate_proj only")
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--n-restarts", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)

    layers = MOE_LAYERS[:2] if args.smoke else MOE_LAYERS
    matrix_types = ["gate_proj"] if args.smoke else MATRIX_TYPES
    n_experts = 8 if args.smoke else N_EXPERTS
    out_path = OUT_DIR / ("robust_mix_weights_smoke.json" if args.smoke else "robust_mix_weights.json")

    print(f"[robust-mix] loading seed{SEED} svd_scale_processed.pt for {len(LANGS)} languages "
          f"(K_PRIMARY={K_PRIMARY}, LOW_HIT_THRESHOLD={LOW_HIT_THRESHOLD})...")
    handles, excluded_by_lang = {}, {}
    for lang in LANGS:
        t0 = time.time()
        handles[lang], excluded_by_lang[lang] = load_node(LANG_CONDITIONS[lang])
        print(f"[robust-mix]   opened {LANG_CONDITIONS[lang]}/seed{SEED} ({lang}) in {time.time()-t0:.1f}s")

    cells = build_all_cells(handles, excluded_by_lang, layers, matrix_types, n_experts)

    # ---- global solve: pool every cell's Gram matrices into one batch ----
    global_batch = np.concatenate([g for g, _ in cells.values() if g.shape[0] > 0], axis=0)
    print(f"[robust-mix] global solve over {global_batch.shape[0]} pooled (layer,matrix,expert) cells...")
    w_global, val_global = solve_minimax(global_batch, n_restarts=args.n_restarts)
    balanced_relfrob_global = relfrob_all(BALANCED_W, global_batch).mean(axis=0)
    wstar_relfrob_global = relfrob_all(w_global, global_batch).mean(axis=0)

    # ---- per-layer solve: pool each layer's matrix_types x experts ----
    per_layer = {}
    for layer in layers:
        layer_batch = np.concatenate(
            [cells[(layer, mt)][0] for mt in matrix_types if cells[(layer, mt)][0].shape[0] > 0], axis=0)
        if layer_batch.shape[0] == 0:
            print(f"[robust-mix] WARNING layer={layer}: zero common-expert cells, skipping")
            continue
        w_l, val_l = solve_minimax(layer_batch, n_restarts=args.n_restarts)
        balanced_l = relfrob_all(BALANCED_W, layer_batch).mean(axis=0)
        per_layer[layer] = {
            "w_star": dict(zip(LANGS, w_l.tolist())),
            "max_relfrob_wstar": float(val_l),
            "max_relfrob_balanced": float(balanced_l.max()),
            "per_lang_relfrob_wstar": dict(zip(LANGS, relfrob_all(w_l, layer_batch).mean(axis=0).tolist())),
            "per_lang_relfrob_balanced": dict(zip(LANGS, balanced_l.tolist())),
            "n_cells": int(layer_batch.shape[0]),
        }

    improved = float(wstar_relfrob_global.max()) < float(balanced_relfrob_global.max())
    result = {
        "seed": SEED, "k_primary": K_PRIMARY, "n_restarts": args.n_restarts, "smoke": args.smoke,
        "langs_order": LANGS,
        "global": {
            "w_star": dict(zip(LANGS, w_global.tolist())),
            "max_relfrob_wstar": float(val_global),
            "max_relfrob_balanced": float(balanced_relfrob_global.max()),
            "per_lang_relfrob_wstar": dict(zip(LANGS, wstar_relfrob_global.tolist())),
            "per_lang_relfrob_balanced": dict(zip(LANGS, balanced_relfrob_global.tolist())),
            "improvement_over_balanced_pct": 100.0 * (1.0 - float(val_global) / float(balanced_relfrob_global.max())),
            "n_cells": int(global_batch.shape[0]),
        },
        "per_layer": per_layer,
        "gate_passed": improved,
        "gate_note": ("w* improves max relFrob over balanced -- proceed to STEP 2/3" if improved
                      else "w* does NOT improve max relFrob over balanced -- STOP per task spec, do not proceed to STEP 2/3"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[robust-mix] wrote {out_path}")
    print(f"[robust-mix] GLOBAL w* = {dict(zip(LANGS, np.round(w_global, 4).tolist()))}")
    print(f"[robust-mix] GLOBAL max relFrob: balanced={balanced_relfrob_global.max():.4f} "
          f"w*={val_global:.4f} (improvement {result['global']['improvement_over_balanced_pct']:.1f}%)")
    print(f"[robust-mix] GATE: {'PASSED' if improved else 'FAILED'} -- {result['gate_note']}")


if __name__ == "__main__":
    main()
