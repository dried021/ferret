"""Phase 1 whitening-geometry extraction + pairwise comparison (09_부족한_
실험_정리.md item 3 "Whitening mechanism 직접 검증" / RQ-W1/W2 in the 0729
whitening-mechanism design note): direct evidence that different calibration
languages induce different whitening subspaces, not just different
downstream performance.

Reuses phase1_svd_scale.py's ALREADY-COMPUTED output (svd_scale_processed.pt,
one file per condition/seed, {layer: {"mlp.experts.<e>.<proj>": cholesky_L}})
for all 6 conditions the disagreement pipeline already defines
(disagreement_common.conditions_for("mixed_5lang") = EN/ZH/KO/SW/BN/mixed5)
x 3 seeds -- NO new GPU calibration run is needed; this script is pure CPU
post-processing over data already on disk.

Why eigenvectors from a Cholesky factor at all: process_scaling_matrix()
(phase1_svd_scale.py / D2MoE's get_scale.py) saves ONLY the Cholesky factor L
of the (regularized) activation covariance C = L L^T, not C itself or its
eigendecomposition -- the object D2MoE's SVD merge actually consumes. But
C's eigenvectors are recoverable from L alone: if L = U_L S_L V_L^T (SVD of
L), then C = L L^T = U_L S_L^2 U_L^T, i.e. L's LEFT singular vectors are
exactly C's eigenvectors, and L's squared singular values are exactly C's
eigenvalues. So a truncated SVD of L (torch.svd_lowrank, cheap, no need to
ever form the dense d x d covariance) gives the top-k object every geometry
metric below needs.

Storage/compute budget (checked live on this machine before writing this
script, 2026-07-29): /mnt/HDD is 99% full (83GB free) and the project's home
partition is 98% full (51GB free), both SHARED with other users' active
runs. This script therefore holds NO large intermediate artifact on disk at
all -- not even the top-k eigenvector caches the design note suggests
persisting. Instead, for each (layer, matrix_type) it: (1) opens all 18
condition x seed svd_scale_processed.pt files ONCE via mmap (torch.load(...,
mmap=True) -- confirmed live to keep resident memory in the hundreds-of-MB
range per file, not the full ~34GB on-disk size), (2) extracts a transient,
in-RAM top-k (eigvec, eigval) cache for just that layer/matrix_type's 64
experts x 18 nodes (~repro ~600MB peak), (3) computes every pairwise metric
analytically from that transient cache (see whitening_distance/
covariance_distance below -- both reduce to a k x k computation on already-
extracted eigenvectors/eigenvalues, no dense d x d matrix ever formed), (4)
appends the resulting small numeric rows to one CSV, then discards the
transient cache before moving to the next (layer, matrix_type). Peak
resident memory stays under ~1GB regardless of how many layers/conditions
have already been processed; total new disk footprint is the final CSV
(order 10 MB).

Dead/low-hit experts (phase1_svd_scale.py's routing_coverage_flags.json,
kind="dead_fallback_identity"/"low_hit_flagged", source="svd_scale") are
EXCLUDED from geometry comparison per (condition, seed, layer, expert): a
dead expert's Cholesky is an identity fallback (see phase1_svd_scale.py's
elementwise_scale_for_layer docstring) carrying no real activation-geometry
information, and including it would spuriously shrink cross-language
distances toward zero for pairs that happen to share a dead expert. Only
experts routed to on BOTH sides of a pair (common_experts) are compared; if
a pair shares zero eligible experts at a given layer/matrix_type, that row
is skipped and logged (should not happen in practice -- 64 experts, only a
handful ever flagged, see swahili_only/bengali_only's flags files).

Metrics computed per (layer, matrix_type, condition_a, seed_a, condition_b,
seed_b), averaged over common included experts, for k in {16, 32, 64}
(k=64 primary, matching the design note's "논문에는 k=64를 쓰되 appendix에
다른 k에서도 추세가 유지된다는 것을 보여주면 됩니다"; 16/32 as robustness,
all three free -- extracted from the SAME top-64 truncated SVD, no extra
compute):
  - mean_principal_angle_deg_k{K}: arccos of M=Ua^T Ub's singular values
  - max_principal_angle_deg_k{K}
  - projection_distance_k{K}: (1/sqrt2)*||P_a - P_b||_F, P=UU^T, computed
    from the closed form sqrt(k - ||M||_F^2) (no d x d projector ever formed)
  - whitening_reldist_k{K}: relative Frobenius distance between the top-k
    truncated whitening transforms W = U (Lambda+eps I)^{-1/2} U^T, computed
    analytically from M/eigval_a/eigval_b (see whitening_distance() docstring
    for the trace-identity derivation) -- again no d x d matrix ever formed
  - covariance_reldist_k{K}: same trick for the top-k truncated covariance
    C = U diag(eigval) U^T itself (secondary metric per design note §7)

Per-expert values are averaged (not summed) within a (layer, matrix_type,
pair) cell -- an expert-pooled proxy for the design note's "Level 1
layer-level pooled activation" (which would need re-running calibration with
un-gated, all-token hooks; not attempted here -- see script docstring epilogue
in 00_docs/09_부족한_실험_정리.md item 3 for why this is documented as a
scope decision, not silently substituted).

Usage:
    conda run -n d2moe_env python phase1_whitening_geometry.py [--smoke]
    (resumable: re-running skips any (layer, matrix_type) whose rows are
    already in the output CSV)
"""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from disagreement_common import conditions_for  # noqa: E402
from phase1_fisher import MOE_LAYERS  # noqa: E402 -- 1..27 for DeepSeek-MoE-16B

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
OUT_DIR = SCRIPT_DIR.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = conditions_for("mixed_5lang")  # EN, ZH, KO, SW, BN, mixed_5lang -- see disagreement_common.py
SEEDS = [0, 1, 2]
MATRIX_TYPES = ["gate_proj", "up_proj", "down_proj"]
N_EXPERTS = 64  # DeepSeek-MoE-16B routed experts (shared experts excluded, matches phase1_svd_scale.py)
K_LIST = [16, 32, 64]
K_PRIMARY = 64
Q_EXTRACT = 72  # oversampling above K_PRIMARY for torch.svd_lowrank accuracy (Halko et al. rule of thumb)
NITER = 2
EPS_WHITEN = 1e-5  # matches process_scaling_matrix's eigenvalue clamp (get_scale.py / phase1_svd_scale.py)

# Same threshold phase1_fisher.py/phase1_svd_scale.py/scan_disagreement_experts.py already use project-wide.
LOW_HIT_THRESHOLD = 5

CSV_COLUMNS = (
    ["layer", "matrix_type", "cond_a", "seed_a", "cond_b", "seed_b", "n_experts_common",
     "n_excluded_a", "n_excluded_b"]
    + [f"mean_angle_deg_k{k}" for k in K_LIST]
    + [f"max_angle_deg_k{k}" for k in K_LIST]
    + [f"proj_dist_k{k}" for k in K_LIST]
    + [f"whitening_reldist_k{k}" for k in K_LIST]
    + [f"cov_reldist_k{k}" for k in K_LIST]
)


def expert_key(e, matrix_type):
    return f"mlp.experts.{e}.{matrix_type}"


def load_excluded_experts(condition, seed):
    """{layer: set(expert_idx)} of experts to skip for THIS condition/seed --
    dead (identity-fallback) or low-hit (<LOW_HIT_THRESHOLD routed tokens)
    experts from phase1_svd_scale.py's own routing_coverage_flags.json.
    Returns an all-empty defaultdict-like dict if the file doesn't exist
    (English/Korean/Chinese conditions had zero flags -- see script docstring)."""
    path = RESULTS_ROOT / condition / f"seed{seed}" / "routing_coverage_flags.json"
    excluded = {}
    if not path.exists():
        return excluded
    data = json.loads(path.read_text())
    for layer_str, flags in data.items():
        layer = int(layer_str)
        for f in flags:
            if f.get("source") != "svd_scale":
                continue
            if f.get("kind") in ("dead_fallback_identity", "low_hit_flagged"):
                excluded.setdefault(layer, set()).add(int(f["expert"]))
    return excluded


def extract_topk(L_bf16, q=Q_EXTRACT, kmax=K_PRIMARY, niter=NITER):
    """L_bf16: [d,d] Cholesky factor (C = L L^T). Returns (U[:, :kmax] float32,
    eigval[:kmax] float32) -- eigval descending, matching C's top-kmax
    eigendecomposition (see module docstring for the SVD(L)-gives-eig(C)
    derivation)."""
    L = L_bf16.float()
    U, S, _V = torch.svd_lowrank(L, q=q, niter=niter)
    eigval = S[:kmax].pow(2)
    return U[:, :kmax].contiguous(), eigval.contiguous()


def build_node_cache(handle, layer, matrix_type, excluded_experts, n_experts=N_EXPERTS):
    """Stacks all non-excluded experts' (U, eigval) for one (condition, seed,
    layer, matrix_type) node into batched tensors for vectorized pairwise
    comparison. Returns (expert_ids: list[int], U_stack: [n,d,kmax],
    eigval_stack: [n,kmax])."""
    layer_dict = handle[layer]
    excl = excluded_experts.get(layer, set())
    expert_ids, Us, Es = [], [], []
    for e in range(n_experts):
        if e in excl:
            continue
        key = expert_key(e, matrix_type)
        if key not in layer_dict:
            continue
        U, eigval = extract_topk(layer_dict[key])
        expert_ids.append(e)
        Us.append(U)
        Es.append(eigval)
    if not expert_ids:
        return [], None, None
    return expert_ids, torch.stack(Us, dim=0), torch.stack(Es, dim=0)


def whitening_distance(M, eigval_a, eigval_b, k, eps=EPS_WHITEN):
    """Relative Frobenius distance between top-k truncated whitening
    transforms W = U (Lambda+eps I)^{-1/2} U^T, WITHOUT ever forming the
    dense d x d W matrices. Derivation: for symmetric W_a/W_b,
      tr(W_a^2)  = sum_i 1/(la_i+eps)                       (la = eigval_a)
      tr(W_b^2)  = sum_i 1/(lb_i+eps)
      tr(W_a W_b) = tr(Ua La^-1/2 (Ua^T Ub) Lb^-1/2 Ub^T Ua ... )
                  = sum_{ij} M_ij^2 * la_i^{-1/2} * lb_j^{-1/2}   (M = Ua^T Ub)
    ||W_a-W_b||_F^2 = tr(W_a^2)+tr(W_b^2)-2 tr(W_a W_b); relative distance
    divides by the mean of ||W_a||_F, ||W_b||_F (design note §7 "Primary:
    relative Frobenius distance"). Batched over the leading (expert) dim.
    M: [n,k,k], eigval_a/b: [n,k]."""
    la = eigval_a[:, :k] + eps
    lb = eigval_b[:, :k] + eps
    inv_sqrt_la = la.pow(-0.5)
    inv_sqrt_lb = lb.pow(-0.5)
    Mk = M[:, :k, :k]
    tr_wa2 = (1.0 / la).sum(dim=1)
    tr_wb2 = (1.0 / lb).sum(dim=1)
    tr_wawb = ((Mk ** 2) * inv_sqrt_la.unsqueeze(2) * inv_sqrt_lb.unsqueeze(1)).sum(dim=(1, 2))
    d2 = (tr_wa2 + tr_wb2 - 2 * tr_wawb).clamp(min=0)
    denom = 0.5 * (tr_wa2.sqrt() + tr_wb2.sqrt())
    return (d2.sqrt() / (denom + 1e-12))


def covariance_distance(M, eigval_a, eigval_b, k):
    """Same closed-form trick as whitening_distance(), applied to the top-k
    truncated covariance C = U diag(eigval) U^T itself (design note §7
    secondary metric) -- tr(C_a C_b) = sum_ij eigval_a_i * eigval_b_j * M_ij^2."""
    la = eigval_a[:, :k]
    lb = eigval_b[:, :k]
    Mk = M[:, :k, :k]
    tr_ca2 = (la ** 2).sum(dim=1)
    tr_cb2 = (lb ** 2).sum(dim=1)
    tr_cacb = ((Mk ** 2) * la.unsqueeze(2) * lb.unsqueeze(1)).sum(dim=(1, 2))
    d2 = (tr_ca2 + tr_cb2 - 2 * tr_cacb).clamp(min=0)
    denom = 0.5 * (tr_ca2.sqrt() + tr_cb2.sqrt())
    return (d2.sqrt() / (denom + 1e-12))


def pairwise_metrics(Ua, Ea, Ub, Eb, k_list=K_LIST):
    """Ua/Ub: [n,d,kmax], Ea/Eb: [n,kmax] for the SAME n common experts
    (caller aligns indices first). Returns a dict of k -> per-expert tensors
    (mean/max over experts computed by the caller, so per-expert values stay
    available if a finer-grained analysis wants them later)."""
    M = torch.bmm(Ua.transpose(1, 2), Ub)  # [n, kmax, kmax], full kmax so any k<=kmax slices it
    out = {}
    for k in k_list:
        Mk = M[:, :k, :k]
        sigma = torch.linalg.svdvals(Mk).clamp(max=1.0, min=-1.0)  # [n,k]
        theta_deg = torch.rad2deg(torch.arccos(sigma))
        proj_dist = (k - (sigma ** 2).sum(dim=1)).clamp(min=0).sqrt()
        out[k] = {
            "mean_angle": theta_deg.mean(dim=1),
            "max_angle": theta_deg.max(dim=1).values,
            "proj_dist": proj_dist,
            "whitening_reldist": whitening_distance(M, Ea, Eb, k),
            "cov_reldist": covariance_distance(M, Ea, Eb, k),
        }
    return out


def align_common(ids_a, Ua, Ea, ids_b, Ub, Eb):
    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        return [], None, None, None, None
    idx_a = torch.tensor([ids_a.index(e) for e in common], dtype=torch.long)
    idx_b = torch.tensor([ids_b.index(e) for e in common], dtype=torch.long)
    return common, Ua[idx_a], Ea[idx_a], Ub[idx_b], Eb[idx_b]


def load_done_cells(csv_path):
    """Resumability: {(layer, matrix_type)} already fully written."""
    if not csv_path.exists():
        return set()
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["layer", "matrix_type"])
    return set(zip(df["layer"].tolist(), df["matrix_type"].tolist()))


def append_rows(csv_path, rows):
    import pandas as pd
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                         help="2 layers, 8 experts, gate_proj only -- exercises the full pipeline in ~1 min")
    parser.add_argument("--num-threads", type=int, default=6,
                         help="torch.set_num_threads -- kept modest by default since this machine runs other "
                              "users'/this project's own GPU jobs concurrently (see module docstring)")
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)

    layers = MOE_LAYERS[:2] if args.smoke else MOE_LAYERS
    matrix_types = ["gate_proj"] if args.smoke else MATRIX_TYPES
    n_experts = 8 if args.smoke else N_EXPERTS
    csv_path = OUT_DIR / ("whitening_geometry_pairs_smoke.csv" if args.smoke else "whitening_geometry_pairs.csv")

    nodes = [(c, s) for c in CONDITIONS for s in SEEDS]
    print(f"[whitening-geom] {len(nodes)} condition x seed nodes, {len(layers)} layers, "
          f"{len(matrix_types)} matrix types, n_experts={n_experts}, smoke={args.smoke}")

    print("[whitening-geom] opening svd_scale_processed.pt for all nodes via mmap (one-time ~30s/file)...")
    handles, excluded_by_node = {}, {}
    for cond, seed in nodes:
        path = RESULTS_ROOT / cond / f"seed{seed}" / "svd_scale_processed.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run phase1_svd_scale.py --condition {cond} --seed {seed} first")
        t0 = time.time()
        handles[(cond, seed)] = torch.load(path, map_location="cpu", mmap=True)
        excluded_by_node[(cond, seed)] = load_excluded_experts(cond, seed)
        print(f"[whitening-geom]   opened {cond}/seed{seed} in {time.time()-t0:.1f}s")

    done_cells = load_done_cells(csv_path)
    if done_cells:
        print(f"[whitening-geom] resuming: {len(done_cells)} (layer, matrix_type) cells already done, skipping those")

    pairs = list(itertools.combinations(nodes, 2))
    total_cells = len(layers) * len(matrix_types)
    cell_i = 0
    for layer in layers:
        for matrix_type in matrix_types:
            cell_i += 1
            if (layer, matrix_type) in done_cells:
                print(f"[whitening-geom] [{cell_i}/{total_cells}] layer={layer} {matrix_type}: already done, skipping")
                continue
            t0 = time.time()
            node_cache = {}
            n_excluded = {}
            for cond, seed in nodes:
                excl = excluded_by_node[(cond, seed)]
                ids, Us, Es = build_node_cache(handles[(cond, seed)], layer, matrix_type, excl, n_experts)
                node_cache[(cond, seed)] = (ids, Us, Es)
                n_excluded[(cond, seed)] = len(excl.get(layer, set()))

            rows = []
            for (ca, sa), (cb, sb) in pairs:
                ids_a, Ua, Ea = node_cache[(ca, sa)]
                ids_b, Ub, Eb = node_cache[(cb, sb)]
                if Ua is None or Ub is None:
                    print(f"[whitening-geom] WARNING layer={layer} {matrix_type} {ca}/s{sa} vs {cb}/s{sb}: "
                          f"one side has zero eligible experts, skipping")
                    continue
                common, Ua_al, Ea_al, Ub_al, Eb_al = align_common(ids_a, Ua, Ea, ids_b, Ub, Eb)
                if not common:
                    print(f"[whitening-geom] WARNING layer={layer} {matrix_type} {ca}/s{sa} vs {cb}/s{sb}: "
                          f"zero common eligible experts, skipping")
                    continue
                metrics = pairwise_metrics(Ua_al, Ea_al, Ub_al, Eb_al)
                row = {
                    "layer": layer, "matrix_type": matrix_type,
                    "cond_a": ca, "seed_a": sa, "cond_b": cb, "seed_b": sb,
                    "n_experts_common": len(common),
                    "n_excluded_a": n_excluded[(ca, sa)], "n_excluded_b": n_excluded[(cb, sb)],
                }
                for k in K_LIST:
                    row[f"mean_angle_deg_k{k}"] = metrics[k]["mean_angle"].mean().item()
                    row[f"max_angle_deg_k{k}"] = metrics[k]["max_angle"].mean().item()
                    row[f"proj_dist_k{k}"] = metrics[k]["proj_dist"].mean().item()
                    row[f"whitening_reldist_k{k}"] = metrics[k]["whitening_reldist"].mean().item()
                    row[f"cov_reldist_k{k}"] = metrics[k]["cov_reldist"].mean().item()
                rows.append(row)

            append_rows(csv_path, rows)
            dt = time.time() - t0
            print(f"[whitening-geom] [{cell_i}/{total_cells}] layer={layer} {matrix_type}: "
                  f"{len(rows)}/{len(pairs)} pairs written in {dt:.1f}s")

    print(f"[whitening-geom] done, wrote {csv_path}")


if __name__ == "__main__":
    main()
