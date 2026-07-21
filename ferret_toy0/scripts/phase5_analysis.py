"""Phase 5: heterogeneity analysis, equal-budget baselines, bootstrap,
figures, and Gate 1-3 decisions. Uses the activation-aware rank-sweep
results (Phase 4b), which is the "retry" decomposition per Section 19 after
the naive weight-SVD (Phase 4) showed a near-flat singular spectrum.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/home/minjeong/project/FERRET/ferret_toy0/results"
FIG_DIR = "/home/minjeong/project/FERRET/ferret_toy0/figures"
RANKS = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 480, 512]
PRIMARY_EPS = 0.05
SECONDARY_EPS = [0.01, 0.02, 0.10]
N_BOOT = 2000
N_MIX_SEEDS = 10
RNG = np.random.default_rng(0)


def load():
    tok = pd.read_parquet(f"{RESULTS_DIR}/token_rank_errors_actaware.parquet")
    oracle = pd.read_parquet(f"{RESULTS_DIR}/oracle_ranks_actaware.parquet")
    return tok, oracle


def err_matrix(tok_g):
    """Return (N, n_ranks) numpy array of relative L2 errors, ranks list actually present."""
    ranks = [r for r in RANKS if f"err_rel_r{r}" in tok_g.columns and tok_g[f"err_rel_r{r}"].notna().all()]
    M = tok_g[[f"err_rel_r{r}" for r in ranks]].to_numpy()
    return M, ranks


def heterogeneity_stats(oracle_g, eps_col, tag):
    vc = oracle_g[eps_col].value_counts()
    total = len(oracle_g)
    fail_frac = (oracle_g[eps_col] == -1).mean()
    non_fail = oracle_g[oracle_g[eps_col] != -1][eps_col]
    dominant_frac = vc.max() / total if len(vc) else float("nan")
    occupied_bins = (vc[vc.index != -1] > 0).sum()
    bins_ge_10pct = ((vc[vc.index != -1] / total) >= 0.10).sum()
    probs = (vc[vc.index != -1] / (total - vc.get(-1, 0))).to_numpy() if (total - vc.get(-1, 0)) > 0 else np.array([])
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if len(probs) else float("nan")
    q25, q75 = (non_fail.quantile(0.25), non_fail.quantile(0.75)) if len(non_fail) else (float("nan"),) * 2
    return {
        "tag": tag, "eps": eps_col, "n": total, "fail_frac": fail_frac,
        "dominant_rank": int(vc.idxmax()) if len(vc) else None,
        "dominant_frac": dominant_frac, "occupied_bins": int(occupied_bins),
        "bins_ge_10pct": int(bins_ge_10pct), "rank_entropy_bits": entropy,
        "median_oracle_rank": float(non_fail.median()) if len(non_fail) else float("nan"),
        "mean_oracle_rank": float(non_fail.mean()) if len(non_fail) else float("nan"),
        "q25_oracle_rank": float(q25), "q75_oracle_rank": float(q75),
    }


def bootstrap_reduction(sample_ids, err_static, err_oracle, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sample_ids)
    diffs = []
    for _ in range(n_boot):
        boot_samples = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(sample_ids == s)[0] for s in boot_samples])
        m_static = err_static[idx].mean()
        m_oracle = err_oracle[idx].mean()
        diffs.append(m_static - m_oracle)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    tok, oracle = load()
    eps_col = f"oracle_rank_eps{PRIMARY_EPS}"

    het_rows = []
    for eps in [PRIMARY_EPS] + SECONDARY_EPS:
        col = f"oracle_rank_eps{eps}"
        for tag, g in oracle.groupby("tag"):
            het_rows.append(heterogeneity_stats(g, col, tag))
    het_df = pd.DataFrame(het_rows)
    het_df.to_csv(f"{RESULTS_DIR}/heterogeneity_stats.csv", index=False)

    # ---------- per-expert Gate1 / Gate3 numbers at PRIMARY_EPS ----------
    gate_rows = []
    curve_rows = []  # for quality-compute Pareto figure
    mixture_rows = []
    per_tag_full = {}

    for tag, tok_g in tok.groupby("tag"):
        tok_g = tok_g.reset_index(drop=True)
        M, ranks = err_matrix(tok_g)
        oracle_g = oracle[oracle["tag"] == tag].reset_index(drop=True)
        r_star = oracle_g[eps_col].to_numpy()
        assert (r_star != -1).all(), "unexpected FAIL at primary epsilon"

        rank_to_idx = {r: i for i, r in enumerate(ranks)}
        oracle_err = M[np.arange(len(M)), [rank_to_idx[r] for r in r_star]]
        oracle_avg_rank = r_star.mean()

        # best expert-wise static rank at <= oracle average rank (largest such grid rank)
        candidates = [r for r in ranks if r <= oracle_avg_rank]
        r_static = max(candidates) if candidates else min(ranks)
        static_err = M[:, rank_to_idx[r_static]]

        sample_ids = tok_g["sample_id"].to_numpy()
        mean_diff, ci_lo, ci_hi = bootstrap_reduction(sample_ids, static_err, oracle_err)

        rel_reduction = (static_err.mean() - oracle_err.mean()) / static_err.mean()
        viol_static = (static_err > PRIMARY_EPS).mean()
        viol_oracle = (oracle_err > PRIMARY_EPS).mean()
        viol_reduction = (viol_static - viol_oracle) / viol_static if viol_static > 0 else float("nan")

        # random equal-budget mixture bracketing oracle_avg_rank
        below = [r for r in ranks if r <= oracle_avg_rank]
        above = [r for r in ranks if r >= oracle_avg_rank]
        r_lo = max(below) if below else min(ranks)
        r_hi = min(above) if above else max(ranks)
        if r_lo == r_hi:
            mix_errs = [M[:, rank_to_idx[r_lo]].mean()] * N_MIX_SEEDS
        else:
            w = (r_hi - oracle_avg_rank) / (r_hi - r_lo)  # weight on r_lo
            mix_errs = []
            for s in range(N_MIX_SEEDS):
                rng = np.random.default_rng(1000 + s)
                assign_lo = rng.random(len(tok_g)) < w
                e = np.where(assign_lo, M[:, rank_to_idx[r_lo]], M[:, rank_to_idx[r_hi]])
                mix_errs.append(e.mean())
        mixture_mean = float(np.mean(mix_errs))
        mixture_std = float(np.std(mix_errs))

        gate_rows.append({
            "tag": tag, "n_tokens": len(tok_g), "R_max": max(ranks),
            "oracle_avg_rank": oracle_avg_rank, "oracle_avg_rank_pct_of_max": oracle_avg_rank / max(ranks),
            "r_static": r_static, "r_static_pct_of_max": r_static / max(ranks),
            "mean_err_static": static_err.mean(), "median_err_static": np.median(static_err),
            "mean_err_oracle": oracle_err.mean(), "median_err_oracle": np.median(oracle_err),
            "mean_err_random_mixture": mixture_mean, "std_err_random_mixture": mixture_std,
            "rel_error_reduction_vs_static": rel_reduction,
            "violation_rate_static": viol_static, "violation_rate_oracle": viol_oracle,
            "violation_rate_reduction": viol_reduction,
            "bootstrap_mean_diff": mean_diff, "bootstrap_ci95_lo": ci_lo, "bootstrap_ci95_hi": ci_hi,
            "gate3_pass": bool(ci_lo > 0 and (rel_reduction >= 0.10 or (not np.isnan(viol_reduction) and viol_reduction >= 0.20))),
        })

        for r in ranks:
            curve_rows.append({"tag": tag, "policy": "expert_static", "rank": r,
                                "mean_err": M[:, rank_to_idx[r]].mean(),
                                "median_err": np.median(M[:, rank_to_idx[r]])})
        curve_rows.append({"tag": tag, "policy": "oracle_token_wise", "rank": oracle_avg_rank,
                            "mean_err": oracle_err.mean(), "median_err": np.median(oracle_err)})
        curve_rows.append({"tag": tag, "policy": "random_equal_budget_mixture", "rank": oracle_avg_rank,
                            "mean_err": mixture_mean, "median_err": mixture_mean})

        per_tag_full[tag] = {"M": M, "ranks": ranks, "sample_ids": sample_ids}

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(f"{RESULTS_DIR}/static_baselines.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(f"{RESULTS_DIR}/quality_compute_curve.csv", index=False)

    # ---------- Gate 1 ----------
    gate1_pass_count = 0
    gate1_rows = []
    for tag, sub in het_df[het_df["eps"] == eps_col].iterrows():
        pass
    for tag in gate_df["tag"]:
        row = gate_df[gate_df["tag"] == tag].iloc[0]
        R_max = row["R_max"]
        med_at_half = None
        M, ranks = per_tag_full[tag]["M"], per_tag_full[tag]["ranks"]
        half_rank_candidates = [r for r in ranks if r <= 0.5 * R_max]
        if half_rank_candidates:
            r_half = max(half_rank_candidates)
            idx = ranks.index(r_half)
            med = float(np.median(M[:, idx]))
            p90 = float(np.percentile(M[:, idx], 90))
        else:
            med, p90 = float("nan"), float("nan")
        passed = (med is not None) and med < 0.05 and p90 < 0.10
        gate1_rows.append({"tag": tag, "r_half": r_half if half_rank_candidates else None,
                            "median_err_at_half_rank": med, "p90_err_at_half_rank": p90, "gate1_pass": passed})
        gate1_pass_count += int(passed)
    gate1_df = pd.DataFrame(gate1_rows)
    gate1_df.to_csv(f"{RESULTS_DIR}/gate1_lowrank_feasibility.csv", index=False)

    # ---------- Gate 2 ----------
    het_primary = het_df[het_df["eps"] == eps_col].copy()
    het_primary["gate2_pass"] = (het_primary["dominant_frac"] < 0.80) & (het_primary["occupied_bins"] >= 3) & (het_primary["bins_ge_10pct"] >= 2)
    het_primary.to_csv(f"{RESULTS_DIR}/gate2_heterogeneity.csv", index=False)

    # ---------- summary ----------
    summary = {
        "primary_epsilon": PRIMARY_EPS,
        "gate1_pass_count": int(gate1_pass_count), "gate1_total": len(gate1_df),
        "gate1_overall_pass": gate1_pass_count >= 3,
        "gate2_pass_count": int(het_primary["gate2_pass"].sum()), "gate2_total": len(het_primary),
        "gate2_overall_pass": bool(het_primary["gate2_pass"].sum() >= 3),
        "gate3_pass_count": int(gate_df["gate3_pass"].sum()), "gate3_total": len(gate_df),
        "gate3_overall_pass": bool(gate_df["gate3_pass"].sum() >= 3),
    }
    with open(f"{RESULTS_DIR}/gate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(gate_df.to_string())
    print(het_primary.to_string())

    # ================= FIGURES =================
    import os
    os.makedirs(FIG_DIR, exist_ok=True)

    # Figure 1: rank vs median/p90 error per expert
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag in per_tag_full:
        M, ranks = per_tag_full[tag]["M"], per_tag_full[tag]["ranks"]
        med = [np.median(M[:, i]) for i in range(len(ranks))]
        ax.plot(ranks, med, marker="o", label=f"{tag} (median)")
    ax.axhline(PRIMARY_EPS, color="gray", linestyle="--", label=f"eps={PRIMARY_EPS}")
    ax.set_xlabel("rank r"); ax.set_ylabel("relative L2 output error")
    ax.set_title("Figure 1: Rank vs median output error (down_proj, activation-aware SVD)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rank_error_curve.png", dpi=150)
    plt.close(fig)

    # Figure 2: oracle rank histograms conditioned on layer/expert
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (tag, g) in zip(axes.flat, oracle.groupby("tag")):
        vals = g[eps_col]
        vals = vals[vals != -1]
        ax.hist(vals, bins=sorted(vals.unique()))
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("oracle rank"); ax.set_ylabel("count")
    fig.suptitle(f"Figure 2: Oracle rank histograms (eps={PRIMARY_EPS}, activation-aware SVD)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/oracle_rank_histograms.png", dpi=150)
    plt.close(fig)

    # Figure 3: quality-compute Pareto curve
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, tag in zip(axes.flat, per_tag_full.keys()):
        sub = curve_df[curve_df["tag"] == tag]
        es = sub[sub.policy == "expert_static"].sort_values("rank")
        ax.plot(es["rank"], es["mean_err"], marker="o", label="expert-wise static")
        orow = sub[sub.policy == "oracle_token_wise"].iloc[0]
        ax.scatter([orow["rank"]], [orow["mean_err"]], color="red", zorder=5, label="oracle token-wise", marker="*", s=150)
        mrow = sub[sub.policy == "random_equal_budget_mixture"].iloc[0]
        ax.scatter([mrow["rank"]], [mrow["mean_err"]], color="green", zorder=5, label="random equal-budget mix", marker="s", s=60)
        ax.set_title(tag, fontsize=9); ax.set_xlabel("average rank"); ax.set_ylabel("mean relative error")
        ax.legend(fontsize=7)
    fig.suptitle("Figure 3: Quality-compute comparison at matched average rank")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/quality_compute_curve.png", dpi=150)
    plt.close(fig)

    # Figure 4: oracle gain over best expert-static baseline
    fig, ax = plt.subplots(figsize=(7, 5))
    tags = gate_df["tag"].tolist()
    reductions = gate_df["rel_error_reduction_vs_static"].to_numpy() * 100
    ci_lo = gate_df["bootstrap_ci95_lo"].to_numpy()
    ci_hi = gate_df["bootstrap_ci95_hi"].to_numpy()
    ax.bar(tags, reductions, color=["tab:green" if p else "tab:red" for p in gate_df["gate3_pass"]])
    ax.set_ylabel("relative mean-error reduction vs expert-static (%)")
    ax.set_title("Figure 4: Oracle gain over best expert-static baseline (green = Gate3 pass)")
    ax.axhline(10, color="gray", linestyle="--", label="10% target")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/per_expert_gain.png", dpi=150)
    plt.close(fig)

    print("figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
