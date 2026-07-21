"""Phase 4b: rank sweep using the activation-aware (whitened) SVD factors,
evaluated on the held-out EVAL traces (never used to build the
decomposition -- Section 4's calibration/evaluation split). Uses a finer
rank grid than Phase 4 since the naive weight-SVD result showed error mass
concentrated near full rank; we want to resolve the curve's shape properly
this time, from Section 8's "insert additional ranks" guidance.
"""
import json
import torch

TRACE_DIR = "/home/minjeong/project/FERRET/ferret_toy0/traces"
FACTOR_DIR = "/home/minjeong/project/FERRET/ferret_toy0/factors"
RESULTS_DIR = "/home/minjeong/project/FERRET/ferret_toy0/results"

RANKS = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 480, 512]
THRESHOLDS = [0.01, 0.02, 0.05, 0.10]
DELTA = 1e-6


def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(f"{TRACE_DIR}/manifest.json") as f:
        manifest = json.load(f)

    device = "cuda:0"
    all_token_rows = []
    all_oracle_rows = []

    for tag, info in manifest.items():
        z = torch.load(f"{TRACE_DIR}/{tag}_z.pt", weights_only=True).float().to(device)
        y = torch.load(f"{TRACE_DIR}/{tag}_y.pt", weights_only=True).float().to(device)
        meta = torch.load(f"{TRACE_DIR}/{tag}_metadata.pt", weights_only=True)
        svd = torch.load(f"{FACTOR_DIR}/{tag}_downproj_svd_actaware.pt", weights_only=True)

        U = svd["U"].float().to(device)
        S = svd["S"].float().to(device)
        Vh = svd["Vh"].float().to(device)
        Linv = svd["Linv"].float().to(device)
        R_max = svd["R"]
        ranks = [r for r in RANKS if r <= R_max]

        z_whitened = z @ Linv.T                      # (N, d_in) whitened input
        v_proj = z_whitened @ Vh.T                    # (N, R)
        y_norm = y.norm(dim=-1)

        N = z.shape[0]
        err_rel = torch.zeros(N, len(ranks), device=device)
        err_cos = torch.zeros(N, len(ranks), device=device)
        for ri, r in enumerate(ranks):
            y_hat = (v_proj[:, :r] * S[:r]) @ U[:, :r].T
            diff = y - y_hat
            err_rel[:, ri] = diff.norm(dim=-1) / (y_norm + DELTA)
            cos = (y * y_hat).sum(-1) / (y_norm * y_hat.norm(dim=-1) + DELTA)
            err_cos[:, ri] = 1 - cos

        err_rel_cpu, err_cos_cpu = err_rel.cpu(), err_cos.cpu()
        sample_id, token_pos = meta["sample_id"], meta["token_pos"]
        router_weight = meta["router_weight"].float()
        router_margin, router_entropy = meta["router_margin"], meta["router_entropy"]
        layer_id, expert_id = info["layer_id"], info["expert_id"]

        for t in range(N):
            row = {"tag": tag, "layer_id": layer_id, "expert_id": expert_id,
                   "sample_id": int(sample_id[t]), "token_pos": int(token_pos[t]),
                   "router_weight": float(router_weight[t]),
                   "router_margin": float(router_margin[t]),
                   "router_entropy": float(router_entropy[t])}
            for ri, r in enumerate(ranks):
                row[f"err_rel_r{r}"] = float(err_rel_cpu[t, ri])
                row[f"err_cos_r{r}"] = float(err_cos_cpu[t, ri])
            all_token_rows.append(row)

            oracle_row = {"tag": tag, "layer_id": layer_id, "expert_id": expert_id,
                           "sample_id": int(sample_id[t]), "token_pos": int(token_pos[t])}
            for eps in THRESHOLDS:
                r_star = -1
                for ri, r in enumerate(ranks):
                    if err_rel_cpu[t, ri].item() <= eps:
                        r_star = r
                        break
                oracle_row[f"oracle_rank_eps{eps}"] = r_star
            all_oracle_rows.append(oracle_row)

        med = {r: round(err_rel_cpu[:, ri].median().item(), 4) for ri, r in enumerate(ranks)}
        print(f"{tag}: N={N} R_max={R_max}")
        print("  median err by rank:", med)

    import pandas as pd
    df_tok = pd.DataFrame(all_token_rows)
    df_oracle = pd.DataFrame(all_oracle_rows)
    df_tok.to_parquet(f"{RESULTS_DIR}/token_rank_errors_actaware.parquet")
    df_oracle.to_parquet(f"{RESULTS_DIR}/oracle_ranks_actaware.parquet")
    with open(f"{RESULTS_DIR}/rank_grid_actaware.json", "w") as f:
        json.dump({"ranks": RANKS}, f, indent=2)
    print("saved", len(df_tok), "rows")


if __name__ == "__main__":
    main()
