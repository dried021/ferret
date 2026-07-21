"""Phase 4: local rank sweep (offline replay, no full model on GPU).

For each selected (layer, expert) pair, load its saved down_proj traces
(z_t, y_t) and SVD factors, then compute the rank-prefix approximation
error for every token x every candidate rank in the grid (config's
rank.r_compress + rank.r_diagnostic). Also derives the per-token oracle
rank at each threshold in thresholds_epsilon (methodology section 10).
"""
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment_config.yaml"
TRACE_DIR = ROOT / "traces"
FACTOR_DIR = ROOT / "factors"
RESULTS_DIR = ROOT / "results"
DELTA = 1e-6


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(TRACE_DIR / "manifest.json") as f:
        manifest = json.load(f)

    rank_cfg = cfg["rank"]
    full_grid = sorted(set(rank_cfg["r_compress"]) | set(rank_cfg["r_diagnostic"]))
    thresholds = cfg["thresholds_epsilon"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    all_token_rows = []
    all_oracle_rows = []
    grid_info = {}

    for tag, info in manifest.items():
        z = torch.load(TRACE_DIR / f"{tag}_z.pt", weights_only=True).float().to(device)   # (N, d_in)
        y = torch.load(TRACE_DIR / f"{tag}_y.pt", weights_only=True).float().to(device)   # (N, d_out)
        meta = torch.load(TRACE_DIR / f"{tag}_metadata.pt", weights_only=True)
        svd = torch.load(FACTOR_DIR / f"{tag}_downproj_svd.pt", weights_only=True)

        U = svd["U"].float().to(device)     # (d_out, R)
        S = svd["S"].float().to(device)     # (R,)
        Vh = svd["Vh"].float().to(device)   # (R, d_in)
        R_max = svd["R"]

        ranks = [r for r in full_grid if r <= R_max]
        grid_info[tag] = ranks

        v_proj = z @ Vh.T                          # (N, R)  == V^T z for every token, all ranks at once
        y_norm = y.norm(dim=-1)                     # (N,)

        N = z.shape[0]
        err_rel = torch.zeros(N, len(ranks), device=device)
        err_cos = torch.zeros(N, len(ranks), device=device)

        for ri, r in enumerate(ranks):
            y_hat = (v_proj[:, :r] * S[:r]) @ U[:, :r].T     # (N, d_out)
            diff = y - y_hat
            err_rel[:, ri] = diff.norm(dim=-1) / (y_norm + DELTA)
            cos = (y * y_hat).sum(-1) / (y_norm * y_hat.norm(dim=-1) + DELTA)
            err_cos[:, ri] = 1 - cos

        err_rel_cpu = err_rel.cpu()
        err_cos_cpu = err_cos.cpu()

        sample_id = meta["sample_id"]
        token_pos = meta["token_pos"]
        router_weight = meta["router_weight"].float()
        router_margin = meta["router_margin"]
        router_entropy = meta["router_entropy"]

        layer_id, expert_id = info["layer_id"], info["expert_id"]

        for t in range(N):
            row = {
                "tag": tag, "layer_id": layer_id, "expert_id": expert_id,
                "sample_id": int(sample_id[t]), "token_pos": int(token_pos[t]),
                "router_weight": float(router_weight[t]),
                "router_margin": float(router_margin[t]),
                "router_entropy": float(router_entropy[t]),
            }
            for ri, r in enumerate(ranks):
                row[f"err_rel_r{r}"] = float(err_rel_cpu[t, ri])
                row[f"err_cos_r{r}"] = float(err_cos_cpu[t, ri])
            all_token_rows.append(row)

            oracle_row = {"tag": tag, "layer_id": layer_id, "expert_id": expert_id,
                           "sample_id": int(sample_id[t]), "token_pos": int(token_pos[t])}
            for eps in thresholds:
                r_star = -1
                for ri, r in enumerate(ranks):
                    if err_rel_cpu[t, ri].item() <= eps:
                        r_star = r
                        break
                oracle_row[f"oracle_rank_eps{eps}"] = r_star
            all_oracle_rows.append(oracle_row)

        primary_eps = cfg["primary_epsilon"]
        fail_rate = sum(1 for row in all_oracle_rows[-N:] if row[f"oracle_rank_eps{primary_eps}"] == -1) / N
        print(f"{tag}: N={N} R_max={R_max} ranks={ranks} FAIL@eps={primary_eps}: {fail_rate:.3f} "
              f"median_err_full_rank={err_rel_cpu[:, -1].median().item():.5f}")

    import pandas as pd
    df_tok = pd.DataFrame(all_token_rows)
    df_oracle = pd.DataFrame(all_oracle_rows)
    df_tok.to_parquet(RESULTS_DIR / "token_rank_errors.parquet")
    df_oracle.to_parquet(RESULTS_DIR / "oracle_ranks.parquet")
    with open(RESULTS_DIR / "rank_grid.json", "w") as f:
        json.dump(grid_info, f, indent=2)
    print("saved", len(df_tok), "token rows and", len(df_oracle), "oracle rows")


if __name__ == "__main__":
    main()
