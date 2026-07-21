"""Phase 6: limited full-model validation (Gate 4).

Picks ONE (layer, expert) pair (the one with the largest Gate-3 gain:
layer19_expert14) and re-runs the *actual* model forward pass on the 16
eval sequences four times per sequence, substituting only that expert's
down_proj computation:

  1. original          -- untouched model (reference)
  2. expert_static      -- fixed rank r_static (=480, from Phase 5) for
                            every token routed to the target expert
  3. oracle_token_wise  -- per-token oracle rank (eps=0.05, from Phase 4b),
                            looked up by (sample_id, token_position)
  4. random_equal_budget-- same fixed mixture-of-two-ranks policy as
                            Phase 5's baseline (one fixed seed)

All other experts / layers run with their exact original weights. Metrics:
delta NLL vs original, mean logits KL divergence vs original, top-1 token
agreement vs original.
"""
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from moe_hooks import get_moe_layers

MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
TRACE_DIR = "/home/minjeong/project/FERRET/ferret_toy0/traces"
FACTOR_DIR = "/home/minjeong/project/FERRET/ferret_toy0/factors"
RESULTS_DIR = "/home/minjeong/project/FERRET/ferret_toy0/results"

TARGET_TAG = "layer19_expert14"
TARGET_LAYER = 19
TARGET_EXPERT = 14
R_STATIC = 480
PRIMARY_EPS = 0.05


def build_oracle_lookup():
    oracle = pd.read_parquet(f"{RESULTS_DIR}/oracle_ranks_actaware.parquet")
    g = oracle[oracle["tag"] == TARGET_TAG]
    return {(int(s), int(p)): int(r) for s, p, r in zip(g.sample_id, g.token_pos, g[f"oracle_rank_eps{PRIMARY_EPS}"])}


def install_rank_substitution(experts_module, target_expert, U, S, Vh, Linv, rank_fn):
    """rank_fn(token_idx_tensor) -> LongTensor of ranks (one per routed token),
    using the *current sequence context* closed over by the caller via rank_fn."""
    orig_forward = experts_module.forward

    def patched_forward(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = experts_module.num_experts
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            z = experts_module.act_fn(gate) * up

            if expert_idx == target_expert:
                ranks = rank_fn(token_idx)  # (n_hit,) LongTensor
                z32 = z.float()
                zw = z32 @ Linv.T
                y = torch.zeros(z32.shape[0], U.shape[0], device=z.device, dtype=torch.float32)
                for r in ranks.unique().tolist():
                    sel = (ranks == r)
                    v = zw[sel] @ Vh[:r, :].T
                    y[sel] = (v * S[:r]) @ U[:, :r].T
                y = y.to(z.dtype)
            else:
                y = F.linear(z, experts_module.down_proj[expert_idx])

            weighted = y * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
        return final_hidden_states

    experts_module.forward = patched_forward

    class Handle:
        def remove(self):
            experts_module.forward = orig_forward
    return Handle()


def main():
    device = "cuda:0"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device).eval()

    svd = torch.load(f"{FACTOR_DIR}/{TARGET_TAG}_downproj_svd_actaware.pt", weights_only=True)
    U = svd["U"].float().to(device)
    S = svd["S"].float().to(device)
    Vh = svd["Vh"].float().to(device)
    Linv = svd["Linv"].float().to(device)

    oracle_lookup = build_oracle_lookup()
    seqs = torch.load(f"{TRACE_DIR}/_sequences.pt", weights_only=False)["eval"]

    bsm = model.model.layers[TARGET_LAYER].block_sparse_moe

    # mixture policy: same r_lo/r_hi/weight logic as Phase5, one fixed seed
    baselines = pd.read_csv(f"{RESULTS_DIR}/static_baselines.csv")
    row = baselines[baselines.tag == TARGET_TAG].iloc[0]
    oracle_avg_rank = row["oracle_avg_rank"]
    ranks_grid = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 480, 512]
    below = [r for r in ranks_grid if r <= oracle_avg_rank]
    above = [r for r in ranks_grid if r >= oracle_avg_rank]
    r_lo, r_hi = max(below), min(above)
    w_lo = (r_hi - oracle_avg_rank) / (r_hi - r_lo) if r_hi != r_lo else 1.0
    mix_rng = np.random.default_rng(1000)  # seed 0 of the N_MIX_SEEDS used in Phase5

    results = []
    cur_sample_id = [-1]

    def make_rank_fn(policy):
        def rank_fn(token_idx):
            n = token_idx.numel()
            if policy == "expert_static":
                return torch.full((n,), R_STATIC, dtype=torch.long, device=token_idx.device)
            if policy == "oracle_token_wise":
                sid = cur_sample_id[0]
                r = [oracle_lookup.get((sid, int(p)), R_STATIC) for p in token_idx.tolist()]
                return torch.tensor(r, dtype=torch.long, device=token_idx.device)
            if policy == "random_equal_budget":
                lo_mask = mix_rng.random(n) < w_lo
                r = np.where(lo_mask, r_lo, r_hi)
                return torch.tensor(r, dtype=torch.long, device=token_idx.device)
            raise ValueError(policy)
        return rank_fn

    with torch.inference_mode():
        for si, ids in enumerate(seqs):
            cur_sample_id[0] = si
            ids_b = ids.unsqueeze(0).to(device)
            labels = ids_b.clone()

            out_orig = model(input_ids=ids_b, labels=labels, use_cache=False)
            logits_orig = out_orig.logits.float()
            loss_orig = out_orig.loss.item()
            probs_orig = torch.softmax(logits_orig, dim=-1)
            top1_orig = probs_orig.argmax(-1)

            row_res = {"sample_id": si, "nll_original": loss_orig}
            for policy in ["expert_static", "oracle_token_wise", "random_equal_budget"]:
                handle = install_rank_substitution(bsm.experts, TARGET_EXPERT, U, S, Vh, Linv, make_rank_fn(policy))
                out = model(input_ids=ids_b, labels=labels, use_cache=False)
                handle.remove()

                logits = out.logits.float()
                loss = out.loss.item()
                probs = torch.softmax(logits, dim=-1)
                kl = (probs_orig * (probs_orig.clamp_min(1e-12).log() - probs.clamp_min(1e-12).log())).sum(-1).mean().item()
                top1 = probs.argmax(-1)
                agree = (top1 == top1_orig).float().mean().item()

                row_res[f"nll_{policy}"] = loss
                row_res[f"delta_nll_{policy}"] = loss - loss_orig
                row_res[f"kl_{policy}"] = kl
                row_res[f"top1_agree_{policy}"] = agree

            results.append(row_res)
            print(si, row_res)

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/phase6_model_validation.csv", index=False)

    summary = {
        "target": TARGET_TAG, "r_static": R_STATIC, "oracle_avg_rank": float(oracle_avg_rank),
        "mean_delta_nll_expert_static": df["delta_nll_expert_static"].mean(),
        "mean_delta_nll_oracle_token_wise": df["delta_nll_oracle_token_wise"].mean(),
        "mean_delta_nll_random_equal_budget": df["delta_nll_random_equal_budget"].mean(),
        "mean_kl_expert_static": df["kl_expert_static"].mean(),
        "mean_kl_oracle_token_wise": df["kl_oracle_token_wise"].mean(),
        "mean_kl_random_equal_budget": df["kl_random_equal_budget"].mean(),
        "mean_top1_agree_expert_static": df["top1_agree_expert_static"].mean(),
        "mean_top1_agree_oracle_token_wise": df["top1_agree_oracle_token_wise"].mean(),
        "mean_top1_agree_random_equal_budget": df["top1_agree_random_equal_budget"].mean(),
    }
    summary["gate4_pass"] = bool(summary["mean_delta_nll_oracle_token_wise"] < summary["mean_delta_nll_expert_static"])
    with open(f"{RESULTS_DIR}/phase6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
