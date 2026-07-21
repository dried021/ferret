"""Phase 1: cheap routing-count pass.

Loads the model once, runs it over the calibration sequences, and for every
(layer, expert) pair counts how many routed tokens hit it. Then selects
1 middle + 1 late MoE layer, and the 2 most frequent experts in each.
"""
import json
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM

from moe_hooks import install_capture_hook, get_moe_layers

MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
OUT_DIR = "/home/minjeong/project/FERRET/ferret_toy0/routing_counts"


def main():
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES pins this to the intended physical GPU
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model = model.to(device).eval()

    counts = defaultdict(lambda: defaultdict(int))  # counts[layer_id][expert_idx] = n_tokens

    def cb(layer_id, expert_idx, token_idx, **kw):
        counts[layer_id][expert_idx] += token_idx.numel()

    handles = []
    for layer_id, experts_mod_owner in get_moe_layers(model):
        h = install_capture_hook(experts_mod_owner.experts, layer_id, cb)
        handles.append(h)

    seqs = torch.load(
        "/home/minjeong/project/FERRET/ferret_toy0/traces/_sequences.pt", weights_only=False
    )["calib"]

    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            ids = ids.unsqueeze(0).to(device)
            model(input_ids=ids, use_cache=False)
            print(f"processed calib seq {i+1}/{len(seqs)}")

    for h in handles:
        h.remove()

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_local_experts

    result = {
        str(l): {str(e): counts[l][e] for e in range(n_experts)}
        for l in range(n_layers)
    }

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/routing_counts.json", "w") as f:
        json.dump(result, f, indent=2)

    # --- layer/expert selection ---
    moe_layer_ids = sorted(counts.keys())
    mid_layer = moe_layer_ids[len(moe_layer_ids) // 2]
    late_layer = moe_layer_ids[int(len(moe_layer_ids) * 0.8)]

    selection = {}
    for tag, lid in [("middle", mid_layer), ("late", late_layer)]:
        top2 = sorted(counts[lid].items(), key=lambda kv: -kv[1])[:2]
        selection[tag] = {"layer_id": lid, "experts": [{"expert_id": e, "count": c} for e, c in top2]}

    with open(f"{OUT_DIR}/selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    print(json.dumps(selection, indent=2))
    print("total MoE layers with routing:", len(moe_layer_ids))


if __name__ == "__main__":
    main()
