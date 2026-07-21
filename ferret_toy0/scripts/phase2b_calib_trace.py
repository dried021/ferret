"""Phase 2b (added after seeing Phase-4 results): collect down_proj INPUT
traces (z_t only) on the CALIBRATION sequences for the 4 selected experts.

Reason: the naive weight-only SVD (Phase 3) showed a near-flat singular
value spectrum -- these down_proj matrices (1024x512) are close to
full-rank in the generic operator-norm sense, so naive low-rank SVD fails
Gate 1 almost everywhere except near full rank. Section 19 of the Toy0
methodology explicitly allows one retry with an *activation-aware*
decomposition before concluding the expert is not low-rank. Activation-aware
SVD only needs the *input* distribution (z_t), not paired outputs, and must
be built from the calibration split (not eval) to keep the Gate-3 evaluation
unbiased.
"""
import json
import torch
from transformers import AutoModelForCausalLM

from moe_hooks import install_capture_hook, get_moe_layers

MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
TRACE_DIR = "/home/minjeong/project/FERRET/ferret_toy0/traces"
SEL_PATH = "/home/minjeong/project/FERRET/ferret_toy0/routing_counts/selection.json"


def main():
    device = "cuda:0"
    with open(SEL_PATH) as f:
        selection = json.load(f)

    targets = {}
    for tag, info in selection.items():
        lid = info["layer_id"]
        for e in info["experts"]:
            targets[(lid, e["expert_id"])] = {"z": []}

    target_layers = sorted({lid for lid, _ in targets.keys()})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model = model.to(device).eval()

    def cb(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t, router_weight):
        key = (layer_id, expert_idx)
        if key not in targets:
            return
        targets[key]["z"].append(z_t.detach().to("cpu", torch.float32))

    handles = []
    for layer_id, bsm in get_moe_layers(model):
        if layer_id in target_layers:
            handles.append(install_capture_hook(bsm.experts, layer_id, cb))

    seqs = torch.load(f"{TRACE_DIR}/_sequences.pt", weights_only=False)["calib"]
    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            ids_b = ids.unsqueeze(0).to(device)
            model(input_ids=ids_b, use_cache=False)
            print(f"calib seq {i+1}/{len(seqs)}")

    for h in handles:
        h.remove()

    for (lid, eid), acc in targets.items():
        z = torch.cat(acc["z"], dim=0)
        tag = f"layer{lid:02d}_expert{eid:02d}"
        torch.save(z, f"{TRACE_DIR}/{tag}_z_calib.pt")
        print(tag, "calib n_tokens=", z.shape[0])


if __name__ == "__main__":
    main()
