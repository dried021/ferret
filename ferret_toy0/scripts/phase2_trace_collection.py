"""Phase 2: trace collection on the EVAL sequences for the 4 selected
(layer, expert) pairs. Saves z_t (down_proj input), y_t (down_proj output),
router_weight g_t, and (sample_id, token_position) metadata, plus router
margin/entropy/top1-prob for optional exploratory analysis (Figure 5).

Everything is moved to CPU / FP16 immediately after capture, per the
methodology's memory-saving rules.
"""
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from moe_hooks import install_capture_hook, get_moe_layers

MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
TRACE_DIR = "/home/minjeong/project/FERRET/ferret_toy0/traces"
SEL_PATH = "/home/minjeong/project/FERRET/ferret_toy0/routing_counts/selection.json"

TARGET_MIN, TARGET_MAX = 500, 3000


def main():
    device = "cuda:0"
    with open(SEL_PATH) as f:
        selection = json.load(f)

    targets = {}  # (layer_id, expert_id) -> accumulator dict
    for tag, info in selection.items():
        lid = info["layer_id"]
        for e in info["experts"]:
            targets[(lid, e["expert_id"])] = {
                "z": [], "y": [], "g": [], "sample_id": [], "token_pos": [],
            }

    target_layers = sorted({lid for lid, _ in targets.keys()})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model = model.to(device).eval()

    # router logits capture, only for selected layers (for optional Figure 5)
    router_logits_by_layer_seq = {}  # (layer_id, sample_id) -> (seq_len, n_experts) fp16 cpu tensor

    router_handles = []

    def make_router_hook(layer_id):
        def hook(module, inputs, output):
            hidden_states = inputs[0]
            router_logits = F.linear(hidden_states, module.weight).float()
            router_logits_by_layer_seq[(layer_id, hook.cur_sample_id)] = router_logits.detach().cpu().half()
        hook.cur_sample_id = -1
        return hook

    router_hook_fns = {}
    for layer_id, bsm in get_moe_layers(model):
        if layer_id in target_layers:
            fn = make_router_hook(layer_id)
            router_hook_fns[layer_id] = fn
            router_handles.append(bsm.router.register_forward_hook(fn))

    capture_handles = []

    def make_cb(sample_id_box):
        def cb(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t, router_weight):
            key = (layer_id, expert_idx)
            if key not in targets:
                return
            acc = targets[key]
            if len(acc["sample_id"]) > 0 and sum(len(s) for s in [acc["sample_id"]]) >= TARGET_MAX * 1.5:
                return
            acc["z"].append(z_t.detach().to("cpu", torch.float16))
            acc["y"].append(y_t.detach().to("cpu", torch.float16))
            acc["g"].append(router_weight.detach().to("cpu", torch.float16))
            acc["sample_id"].append(torch.full_like(token_idx.cpu(), sample_id_box[0]))
            acc["token_pos"].append(token_idx.detach().cpu().clone())
        return cb

    sample_id_box = [-1]
    cb = make_cb(sample_id_box)
    for layer_id, bsm in get_moe_layers(model):
        if layer_id in target_layers:
            capture_handles.append(install_capture_hook(bsm.experts, layer_id, cb))

    seqs = torch.load(f"{TRACE_DIR}/_sequences.pt", weights_only=False)["eval"]

    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            sample_id_box[0] = i
            for fn in router_hook_fns.values():
                fn.cur_sample_id = i
            ids_b = ids.unsqueeze(0).to(device)
            model(input_ids=ids_b, use_cache=False)
            counts_now = {k: sum(t.numel() for t in v["sample_id"]) for k, v in targets.items()}
            print(f"eval seq {i+1}/{len(seqs)} -> running token counts: {counts_now}")

    for h in capture_handles:
        h.remove()
    for h in router_handles:
        h.remove()

    os.makedirs(TRACE_DIR, exist_ok=True)
    manifest = {}
    for (lid, eid), acc in targets.items():
        z = torch.cat(acc["z"], dim=0)
        y = torch.cat(acc["y"], dim=0)
        g = torch.cat(acc["g"], dim=0)
        sample_id = torch.cat(acc["sample_id"], dim=0)
        token_pos = torch.cat(acc["token_pos"], dim=0)

        # router margin/entropy per captured token, joined from router_logits_by_layer_seq
        margins, entropies, top1p = [], [], []
        for sid, pos in zip(sample_id.tolist(), token_pos.tolist()):
            logits = router_logits_by_layer_seq[(lid, sid)][pos].float()
            probs = torch.softmax(logits, dim=-1)
            top2 = torch.topk(probs, 2).values
            margins.append((top2[0] - top2[1]).item())
            entropies.append((-(probs * (probs.clamp_min(1e-12)).log()).sum()).item())
            top1p.append(top2[0].item())

        tag = f"layer{lid:02d}_expert{eid:02d}"
        torch.save(z, f"{TRACE_DIR}/{tag}_z.pt")
        torch.save(y, f"{TRACE_DIR}/{tag}_y.pt")
        meta = {
            "router_weight": g,
            "sample_id": sample_id,
            "token_pos": token_pos,
            "router_margin": torch.tensor(margins),
            "router_entropy": torch.tensor(entropies),
            "router_top1_prob": torch.tensor(top1p),
        }
        torch.save(meta, f"{TRACE_DIR}/{tag}_metadata.pt")
        manifest[tag] = {"layer_id": lid, "expert_id": eid, "n_tokens": int(z.shape[0]), "z_dim": int(z.shape[1]), "y_dim": int(y.shape[1])}
        print(tag, "n_tokens=", z.shape[0], "z_dim=", z.shape[1], "y_dim=", y.shape[1])

    with open(f"{TRACE_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
