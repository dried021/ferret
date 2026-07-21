"""Phase 2: trace collection on the EVAL sequences for the selected
(layer, expert) pairs. Saves z_t (down_proj input), y_t (down_proj output),
router_weight g_t, and (sample_id, token_position) metadata, plus router
margin/entropy/top1-prob for optional exploratory analysis (Figure 5).

Everything is moved to CPU / FP16 immediately after capture, per the
methodology's memory-saving rules (section 5). Model loaded INT8 balanced
across cuda:0,1, same as phase0/phase1 (Stage A backbone precision).
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from moe_hooks import get_moe_layers, install_capture_hook

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
ROOT = Path(__file__).resolve().parents[1]
DEVICE_MAP_CONFIG = ROOT / "configs" / "device_map.json"
TRACE_DIR = ROOT / "traces"
SEL_PATH = ROOT / "routing_counts" / "selection.json"
LOG_PATH = ROOT / "logs" / "phase2_trace_collection.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    with open(SEL_PATH) as f:
        selection = json.load(f)
    with open(DEVICE_MAP_CONFIG) as f:
        dm_cfg = json.load(f)

    targets = {}  # (layer_id, expert_id) -> accumulator dict
    for tag, info in selection.items():
        lid = info["layer_id"]
        for e in info["experts"]:
            targets[(lid, e["expert_id"])] = {
                "z": [], "y": [], "g": [], "sample_id": [], "token_pos": [],
            }
    target_layers = sorted({lid for lid, _ in targets.keys()})
    log(f"target (layer, expert) pairs: {list(targets.keys())}")

    max_memory = {(int(k) if k.isdigit() else k): v for k, v in dm_cfg["max_memory"].items()}
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

    log("loading model in INT8 across cuda:0,1 (balanced device_map)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=dm_cfg["strategy"],
        max_memory=max_memory,
        dtype=torch.bfloat16,
    )
    model.eval()
    log(f"model loaded in {time.time() - t0:.1f}s")

    # router logits capture, only for target layers (for optional Figure 5)
    router_logits_by_layer_seq = {}  # (layer_id, sample_id) -> (seq_len, n_experts) fp16 cpu tensor
    router_handles = []
    router_hook_fns = {}

    def make_router_hook(layer_id):
        def hook(module, inputs, output):
            hidden_states = inputs[0]
            router_logits = F.linear(hidden_states, module.weight).float()
            router_logits_by_layer_seq[(layer_id, hook.cur_sample_id)] = router_logits.detach().cpu().half()
        hook.cur_sample_id = -1
        return hook

    moe_layers = get_moe_layers(model, layer_ids=target_layers)
    for lid, block in moe_layers:
        fn = make_router_hook(lid)
        router_hook_fns[lid] = fn
        router_handles.append(block.gate.register_forward_hook(fn))

    capture_handles = []

    def make_cb(sample_id_box):
        def cb(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t, router_weight):
            key = (layer_id, expert_idx)
            if key not in targets:
                return
            acc = targets[key]
            acc["z"].append(z_t.detach().to("cpu", torch.float16))
            acc["y"].append(y_t.detach().to("cpu", torch.float16))
            acc["g"].append(router_weight.detach().to("cpu", torch.float16))
            acc["sample_id"].append(torch.full_like(token_idx.cpu(), sample_id_box[0]))
            acc["token_pos"].append(token_idx.detach().cpu().clone())
        return cb

    sample_id_box = [-1]
    cb = make_cb(sample_id_box)
    for lid, block in moe_layers:
        capture_handles.append(install_capture_hook(block.experts, lid, cb))

    seqs = torch.load(TRACE_DIR / "_sequences.pt", weights_only=False)["eval"]
    log(f"loaded {len(seqs)} eval sequences")

    input_device = next(model.parameters()).device
    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            sample_id_box[0] = i
            for fn in router_hook_fns.values():
                fn.cur_sample_id = i
            ids_b = ids.unsqueeze(0).to(input_device)
            model(input_ids=ids_b, use_cache=False)
            counts_now = {f"{k[0]}/{k[1]}": sum(t.numel() for t in v["sample_id"]) for k, v in targets.items()}
            log(f"eval seq {i + 1}/{len(seqs)} -> running token counts: {counts_now}")

    for h in capture_handles:
        h.remove()
    for h in router_handles:
        h.remove()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for (lid, eid), acc in targets.items():
        z = torch.cat(acc["z"], dim=0)
        y = torch.cat(acc["y"], dim=0)
        g = torch.cat(acc["g"], dim=0)
        sample_id = torch.cat(acc["sample_id"], dim=0)
        token_pos = torch.cat(acc["token_pos"], dim=0)

        margins, entropies, top1p = [], [], []
        for sid, pos in zip(sample_id.tolist(), token_pos.tolist()):
            logits = router_logits_by_layer_seq[(lid, sid)][pos].float()
            probs = torch.softmax(logits, dim=-1)
            top2 = torch.topk(probs, 2).values
            margins.append((top2[0] - top2[1]).item())
            entropies.append((-(probs * (probs.clamp_min(1e-12)).log()).sum()).item())
            top1p.append(top2[0].item())

        tag = f"layer{lid:02d}_expert{eid:02d}"
        torch.save(z, TRACE_DIR / f"{tag}_z.pt")
        torch.save(y, TRACE_DIR / f"{tag}_y.pt")
        meta = {
            "router_weight": g,
            "sample_id": sample_id,
            "token_pos": token_pos,
            "router_margin": torch.tensor(margins),
            "router_entropy": torch.tensor(entropies),
            "router_top1_prob": torch.tensor(top1p),
        }
        torch.save(meta, TRACE_DIR / f"{tag}_metadata.pt")
        manifest[tag] = {
            "layer_id": lid, "expert_id": eid,
            "n_tokens": int(z.shape[0]), "z_dim": int(z.shape[1]), "y_dim": int(y.shape[1]),
        }
        log(f"{tag} n_tokens={z.shape[0]} z_dim={z.shape[1]} y_dim={y.shape[1]}")

    with open(TRACE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("PHASE 2 DONE")


if __name__ == "__main__":
    main()
