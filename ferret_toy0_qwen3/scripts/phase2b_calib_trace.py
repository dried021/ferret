"""Phase 2b (added after seeing Phase 4 results): collect down_proj INPUT
traces (z_t only) on the CALIBRATION sequences for the selected experts.

Reason: naive weight-only SVD (Phase 3/4) showed oracle_rank == R_max (768,
full rank) for every captured token across all 4 selected experts -- these
down_proj matrices are close to full-rank in the generic operator-norm
sense, so naive low-rank SVD fails Gate 1 everywhere. Section 19 of the
methodology explicitly allows one retry with an activation-aware
decomposition before concluding the expert is not low-rank.
Activation-aware SVD only needs the input distribution (z_t), not paired
outputs, and must be built from the calibration split (not eval) to keep
the Gate-3 evaluation unbiased.
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from moe_hooks import get_moe_layers, install_capture_hook

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
ROOT = Path(__file__).resolve().parents[1]
DEVICE_MAP_CONFIG = ROOT / "configs" / "device_map.json"
TRACE_DIR = ROOT / "traces"
SEL_PATH = ROOT / "routing_counts" / "selection.json"
LOG_PATH = ROOT / "logs" / "phase2b_calib_trace.log"


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

    targets = {}
    for tag, info in selection.items():
        lid = info["layer_id"]
        for e in info["experts"]:
            targets[(lid, e["expert_id"])] = {"z": []}
    target_layers = sorted({lid for lid, _ in targets.keys()})

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

    def cb(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t, router_weight):
        key = (layer_id, expert_idx)
        if key not in targets:
            return
        targets[key]["z"].append(z_t.detach().to("cpu", torch.float32))

    handles = []
    for lid, block in get_moe_layers(model, layer_ids=target_layers):
        handles.append(install_capture_hook(block.experts, lid, cb))

    seqs = torch.load(TRACE_DIR / "_sequences.pt", weights_only=False)["calib"]
    log(f"loaded {len(seqs)} calibration sequences")

    input_device = next(model.parameters()).device
    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            ids_b = ids.unsqueeze(0).to(input_device)
            model(input_ids=ids_b, use_cache=False)
            log(f"calib seq {i + 1}/{len(seqs)}")

    for h in handles:
        h.remove()

    for (lid, eid), acc in targets.items():
        z = torch.cat(acc["z"], dim=0)
        tag = f"layer{lid:02d}_expert{eid:02d}"
        torch.save(z, TRACE_DIR / f"{tag}_z_calib.pt")
        log(f"{tag} calib n_tokens={z.shape[0]} nan={torch.isnan(z).sum().item()} inf={torch.isinf(z).sum().item()}")
    log("PHASE 2b DONE")


if __name__ == "__main__":
    main()
