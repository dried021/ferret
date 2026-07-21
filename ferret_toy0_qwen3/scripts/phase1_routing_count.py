"""Phase 1: cheap routing-count pass (methodology doc section 6).

Loads Qwen3-30B-A3B once (INT8 balanced across cuda:0,1, matching phase0),
runs it over the calibration sequences, and for every (layer, expert) pair
in the candidate minimal-run layers counts how many routed tokens hit it.
Then, per position (middle/late), picks whichever candidate layer has the
higher combined top-2-expert count, and records its 2 most frequent experts.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from moe_hooks import get_moe_layers, install_capture_hook

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment_config.yaml"
DEVICE_MAP_CONFIG = ROOT / "configs" / "device_map.json"
SEQ_PATH = ROOT / "traces" / "_sequences.pt"
OUT_DIR = ROOT / "routing_counts"
LOG_PATH = ROOT / "logs" / "phase1_routing_count.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(DEVICE_MAP_CONFIG) as f:
        dm_cfg = json.load(f)

    candidates = cfg["layer_selection"]["candidates"]
    minimal_positions = cfg["layer_selection"]["minimal_positions"]
    n_experts_per_layer = cfg["expert_selection"]["n_experts_per_layer"]
    min_token_count = cfg["expert_selection"]["min_eval_token_count"]

    candidate_layer_ids = sorted({lid for pos in minimal_positions for lid in candidates[pos]})
    log(f"minimal-run positions: {minimal_positions}, candidate layers: {candidate_layer_ids}")

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

    counts = defaultdict(lambda: defaultdict(int))  # counts[layer_id][expert_idx] = n_tokens

    def cb(layer_id, expert_idx, token_idx, **kw):
        counts[layer_id][expert_idx] += token_idx.numel()

    moe_layers = get_moe_layers(model, layer_ids=candidate_layer_ids)
    log(f"hooking {len(moe_layers)} MoE layers: {[lid for lid, _ in moe_layers]}")
    handles = [install_capture_hook(block.experts, lid, cb) for lid, block in moe_layers]

    seqs = torch.load(SEQ_PATH, weights_only=False)["calib"]
    log(f"loaded {len(seqs)} calibration sequences from {SEQ_PATH}")

    input_device = next(model.parameters()).device
    with torch.inference_mode():
        for i, ids in enumerate(seqs):
            ids = ids.unsqueeze(0).to(input_device)
            model(input_ids=ids, use_cache=False)
            log(f"processed calib seq {i + 1}/{len(seqs)}")

    for h in handles:
        h.remove()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        str(lid): {str(e): counts[lid][e] for e in sorted(counts[lid])}
        for lid in candidate_layer_ids
    }
    with open(OUT_DIR / "routing_counts.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"routing counts written to {OUT_DIR / 'routing_counts.json'}")

    # --- per-position layer selection: pick the candidate with the higher
    # combined top-n_experts_per_layer count, matching section 6's frequency
    # criterion ("빈도가 주요 기준이어야 한다") ---
    selection = {}
    for pos in minimal_positions:
        best = None
        for lid in candidates[pos]:
            top = sorted(counts[lid].items(), key=lambda kv: -kv[1])[:n_experts_per_layer]
            total = sum(c for _, c in top)
            log(f"  candidate layer {lid} ({pos}): top-{n_experts_per_layer} experts={top}, total={total}")
            if best is None or total > best[1]:
                best = (lid, total, top)
        lid, total, top = best
        meets_threshold = all(c >= min_token_count for _, c in top)
        selection[pos] = {
            "layer_id": lid,
            "experts": [{"expert_id": e, "count": c} for e, c in top],
            "meets_min_token_count": meets_threshold,
        }
        log(f"selected {pos} layer={lid} experts={top} meets_min_token_count={meets_threshold}")

    with open(OUT_DIR / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    log(json.dumps(selection, indent=2))
    log(f"PHASE 1 {'PASS' if all(v['meets_min_token_count'] for v in selection.values()) else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
