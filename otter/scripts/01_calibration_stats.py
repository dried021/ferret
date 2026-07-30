"""Toy0: does calibration language change routing coverage and per-expert
importance, before any actual D^2-MoE compression exists in this repo?

For each calibration condition (en_only, en_only_control, ko_only, zh_only,
balanced -- see data/languages.yaml), runs a batched forward over that
condition's sentences with the capture hook installed, and accumulates per
(layer, expert):
  - hit_count           routing coverage
  - fisher_proxy        sum_t (router_weight_t * ||y_t||)^2, a cheap
                         activation-based importance proxy. This is NOT the
                         gradient-based Fisher information D^2-MoE actually
                         uses -- it's a stand-in to check whether *any*
                         calibration-language signal exists at all before
                         investing in the real (gradient) computation. See
                         README.md "Fisher-proxy 한계".

en_only_control uses a disjoint sentence set of the *same* language as
en_only -- this is the same-language noise floor that 02_analyze_toy0.py
compares cross-language divergence against.

Usage:
    conda run -n torch_env python 01_calibration_stats.py [--smoke]

--smoke restricts to 5 sentences/condition-language-row and writes to
results/toy0_routing_fisher_stats_smoke.json.

Output:
    results/toy0_routing_fisher_stats.json
        {condition_id: {"hit_count": {layer: [num_experts]},
                         "fisher_proxy": {layer: [num_experts]},
                         "n_sentences": int, "n_tokens": int}}
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import moe_hooks
import data_utils


def load_config():
    spec = importlib.util.spec_from_file_location("d2moe_ml_config", SCRIPT_DIR / "00_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_path(p, smoke):
    return p if not smoke else p.with_name(p.stem + "_smoke" + p.suffix)


def run_condition(model, tokenizer, cfg, moe_layers, num_experts, source, split, prefix,
                   condition_id, rows, n_sentences_override, eval_batch_size):
    hit_count = {lid: torch.zeros(num_experts, dtype=torch.long) for lid, _ in moe_layers}
    fisher_proxy = {lid: torch.zeros(num_experts, dtype=torch.float64) for lid, _ in moe_layers}

    def callback(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t, router_weight):
        n = y_t.shape[0]
        hit_count[layer_id][expert_idx] += n
        # per-token proxy importance: (router-weighted output norm)^2, summed.
        # A real Fisher-weighted score would use squared gradients instead --
        # see module docstring and README.md for why this is a placeholder.
        per_token_norm = y_t.detach().to(torch.float64).norm(dim=-1)
        weighted_sq = (router_weight.detach().to(torch.float64) * per_token_norm) ** 2
        fisher_proxy[layer_id][expert_idx] += weighted_sq.sum().item()

    handles = [moe_hooks.install_capture_hook(mod.experts, lid, callback) for lid, mod in moe_layers]

    total_sentences = 0
    total_tokens = 0
    try:
        for lang_code, n_sentences, offset in rows:
            n = n_sentences_override if n_sentences_override is not None else n_sentences
            sentences = data_utils.load_flores_sentences(source, split, prefix, lang_code, n, offset)
            nll_sums, n_tokens = data_utils.evaluate_sentences(model, tokenizer, sentences, eval_batch_size)
            total_sentences += len(sentences)
            total_tokens += sum(n_tokens)
            print(f"[01] {condition_id}/{lang_code}: {len(sentences)} sentences, {sum(n_tokens)} tokens")
    finally:
        for h in handles:
            h.remove()

    return {
        "hit_count": {lid: hit_count[lid].tolist() for lid, _ in moe_layers},
        "fisher_proxy": {lid: fisher_proxy[lid].tolist() for lid, _ in moe_layers},
        "n_sentences": total_sentences,
        "n_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    source, split, prefix, conditions = cfg.load_conditions()
    n_override = 5 if args.smoke else None

    print(f"[01] {'SMOKE' if args.smoke else 'FULL'} run: layers={cfg.TOY_LAYER_INDICES}, "
          f"conditions={list(conditions)}")
    print(f"[01] loading {cfg.MODEL_NAME} ...")
    model, tokenizer = cfg.load_model_and_tokenizer()

    moe_layers = moe_hooks.get_moe_layers(model, cfg.TOY_LAYER_INDICES)
    if not moe_layers:
        raise RuntimeError(f"no MoE layers found among {cfg.TOY_LAYER_INDICES}")
    num_experts = moe_layers[0][1].experts.num_experts
    print(f"[01] {len(moe_layers)} MoE layers, {num_experts} experts/layer")

    all_results = {}
    for condition_id, rows in conditions.items():
        print(f"[01] === condition: {condition_id} ===")
        all_results[condition_id] = run_condition(
            model, tokenizer, cfg, moe_layers, num_experts, source, split, prefix,
            condition_id, rows, n_override, cfg.EVAL_BATCH_SIZE,
        )

    out_path = smoke_path(cfg.ROUTING_STATS_JSON, args.smoke)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"[01] wrote {out_path}")


if __name__ == "__main__":
    main()
