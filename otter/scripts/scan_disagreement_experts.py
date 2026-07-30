"""Stage 1 of the disagreement-aware calibration method (01_연구설계.md
Section 23, 06_논문_구성.md §6 "제안 방법: 불일치-표적 calibration budget
배분"): a single forward-only pass over each of Phase 1's 6 calibration
conditions (phase1_calib_data.py's 5 single-language conditions plus one
balanced-type condition), captured simultaneously across ALL 27 DeepSeek-
MoE-16B MoE layers -- not just the 4 role layers
make_figure_expert_language_heatmap.py plots -- so Stage 2's targeted-budget
allocator (phase1_6_targeted_budget.py) has proxy/hit_count coverage for
whichever layer a disagreement expert actually lives in, not only the 4
pre-selected role layers.

Unlike fisher_pilot_a.py/phase1_fisher.py, this needs no backward pass and no
per-layer freeze-loop: proxy capture only needs a forward hook, so every
layer's MoE block is monkey-patched at once and a single forward pass over a
calibration sample updates all 27 layers' proxy/hit_count together. This is
the "forward-only proxy 통계 수집... 3090이 커버, 무료" scan
00_docs/07_실행_리소스_계획.md describes -- cheap enough to run over every
MoE layer, unlike fisher_pilot_a.py's real-gradient pilot which was scoped
down to 4 layers because backward passes are expensive.

Calibration sampling reuses phase1_fisher.py's build_samples() verbatim (same
condition text pool, same random-window sampling, same --n-samples/--seqlen
defaults) so this scan's calibration text is constructed identically to what
the real Fisher run (Stage 2) will later use for the SAME condition -- a
prerequisite for the proxy-vs-real Spearman comparison
analyze_fisher_pilot_a.py-style scripts already do.

Output schema matches make_figure_expert_language_heatmap.py's
load_scan_results() contract exactly (that function's docstring was written
as the de facto interface spec for this script before this script existed):
    {"<layer_idx>": {"role": "<early|transition|sensitive|final|other>",
                      "<condition>": {"proxy": [float, ...], "hit_count": [int, ...]},
                      ...for each condition...,
                      "disagreement_experts": [int, ...]}}
"disagreement_experts" reuses make_figure_expert_language_heatmap.py's own
normalize_layer()/rank_by_disagreement() math directly (imported, not
reimplemented) so Stage 1's notion of "disagreement" is identical to what the
paper figure shows, restricted to experts with >= --min-hit-count routed
tokens in EVERY single-language condition (an expert a language barely
routes to cannot have its cross-language variance trusted -- same worry
phase1_fisher.py's dead-expert check guards against for real Fisher).

Usage:
    conda run -n d2moe_env python scan_disagreement_experts.py [--smoke]
        [--n-samples 128] [--seqlen 512] [--seed 0]
        [--balanced-condition balanced] [--top-k 30] [--min-hit-count 5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402
import phase1_fisher  # noqa: E402 -- reuses build_samples() verbatim
from fisher_pilot_a import LAYER_ROLE  # noqa: E402 -- reused, not redefined
from disagreement_common import (  # noqa: E402 -- shared with make_figure_expert_language_heatmap.py, no plotting deps
    LANG_CONDITIONS, conditions_for, normalize_layer, rank_by_disagreement,
)

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/scan_disagreement_experts")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_HIDDEN_LAYERS = 28  # 0 = dense, 1..27 = MoE (see phase1_fisher.py)
MOE_LAYERS = list(range(1, NUM_HIDDEN_LAYERS))


def make_capturing_forward(moe_block, proxy_accum, hit_accum):
    """Same (router_weight * ||expert_output||)^2 proxy as
    fisher_pilot_a.py's make_capturing_forward, extended with a per-expert
    routed-token counter (hit_accum) -- the field make_figure_expert_
    language_heatmap.py needs to tell a genuine language-specific spike from
    a low-routing-sample artifact (its review point 3)."""
    experts = moe_block.experts
    top_k = moe_block.num_experts_per_tok

    def captured_forward(hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _aux_loss = moe_block.gate(hidden_states)
        hs = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        flat_topk_weight = topk_weight.view(-1)
        hs_rep = hs.repeat_interleave(top_k, dim=0)
        y = torch.empty_like(hs_rep)
        for i, expert in enumerate(experts):
            mask = flat_topk_idx == i
            if not mask.any():
                continue
            tok = hs_rep[mask]
            out = expert(tok)
            w = flat_topk_weight[mask]
            proxy_accum[i] += (w.float() * out.float().norm(dim=-1)).pow(2).sum().item()
            hit_accum[i] += int(mask.sum().item())
            y[mask] = out
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)
        if moe_block.config.n_shared_experts is not None:
            y = y + moe_block.shared_experts(identity)
        return y

    return captured_forward


def install_hooks(model, n_experts):
    """Monkey-patches EVERY MoE layer's forward at once (not one layer at a
    time like fisher_pilot_a.py) -- a proxy forward pass has no gradient-
    memory reason to freeze/isolate layers, so one pass over a calibration
    sample can update all 27 layers' accumulators together."""
    proxy = {layer_idx: [0.0] * n_experts for layer_idx in MOE_LAYERS}
    hits = {layer_idx: [0] * n_experts for layer_idx in MOE_LAYERS}
    originals = {}
    for layer_idx in MOE_LAYERS:
        moe_block = model.model.layers[layer_idx].mlp
        originals[layer_idx] = moe_block.forward
        moe_block.forward = make_capturing_forward(moe_block, proxy[layer_idx], hits[layer_idx])
    return proxy, hits, originals


def remove_hooks(model, originals):
    for layer_idx, orig in originals.items():
        model.model.layers[layer_idx].mlp.forward = orig


def scan_condition(model, tokenizer, texts, seqlen, n_experts):
    """Returns (proxy, hits, n_used, n_tokens). n_tokens (2026-07-27, added
    for Stage 2) is the TOTAL input tokens actually processed for this
    condition -- identical across every layer (one shared forward pass), so
    it's tracked once here, not per-layer. phase1_6_targeted_budget.py
    divides each layer/expert's hit_count by this to get a per-token hit
    RATE it can extrapolate to phase1_fisher.py's (larger) real-Fisher
    calibration budget."""
    embed_device = model.get_input_embeddings().weight.device
    proxy, hits, originals = install_hooks(model, n_experts)
    model.eval()
    n_used = 0
    n_tokens = 0
    try:
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
                input_ids = enc["input_ids"].to(embed_device)
                if input_ids.shape[1] < 2:
                    continue
                model(input_ids=input_ids)
                n_used += 1
                n_tokens += input_ids.shape[1]
    finally:
        remove_hooks(model, originals)
    if n_used == 0:
        raise RuntimeError("no usable samples (all shorter than 2 tokens)")
    return proxy, hits, n_used, n_tokens


def find_disagreement_experts(layer_entry, conditions, top_k, min_hit_count):
    """Reuses make_figure_expert_language_heatmap.py's own column-wise
    normalize_layer()/rank_by_disagreement() math -- Stage 1's notion of
    "disagreement" must match what the paper figure shows, not a separate
    reimplementation that could silently drift from it.

    An expert is only eligible if it has >= min_hit_count routed tokens in
    EVERY single-language condition (LANG_CONDITIONS) -- an expert a
    language barely routes to cannot have its cross-language variance
    trusted (2026-07-27 review point 3's worry, applied at the source
    instead of only flagged downstream)."""
    proxy_by_cond = {c: np.asarray(layer_entry[c]["proxy"], dtype=np.float64) for c in conditions}
    hit_by_cond = {c: np.asarray(layer_entry[c]["hit_count"], dtype=np.int64) for c in conditions}
    n_experts = len(next(iter(proxy_by_cond.values())))

    eligible = np.ones(n_experts, dtype=bool)
    for c in LANG_CONDITIONS:
        eligible &= hit_by_cond[c] >= min_hit_count
    n_ineligible = int((~eligible).sum())

    normed = normalize_layer(proxy_by_cond, conditions)
    _, variance = rank_by_disagreement(normed)
    variance_eligible = np.where(eligible, variance, -np.inf)
    order = np.argsort(-variance_eligible)
    top = [int(e) for e in order[:top_k] if eligible[e]]
    return top, n_ineligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=128, help="matches phase1_fisher.py's default so calibration text is built identically")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balanced-condition", default="balanced", choices=["balanced", "mixed_5lang"])
    parser.add_argument("--top-k", type=int, default=30, help="candidate disagreement-expert count per layer, matches the figure's default")
    parser.add_argument("--min-hit-count", type=int, default=5)
    args = parser.parse_args()

    n_samples = 4 if args.smoke else args.n_samples
    seqlen = 128 if args.smoke else args.seqlen
    conditions = conditions_for(args.balanced_condition)

    n_gpu = torch.cuda.device_count()
    max_memory = {i: "16GiB" for i in range(n_gpu)}
    print(f"[scan] {'SMOKE' if args.smoke else 'FULL'}: conditions={conditions} n_samples={n_samples} "
          f"seqlen={seqlen} n_layers={len(MOE_LAYERS)} max_memory={max_memory}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", max_memory=max_memory, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    n_experts = model.config.n_routed_experts
    print(f"[scan] {n_experts} routed experts, top_k={model.config.num_experts_per_tok}")

    results = {layer_idx: {"role": LAYER_ROLE.get(layer_idx, "other")} for layer_idx in MOE_LAYERS}
    meta = {"n_tokens": {}}
    out_path = RESULTS_DIR / ("scan_results_smoke.json" if args.smoke else "scan_results.json")

    def write_checkpoint():
        payload = {str(k): v for k, v in results.items()}
        payload["_meta"] = meta
        out_path.write_text(json.dumps(payload, indent=2))

    for cond in conditions:
        print(f"[scan] === condition {cond} ===")
        texts = phase1_fisher.build_samples(cond, n_samples, seqlen, args.seed)
        proxy, hits, n_used, n_tokens = scan_condition(model, tokenizer, texts, seqlen, n_experts)
        for layer_idx in MOE_LAYERS:
            results[layer_idx][cond] = {"proxy": proxy[layer_idx], "hit_count": hits[layer_idx]}
        meta["n_tokens"][cond] = n_tokens
        nonzero_l4 = sum(1 for v in proxy[MOE_LAYERS[0]] if v > 0)
        print(f"[scan] {cond}: {n_used}/{len(texts)} samples used, {n_tokens} tokens, "
              f"layer {MOE_LAYERS[0]} nonzero-proxy experts={nonzero_l4}/{n_experts}")
        # Checkpoint after every condition (6 conditions x up to 128 samples
        # each, all 27 layers per sample -- cheap per-sample but still worth
        # not losing on an interrupted run, same rationale as
        # phase1_fisher.py's per-layer checkpoint).
        write_checkpoint()
        print(f"[scan] checkpoint-saved {out_path}")
        torch.cuda.empty_cache()

    print("[scan] scanning done, computing per-layer disagreement_experts candidate lists ...")
    total_ineligible = 0
    for layer_idx in MOE_LAYERS:
        top, n_ineligible = find_disagreement_experts(results[layer_idx], conditions, args.top_k, args.min_hit_count)
        results[layer_idx]["disagreement_experts"] = top
        total_ineligible += n_ineligible
        role = results[layer_idx]["role"]
        print(f"[scan] layer {layer_idx} ({role}): {len(top)} disagreement candidates "
              f"({n_ineligible} experts excluded for < {args.min_hit_count} hits in some language)")

    write_checkpoint()
    print(f"[scan] wrote {out_path} ({total_ineligible} total low-sample exclusions across all layers)")


if __name__ == "__main__":
    main()
