"""Fisher Pilot A: does the forward-only Fisher-proxy correlate with the real
gradient-based Fisher information, and does the EN-vs-non-EN gap show up in
both, on DeepSeek-MoE-16B (the model Phase 0 actually reproduces on)?

Scoped-down per plan: 4 layers x 4 conditions x 1 seed, not the full
5-layer x 5-condition x 3-seed factorial -- this only needs to answer two
narrow questions before any larger run:
  1. does proxy expert-rank correlate positively with real-gradient
     expert-rank (Spearman), per layer/condition?
  2. is EN-KO/EN-ZH divergence bigger than within-EN (A vs B) divergence at
     the "sensitive"/"final" layers vs the "early"/"transition" layers?

Memory design (why this fits on 2x24GB GPUs when the full-model, all-params
get_fisher.py did not):
  - only ONE MoE layer's expert parameters have requires_grad=True at a time
    (everything else frozen) -- gradient memory drops from ~model-size to
    ~one-layer's-experts-size (~1-2GB instead of ~31GB).
  - Fisher is accumulated as a single scalar per expert (sum of squared
    gradients across that expert's gate/up/down projections), not stored
    element-wise -- this is the "expert scalar Fisher" the merge-relevant
    question (does calibration language reorder which experts matter) only
    needs; element-wise Fisher is deferred to when the actual D^2-MoE merge
    step needs it.
  - short sequences (SEQLEN tokens, well under the paper's 2048), batch
    size 1, model.zero_grad(set_to_none=True) after every sample, no
    optimizer, immediate CPU float() for anything retained across samples.
  - proxy computation reuses the real DeepSeek-MoE-16B checkpoint's own
    DeepseekMoE.forward logic (copied here, not reimplemented from scratch)
    so it's an exact behavioral match, just with capture hooks and
    torch.no_grad() instead of gradient tracking.

Usage:
    conda run -n d2moe_env python fisher_pilot_a.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import data_utils  # noqa: E402 (reused as-is: load_full_column, token_budget_chunks)

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/fisher_pilot_a")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DeepSeek-MoE-16B has 28 layers (0 dense, 1-27 MoE) -- not the 48-layer
# Qwen3 scheme Toy0/Phase 0.5/0 used. Proportionally-equivalent depths:
# early ~14%, transition ~61%, sensitive/late ~86%, final =100%.
LAYERS = [4, 16, 24, 27]
LAYER_ROLE = {4: "early", 16: "transition", 24: "sensitive", 27: "final"}

CONDITIONS = ["english_a", "english_b", "korean", "chinese"]
LANG_OF = {"english_a": "eng_Latn", "english_b": "eng_Latn", "korean": "kor_Hang", "chinese": "zho_Hans"}
SOURCE, SPLIT, COL_PREFIX = "israel/flores-parallel", "test", "sentence_"
SEED = 0


def build_condition_sentences(tokenizer, token_budget):
    pools = {lang: data_utils.load_full_column(SOURCE, SPLIT, COL_PREFIX, lang) for lang in set(LANG_OF.values())}
    eng_a, eng_b = data_utils.token_budget_chunks(tokenizer, pools["eng_Latn"], SEED, [token_budget, token_budget])
    (kor,) = data_utils.token_budget_chunks(tokenizer, pools["kor_Hang"], SEED, [token_budget])
    (zho,) = data_utils.token_budget_chunks(tokenizer, pools["zho_Hans"], SEED, [token_budget])
    return {"english_a": eng_a, "english_b": eng_b, "korean": kor, "chinese": zho}


def freeze_all_but_layer(model, layer_idx):
    for p in model.parameters():
        p.requires_grad_(False)
    moe_block = model.model.layers[layer_idx].mlp
    for p in moe_block.experts.parameters():
        p.requires_grad_(True)
    return moe_block


def real_fisher_for_condition(model, moe_block, sentences, tokenizer, n_experts, seqlen):
    fisher = [0.0] * n_experts
    embed_device = model.get_input_embeddings().weight.device
    model.train()
    for text in sentences:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
        input_ids = enc["input_ids"].to(embed_device)
        if input_ids.shape[1] < 2:
            continue
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        for e_idx, expert in enumerate(moe_block.experts):
            g2 = 0.0
            for p in expert.parameters():
                if p.grad is not None:
                    g2 += p.grad.detach().float().pow(2).sum().item()
            fisher[e_idx] += g2
        model.zero_grad(set_to_none=True)
    return fisher


def make_capturing_forward(moe_block, proxy_accum):
    """Reproduces DeepseekMoE.forward's training-mode branch verbatim (see
    modeling_deepseek.py), except every expert call is wrapped to accumulate
    (router_weight * ||expert_output||)^2 per expert -- the same proxy
    formula Toy0/Phase 0.5/0 used on Qwen3, applied here to DeepSeek so the
    comparison against real_fisher_for_condition is same-model, same-data."""
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
            y[mask] = out
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)
        if moe_block.config.n_shared_experts is not None:
            y = y + moe_block.shared_experts(identity)
        return y

    return captured_forward


def proxy_for_condition(model, moe_block, sentences, tokenizer, n_experts, seqlen):
    proxy = [0.0] * n_experts
    embed_device = model.get_input_embeddings().weight.device
    orig_forward = moe_block.forward
    moe_block.forward = make_capturing_forward(moe_block, proxy)
    model.eval()
    try:
        with torch.no_grad():
            for text in sentences:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
                input_ids = enc["input_ids"].to(embed_device)
                if input_ids.shape[1] < 2:
                    continue
                model(input_ids=input_ids)
    finally:
        moe_block.forward = orig_forward
    return proxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--token-budget", type=int, default=768, help="per-condition token budget (512-1024 recommended)")
    parser.add_argument("--seqlen", type=int, default=192, help="max tokens per calibration sample (128-256 recommended)")
    args = parser.parse_args()

    token_budget = 128 if args.smoke else args.token_budget
    seqlen = args.seqlen
    layers = LAYERS[:1] if args.smoke else LAYERS

    n_gpu = torch.cuda.device_count()
    max_memory = {i: "16GiB" for i in range(n_gpu)}
    print(f"[pilot] {'SMOKE' if args.smoke else 'FULL'} run: layers={layers}, token_budget={token_budget}, "
          f"seqlen={seqlen}, max_memory={max_memory}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", max_memory=max_memory, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    n_experts = model.config.n_routed_experts
    print(f"[pilot] {n_experts} routed experts, top_k={model.config.num_experts_per_tok}")

    print("[pilot] building calibration sentence sets ...")
    condition_sentences = build_condition_sentences(tokenizer, token_budget)
    for cond, sents in condition_sentences.items():
        n_tok = sum(len(tokenizer(s)["input_ids"]) for s in sents)
        print(f"[pilot] {cond}: {len(sents)} sentences, ~{n_tok} tokens")

    results = {}
    out_path = RESULTS_DIR / ("pilot_a_results_smoke.json" if args.smoke else "pilot_a_results.json")
    for layer_idx in layers:
        print(f"[pilot] === layer {layer_idx} ({LAYER_ROLE.get(layer_idx, '?')}) ===")
        moe_block = freeze_all_but_layer(model, layer_idx)
        results[layer_idx] = {"role": LAYER_ROLE.get(layer_idx, "?")}
        for cond in CONDITIONS:
            sentences = condition_sentences[cond]
            real = real_fisher_for_condition(model, moe_block, sentences, tokenizer, n_experts, seqlen)
            proxy = proxy_for_condition(model, moe_block, sentences, tokenizer, n_experts, seqlen)
            results[layer_idx][cond] = {"real_fisher": real, "proxy": proxy}
            print(f"[pilot] layer {layer_idx} / {cond}: done "
                  f"(real_fisher nonzero experts={sum(1 for v in real if v > 0)}, "
                  f"proxy nonzero experts={sum(1 for v in proxy if v > 0)})")
            torch.cuda.empty_cache()
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[pilot] checkpoint-saved {out_path}")

    print(f"[pilot] wrote {out_path}")


if __name__ == "__main__":
    main()
