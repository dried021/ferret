"""Phase 1: real element-wise gradient Fisher, for ALL 27 MoE layers, for one
calibration condition -- the artifact D2-deepseek.py's merge step actually
needs (Fisher Pilot A's expert-scalar version was only for the proxy-vs-real
validation question, not for merging).

Same per-layer freeze trick as fisher_pilot_a.py (see its docstring for why):
only the target layer's expert parameters get requires_grad=True at a time,
so gradient memory stays ~O(one layer) instead of ~O(whole model). Unlike
Pilot A, this KEEPS the full per-parameter gradient tensor (not collapsed to
a scalar) since D2-deepseek.py's Merge_deepseekMoE.merge_experts() needs
per-weight Fisher for its weighted-merge math -- and it's accumulated on
CPU throughout (not GPU) for the same reason get_fisher.py's original
GPU-resident accumulator was part of what OOM'd.

Output format matches what D2-deepseek.py expects to torch.load() directly:
{layer_idx: {"mlp.experts.<e>.gate_proj": tensor, ...}} (post_process_fisher_info's
shape in the original get_fisher.py), so no further conversion is needed
before running the actual merge step.

Checkpointed after every layer (a 128-sample x 27-layer run is not cheap --
see 00_docs/03_기술노트.md Phase 1 timing notes) so a smoke run or an
interruption doesn't lose completed layers.

Usage:
    conda run -n d2moe_env python phase1_fisher.py --condition english_only [--smoke]
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
NUM_HIDDEN_LAYERS = 28  # 0 = dense, 1..27 = MoE (see 00_docs/03_기술노트.md Phase 0)
MOE_LAYERS = list(range(1, NUM_HIDDEN_LAYERS))

# Below this many routed samples, a per-expert statistic is real but noisy --
# flagged (not replaced) for paper transparency. Reuses the threshold already
# established elsewhere in this codebase for the same judgment call
# (scan_disagreement_experts.py, make_figure_expert_language_heatmap.py,
# disagreement_common.py all default min_hit_count=5) so "low-confidence
# routing coverage" means the same thing project-wide, not a per-script value.
LOW_HIT_THRESHOLD = 5


def build_samples(condition, n_samples, seqlen, seed):
    """Short, batch-size-1 samples -- random windows over the condition's
    seed-shuffled sentence pool, joined the same way get_calib_data does."""
    import random

    sentences = phase1_calib_data.build_condition_sentences(condition, seed)
    tot_text = "\n\n".join(sentences)
    rng = random.Random(seed + 1)  # offset from the sentence-shuffle seed
    samples = []
    max_start = max(1, len(tot_text) - seqlen * 4 - 1)
    for _ in range(n_samples):
        i = rng.randint(0, max_start)
        j = i + seqlen * 4
        samples.append(tot_text[i:j])
    return samples


def freeze_all_but_layer(model, layer_idx):
    for p in model.parameters():
        p.requires_grad_(False)
    moe_block = model.model.layers[layer_idx].mlp
    for p in moe_block.experts.parameters():
        p.requires_grad_(True)
    return moe_block


def elementwise_fisher_for_layer(model, moe_block, tokenizer, texts, seqlen, embed_device, layer_idx):
    """Returns ({module_name: cpu_tensor (bf16)}, flags), module_name like
    "mlp.experts.<e>.gate_proj" -- matches D2-deepseek.py's expected key
    shape. Accumulates in fp32 on CPU for numerical safety, casts down to
    bf16 only in the returned/saved copy (each layer's dict is ~2.2GB in
    fp32, same order of magnitude as that layer's own weights -- storing
    27 of these per condition adds up, so bf16-on-disk matters).

    An expert with zero gradient across every sample used to always raise --
    that guard exists because a 2026-07-24 incident let a layer whose
    parameters got accelerate-offloaded to CPU/disk (backward through
    offloaded params silently produces no .grad, no error) save an
    empty/incomplete layer with no indication anything was wrong until the
    downstream merge step KeyError'd on a missing expert. See
    00_docs/03_기술노트.md Phase 1.

    2026-07-28: that incident is specifically an offload artifact, not
    routing sparsity itself, so we now check which one actually happened
    before deciding: if any of this layer's expert params are NOT on a CUDA
    device, the offload bug is live and we still raise immediately (same
    protection as before, the real fix is still more GPU memory). Only when
    every expert param is confirmed on-GPU do we trust a zero-hit count as
    genuine routing sparsity -- routing an MoE model never selects a given
    expert for some calibration seeds/conditions, which is a real, expected
    (if inconvenient) property of the router, not a bug -- and fall back to
    F=0 (an all-zero Fisher block: "no evidence this expert matters here",
    the same null answer phase1_svd_scale.py's identity-whitening fallback
    encodes for the same situation). Experts with 0 < hits < LOW_HIT_THRESHOLD
    keep their real (noisy) statistic -- there IS data, just thin -- but are
    flagged the same way, for the same reason the rest of this codebase
    already flags (not discards) hit_count < 5 cells (see LOW_HIT_THRESHOLD)."""
    accum = {}
    per_expert_hits = [0] * len(moe_block.experts)
    n_used = 0
    model.train()
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
        input_ids = enc["input_ids"].to(embed_device)
        if input_ids.shape[1] < 2:
            continue
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        for e_idx, expert in enumerate(moe_block.experts):
            got_grad_this_sample = False
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(expert, proj_name).weight
                if w.grad is None:
                    continue
                got_grad_this_sample = True
                key = f"mlp.experts.{e_idx}.{proj_name}"
                g2 = w.grad.detach().float().pow(2).cpu()
                if key in accum:
                    accum[key] += g2
                else:
                    accum[key] = g2
            if got_grad_this_sample:
                per_expert_hits[e_idx] += 1
        model.zero_grad(set_to_none=True)
        n_used += 1
    if n_used == 0:
        raise RuntimeError("no usable samples (all shorter than 2 tokens)")

    dead_experts = [e for e, hits in enumerate(per_expert_hits) if hits == 0]
    flags = []
    if dead_experts:
        offloaded = sorted({
            e_idx for e_idx in dead_experts
            for p in (getattr(moe_block.experts[e_idx], n).weight for n in ("gate_proj", "up_proj", "down_proj"))
            if p.device.type != "cuda"
        })
        if offloaded:
            raise RuntimeError(
                f"{len(dead_experts)}/{len(moe_block.experts)} experts got zero gradient across all "
                f"{n_used} samples: {dead_experts[:10]}{'...' if len(dead_experts) > 10 else ''}, and "
                f"{len(offloaded)} of them ({offloaded[:10]}) are NOT on a CUDA device right now -- this "
                "looks like the 2026-07-24 accelerate-offload gradient bug, not routing sparsity. Raise "
                "max_memory so the model doesn't spill to CPU/disk -- see 00_docs/03_기술노트.md Phase 1."
            )
        # Confirmed on-GPU with zero hits -> genuine routing sparsity for this
        # seed/condition. F=0: no gradient evidence, so no contribution to the merge.
        # (accum can only be empty if literally every expert in the layer is
        # dead -- implausible for a real MoE router, but fall back to a fixed
        # dtype/device rather than crash on next(iter(...)) if it ever happens.)
        example_w = next(iter(accum.values())) if accum else torch.zeros(1, dtype=torch.float32, device="cpu")
        for e_idx in dead_experts:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                key = f"mlp.experts.{e_idx}.{proj_name}"
                w = getattr(moe_block.experts[e_idx], proj_name).weight
                accum[key] = torch.zeros_like(w, dtype=example_w.dtype, device=example_w.device)
            flags.append({"layer": layer_idx, "expert": e_idx, "hit_count": 0, "kind": "dead_fallback_F0", "source": "fisher"})
            print(f"[phase1-fisher] layer {layer_idx} expert {e_idx}: 0/{n_used} hits (confirmed on-GPU, "
                  "genuine routing sparsity), falling back to F=0")

    for e_idx, hits in enumerate(per_expert_hits):
        if 0 < hits < LOW_HIT_THRESHOLD:
            flags.append({"layer": layer_idx, "expert": e_idx, "hit_count": hits, "kind": "low_hit_flagged", "source": "fisher"})
            print(f"[phase1-fisher] layer {layer_idx} expert {e_idx}: {hits}/{n_used} hits (< "
                  f"{LOW_HIT_THRESHOLD}), keeping real statistic but flagging as low-confidence")

    return {key: val.div(n_used).to(torch.bfloat16) for key, val in accum.items()}, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_samples = 4 if args.smoke else args.n_samples
    seqlen = 128 if args.smoke else args.seqlen
    layers = MOE_LAYERS[:2] if args.smoke else MOE_LAYERS

    out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
    suffix = "_smoke" if args.smoke else ""
    # One file per layer (each ~1.1GB at bf16) instead of re-saving one
    # ever-growing dict every layer -- the first version of this script
    # resaved the full accumulated dict on every checkpoint, so total write
    # volume grew O(n_layers^2) (~800GB of actual disk I/O for one 27-layer
    # condition). assemble_fisher_dict() below combines these into the
    # single file D2-deepseek.py expects, once, after all layers are done.
    layer_dir = out_dir / f"fisher_layers{suffix}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fisher_processed{suffix}.pt"
    # Routing-coverage paper material (2026-07-28): dead/low-hit experts flagged
    # by elementwise_fisher_for_layer, written incrementally (survives a
    # resumed/interrupted run same as the per-layer checkpoints above).
    flags_path = out_dir / "routing_coverage_flags.json"

    n_gpu = torch.cuda.device_count()
    max_memory = {i: "22GiB" for i in range(n_gpu)}
    print(f"[phase1-fisher] condition={args.condition} smoke={args.smoke} layers={layers} "
          f"n_samples={n_samples} seqlen={seqlen} max_memory={max_memory}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", max_memory=max_memory, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    embed_device = model.get_input_embeddings().weight.device

    print("[phase1-fisher] building calibration samples ...")
    texts = build_samples(args.condition, n_samples, seqlen, args.seed)
    calib_hash = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
    print(f"[phase1-fisher] {len(texts)} samples ready, calib_text_hash={calib_hash} "
          f"first_sample_preview={texts[0][:60]!r}")

    for layer_idx in layers:
        layer_path = layer_dir / f"layer_{layer_idx}.pt"
        if layer_path.exists():
            print(f"[phase1-fisher] layer {layer_idx}: already done ({layer_path.name}), skipping")
            continue
        t0 = time.time()
        moe_block = freeze_all_but_layer(model, layer_idx)
        layer_fisher, flags = elementwise_fisher_for_layer(model, moe_block, tokenizer, texts, seqlen, embed_device, layer_idx)
        torch.save(layer_fisher, layer_path)
        if flags:
            import json
            existing = json.loads(flags_path.read_text()) if flags_path.exists() else {}
            existing[str(layer_idx)] = [f for f in existing.get(str(layer_idx), []) if f.get("source") != "fisher"] + flags
            flags_path.write_text(json.dumps(existing, indent=2))
        dt = time.time() - t0
        n_done = len(list(layer_dir.glob("layer_*.pt")))
        print(f"[phase1-fisher] layer {layer_idx}: done in {dt:.1f}s, wrote {layer_path.name} "
              f"({n_done}/{len(layers)} layers)")
        torch.cuda.empty_cache()

    n_done = len(list(layer_dir.glob("layer_*.pt")))
    if n_done < len(layers):
        print(f"[phase1-fisher] {n_done}/{len(layers)} layers done -- not assembling {out_path} yet "
              f"(rerun this script to resume the rest)")
        return

    print(f"[phase1-fisher] all {len(layers)} layers done, assembling {out_path} ...")
    assembled = {}
    for layer_idx in layers:
        assembled[layer_idx] = torch.load(layer_dir / f"layer_{layer_idx}.pt", map_location="cpu")
    torch.save(assembled, out_path)
    print(f"[phase1-fisher] wrote {out_path}")


if __name__ == "__main__":
    main()
