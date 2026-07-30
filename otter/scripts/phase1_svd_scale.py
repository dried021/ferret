"""Phase 1: reimplementation of get_scale.py's activation-aware SVD scale
statistic (scaling_diag_matrix = sum of X^T X over calibration tokens, per
expert Linear submodule), for ALL 27 MoE layers, one calibration condition
at a time.

Why this exists (2026-07-26, see 00_docs/03_기술노트.md "2) whitening 복원"):
the vendored D2MoE/preprocess/get_scale.py had two real GPU memory leaks
(found and fixed) and, after those were fixed, a THIRD, unrelated bug: a
deterministic "CUDA error: illegal memory access" inside DeepSeek's own
vendored moe_infer's `scatter_reduce_` call, reproducing at the exact same
layer/timestamp across independent runs (seed=0 fixed). That crash lives
inside get_scale.py's own per-layer, batch-size-1, no-attention-mask forward
mechanism (`layer(inps[j].unsqueeze(0))[0]`) -- a nonstandard forward path
get_scale.py invented to chain activations layer-by-layer without rerunning
the whole model. phase1_fisher.py's `model(input_ids=...)` standard forward
call has run cleanly, per-layer, across EN/KO/ZH/Swahili/placebo conditions
(dozens of successful runs) -- so this script computes the exact same
scaling_diag_matrix statistic using THAT proven forward path instead,
avoiding the vendored bug entirely rather than debugging it further.

No backward pass is needed here (unlike phase1_fisher.py) -- just forward
activations -- so this runs under @torch.no_grad() and skips the
freeze/requires_grad dance entirely; expected to be faster than Fisher.

Output format matches what merge_experts() expects for `svd_scale`:
{layer_idx: {"mlp.experts.<e>.gate_proj": cholesky_matrix, ...}}, same keys
as get_scale.py's own saved dict, so phase1_merge_eval.py's --scale-condition
loading path works against this file the same way (see SCALE_FILENAME).

Usage:
    conda run -n d2moe_env python phase1_svd_scale.py --condition english_only [--smoke]
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
from phase1_fisher import build_samples, MOE_LAYERS  # noqa: E402 -- identical sample-building, reused as-is

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SCALE_FILENAME = "svd_scale_processed.pt"  # phase1_merge_eval.py looks for this first (see its --scale-condition path)

# See phase1_fisher.py's LOW_HIT_THRESHOLD -- same value, same reasoning,
# reused project-wide (scan_disagreement_experts.py, make_figure_expert_
# language_heatmap.py, disagreement_common.py all default min_hit_count=5).
LOW_HIT_THRESHOLD = 5


def process_scaling_matrix(raw_matrix, device):
    """Symmetrize + eigenvalue-clamp + Cholesky -- identical math to
    get_scale.py's process_scaling_matrix(). raw_matrix is accumulated on CPU
    (see elementwise_scale_for_layer) to avoid the GPU-resident-forever leak
    that hit get_scale.py, but eigh/Cholesky on a [2048,2048] matrix is far
    cheaper on GPU than CPU (~60s/layer vs a few seconds in an early smoke
    test) -- so this brings just ONE matrix at a time onto GPU for the
    decomposition, then the caller moves the small bf16 result back to CPU.
    Transient, not accumulated, so no leak risk."""
    raw_matrix = raw_matrix.to(device)
    matrix = (raw_matrix + raw_matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if eigenvalues.min().item() <= 0:
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)
        matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return torch.linalg.cholesky(matrix)


def elementwise_scale_for_layer(model, moe_block, tokenizer, texts, seqlen, embed_device, layer_idx):
    """Returns ({module_name: cpu_tensor (bf16)}, flags), module_name like
    "mlp.experts.<e>.gate_proj". Accumulates X^T X in fp64 on CPU (matches
    get_scale.py's .double() before Cholesky, and keeps this script's own
    memory flat across layers -- the same CPU-offload discipline that fixed
    get_scale.py's second leak).

    2026-07-28: an expert with zero forward hits across every calibration
    sample used to raise (see phase1_fisher.py's docstring for why THAT
    script is cautious about zero-hit experts -- an accelerate-offload bug
    can silently produce the same symptom there). That risk doesn't apply
    here: this runs entirely under @torch.no_grad() with no backward pass,
    so a forward hook simply not firing means the router genuinely never
    selected this expert for any of this seed's calibration text -- real
    routing sparsity, not an artifact. Falls back to an identity scaling
    matrix per missing (expert, projection) -- process_scaling_matrix's own
    eigenvalue-clamp+Cholesky of an identity is just the identity, i.e. this
    expert's SVD merge gets no activation-aware whitening, which is this
    project's own existing "plain SVD, whitening off" convention (see
    01_plans/06_불일치표적배분_실측결과.md), just applied to one expert
    instead of the whole run. Experts with 0 < hits < LOW_HIT_THRESHOLD keep
    their real (noisy) statistic but are flagged the same way."""
    accum = {}
    hit_counts = [0] * len(moe_block.experts)

    def make_hook(e_idx, proj_name, key):
        def hook(module, input, output):
            inp = input[0].detach().float()
            inp = inp.reshape(-1, inp.size(-1))
            adds = torch.matmul(inp.transpose(0, 1), inp).double().cpu()
            if key in accum:
                accum[key] += adds
            else:
                accum[key] = adds
            hit_counts[e_idx] += 1
        return hook

    handles = []
    for e_idx, expert in enumerate(moe_block.experts):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            key = f"mlp.experts.{e_idx}.{proj_name}"
            handles.append(getattr(expert, proj_name).register_forward_hook(make_hook(e_idx, proj_name, key)))

    model.eval()
    n_used = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
            input_ids = enc["input_ids"].to(embed_device)
            if input_ids.shape[1] < 2:
                continue
            model(input_ids=input_ids)
            n_used += 1

    for h in handles:
        h.remove()

    if n_used == 0:
        raise RuntimeError("no usable samples (all shorter than 2 tokens)")

    dead_experts = [e for e, hits in enumerate(hit_counts) if hits == 0]
    flags = []
    for e_idx in dead_experts:
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            key = f"mlp.experts.{e_idx}.{proj_name}"
            in_features = getattr(moe_block.experts[e_idx], proj_name).weight.shape[1]
            accum[key] = torch.eye(in_features, dtype=torch.float64)
        flags.append({"layer": layer_idx, "expert": e_idx, "hit_count": 0, "kind": "dead_fallback_identity", "source": "svd_scale"})
        print(f"[phase1-svd-scale] layer {layer_idx} expert {e_idx}: 0/{n_used} hits, falling back to "
              "identity scaling matrix (no whitening for this expert)")

    for e_idx, hits in enumerate(hit_counts):
        if 0 < hits < LOW_HIT_THRESHOLD:
            flags.append({"layer": layer_idx, "expert": e_idx, "hit_count": hits, "kind": "low_hit_flagged", "source": "svd_scale"})
            print(f"[phase1-svd-scale] layer {layer_idx} expert {e_idx}: {hits}/{n_used} hits (< "
                  f"{LOW_HIT_THRESHOLD}), keeping real statistic but flagging as low-confidence")

    return {key: process_scaling_matrix(raw, embed_device).to(torch.bfloat16).cpu() for key, raw in accum.items()}, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_samples = 4 if args.smoke else args.n_samples
    seqlen = 128 if args.smoke else args.seqlen
    layers = MOE_LAYERS[:2] if args.smoke else MOE_LAYERS

    out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
    suffix = "_smoke" if args.smoke else ""
    layer_dir = out_dir / f"svd_scale_layers{suffix}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"svd_scale_processed{suffix}.pt"
    # See phase1_fisher.py's flags_path -- same convention, same file name, so
    # a condition/seed's dead+low-hit experts from both scripts land together.
    flags_path = out_dir / "routing_coverage_flags.json"

    n_gpu = torch.cuda.device_count()
    max_memory = {i: "22GiB" for i in range(n_gpu)}
    print(f"[phase1-svd-scale] condition={args.condition} smoke={args.smoke} layers={layers} "
          f"n_samples={n_samples} seqlen={seqlen} max_memory={max_memory}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", max_memory=max_memory, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    embed_device = model.get_input_embeddings().weight.device

    print("[phase1-svd-scale] building calibration samples ...")
    texts = build_samples(args.condition, n_samples, seqlen, args.seed)
    calib_hash = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
    print(f"[phase1-svd-scale] {len(texts)} samples ready, calib_text_hash={calib_hash} "
          f"first_sample_preview={texts[0][:60]!r}")

    for layer_idx in layers:
        layer_path = layer_dir / f"layer_{layer_idx}.pt"
        if layer_path.exists():
            print(f"[phase1-svd-scale] layer {layer_idx}: already done ({layer_path.name}), skipping")
            continue
        t0 = time.time()
        moe_block = model.model.layers[layer_idx].mlp
        layer_scale, flags = elementwise_scale_for_layer(model, moe_block, tokenizer, texts, seqlen, embed_device, layer_idx)
        torch.save(layer_scale, layer_path)
        if flags:
            import json
            existing = json.loads(flags_path.read_text()) if flags_path.exists() else {}
            existing[str(layer_idx)] = [f for f in existing.get(str(layer_idx), []) if f.get("source") != "svd_scale"] + flags
            flags_path.write_text(json.dumps(existing, indent=2))
        dt = time.time() - t0
        n_done = len(list(layer_dir.glob("layer_*.pt")))
        print(f"[phase1-svd-scale] layer {layer_idx}: done in {dt:.1f}s, wrote {layer_path.name} "
              f"({n_done}/{len(layers)} layers)")
        torch.cuda.empty_cache()

    n_done = len(list(layer_dir.glob("layer_*.pt")))
    if n_done < len(layers):
        print(f"[phase1-svd-scale] {n_done}/{len(layers)} layers done -- not assembling {out_path} yet "
              f"(rerun this script to resume the rest)")
        return

    print(f"[phase1-svd-scale] all {len(layers)} layers done, assembling {out_path} ...")
    assembled = {}
    for layer_idx in layers:
        assembled[layer_idx] = torch.load(layer_dir / f"layer_{layer_idx}.pt", map_location="cpu")
    torch.save(assembled, out_path)
    print(f"[phase1-svd-scale] wrote {out_path}")


if __name__ == "__main__":
    main()
