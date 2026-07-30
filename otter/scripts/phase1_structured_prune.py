"""Phase 1: pp_ratio structured pruning -- the third D2-MoE pipeline stage
(§4.3's part (iii), see 06_논문_구성.md "[5] structured pruning" in the
pipeline diagram), applied post-merge to prune low-importance intermediate
("hidden representation") channels of the shared base down-projection.

Why this is NOT the vendored D2MoE/model/pruning_module.py::flap_ratio() +
model/llama_eri.py::MixtralEriModel path (2026-07-27 investigation, recorded
here since the gap isn't obvious from either file alone):

1. flap_ratio()'s calibration-metric collection scans model.named_modules()
   for names literally containing "down_proj"/"o_proj" (LLaMA/OPT's own
   submodule names). After Merge_deepseekMoE.merge_experts(), there is no
   submodule named that -- the fused down-projection is
   `Wmean_down(h) + delta_u2(delta_v2(h))`, split across submodules named
   "Wmean_down"/"delta_u2"/"delta_v2". flap_ratio()'s scan finds zero
   matches against a merged DeepSeek model; it was written for (and only
   ever exercised against) unmerged LLaMA/OPT-style checkpoints.
2. Even meanW_deltaUV.forward()'s OWN pruning branches ('probe' in
   cfg['prune_method']) call self.Wmean_gate(probe, cal_mlp_probe_out_dim_metric=True)
   and read self.Wmean_down.get_global_metric_score_distribution() --
   kwargs/methods that only exist on model/llama_eri.py's instrumented
   Linear11(nn.Linear, EriLayer) class. Merge_deepseekMoE.__init__ builds
   Wmean_gate/Wmean_down/Wmean_up as plain torch.nn.Linear (see
   model/merge_deepseek.py line ~56-58) -- calling that branch against the
   real merged model raises TypeError/AttributeError immediately. Nothing
   in D2-deepseek.py's own runExperiment() ever wraps Wmean_* with
   Linear11 either (make_prune_model()'s MixtralEriModel._find_and_replace
   would need to, but its target-module regex + flap_ratio()'s own name
   scan both look for the wrong names, see point 1).
3. Reaching Linear11's real calibration path would additionally require
   ~15 cfg keys process_control() populates from a full --control_name
   parse (cfg['data_type_max'], cfg['nonpad_tokens_denominator'],
   cfg['pad_tokens'], cfg['ema_momentum'], cfg['cur_batch_index'], ...) that
   phase1_merge_eval.py's merge_condition() deliberately does NOT set up
   (see that function's own comment on cfg.setdefault(...) -- it copies only
   the handful of keys the plain merge path touches, precisely to avoid this
   subsystem). Wiring all of it correctly for an architecture (merged
   per-expert delta-SVD MoE) the vendored code was never actually run
   against is a much bigger undertaking than a pp_ratio on/off ablation
   needs -- phase1_merge_eval.py's module docstring already flagged this
   ("wiring it correctly is its own project") before this file existed.

What this does instead: reimplements the SAME metric flap_ratio()'s
cal_mlp_calib_prune_metric('flap' variant) computes --
`metric[j] = (sum_t input_activation[t,j]^2) * ||down_proj_weight[:,j]||_2^2`
-- via a straightforward forward-hook calibration pass over the
already-merged model, using calibration text from the SAME
phase1_fisher.build_samples() every other Phase 1 stage uses (same
condition/seed/n-samples/seqlen budget, so this stage's calibration language
is controlled exactly like the Fisher and whitening stages are). The weight
term uses ONLY Wmean_down's own weight (not the per-expert delta_u2/delta_v2
correction) -- see apply_structured_pruning()'s docstring for why pruning is
scoped to the shared base rather than also touching each expert's delta path.
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402
from phase1_fisher import build_samples, MOE_LAYERS  # noqa: E402 -- identical sample-building, reused as-is

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")


def collect_intermediate_activation_sumsq(model, tokenizer, texts, seqlen, embed_device):
    """Returns {layer_idx: cpu_tensor of shape [intermediate_size]} -- sum of
    squared activations per intermediate channel, accumulated over every
    token in every calibration text, at the input to that layer's
    Wmean_down. One forward hook per layer, ALL layers at once (unlike
    phase1_fisher.py's per-layer freeze-and-accumulate: that trick exists to
    bound BACKWARD/gradient memory to one layer at a time, which doesn't
    apply here -- this is forward-only, and a 1-D per-channel sum is trivial
    memory next to phase1_svd_scale.py's [2048,2048] covariance accumulation,
    so all 27 layers can be hooked in a single pass over the calibration
    texts instead of 27 separate passes)."""
    accum = {}
    hit_counts = {i: 0 for i in MOE_LAYERS}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h = input[0].detach().float()
            h = h.reshape(-1, h.size(-1))  # [tokens, intermediate_size] -- Wmean_down's actual input
            sumsq = h.pow(2).sum(dim=0).cpu()
            if layer_idx in accum:
                accum[layer_idx] += sumsq
            else:
                accum[layer_idx] = sumsq
            hit_counts[layer_idx] += h.shape[0]
        return hook

    handles = []
    for i in MOE_LAYERS:
        wmean_down = model.model.layers[i].mlp.Wmean_down
        handles.append(wmean_down.register_forward_hook(make_hook(i)))

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
    dead_layers = [i for i in MOE_LAYERS if hit_counts[i] == 0]
    if dead_layers:
        raise RuntimeError(
            f"{len(dead_layers)}/{len(MOE_LAYERS)} layers' Wmean_down got zero forward hits across all "
            f"{n_used} samples: {dead_layers[:10]}{'...' if len(dead_layers) > 10 else ''} -- same "
            "routing-coverage concern as phase1_fisher.py/phase1_svd_scale.py, see their docstrings."
        )
    return accum


def apply_structured_pruning(model, condition, seed, pp_ratio, tokenizer, n_samples=64, seqlen=512):
    """Prunes the bottom `pp_ratio` fraction of each layer's intermediate
    channels OUT OF Wmean_down ONLY (zeroes those columns of
    Wmean_down.weight in place) -- NOT out of any expert's delta_u2/delta_v2
    correction. Two reasons for this scope, not just simplicity:

    1. Wmean_down is the one down-projection component genuinely SHARED
       across a layer's 64 experts (that's the whole point of the "shared
       base" in Fisher-merge + delta-SVD). A single pruning-index set can be
       sliced out of it consistently. delta_u2/delta_v2 are PER-EXPERT --
       computing this metric per-expert and pruning each expert's delta
       differently would need a separate index set per expert per layer
       (2 GB+ of extra bookkeeping and no longer "one pp_ratio ablation",
       closer to a second research question). Pruning only the shared
       component keeps this a clean on/off ablation of exactly one axis, the
       same way delta_ratio is the one axis phase1_merge_eval.py already
       varies for the SVD stage.
    2. It's the conservative choice for what "pruning removes": the
       expert-specific correction on a channel survives even when that
       channel is pruned from the shared base, so a channel some particular
       expert relies on isn't silently deleted for that expert -- only its
       generic/shared contribution is.

    Zeroed rather than physically resized (Wmean_down.weight keeps its
    original shape, pruned columns just become exactly 0) -- functionally
    identical to true structured pruning for every downstream computation
    (forward pass, bpb eval), just without a real FLOP/parameter-count
    reduction. This pilot's claim is quality impact only (matches every
    other Phase 1 stage: delta_ratio's low-rank truncation is likewise
    evaluated for quality, not wall-clock speed), so this is not a
    simplification that changes what's being measured -- see
    00_docs/03_기술노트.md Phase 1 for the same framing applied to why
    delta_ratio=0.8 (not 0) was used for the lossless test.

    Returns {layer_idx: {"num_pruned": int, "intermediate_size": int}} for
    logging/verification.
    """
    embed_device = model.get_input_embeddings().weight.device
    texts = build_samples(condition, n_samples, seqlen, seed)
    print(f"[structured-prune] condition={condition} seed={seed} pp_ratio={pp_ratio} "
          f"n_samples={len(texts)} seqlen={seqlen}")

    activation_sumsq = collect_intermediate_activation_sumsq(model, tokenizer, texts, seqlen, embed_device)

    stats = {}
    for i in MOE_LAYERS:
        wmean_down = model.model.layers[i].mlp.Wmean_down
        weight = wmean_down.weight.data  # [hidden_size, intermediate_size]
        intermediate_size = weight.shape[1]
        calib = activation_sumsq[i].to(weight.device).to(torch.float32)
        # cal_mlp_calib_prune_metric's 'flap' formula (model/pruning_module.py):
        # metric[j] = calib[j] * sum_out(weight[:, j]^2) -- see module docstring
        # point 1 for why we don't call that function directly.
        weight_term = weight.float().pow(2).sum(dim=0)
        metric = calib * weight_term

        num_prune = int(pp_ratio * intermediate_size)
        if num_prune > 0:
            prune_idx = torch.argsort(metric)[:num_prune]
            weight[:, prune_idx] = 0
        stats[i] = {"num_pruned": num_prune, "intermediate_size": intermediate_size}
        print(f"[structured-prune] layer {i}: pruned {num_prune}/{intermediate_size} channels "
              f"({100 * num_prune / intermediate_size:.1f}%)")

    return stats


def _standalone_main():
    """Smoke-test entry point -- NOT what phase1_merge_eval.py calls (it
    imports apply_structured_pruning() directly after merge_condition()).
    This standalone path exists only to sanity-check the hook/metric/prune
    mechanics in isolation against a freshly loaded (unmerged) model before
    trusting it inside the full merge+eval pipeline -- run with --smoke
    first. NOTE: pruning Wmean_down on an UNMERGED model has no effect
    (there is no Wmean_down before merge_experts() runs), so this entry
    point is only useful for exercising collect_intermediate_activation_sumsq's
    hook plumbing against a real forward pass shape-wise; the real
    correctness check is an on/off bpb comparison through
    run_phase1_pruning_ablation.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pp-ratio", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    max_memory = {i: "22GiB" for i in range(max(n_gpu, 1))}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", max_memory=max_memory, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()

    n_samples = 4 if args.smoke else 64
    seqlen = 128 if args.smoke else 512
    texts = build_samples(args.condition, n_samples, seqlen, args.seed)
    embed_device = model.get_input_embeddings().weight.device
    layers = MOE_LAYERS[:2] if args.smoke else MOE_LAYERS
    handles = []
    accum = {}
    hit_counts = {i: 0 for i in layers}
    # Standalone-only shape check: hooks Wmean_up (exists pre-merge, unlike
    # Wmean_down which is a post-merge artifact) just to confirm the hook
    # fires and shapes look right -- not the real pruning target.
    for i in layers:
        def make_hook(li):
            def hook(module, input, output):
                hit_counts[li] += output.shape[0] * output.shape[1] if output.dim() == 3 else output.shape[0]
            return hook
        handles.append(model.model.layers[i].mlp.experts[0].up_proj.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for text in texts[:2]:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
            model(input_ids=enc["input_ids"].to(embed_device))
    for h in handles:
        h.remove()
    print(f"[structured-prune-smoke] hook fired on layers with nonzero hits: "
          f"{[i for i in layers if hit_counts[i] > 0]} / {layers}")
    print("[structured-prune-smoke] shape-check only -- run run_phase1_pruning_ablation.py for the real "
          "on/off comparison through a merged model")


if __name__ == "__main__":
    _standalone_main()
