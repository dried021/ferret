"""Phase 1: merge (compress) DeepSeek-MoE-16B using D2MoE's own
Merge_deepseekMoE class directly, for a given calibration condition's
preprocessed artifacts (expert_freq / real Fisher), then evaluate
FLORES-200 perplexity per language (EN/KO/ZH).

Two deliberate scope cuts for this pilot (both documented in
00_docs/03_기술노트.md Phase 1, not silent):

1. Bypasses D2-deepseek.py's cfg/control_name/pruning-framework machinery
   (LLMPruner, run_lm_eval harness integration) -- that machinery is a
   generic multi-method pruning framework this repo is built on top of, and
   wiring IT correctly (i.e. flap_ratio()/model/llama_eri.py's instrumented
   Linear11) is its own project -- see phase1_structured_prune.py's module
   docstring for exactly why (concretely: it scans model.named_modules()
   for LLaMA/OPT-style "down_proj"/"o_proj" names that don't exist after
   Merge_deepseekMoE.merge_experts(), and its calibration-metric machinery
   needs Wmean_gate/down/up to be instances of a wrapper class this repo
   never actually constructs for DeepSeek). This script calls the exact same
   core API D2-deepseek.py calls per layer (`Merge_deepseekMoE(...).
   merge_experts(...)`, same class, same "fisher" merge_method) -- i.e. the
   actual Fisher-weighted-merge math our research question is about.
   pp_ratio structured pruning of the shared base IS available (2026-07-27)
   via --pp-ratio, using phase1_structured_prune.py's own reimplementation
   of the same FLAP-style metric instead of the vendored (and, for this
   architecture, non-functional) path -- see that module's docstring.

2. svd_scale=None (plain SVD for the delta factors, not D²-MoE's
   "activation-aware" whitened SVD) -- preprocess/get_scale.py (which
   computes that whitening scale) has a real GPU memory leak on this host
   (OOMs even at 4 samples/seqlen 128 after ~6 of 28 layers, independent of
   budget; see 00_docs/03_기술노트.md). merge_experts()'s
   delta_share_V=False/delta_share_U=False branch has an explicit
   `if svd_scale is not None: ... else: self.svd_delta(delta, ratio=...)`
   fallback to plain SVD for exactly this case, so DELTA_RATIO can still be
   the paper example's 0.8 -- only the whitening enhancement is skipped, not
   delta retention itself. (DELTA_RATIO=0 was tried first as a way to avoid
   get_scale.py entirely, but meanW_deltaUV never allocates delta_u/v
   modules when delta_ratio==0, and forward() unconditionally references
   them -- a structurally unsupported corner case in this repo, not
   something fixable by us without patching forward() itself.)

Usage:
    conda run -n d2moe_env python phase1_merge_eval.py --condition english_only [--smoke]
    conda run -n d2moe_env python phase1_merge_eval.py --baseline           # uncompressed model, once
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = SCRIPT_DIR.parent / "D2MoE"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(D2MOE_DIR))
import data_utils  # noqa: E402 -- FLORES loading + PPL helpers, reused as-is
import phase1_calib_data  # noqa: E402
import phase1_structured_prune  # noqa: E402 -- pp_ratio on/off ablation, see its module docstring

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
NUM_HIDDEN_LAYERS = 28
MOE_LAYERS = list(range(1, NUM_HIDDEN_LAYERS))

EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX = "israel/flores-parallel", "test", "sentence_"
EVAL_LANGS = {"eng_Latn": "English", "kor_Hang": "Korean", "zho_Hans": "Chinese", "swh_Latn": "Swahili",
              "ben_Beng": "Bengali"}
EVAL_N_SENTENCES = 60  # held-out FLORES devtest sentences per language

# DELTA_RATIO matches D2-deepseek.py's README example exactly; svd_scale=None
# is the deviation (see module docstring point 2).
DELTA_RATIO = 0.8
SHARE_RATIO = 1.0
MERGE_METHOD = "fisher"


def load_model(cpu_gib=0):
    """cpu_gib (2026-07-27, lossless test): lets accelerate spill excess
    layers to CPU RAM when a merged model doesn't fit in n_gpu*22GiB (e.g.
    delta_ratio=2.0 for the full-rank test uses ~2.5x bigger delta_u/delta_v
    per layer than the normal delta_ratio=0.8 pipeline was budgeted for).
    Defaults to 0 (no "cpu" key in max_memory) so every existing call site is
    unchanged. Safe here specifically because eval_flores_ppl's forward pass
    is @torch.no_grad() (data_utils.batch_sentence_nll) -- the known
    accelerate-offload bug (see phase1_fisher.py) is a BACKWARD/gradient-only
    failure mode and doesn't apply to a no-grad forward."""
    n_gpu = torch.cuda.device_count()
    max_memory = {i: "22GiB" for i in range(n_gpu)}
    if cpu_gib:
        max_memory["cpu"] = f"{cpu_gib}GiB"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", max_memory=max_memory, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def merge_condition(model, condition, seed, scale_condition=None, scale_seed=None, delta_ratio=None):
    """scale_condition/scale_seed (2026-07-26, Fisher x Scale 2x2): when given,
    loads SVD_scale from a DIFFERENT condition/seed's get_scale.py output than
    the Fisher/expert_freq data, so Fisher-calibration-language and
    whitening-calibration-language can be varied independently. Defaults to
    None (svd_scale=None, plain SVD) to keep every prior call to this
    function -- and every already-computed result -- unchanged.

    delta_ratio (2026-07-27, lossless test): overrides module-level
    DELTA_RATIO. svd_delta's `ratio` maps to rank via
    num_s = dim0*dim1*ratio/(dim0+dim1), NOT ratio*min(dim0,dim1) -- so
    ratio=1.0 is NOT full rank here (gives rank 834 of 1408 for this model's
    2048x1408 expert matrices). ratio must be >= ~1.6875 to reach full rank
    1408 (slicing S[:num_s] with num_s > len(S) just takes all of S, so
    over-shooting is harmless) -- see 00_docs/03_기술노트.md lossless test."""
    if delta_ratio is None:
        delta_ratio = DELTA_RATIO
    import os
    old_cwd = Path.cwd()
    os.chdir(D2MOE_DIR)  # model/model.py -> config.py reads './config.yml' by relative path
    try:
        from config import cfg
        from model.merge_deepseek import Merge_deepseekMoE
        # meanW_deltaUV (the delta-expert class) unconditionally constructs a
        # HiddenRepresentationPruning, which reads a few top-level cfg keys
        # that D2-deepseek.py's full pipeline only populates via
        # module/hyper.py::process_control() -- a large function (GPU-name
        # detection, prune_method string validation, mode checks) that's
        # part of the generic multi-method pruning framework we're
        # deliberately not wiring up (see module docstring). Copying just
        # the two keys HiddenRepresentationPruning.__init__ actually reads
        # avoids that whole subsystem; if a real forward pass through the
        # pruning path ever needs more, that'll surface as a clear KeyError.
        # Values match exactly what D2-deepseek.py's runExperiment() /
        # module/hyper.py::process_control() set before calling this same
        # merge_experts() path, for a merge_method="fisher" (no probe-based
        # activation-aware pruning) run -- found by grepping every cfg[...]
        # key model/merge_deepseek.py's forward path touches.
        cfg.setdefault("prune_metric", cfg["control"]["prune_metric"])
        cfg.setdefault("prune_ratio", 0.0)  # no pp_ratio structured pruning in this pilot
        cfg.setdefault("test_stage", False)
        cfg.setdefault("no_probe_process", False)
        # meanW_deltaUV.forward(): `if self.layer_order not in cfg['skip_layers']:` is
        # the ONLY branch that takes the probe/pp_ratio-pruning path; the `else` computes
        # the plain mean+delta formula with no dimension selection. Since this pilot does
        # no pp_ratio pruning at all (DELTA_RATIO covers the only compression axis we're
        # testing), every expert must take the `else`. Merge_deepseekMoE.__init__ never
        # actually passes `layer_order` when constructing experts (repo bug/incomplete
        # feature), so every expert's self.layer_order is the parameter's own default,
        # `[]` -- not an int. skip_layers must therefore contain the literal `[]`, not
        # layer indices, for `self.layer_order not in cfg['skip_layers']` to be False.
        cfg.setdefault("skip_layers", [[]])
        cfg.setdefault("calibration_stage", False)
        cfg.setdefault("batch_size", cfg["control"]["batch_size"])
        cfg.setdefault("mode", cfg["control"]["mode"])
        cfg.setdefault("prune_method", cfg["control"]["prune_method"])
        cfg.setdefault("tc_multiple", 64)
        cfg.setdefault("onlyprobe", False)
        cfg.setdefault("gate_probe_ratio", 1.0)  # not expected to be read when onlyprobe=False
        cfg.setdefault("up_probe_ratio", 1.0)
    finally:
        os.chdir(old_cwd)

    cond_dir = RESULTS_ROOT / condition / f"seed{seed}"
    expert_freq_paths = list(cond_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
    if not expert_freq_paths:
        raise FileNotFoundError(f"no expert_freq json in {cond_dir} -- run phase1_run_freq_and_scale.py first")
    expert_freq = json.loads(expert_freq_paths[0].read_text())

    fisher_path = cond_dir / "fisher_processed.pt"
    fisher_info = torch.load(fisher_path, map_location="cpu")

    # Prior pilots (delta_ratio=0.8, svd_scale=None) used plain SVD because
    # get_scale.py leaked memory (2 leaks found+fixed 2026-07-26), then hit a
    # third, unrelated bug (deterministic CUDA illegal-memory-access inside
    # DeepSeek's own vendored moe_infer, see 00_docs/03_기술노트.md "2)
    # whitening 복원") -- so the Fisher x Scale 2x2 uses phase1_svd_scale.py's
    # reimplementation (same scaling_diag_matrix statistic, computed via the
    # proven model(input_ids=...) forward path) instead of the vendored
    # script. scale_condition/scale_seed opts into loading it; svd_scale
    # stays None (plain SVD) by default for every prior call site.
    svd_scale = None
    if scale_condition is not None:
        scale_dir = RESULTS_ROOT / scale_condition / f"seed{scale_seed}"
        scale_path = scale_dir / "svd_scale_processed.pt"
        if not scale_path.exists():
            raise FileNotFoundError(f"no svd_scale_processed.pt in {scale_dir} -- run "
                                     f"phase1_svd_scale.py --condition {scale_condition} --seed {scale_seed} first")
        svd_scale = torch.load(scale_path, map_location="cpu")

    import gc

    for i in MOE_LAYERS:
        merge_block = Merge_deepseekMoE(
            model.config, share_ratio=SHARE_RATIO, delta_ratio=delta_ratio,
            expert_freq=expert_freq[str(i)], delta_share_V=False, delta_share_U=False,
            merge_method=MERGE_METHOD, shared_infer=False,
        ).to(model.model.layers[i].mlp.gate.weight.device)
        merge_block.merge_experts(
            model.model.layers[i].mlp,
            svd_scale=svd_scale[i] if svd_scale is not None else None,
            hessian=fisher_info[i], scale_type="svdllm",
        )
        # The original 64-expert mlp this replaces (~1.1GB in bf16) needs to
        # actually be freed before the next layer's merge math runs its own
        # transient 64-expert-wide tensor lists on top of an already ~31GB
        # model -- across 27 layers, relying on GC timing alone let this
        # grow until OOM (2026-07-24 incident). Explicit del + empty_cache
        # forces it each iteration instead of hoping the allocator gets to it.
        old_mlp = model.model.layers[i].mlp
        model.model.layers[i].mlp = merge_block
        del old_mlp
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[phase1-eval] layer {i}: merged")
    return model


def eval_flores_ppl(model, tokenizer):
    """Reports both per-token PPL (kept for continuity with the first pilot)
    and bits-per-byte (the metric to actually compare across languages on --
    per-token PPL is not comparable across languages when the tokenizer
    segments them at very different granularities. DeepSeek's tokenizer byte-
    falls-back on Hangul (observed ~0.70 tokens/byte for Korean vs ~0.21-0.24
    for English/Chinese in this pilot's baseline), which mechanically deflates
    Korean's per-token PPL regardless of true model quality -- see
    00_docs/03_기술노트.md "Phase 1 pilot" for the diagnostic that found this."""
    results = {}
    for lang_code, lang_name in EVAL_LANGS.items():
        sentences = data_utils.load_flores_sentences(EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX, lang_code, EVAL_N_SENTENCES)
        nll_sums, n_tokens = data_utils.evaluate_sentences(model, tokenizer, sentences, batch_size=8)
        ppl = data_utils.corpus_ppl(nll_sums, n_tokens)
        total_nll_nats = sum(nll_sums)
        total_bytes = sum(len(s.encode("utf-8")) for s in sentences)
        bpb = (total_nll_nats / math.log(2)) / total_bytes
        results[lang_code] = {
            "name": lang_name, "ppl": ppl, "n_sentences": len(sentences), "n_tokens": sum(n_tokens),
            "n_bytes": total_bytes, "bits_per_byte": bpb,
        }
        print(f"[phase1-eval] {lang_name}: PPL={ppl:.3f} over {sum(n_tokens)} tokens, bits_per_byte={bpb:.4f} over {total_bytes} bytes")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--baseline", action="store_true", help="evaluate the uncompressed model instead")
    parser.add_argument("--seed", type=int, default=0, help="must match the seed used for phase1_run_freq_and_scale.py / phase1_fisher.py")
    parser.add_argument("--scale-condition", default=None, help="2x2: load SVD_scale from a different condition than Fisher (default: none, plain SVD)")
    parser.add_argument("--scale-seed", type=int, default=None, help="seed for --scale-condition (defaults to --seed if --scale-condition is given)")
    parser.add_argument("--delta-ratio", type=float, default=None,
                         help="override DELTA_RATIO=0.8 (e.g. 2.0 for the full-rank/lossless-reconstruction test; "
                              "see merge_condition() docstring -- ratio=1.0 is NOT full rank for this model)")
    parser.add_argument("--cpu-offload-gib", type=int, default=0,
                         help="allow accelerate to spill this many GiB to CPU RAM if the merged model "
                              "doesn't fit in n_gpu*22GiB (only safe for no-grad forward paths, e.g. eval)")
    parser.add_argument("--pp-ratio", type=float, default=None,
                         help="§4.3(iii) on/off ablation: fraction of each layer's shared-base intermediate "
                              "channels to structurally prune post-merge (e.g. 0.2, D2-deepseek.py's own "
                              "default). Default None = off (no pruning, prior behavior unchanged). Uses "
                              "phase1_structured_prune.py's own FLAP-style reimplementation, NOT the vendored "
                              "flap_ratio() path -- see that module's docstring for why.")
    parser.add_argument("--pp-condition", default=None,
                         help="calibration condition for the pruning-importance calibration pass (default: "
                              "same as --condition). Lets pp_ratio's calibration language be varied "
                              "independently of Fisher's, mirroring --scale-condition, if that's ever needed; "
                              "§4.3(iii) as currently scoped keeps this equal to --condition (on/off only, "
                              "not a second 2x2).")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.baseline and not args.condition:
        parser.error("either --condition or --baseline is required")
    if args.scale_condition is not None and args.scale_seed is None:
        args.scale_seed = args.seed
    if args.pp_ratio is not None and args.pp_condition is None:
        args.pp_condition = args.condition

    global EVAL_N_SENTENCES
    if args.smoke:
        EVAL_N_SENTENCES = 5

    print(f"[phase1-eval] {'BASELINE (uncompressed)' if args.baseline else f'condition={args.condition}'} "
          f"smoke={args.smoke}")
    model, tokenizer = load_model(cpu_gib=args.cpu_offload_gib)

    if not args.baseline:
        effective_delta_ratio = args.delta_ratio if args.delta_ratio is not None else DELTA_RATIO
        scale_note = f" scale=({args.scale_condition},seed{args.scale_seed})" if args.scale_condition else " scale=none(plain SVD)"
        pp_note = f" pp_ratio={args.pp_ratio}(calib={args.pp_condition})" if args.pp_ratio is not None else " pp_ratio=off"
        print(f"[phase1-eval] merging with delta_ratio={effective_delta_ratio} share_ratio={SHARE_RATIO} "
              f"merge_method={MERGE_METHOD}{scale_note}{pp_note}")
        model = merge_condition(model, args.condition, args.seed, args.scale_condition, args.scale_seed,
                                 delta_ratio=args.delta_ratio)
        if args.pp_ratio is not None:
            phase1_structured_prune.apply_structured_pruning(
                model, args.pp_condition, args.seed, args.pp_ratio, tokenizer)

    results = eval_flores_ppl(model, tokenizer)

    # 2026-07-28: seen twice under heavy concurrent host load (3 local GPU jobs
    # at once) -- accelerate silently falls back to disk-offload (not cpu),
    # and the merged model then produces all-NaN logits with no exception
    # raised anywhere. Left unchecked this writes a garbage eval_ppl.json that
    # looks like a completed result. Fail loudly instead so the caller's
    # retry loop (local_gpu_driver.sh / pod*_gpu_driver.sh) treats this as a
    # transient failure and retries, the same as an OOM crash, rather than
    # silently accepting corrupt output.
    nan_langs = [name for name, r in results.items() if r["bits_per_byte"] != r["bits_per_byte"]]
    if nan_langs:
        raise RuntimeError(
            f"eval produced NaN bits_per_byte for {nan_langs} -- likely disk-offload under host memory "
            "pressure (check for 'offloaded to the disk' in this run's log), not a real result. Refusing "
            "to write eval_ppl.json; let the caller retry.")

    if args.baseline:
        out_dir = RESULTS_ROOT / "baseline"
    elif args.scale_condition is not None:
        out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}" / f"scale_{args.scale_condition}_seed{args.scale_seed}"
    else:
        out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
    if args.delta_ratio is not None:
        out_dir = out_dir / f"delta_ratio_{args.delta_ratio}"
    if args.pp_ratio is not None:
        out_dir = out_dir / f"pp_ratio_{args.pp_ratio}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("eval_ppl_smoke.json" if args.smoke else "eval_ppl.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[phase1-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
