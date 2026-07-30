"""Phase 0.5: does the Toy0 (2026-07-23) pattern reproduce with more seeds
and a token-budget-controlled sample, before scaling up to Phase 0/1?

Same conditions, same metric, same 3 layers-of-interest as Toy0, plus 3 more
in between (see data/phase0_5_config.yaml) -- no new languages, no new
analysis. Differences from Toy0, per the reproducibility-check plan:

  - conditions are sampled by tokenizer token budget, not sentence count
    (data_utils.token_budget_chunks) -- the biggest confound isn't language,
    it's tokenization/sentence-length, so token count must match exactly
    across conditions (balanced splits the same total budget 3 ways).
  - en_only/en_only_control are renamed english_a/english_b (same role: the
    same-language "placebo" noise floor) so it reads directly off figures.
  - the whole thing repeats for multiple seeds (each reshuffling the FLORES
    pool independently), so 02b_analyze_repro.py can check whether Toy0's
    layer-8-FAIL / layer-22,38-PASS pattern is a stable trend or noise.

Usage:
    conda run -n torch_env python 01b_phase0_5_stats.py [--smoke]
    conda run -n torch_env python 01b_phase0_5_stats.py --config ../data/layer_locality_config.yaml --prefix layer_locality

--smoke uses a 500-token budget and only the first configured seed, to check
wiring quickly. --config/--prefix let this same probe be rerun against a
differently-shaped yaml (e.g. a denser late-layer sweep) without forking a
new script -- the procedure is identical, only which layers/seeds/budget to
use changes (see data/layer_locality_config.yaml).

Output:
    results/{prefix}_stats_seed{seed}.json  -- one file per seed, same shape
        as Toy0's routing_fisher_stats.json plus a "sentence_stats" block
        (n_sentences/n_tokens per condition/language, for the sequence-
        length-distribution audit the reproducibility plan calls for).
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


def build_condition_sentences(pools, tokenizer, seed, total_budget):
    """english_a/english_b/balanced-EN draw 3 disjoint chunks from the same
    English pool; korean/chinese each draw 2 (their own + balanced's share).
    See data_utils.token_budget_chunks for the disjointness mechanics."""
    eng_a, eng_b, eng_bal = data_utils.token_budget_chunks(
        tokenizer, pools["eng_Latn"], seed, [total_budget, total_budget, total_budget / 3])
    kor_only, kor_bal = data_utils.token_budget_chunks(
        tokenizer, pools["kor_Hang"], seed, [total_budget, total_budget / 3])
    zho_only, zho_bal = data_utils.token_budget_chunks(
        tokenizer, pools["zho_Hans"], seed, [total_budget, total_budget / 3])

    return {
        "english_a": [("eng_Latn", eng_a)],
        "english_b": [("eng_Latn", eng_b)],
        "korean": [("kor_Hang", kor_only)],
        "chinese": [("zho_Hans", zho_only)],
        "balanced": [("eng_Latn", eng_bal), ("kor_Hang", kor_bal), ("zho_Hans", zho_bal)],
    }


def run_condition(model, tokenizer, moe_layers, num_experts, condition_id, lang_sentence_pairs, eval_batch_size):
    hit_count = {lid: torch.zeros(num_experts, dtype=torch.long) for lid, _ in moe_layers}
    fisher_proxy = {lid: torch.zeros(num_experts, dtype=torch.float64) for lid, _ in moe_layers}

    def callback(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t, router_weight):
        n = y_t.shape[0]
        hit_count[layer_id][expert_idx] += n
        per_token_norm = y_t.detach().to(torch.float64).norm(dim=-1)
        weighted_sq = (router_weight.detach().to(torch.float64) * per_token_norm) ** 2
        fisher_proxy[layer_id][expert_idx] += weighted_sq.sum().item()

    handles = [moe_hooks.install_capture_hook(mod.experts, lid, callback) for lid, mod in moe_layers]

    sentence_stats = []
    try:
        for lang_code, sentences in lang_sentence_pairs:
            nll_sums, n_tokens = data_utils.evaluate_sentences(model, tokenizer, sentences, eval_batch_size)
            sentence_stats.append({
                "lang": lang_code, "n_sentences": len(sentences), "n_tokens": sum(n_tokens),
                "mean_tokens_per_sentence": sum(n_tokens) / len(n_tokens) if n_tokens else 0.0,
            })
            print(f"[01b] {condition_id}/{lang_code}: {len(sentences)} sentences, {sum(n_tokens)} tokens")
    finally:
        for h in handles:
            h.remove()

    return {
        "hit_count": {lid: hit_count[lid].tolist() for lid, _ in moe_layers},
        "fisher_proxy": {lid: fisher_proxy[lid].tolist() for lid, _ in moe_layers},
        "sentence_stats": sentence_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default=None, help="path to a phase0_5_config.yaml-shaped file (default: data/phase0_5_config.yaml)")
    parser.add_argument("--prefix", default="phase0_5", help="output filename prefix under results/ (default: phase0_5)")
    args = parser.parse_args()

    cfg = load_config()
    spec = cfg.load_spec(args.config) if args.config else cfg.load_phase0_5_spec()
    source, split, col_prefix = spec["source"], spec["split"], spec["column_prefix"]
    languages = spec["languages"]
    layer_indices = spec["layer_indices"]
    seeds = [spec["seeds"][0]] if args.smoke else spec["seeds"]
    total_budget = 500 if args.smoke else spec["total_token_budget"]

    print(f"[01b] {'SMOKE' if args.smoke else 'FULL'} run (prefix={args.prefix}): seeds={seeds}, "
          f"layers={layer_indices}, total_token_budget={total_budget}")
    print(f"[01b] loading {cfg.MODEL_NAME} ...")
    model, tokenizer = cfg.load_model_and_tokenizer(layer_indices)

    moe_layers = moe_hooks.get_moe_layers(model, layer_indices)
    if not moe_layers:
        raise RuntimeError(f"no MoE layers found among {layer_indices}")
    num_experts = moe_layers[0][1].experts.num_experts
    print(f"[01b] {len(moe_layers)} MoE layers, {num_experts} experts/layer")

    print("[01b] loading full language pools ...")
    pools = {code: data_utils.load_full_column(source, split, col_prefix, code) for code in languages}
    for code, pool in pools.items():
        print(f"[01b] {code}: pool size {len(pool)} sentences")

    for seed in seeds:
        print(f"[01b] === seed {seed} ===")
        conditions = build_condition_sentences(pools, tokenizer, seed, total_budget)
        seed_results = {}
        for condition_id, lang_sentence_pairs in conditions.items():
            seed_results[condition_id] = run_condition(
                model, tokenizer, moe_layers, num_experts, condition_id, lang_sentence_pairs, cfg.EVAL_BATCH_SIZE,
            )
        out_path = cfg.stats_json(args.prefix, seed)
        if args.smoke:
            out_path = out_path.with_name(out_path.stem + "_smoke" + out_path.suffix)
        out_path.write_text(json.dumps(seed_results, indent=2))
        print(f"[01b] wrote {out_path}")


if __name__ == "__main__":
    main()
