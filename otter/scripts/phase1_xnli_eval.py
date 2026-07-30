"""Phase 1: XNLI downstream accuracy for the D2-MoE compression pipeline,
the "official XNLI, English-fixed 6-shot conditional-likelihood" protocol
worked out for this project's calibration-language-transfer question (see
the 2026-07-29 design discussion this script implements).

Why only 3 XNLI languages, not this project's full 5: official XNLI (the
HF `xnli` dataset, an alias of facebook/xnli) covers 15 languages, and only
English/Chinese/Swahili of this project's 5 calibration languages
(EN/KO/ZH/SW/BN) are among them -- Korean and Bengali have NO official XNLI
split. Rather than substitute a different dataset (KorNLI, IndicXNLI) with
its own translation/annotation pipeline into the same table -- which would
silently mix "XNLI accuracy" numbers with numbers from a differently-built
dataset -- this script evaluates every calibration checkpoint (including
korean_only/bengali_only/mixed_5lang) on ONLY {en, zh, sw}. The Korean- and
Bengali-calibrated checkpoints are still fully in scope: what's missing is
an evaluation LANGUAGE for them, not a checkpoint -- their effect on
en/zh/sw downstream accuracy (transfer/interference from an eval-unsupported
calibration language) is exactly one of this experiment's questions.

Protocol (deliberately NOT lm-eval-harness's vendored xnli_en/xnli_zh/
xnli_sw tasks, see lm_eval/tasks/xnli/*.yaml -- those use a different
per-language natural-sentence template ("premise, right? Yes, hypothesis"
etc.), 0-shot, and score full-hypothesis continuations, none of which match
what this experiment calls for):
  - instruction fixed in English for every eval language (isolates the
    calibration-language variable from an instruction-translation-quality
    variable)
  - premise/hypothesis in the eval language (en/zh/sw)
  - 6-shot demonstrations, ALWAYS drawn from English XNLI dev (2 per label,
    fixed by --demo-seed=42), identical across every checkpoint and every
    eval language -- a fixed English-demonstration -> multilingual-test
    transfer setup, not per-language few-shot
  - one fixed label mapping only (A=entailment, B=neutral, C=contradiction,
    matching the dataset's own ClassLabel order 0/1/2). The design doc's
    3-permutation label-order bias check (§6, "Sanity check" scope) is NOT
    implemented here -- out of scope for this pass; add a second template
    with a rotated CHOICE_LETTERS/LABEL_NAMES mapping if that check is
    later needed.
  - answer scored as the single next-token log-prob of " A"/" B"/" C"
    (confirmed single-token for this tokenizer by inspection -- see
    CHOICE_TOKENS), via ONE forward pass per example (no generation)

Reuses phase1_merge_eval.py's load_model()/merge_condition() and
phase1_structured_prune.apply_structured_pruning() UNCHANGED, exactly like
phase1_belebele_eval.py -- this script's only job is another eval pass
(XNLI accuracy instead of Belebele accuracy or FLORES bpb) over the same
in-memory model object every other Phase 1 stage produces.

Cross-checkpoint metrics (macro avg, worst-language, retention, delta from
baseline, own-language gain) are NOT computed here -- this script only
produces one checkpoint's raw per-language accuracy, same division of labor
as phase1_merge_eval.py/phase1_belebele_eval.py vs phase1_41_headline_gate.py.
See phase1_xnli_analyze.py for those, read across every condition's
eval_xnli.json.

Usage:
    conda run -n d2moe_env python phase1_xnli_eval.py --baseline --smoke
    conda run -n d2moe_env python phase1_xnli_eval.py --condition korean_only --seed 0 --smoke
    conda run -n d2moe_env python phase1_xnli_eval.py --condition mixed_5lang --seed 0 \\
        --scale-condition mixed_5lang --scale-seed 0 --limit 500   # pilot before the full 5,010/language test set
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = SCRIPT_DIR.parent / "D2MoE"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(D2MOE_DIR))
import phase1_calib_data  # noqa: E402
import phase1_structured_prune  # noqa: E402
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402 -- reused unchanged

# HF dataset_path "xnli" -- same underlying data lm_eval/tasks/xnli/xnli_common_yaml
# points at (dataset_path: xnli), the canonical alias of facebook/xnli.
XNLI_DATASET = "xnli"
# Official-XNLI intersection of this project's 5 calibration languages
# (EN/KO/ZH/SW/BN) -- Korean ("ko") and Bengali ("bn") are NOT in the
# official 15-language XNLI and are deliberately excluded as EVAL languages
# (see module docstring). HF config codes, not this project's FLORES codes.
XNLI_LANGS = {"en": "English", "zh": "Chinese", "sw": "Swahili"}

INSTRUCTION = "Determine the relationship between the premise and the hypothesis."
# label int (dataset's own ClassLabel order: 0=entailment,1=neutral,2=contradiction)
# -> option letter. Fixed Template 1 (A=entailment,B=neutral,C=contradiction) only
# -- see module docstring for why the 3-permutation bias-control variant isn't
# implemented here.
LABEL_NAMES = ["Entailment", "Neutral", "Contradiction"]
CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_TOKENS = [" A", " B", " C"]  # each confirmed a single token for this tokenizer (see eval_xnli())

DEMO_SEED_DEFAULT = 42
DEMOS_PER_LABEL_DEFAULT = 2
DEMO_MAX_WORDS = 40  # premise+hypothesis word count cap, keeps demonstrations short (§8 of the design)


def build_demonstrations(seed=DEMO_SEED_DEFAULT, per_label=DEMOS_PER_LABEL_DEFAULT, max_words=DEMO_MAX_WORDS):
    """Fixed English-dev demonstrations, `per_label` examples per label,
    identical across every checkpoint/eval-language call (never re-drawn
    per language) -- §8's requirement. Dev (validation) split only, never
    test (§7)."""
    from datasets import load_dataset

    ds = load_dataset(XNLI_DATASET, "en", split="validation")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    buckets = {0: [], 1: [], 2: []}
    for i in order:
        ex = ds[i]
        label = ex["label"]
        if len(buckets[label]) >= per_label:
            continue
        if len(ex["premise"].split()) + len(ex["hypothesis"].split()) > max_words:
            continue
        buckets[label].append(ex)
        if all(len(v) >= per_label for v in buckets.values()):
            break
    missing = [l for l, v in buckets.items() if len(v) < per_label]
    if missing:
        raise RuntimeError(f"could not find {per_label} demonstrations under {max_words} words for label(s) {missing}")

    demos = buckets[0] + buckets[1] + buckets[2]
    rng.shuffle(demos)  # interleave label order in the printed demo block
    return demos


def _format_block(premise, hypothesis, answer_letter=None):
    block = (f"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
             f"A. {LABEL_NAMES[0]}\nB. {LABEL_NAMES[1]}\nC. {LABEL_NAMES[2]}\n\nAnswer:")
    if answer_letter is not None:
        block += f" {answer_letter}\n\n"
    return block


def build_prompt(demos, premise, hypothesis):
    parts = [INSTRUCTION, ""]
    for ex in demos:
        parts.append(_format_block(ex["premise"], ex["hypothesis"], CHOICE_LETTERS[ex["label"]]))
    parts.append(_format_block(premise, hypothesis))
    return "\n".join(parts)


@torch.no_grad()
def score_batch(model, tokenizer, prompts, choice_token_ids, device=None):
    """One forward pass per batch: right-pads (matches data_utils.
    batch_sentence_nll's convention -- keeps every real token's position id
    the arange default `0..real_len-1`, so right-padding never perturbs the
    positions that matter, unlike left-padding which needs correct
    position_ids threaded through explicitly), then reads each row's logits
    at its own last REAL token (index attention_mask.sum()-1, not a
    uniform -1) -- that's the position predicting the token right after
    "Answer:". Returns (batch, 3) log-probs, one column per CHOICE_TOKENS
    entry, in CPU float32."""
    if device is None:
        device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # Re-derive the device from the actual output, not the pre-forward
    # `device` var: load_model() uses device_map="auto" across 2-4 GPUs, and
    # accelerate is free to place the final lm_head (hence out.logits) on a
    # different shard than the first embedding parameter. Fancy-indexing
    # out.logits with index tensors built on the wrong device raises
    # "Expected all tensors to be on the same device" -- unlike data_utils.
    # batch_sentence_nll's plain slice (out.logits[:, :-1, :]), which has no
    # index tensors and so never hits this.
    out_device = out.logits.device
    last_idx = (attention_mask.sum(dim=1) - 1).to(out_device)
    batch_idx = torch.arange(input_ids.size(0), device=out_device)
    last_logits = out.logits[batch_idx, last_idx, :]
    logprobs = F.log_softmax(last_logits.float(), dim=-1)
    return logprobs[:, choice_token_ids].cpu()


def eval_xnli(model, tokenizer, batch_size=4, limit=None, demo_seed=DEMO_SEED_DEFAULT):
    """Returns {lang_code: {"acc": float, "n_samples": int}} for every
    XNLI_LANGS entry, evaluated on the `test` split (§7 -- dev is for
    demonstration/template selection only, never for reported accuracy)."""
    from datasets import load_dataset

    demos = build_demonstrations(seed=demo_seed)
    choice_token_ids = []
    for tok_str in CHOICE_TOKENS:
        ids = tokenizer(tok_str, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"choice token {tok_str!r} is {len(ids)} tokens for this tokenizer, "
                                f"not 1 -- the single-forward-pass scoring in score_batch() assumes single-token "
                                f"choices; CHOICE_TOKENS needs revisiting for this model.")
        choice_token_ids.append(ids[0])

    results = {}
    for lang_code, lang_name in XNLI_LANGS.items():
        ds = load_dataset(XNLI_DATASET, lang_code, split="test")
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        preds, gold = [], []
        for i in range(0, len(ds), batch_size):
            chunk = ds[i:i + batch_size]
            prompts = [build_prompt(demos, p, h) for p, h in zip(chunk["premise"], chunk["hypothesis"])]
            logprobs = score_batch(model, tokenizer, prompts, choice_token_ids)
            preds.extend(logprobs.argmax(dim=-1).tolist())
            gold.extend(chunk["label"])

        correct = sum(p == g for p, g in zip(preds, gold))
        acc = correct / len(gold) if gold else float("nan")
        results[lang_code] = {"name": lang_name, "acc": acc, "n_samples": len(gold)}
        print(f"[xnli-eval] {lang_name} ({lang_code}): acc={acc:.4f} (n={len(gold)})")

    results["_demonstrations"] = [
        {"premise": ex["premise"], "hypothesis": ex["hypothesis"], "label": ex["label"]} for ex in demos
    ]
    results["_demo_seed"] = demo_seed
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--baseline", action="store_true", help="evaluate the uncompressed model instead")
    parser.add_argument("--seed", type=int, default=0,
                         help="must match the seed used for phase1_run_freq_and_scale.py / phase1_fisher.py")
    parser.add_argument("--scale-condition", default=None,
                         help="load svd_scale (whitened/truncation-aware SVD) from this condition; "
                              "default None = plain SVD. Pass --scale-condition <same as --condition> for the "
                              "'own-language whitened SVD' full-pipeline configuration (matches phase1_belebele_eval.py).")
    parser.add_argument("--scale-seed", type=int, default=None, help="seed for --scale-condition (defaults to --seed)")
    parser.add_argument("--delta-ratio", type=float, default=None, help="override DELTA_RATIO=0.8 (see phase1_merge_eval.py)")
    parser.add_argument("--cpu-offload-gib", type=int, default=0)
    parser.add_argument("--pp-ratio", type=float, default=None,
                         help="structured-pruning ratio (third pipeline stage); default None = off")
    parser.add_argument("--pp-condition", default=None, help="calibration condition for pruning importance calib (default: --condition)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--demo-seed", type=int, default=DEMO_SEED_DEFAULT,
                         help="fixed English-dev demonstration selection seed (must match across every checkpoint/language for a valid comparison)")
    parser.add_argument("--limit", type=int, default=None,
                         help="subsample this many test examples per language; recommended 500 for a pilot before "
                              "the full ~5,010/language test set")
    parser.add_argument("--smoke", action="store_true", help="limit=20, for wiring verification only")
    args = parser.parse_args()
    if not args.baseline and not args.condition:
        parser.error("either --condition or --baseline is required")
    if args.scale_condition is not None and args.scale_seed is None:
        args.scale_seed = args.seed
    if args.pp_ratio is not None and args.pp_condition is None:
        args.pp_condition = args.condition

    limit = 20 if args.smoke else args.limit

    print(f"[xnli-eval] {'BASELINE (uncompressed)' if args.baseline else f'condition={args.condition}'} "
          f"smoke={args.smoke} limit={limit}")
    model, tokenizer = load_model(cpu_gib=args.cpu_offload_gib)

    if not args.baseline:
        scale_note = f" scale=({args.scale_condition},seed{args.scale_seed})" if args.scale_condition else " scale=none(plain SVD)"
        pp_note = f" pp_ratio={args.pp_ratio}(calib={args.pp_condition})" if args.pp_ratio is not None else " pp_ratio=off"
        print(f"[xnli-eval] merging condition={args.condition} seed={args.seed}{scale_note}{pp_note}")
        model = merge_condition(model, args.condition, args.seed, args.scale_condition, args.scale_seed,
                                 delta_ratio=args.delta_ratio)
        if args.pp_ratio is not None:
            phase1_structured_prune.apply_structured_pruning(
                model, args.pp_condition, args.seed, args.pp_ratio, tokenizer)

    results = eval_xnli(model, tokenizer, batch_size=args.batch_size, limit=limit, demo_seed=args.demo_seed)

    nan_langs = [lang for lang in XNLI_LANGS if results[lang]["acc"] != results[lang]["acc"]]
    if nan_langs:
        raise RuntimeError(
            f"xnli eval produced NaN acc for {nan_langs} -- likely a silent model failure (e.g. disk-offload "
            "under host memory pressure, see phase1_merge_eval.py's NaN guard for the same failure mode on FLORES "
            "bpb). Refusing to write eval_xnli.json; let the caller retry.")

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
    out_path = out_dir / ("eval_xnli_smoke.json" if args.smoke else "eval_xnli.json")
    out_path.write_text(json.dumps({"num_fewshot": len(CHOICE_LETTERS) * DEMOS_PER_LABEL_DEFAULT,
                                     "demo_seed": args.demo_seed, "limit": limit, "results": results}, indent=2))
    print(f"[xnli-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
