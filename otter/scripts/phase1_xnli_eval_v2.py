"""Phase 1: XNLI downstream accuracy, PROTOCOL v2 -- replaces the "Answer:
A/B/C" letter-token scoring in phase1_xnli_eval.py (v1) after that protocol's
baseline diagnosed a label collapse, not a compression-pipeline bug (see the
2026-07-30 diagnosis this script implements): with v1's 6-shot lettered-MC
template, baseline predictions piled onto B/Neutral (EN 17/20, SW 19/20) and
the A/C candidate logprobs were IDENTICAL to several decimal places
regardless of premise/hypothesis content -- this base model (never
instruction-tuned) cannot read a structured "A. Entailment / B. Neutral /
C. Contradiction" multiple-choice layout as a live per-item decision; it was
reading the letter distribution of the demonstrations, not the item. v1's
prompt formatting/demo balance were separately re-checked and are NOT the
cause. v1's eval_xnli_smoke.json outputs are left in place but renamed
*_deprecated.json (not deleted) -- do not read them as protocol-v2 numbers.

Template (deliberately the lm-eval-harness-style base-model XNLI phrasing,
not v1's lettered MC -- lm_eval/tasks/xnli/*.yaml's own per-language natural-
sentence template was considered and rejected, see below):
    {premise}
    Question: {hypothesis} True, False, or Neither?
    Answer:
scored against three continuations: " True" (entailment), " Neither"
(neutral), " False" (contradiction). The question/candidate words are fixed
in ENGLISH for every eval language (only premise/hypothesis switch) --
standard XNLI practice (also what v1 did for its instruction line), so that
switching eval language never also switches an instruction-translation-
quality variable, keeping languages comparable. This is why lm_eval's own
vendored xnli_en/xnli_zh/xnli_sw tasks are NOT reused here: those templates
translate the *phrasing* itself per language ("premise, right? Yes/No/Also,
hypothesis"), which would confound eval-language with instruction-language
exactly like v1's design deliberately avoided.

Scoring: each candidate continuation is confirmed (see CHOICE_WORDS /
_choice_token_ids() below) to be exactly ONE token for this tokenizer, same
as v1's " A"/" B"/" C" were -- so the "sum of per-token logprobs" and the
"length-normalized (sum / token count)" versions are numerically identical
here (n_tokens=1 for all three candidates); both are still computed and
written to the output JSON as separate fields (acc / acc_unnormalized, and
per-candidate logprob_sum / logprob_normalized in _sample_items) per the
2026-07-30 spec, so the distinction is on record even though it's a no-op
for this tokenizer. Judged accuracy uses the normalized field. If a future
model/candidate-set ever makes a candidate multi-token, CHOICE_WORDS's
single-token assumption (asserted in _choice_token_ids()) will need a real
multi-token summed-logprob loop -- not implemented here since it can't
happen for the current tokenizer.

6-shot demonstrations: reuses phase1_xnli_eval.build_demonstrations()
UNCHANGED (same seed=42 default, same 2-per-label English-dev selection) --
only the rendering (build_prompt() below) is new; the demo *content*
selection logic is identical to v1's, per the "재사용 가능하면 형식만 변환"
instruction.

Reuses phase1_merge_eval.py's load_model()/merge_condition() and
phase1_structured_prune.apply_structured_pruning() UNCHANGED, and
phase1_xnli_eval.py's score_batch() UNCHANGED (it's already generic over any
3-way single-token choice-id list) -- same division of labor as v1 and every
other Phase 1 eval script.

Usage:
    conda run -n d2moe_env python phase1_xnli_eval_v2.py --baseline --smoke --limit 50 --langs en   # STEP 1 gate
    conda run -n d2moe_env python phase1_xnli_eval_v2.py --baseline --limit 200                     # STEP 2
    conda run -n d2moe_env python phase1_xnli_eval_v2.py --condition korean_only --seed 0 \\
        --scale-condition korean_only --scale-seed 0 --limit 200                                    # STEP 3
"""
import argparse
import json
import sys
from pathlib import Path

import torch

# Environment workaround (2026-07-30, found while debugging why STEP 1 couldn't even
# load the model here): transformers.dynamic_module_utils.check_imports() runs on
# EVERY trust_remote_code=True from_pretrained() call, unconditionally, against the
# pristine hub snapshot file (not the locally-cached/patched copy under
# ~/.cache/huggingface/modules/) -- so it always sees this model's modeling_deepseek.py
# guarding `from flash_attn import ...` with `if is_flash_attn_2_available():`, NOT a
# try/except (get_imports() only strips try/except blocks, not if-blocks), and always
# raises ImportError for flash_attn, which was never actually installed in d2moe_env
# (a prior install attempt failed on missing CUDA_HOME, see
# otter/logs/d2moe_flashattn_install.log) and isn't needed: is_flash_attn_2_available()
# correctly returns False, so that guarded import branch never executes at runtime --
# the check_imports failure is a false positive for an optional, properly-gated import.
# Confirmed by direct reproduction (both online and HF_HUB_OFFLINE=1) that
# AutoModelForCausalLM.from_pretrained() cannot load this model at all without this,
# and that dropping "flash_attn" from the required-imports list is sufficient. Scoped
# to this process only (patches the module object in memory, never touches the
# installed transformers package on disk) so it can't affect any other concurrently
# running session's already-imported code.
import transformers.dynamic_module_utils as _dmu  # noqa: E402
_orig_get_imports = _dmu.get_imports
_dmu.get_imports = lambda filename: [imp for imp in _orig_get_imports(filename) if imp != "flash_attn"]

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = SCRIPT_DIR.parent / "D2MoE"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(D2MOE_DIR))
import phase1_calib_data  # noqa: E402
import phase1_structured_prune  # noqa: E402
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402 -- reused unchanged
from phase1_xnli_eval import (  # noqa: E402 -- reused unchanged, see module docstring
    XNLI_DATASET, XNLI_LANGS, build_demonstrations, score_batch,
    DEMO_SEED_DEFAULT, DEMOS_PER_LABEL_DEFAULT,
)

# label int (dataset's own ClassLabel order: 0=entailment,1=neutral,2=contradiction)
# -> continuation word, WITH the leading space the tokenizer treats as part of
# the token (see _choice_token_ids() -- confirmed single-token: True=8173,
# Neither=46136, False=13813 for this tokenizer, by direct inspection).
CHOICE_WORDS = [" True", " Neither", " False"]
LABEL_KEYS = ["entailment", "neutral", "contradiction"]

QUESTION_SUFFIX = "True, False, or Neither?"  # fixed English wording for every eval language, see module docstring


def _choice_token_ids(tokenizer):
    ids = []
    for word in CHOICE_WORDS:
        toks = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            raise RuntimeError(
                f"choice word {word!r} is {len(toks)} tokens for this tokenizer, not 1 -- the single-forward-pass "
                f"scoring here (and the sum==normalized shortcut documented in the module docstring) assumes "
                f"single-token candidates; CHOICE_WORDS needs a real multi-token summed-logprob loop for this model.")
        ids.append(toks[0])
    return ids


def format_example(premise, hypothesis, answer_word=None):
    """One demo/query block. answer_word (e.g. CHOICE_WORDS[label], leading
    space included) appended in-line after 'Answer:' for demos; omitted
    (block ends right after 'Answer:') for the item being scored -- the
    continuation logprob is read at exactly that position, same convention
    v1 used for its lettered choices."""
    block = f"{premise}\nQuestion: {hypothesis} {QUESTION_SUFFIX}\nAnswer:"
    if answer_word is not None:
        block += f"{answer_word}\n\n"
    return block


def build_prompt(demos, premise, hypothesis):
    parts = [format_example(ex["premise"], ex["hypothesis"], CHOICE_WORDS[ex["label"]]) for ex in demos]
    parts.append(format_example(premise, hypothesis))
    return "\n".join(parts)


def eval_xnli(model, tokenizer, batch_size=4, limit=None, demo_seed=DEMO_SEED_DEFAULT,
              langs=None, n_sample_items=10):
    """Returns {lang_code: {...}} for every requested XNLI language (default:
    every XNLI_LANGS entry), evaluated on the test split. Each language's
    dict carries acc (normalized, judged metric), acc_unnormalized (reference
    only -- identical to acc here, see module docstring), pred_distribution
    (label -> count, for the STEP-1/STEP-2 label-collapse gate check), and
    n_sample_items per-item records (premise/hypothesis/gold/pred plus every
    candidate's logprob_sum/logprob_normalized) so a human can confirm
    logprobs actually vary with item content instead of collapsing like v1's
    did."""
    from datasets import load_dataset

    if langs is None:
        langs = list(XNLI_LANGS.keys())

    demos = build_demonstrations(seed=demo_seed)
    choice_token_ids = _choice_token_ids(tokenizer)

    results = {}
    for lang_code in langs:
        lang_name = XNLI_LANGS[lang_code]
        ds = load_dataset(XNLI_DATASET, lang_code, split="test")
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        pred_sum, pred_norm, gold = [], [], []
        sample_items = []
        for i in range(0, len(ds), batch_size):
            chunk = ds[i:i + batch_size]
            prompts = [build_prompt(demos, p, h) for p, h in zip(chunk["premise"], chunk["hypothesis"])]
            logprobs = score_batch(model, tokenizer, prompts, choice_token_ids)  # (batch, 3), n_tokens=1 per candidate
            logprobs_norm = logprobs  # / 1 token each -- see module docstring; kept as a separate name, not folded
            # into logprobs, so a future multi-token CHOICE_WORDS change only has to edit this one line.
            batch_pred_sum = logprobs.argmax(dim=-1).tolist()
            batch_pred_norm = logprobs_norm.argmax(dim=-1).tolist()
            pred_sum.extend(batch_pred_sum)
            pred_norm.extend(batch_pred_norm)
            gold.extend(chunk["label"])

            for j in range(len(batch_pred_norm)):
                if len(sample_items) >= n_sample_items:
                    continue
                sample_items.append({
                    "premise": chunk["premise"][j], "hypothesis": chunk["hypothesis"][j], "gold": chunk["label"][j],
                    "pred": batch_pred_norm[j],
                    "logprobs": {LABEL_KEYS[k]: {"sum": logprobs[j, k].item(), "normalized": logprobs_norm[j, k].item()}
                                 for k in range(3)},
                })

        n = len(gold)
        acc = sum(p == g for p, g in zip(pred_norm, gold)) / n if n else float("nan")
        acc_unnorm = sum(p == g for p, g in zip(pred_sum, gold)) / n if n else float("nan")
        dist = {key: 0 for key in LABEL_KEYS}
        for p in pred_norm:
            dist[LABEL_KEYS[p]] += 1
        results[lang_code] = {
            "name": lang_name, "acc": acc, "acc_unnormalized": acc_unnorm, "n_samples": n,
            "pred_distribution": dist, "_sample_items": sample_items,
        }
        top_frac = max(dist.values()) / n if n else float("nan")
        print(f"[xnli-eval-v2] {lang_name} ({lang_code}): acc={acc:.4f} acc_unnorm={acc_unnorm:.4f} (n={n}) "
              f"pred_dist={dist} top_label_frac={top_frac:.2f}")

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
                              "'own-language whitened SVD' full-pipeline configuration.")
    parser.add_argument("--scale-seed", type=int, default=None, help="seed for --scale-condition (defaults to --seed)")
    parser.add_argument("--delta-ratio", type=float, default=None, help="override DELTA_RATIO=0.8 (see phase1_merge_eval.py)")
    parser.add_argument("--cpu-offload-gib", type=int, default=0)
    parser.add_argument("--pp-ratio", type=float, default=None,
                         help="structured-pruning ratio (third pipeline stage); default None = off")
    parser.add_argument("--pp-condition", default=None, help="calibration condition for pruning importance calib (default: --condition)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--demo-seed", type=int, default=DEMO_SEED_DEFAULT,
                         help="fixed English-dev demonstration selection seed (must match across every checkpoint/language for a valid comparison)")
    parser.add_argument("--langs", default=None,
                         help="comma-separated subset of en,zh,sw (default: all three) -- e.g. --langs en for the STEP-1 gate")
    parser.add_argument("--limit", type=int, default=None, help="subsample this many test examples per language")
    parser.add_argument("--smoke", action="store_true",
                         help="write to the _smoke output filename; also sets limit=20 unless --limit is also given")
    args = parser.parse_args()
    if not args.baseline and not args.condition:
        parser.error("either --condition or --baseline is required")
    if args.scale_condition is not None and args.scale_seed is None:
        args.scale_seed = args.seed
    if args.pp_ratio is not None and args.pp_condition is None:
        args.pp_condition = args.condition

    limit = args.limit if args.limit is not None else (20 if args.smoke else None)
    langs = args.langs.split(",") if args.langs else None
    for lang in (langs or []):
        if lang not in XNLI_LANGS:
            parser.error(f"--langs: {lang!r} not in {list(XNLI_LANGS)}")

    print(f"[xnli-eval-v2] {'BASELINE (uncompressed)' if args.baseline else f'condition={args.condition}'} "
          f"smoke={args.smoke} limit={limit} langs={langs or list(XNLI_LANGS)}")
    model, tokenizer = load_model(cpu_gib=args.cpu_offload_gib)

    if not args.baseline:
        scale_note = f" scale=({args.scale_condition},seed{args.scale_seed})" if args.scale_condition else " scale=none(plain SVD)"
        pp_note = f" pp_ratio={args.pp_ratio}(calib={args.pp_condition})" if args.pp_ratio is not None else " pp_ratio=off"
        print(f"[xnli-eval-v2] merging condition={args.condition} seed={args.seed}{scale_note}{pp_note}")
        model = merge_condition(model, args.condition, args.seed, args.scale_condition, args.scale_seed,
                                 delta_ratio=args.delta_ratio)
        if args.pp_ratio is not None:
            phase1_structured_prune.apply_structured_pruning(
                model, args.pp_condition, args.seed, args.pp_ratio, tokenizer)

    results = eval_xnli(model, tokenizer, batch_size=args.batch_size, limit=limit, demo_seed=args.demo_seed, langs=langs)

    eval_langs = langs or list(XNLI_LANGS)
    nan_langs = [lang for lang in eval_langs if results[lang]["acc"] != results[lang]["acc"]]
    if nan_langs:
        raise RuntimeError(
            f"xnli-v2 eval produced NaN acc for {nan_langs} -- likely a silent model failure (e.g. disk-offload "
            "under host memory pressure, see phase1_merge_eval.py's NaN guard for the same failure mode). Refusing "
            "to write eval_xnli_v2 json; let the caller retry.")

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
    out_path = out_dir / ("eval_xnli_v2_smoke.json" if args.smoke else "eval_xnli_v2.json")
    out_path.write_text(json.dumps({
        "protocol": "v2_true_false_neither",
        "num_fewshot": len(LABEL_KEYS) * DEMOS_PER_LABEL_DEFAULT,
        "demo_seed": args.demo_seed, "limit": limit, "langs": eval_langs, "results": results,
    }, indent=2))
    print(f"[xnli-eval-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
