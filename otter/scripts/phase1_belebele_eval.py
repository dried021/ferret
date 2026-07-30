"""Phase 1: Belebele zero-shot accuracy for the FULL D2-MoE pipeline (Fisher-
weighted merge + truncation-aware/whitened delta-SVD + pp_ratio structured
pruning), the "full pipeline + downstream task" reconfirmation flagged as
top priority in 00_docs/04_전체요약.md ("논문 주장의 현재 형태" -- 미해결 항목:
"지금까지의 own-language 효과와 §4.3 whitening 지배 결과는 FLORES bpb 지표로만
확인됐고, pruning까지 포함한 전체 파이프라인 + 경량 downstream task(Belebele/XNLI)
에서 재확인된 적은 아직 없다") and in 01_plans/claude_plan.md ("핵심 주장은
Fisher-merge 단독이 아니라 full pipeline + downstream task에서 재확인되어야 함").

Every prior Phase 1 result (own-language protective effect for KO/Swahili,
whitening-dominates-Fisher finding) was measured on FLORES-200 bits-per-byte
-- a language-modeling-likelihood metric, not a task the model is actually
"used for". This script instead measures Belebele (FLORES-passage-grounded,
122-language, 4-way multiple-choice reading comprehension) zero-shot
accuracy, so the same own-language claim can be checked against a real
downstream task, on the SAME compressed model phase1_merge_eval.py builds --
not a different pipeline. Belebele's per-language codes are the same
FLORES-200 codes this project already standardized on (eng_Latn, kor_Hang,
zho_Hans, swh_Latn, ben_Beng), so no new calibration/eval language mapping
is needed.

Deliberately reuses phase1_merge_eval.py's load_model()/merge_condition()
and phase1_structured_prune.apply_structured_pruning() UNCHANGED (imported,
not reimplemented) -- this script's only job is to swap eval_flores_ppl()
for a Belebele accuracy pass through the vendored lm-evaluation-harness, on
the exact same in-memory model object every other Phase 1 stage produces.
See phase1_merge_eval.py's module docstring for why merge_experts()/
apply_structured_pruning() are used directly instead of D2-deepseek.py's
full --control_name pipeline.

Belebele harness wiring: D2MoE/lm-evaluation-harness is EleutherAI's
lm-eval-harness (v0.4.3) with a full belebele task set already vendored in
(lm_eval/tasks/belebele/, 122 languages) and a custom "svd"/"SVD" registered
model class (lm_eval/models/SVDmodel.py) that accepts an already-initialized
transformers.PreTrainedModel via `pretrained=` (falls through
`isinstance(pretrained, str)` to `self._model = pretrained` directly, no
re-loading) -- exactly the object load_model()+merge_condition() produce.
This is the same call pattern D2MoE/utils.py::run_lm_eval() already uses
(the only other call site for this harness in the whole repo, invoked from
D2-deepseek.py's own downstream-task eval).

Output type is multiple_choice (loglikelihood-based, not free generation):
each question's 4 answer letters are scored as continuations of the same
passage+question context, so cost is dominated by one forward pass per
question, not per (question, choice) pair (lm-eval's context-grouping cache
reuses logits across the 4 single-token continuations).

Usage:
    conda run -n d2moe_env python phase1_belebele_eval.py --baseline --smoke
    conda run -n d2moe_env python phase1_belebele_eval.py --condition korean_only --seed 0 --smoke
    # full pipeline: own-language Fisher + own-language whitened SVD + pruning
    conda run -n d2moe_env python phase1_belebele_eval.py --condition korean_only --seed 0 \\
        --scale-condition korean_only --scale-seed 0 --pp-ratio 0.2 --limit 200
    # first full-scale run: recommend --limit 100-200 (pilot) before the full ~900/language set
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = SCRIPT_DIR.parent / "D2MoE"
LM_EVAL_DIR = D2MOE_DIR / "lm-evaluation-harness"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(D2MOE_DIR))
sys.path.insert(0, str(LM_EVAL_DIR))  # fallback if not `pip install -e .`-ed into the active env
import phase1_calib_data  # noqa: E402
import phase1_structured_prune  # noqa: E402
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402 -- reused unchanged

# Same 5 languages phase1_merge_eval.py's EVAL_LANGS uses for FLORES bpb, so
# every Belebele result here is directly comparable (same calibration
# conditions, same eval languages) to the existing bpb table.
BELEBELE_LANGS = {
    "eng_Latn": "belebele_eng_Latn",
    "kor_Hang": "belebele_kor_Hang",
    "zho_Hans": "belebele_zho_Hans",
    "swh_Latn": "belebele_swh_Latn",
    "ben_Beng": "belebele_ben_Beng",
}


def _extract_metric(task_result, metric):
    """lm-eval 0.4.x keys metrics as '<metric>,<filter>' (e.g. 'acc,none');
    fall back to the bare name in case a different harness version is ever
    swapped in."""
    for key in (f"{metric},none", metric):
        if key in task_result:
            return task_result[key]
    return None


def eval_belebele(model, tokenizer, batch_size=4, limit=None, num_fewshot=0, langs=None):
    """Runs lm-eval-harness's Belebele tasks against an already-built model
    (baseline or compressed -- caller decides). Returns
    {lang_code: {"acc": float, "acc_norm": float, "n_samples": int}}.

    batch_size defaults lower than eval_flores_ppl's 8 -- Belebele passages
    (FLORES paragraphs, not single sentences) are considerably longer than
    the FLORES sentence-level eval this project's bpb metric uses, so a
    smaller batch keeps peak activation memory comparable on a single GPU.

    langs: optional subset of BELEBELE_LANGS keys to restrict evaluation to
    (default None = all 5). Used to skip at-chance/irrelevant languages for
    a given grid (e.g. §4.2's eng/kor/zho-only expansion), not to change the
    task definitions themselves.
    """
    from lm_eval import evaluator, tasks

    lang_subset = {l: BELEBELE_LANGS[l] for l in langs} if langs else BELEBELE_LANGS
    task_names = list(lang_subset.values())
    raw = evaluator.simple_evaluate(
        model=model,
        tokenizer=tokenizer,
        tasks=task_names,
        batch_size=batch_size,
        device=next(model.parameters()).device,
        write_out=False,
        log_samples=False,
        verbosity="WARNING",
        num_fewshot=num_fewshot,
        task_manager=tasks.TaskManager(),
        limit=limit,
    )

    out = {}
    for lang_code, task_name in lang_subset.items():
        r = raw["results"].get(task_name)
        if r is None:
            print(f"[belebele-eval] WARNING: no results for {task_name}, skipping")
            continue
        n_samples = raw.get("n-samples", {}).get(task_name, {}).get("effective", None)
        acc, acc_norm = _extract_metric(r, "acc"), _extract_metric(r, "acc_norm")
        out[lang_code] = {"task": task_name, "acc": acc, "acc_norm": acc_norm, "n_samples": n_samples}
        print(f"[belebele-eval] {lang_code}: acc={acc:.4f} acc_norm={acc_norm:.4f} "
              f"(n={n_samples}, num_fewshot={num_fewshot})")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--baseline", action="store_true", help="evaluate the uncompressed model instead")
    parser.add_argument("--seed", type=int, default=0,
                         help="must match the seed used for phase1_run_freq_and_scale.py / phase1_fisher.py")
    parser.add_argument("--scale-condition", default=None,
                         help="load svd_scale (whitened/truncation-aware SVD) from this condition; "
                              "default None = plain SVD (matches phase1_merge_eval.py's own default scope cut). "
                              "Pass --scale-condition <same as --condition> for the 'own-language whitened SVD' "
                              "full-pipeline configuration.")
    parser.add_argument("--scale-seed", type=int, default=None, help="seed for --scale-condition (defaults to --seed)")
    parser.add_argument("--delta-ratio", type=float, default=None, help="override DELTA_RATIO=0.8 (see phase1_merge_eval.py)")
    parser.add_argument("--cpu-offload-gib", type=int, default=0)
    parser.add_argument("--pp-ratio", type=float, default=None,
                         help="structured-pruning ratio (third pipeline stage); default None = off")
    parser.add_argument("--pp-condition", default=None, help="calibration condition for pruning importance calib (default: --condition)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                         help="subsample this many examples per language (lm-eval's own --limit); "
                              "recommended 100-200 for a first pilot before the full ~900/language set")
    parser.add_argument("--smoke", action="store_true", help="limit=20, for wiring verification only")
    parser.add_argument("--langs", nargs="+", default=None, choices=list(BELEBELE_LANGS.keys()),
                         help="subset of languages to evaluate (default: all 5). Use to skip languages "
                              "already known to be at-chance in the baseline (phase1_belebele_floor_check.py) "
                              "and save time -- does not change task definitions, just which tasks run.")
    args = parser.parse_args()
    if not args.baseline and not args.condition:
        parser.error("either --condition or --baseline is required")
    if args.scale_condition is not None and args.scale_seed is None:
        args.scale_seed = args.seed
    if args.pp_ratio is not None and args.pp_condition is None:
        args.pp_condition = args.condition

    limit = 20 if args.smoke else args.limit

    print(f"[belebele-eval] {'BASELINE (uncompressed)' if args.baseline else f'condition={args.condition}'} "
          f"smoke={args.smoke} limit={limit}")
    model, tokenizer = load_model(cpu_gib=args.cpu_offload_gib)

    if not args.baseline:
        scale_note = f" scale=({args.scale_condition},seed{args.scale_seed})" if args.scale_condition else " scale=none(plain SVD)"
        pp_note = f" pp_ratio={args.pp_ratio}(calib={args.pp_condition})" if args.pp_ratio is not None else " pp_ratio=off"
        print(f"[belebele-eval] merging condition={args.condition} seed={args.seed}{scale_note}{pp_note}")
        model = merge_condition(model, args.condition, args.seed, args.scale_condition, args.scale_seed,
                                 delta_ratio=args.delta_ratio)
        if args.pp_ratio is not None:
            phase1_structured_prune.apply_structured_pruning(
                model, args.pp_condition, args.seed, args.pp_ratio, tokenizer)

    results = eval_belebele(model, tokenizer, batch_size=args.batch_size, limit=limit, num_fewshot=args.num_fewshot,
                             langs=args.langs)

    nan_langs = [lang for lang, r in results.items() if r["acc"] is None or r["acc"] != r["acc"]]
    if nan_langs:
        raise RuntimeError(
            f"belebele eval produced no/NaN acc for {nan_langs} -- likely a silent model failure (e.g. disk-offload "
            "under host memory pressure, see phase1_merge_eval.py's NaN guard for the same failure mode on FLORES "
            "bpb). Refusing to write eval_belebele.json; let the caller retry.")

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
    if args.num_fewshot:
        # 2026-07-29: without this, a --num-fewshot run silently overwrote the
        # 0-shot baseline's eval_belebele.json (same path, no fewshot in the
        # name) -- baseline 0-shot floor-check numbers were lost once before
        # being manually backed up. Only 0-shot (the prior default/implicit
        # behavior) keeps the old bare filename, so existing 0-shot results
        # anywhere in the tree stay valid without needing to be re-run.
        out_dir = out_dir / f"fewshot_{args.num_fewshot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("eval_belebele_smoke.json" if args.smoke else "eval_belebele.json")
    out_path.write_text(json.dumps({"num_fewshot": args.num_fewshot, "limit": limit,
                                     "langs_evaluated": args.langs or list(BELEBELE_LANGS.keys()),
                                     "results": results}, indent=2))
    print(f"[belebele-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
