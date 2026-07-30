"""§4.1/§4.3 full-pipeline + downstream-task reconfirmation -- retention report
for Belebele zero-shot accuracy, mirroring FLORES bpb's retention formula
(00_docs/01_연구설계.md §10: retention(language) = compressed_score(language) /
original_score(language)) and phase1_pruning_gate.py's on/off-ablation
reporting style, applied to a real task metric instead of a likelihood metric.

Compares, per language and per seed:
  - baseline (uncompressed) acc
  - condition, pp_ratio OFF (Fisher-merge + [whitened] SVD only, no pruning)
  - condition, pp_ratio ON  (+ structured pruning) -- the full 3-stage pipeline
    this project's own docs (00_docs/04_전체요약.md, 01_plans/claude_plan.md)
    flag as the necessary condition for the paper's core claim

...and reports retention_off/retention_on per language, plus macro-average
and worst-language retention (the project's standard summary cut, §10).

Any language phase1_belebele_floor_check.py flagged as at-chance in the
baseline is reported but EXCLUDED from macro-average/worst-language --
retention isn't a meaningful number there regardless of compression (see
that script's docstring for why).

Usage:
    conda run -n d2moe_env python phase1_belebele_gate.py --condition korean_only --pp-ratio 0.2
    conda run -n d2moe_env python phase1_belebele_gate.py --condition swahili_only --scale-condition swahili_only
"""
import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
EVAL_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="korean_only")
    parser.add_argument("--pp-ratio", type=float, default=0.2)
    parser.add_argument("--scale-condition", default=None,
                         help="pass the same value used for the grid run (e.g. --scale-condition korean_only "
                              "for the own-language whitened-SVD full pipeline) to locate the right output dir")
    parser.add_argument("--metric", default="acc", choices=["acc", "acc_norm"])
    parser.add_argument("--num-fewshot", type=int, default=0,
                         help="must match the --num-fewshot used for both the baseline and grid runs being "
                              "compared here (0-shot Belebele sat too close to chance to be usable, see "
                              "phase1_belebele_floor_check.py and run_phase1_belebele_grid.py's docstrings)")
    args = parser.parse_args()
    cond, pp_ratio, metric = args.condition, args.pp_ratio, args.metric
    fewshot_suffix = f"fewshot_{args.num_fewshot}" if args.num_fewshot else None

    baseline_dir = RESULTS_ROOT / "baseline"
    if fewshot_suffix:
        baseline_dir = baseline_dir / fewshot_suffix
    baseline_path = baseline_dir / "eval_belebele.json"
    if not baseline_path.exists():
        raise SystemExit(f"{baseline_path} not found -- run phase1_belebele_eval.py --baseline "
                          f"--num-fewshot {args.num_fewshot} first")
    baseline = json.loads(baseline_path.read_text())["results"]

    floor_path = RESULTS_ROOT / (f"belebele_floor_check_result_fewshot{args.num_fewshot}.json" if args.num_fewshot
                                  else "belebele_floor_check_result.json")
    flagged = set(json.loads(floor_path.read_text())["flagged"]) if floor_path.exists() else set()
    if flagged:
        print(f"NOTE: {sorted(flagged)} flagged at-chance by phase1_belebele_floor_check.py -- retention numbers "
              f"for these languages are excluded from macro/worst-language below.\n")

    def cond_dir(seed):
        # base dir only (no pp_ratio/fewshot) -- run_phase1_belebele_grid.py's out_dir_for()
        # nests pp_ratio_X BEFORE fewshot_N on disk (.../scale_.../pp_ratio_0.2/fewshot_5/...),
        # so the ON path below must apply pp_ratio first, fewshot second -- NOT go through a
        # cond_dir that already appended fewshot (that produced a nonexistent
        # .../fewshot_5/pp_ratio_0.2/... path and made every ON lookup silently miss).
        d = RESULTS_ROOT / cond / f"seed{seed}"
        if args.scale_condition:
            d = d / f"scale_{args.scale_condition}_seed{seed}"
        return d

    print(f"=== full-pipeline Belebele retention: condition={cond} pp_ratio={pp_ratio} metric={metric} ===\n")

    per_lang = {lang: {"off": [], "on": []} for lang in EVAL_LANGS}
    for seed in SEEDS:
        base = cond_dir(seed)
        off_path = (base / fewshot_suffix / "eval_belebele.json") if fewshot_suffix else (base / "eval_belebele.json")
        on_base = base / f"pp_ratio_{pp_ratio}"
        on_path = (on_base / fewshot_suffix / "eval_belebele.json") if fewshot_suffix else (on_base / "eval_belebele.json")
        if not off_path.exists() or not on_path.exists():
            print(f"seed {seed}: missing ({'OFF' if not off_path.exists() else 'ON'} not found), skipping")
            continue
        off_data = json.loads(off_path.read_text())["results"]
        on_data = json.loads(on_path.read_text())["results"]
        print(f"--- seed {seed} ---")
        for lang in EVAL_LANGS:
            if lang not in baseline or lang not in off_data or lang not in on_data:
                continue
            base_acc = baseline[lang][metric]
            off_acc, on_acc = off_data[lang][metric], on_data[lang][metric]
            if not base_acc:
                continue
            ret_off, ret_on = off_acc / base_acc, on_acc / base_acc
            per_lang[lang]["off"].append(ret_off)
            per_lang[lang]["on"].append(ret_on)
            flag_note = " [FLAGGED-AT-CHANCE]" if lang in flagged else ""
            print(f"  {lang}: baseline={base_acc:.4f}  OFF={off_acc:.4f}(retention={ret_off:.3f})  "
                  f"ON={on_acc:.4f}(retention={ret_on:.3f}){flag_note}")

    print("\n=== summary (mean retention over available seeds) ===\n")
    summary = {}
    macro_off, macro_on = [], []
    for lang in EVAL_LANGS:
        offs, ons = per_lang[lang]["off"], per_lang[lang]["on"]
        if not offs:
            continue
        mean_off, mean_on = sum(offs) / len(offs), sum(ons) / len(ons)
        summary[lang] = {"retention_off": mean_off, "retention_on": mean_on,
                          "n_seeds": len(offs), "flagged_at_chance": lang in flagged}
        excl_note = " [excluded from macro/worst -- at chance]" if lang in flagged else ""
        if lang not in flagged:
            macro_off.append(mean_off)
            macro_on.append(mean_on)
        print(f"{lang}: retention_off={mean_off:.3f} retention_on={mean_on:.3f} (n={len(offs)} seeds){excl_note}")

    macro_avg_off = macro_avg_on = worst_off = worst_on = None
    if macro_off:
        macro_avg_off, macro_avg_on = sum(macro_off) / len(macro_off), sum(macro_on) / len(macro_on)
        worst_off, worst_on = min(macro_off), min(macro_on)
        print(f"\nmacro-average retention: OFF={macro_avg_off:.3f} ON={macro_avg_on:.3f}")
        print(f"worst-language retention: OFF={worst_off:.3f} ON={worst_on:.3f}")
        verdict = ("Belebele accuracy retention SURVIVES the full pipeline (Fisher+SVD+pruning) -- "
                   "worst-language retention stays close to 1.0" if worst_on > 0.9 else
                   "Effectiveness DROPS meaningfully once pruning is added -- worst-language retention below 0.9")
        print(f"\n=> {verdict}")
    else:
        print("\nNo non-flagged languages with complete data -- cannot compute macro/worst-language retention.")

    out = {"condition": cond, "pp_ratio": pp_ratio, "metric": metric, "per_language": summary,
           "macro_average_retention": {"off": macro_avg_off, "on": macro_avg_on},
           "worst_language_retention": {"off": worst_off, "on": worst_on}}
    out_path = RESULTS_ROOT / f"phase1_belebele_gate_{cond}_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
