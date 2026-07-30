"""Phase 1: cross-checkpoint XNLI metrics -- macro average, worst-language
accuracy, retention/delta vs baseline, and own-language downstream gain --
read across every condition's eval_xnli.json (phase1_xnli_eval.py). Same
division of labor as phase1_41_headline_gate.py vs phase1_merge_eval.py:
this script computes nothing new from the model, only aggregates already-
written per-checkpoint results.

Only {en, zh, sw} are scored (official-XNLI intersection of this project's
5 calibration languages -- see phase1_xnli_eval.py's module docstring).
korean_only/bengali_only checkpoints ARE included as calibration rows (their
transfer/interference effect on en/zh/sw is exactly the RQ-D2 question) --
they just have no own-language column (own_language_gain() returns None for
them, not an error).

Usage:
    conda run -n d2moe_env python phase1_xnli_analyze.py [--smoke] [--scale-diagonal]
"""
import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")

SINGLE_LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
BALANCED_CONDITION = "mixed_5lang"
ALL_CONDITIONS = SINGLE_LANG_CONDITIONS + [BALANCED_CONDITION]
CONDITION_LABEL = {
    "english_only": "English", "korean_only": "Korean", "chinese_only": "Chinese",
    "swahili_only": "Swahili", "bengali_only": "Bengali", "mixed_5lang": "Balanced",
}
# Only conditions whose calibration language IS an XNLI eval language have an
# own-language cell (RQ-D3, §10 of the design) -- korean_only/bengali_only
# never appear as a key here, by construction, not omission.
MATCHED_CONDITION = {"en": "english_only", "zh": "chinese_only", "sw": "swahili_only"}
XNLI_LANGS = ["en", "zh", "sw"]


def result_path(condition, seed, smoke=False, scale_diagonal=False):
    if condition == "baseline":
        base = RESULTS_ROOT / "baseline"
    elif scale_diagonal:
        base = RESULTS_ROOT / condition / f"seed{seed}" / f"scale_{condition}_seed{seed}"
    else:
        base = RESULTS_ROOT / condition / f"seed{seed}"
    return base / ("eval_xnli_smoke.json" if smoke else "eval_xnli.json")


def load_acc(condition, seed, lang, smoke=False, scale_diagonal=False):
    p = result_path(condition, seed, smoke, scale_diagonal)
    return json.loads(p.read_text())["results"][lang]["acc"]


def missing_paths(seeds, smoke=False, scale_diagonal=False):
    missing = []
    for condition in ALL_CONDITIONS:
        for seed in seeds:
            p = result_path(condition, seed, smoke, scale_diagonal)
            if not p.exists():
                missing.append(str(p))
    bp = result_path("baseline", None, smoke, False)
    if not bp.exists():
        missing.append(str(bp))
    return missing


def macro_avg(condition, seed, smoke=False, scale_diagonal=False):
    accs = [load_acc(condition, seed, lang, smoke, scale_diagonal) for lang in XNLI_LANGS]
    return sum(accs) / len(accs)


def worst_lang(condition, seed, smoke=False, scale_diagonal=False):
    accs = {lang: load_acc(condition, seed, lang, smoke, scale_diagonal) for lang in XNLI_LANGS}
    worst = min(accs, key=accs.get)
    return worst, accs[worst]


def retention_pct(condition, seed, lang, baseline_acc, smoke=False, scale_diagonal=False):
    return 100 * load_acc(condition, seed, lang, smoke, scale_diagonal) / baseline_acc[lang]


def delta_acc(condition, seed, lang, baseline_acc, smoke=False, scale_diagonal=False):
    return load_acc(condition, seed, lang, smoke, scale_diagonal) - baseline_acc[lang]


def own_language_gain(lang, seed, smoke=False, scale_diagonal=False):
    """§10: own-calibration accuracy minus the mean of every OTHER
    single-language calibration's accuracy on this XNLI language (mixed_5lang
    excluded from the "other" average -- it's a different-kind baseline, see
    balanced_gap()). None if `lang` has no matched single-language condition
    (never true for en/zh/sw given MATCHED_CONDITION, but kept as a guard)."""
    matched = MATCHED_CONDITION.get(lang)
    if matched is None:
        return None
    others = [c for c in SINGLE_LANG_CONDITIONS if c != matched]
    other_acc = [load_acc(c, seed, lang, smoke, scale_diagonal) for c in others]
    matched_acc = load_acc(matched, seed, lang, smoke, scale_diagonal)
    return matched_acc - sum(other_acc) / len(other_acc)


def balanced_gap(lang, seed, smoke=False, scale_diagonal=False):
    matched = MATCHED_CONDITION.get(lang)
    if matched is None:
        return None
    return (load_acc(matched, seed, lang, smoke, scale_diagonal)
            - load_acc(BALANCED_CONDITION, seed, lang, smoke, scale_diagonal))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="read eval_xnli_smoke.json instead of eval_xnli.json")
    parser.add_argument("--scale-diagonal", action="store_true",
                         help="read the own-language-whitened-SVD diagonal (scale_<condition>_seed<seed>/) "
                              "instead of the plain-SVD path -- matches phase1_41_headline_gate.py's diag_path()")
    args = parser.parse_args()

    missing = missing_paths([args.seed], args.smoke, args.scale_diagonal)
    if missing:
        print("[xnli-analyze] MISSING results, run phase1_xnli_eval.py first:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)

    baseline_acc = {lang: load_acc("baseline", None, lang, args.smoke, False) for lang in XNLI_LANGS}
    print(f"[xnli-analyze] baseline: " + ", ".join(f"{l}={baseline_acc[l]:.4f}" for l in XNLI_LANGS))
    print()

    header = f"{'Calibration':<12} " + " ".join(f"{l.upper():>14}" for l in XNLI_LANGS) + f" {'MacroAvg':>10} {'Worst':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Original':<12} " + " ".join(f"{baseline_acc[l]:>14.4f}" for l in XNLI_LANGS)
          + f" {sum(baseline_acc.values()) / 3:>10.4f} {min(baseline_acc.values()):>10.4f}")

    for condition in ALL_CONDITIONS:
        cells = []
        for lang in XNLI_LANGS:
            acc = load_acc(condition, args.seed, lang, args.smoke, args.scale_diagonal)
            delta = delta_acc(condition, args.seed, lang, baseline_acc, args.smoke, args.scale_diagonal)
            cells.append(f"{acc:.4f}({delta:+.4f})")
        m = macro_avg(condition, args.seed, args.smoke, args.scale_diagonal)
        w_lang, w_acc = worst_lang(condition, args.seed, args.smoke, args.scale_diagonal)
        print(f"{CONDITION_LABEL[condition]:<12} " + " ".join(f"{c:>14}" for c in cells)
              + f" {m:>10.4f} {w_acc:>10.4f}({w_lang})")

    print()
    print("[xnli-analyze] own-language gain (RQ-D3, only defined for en/zh/sw-matched conditions):")
    for lang in XNLI_LANGS:
        gain = own_language_gain(lang, args.seed, args.smoke, args.scale_diagonal)
        gap = balanced_gap(lang, args.seed, args.smoke, args.scale_diagonal)
        print(f"  {lang}: own_gain={gain:+.4f}  balanced_gap={gap:+.4f}")

    print()
    print("[xnli-analyze] retention % vs original:")
    for condition in ALL_CONDITIONS:
        cells = [f"{lang}={retention_pct(condition, args.seed, lang, baseline_acc, args.smoke, args.scale_diagonal):.1f}%"
                 for lang in XNLI_LANGS]
        print(f"  {CONDITION_LABEL[condition]:<12} " + " ".join(cells))


if __name__ == "__main__":
    main()
