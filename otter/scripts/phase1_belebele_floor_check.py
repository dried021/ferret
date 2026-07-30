"""Pre-flight check, run BEFORE the compressed-model Belebele grid (see
phase1_belebele_eval.py's module docstring for why Belebele was chosen).

DeepSeek-MoE-16B-base is a base (not instruction-tuned) model. For
low-resource languages (Swahili, Bengali especially) multiple-choice
accuracy may sit at or near chance regardless of compression -- if the
UNCOMPRESSED baseline is already at chance for a language, no compression
condition can show a real effect there; any observed difference between
conditions would be noise on top of a floor, not signal. (2026-07-29,
user-flagged methodological risk, raised before committing to the full
pipeline grid.)

0-shot check (2026-07-29): with n=900/language, ALL FIVE languages were
"statistically" above chance by a one-sided binomial test, but the margins
were thin across the board -- even English was only +10.2pp over chance,
Korean/Swahili only +4pp, and Bengali (+3.1pp, p=0.018) failed the alpha=0.01
bar outright. A p-value alone doesn't capture this: with n=900, even a
practically useless +3pp margin can look "significant". So this script
checks BOTH significance (p < ALPHA) AND a practical margin floor
(MIN_MARGIN) -- a language can be statistically above chance yet still be
too close to it for a compression effect to be distinguishable from seed
noise. Given the 0-shot margins were uniformly thin, the project switched to
few-shot (see run_phase1_belebele_grid.py's --num-fewshot); re-run this
check on the few-shot baseline before trusting the grid's retention numbers.

This reads phase1_belebele_eval.py --baseline's output and, per language,
runs a one-sided binomial test of (n_correct out of n_samples) against
p=0.25 (4-way MC chance). It does NOT run anything on GPU itself.

Usage:
    conda run -n d2moe_env python phase1_belebele_eval.py --baseline --num-fewshot 5 --limit 200
    conda run -n d2moe_env python phase1_belebele_floor_check.py --num-fewshot 5
"""
import argparse
import json
from pathlib import Path

from scipy import stats

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
CHANCE = 0.25
ALPHA = 0.01  # conservative on purpose -- a false "looks fine" here wastes the whole downstream grid
MIN_MARGIN = 0.05  # practical floor: below this, even a "significant" margin leaves too little room to
                    # detect a compression-induced drop against seed-to-seed noise (2026-07-29 0-shot finding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-fewshot", type=int, default=0,
                         help="must match the --num-fewshot used for the baseline eval_belebele.json being checked")
    args = parser.parse_args()

    base_dir = RESULTS_ROOT / "baseline"
    if args.num_fewshot:
        base_dir = base_dir / f"fewshot_{args.num_fewshot}"
    path = base_dir / "eval_belebele.json"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `phase1_belebele_eval.py --baseline "
                          f"--num-fewshot {args.num_fewshot}` first (no --limit, or a generous one like "
                          "--limit 400, so the binomial test below has power)")
    data = json.loads(path.read_text())["results"]

    print(f"=== Belebele baseline floor check (chance = 25%, 4-way MC, num_fewshot={args.num_fewshot}) ===\n")
    flagged = []
    thin = []
    for lang, r in data.items():
        acc, n = r.get("acc"), r.get("n_samples")
        if acc is None:
            print(f"{lang}: no acc recorded, skipping")
            continue
        if not n:
            print(f"{lang}: acc={acc:.4f} but n_samples unknown -- can't run binomial test, inspect manually")
            continue
        n_correct = round(acc * n)
        pvalue = stats.binomtest(n_correct, n, CHANCE, alternative="greater").pvalue
        margin = acc - CHANCE
        above_chance = pvalue < ALPHA
        wide_enough = margin >= MIN_MARGIN
        if not above_chance:
            status, flagged = "FLOOR -- at/near chance, flagged", flagged + [lang]
        elif not wide_enough:
            status, thin = f"THIN -- stat-sig but margin < {MIN_MARGIN:.0%}, flagged", thin + [lang]
        else:
            status = "OK (reliably above chance, usable margin)"
        print(f"{lang}: acc={acc:.4f} acc_norm={r.get('acc_norm'):.4f} (n={n}, margin over chance={margin:+.4f}, "
              f"one-sided p={pvalue:.4g})  -> {status}")

    print()
    all_flagged = flagged + thin
    if all_flagged:
        if flagged:
            print(f"AT CHANCE: {flagged} -- Belebele accuracy cannot show a compression effect here at all.")
        if thin:
            print(f"THIN MARGIN (< {MIN_MARGIN:.0%}): {thin} -- technically above chance but likely too close "
                  "to it to distinguish a real compression effect from seed noise.")
        print("Recommendation: for these languages, supplement/replace the Belebele retention claim with a "
              "generation-based metric (e.g. FLORES En->X chrF++), or increase --num-fewshot, before drawing "
              "any full-pipeline conclusion about them.")
    else:
        print("No languages flagged -- Belebele accuracy is usable as the full-pipeline retention metric "
              "for all languages checked.")

    out = {"chance": CHANCE, "alpha": ALPHA, "min_margin": MIN_MARGIN, "num_fewshot": args.num_fewshot,
           "at_chance": flagged, "thin_margin": thin, "flagged": all_flagged,
           "per_language": {lang: {"acc": r.get("acc"), "n_samples": r.get("n_samples")} for lang, r in data.items()}}
    out_name = f"belebele_floor_check_result_fewshot{args.num_fewshot}.json" if args.num_fewshot else "belebele_floor_check_result.json"
    out_path = RESULTS_ROOT / out_name
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
