"""Pre-registered gate for Phase 1's seed-replicated run (2026-07-24, see
00_docs/03_기술노트.md "1) 예산 정상화 + seed 3개" -- this script is written
and committed BEFORE any of the 3-seed results exist, so its criteria can't
be quietly adjusted after seeing the numbers.

H2 direction being tested: calibration in a language helps retain that same
language's performance after compression ("own-language advantage").

For each language L in {eng_Latn, kor_Hang, zho_Hans}:
    own_gain(L) = mean(bpb_increase(other single-lang conditions, L))
                  - bpb_increase(L's matched condition, L)
    (positive = matched calibration helps L, the H2 direction)

Verdict (fixed in advance, in order of strength):
  - "H2_HELD": all 3 languages have own_gain(L) > 0 in every individual seed
    AND the 3-seed-bootstrapped 95% CI of the mean excludes 0.
  - "H2_PARTIAL": 1-2 languages meet the all-seeds-positive bar, or CIs are
    directionally positive but don't cleanly exclude 0 (acknowledged as
    expected given only 3 seeds -- bootstrap CIs from n=3 are wide).
  - "H2_REJECTED": no language shows a consistent positive own_gain.

Usage: python phase1_seed_gate.py   (run after all 12 seed x condition runs finish)
"""
import json
import random
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
SINGLE_LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only"]
MATCHED_CONDITION = {"eng_Latn": "english_only", "kor_Hang": "korean_only", "zho_Hans": "chinese_only"}
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans"]
N_BOOTSTRAP = 20000


def load_bpb(condition, seed, lang):
    path = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
    data = json.loads(path.read_text())
    return data[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    data = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    return data[lang]["bits_per_byte"]


def bpb_increase_pct(condition, seed, lang, baseline_bpb):
    return 100 * (load_bpb(condition, seed, lang) / baseline_bpb - 1)


def own_gain_per_seed(lang, seed, baseline_bpb):
    matched = MATCHED_CONDITION[lang]
    others = [c for c in SINGLE_LANG_CONDITIONS if c != matched]
    other_incr = [bpb_increase_pct(c, seed, lang, baseline_bpb) for c in others]
    matched_incr = bpb_increase_pct(matched, seed, lang, baseline_bpb)
    return sum(other_incr) / len(other_incr) - matched_incr


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=12345):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def main():
    print("=== Phase 1 seed-gate: own-language advantage (bits-per-byte) ===\n")
    all_seeds_positive = {}
    cis = {}
    for lang in LANGS:
        baseline_bpb = load_baseline_bpb(lang)
        gains = [own_gain_per_seed(lang, seed, baseline_bpb) for seed in SEEDS]
        mean_gain = sum(gains) / len(gains)
        lo, hi = bootstrap_ci(gains)
        excludes_zero = lo > 0
        all_positive = all(g > 0 for g in gains)
        all_seeds_positive[lang] = all_positive
        cis[lang] = (mean_gain, lo, hi, excludes_zero)
        print(f"{lang}: per-seed gains={[f'{g:+.2f}' for g in gains]} "
              f"mean={mean_gain:+.2f} 95% CI=[{lo:+.2f}, {hi:+.2f}] "
              f"excludes_zero={excludes_zero} all_seeds_positive={all_positive}")

    n_strict = sum(1 for lang in LANGS if all_seeds_positive[lang] and cis[lang][3])
    n_all_positive = sum(1 for lang in LANGS if all_seeds_positive[lang])

    print()
    if n_strict == 3:
        verdict = "H2_HELD"
    elif n_all_positive >= 1 or any(cis[lang][0] > 0 for lang in LANGS):
        verdict = "H2_PARTIAL"
    else:
        verdict = "H2_REJECTED"
    print(f"VERDICT: {verdict} (strict pass on {n_strict}/3 languages, "
          f"{n_all_positive}/3 all-seeds-positive)")

    out = {
        "verdict": verdict,
        "per_language": {
            lang: {"per_seed_gains": [own_gain_per_seed(lang, s, load_baseline_bpb(lang)) for s in SEEDS],
                   "mean_gain": cis[lang][0], "ci_lo": cis[lang][1], "ci_hi": cis[lang][2],
                   "excludes_zero": cis[lang][3], "all_seeds_positive": all_seeds_positive[lang]}
            for lang in LANGS
        },
    }
    out_path = RESULTS_ROOT / "phase1_seed_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
