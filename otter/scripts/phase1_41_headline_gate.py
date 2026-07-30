"""§4.1 headline gate (2026-07-27) -- pre-registered per 06_논문_구성.md §4.1,
written before run_phase1_41_diagonal.py's grid finishes so its criteria
can't be adjusted after seeing the numbers (same discipline as
phase1_seed_gate.py/phase1_placebo_gate.py/phase1_swahili_gate.py, which
this script supersedes as the paper's Table 1 / Figure 1 source -- it does
NOT delete or reinterpret their results, which stay valid as whitening-OFF,
now §4.3's "Fisher-path-alone" evidence per 01_plans/claude_plan.md).

Grid: 6 calibration conditions {english_only, korean_only, chinese_only,
swahili_only, bengali_only, mixed_5lang("Balanced")} x 5 eval languages
{eng_Latn, kor_Hang, zho_Hans, swh_Latn, ben_Beng}, whitening-ON DIAGONAL
(Fisher-language == Scale-language, same seed) -- the full D2-MoE pipeline,
not the Fisher-path-only pilot the earlier per-condition gates used.

own_gain(L, seed) = mean_{single-lang C != matched(L)} bpb_incr(C, seed, L)
                     - bpb_incr(matched(L), seed, L)
"mixed_5lang" is EXCLUDED from the "other conditions" average (it isn't a
foreign-language condition for any L, and isn't anyone's "matched"
condition either) -- own_gain is defined only among the 5 single-language
conditions, same shape as phase1_seed_gate.py's original 3-language version.

Per 06_논문_구성.md §4.1's own reporting rule (chosen there specifically
BECAUSE n=3 seeds is too small for a trustworthy bootstrap CI): report
per-seed raw own_gain values and whether all 3 seeds agree in sign, not a
confidence interval.

noise_floor(L) = max over seed of |bpb_incr(matched(L),seed,L) - bpb_incr(matched(L)+"_b",seed,L)|
-- only defined for {eng_Latn, kor_Hang, swh_Latn} (english_only_b,
korean_only_b, swahili_only_b exist); zho_Hans and ben_Beng are reported
NO_PLACEBO, not silently omitted (same convention as phase1_placebo_gate.py
for zho_Hans -- chinese_only_b / bengali_only_b were never run, out of
04_전체요약.md's stated budget prioritization).

Verdict per language (fixed in advance, in order of strength):
  - SUPPORTED:              sign-consistent across all 3 seeds AND (if a
                             placebo exists) mean(own_gain) > 2 * noise_floor
  - SIGN_CONSISTENT_NO_FLOOR: sign-consistent across all 3 seeds, but no
                             placebo run exists to compare against (zho_Hans,
                             ben_Beng) -- direction is trustworthy, magnitude
                             vs noise is not yet checked
  - INCONCLUSIVE:            sign-consistent but mean(own_gain) <= 2*floor
                             (placebo exists and gain doesn't clear it)
  - NOT_SUPPORTED:           not sign-consistent across the 3 seeds

Usage: python phase1_41_headline_gate.py   (run after run_phase1_41_diagonal.py finishes)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
MARGIN = 2.0

SINGLE_LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
BALANCED_CONDITION = "mixed_5lang"
ALL_CONDITIONS = SINGLE_LANG_CONDITIONS + [BALANCED_CONDITION]
MATCHED_CONDITION = {
    "eng_Latn": "english_only", "kor_Hang": "korean_only", "zho_Hans": "chinese_only",
    "swh_Latn": "swahili_only", "ben_Beng": "bengali_only",
}
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
PLACEBO_CONDITION = {"eng_Latn": "english_only_b", "kor_Hang": "korean_only_b", "swh_Latn": "swahili_only_b"}
CONDITION_LABEL = {
    "english_only": "English", "korean_only": "Korean", "chinese_only": "Chinese",
    "swahili_only": "Swahili", "bengali_only": "Bengali", "mixed_5lang": "Balanced",
}


def diag_path(condition, seed):
    return RESULTS_ROOT / condition / f"seed{seed}" / f"scale_{condition}_seed{seed}" / "eval_ppl.json"


def load_bpb(condition, seed, lang):
    return json.loads(diag_path(condition, seed).read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    return json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())[lang]["bits_per_byte"]


def bpb_incr(condition, seed, lang, baseline_bpb):
    return 100 * (load_bpb(condition, seed, lang) / baseline_bpb - 1)


def missing_paths():
    missing = []
    for condition in ALL_CONDITIONS:
        for seed in SEEDS:
            p = diag_path(condition, seed)
            if not p.exists():
                missing.append(str(p))
    if not (RESULTS_ROOT / "baseline" / "eval_ppl.json").exists():
        missing.append(str(RESULTS_ROOT / "baseline" / "eval_ppl.json"))
    return missing


def own_gain_per_seed(lang, seed, baseline_bpb):
    matched = MATCHED_CONDITION[lang]
    others = [c for c in SINGLE_LANG_CONDITIONS if c != matched]
    other_incr = [bpb_incr(c, seed, lang, baseline_bpb) for c in others]
    matched_incr = bpb_incr(matched, seed, lang, baseline_bpb)
    return sum(other_incr) / len(other_incr) - matched_incr


def noise_floor(lang, baseline_bpb):
    if lang not in PLACEBO_CONDITION:
        return None
    matched = MATCHED_CONDITION[lang]
    placebo = PLACEBO_CONDITION[lang]
    diffs = []
    for seed in SEEDS:
        a = bpb_incr(matched, seed, lang, baseline_bpb)
        b = bpb_incr(placebo, seed, lang, baseline_bpb)
        diffs.append(abs(a - b))
    return max(diffs)


def main():
    missing = missing_paths()
    if missing:
        print("=== NOT READY: missing eval_ppl.json files ===")
        for m in missing:
            print(f"  {m}")
        print(f"\n{len(missing)} file(s) missing -- run run_phase1_41_diagonal.py first (or check its progress).")
        return

    print("=== §4.1 headline grid: bpb increase % over baseline, whitening-ON diagonal ===\n")
    grid = {}
    baseline_bpb = {lang: load_baseline_bpb(lang) for lang in LANGS}
    header = "Calib \\ Eval".ljust(16) + "".join(lang.split("_")[0].rjust(9) for lang in LANGS)
    print(header)
    for condition in ALL_CONDITIONS:
        row_vals = []
        for lang in LANGS:
            vals = [bpb_incr(condition, seed, lang, baseline_bpb[lang]) for seed in SEEDS]
            mean_val = sum(vals) / len(vals)
            grid[(condition, lang)] = {"per_seed": vals, "mean": mean_val}
            row_vals.append(mean_val)
        print(CONDITION_LABEL[condition].ljust(16) + "".join(f"{v:9.3f}" for v in row_vals))

    print("\n=== own-language gain (own_gain > 0 == matched calibration helps) ===\n")
    per_lang_result = {}
    for lang in LANGS:
        gains = [own_gain_per_seed(lang, seed, baseline_bpb[lang]) for seed in SEEDS]
        mean_gain = sum(gains) / len(gains)
        sign_consistent = all(g > 0 for g in gains) or all(g < 0 for g in gains)
        floor = noise_floor(lang, baseline_bpb[lang])

        if not sign_consistent:
            verdict = "NOT_SUPPORTED"
        elif floor is None:
            verdict = "SIGN_CONSISTENT_NO_FLOOR"
        elif mean_gain > MARGIN * floor:
            verdict = "SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"

        per_lang_result[lang] = {
            "per_seed_gain": gains, "mean_gain": mean_gain, "sign_consistent": sign_consistent,
            "noise_floor": floor, "verdict": verdict,
        }
        floor_str = f"{floor:.3f}" if floor is not None else "NO_PLACEBO"
        print(f"{lang}: per-seed={[f'{g:+.3f}' for g in gains]} mean={mean_gain:+.3f} "
              f"sign_consistent={sign_consistent} noise_floor={floor_str} -> {verdict}")

    out = {
        "grid": {f"{c}|{l}": v for (c, l), v in grid.items()},
        "own_gain": per_lang_result,
        "conditions": ALL_CONDITIONS,
        "langs": LANGS,
    }
    out_path = RESULTS_ROOT / "phase1_41_headline_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
