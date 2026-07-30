"""Pre-registered gate for Bengali's own-language calibration gain -- mirrors
phase1_swahili_gate.py exactly (same 3-tier placebo-verified verdict, same
vulnerability-proportional check), applied to Bengali instead of Swahili.
Written 2026-07-27, before run_phase1_bengali_grid.py's 9 new runs exist.

Question: does Bengali -- a second, independently-added low-resource/
non-Latin-script language -- show an own-language calibration gain that
clears its own placebo-verified noise floor (same 2x margin standard as
phase1_placebo_gate.py / phase1_swahili_gate.py)?

own_gain(Bengali) = bpb_increase(english_only, seed, ben) - bpb_increase(bengali_only, seed, ben)
noise_floor(Bengali) = max over seed in {0,1,2} of
    |bpb_increase(bengali_only, seed, ben) - bpb_increase(bengali_only_b, seed, ben)|

Verdict (fixed in advance, same three-tier structure as phase1_swahili_gate.py):
    own_gain > 2 * noise_floor  -> SUPPORTED
    0 < own_gain <= 2*floor     -> INCONCLUSIVE
    own_gain <= 0               -> NOT_SUPPORTED

"Vulnerability-proportional" gate (also pre-registered, extending
phase1_swahili_gate.py's KO-vs-Swahili check to a 3rd point): Bengali's
own_gain is compared against BOTH Korean's (5.598%p abs / 24.23% rel) and
Swahili's (7.845%p abs / 37.14% rel, phase1_swahili_gate_result.json) own
gains, on both absolute and relative-ratio scales. Bengali's baseline bpb
(0.8857, LOWER than English's 0.9137 -- see claude_plan.md D-8 checkbox) is
the odd one out among the vulnerability-proportional pattern's 3 prior data
points (EN < KO < Swahili baseline bpb, matched by EN < KO < Swahili
own-gain) -- Bengali baseline bpb doesn't slot cleanly into that ordering
(lower than English's despite being nominally "low-resource"), so this gate
reports where Bengali's own_gain falls RELATIVE to KO/Swahili without
presupposing it must exceed both -- that presupposition only made sense
when baseline bpb ordering and vulnerability were expected to track exactly.

Usage: python phase1_bengali_gate.py   (run after all of
       run_phase1_bengali_grid.py's stream a + b tasks finish)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
MARGIN = 2.0
LANG = "ben_Beng"
KO_OWN_GAIN_ABS = 5.598    # kor_Hang, phase1_seed_gate_result.json
KO_OWN_GAIN_REL = 24.23    # %
SWAHILI_OWN_GAIN_ABS = 7.845   # swh_Latn, phase1_swahili_gate_result.json
SWAHILI_OWN_GAIN_REL = 37.14   # %


def load_bpb(condition, seed, lang):
    path = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    data = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    return data[lang]["bits_per_byte"]


def bpb_increase_pct(condition, seed, lang, baseline_bpb):
    return 100 * (load_bpb(condition, seed, lang) / baseline_bpb - 1)


def main():
    baseline_bpb = load_baseline_bpb(LANG)

    per_seed_gain = []
    per_seed_english = []
    per_seed_bengali = []
    for seed in SEEDS:
        english_incr = bpb_increase_pct("english_only", seed, LANG, baseline_bpb)
        bengali_incr = bpb_increase_pct("bengali_only", seed, LANG, baseline_bpb)
        per_seed_english.append(english_incr)
        per_seed_bengali.append(bengali_incr)
        per_seed_gain.append(english_incr - bengali_incr)
    mean_gain = sum(per_seed_gain) / len(per_seed_gain)
    mean_other = sum(per_seed_english) / len(per_seed_english)
    relative_gain = 100 * mean_gain / mean_other

    per_seed_placebo_diff = []
    for seed in SEEDS:
        bengali_incr = bpb_increase_pct("bengali_only", seed, LANG, baseline_bpb)
        bengali_b_incr = bpb_increase_pct("bengali_only_b", seed, LANG, baseline_bpb)
        per_seed_placebo_diff.append(abs(bengali_incr - bengali_b_incr))
    noise_floor = max(per_seed_placebo_diff)

    if mean_gain <= 0:
        verdict = "NOT_SUPPORTED"
    elif mean_gain > MARGIN * noise_floor:
        verdict = "SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    exceeds_ko_abs = mean_gain > KO_OWN_GAIN_ABS
    exceeds_ko_rel = relative_gain > KO_OWN_GAIN_REL
    exceeds_swahili_abs = mean_gain > SWAHILI_OWN_GAIN_ABS
    exceeds_swahili_rel = relative_gain > SWAHILI_OWN_GAIN_REL

    print("=== Phase 1 Bengali gate ===\n")
    print(f"baseline ben bpb: {baseline_bpb:.4f}")
    print(f"per-seed english_only ben increase %: {[f'{v:.3f}' for v in per_seed_english]}")
    print(f"per-seed bengali_only ben increase %: {[f'{v:.3f}' for v in per_seed_bengali]}")
    print(f"per-seed own_gain (english - bengali): {[f'{v:.3f}' for v in per_seed_gain]}")
    print(f"mean own_gain: {mean_gain:.3f}%p   relative_gain: {relative_gain:.2f}%\n")

    print(f"per-seed |bengali_only - bengali_only_b| diff: {[f'{v:.3f}' for v in per_seed_placebo_diff]}")
    print(f"noise_floor(max): {noise_floor:.3f}%p   threshold(2x): {MARGIN * noise_floor:.3f}%p")
    print(f"VERDICT: {verdict}\n")

    print(f"vs KO: own_gain {mean_gain:.3f}%p {'>' if exceeds_ko_abs else '<='} KO's {KO_OWN_GAIN_ABS}%p "
          f"(abs {'EXCEEDS' if exceeds_ko_abs else 'does not exceed'})")
    print(f"vs KO: relative_gain {relative_gain:.2f}% {'>' if exceeds_ko_rel else '<='} KO's {KO_OWN_GAIN_REL}% "
          f"(rel {'EXCEEDS' if exceeds_ko_rel else 'does not exceed'})")
    print(f"vs Swahili: own_gain {mean_gain:.3f}%p {'>' if exceeds_swahili_abs else '<='} Swahili's "
          f"{SWAHILI_OWN_GAIN_ABS}%p (abs {'EXCEEDS' if exceeds_swahili_abs else 'does not exceed'})")
    print(f"vs Swahili: relative_gain {relative_gain:.2f}% {'>' if exceeds_swahili_rel else '<='} Swahili's "
          f"{SWAHILI_OWN_GAIN_REL}% (rel {'EXCEEDS' if exceeds_swahili_rel else 'does not exceed'})")
    print("\nNote: unlike phase1_swahili_gate.py's KO-only comparison, this does NOT collapse the above into a\n"
          "single 'vulnerability_proportional' verdict -- Bengali's baseline bpb (see module docstring) breaks\n"
          "the EN<KO<Swahili ordering the hypothesis was built on, so read the four exceeds_* fields directly.")

    out = {
        "baseline_bpb": baseline_bpb,
        "per_seed_english_only_incr": per_seed_english,
        "per_seed_bengali_only_incr": per_seed_bengali,
        "per_seed_own_gain": per_seed_gain,
        "mean_own_gain": mean_gain,
        "relative_gain_pct": relative_gain,
        "per_seed_placebo_diff": per_seed_placebo_diff,
        "noise_floor": noise_floor,
        "margin_threshold": MARGIN * noise_floor,
        "verdict": verdict,
        "exceeds_ko_absolute": exceeds_ko_abs,
        "exceeds_ko_relative": exceeds_ko_rel,
        "exceeds_swahili_absolute": exceeds_swahili_abs,
        "exceeds_swahili_relative": exceeds_swahili_rel,
    }
    out_path = RESULTS_ROOT / "phase1_bengali_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
