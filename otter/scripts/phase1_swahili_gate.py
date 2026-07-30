"""Pre-registered gate for the Swahili low-resource-language experiment
(2026-07-25 -- criteria fixed in the /loop prompt used to launch
run_phase1_swahili.py, before any of its 6 runs existed).

Question: does Swahili -- more vulnerable/lower-resource for this model than
Korean -- show an own-language calibration gain that exceeds Korean's, on
BOTH absolute and relative-ratio terms, and does that gain clear its own
placebo-verified noise floor (same 2x-margin standard as
phase1_placebo_gate.py)?

own_gain(Swahili) = bpb_increase(english_only, seed, swh) - bpb_increase(swahili_only, seed, swh)
    -- English stands in as the "distant/default calibration" reference
    (reusing englis_only's ALREADY-COMPUTED Fisher/freq artifacts, re-run
    only through merge+eval so its Swahili score exists).
noise_floor(Swahili) = max over seed in {0,1,2} of
    |bpb_increase(swahili_only, seed, swh) - bpb_increase(swahili_only_b, seed, swh)|

Verdict (fixed in advance, same three-tier structure as phase1_placebo_gate.py):
    own_gain > 2 * noise_floor  -> SUPPORTED
    0 < own_gain <= 2*floor     -> INCONCLUSIVE
    own_gain <= 0               -> NOT_SUPPORTED

"Vulnerability-proportional" gate (also pre-registered, in 04_전체요약.md
"다음 단계" 2): Swahili's own_gain must exceed KO's on BOTH the absolute
scale (> 5.598%p, kor_Hang's own_gain from phase1_seed_gate_result.json) and
the relative-ratio scale (> 24.23%, own_gain / mean(other-conditions' bpb
increase)) for the "vulnerability-proportional" hypothesis to be supported --
otherwise KO was an exceptional case, not a pattern.

Usage: python phase1_swahili_gate.py   (run after all 6 swahili_only(_b) runs finish)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
MARGIN = 2.0
LANG = "swh_Latn"
KO_OWN_GAIN_ABS = 5.598   # kor_Hang, from phase1_seed_gate_result.json (2026-07-25 run)
KO_OWN_GAIN_REL = 24.23   # %, from the same run's relative-ratio check


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
    per_seed_swahili = []
    for seed in SEEDS:
        english_incr = bpb_increase_pct("english_only", seed, LANG, baseline_bpb)
        swahili_incr = bpb_increase_pct("swahili_only", seed, LANG, baseline_bpb)
        per_seed_english.append(english_incr)
        per_seed_swahili.append(swahili_incr)
        per_seed_gain.append(english_incr - swahili_incr)
    mean_gain = sum(per_seed_gain) / len(per_seed_gain)
    mean_other = sum(per_seed_english) / len(per_seed_english)
    relative_gain = 100 * mean_gain / mean_other

    per_seed_placebo_diff = []
    for seed in SEEDS:
        swahili_incr = bpb_increase_pct("swahili_only", seed, LANG, baseline_bpb)
        swahili_b_incr = bpb_increase_pct("swahili_only_b", seed, LANG, baseline_bpb)
        per_seed_placebo_diff.append(abs(swahili_incr - swahili_b_incr))
    noise_floor = max(per_seed_placebo_diff)

    if mean_gain <= 0:
        verdict = "NOT_SUPPORTED"
    elif mean_gain > MARGIN * noise_floor:
        verdict = "SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    exceeds_ko_abs = mean_gain > KO_OWN_GAIN_ABS
    exceeds_ko_rel = relative_gain > KO_OWN_GAIN_REL
    vulnerability_proportional = exceeds_ko_abs and exceeds_ko_rel

    print("=== Phase 1 Swahili gate ===\n")
    print(f"baseline swh bpb: {baseline_bpb:.4f}")
    print(f"per-seed english_only swh increase %: {[f'{v:.3f}' for v in per_seed_english]}")
    print(f"per-seed swahili_only swh increase %: {[f'{v:.3f}' for v in per_seed_swahili]}")
    print(f"per-seed own_gain (english - swahili): {[f'{v:.3f}' for v in per_seed_gain]}")
    print(f"mean own_gain: {mean_gain:.3f}%p   relative_gain: {relative_gain:.2f}%\n")

    print(f"per-seed |swahili_only - swahili_only_b| diff: {[f'{v:.3f}' for v in per_seed_placebo_diff]}")
    print(f"noise_floor(max): {noise_floor:.3f}%p   threshold(2x): {MARGIN * noise_floor:.3f}%p")
    print(f"VERDICT: {verdict}\n")

    print(f"vs KO: own_gain {mean_gain:.3f}%p {'>' if exceeds_ko_abs else '<='} KO's {KO_OWN_GAIN_ABS}%p "
          f"(abs {'EXCEEDS' if exceeds_ko_abs else 'does not exceed'})")
    print(f"vs KO: relative_gain {relative_gain:.2f}% {'>' if exceeds_ko_rel else '<='} KO's {KO_OWN_GAIN_REL}% "
          f"(rel {'EXCEEDS' if exceeds_ko_rel else 'does not exceed'})")
    print(f"VULNERABILITY-PROPORTIONAL HYPOTHESIS: "
          f"{'SUPPORTED' if vulnerability_proportional else 'NOT SUPPORTED (KO may be an exceptional case)'}")

    out = {
        "baseline_bpb": baseline_bpb,
        "per_seed_english_only_incr": per_seed_english,
        "per_seed_swahili_only_incr": per_seed_swahili,
        "per_seed_own_gain": per_seed_gain,
        "mean_own_gain": mean_gain,
        "relative_gain_pct": relative_gain,
        "per_seed_placebo_diff": per_seed_placebo_diff,
        "noise_floor": noise_floor,
        "margin_threshold": MARGIN * noise_floor,
        "verdict": verdict,
        "exceeds_ko_absolute": exceeds_ko_abs,
        "exceeds_ko_relative": exceeds_ko_rel,
        "vulnerability_proportional_hypothesis": vulnerability_proportional,
    }
    out_path = RESULTS_ROOT / "phase1_swahili_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
