"""Pre-registered placebo gate (2026-07-25, written and committed BEFORE the
english_only_b / korean_only_b runs exist -- see 00_docs/03_기술노트.md
"placebo" section for the full rationale).

Question: is each language's own-language gain (from phase1_seed_gate.py,
computed from arm-A conditions only) distinguishable from the noise floor of
simply recalibrating on a DIFFERENT sample of the SAME language?

Noise floor definition (fixed in advance):
    noise_floor(lang) = max over seed in {0,1,2} of
        | bpb_increase(A-arm condition, seed, lang) - bpb_increase(B-arm condition, seed, lang) |
    i.e. the largest same-seed A-vs-B disagreement, in percentage points of
    relative bpb increase, across the 3 seed pairs. This is deliberately the
    max (not the mean) -- a conservative (hard-to-pass) floor.

Verdict per language (fixed in advance, in order of strength):
    own_gain(lang) > 2 * noise_floor(lang)  -> "SUPPORTED" (margin matches the
        2x factor used for this check, chosen to be stricter than Toy0's 1.5x
        gate since this is a placebo-vs-effect comparison, not a repeatability
        check)
    0 < own_gain(lang) <= 2 * noise_floor(lang) -> "INCONCLUSIVE" (gain and
        floor are the same order of magnitude -- can't rule out noise)
    own_gain(lang) <= 0 -> "NOT_SUPPORTED"

Only EN and KO have a placebo arm (english_only_b, korean_only_b) -- ZH has
none (see 00_docs/03_기술노트.md for why: EN-B confirms the "already flagged
as inconclusive" downgrade, KO-B defends the paper's central claim; ZH was
deprioritized given budget). ZH is reported as "NO_PLACEBO" throughout, not
silently omitted.

Usage: python phase1_placebo_gate.py   (run after both placebo conditions x 3 seeds finish)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
MARGIN = 2.0

PAIRS = {
    "eng_Latn": ("english_only", "english_only_b"),
    "kor_Hang": ("korean_only", "korean_only_b"),
}
# own_gain values from phase1_seed_gate.py's 2026-07-25 run (arm-A only,
# unaffected by the placebo arm -- reproduced here as fixed reference values
# so this script doesn't need to re-derive them; cross-checked against
# phase1_seed_gate_result.json at runtime, see main()).
EXPECTED_OWN_GAIN = {"eng_Latn": 0.75, "kor_Hang": 5.60, "zho_Hans": 0.71}


def load_bpb(condition, seed, lang):
    path = RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    data = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    return data[lang]["bits_per_byte"]


def bpb_increase_pct(condition, seed, lang, baseline_bpb):
    return 100 * (load_bpb(condition, seed, lang) / baseline_bpb - 1)


def main():
    gate_result_path = RESULTS_ROOT / "phase1_seed_gate_result.json"
    own_gain = dict(EXPECTED_OWN_GAIN)
    if gate_result_path.exists():
        gate_result = json.loads(gate_result_path.read_text())
        for lang in own_gain:
            recorded = gate_result["per_language"][lang]["mean_gain"]
            if abs(recorded - own_gain[lang]) > 0.01:
                print(f"WARNING: {lang} own_gain mismatch -- expected {own_gain[lang]}, "
                      f"phase1_seed_gate_result.json has {recorded:.2f}. Using the recorded value.")
            own_gain[lang] = recorded

    print("=== Phase 1 placebo gate: same-language resampling noise floor ===\n")
    results = {}
    for lang, (cond_a, cond_b) in PAIRS.items():
        baseline_bpb = load_baseline_bpb(lang)
        per_seed_diff = []
        for seed in SEEDS:
            incr_a = bpb_increase_pct(cond_a, seed, lang, baseline_bpb)
            incr_b = bpb_increase_pct(cond_b, seed, lang, baseline_bpb)
            per_seed_diff.append(abs(incr_a - incr_b))
        floor = max(per_seed_diff)
        gain = own_gain[lang]
        if gain <= 0:
            verdict = "NOT_SUPPORTED"
        elif gain > MARGIN * floor:
            verdict = "SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"
        results[lang] = {"per_seed_abs_diff": per_seed_diff, "noise_floor": floor,
                          "own_gain": gain, "margin_threshold": MARGIN * floor, "verdict": verdict}
        print(f"{lang}: per-seed |A-B| diff={[f'{d:.3f}' for d in per_seed_diff]} "
              f"noise_floor(max)={floor:.3f} own_gain={gain:.3f} "
              f"threshold({MARGIN}x floor)={MARGIN*floor:.3f} -> {verdict}")

    results["zho_Hans"] = {"verdict": "NO_PLACEBO", "own_gain": own_gain["zho_Hans"]}
    print(f"zho_Hans: NO_PLACEBO (no chinese_only_b run) own_gain={own_gain['zho_Hans']:.3f}")

    out_path = RESULTS_ROOT / "phase1_placebo_gate_result.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
