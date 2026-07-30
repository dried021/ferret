"""Generalized pre-registered gate for the Fisher x Scale 2x2, parametrized
by target language -- same statistical logic as phase1_2x2_verify_gate.py
(which is hardcoded to Korean and already gated SUPPORTED, see
00_docs/03_기술노트.md §2.5), factored out so it can be re-applied to
run_phase1_2x2_lang.py's output for any language (e.g. Swahili, the
language §4.1 actually found most sensitive -- see run_phase1_2x2_lang.py's
docstring for why that matters for §4.3's "centered on the most sensitive
language" pre-registration).

Question: does own_scale_gain(<lang> | Fisher=<lang>) -- the KO 2x2's
"whitening language dominates Fisher language" finding -- replicate for a
different calibration language, at 3 seeds, past a real placebo?

own_scale_gain(seed) = bpb_incr(Fisher=<lang>,Scale=english_only,seed,<lang>)
                        - bpb_incr(Fisher=<lang>,Scale=<lang>,seed,<lang>)

noise_floor_scale = max over seed and over {lang-side, EN-side} placebo pairs of
    |bpb_incr(Fisher=<lang>,Scale=<lang>,seed,<lang>)   - bpb_incr(Fisher=<lang>,Scale=<lang>_b,seed,<lang>)|
    |bpb_incr(Fisher=<lang>,Scale=english_only,seed,<lang>) - bpb_incr(Fisher=<lang>,Scale=english_only_b,seed,<lang>)|

Verdict (fixed in advance, same 3-tier structure as phase1_2x2_verify_gate.py):
    mean(own_scale_gain) > 2 * noise_floor_scale -> SUPPORTED
    0 < mean(own_scale_gain) <= 2*floor          -> INCONCLUSIVE
    mean(own_scale_gain) <= 0                     -> NOT_SUPPORTED

Path-independence (secondary, not a verdict criterion): same computation
holding Fisher=english_only instead.

Usage:
    python phase1_2x2_gate_lang.py --lang-cond swahili_only \\
        --lang-cond-b swahili_only_b --lang-code swh_Latn
    (after run_phase1_2x2_lang.py's tasks finish for the same language)
"""
import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
MARGIN = 2.0
EN, EN_B = "english_only", "english_only_b"


def load_bpb(fisher_cond, scale_cond, seed, lang):
    path = RESULTS_ROOT / fisher_cond / f"seed{seed}" / f"scale_{scale_cond}_seed{seed}" / "eval_ppl.json"
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    return json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())[lang]["bits_per_byte"]


def incr(fisher_cond, scale_cond, seed, lang, baseline_bpb):
    return 100 * (load_bpb(fisher_cond, scale_cond, seed, lang) / baseline_bpb - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-cond", required=True, help="e.g. swahili_only")
    parser.add_argument("--lang-cond-b", required=True, help="e.g. swahili_only_b")
    parser.add_argument("--lang-code", required=True, help="FLORES code, e.g. swh_Latn")
    args = parser.parse_args()
    lang, lang_b, LANG = args.lang_cond, args.lang_cond_b, args.lang_code

    baseline_bpb = load_baseline_bpb(LANG)

    print(f"=== per-seed {LANG} bpb increase %, Fisher={lang} fixed ===\n")
    per_seed_gain_lang_fisher = []
    per_seed_gain_en_fisher = []
    floor_candidates = []
    for seed in SEEDS:
        scale_en = incr(lang, EN, seed, LANG, baseline_bpb)
        scale_lang = incr(lang, lang, seed, LANG, baseline_bpb)
        scale_en_b = incr(lang, EN_B, seed, LANG, baseline_bpb)
        scale_lang_b = incr(lang, lang_b, seed, LANG, baseline_bpb)
        gain = scale_en - scale_lang
        per_seed_gain_lang_fisher.append(gain)
        floor_lang_side = abs(scale_lang - scale_lang_b)
        floor_en_side = abs(scale_en - scale_en_b)
        floor_candidates.extend([floor_lang_side, floor_en_side])
        print(f"seed {seed}: Scale=EN {scale_en:.2f}%  Scale={lang} {scale_lang:.2f}%  "
              f"Scale=EN_b {scale_en_b:.2f}%  Scale={lang}_b {scale_lang_b:.2f}%  "
              f"own_scale_gain={gain:.2f}%p  floor({lang}-side)={floor_lang_side:.2f}  floor(EN-side)={floor_en_side:.2f}")

    print(f"\n=== per-seed {LANG} bpb increase %, Fisher={EN} (path-independence check) ===\n")
    for seed in SEEDS:
        try:
            scale_en = incr(EN, EN, seed, LANG, baseline_bpb)
            scale_lang = incr(EN, lang, seed, LANG, baseline_bpb)
            gain = scale_en - scale_lang
            per_seed_gain_en_fisher.append(gain)
            print(f"seed {seed}: Scale=EN {scale_en:.2f}%  Scale={lang} {scale_lang:.2f}%  own_scale_gain={gain:.2f}%p")
        except FileNotFoundError:
            print(f"seed {seed}: not available")

    mean_gain = sum(per_seed_gain_lang_fisher) / len(per_seed_gain_lang_fisher)
    noise_floor_scale = max(floor_candidates)
    threshold = MARGIN * noise_floor_scale

    if mean_gain <= 0:
        verdict = "NOT_SUPPORTED"
    elif mean_gain > threshold:
        verdict = "SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nper-seed own_scale_gain (Fisher={lang}): {[f'{g:.2f}' for g in per_seed_gain_lang_fisher]}")
    print(f"mean own_scale_gain({LANG} | Fisher={lang}) = {mean_gain:.3f}%p")
    print(f"noise_floor_scale (max over seeds and EN/{lang}-side placebo pairs) = {noise_floor_scale:.3f}%p")
    print(f"threshold (2x floor) = {threshold:.3f}%p")
    print(f"\nVERDICT: {verdict}")

    if per_seed_gain_en_fisher:
        consistency = abs(mean_gain - sum(per_seed_gain_en_fisher) / len(per_seed_gain_en_fisher))
        print(f"\nPath-independence: |mean_gain(Fisher={lang}) - mean_gain(Fisher=EN)| = {consistency:.3f}%p")

    out = {
        "lang_code": LANG,
        "lang_cond": lang,
        "per_seed_gain_fisher_lang": per_seed_gain_lang_fisher,
        "per_seed_gain_fisher_en": per_seed_gain_en_fisher,
        "mean_own_scale_gain": mean_gain,
        "noise_floor_scale": noise_floor_scale,
        "threshold": threshold,
        "verdict": verdict,
    }
    out_path = RESULTS_ROOT / f"phase1_2x2_gate_{lang}_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
