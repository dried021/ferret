"""Pre-registered gate for the seed-replicated + placebo-verified Fisher x
Scale 2x2 (2026-07-26, written before run_phase1_2x2_verify.py's 24 tasks
existed -- see 00_docs/03_기술노트.md "2) whitening 복원").

Question: does own_scale_gain(KO | Fisher=KO) -- the core claim from the
seed=1 preliminary 2x2 ("whitening language dominates Fisher language for
KO") -- survive (a) 3-seed replication and (b) a real placebo, the same two
checks the main Phase 1 Fisher result had to pass?

own_scale_gain(KO | Fisher=KO, seed) = bpb_incr(Fisher=KO,Scale=EN,seed,KO)
                                        - bpb_incr(Fisher=KO,Scale=KO,seed,KO)

noise_floor_scale = max over seed and over {KO-side, EN-side} placebo pairs of
    |bpb_incr(Fisher=KO,Scale=KO,seed,KO)   - bpb_incr(Fisher=KO,Scale=KO_b,seed,KO)|
    |bpb_incr(Fisher=KO,Scale=EN,seed,KO)   - bpb_incr(Fisher=KO,Scale=EN_b,seed,KO)|
(same-Scale-language, different-sample-pool comparison -- Toy0/phase1_placebo_gate.py's
placebo logic, applied to the Scale axis instead of Fisher axis.)

Verdict (fixed in advance, same 3-tier structure as phase1_placebo_gate.py):
    mean(own_scale_gain) > 2 * noise_floor_scale -> SUPPORTED
    0 < mean(own_scale_gain) <= 2*floor          -> INCONCLUSIVE
    mean(own_scale_gain) <= 0                     -> NOT_SUPPORTED

Path-independence (secondary, not a verdict criterion): same computation
holding Fisher=EN instead, to see if the whitening effect's size/sign is
consistent regardless of which Fisher was used.

Usage: python phase1_2x2_verify_gate.py   (after all 24 tasks in
       run_phase1_2x2_verify.py finish)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
MARGIN = 2.0
LANG = "kor_Hang"


def load_bpb(fisher_cond, scale_cond, seed, lang):
    path = RESULTS_ROOT / fisher_cond / f"seed{seed}" / f"scale_{scale_cond}_seed{seed}" / "eval_ppl.json"
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    return json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())[lang]["bits_per_byte"]


def incr(fisher_cond, scale_cond, seed, lang, baseline_bpb):
    return 100 * (load_bpb(fisher_cond, scale_cond, seed, lang) / baseline_bpb - 1)


def main():
    baseline_bpb = load_baseline_bpb(LANG)

    print("=== per-seed KO bpb increase %, Fisher=korean_only fixed ===\n")
    per_seed_gain_ko_fisher = []
    per_seed_gain_en_fisher = []
    floor_candidates = []
    for seed in SEEDS:
        scale_en = incr("korean_only", "english_only", seed, LANG, baseline_bpb)
        scale_ko = incr("korean_only", "korean_only", seed, LANG, baseline_bpb)
        scale_en_b = incr("korean_only", "english_only_b", seed, LANG, baseline_bpb)
        scale_ko_b = incr("korean_only", "korean_only_b", seed, LANG, baseline_bpb)
        gain = scale_en - scale_ko
        per_seed_gain_ko_fisher.append(gain)
        floor_ko_side = abs(scale_ko - scale_ko_b)
        floor_en_side = abs(scale_en - scale_en_b)
        floor_candidates.extend([floor_ko_side, floor_en_side])
        print(f"seed {seed}: Scale=EN {scale_en:.2f}%  Scale=KO {scale_ko:.2f}%  "
              f"Scale=EN_b {scale_en_b:.2f}%  Scale=KO_b {scale_ko_b:.2f}%  "
              f"own_scale_gain={gain:.2f}%p  floor(KO-side)={floor_ko_side:.2f}  floor(EN-side)={floor_en_side:.2f}")

    print("\n=== per-seed KO bpb increase %, Fisher=english_only (path-independence check) ===\n")
    for seed in [0, 1, 2]:
        try:
            scale_en = incr("english_only", "english_only", seed, LANG, baseline_bpb)
            scale_ko = incr("english_only", "korean_only", seed, LANG, baseline_bpb)
            gain = scale_en - scale_ko
            per_seed_gain_en_fisher.append(gain)
            print(f"seed {seed}: Scale=EN {scale_en:.2f}%  Scale=KO {scale_ko:.2f}%  own_scale_gain={gain:.2f}%p")
        except FileNotFoundError:
            print(f"seed {seed}: not available")

    mean_gain = sum(per_seed_gain_ko_fisher) / len(per_seed_gain_ko_fisher)
    noise_floor_scale = max(floor_candidates)
    threshold = MARGIN * noise_floor_scale

    if mean_gain <= 0:
        verdict = "NOT_SUPPORTED"
    elif mean_gain > threshold:
        verdict = "SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nper-seed own_scale_gain (Fisher=KO): {[f'{g:.2f}' for g in per_seed_gain_ko_fisher]}")
    print(f"mean own_scale_gain(KO | Fisher=KO) = {mean_gain:.3f}%p")
    print(f"noise_floor_scale (max over seeds and EN/KO-side placebo pairs) = {noise_floor_scale:.3f}%p")
    print(f"threshold (2x floor) = {threshold:.3f}%p")
    print(f"\nVERDICT: {verdict}")

    if per_seed_gain_en_fisher:
        consistency = abs(mean_gain - sum(per_seed_gain_en_fisher) / len(per_seed_gain_en_fisher))
        print(f"\nPath-independence: |mean_gain(Fisher=KO) - mean_gain(Fisher=EN)| = {consistency:.3f}%p")

    out = {
        "per_seed_gain_fisher_ko": per_seed_gain_ko_fisher,
        "per_seed_gain_fisher_en": per_seed_gain_en_fisher,
        "mean_own_scale_gain": mean_gain,
        "noise_floor_scale": noise_floor_scale,
        "threshold": threshold,
        "verdict": verdict,
    }
    out_path = RESULTS_ROOT / "phase1_2x2_verify_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
