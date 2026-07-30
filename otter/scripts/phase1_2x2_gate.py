"""Fisher x Scale 2x2 analysis (2026-07-26, criteria fixed in
00_docs/03_기술노트.md "2) whitening 복원" BEFORE any of the 4 cells existed).

Question: is KO's own-language advantage driven by the Fisher-merge path, the
whitening(SVD scale) path, or both? Compare, holding Fisher fixed at KO, how
much swapping the SCALE calibration language (EN vs KO) moves KO's bpb
increase -- against the ALREADY-ESTABLISHED Fisher-alone effect (+5.598%p,
from phase1_seed_gate_result.json, Phase 1's main 3-seed run).

own_scale_gain(KO | Fisher=X) = bpb_increase(Fisher=X, Scale=EN, KO-eval)
                                 - bpb_increase(Fisher=X, Scale=KO, KO-eval)

Verdict (fixed in advance):
    own_scale_gain(KO | Fisher=KO) > FISHER_ALONE_GAIN (5.598%p) -> "WHITENING_DOMINANT"
    own_scale_gain(KO | Fisher=KO) <= FISHER_ALONE_GAIN            -> "FISHER_DOMINANT"
(sign of own_scale_gain itself: positive means matched-language scale still
helps KO, consistent direction with the Fisher-alone result; negative would
mean whitening acts AGAINST the Fisher-alone effect for KO -- worth flagging
regardless of the magnitude comparison above.)

Secondary (path-independence check, not a verdict criterion): repeat with
Fisher=EN fixed -- if own_scale_gain(KO | Fisher=EN) is similar in sign and
rough magnitude to own_scale_gain(KO | Fisher=KO), the whitening effect looks
independent of which Fisher was used; if very different, the two paths
interact.

Usage: python phase1_2x2_gate.py   (run after all 4 cells in run_phase1_2x2.py finish)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEED = 0
LANG = "kor_Hang"
FISHER_ALONE_GAIN = 5.598  # kor_Hang own_gain from phase1_seed_gate_result.json (2026-07-25, 3-seed mean)


def load_bpb(fisher_cond, scale_cond, seed, lang):
    path = RESULTS_ROOT / fisher_cond / f"seed{seed}" / f"scale_{scale_cond}_seed{seed}" / "eval_ppl.json"
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    data = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    return data[lang]["bits_per_byte"]


def incr(fisher_cond, scale_cond, seed, lang, baseline_bpb):
    return 100 * (load_bpb(fisher_cond, scale_cond, seed, lang) / baseline_bpb - 1)


def main():
    baseline_bpb = load_baseline_bpb(LANG)

    grid = {}
    for fisher_cond in ["english_only", "korean_only"]:
        for scale_cond in ["english_only", "korean_only"]:
            grid[(fisher_cond, scale_cond)] = incr(fisher_cond, scale_cond, SEED, LANG, baseline_bpb)

    print("=== Fisher x Scale 2x2 (KO bpb increase %) ===\n")
    header_label = "Fisher \\ Scale"
    print(f"{header_label:16s}{'EN':>10s}{'KO':>10s}")
    for fisher_cond in ["english_only", "korean_only"]:
        row = f"{fisher_cond:16s}"
        for scale_cond in ["english_only", "korean_only"]:
            row += f"{grid[(fisher_cond, scale_cond)]:10.3f}"
        print(row)

    gain_fisher_ko = grid[("korean_only", "english_only")] - grid[("korean_only", "korean_only")]
    gain_fisher_en = grid[("english_only", "english_only")] - grid[("english_only", "korean_only")]

    print(f"\nown_scale_gain(KO | Fisher=KO) = {gain_fisher_ko:.3f}%p")
    print(f"own_scale_gain(KO | Fisher=EN) = {gain_fisher_en:.3f}%p")
    print(f"Fisher-alone gain (reference, from main Phase 1 run) = {FISHER_ALONE_GAIN}%p")

    if gain_fisher_ko > FISHER_ALONE_GAIN:
        verdict = "WHITENING_DOMINANT"
    else:
        verdict = "FISHER_DOMINANT"
    print(f"\nVERDICT: {verdict}")

    direction_note = "same direction as Fisher (helps KO)" if gain_fisher_ko > 0 else "OPPOSITE direction (hurts KO)"
    print(f"Direction check: own_scale_gain(KO | Fisher=KO) is {direction_note}")

    consistency = abs(gain_fisher_ko - gain_fisher_en)
    print(f"\nPath-independence check: |gain(Fisher=KO) - gain(Fisher=EN)| = {consistency:.3f}%p "
          f"({'similar -> paths look independent' if consistency < 1.0 else 'different -> paths interact'})")

    out = {
        "grid": {f"{f}|{s}": v for (f, s), v in grid.items()},
        "own_scale_gain_fisher_ko": gain_fisher_ko,
        "own_scale_gain_fisher_en": gain_fisher_en,
        "fisher_alone_gain_reference": FISHER_ALONE_GAIN,
        "verdict": verdict,
        "path_independence_diff": consistency,
    }
    out_path = RESULTS_ROOT / "phase1_2x2_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
