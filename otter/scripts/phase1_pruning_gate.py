"""§4.3(iii) pruning on/off ablation report -- reads run_phase1_pruning_ablation.py's
output and reports, per language, how much pruning (on top of the already-
merged Fisher+SVD-delta model) changes the bpb increase, at 3 seeds.

Unlike every other Phase 1 gate (phase1_placebo_gate.py, phase1_swahili_gate.py,
phase1_2x2_verify_gate.py, ...), this deliberately does NOT compute a
SUPPORTED/INCONCLUSIVE/NOT_SUPPORTED verdict against a noise floor: those
gates all compare against a PLACEBO (same language, different sample pool) to
estimate resampling noise. There is no equivalent placebo for an on/off
toggle -- "prune vs don't prune" isn't a resampling question, so a 2x-margin
noise-floor test doesn't apply here without inventing a new control (e.g. a
"prune a RANDOM pp_ratio-fraction of channels instead of the lowest-metric
ones" sham condition, which run_phase1_pruning_ablation.py doesn't currently
run). Instead this reports the closest honest analogue: whether the sign of
(bpb_increase_ON - bpb_increase_OFF) agrees across all 3 seeds per language --
seed-consistent-direction is evidence the effect isn't seed noise, without
claiming a specific noise floor was cleared.

Usage: python phase1_pruning_gate.py --condition korean_only --pp-ratio 0.2
    (after run_phase1_pruning_ablation.py finishes for the same args)
"""
import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
EVAL_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]


def load_bpb(path, lang):
    return json.loads(path.read_text())[lang]["bits_per_byte"]


def bpb_increase_pct(bpb, baseline_bpb):
    return 100 * (bpb / baseline_bpb - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="korean_only")
    parser.add_argument("--pp-ratio", type=float, default=0.2)
    args = parser.parse_args()
    cond, pp_ratio = args.condition, args.pp_ratio

    baseline = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())

    print(f"=== §4.3(iii) pruning on/off ablation: condition={cond} pp_ratio={pp_ratio} ===\n")

    per_lang_deltas = {lang: [] for lang in EVAL_LANGS}
    for seed in SEEDS:
        off_path = RESULTS_ROOT / cond / f"seed{seed}" / "eval_ppl.json"
        on_path = RESULTS_ROOT / cond / f"seed{seed}" / f"pp_ratio_{pp_ratio}" / "eval_ppl.json"
        if not off_path.exists() or not on_path.exists():
            print(f"seed {seed}: missing ({'OFF' if not off_path.exists() else 'ON'} not found), skipping")
            continue
        off_data = json.loads(off_path.read_text())
        on_data = json.loads(on_path.read_text())
        print(f"--- seed {seed} ---")
        for lang in EVAL_LANGS:
            if lang not in off_data or lang not in on_data:
                continue
            baseline_bpb = baseline[lang]["bits_per_byte"]
            off_incr = bpb_increase_pct(off_data[lang]["bits_per_byte"], baseline_bpb)
            on_incr = bpb_increase_pct(on_data[lang]["bits_per_byte"], baseline_bpb)
            delta = on_incr - off_incr
            per_lang_deltas[lang].append(delta)
            print(f"  {lang}: OFF={off_incr:.2f}%  ON={on_incr:.2f}%  delta(ON-OFF)={delta:+.2f}%p")

    print("\n=== summary (mean over available seeds, sign-consistency across seeds) ===\n")
    summary = {}
    for lang, deltas in per_lang_deltas.items():
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in deltas}
        sign_consistent = len(signs) == 1 and len(deltas) == len(SEEDS)
        direction = "PRUNING_HURTS" if mean_delta > 0 else ("PRUNING_HELPS" if mean_delta < 0 else "NO_CHANGE")
        summary[lang] = {"mean_delta_pp": mean_delta, "per_seed_delta": deltas,
                          "sign_consistent_across_seeds": sign_consistent, "direction": direction}
        print(f"{lang}: mean_delta={mean_delta:+.3f}%p over {len(deltas)} seeds  "
              f"sign_consistent={sign_consistent}  direction={direction}")

    out = {"condition": cond, "pp_ratio": pp_ratio, "per_language": summary}
    out_path = RESULTS_ROOT / f"phase1_pruning_gate_{cond}_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
