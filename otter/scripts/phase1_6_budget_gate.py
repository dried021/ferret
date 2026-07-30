"""§6 main comparison gate (Table 3), pre-registered per 06_논문_구성.md §6
-- written before the disagreement_targeted merge+eval runs finish so the
criterion can't be adjusted after seeing the numbers (same discipline as
phase1_41_headline_gate.py/phase1_placebo_gate.py).

Compares, at the SAME total calibration budget (see phase1_6_targeted_
budget.py's reference_budget_tokens): {제안법 (disagreement_targeted),
Balanced (mixed_5lang), 최선 단일 언어 (best single language)} on the worst-
language bpb degradation (primary) and mean bpb degradation (secondary).

Pre-registered verdict (06_논문_구성.md §6, the ONLY gating criterion --
"Balanced는 누구나 떠올리는 기본 대안이므로, 이를 이기지 못하면 방법의 기여를
주장하지 않는다"):
    SUPPORTED     if disagreement_targeted's worst-language bpb increase is
                  STRICTLY LOWER than mixed_5lang's, in ALL 3 seeds.
    NOT_SUPPORTED otherwise.
"최선 단일 언어" is reported as additional context (which single-language
condition has the lowest MEAN-across-seeds worst-language degradation -- one
fixed condition, not re-picked per seed, so it's not cherry-picked after the
fact), but per the doc it does not itself gate the verdict -- only losing to
Balanced does.

Usage: python phase1_6_budget_gate.py   (run after disagreement_targeted's
    phase1_merge_eval.py finishes for seeds 0,1,2 -- same eval_ppl.json path
    every other plain (non-2x2-diagonal) condition uses)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]

PROPOSED_CONDITION = "disagreement_targeted"
BALANCED_CONDITION = "mixed_5lang"  # all-5-language balanced, matching disagreement_targeted's language scope
SINGLE_LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
CONDITION_LABEL = {
    "disagreement_targeted": "Proposed (targeted)", "mixed_5lang": "Balanced",
    "english_only": "English", "korean_only": "Korean", "chinese_only": "Chinese",
    "swahili_only": "Swahili", "bengali_only": "Bengali",
}


def eval_path(condition, seed):
    return RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"


def load_bpb(condition, seed, lang):
    return json.loads(eval_path(condition, seed).read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    return json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())[lang]["bits_per_byte"]


def bpb_incr(condition, seed, lang, baseline_bpb):
    return 100 * (load_bpb(condition, seed, lang) / baseline_bpb - 1)


def missing_paths():
    missing = []
    for condition in [PROPOSED_CONDITION, BALANCED_CONDITION] + SINGLE_LANG_CONDITIONS:
        for seed in SEEDS:
            p = eval_path(condition, seed)
            if not p.exists():
                missing.append(str(p))
    if not (RESULTS_ROOT / "baseline" / "eval_ppl.json").exists():
        missing.append(str(RESULTS_ROOT / "baseline" / "eval_ppl.json"))
    return missing


def worst_and_mean(condition, seed, baseline_bpb):
    incrs = {lang: bpb_incr(condition, seed, lang, baseline_bpb[lang]) for lang in LANGS}
    worst_lang = max(incrs, key=incrs.get)
    return incrs, incrs[worst_lang], worst_lang, sum(incrs.values()) / len(incrs)


def main():
    missing = missing_paths()
    if missing:
        print("=== NOT READY: missing eval_ppl.json files ===")
        for m in missing:
            print(f"  {m}")
        print(f"\n{len(missing)} file(s) missing -- finish disagreement_targeted's phase1_run_freq_and_scale.py -> "
              f"phase1_fisher.py -> phase1_merge_eval.py for seeds {SEEDS} first (phase1_6_targeted_budget.py must "
              f"have already written its budget_allocation.json for each seed).")
        return

    baseline_bpb = {lang: load_baseline_bpb(lang) for lang in LANGS}

    print("=== §6 Table 3: worst-language / mean bpb increase %% over baseline ===\n")
    rows = {}
    for condition in [PROPOSED_CONDITION, BALANCED_CONDITION] + SINGLE_LANG_CONDITIONS:
        per_seed = [worst_and_mean(condition, seed, baseline_bpb) for seed in SEEDS]
        worst_vals = [w for (_, w, _, _) in per_seed]
        mean_vals = [m for (_, _, _, m) in per_seed]
        worst_langs = [wl for (_, _, wl, _) in per_seed]
        rows[condition] = {
            "per_seed_worst": worst_vals, "per_seed_worst_lang": worst_langs, "per_seed_mean": mean_vals,
            "mean_of_worst": sum(worst_vals) / len(worst_vals), "mean_of_mean": sum(mean_vals) / len(mean_vals),
        }
        print(f"{CONDITION_LABEL[condition]:22s} worst(mean over seeds)={rows[condition]['mean_of_worst']:7.3f}  "
              f"mean(mean over seeds)={rows[condition]['mean_of_mean']:7.3f}  "
              f"per-seed worst={[f'{v:.3f}' for v in worst_vals]} (lang: {worst_langs})")

    # "최선 단일 언어": the ONE single-language condition with the lowest
    # mean-across-seeds worst-language degradation -- fixed once here, not
    # re-picked per seed (that would be cherry-picking after the fact).
    best_single = min(SINGLE_LANG_CONDITIONS, key=lambda c: rows[c]["mean_of_worst"])
    print(f"\n최선 단일 언어 (lowest mean worst-language degradation): {CONDITION_LABEL[best_single]} "
          f"(mean_of_worst={rows[best_single]['mean_of_worst']:.3f})")

    # Pre-registered gate: PROPOSED strictly beats BALANCED on worst-language
    # degradation in ALL 3 seeds. This is the ONLY criterion 06_논문_구성.md
    # §6 registers -- beating best-single-language is reported for context,
    # not gated on.
    per_seed_wins = [rows[PROPOSED_CONDITION]["per_seed_worst"][i] < rows[BALANCED_CONDITION]["per_seed_worst"][i]
                      for i in range(len(SEEDS))]
    verdict = "SUPPORTED" if all(per_seed_wins) else "NOT_SUPPORTED"

    print(f"\n=== Pre-registered verdict: does {CONDITION_LABEL[PROPOSED_CONDITION]} beat "
          f"{CONDITION_LABEL[BALANCED_CONDITION]} on worst-language bpb increase in ALL {len(SEEDS)} seeds? ===")
    for seed, win, p, b in zip(SEEDS, per_seed_wins,
                                rows[PROPOSED_CONDITION]["per_seed_worst"], rows[BALANCED_CONDITION]["per_seed_worst"]):
        print(f"  seed {seed}: proposed={p:.3f} vs balanced={b:.3f} -> {'WIN' if win else 'LOSE'}")
    print(f"\n-> {verdict}"
          + ("" if verdict == "SUPPORTED" else
             " (per 06_논문_구성.md §6: 'Balanced를 이기지 못하면 방법의 기여를 주장하지 않는다' -- "
             "does not clear the pre-registered bar)"))

    also_beats_best_single = rows[PROPOSED_CONDITION]["mean_of_worst"] < rows[best_single]["mean_of_worst"]
    print(f"\n(context, not gating) proposed also beats best-single-language "
          f"({CONDITION_LABEL[best_single]}) on mean worst-language degradation: {also_beats_best_single}")

    out = {
        "rows": rows, "best_single_lang_condition": best_single,
        "per_seed_wins_vs_balanced": per_seed_wins, "verdict": verdict,
        "also_beats_best_single_language": also_beats_best_single,
        "proposed_condition": PROPOSED_CONDITION, "balanced_condition": BALANCED_CONDITION,
        "seeds": SEEDS, "langs": LANGS,
    }
    out_path = RESULTS_ROOT / "phase1_6_budget_gate_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
