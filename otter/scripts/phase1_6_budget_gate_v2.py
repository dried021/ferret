"""§6 comparison gate v2 -- pre-registered per 01_plans/06_불일치표적배분_
해결계획_0729.md §1 (P3), written BEFORE any of the new conditions' merge+eval
runs finish, same discipline as phase1_6_budget_gate.py (v1) and
phase1_41_headline_gate.py/phase1_placebo_gate.py.

Does NOT replace or reinterpret phase1_6_budget_gate.py -- that gate's
NOT_SUPPORTED verdict for disagreement_targeted/disagreement_targeted_cap50
(01_plans/06_불일치표적배분_실측결과.md) stays exactly as reported (R2 in the
plan doc: "기존 negative result는 삭제하지 않고 본문에 유지... 사전등록
게이트를 정직하게 보고한 실패가 성공안의 설득력을 오히려 올린다"). This
script ADDS the replacement candidates the failure analysis produced:

  CANDIDATES (label -> (condition, scale_condition)):
    Track A (gap)     -> (vulnerability_targeted, None)
    Track A (abs)     -> (vulnerability_targeted_abs, None)          [sensitivity, signal (i)]
    Track A (random)  -> (vulnerability_targeted_random, None)       [R1 control, NOT gated]
    Track B (minimax) -> (interference_minimax, None)
    Track B (random)  -> (interference_minimax_random, None)         [R1 control, NOT gated]
    Track C (A)       -> (mixed_5lang, vulnerability_targeted)       [whitening-only, Track A alloc]
    Track C (B)       -> (mixed_5lang, interference_minimax)         [whitening-only, Track B alloc]

Pre-registered PRIMARY criterion (identical to v1, per the plan's explicit
instruction "판정: 3 seed 모두 Balanced의 worst보다 낮아야 SUPPORTED --
기존과 동일 기준 유지 -- 기준을 완화해서 통과시키는 모양새는 절대 금지"):
    SUPPORTED     if a candidate's worst-language bpb increase is STRICTLY
                  LOWER than mixed_5lang's, in ALL 3 seeds.
    NOT_SUPPORTED otherwise.
Only the 5 non-control candidates (Track A gap/abs, Track B, Track C x2) are
eligible to flip the overall §6 verdict to SUPPORTED; the two *_random
controls are reported for context only (R1's purpose is to show that an
arbitrary same-entropy reallocation does NOT clear the bar, not to itself
clear it).

SECONDARY fallback (01_plans/..._해결계획_0729.md Step 3, "모델의 예측력"
branch): if every candidate above is NOT_SUPPORTED, this script also reports
Track B's leave-one-condition-out diagnostics (written into
interference_minimax's budget_allocation.json by phase1_6_interference_model.py)
so §6 can still report "간섭 구조 발견 + 예측 모델" as a positive
contribution per the plan's explicit fallback framing, rather than a bare
negative result -- this is DIAGNOSTIC ONLY and never flips overall_verdict.

Usage: python phase1_6_budget_gate_v2.py   (run after whichever candidates'
    phase1_merge_eval.py runs have finished for seeds 0,1,2 -- partial runs
    are fine, this script reports NOT_READY per-candidate, not all-or-nothing)
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]

BALANCED_CONDITION = "mixed_5lang"
SINGLE_LANG_CONDITIONS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]

# (label, condition, scale_condition_or_None, gates_overall_verdict)
CANDIDATES = [
    ("Track A (gap)", "vulnerability_targeted", None, True),
    ("Track A (abs)", "vulnerability_targeted_abs", None, True),
    ("Track A (random control)", "vulnerability_targeted_random", None, False),
    ("Track B (minimax)", "interference_minimax", None, True),
    ("Track B (random control)", "interference_minimax_random", None, False),
    ("Track C (whitening=A)", "mixed_5lang", "vulnerability_targeted", True),
    ("Track C (whitening=B)", "mixed_5lang", "interference_minimax", True),
]
CONDITION_LABEL = {
    "mixed_5lang": "Balanced", "english_only": "English", "korean_only": "Korean",
    "chinese_only": "Chinese", "swahili_only": "Swahili", "bengali_only": "Bengali",
}


def eval_path(condition, seed, scale_condition=None):
    if scale_condition is None:
        return RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"
    return RESULTS_ROOT / condition / f"seed{seed}" / f"scale_{scale_condition}_seed{seed}" / "eval_ppl.json"


def load_bpb(condition, seed, lang, scale_condition=None):
    return json.loads(eval_path(condition, seed, scale_condition).read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    return json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())[lang]["bits_per_byte"]


def bpb_incr(condition, seed, lang, baseline_bpb, scale_condition=None):
    return 100 * (load_bpb(condition, seed, lang, scale_condition) / baseline_bpb - 1)


def worst_and_mean(condition, seed, baseline_bpb, scale_condition=None):
    incrs = {lang: bpb_incr(condition, seed, lang, baseline_bpb[lang], scale_condition) for lang in LANGS}
    worst_lang = max(incrs, key=incrs.get)
    return incrs, incrs[worst_lang], worst_lang, sum(incrs.values()) / len(incrs)


def candidate_ready(condition, seed, scale_condition=None):
    return eval_path(condition, seed, scale_condition).exists()


def summarize(condition, seed_list, baseline_bpb, scale_condition=None):
    per_seed = [worst_and_mean(condition, seed, baseline_bpb, scale_condition) for seed in seed_list]
    worst_vals = [w for (_, w, _, _) in per_seed]
    mean_vals = [m for (_, _, _, m) in per_seed]
    worst_langs = [wl for (_, _, wl, _) in per_seed]
    return {
        "per_seed_worst": worst_vals, "per_seed_worst_lang": worst_langs, "per_seed_mean": mean_vals,
        "mean_of_worst": sum(worst_vals) / len(worst_vals), "mean_of_mean": sum(mean_vals) / len(mean_vals),
    }


def main():
    if not (RESULTS_ROOT / "baseline" / "eval_ppl.json").exists():
        print("=== NOT READY: baseline/eval_ppl.json missing ===")
        return
    baseline_bpb = {lang: load_baseline_bpb(lang) for lang in LANGS}

    balanced_ready = all(candidate_ready(BALANCED_CONDITION, s) for s in SEEDS)
    if not balanced_ready:
        print(f"=== NOT READY: {BALANCED_CONDITION} (Balanced) missing for some seed in {SEEDS} ===")
        return
    balanced_row = summarize(BALANCED_CONDITION, SEEDS, baseline_bpb)

    print("=== §6 Table 3v2: worst-language / mean bpb increase % over baseline ===\n")
    print(f"{'Balanced':32s} worst(mean over seeds)={balanced_row['mean_of_worst']:7.3f}  "
          f"mean(mean over seeds)={balanced_row['mean_of_mean']:7.3f}  "
          f"per-seed worst={[f'{v:.3f}' for v in balanced_row['per_seed_worst']]} "
          f"(lang: {balanced_row['per_seed_worst_lang']})")

    single_rows = {}
    for c in SINGLE_LANG_CONDITIONS:
        if all(candidate_ready(c, s) for s in SEEDS):
            single_rows[c] = summarize(c, SEEDS, baseline_bpb)
    best_single = min(single_rows, key=lambda c: single_rows[c]["mean_of_worst"]) if single_rows else None
    if best_single:
        print(f"\n최선 단일 언어 (context): {CONDITION_LABEL[best_single]} "
              f"(mean_of_worst={single_rows[best_single]['mean_of_worst']:.3f})")

    print("\n=== candidates ===\n")
    results = {}
    any_supported = False
    for label, condition, scale_condition, gates in CANDIDATES:
        ready_seeds = [s for s in SEEDS if candidate_ready(condition, s, scale_condition)]
        if len(ready_seeds) < len(SEEDS):
            missing = len(SEEDS) - len(ready_seeds)
            print(f"{label:32s} NOT READY ({missing}/{len(SEEDS)} seed(s) missing"
                  + (f", scale_condition={scale_condition}" if scale_condition else "") + ")")
            results[label] = {"ready": False, "condition": condition, "scale_condition": scale_condition}
            continue

        row = summarize(condition, SEEDS, baseline_bpb, scale_condition)
        per_seed_wins = [row["per_seed_worst"][i] < balanced_row["per_seed_worst"][i] for i in range(len(SEEDS))]
        verdict = "SUPPORTED" if all(per_seed_wins) else "NOT_SUPPORTED"
        if gates and verdict == "SUPPORTED":
            any_supported = True

        print(f"{label:32s} worst(mean)={row['mean_of_worst']:7.3f}  mean(mean)={row['mean_of_mean']:7.3f}  "
              f"per-seed worst={[f'{v:.3f}' for v in row['per_seed_worst']]} (lang: {row['per_seed_worst_lang']})  "
              f"vs Balanced: {['WIN' if w else 'LOSE' for w in per_seed_wins]} -> {verdict}"
              + ("" if gates else " (control, not gating)"))

        results[label] = {
            "ready": True, "condition": condition, "scale_condition": scale_condition, "gates_overall": gates,
            "row": row, "per_seed_wins_vs_balanced": per_seed_wins, "verdict": verdict,
        }

    overall_verdict = "SUPPORTED" if any_supported else "NOT_SUPPORTED"
    print(f"\n=== overall §6 verdict (ANY gating candidate beats Balanced in all {len(SEEDS)} seeds): "
          f"{overall_verdict} ===")

    # Secondary fallback (Step 3 middle branch): report Track B's LOO fit
    # quality regardless of overall_verdict, so a NOT_SUPPORTED run can still
    # cite "interference structure discovered + validated predictive model"
    # if LOO shows real predictive power (01_plans/..._해결계획_0729.md B4).
    interference_alloc_paths = [RESULTS_ROOT / "interference_minimax" / f"seed{s}" / "budget_allocation.json"
                                 for s in SEEDS]
    interference_alloc_path = next((p for p in interference_alloc_paths if p.exists()), None)
    if interference_alloc_path is not None:
        diag = json.loads(interference_alloc_path.read_text())
        loo_rmse, loo_corr = diag.get("loo_rmse"), diag.get("loo_corr")
        print(f"\n(secondary, non-gating) Track B interference-model LOO fit: RMSE={loo_rmse:.3f} bpb %pts, "
              f"corr={loo_corr:+.3f} -- {'model has real predictive power' if (loo_corr is not None and loo_corr > 0.5) else 'weak/no demonstrated predictive power'} "
              f"per 01_plans/06_불일치표적배분_해결계획_0729.md Step 3's fallback framing "
              f"('간섭 구조 발견 + 예측 모델' is only warranted if this number is good).")
    else:
        print(f"\n(secondary, non-gating) Track B interference-model LOO diagnostics not found -- "
              f"run phase1_6_interference_model.py first to populate the fallback framing check.")

    out = {
        "balanced_row": balanced_row, "best_single_lang_condition": best_single,
        "single_rows": single_rows, "candidates": results, "overall_verdict": overall_verdict,
        "seeds": SEEDS, "langs": LANGS,
    }
    out_path = RESULTS_ROOT / "phase1_6_budget_gate_v2_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
