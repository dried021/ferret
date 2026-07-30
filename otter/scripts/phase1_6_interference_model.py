"""Track B of 01_plans/06_불일치표적배분_해결계획_0729.md §3 -- the "본명"
(primary) replacement for 06_논문_구성.md §6's proposed method, built directly
in response to 01_plans/06_불일치표적배분_실측결과.md §4-3's decisive
counter-evidence: cap50 gave Bengali MORE budget than Balanced (35%%>20%%)
and Bengali still got WORSE, because Swahili's share also rose in the same
step (20%%->50%%). Track A (phase1_6_vulnerability_budget.py) only fixes the
SIGNAL (vulnerability instead of coverage); this script additionally models
the INTERFERENCE those numbers point to -- languages' calibration shares do
not affect only their own compression quality, they affect everyone else's
too, and (§0-d of the plan) Swahili-heavy allocations look like the main
offender across all 4 already-run conditions:

    Swahili share   0%      20%     50%     73%
    Bengali degr.   ~17.7   22.03   22.81   31.38   (monotonic in SW share,
                                                       not in Bengali's OWN share)

Model (01_plans/..._해결계획_0729.md B1): D_l(s) ~= sum_k a_lk * s_k, where
D_l is language l's bpb increase %% and s is the calibration budget SHARE
vector (sums to 1). NOTE: this drops the doc's separate intercept b_l --
because sum_k s_k == 1 identically for every condition, b_l + sum_k a_lk*s_k
is EXACTLY reproducible with no intercept term at all (absorb b_l/5 into
each a_lk) -- so dropping it removes the "절편과 계수 하나가 비식별" problem
the doc flags as needing a sum-to-zero constraint, instead of imposing one:
with no intercept, the 5 single-language conditions (s = one-hot) alone give
a full-rank 5x5 system, so a_lk is already identified as "language l's
degradation when calibration is 100%% language k" (literally the observed
single-language grid cell in the noiseless limit) -- adding mixed_5lang/
disagreement_targeted/disagreement_targeted_cap50's rows (each a convex
combination of the same 5 one-hot rows, hence no new rank, only new
noisy-but-informative equations for the same 5*5 unknowns per language) just
gives ridge regression 8 conditions x 3 seeds = 24 equations to fit 5
unknowns per language instead of exactly 5 -- overdetermined, which is what
lets leave-one-condition-out validation (below) mean anything.

Fit data: the SAME 8 conditions 06_불일치표적배분_실측결과.md already ran to
merge+eval (english/korean/chinese/swahili/bengali_only, mixed_5lang,
disagreement_targeted, disagreement_targeted_cap50) -- zero new GPU cost to
FIT the model, only the one condition it recommends (interference_minimax)
needs a new merge+eval run, exactly like Track A.

Validation (B4, two-layer safety net):
  1. Primary: interference_minimax vs mixed_5lang on worst-language bpb
     degradation, same pre-registered gate as every other §6 candidate
     (phase1_6_budget_gate_v2.py).
  2. Secondary (model's own predictive power, independent of #1 winning):
     leave-one-condition-out (LOO) -- refit excluding one condition's 3
     seeds, predict its D_l from its known s vector, compare to the actual
     mean. Reported per 01_plans/..._해결계획_0729.md B4/Step 3: if the
     primary gate fails but LOO shows real predictive power, §6 can still
     report "interference structure discovered + validated predictive
     model" as a positive contribution (the plan's explicit fallback
     framing), rather than a bare negative result.

a_{.,swh_Latn} sign check (B1's headline hypothesis): reports how many of
the 4 OTHER languages' a_{l,swh_Latn} coefficients are positive (Swahili
calibration hurts them) -- if broadly positive, that is the paper's
"language calibration interferes with other languages' statistics" claim in
numbers, not just the 4-point eyeball pattern in the plan doc.

Track C invocation (01_plans/..._해결계획_0729.md §4, whitening-only
targeting -- needs NO new code): phase1_merge_eval.py loads Fisher/expert_freq
from `--condition`'s own results dir but SVD-scale from `--scale-condition`'s
(see its `svd_scale = ...` block, RESULTS_ROOT/<scale_condition>/seed<seed>/
svd_scale_processed.pt) -- exactly the split Track C wants. Only the SCALE
half needs a new run (mixed_5lang's own Fisher/expert_freq already exist from
its own prior merge+eval):
    conda run -n d2moe_env python phase1_svd_scale.py \\
        --condition interference_minimax --seed <seed>
    conda run -n d2moe_env python phase1_merge_eval.py \\
        --condition mixed_5lang --seed <seed> \\
        --scale-condition interference_minimax --scale-seed <seed>
This is phase1_merge_eval.py's existing 2x2 --scale-condition mechanism (the
same one run_phase1_41_diagonal.py already exercises) -- Fisher comes from
Balanced, whitening/SVD-scale comes from Track B's minimax allocation,
D2-MoE's algorithm itself is untouched. Output lands at
RESULTS_ROOT/mixed_5lang/seed<seed>/scale_interference_minimax_seed<seed>/
eval_ppl.json -- phase1_6_budget_gate_v2.py's Track C candidates read exactly
that path (condition=mixed_5lang, scale_condition=interference_minimax).

Usage:
    conda run -n d2moe_env python phase1_6_interference_model.py --seed 0
        [--ridge-lambda 0.5] [--floor-share 0.05] [--max-share 0.40]
        [--random-control] [--out-condition NAME]

    (No GPU needed for the fit/LP -- pure post-processing of already-written
    eval_ppl.json + budget_allocation.json files; requires ALL 8 conditions'
    eval_ppl.json for seeds 0-2, i.e. 06_불일치표적배분_실측결과.md's runs
    plus Plan §1's P1/P2 prerequisites.)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_calib_data  # noqa: E402
from phase1_6_vulnerability_budget import random_control_weights  # noqa: E402

RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")
SEEDS = [0, 1, 2]
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
MATCHED_CONDITION = {
    "eng_Latn": "english_only", "kor_Hang": "korean_only", "zho_Hans": "chinese_only",
    "swh_Latn": "swahili_only", "ben_Beng": "bengali_only",
}
SINGLE_LANG_CONDITIONS = list(MATCHED_CONDITION.values())
BALANCED_CONDITION = "mixed_5lang"
# The two already-run §6 candidates whose allocation vectors we also fit on
# (see module docstring: "overdetermined, not needed for identifiability,
# but improves the fit / gives LOO something informative to hold out").
DISAGREEMENT_CONDITIONS = [phase1_calib_data.TARGETED_CONDITION, phase1_calib_data.TARGETED_CONDITION_CAP50]
FIT_CONDITIONS = SINGLE_LANG_CONDITIONS + [BALANCED_CONDITION] + DISAGREEMENT_CONDITIONS


def eval_path(condition, seed):
    return RESULTS_ROOT / condition / f"seed{seed}" / "eval_ppl.json"


def load_bpb(condition, seed, lang):
    return json.loads(eval_path(condition, seed).read_text())[lang]["bits_per_byte"]


def load_baseline_bpb(lang):
    return json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())[lang]["bits_per_byte"]


def bpb_incr(condition, seed, lang, baseline_bpb):
    return 100 * (load_bpb(condition, seed, lang) / baseline_bpb - 1)


def allocation_vector(condition):
    """s vector ({lang: share}) for `condition`, in the same LANGS order --
    one-hot for single-language conditions, uniform for mixed_5lang (its
    build_condition_sentences() path uses build_weighted_sentences() with
    equal weight per language -- see phase1_calib_data.py's
    CHAR_BALANCED_MIXES), and the saved Stage-2 allocation for the
    disagreement-targeted conditions."""
    if condition in SINGLE_LANG_CONDITIONS:
        lang = phase1_calib_data.LANG_OF_SIMPLE[condition]
        return {l: (1.0 if l == lang else 0.0) for l in LANGS}
    if condition == BALANCED_CONDITION:
        return {l: 1.0 / len(LANGS) for l in LANGS}
    if condition in DISAGREEMENT_CONDITIONS:
        alloc_path = phase1_calib_data.budget_allocation_path(0, condition)
        if not alloc_path.exists():
            raise FileNotFoundError(f"{alloc_path} missing -- {condition} must already have a Stage-2 allocation "
                                     f"(it was carried through a real merge+eval per 01_plans/06_불일치표적배분_실측결과.md)")
        weights = json.loads(alloc_path.read_text())["weights"]
        total = sum(weights.values())
        return {l: weights.get(l, 0.0) / total for l in LANGS}
    raise ValueError(f"no allocation_vector rule for condition {condition!r}")


def missing_paths():
    missing = []
    for condition in FIT_CONDITIONS:
        for seed in SEEDS:
            p = eval_path(condition, seed)
            if not p.exists():
                missing.append(str(p))
    for condition in DISAGREEMENT_CONDITIONS:
        p = phase1_calib_data.budget_allocation_path(0, condition)
        if not p.exists():
            missing.append(str(p))
    if not (RESULTS_ROOT / "baseline" / "eval_ppl.json").exists():
        missing.append(str(RESULTS_ROOT / "baseline" / "eval_ppl.json"))
    return missing


def build_dataset():
    """Returns (S: (n_obs,5) design matrix, D: {lang: (n_obs,) array},
    obs_conditions: [condition]*n_obs) across all FIT_CONDITIONS x SEEDS."""
    baseline_bpb = {lang: load_baseline_bpb(lang) for lang in LANGS}
    s_by_cond = {c: allocation_vector(c) for c in FIT_CONDITIONS}

    rows, obs_conditions = [], []
    D = {lang: [] for lang in LANGS}
    for condition in FIT_CONDITIONS:
        s = np.array([s_by_cond[condition][l] for l in LANGS])
        for seed in SEEDS:
            rows.append(s)
            obs_conditions.append(condition)
            for lang in LANGS:
                D[lang].append(bpb_incr(condition, seed, lang, baseline_bpb[lang]))
    S = np.stack(rows, axis=0)
    D = {lang: np.array(vals) for lang, vals in D.items()}
    return S, D, obs_conditions, s_by_cond


def fit_ridge_no_intercept(S, y, ridge_lambda):
    """a = argmin_a ||S a - y||^2 + ridge_lambda ||a||^2, closed form. No
    intercept term -- see module docstring for why that's the identifiability
    fix used here instead of an explicit sum-to-zero constraint."""
    n_feat = S.shape[1]
    return np.linalg.solve(S.T @ S + ridge_lambda * np.eye(n_feat), S.T @ y)


def fit_all_languages(S, D, ridge_lambda):
    return {lang: fit_ridge_no_intercept(S, D[lang], ridge_lambda) for lang in LANGS}


def leave_one_condition_out(S, D, obs_conditions, ridge_lambda):
    """For each FIT_CONDITIONS entry, refit on the other 7 conditions' rows
    and predict this condition's mean D_l from its (held-out, but KNOWN --
    the s vector isn't secret, only its outcome is) allocation vector.
    Returns {condition: {lang: {"predicted": float, "actual": float}}} plus
    overall RMSE/correlation across all (condition, lang) cells."""
    obs_conditions = np.array(obs_conditions)
    results = {}
    preds, actuals = [], []
    for held_out in FIT_CONDITIONS:
        mask = obs_conditions != held_out
        S_train = S[mask]
        held_mask = obs_conditions == held_out
        s_held = S[held_mask][0]  # identical across that condition's 3 seed-rows
        results[held_out] = {}
        for lang in LANGS:
            a = fit_ridge_no_intercept(S_train, D[lang][mask], ridge_lambda)
            predicted = float(s_held @ a)
            actual = float(D[lang][held_mask].mean())
            results[held_out][lang] = {"predicted": predicted, "actual": actual}
            preds.append(predicted)
            actuals.append(actual)
    preds, actuals = np.array(preds), np.array(actuals)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    corr = float(np.corrcoef(preds, actuals)[0, 1]) if len(preds) > 1 else float("nan")
    return results, rmse, corr


def solve_minimax(a_matrix, floor_share, max_share):
    """min_s max_l (a_l . s), s in the capped simplex {sum(s)=1,
    floor<=s_k<=cap}. Standard LP epigraph trick: variables x=[s(5), t],
    minimize t subject to a_l . s - t <= 0 for every l.

    a_matrix: (5 langs, 5 langs) array, row l = a_lk fitted above.
    Returns (s*: {lang: float}, t*: float, predicted_D: {lang: float})."""
    n = len(LANGS)
    c = np.concatenate([np.zeros(n), [1.0]])
    A_ub = np.hstack([a_matrix, -np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.concatenate([np.ones(n), [0.0]]).reshape(1, -1)
    b_eq = np.array([1.0])
    bounds = [(floor_share, max_share)] * n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"minimax LP failed to solve: {res.message}")
    s_star = {lang: float(res.x[i]) for i, lang in enumerate(LANGS)}
    t_star = float(res.x[-1])
    predicted_D = {lang: float(a_matrix[i] @ res.x[:n]) for i, lang in enumerate(LANGS)}
    return s_star, t_star, predicted_D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="which seed subdir to write budget_allocation.json under "
                                                              "(weights don't depend on this seed, see phase1_6_vulnerability_budget.py's convention)")
    parser.add_argument("--ridge-lambda", type=float, default=0.5)
    parser.add_argument("--floor-share", type=float, default=0.05)
    parser.add_argument("--max-share", type=float, default=0.40,
                         help="also bounds the LP -- without it, minimax over a LINEAR model fit mostly from "
                              "convex-hull-interior points can push s* to a simplex corner the model never saw, "
                              "an extrapolation risk on top of the concentration risk Track A's cap also guards against")
    parser.add_argument("--random-control", action="store_true",
                         help="R1: write a same-entropy random permutation of s* instead (forces --out-condition "
                              "to the *_random condition)")
    parser.add_argument("--out-condition", default=None)
    args = parser.parse_args()

    missing = missing_paths()
    if missing:
        print("=== NOT READY: missing files ===")
        for m in missing:
            print(f"  {m}")
        print(f"\n{len(missing)} file(s) missing -- this script needs all of {FIT_CONDITIONS} eval_ppl.json "
              f"for seeds {SEEDS}, plus the disagreement-targeted conditions' budget_allocation.json.")
        return

    S, D, obs_conditions, s_by_cond = build_dataset()
    a_by_lang = fit_all_languages(S, D, args.ridge_lambda)
    a_matrix = np.stack([a_by_lang[lang] for lang in LANGS], axis=0)  # rows=l, cols=k, both in LANGS order

    print("=== Track B: fitted interference matrix a_lk (row=affected language l, col=source language k) ===\n")
    header = "l \\ k".ljust(10) + "".join(l.rjust(11) for l in LANGS)
    print(header)
    for i, lang in enumerate(LANGS):
        print(lang.ljust(10) + "".join(f"{a_matrix[i, j]:11.3f}" for j in range(len(LANGS))))

    sw_idx = LANGS.index("swh_Latn")
    sw_col = [a_matrix[i, sw_idx] for i in range(len(LANGS)) if LANGS[i] != "swh_Latn"]
    n_positive = sum(1 for v in sw_col if v > 0)
    print(f"\n[interference] a_{{.,swh_Latn}} (excluding Swahili's own row): {[f'{v:.3f}' for v in sw_col]} "
          f"-> {n_positive}/{len(sw_col)} positive (Swahili calibration raises OTHER languages' degradation)")

    loo_results, loo_rmse, loo_corr = leave_one_condition_out(S, D, obs_conditions, args.ridge_lambda)
    print(f"\n[interference] leave-one-condition-out validation: RMSE={loo_rmse:.3f} (bpb %%pts), "
          f"corr(predicted, actual)={loo_corr:+.3f} across {len(FIT_CONDITIONS) * len(LANGS)} (condition, lang) cells")
    for condition in FIT_CONDITIONS:
        for lang in LANGS:
            r = loo_results[condition][lang]
            print(f"  {condition:28s} {lang}: predicted={r['predicted']:7.3f} actual={r['actual']:7.3f} "
                  f"diff={r['predicted'] - r['actual']:+.3f}")

    s_star, t_star, predicted_D = solve_minimax(a_matrix, args.floor_share, args.max_share)
    print(f"\n[interference] minimax allocation s* (floor={args.floor_share}, cap={args.max_share}), "
          f"predicted worst-language degradation t*={t_star:.3f}:")
    baseline_share = 1.0 / len(LANGS)
    for lang, w in sorted(s_star.items(), key=lambda kv: -kv[1]):
        print(f"  {lang}: weight={w:.3f} ({'above' if w > baseline_share else 'below' if w < baseline_share else '='} "
              f"equal-share {baseline_share:.3f}) predicted_D={predicted_D[lang]:.3f}")

    out_condition = args.out_condition
    weights = s_star
    if args.random_control:
        weights = random_control_weights(s_star, args.seed)
        out_condition = out_condition or phase1_calib_data.TRACK_B_CONDITION_RANDOM
        print(f"\n[interference] --random-control: permuted s* VALUES across languages (seed={args.seed}):")
        for lang, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f"  {lang}: weight={w:.3f}")
    elif out_condition is None:
        out_condition = phase1_calib_data.TRACK_B_CONDITION

    out_dir = RESULTS_ROOT / out_condition / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "budget_allocation.json"
    payload = {
        "weights": weights,
        "random_control": args.random_control,
        "a_matrix": {LANGS[i]: {LANGS[j]: float(a_matrix[i, j]) for j in range(len(LANGS))} for i in range(len(LANGS))},
        "t_star": t_star, "predicted_D": predicted_D,
        "loo_rmse": loo_rmse, "loo_corr": loo_corr, "loo_results": loo_results,
        "ridge_lambda": args.ridge_lambda, "floor_share": args.floor_share, "max_share": args.max_share,
        "fit_conditions": FIT_CONDITIONS, "seed": args.seed, "out_condition": out_condition,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[interference] wrote {out_path}")
    print(f"[interference] also wrote model diagnostics (a_matrix/LOO) into the same file for "
          f"phase1_6_budget_gate_v2.py's secondary 'prediction validation' fallback check.")
    print(f"[interference] next: conda run -n d2moe_env python phase1_run_freq_and_scale.py "
          f"--condition {out_condition} --seed {args.seed}, then phase1_fisher.py, then phase1_merge_eval.py "
          f"-- same as any other condition. For Track C (whitening-only), see module docstring instead.")


if __name__ == "__main__":
    main()
