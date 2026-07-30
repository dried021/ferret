"""Shared calibration-text builder for Phase 1's conditions (CONDITIONS below
-- 5 single-language + 4 placebo-B arms + balanced/mixed_5lang, plus
"disagreement_targeted", the §6 proposed method's Stage-2 condition, see
build_weighted_sentences()). Produces a HF `datasets.Dataset` with a "text"
column, in the exact shape D2MoE's preprocess/get_expert_freq.py and
preprocess/get_scale.py expect from `load_dataset('wikitext', ...)` -- so
those scripts' own (already-correct, already-tested) sampling/hook logic can
run completely unmodified against OUR multilingual text via a load_dataset
monkeypatch (see phase1_run_freq_and_scale.py), instead of reimplementing it.
"""
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import data_utils  # noqa: E402

SOURCE, SPLIT, COL_PREFIX = "israel/flores-parallel", "test", "sentence_"
LANG_OF_SIMPLE = {
    "english_only": "eng_Latn", "korean_only": "kor_Hang", "chinese_only": "zho_Hans",
    "english_only_b": "eng_Latn", "korean_only_b": "kor_Hang",
    "swahili_only": "swh_Latn", "swahili_only_b": "swh_Latn",
    "bengali_only": "ben_Beng", "bengali_only_b": "ben_Beng",
}
# "disagreement_targeted"* (2026-07-27, 06_논문_구성.md §6 proposed method):
# TWO variants, both alongside the 5 single-language + balanced/mixed_5lang
# ones above -- see TARGETED_CONDITIONS below for what makes them different.
# "disagreement_targeted" = uncapped (pure proportional-to-deficiency, a
# single language CAN dominate the budget). "disagreement_targeted_cap50"
# (2026-07-27 code review point 2) = the SAME deficiency signal but capped
# at max_share=0.5 per language via phase1_6_targeted_budget.py's
# project_to_capped_simplex() -- carried through the real pipeline as its
# OWN condition (not a post-hoc reweighting) since the cap changes the
# actual calibration TEXT build_weighted_sentences() produces, not just a
# label. Both are run to Fisher/merge/eval per user decision: the cap
# version showed equal-or-better disagreement-expert coverage in the
# Stage-1/2 sanity check with far less concentration risk, so it isn't
# assumed to be strictly worse and dropped -- only real eval settles it.
#
# BOTH variants above went to a real merge+eval and came back NOT_SUPPORTED
# (01_plans/06_불일치표적배분_실측결과.md, 2026-07-28): the disagreement/
# routing-coverage SIGNAL is not a vulnerability signal (Bengali, the actual
# worst language, scored deficiency=5/810 -- "no problem"), and cap50 proved
# the failure isn't merely "too little Bengali budget" either (cap50 raised
# Bengali's share 20%->35% and Bengali still got WORSE, because Swahili's
# share also rose 20%->50% in the same step -- languages interfere with each
# other's calibration statistics, not just compete for a coverage budget).
# 01_plans/06_불일치표적배분_해결계획_0729.md registers three replacement
# signals/allocators that reuse this exact plumbing (only the WEIGHTS
# computation changes -- build_weighted_sentences()/build_condition_sentences
# below are untouched) -- see phase1_6_vulnerability_budget.py (Track A) and
# phase1_6_interference_model.py (Track B). Track C (whitening-only
# targeting) needs no new condition name at all: it is phase1_merge_eval.py's
# existing --scale-condition mechanism applied to "mixed_5lang" (Fisher) +
# one of the conditions below (scale) -- see phase1_6_interference_model.py's
# module docstring "Track C" section for the exact invocation.
TARGETED_CONDITION = "disagreement_targeted"
TARGETED_CONDITION_CAP50 = "disagreement_targeted_cap50"
# Track A (vulnerability/recoverable-gap signal, 2026-07-29): "_abs" uses
# signal (i) (raw Balanced-condition degradation); the unsuffixed condition
# uses signal (ii) (recoverable gap = Balanced degradation minus own-language
# degradation), the doc's theoretically-preferred default. "_random" is
# Track A's R1 same-entropy random-allocation control (a permutation of the
# primary condition's weight VALUES across languages, decoupling the
# signal-language pairing while holding the allocation's entropy fixed).
TRACK_A_CONDITION = "vulnerability_targeted"
TRACK_A_CONDITION_ABS = "vulnerability_targeted_abs"
TRACK_A_CONDITION_RANDOM = "vulnerability_targeted_random"
# Track B (interference-model minimax signal, 2026-07-29): fits
# D_l(s) ~= b_l + sum_k a_lk * s_k from the 8 already-run conditions'
# eval_ppl.json (no new GPU runs to FIT the model) and solves for the
# language-share vector s* that minimizes the worst-language prediction --
# see phase1_6_interference_model.py. "_random" is Track B's R1 control,
# same construction as Track A's.
TRACK_B_CONDITION = "interference_minimax"
TRACK_B_CONDITION_RANDOM = "interference_minimax_random"
TARGETED_CONDITIONS = [
    TARGETED_CONDITION, TARGETED_CONDITION_CAP50,
    TRACK_A_CONDITION, TRACK_A_CONDITION_ABS, TRACK_A_CONDITION_RANDOM,
    TRACK_B_CONDITION, TRACK_B_CONDITION_RANDOM,
]
# §6 whitening-targeted robust mixing (2026-07-30 pilot, new method registered
# alongside Tracks A/B/C above -- NOT a replacement for phase1_6_targeted_
# budget.py's routing-disagreement signal or phase1_6_interference_model.py's
# fitted-interference minimax; this one targets whitening GEOMETRY directly).
# Unlike TARGETED_CONDITIONS (whose weights are per-condition, per-seed
# budget_allocation.json files from a Stage-1/Stage-2 pipeline), this
# condition's weight vector is a SINGLE fixed w* independent of seed --
# phase1_6_robust_mix_weights.py's STEP 1 solves w* = argmin_w max_l
# relFrob(Sigma(w), Sigma_l) once from seed0's already-computed svd_scale_
# processed.pt Cholesky factors (5 single-language conditions), and every
# seed's calibration text for this condition reuses that same w* (only the
# build_weighted_sentences() shuffle differs by seed, exactly like every
# other condition here) -- see robust_mix_weights_path() below.
ROBUST_MIX_CONDITION = "robust_mix"
CONDITIONS = ["english_only", "korean_only", "chinese_only", "balanced", "english_only_b", "korean_only_b",
              "swahili_only", "swahili_only_b", "bengali_only", "bengali_only_b", "mixed_5lang",
              ] + TARGETED_CONDITIONS + [ROBUST_MIX_CONDITION]
PLACEBO_CONDITIONS = ["english_only_b", "korean_only_b", "swahili_only_b", "bengali_only_b"]  # arm "B" -- see build_condition_sentences

# Same results tree phase1_fisher.py/phase1_merge_eval.py define independently
# under their own RESULTS_ROOT constant (intentional duplication -- matches
# how MODEL_PATH is already duplicated across phase1_*.py scripts rather than
# forcing every script to import a shared path module for one string).
RESULTS_ROOT = Path("/mnt/HDD/minjeong/d2moe_results/phase1")

# "balanced" (2026-07-24, EN/KO/ZH) is left untouched -- both its calibration
# text AND its equal-SENTENCE-COUNT mixing mechanism -- so its already-
# published results stay bit-for-bit reproducible. "mixed_5lang" (2026-07-27)
# is a SEPARATE condition for EN/KO/ZH/Swahili/Bengali, not a redefinition.
MIXED_LANGS = {
    "balanced": ("eng_Latn", "kor_Hang", "zho_Hans"),
    "mixed_5lang": ("eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"),
}
# 2026-07-27 code review fix: "balanced"'s equal-SENTENCE-COUNT mix is not
# actually an equal CHARACTER/token mix -- measured on the real pools, EN/KO/ZH
# split 54.7%/27.4%/17.9% by character volume despite 1/3 each by sentence
# count (zho_Hans packs far more content per character). mixed_5lang has never
# been run (unlike "balanced", no published result to preserve), so it goes
# through build_weighted_sentences() with equal per-language weight instead --
# a genuine equal-character-budget mix. See build_weighted_sentences()
# docstring for the mechanism.
CHAR_BALANCED_MIXES = {"mixed_5lang"}

# phase1_merge_eval.py's eval_flores_ppl() evaluates on the first
# EVAL_N_SENTENCES sentences (offset=0) of each language's FLORES pool.
# Dropping that same range here before shuffling for calibration keeps
# calibration and evaluation text disjoint (2026-07-24 fix -- see
# 00_docs/03_기술노트.md "Phase 1 pilot", the earlier pilot did not do this).
EVAL_N_SENTENCES = 60


def _pool_for(lang, arm):
    pool = data_utils.load_full_column(SOURCE, SPLIT, COL_PREFIX, lang)[EVAL_N_SENTENCES:]
    if arm == "B":
        half = len(pool) // 2
        return pool[half:]
    return pool


def budget_allocation_path(seed, condition=TARGETED_CONDITION):
    return RESULTS_ROOT / condition / f"seed{seed}" / "budget_allocation.json"


def robust_mix_weights_path():
    """Single, seed-independent w* file phase1_6_robust_mix_weights.py
    writes (otter/results/, not RESULTS_ROOT -- it's a CPU-only geometry
    artifact, not a per-condition/seed GPU pipeline output)."""
    return SCRIPT_DIR.parent / "results" / "robust_mix_weights.json"


def load_robust_mix_weights():
    """{lang_code: weight} from phase1_6_robust_mix_weights.py's GLOBAL
    solve (module docstring: the pooled-over-all-layers w*, the primary
    output STEP 2/3 use -- the per-layer solves are a robustness comparison,
    not a calibration-text input)."""
    path = robust_mix_weights_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run phase1_6_robust_mix_weights.py first (STEP 1 of the "
            f"whitening-targeted robust mixing pilot must complete before this condition's text can be built)."
        )
    return json.loads(path.read_text())["global"]["w_star"]


def build_weighted_sentences(weights, seed=0):
    """Stage 2 of 06_논문_구성.md §6's proposed method: unlike MIXED_LANGS's
    equal-share interleave (every language contributes its FULL pool, so
    each language's SHARE of the joined calibration text is whatever its
    natural FLORES pool size happens to be -- not a deliberate choice), this
    redistributes the total calibration budget across languages
    PROPORTIONALLY to `weights` (a {lang_code: weight>=0} dict, need not sum
    to 1).

    Why this makes calibration "targeted": phase1_fisher.py's build_samples()
    draws random constant-length windows over the joined sentence text, so a
    language's SHARE of total joined CHARACTER length is (approximately) its
    share of calibration tokens the real Fisher computation actually sees.
    Making that share proportional to how much each language's disagreement
    experts need (phase1_6_targeted_budget.py's allocate_budget()) -- rather
    than equal, like "balanced" -- is the entire mechanism: same total
    character budget as mixed_5lang, redistributed instead of split evenly.

    2026-07-27 fix (code review): this MUST allocate by character count, not
    sentence count -- per-sentence length varies hugely by language (measured
    on the real FLORES pools: ~42 chars/sentence for zho_Hans vs ~129 for
    eng_Latn/ben_Beng), so a sentence-count-based target_n silently gives
    each language a very different character/token share than its weight
    implies. An earlier version of this function computed target_n as a
    sentence count off total_size=sum(len(pool)) (list length, i.e. sentence
    count) -- that version could NOT actually deliver the weighted budget
    split its own docstring promised. Fixed by working in character space
    throughout: target_chars per language, pool cycled sentence-by-sentence
    (not by a precomputed rep count) until that character budget is met.

    Total character budget is fixed at the sum of all 5 single languages'
    natural (post-eval-exclusion) pool sizes -- the same scale mixed_5lang's
    total corpus uses -- so a same-budget comparison against mixed_5lang is
    apples-to-apples. A language given more weight than its natural pool's
    character volume holds has its pool CYCLED (repeated from the start) to
    reach its target share; see the loop below."""
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError(f"weights must sum to > 0, got {weights}")
    all_langs = set(LANG_OF_SIMPLE.values())
    pools = {lang: _pool_for(lang, "A") for lang in all_langs}
    total_chars = sum(sum(len(s) for s in p) for p in pools.values())

    sents = []
    for lang, w in weights.items():
        if w <= 0:
            continue
        if lang not in pools:
            raise ValueError(f"unknown language code {lang!r}, expected one of {sorted(all_langs)}")
        pool = pools[lang]
        target_chars = total_chars * w / total_weight
        chosen = []
        acc_chars = 0
        idx = 0
        while acc_chars < target_chars:
            s = pool[idx % len(pool)]
            chosen.append(s)
            acc_chars += len(s)
            idx += 1
        sents.extend(chosen)
    random.Random(seed).shuffle(sents)
    return sents


def build_condition_sentences(condition, seed=0):
    """Returns a seed-shuffled sentence list for `condition`, with the first
    EVAL_N_SENTENCES sentences of each language held out (reserved for
    evaluation). "balanced"/"mixed_5lang" (see MIXED_LANGS) interleave
    (shuffle together) their constituent languages' full FLORES pools so any
    text window D2MoE later samples is drawn from a genuinely mixed pool, not
    long same-language runs from naive concatenation.

    Placebo arm B (2026-07-25, added per user review to establish a same-
    language resampling noise floor -- see 00_docs/03_기술노트.md "placebo"):
    `english_only_b`/`korean_only_b` draw from the SECOND HALF (by original
    FLORES index, before any shuffling) of the same post-eval-exclusion pool
    that `english_only`/`korean_only` (arm A) draw from -- the same disjoint-
    chunk-carving idea Toy0/Phase0.5 used for english_a/english_b
    (data_utils.token_budget_chunks), applied at the pool level since arm A
    was already run against the undivided pool and can't be retroactively
    re-split. This guarantees arm B's pool is a clean, dedicated half no
    other condition draws from -- it does not retroactively guarantee zero
    overlap with whichever specific windows arm A's own random sampling
    happened to touch (disclosed limitation, not silently assumed away).

    TARGETED_CONDITIONS ("disagreement_targeted"/"disagreement_targeted_cap50",
    2026-07-27, §6 proposed method): reads the per-language weight allocation
    phase1_6_targeted_budget.py computed and saved for THIS condition and
    seed (budget_allocation_path(seed, condition)), and builds the
    calibration pool via build_weighted_sentences() instead of an equal-share
    interleave. Raises a clear FileNotFoundError (not a silent fallback) if
    that file doesn't exist yet -- Stage 2's allocator must run first, with
    --out-condition matching this condition name."""
    if condition in TARGETED_CONDITIONS:
        alloc_path = budget_allocation_path(seed, condition)
        if not alloc_path.exists():
            raise FileNotFoundError(
                f"{alloc_path} missing -- run phase1_6_targeted_budget.py --seed {seed} "
                f"--out-condition {condition} first (needs scan_disagreement_experts.py's output)."
            )
        weights = json.loads(alloc_path.read_text())["weights"]
        return build_weighted_sentences(weights, seed)

    if condition == ROBUST_MIX_CONDITION:
        return build_weighted_sentences(load_robust_mix_weights(), seed)

    if condition in CHAR_BALANCED_MIXES:
        langs = MIXED_LANGS[condition]
        return build_weighted_sentences({lang: 1.0 for lang in langs}, seed)
    if condition in MIXED_LANGS:
        sents = []
        for lang in MIXED_LANGS[condition]:
            sents.extend(_pool_for(lang, "A"))
    elif condition in PLACEBO_CONDITIONS:
        lang = LANG_OF_SIMPLE[condition]
        sents = list(_pool_for(lang, "B"))
    else:
        lang = LANG_OF_SIMPLE[condition]
        sents = list(_pool_for(lang, "A"))
    random.Random(seed).shuffle(sents)
    return sents


def make_fake_wikitext_dataset(condition, seed=0):
    from datasets import Dataset
    sentences = build_condition_sentences(condition, seed)
    return Dataset.from_dict({"text": sentences})
