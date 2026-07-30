"""§6 whitening-targeted robust mixing, STEP 2 tail: realized calibration
token measurement for the new "robust_mix" condition, in the same format as
00_docs/06_논문_구성.md's "[부록] 표 A1: 조건별 실측 calibration 토큰 길이"
(mean raw tokens before truncation / mean realized (512-capped) tokens /
512-cap 도달 비율 / fertility in chars-per-raw-token).

Reuses phase1_fisher.py's build_samples() UNMODIFIED (same seqlen*4-char
window, same per-seed RNG offset) so these numbers describe exactly the text
phase1_fisher.py itself will tokenize for real Fisher estimation -- this
script only adds one extra un-truncated tokenizer call per sample to recover
the "raw tokens before truncation" and fertility columns build_samples()
itself discards (it only needs the truncated encoding).

Usage:
    HF_HOME=/mnt/HDD/minjeong/hf_cache conda run -n d2moe_env python \\
        phase1_6_robust_mix_calib_tokens.py [--condition robust_mix]
        [--n-samples 64] [--seqlen 512] [--seeds 0 1 2]
    (CPU-only -- tokenizer load, no model forward pass.)
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_fisher  # noqa: E402 -- reuse build_samples() verbatim

OUT_DIR = SCRIPT_DIR.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def measure(condition, n_samples, seqlen, seeds):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(phase1_fisher.MODEL_PATH, use_fast=False)
    per_seed = {}
    for seed in seeds:
        samples = phase1_fisher.build_samples(condition, n_samples, seqlen, seed)
        raw_tokens, realized_tokens, fertility = [], [], []
        for text in samples:
            enc = tokenizer(text, return_tensors="pt")  # no truncation -- raw length
            n_raw = enc["input_ids"].shape[1]
            raw_tokens.append(n_raw)
            realized_tokens.append(min(n_raw, seqlen))
            fertility.append(len(text) / n_raw)
        per_seed[seed] = {
            "mean_raw_tokens": sum(raw_tokens) / len(raw_tokens),
            "mean_realized_tokens": sum(realized_tokens) / len(realized_tokens),
            "cap_hit_fraction": sum(1 for t in raw_tokens if t >= seqlen) / len(raw_tokens),
            "mean_fertility_chars_per_token": sum(fertility) / len(fertility),
            "n_samples": len(raw_tokens),
        }
    return per_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="robust_mix")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    per_seed = measure(args.condition, args.n_samples, args.seqlen, args.seeds)
    means = {
        key: sum(per_seed[s][key] for s in args.seeds) / len(args.seeds)
        for key in ("mean_raw_tokens", "mean_realized_tokens", "cap_hit_fraction", "mean_fertility_chars_per_token")
    }
    out = {"condition": args.condition, "n_samples": args.n_samples, "seqlen": args.seqlen,
           "per_seed": per_seed, "mean_over_seeds": means}
    out_path = OUT_DIR / f"calib_token_length_{args.condition}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[calib-tokens] wrote {out_path}")
    print(f"[calib-tokens] {args.condition} (n={args.n_samples} windows/seed, seeds={args.seeds}):")
    print(f"  mean raw tokens (절단 전): {means['mean_raw_tokens']:.1f}")
    print(f"  mean realized tokens: {means['mean_realized_tokens']:.1f}")
    print(f"  512-cap 도달 비율: {100*means['cap_hit_fraction']:.1f}%")
    print(f"  fertility (chars/token): {means['mean_fertility_chars_per_token']:.2f}")


if __name__ == "__main__":
    main()
