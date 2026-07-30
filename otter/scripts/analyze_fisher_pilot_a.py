"""Fisher Pilot A analysis: answers the two questions the pilot was scoped
for (see fisher_pilot_a.py docstring):

  1. Does the forward-only proxy's expert ranking correlate positively with
     the real gradient-Fisher expert ranking, per layer/condition?
  2. Is the EN-vs-non-EN expert-rank gap bigger at the sensitive/final layers
     than at the early/transition layers, in the REAL Fisher -- and does the
     proxy show the same qualitative pattern (even if it disagrees on scale)?

Usage:
    conda run -n d2moe_env python analyze_fisher_pilot_a.py [--smoke]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path("/mnt/HDD/minjeong/d2moe_results/fisher_pilot_a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    in_path = RESULTS_DIR / ("pilot_a_results_smoke.json" if args.smoke else "pilot_a_results.json")
    data = json.loads(in_path.read_text())

    rows = []
    for layer_str, conds in data.items():
        role = conds.get("role", "?")
        cond_scores = {c: v for c, v in conds.items() if c != "role"}

        # Question 1: proxy vs real, per condition
        proxy_vs_real = {}
        for cond, v in cond_scores.items():
            rho, p = spearmanr(v["real_fisher"], v["proxy"])
            proxy_vs_real[cond] = {"rho": float(rho), "p": float(p)}

        # Question 2: within-EN vs EN-vs-non-EN gap, for real and for proxy separately
        def gap(metric_key):
            en_a = cond_scores["english_a"][metric_key]
            en_b = cond_scores["english_b"][metric_key]
            ko = cond_scores["korean"][metric_key]
            zh = cond_scores["chinese"][metric_key]
            within_en, _ = spearmanr(en_a, en_b)
            en_ko, _ = spearmanr(en_a, ko)
            en_zh, _ = spearmanr(en_a, zh)
            return {
                "within_en_rho": float(within_en),
                "en_ko_rho": float(en_ko),
                "en_zh_rho": float(en_zh),
                "gap": float(within_en - np.mean([en_ko, en_zh])),
            }

        real_gap = gap("real_fisher")
        proxy_gap = gap("proxy")

        rows.append({
            "layer": int(layer_str), "role": role,
            "proxy_vs_real": proxy_vs_real,
            "real_gap": real_gap,
            "proxy_gap": proxy_gap,
        })
        print(f"\n[analyze] layer {layer_str} ({role})")
        print(f"  proxy vs real Spearman: " +
              ", ".join(f"{c}={v['rho']:.3f}" for c, v in proxy_vs_real.items()))
        print(f"  REAL  within_en={real_gap['within_en_rho']:.3f} en_ko={real_gap['en_ko_rho']:.3f} "
              f"en_zh={real_gap['en_zh_rho']:.3f} gap={real_gap['gap']:.3f}")
        print(f"  PROXY within_en={proxy_gap['within_en_rho']:.3f} en_ko={proxy_gap['en_ko_rho']:.3f} "
              f"en_zh={proxy_gap['en_zh_rho']:.3f} gap={proxy_gap['gap']:.3f}")

    out_path = RESULTS_DIR / ("pilot_a_analysis_smoke.json" if args.smoke else "pilot_a_analysis.json")
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\n[analyze] wrote {out_path}")

    # Overall verdicts
    mean_proxy_real_rho = np.mean([v["rho"] for r in rows for v in r["proxy_vs_real"].values()])
    early_like = [r for r in rows if r["role"] in ("early", "transition")]
    late_like = [r for r in rows if r["role"] in ("sensitive", "final")]
    real_gap_early = np.mean([r["real_gap"]["gap"] for r in early_like]) if early_like else float("nan")
    real_gap_late = np.mean([r["real_gap"]["gap"] for r in late_like]) if late_like else float("nan")

    print(f"\n[analyze] Q1 verdict: mean proxy-vs-real Spearman rho = {mean_proxy_real_rho:.3f} "
          f"({'POSITIVE -- proxy tracks real Fisher' if mean_proxy_real_rho > 0 else 'NOT positive -- proxy does not track real Fisher'})")
    print(f"[analyze] Q2 verdict: real-Fisher gap early/transition={real_gap_early:.3f} vs "
          f"sensitive/final={real_gap_late:.3f} "
          f"({'CONFIRMS late-layer sensitivity' if real_gap_late > real_gap_early else 'DOES NOT confirm late-layer sensitivity'})")


if __name__ == "__main__":
    main()
