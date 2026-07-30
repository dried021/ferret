"""Isolated numerical test of D2MoE's delta-SVD reconstruction math itself
(2026-07-27), answering: "is the own-language gain/difference seen in the
Fisher x Scale results a bug in the delta decomposition, or a genuine
truncation effect at delta_ratio=0.8?"

Earlier attempt (phase1_merge_eval.py --delta-ratio 2.0, full pipeline,
all 27 layers) OOM'd: forcing rank>=full_rank(1408) via the ratio formula
makes each expert's delta_u+delta_v factorization ~1.69x BIGGER than the
original dense weight matrix (U@V is a memory-inefficient way to store a
full-rank matrix), so a "full-rank" merged model needs ~50.5GB just for
expert deltas across 27 layers -- more than the original 33GB model, and
more than the 44GiB (2x22GiB) GPU budget. That OOM was a resource-size
artifact of the test design, not a pipeline bug.

This script sidesteps that entirely: it merges ONLY ONE MoE layer (not all
27) at delta_ratio=2.0 (rank comfortably >= full rank 1408 for this model's
2048x1408 expert matrices -- see phase1_merge_eval.py's merge_condition()
docstring for the ratio->rank formula), then directly compares the
reconstructed per-expert weight (Wmean + delta_u @ delta_v, the same
composition merge_deepseek.py's own forward() and probe_process() use --
see meanW_deltaUV.forward() line 550 and the `torch.matmul(delta_u2.weight,
delta_v2.weight)` pattern in probe_process()) against the ORIGINAL expert
weight captured before merging. If delta decomposition itself is correct,
this reconstruction should match to bf16 rounding precision (~1e-3 relative)
regardless of calibration condition/language -- because at full rank there
is no compression left to be "language-sensitive" about.

Usage: CUDA_VISIBLE_DEVICES=2,3 conda run -n d2moe_env python phase1_lossless_unit_test.py
"""
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase1_merge_eval as pme  # noqa: E402 -- reuse load_model/RESULTS_ROOT/MODEL_PATH

LAYER_IDX = 5
CONDITION = "english_only"
SEED = 0
FULL_RANK_DELTA_RATIO = 2.0  # rank=1668 >= full_rank=1408, see module docstring


def main():
    model, _tokenizer = pme.load_model()
    mlp = model.model.layers[LAYER_IDX].mlp
    n_experts = len(mlp.experts)
    print(f"[lossless-unit] layer {LAYER_IDX}, {n_experts} experts, device={mlp.gate.weight.device}")

    orig_gate = [mlp.experts[j].gate_proj.weight.detach().float().clone() for j in range(n_experts)]
    orig_up = [mlp.experts[j].up_proj.weight.detach().float().clone() for j in range(n_experts)]
    orig_down = [mlp.experts[j].down_proj.weight.detach().float().clone() for j in range(n_experts)]

    cond_dir = pme.RESULTS_ROOT / CONDITION / f"seed{SEED}"
    expert_freq_paths = list(cond_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
    expert_freq = json.loads(expert_freq_paths[0].read_text())
    fisher_info = torch.load(cond_dir / "fisher_processed.pt", map_location="cpu")

    import os
    old_cwd = Path.cwd()
    os.chdir(pme.D2MOE_DIR)
    try:
        from config import cfg
        from model.merge_deepseek import Merge_deepseekMoE
        cfg.setdefault("prune_metric", cfg["control"]["prune_metric"])
        cfg.setdefault("prune_ratio", 0.0)
        cfg.setdefault("test_stage", False)
        cfg.setdefault("no_probe_process", False)
        cfg.setdefault("skip_layers", [[]])
        cfg.setdefault("calibration_stage", False)
        cfg.setdefault("batch_size", cfg["control"]["batch_size"])
        cfg.setdefault("mode", cfg["control"]["mode"])
        cfg.setdefault("prune_method", cfg["control"]["prune_method"])
        cfg.setdefault("tc_multiple", 64)
        cfg.setdefault("onlyprobe", False)
        cfg.setdefault("gate_probe_ratio", 1.0)
        cfg.setdefault("up_probe_ratio", 1.0)

        merge_block = Merge_deepseekMoE(
            model.config, share_ratio=1.0, delta_ratio=FULL_RANK_DELTA_RATIO,
            expert_freq=expert_freq[str(LAYER_IDX)], delta_share_V=False, delta_share_U=False,
            merge_method="fisher", shared_infer=False,
        ).to(mlp.gate.weight.device)
        merge_block.merge_experts(mlp, svd_scale=None, hessian=fisher_info[LAYER_IDX], scale_type="svdllm")
    finally:
        os.chdir(old_cwd)

    print(f"[lossless-unit] actual delta_low_rank used = {merge_block.experts[0].delta_low_rank} "
          f"(requested via ratio={FULL_RANK_DELTA_RATIO}; full_rank=1408 expected to saturate)")

    def rel_err(recon, orig):
        return (recon - orig).norm().item() / orig.norm().item()

    def max_abs(recon, orig):
        return (recon - orig).abs().max().item()

    rows = []
    for j in range(n_experts):
        e = merge_block.experts[j]
        recon_gate = (merge_block.Wmean_gate.weight.float() + torch.matmul(e.delta_u1.weight.float(), e.delta_v1.weight.float()))
        recon_up = (merge_block.Wmean_up.weight.float() + torch.matmul(e.delta_u3.weight.float(), e.delta_v3.weight.float()))
        recon_down = (merge_block.Wmean_down.weight.float() + torch.matmul(e.delta_u2.weight.float(), e.delta_v2.weight.float()))
        rows.append(("gate", j, rel_err(recon_gate, orig_gate[j]), max_abs(recon_gate, orig_gate[j])))
        rows.append(("up", j, rel_err(recon_up, orig_up[j]), max_abs(recon_up, orig_up[j])))
        rows.append(("down", j, rel_err(recon_down, orig_down[j]), max_abs(recon_down, orig_down[j])))

    rel_errs = [r[2] for r in rows]
    max_abs_errs = [r[3] for r in rows]
    print(f"\n[lossless-unit] over {len(rows)} (proj,expert) reconstructions:")
    print(f"  mean relative Frobenius error = {sum(rel_errs)/len(rel_errs):.6f}")
    print(f"  max  relative Frobenius error = {max(rel_errs):.6f}")
    print(f"  max  abs element error        = {max(max_abs_errs):.6f}")
    worst = max(rows, key=lambda r: r[2])
    print(f"  worst case: proj={worst[0]} expert={worst[1]} rel_err={worst[2]:.6f}")

    verdict = "LOSSLESS (decomposition math is correct)" if max(rel_errs) < 0.01 else "NOT LOSSLESS -- possible bug in delta decomposition"
    print(f"\n[lossless-unit] VERDICT: {verdict}")

    out = {
        "layer_idx": LAYER_IDX, "condition": CONDITION, "seed": SEED,
        "delta_ratio_requested": FULL_RANK_DELTA_RATIO,
        "delta_low_rank_actual": merge_block.experts[0].delta_low_rank,
        "mean_rel_err": sum(rel_errs) / len(rel_errs), "max_rel_err": max(rel_errs),
        "max_abs_err": max(max_abs_errs), "verdict": verdict,
    }
    out_path = pme.RESULTS_ROOT / "phase1_lossless_unit_test_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[lossless-unit] wrote {out_path}")


if __name__ == "__main__":
    main()
