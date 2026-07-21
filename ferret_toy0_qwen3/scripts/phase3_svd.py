"""Phase 3: extract down_proj weight for each selected (layer, expert) pair
and compute one full-rank SVD (max rank = min(d_out, d_in) = 768 here, so
full SVD is cheap -- no need for randomized SVD at this model scale).

Qwen3-30B-A3B is ~60GB in bf16, too large to load the whole model just to
read two small (2048, 768) matrices. The on-disk checkpoint stores each
expert's down_proj as its own 2D tensor (model.layers.{L}.mlp.experts.{E}.
down_proj.weight) -- transformers stacks these into the batched 3D
Qwen3MoeExperts.down_proj parameter at load time, but for Phase 3 we read
the unfused tensor straight out of the safetensors shard via the weight
map, so the full model is never instantiated (methodology section 7 /
Phase 3 step 1: "가능하면 체크포인트를 CPU에서 로드").
"""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = Path(
    "/mnt/HDD/minjeong/hf_cache/hub/models--Qwen--Qwen3-30B-A3B/snapshots/"
    "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)
TRACE_DIR = ROOT / "traces"
FACTOR_DIR = ROOT / "factors"
SEL_PATH = ROOT / "routing_counts" / "selection.json"


def load_down_proj(weight_map, layer_id, expert_id):
    key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
    shard = weight_map[key]
    with safe_open(SNAPSHOT_DIR / shard, framework="pt") as f:
        return f.get_tensor(key)  # (d_out=hidden, d_in=intermediate), bf16


def main():
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    with open(SEL_PATH) as f:
        selection = json.load(f)
    pairs = []
    for tag, info in selection.items():
        for e in info["experts"]:
            pairs.append((info["layer_id"], e["expert_id"]))

    with open(SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    manifest = {}
    for lid, eid in pairs:
        W = load_down_proj(weight_map, lid, eid).float()  # (d_out, d_in)
        d_out, d_in = W.shape
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:(d_out,R) S:(R,) Vh:(R,d_in), R=min(d_out,d_in)
        R = S.shape[0]

        # sanity check: reconstruct at full rank
        recon = (U * S) @ Vh
        err = (recon - W).norm() / W.norm()
        assert err < 1e-4, f"SVD reconstruction error too high: {err}"

        tag = f"layer{lid:02d}_expert{eid:02d}"
        torch.save({"U": U.half(), "S": S.float(), "Vh": Vh.half(), "R": R, "d_out": d_out, "d_in": d_in},
                   FACTOR_DIR / f"{tag}_downproj_svd.pt")
        manifest[tag] = {"layer_id": lid, "expert_id": eid, "d_out": d_out, "d_in": d_in, "R": R,
                          "full_rank_recon_rel_err": err.item()}
        print(tag, "d_out=", d_out, "d_in=", d_in, "R=", R, "full-rank recon err=", err.item())

    with open(FACTOR_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
