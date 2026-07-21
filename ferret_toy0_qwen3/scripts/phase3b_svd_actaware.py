"""Phase 3b: activation-aware (whitened) SVD of down_proj, per Section 19's
retry instruction.

We minimize sum_t || (W - W_r) z_t ||^2 over the calibration activations,
which is equivalent to the ordinary best-rank-r approximation of A = W @ L
(where Gram = E[z z^T] = L L^T, Cholesky), since:

    sum_t ||(W-Wr) z_t||^2  ~=  || (W-Wr) L ||_F^2   (sample covariance form)

Best rank-r A_r = U_r S_r V_r^T (plain SVD of A). Then W_r = A_r @ L^{-1},
and at inference: y_hat = U_r (S_r * (V_r^T @ (L^{-1} z)))
                          = U_r (S_r * (V_r^T @ z')),  z' := L^{-1} z

We store U, S, Vh (of A) and Linv so replay just needs one extra (d_in x
d_in) matmul to whiten z before projecting. down_proj read directly from
the safetensors shard (same approach as Phase 3), not via a loaded model.
"""
import json
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
        return f.get_tensor(key)


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
        tag = f"layer{lid:02d}_expert{eid:02d}"
        W = load_down_proj(weight_map, lid, eid).float()  # (d_out, d_in)
        d_out, d_in = W.shape

        Z = torch.load(TRACE_DIR / f"{tag}_z_calib.pt", weights_only=True).float()  # (Nc, d_in)
        Nc = Z.shape[0]
        Gram = (Z.T @ Z) / Nc                                  # (d_in, d_in)
        ridge = 1e-4 * torch.diag(Gram).mean()
        Gram = Gram + ridge * torch.eye(d_in)
        L = torch.linalg.cholesky(Gram)                        # Gram = L L^T
        Linv = torch.linalg.inv(L)

        A = W @ L                                               # (d_out, d_in)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        R = S.shape[0]

        # sanity: full-rank reconstruction of W via whitened factors
        W_full = (U * S) @ Vh @ Linv
        err = (W_full - W).norm() / W.norm()
        assert err < 1e-3, f"act-aware full-rank recon error too high: {err}"

        torch.save({"U": U.half(), "S": S.float(), "Vh": Vh.half(), "Linv": Linv.half(),
                    "R": R, "d_out": d_out, "d_in": d_in, "Nc": Nc, "ridge": ridge.item()},
                   FACTOR_DIR / f"{tag}_downproj_svd_actaware.pt")
        manifest[tag] = {"layer_id": lid, "expert_id": eid, "d_out": d_out, "d_in": d_in, "R": R,
                          "Nc": Nc, "full_rank_recon_rel_err": err.item()}
        print(tag, "Nc=", Nc, "R=", R, "recon err=", err.item())

    with open(FACTOR_DIR / "manifest_actaware.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
