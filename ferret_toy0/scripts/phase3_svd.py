"""Phase 3: extract down_proj weight for each selected (layer, expert) pair
and compute one full-rank SVD (max rank = min(d_out, d_in) = 512 here, so
full SVD is cheap -- no need for randomized SVD at this model scale).

Only one expert's matrix is ever placed on GPU at a time; the full model is
not needed at all for this phase, so we load only the state_dict of the two
target expert index for each layer (still cheapest to just load the model,
since it's <3GB, but we process experts sequentially regardless per the
methodology's "no simultaneous GPU residency" rule).
"""
import json
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
TRACE_DIR = "/home/minjeong/project/FERRET/ferret_toy0/traces"
FACTOR_DIR = "/home/minjeong/project/FERRET/ferret_toy0/factors"
SEL_PATH = "/home/minjeong/project/FERRET/ferret_toy0/routing_counts/selection.json"


def main():
    import os
    os.makedirs(FACTOR_DIR, exist_ok=True)

    with open(SEL_PATH) as f:
        selection = json.load(f)
    pairs = []
    for tag, info in selection.items():
        for e in info["experts"]:
            pairs.append((info["layer_id"], e["expert_id"]))

    # Load model on CPU only; we never need GPU for a 1024x512 SVD.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()

    manifest = {}
    for lid, eid in pairs:
        bsm = model.model.layers[lid].block_sparse_moe
        W = bsm.experts.down_proj[eid].detach().clone().float()  # (d_out=hidden, d_in=intermediate)
        d_out, d_in = W.shape
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:(d_out,R) S:(R,) Vh:(R,d_in), R=min(d_out,d_in)
        R = S.shape[0]

        # sanity check: reconstruct at full rank
        recon = (U * S) @ Vh
        err = (recon - W).norm() / W.norm()
        assert err < 1e-4, f"SVD reconstruction error too high: {err}"

        tag = f"layer{lid:02d}_expert{eid:02d}"
        torch.save({"U": U.half(), "S": S.float(), "Vh": Vh.half(), "R": R, "d_out": d_out, "d_in": d_in},
                   f"{FACTOR_DIR}/{tag}_downproj_svd.pt")
        manifest[tag] = {"layer_id": lid, "expert_id": eid, "d_out": d_out, "d_in": d_in, "R": R,
                          "full_rank_recon_rel_err": err.item()}
        print(tag, "d_out=", d_out, "d_in=", d_in, "R=", R, "full-rank recon err=", err.item())

    with open(f"{FACTOR_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
