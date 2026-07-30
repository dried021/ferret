"""Instance-level monkeypatch of Qwen3MoeExperts.forward, copied unchanged from
ferret_kr/scripts/moe_hooks.py -- the model (Qwen3-30B-A3B) and module shape
are identical here, only the calibration language conditions differ (see
01_calibration_stats.py). Two hooks are defined, but otter's
Toy0 (01_calibration_stats.py) only uses install_capture_hook:

1. install_capture_hook -- observe routing + accumulate per-(layer,expert)
   hit counts, router-weighted output y_t, for Fisher-proxy importance and
   routing-coverage statistics per calibration condition.
2. install_corruption_hook -- unused in Toy0, kept for parity with ferret_kr
   in case a later phase needs corruption-based ablation on top of this.

Qwen3MoeExperts.forward(hidden_states, top_k_index, top_k_weights):
    for each expert_idx that was hit:
        current_state = hidden_states[token_idx]                       # expert input x_t
        gate, up = linear(current_state, gate_up_proj[expert_idx]).chunk(2)
        z = act_fn(gate) * up                                          # down_proj input
        y = linear(z, down_proj[expert_idx])                           # down_proj output (= W z, no bias)
        out = y * top_k_weights[token_idx, top_k_pos]                  # router-weighted contribution

Module path: model.model.layers[i].mlp.experts, only present when
isinstance(layer.mlp, Qwen3MoeSparseMoeBlock).
"""
import torch
import torch.nn.functional as F


def get_moe_layers(model, layer_ids=None):
    """Returns list of (layer_id, Qwen3MoeSparseMoeBlock) for decoder layers
    whose mlp is a MoE block. If layer_ids is given, restrict to those
    indices (skipping any that turn out to be dense, with a note printed)."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    out = []
    wanted = set(layer_ids) if layer_ids is not None else None
    for i, layer in enumerate(model.model.layers):
        if wanted is not None and i not in wanted:
            continue
        if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            out.append((i, layer.mlp))
        elif wanted is not None:
            print(f"warning: layer {i} requested but mlp is {type(layer.mlp).__name__}, not MoE -- skipped")
    return out


def _hooked_forward(experts_module, layer_id, compute):
    """Wraps experts_module.forward with `compute`, driving any accelerate
    AlignDevicesHook pre/post_forward manually (offloaded params are meta
    tensors outside of pre/post_forward's window) -- mirrors the pattern in
    ferret_toy0_qwen3_ver1/scripts/moe_hooks.py."""
    orig_forward = experts_module.forward
    hf_hook = getattr(experts_module, "_hf_hook", None)

    if hf_hook is not None:
        def patched_forward(hidden_states, top_k_index, top_k_weights):
            args, kwargs = hf_hook.pre_forward(experts_module, hidden_states, top_k_index, top_k_weights)
            out = compute(*args, **kwargs)
            return hf_hook.post_forward(experts_module, out)
    else:
        patched_forward = compute

    experts_module.forward = patched_forward

    class Handle:
        def remove(self):
            experts_module.forward = orig_forward
    return Handle()


def install_capture_hook(experts_module, layer_id, callback):
    """callback(layer_id, expert_idx, token_idx, top_k_pos, x_t, z_t, y_t,
    router_weight) invoked once per (expert_idx, hit-batch), mirroring the
    exact ops of Qwen3MoeExperts.forward. Returns a handle with .remove()."""

    def compute(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = experts_module.num_experts
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            z = experts_module.act_fn(gate) * up
            y = F.linear(z, experts_module.down_proj[expert_idx])
            weighted = y * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))

            callback(
                layer_id=layer_id,
                expert_idx=expert_idx,
                token_idx=token_idx,
                top_k_pos=top_k_pos,
                x_t=current_state,
                z_t=z,
                y_t=y,
                router_weight=top_k_weights[token_idx, top_k_pos],
            )
        return final_hidden_states

    return _hooked_forward(experts_module, layer_id, compute)


def install_corruption_hook(experts_module, layer_id, corrupt_spec, corruption_seed=0):
    """corrupt_spec: {expert_idx: {"mode": "zero"|"mean"|"noise",
    "mean_y": Tensor[hidden] (mean mode), "std_y": Tensor[hidden] (noise
    mode), "noise_scale": float (noise mode)}}. For any hit expert in
    corrupt_spec, y is transformed before the router-weighted accumulate;
    all other experts compute unchanged. No weight tensor is ever written."""

    def compute(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = experts_module.num_experts
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            z = experts_module.act_fn(gate) * up
            y = F.linear(z, experts_module.down_proj[expert_idx])

            spec = corrupt_spec.get(expert_idx)
            if spec is not None:
                mode = spec["mode"]
                if mode == "zero":
                    y = torch.zeros_like(y)
                elif mode == "mean":
                    y = spec["mean_y"].to(dtype=y.dtype, device=y.device).expand_as(y)
                elif mode == "noise":
                    gen = torch.Generator(device="cpu").manual_seed(
                        corruption_seed * 1_000_003 + layer_id * 10_007 + expert_idx
                    )
                    noise = torch.randn(y.shape, generator=gen).to(dtype=y.dtype, device=y.device)
                    std_y = spec["std_y"].to(dtype=y.dtype, device=y.device)
                    y = y + noise * std_y * spec["noise_scale"]
                else:
                    raise ValueError(f"unknown corruption mode: {mode}")

            weighted = y * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
        return final_hidden_states

    return _hooked_forward(experts_module, layer_id, compute)
