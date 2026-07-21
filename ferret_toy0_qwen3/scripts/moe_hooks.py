"""Instance-level monkeypatch of Qwen3MoeExperts.forward so we can observe
per-token routing decisions (top_k_index / top_k_weights) and the exact
down_proj input/output (z_t, y_t) for hit experts, without touching model
weights or requiring output_router_logits plumbing.

Qwen3MoeExperts.forward(hidden_states, top_k_index, top_k_weights):
    for each expert_idx that was hit:
        current_state = hidden_states[token_idx]                       # expert input x_t
        gate, up = linear(current_state, gate_up_proj[expert_idx]).chunk(2)
        z = act_fn(gate) * up                                          # down_proj input
        y = linear(z, down_proj[expert_idx])                           # down_proj output (= W z, no bias)
        out = y * top_k_weights[token_idx, top_k_pos]                  # router-weighted contribution

Module path: model.model.layers[i].mlp.experts, only present when
isinstance(layer.mlp, Qwen3MoeSparseMoeBlock) -- some Qwen3-30B-A3B layers
may be dense MLP instead of MoE.
"""
import torch
import torch.nn.functional as F


def install_capture_hook(experts_module, layer_id, callback):
    """Wrap experts_module.forward. callback(layer_id, expert_idx, token_idx,
    top_k_pos, x_t, z_t, y_t, router_weight) is invoked once per (expert_idx,
    hit-batch) inside the original computation, mirroring the exact ops of
    Qwen3MoeExperts.forward. Returns a handle with .remove().

    Some experts modules carry an accelerate AlignDevicesHook with
    offload=True (params replaced by meta tensors, materialized on the
    execution device only for the duration of one forward call). Directly
    overwriting .forward would bypass that hook's pre_forward/post_forward
    and leave us reading uninitialized meta tensors -- so when present, we
    drive pre_forward/post_forward ourselves around the same computation.
    """
    orig_forward = experts_module.forward
    hf_hook = getattr(experts_module, "_hf_hook", None)

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
