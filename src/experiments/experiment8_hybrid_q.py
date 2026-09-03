"""Experiment 8 hybrid-query / global-key-value attention projections.

Each eight-way GQA group has two query heads in the standard contiguous order:
the even head reads only its matching 160-dimensional chunk and the odd head
reads the complete 1,280-dimensional residual. K, V, and O stay dense.
"""

from __future__ import annotations

import torch
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


HYBRID_Q_GROUPS = 8


class HybridQueryLinear(nn.Module):
    """Interleaved local/global query projection for two-query GQA groups.

    Output rows are ordered ``local_0, global_0, ..., local_7, global_7``.
    Disallowed cross-chunk weights for local heads do not exist as parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int = HYBRID_Q_GROUPS,
        *,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if groups != HYBRID_Q_GROUPS:
            raise ValueError(f"Experiment 8 requires exactly {HYBRID_Q_GROUPS} groups")
        if in_features % groups:
            raise ValueError("input width must divide evenly into hybrid-Q groups")
        if out_features % (2 * groups):
            raise ValueError("hybrid Q requires exactly two equal-width query heads per group")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groups = int(groups)
        self.in_features_per_group = in_features // groups
        self.head_dim = out_features // (2 * groups)
        factory = {"device": device, "dtype": dtype}
        self.local_weight = nn.Parameter(torch.empty(
            groups, self.head_dim, self.in_features_per_group, **factory))
        self.global_weight = nn.Parameter(torch.empty(
            groups, self.head_dim, in_features, **factory))
        self.local_bias = nn.Parameter(torch.empty(groups, self.head_dim, **factory)) \
            if bias else None
        self.global_bias = nn.Parameter(torch.empty(groups, self.head_dim, **factory)) \
            if bias else None

    @classmethod
    def from_dense(cls, dense: nn.Linear, groups: int = HYBRID_Q_GROUPS):
        """Retain matching local-even and global-odd rows from a dense Q layer."""

        hybrid = cls(
            dense.in_features,
            dense.out_features,
            groups,
            bias=dense.bias is not None,
            device=dense.weight.device,
            dtype=dense.weight.dtype,
        )
        if not dense.weight.is_meta:
            weight = dense.weight.view(groups, 2, hybrid.head_dim, dense.in_features)
            local_blocks = [
                weight[group, 0, :,
                       group * hybrid.in_features_per_group:
                       (group + 1) * hybrid.in_features_per_group]
                for group in range(groups)
            ]
            with torch.no_grad():
                hybrid.local_weight.copy_(torch.stack(local_blocks))
                hybrid.global_weight.copy_(weight[:, 1])
                if dense.bias is not None:
                    bias = dense.bias.view(groups, 2, hybrid.head_dim)
                    hybrid.local_bias.copy_(bias[:, 0])
                    hybrid.global_bias.copy_(bias[:, 1])
        return hybrid

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {inputs.shape[-1]}")
        grouped_inputs = inputs.reshape(
            *inputs.shape[:-1], self.groups, self.in_features_per_group)
        local = torch.einsum("...gi,goi->...go", grouped_inputs, self.local_weight)
        global_ = torch.einsum("...i,goi->...go", inputs, self.global_weight)
        if self.local_bias is not None:
            local = local + self.local_bias
            global_ = global_ + self.global_bias
        interleaved = torch.stack((local, global_), dim=-2)
        return interleaved.reshape(*inputs.shape[:-1], self.out_features)

    def dense_weight(self) -> torch.Tensor:
        """Materialize the equivalent masked dense Q matrix for tests only."""

        dense = self.global_weight.new_zeros(self.out_features, self.in_features)
        viewed = dense.view(self.groups, 2, self.head_dim, self.in_features)
        viewed[:, 1].copy_(self.global_weight)
        for group in range(self.groups):
            start = group * self.in_features_per_group
            viewed[group, 0, :, start:start + self.in_features_per_group].copy_(
                self.local_weight[group])
        return dense

    def dense_bias(self):
        if self.local_bias is None:
            return None
        return torch.stack((self.local_bias, self.global_bias), dim=1).reshape(
            self.out_features)


def apply_hybrid_query(model: nn.Module, groups: int = HYBRID_Q_GROUPS) -> None:
    """Replace only Q with the frozen interleaved hybrid projection."""

    if groups != HYBRID_Q_GROUPS:
        raise ValueError(f"Experiment 8 requires exactly {HYBRID_Q_GROUPS} groups")
    config = model.config
    if int(config.num_key_value_heads) != groups:
        raise ValueError("hybrid-Q groups must equal the number of KV heads")
    if int(config.num_attention_heads) != 2 * groups:
        raise ValueError("hybrid Q requires exactly two query heads per KV head")
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        query = attention.q_proj
        if isinstance(query, HybridQueryLinear):
            if query.groups != groups:
                raise ValueError(
                    f"layer {layer_index} q_proj already uses {query.groups} groups")
        elif isinstance(query, nn.Linear):
            attention.q_proj = HybridQueryLinear.from_dense(query, groups)
        else:
            raise TypeError(
                f"layer {layer_index} q_proj must be nn.Linear, got {type(query).__name__}")
        for name in ("k_proj", "v_proj", "o_proj"):
            if not isinstance(getattr(attention, name), nn.Linear):
                raise TypeError(f"Experiment 8 requires dense {name} in layer {layer_index}")
    config.experiment8_hybrid_q_groups = int(groups)
    config.experiment8_local_head_position = "even"
    config.experiment8_global_head_position = "odd"


class Experiment8BaselineForCausalLM(Qwen3ForCausalLM):
    """Ordinary residual with one local and one global Q per GQA group."""

    def __init__(self, config):
        super().__init__(config)
        groups = getattr(config, "experiment8_hybrid_q_groups", None)
        if groups is None:
            raise ValueError("Experiment8BaselineForCausalLM requires hybrid-Q groups")
        apply_hybrid_query(self, int(groups))


class Experiment8MHARForCausalLM(Qwen3AttnResForCausalLM):
    """MHAR-8 with one local and one global Q per routed chunk/GQA group."""

    def __init__(self, config):
        groups = getattr(config, "experiment8_hybrid_q_groups", None)
        if getattr(config, "attnres_mode", None) != "full_mh":
            raise ValueError("Experiment8MHARForCausalLM requires attnres_mode='full_mh'")
        if groups is None or int(groups) != int(config.attnres_num_heads):
            raise ValueError("HQ8 requires hybrid-Q groups == attnres_num_heads")
        super().__init__(config)
        apply_hybrid_query(self, int(groups))


def experiment8_parameter_report(model: nn.Module) -> dict:
    """Report physical projection sizes and per-token linear MAC proxies."""

    report = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "hybrid_q_groups": getattr(model.config, "experiment8_hybrid_q_groups", None),
        "local_q_fraction": 0.5,
    }
    q_local = q_global = 0
    for layer in model.model.layers:
        query = layer.self_attn.q_proj
        if not isinstance(query, HybridQueryLinear):
            raise TypeError("Experiment 8 parameter report requires HybridQueryLinear")
        q_local += query.local_weight.numel()
        q_global += query.global_weight.numel()
        if query.local_bias is not None:
            q_local += query.local_bias.numel()
            q_global += query.global_bias.numel()
    report["q_local_parameters"] = q_local
    report["q_global_parameters"] = q_global
    report["q_parameters"] = q_local + q_global
    for name in ("k_proj", "v_proj", "o_proj"):
        count = sum(
            parameter.numel()
            for layer in model.model.layers
            for parameter in getattr(layer.self_attn, name).parameters()
        )
        key = name.removesuffix("_proj")
        report[f"{key}_parameters"] = count
    for key in ("q", "q_local", "q_global", "k", "v", "o"):
        report[f"{key}_macs_per_token"] = report[f"{key}_parameters"]
    report["qkv_parameters"] = sum(report[f"{key}_parameters"] for key in ("q", "k", "v"))
    report["qkv_macs_per_token"] = report["qkv_parameters"]
    return report
