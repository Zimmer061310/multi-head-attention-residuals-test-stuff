"""Experiment 7 local-query / global-key-value attention projections.

Only ``q_proj`` is block diagonal.  ``k_proj``, ``v_proj``, and ``o_proj``
remain the ordinary dense Qwen3 projections, so every query group attends to
keys and values computed from the complete residual stream.
"""

from __future__ import annotations

from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
from src.experiments.experiment6_coupled_qkv import GroupedLinear, SUPPORTED_GROUPS
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def apply_local_query(model: nn.Module, groups: int) -> None:
    """Replace only Q projections with physical grouped projections."""

    if groups not in SUPPORTED_GROUPS:
        raise ValueError(f"Experiment 7 supports groups {SUPPORTED_GROUPS}, got {groups}")
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        query = attention.q_proj
        if isinstance(query, GroupedLinear):
            if query.groups != groups:
                raise ValueError(
                    f"layer {layer_index} q_proj already uses {query.groups} groups")
        elif isinstance(query, nn.Linear):
            attention.q_proj = GroupedLinear.from_dense(query, groups)
        else:
            raise TypeError(
                f"layer {layer_index} q_proj must be nn.Linear, got {type(query).__name__}")
        for name in ("k_proj", "v_proj", "o_proj"):
            if not isinstance(getattr(attention, name), nn.Linear):
                raise TypeError(f"Experiment 7 requires dense {name} in layer {layer_index}")
    model.config.experiment7_local_q_groups = int(groups)


class Experiment7BaselineForCausalLM(Qwen3ForCausalLM):
    """Ordinary residual stream with local Q and global K/V (BLQ4/BLQ8)."""

    def __init__(self, config):
        super().__init__(config)
        groups = getattr(config, "experiment7_local_q_groups", None)
        if groups is None:
            raise ValueError("Experiment7BaselineForCausalLM requires local-Q groups")
        apply_local_query(self, int(groups))


class Experiment7MHARForCausalLM(Qwen3AttnResForCausalLM):
    """Matched MHAR chunks with local Q and global K/V (LQ4/LQ8)."""

    def __init__(self, config):
        groups = getattr(config, "experiment7_local_q_groups", None)
        if getattr(config, "attnres_mode", None) != "full_mh":
            raise ValueError("Experiment7MHARForCausalLM requires attnres_mode='full_mh'")
        if groups is None or int(groups) != int(config.attnres_num_heads):
            raise ValueError("LQ MHAR requires local-Q groups == attnres_num_heads")
        super().__init__(config)
        apply_local_query(self, int(groups))


def experiment7_parameter_report(model: nn.Module) -> dict:
    """Report real trainable projection sizes and per-token linear MAC proxies."""

    report = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "local_q_groups": getattr(model.config, "experiment7_local_q_groups", None),
    }
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        count = sum(
            parameter.numel()
            for layer in model.model.layers
            for parameter in getattr(layer.self_attn, name).parameters()
        )
        key = name.removesuffix("_proj")
        report[f"{key}_parameters"] = count
        report[f"{key}_macs_per_token"] = count
    report["qkv_parameters"] = sum(
        report[f"{key}_parameters"] for key in ("q", "k", "v"))
    report["qkv_macs_per_token"] = report["qkv_parameters"]
    return report
