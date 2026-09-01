"""Experiment 6 chunk-restricted Q/K/V projections.

This module is deliberately separate from the MHAR routing implementation.
The existing model is initialized first, then only its Q/K/V projections are
replaced by grouped projections copied from the corresponding diagonal blocks
of the dense initialization.  The attention computation, Q/K normalization,
RoPE, GQA repetition, and dense output projection are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


SUPPORTED_GROUPS = (4, 8)


class GroupedLinear(nn.Module):
    """Block-diagonal linear map over contiguous input/output groups.

    Input group ``g`` can affect only output group ``g``.  The parameter tensor
    has shape ``[groups, outputs_per_group, inputs_per_group]`` so disallowed
    dense weights do not exist and are not counted as trainable parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int,
        *,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if groups not in SUPPORTED_GROUPS:
            raise ValueError(f"Experiment 6 supports groups {SUPPORTED_GROUPS}, got {groups}")
        if in_features % groups or out_features % groups:
            raise ValueError(
                f"in_features={in_features} and out_features={out_features} "
                f"must both be divisible by groups={groups}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groups = int(groups)
        self.in_features_per_group = in_features // groups
        self.out_features_per_group = out_features // groups
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(
            groups, self.out_features_per_group, self.in_features_per_group,
            **factory,
        ))
        self.bias = nn.Parameter(torch.empty(groups, self.out_features_per_group, **factory)) \
            if bias else None

    @classmethod
    def from_dense(cls, dense: nn.Linear, groups: int) -> "GroupedLinear":
        """Copy the allowed diagonal blocks from an initialized dense layer."""

        grouped = cls(
            dense.in_features,
            dense.out_features,
            groups,
            bias=dense.bias is not None,
            device=dense.weight.device,
            dtype=dense.weight.dtype,
        )
        if not dense.weight.is_meta:
            dense_weight = dense.weight.view(
                groups, grouped.out_features_per_group, dense.in_features)
            blocks = [
                dense_weight[index, :,
                             index * grouped.in_features_per_group:
                             (index + 1) * grouped.in_features_per_group]
                for index in range(groups)
            ]
            with torch.no_grad():
                grouped.weight.copy_(torch.stack(blocks))
                if grouped.bias is not None:
                    grouped.bias.copy_(dense.bias.view(groups, grouped.out_features_per_group))
        return grouped

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {inputs.shape[-1]}")
        grouped_inputs = inputs.reshape(
            *inputs.shape[:-1], self.groups, self.in_features_per_group)
        outputs = torch.einsum("...gi,goi->...go", grouped_inputs, self.weight)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs.reshape(*inputs.shape[:-1], self.out_features)

    def dense_weight(self) -> torch.Tensor:
        """Materialize the equivalent masked dense matrix for testing only."""

        dense = self.weight.new_zeros(self.out_features, self.in_features)
        for group in range(self.groups):
            out_start = group * self.out_features_per_group
            in_start = group * self.in_features_per_group
            dense[
                out_start:out_start + self.out_features_per_group,
                in_start:in_start + self.in_features_per_group,
            ] = self.weight[group]
        return dense

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"groups={self.groups}, bias={self.bias is not None}")


@dataclass(frozen=True)
class GroupedQKVReport:
    groups: int
    layers: int
    dense_qkv_parameters: int
    grouped_qkv_parameters: int

    @property
    def retained_fraction(self) -> float:
        return self.grouped_qkv_parameters / self.dense_qkv_parameters

    @property
    def removed_parameters(self) -> int:
        return self.dense_qkv_parameters - self.grouped_qkv_parameters


def apply_chunk_restricted_qkv(model: nn.Module, groups: int) -> GroupedQKVReport:
    """Replace Q/K/V in every decoder layer; never replace dense ``o_proj``."""

    if groups not in SUPPORTED_GROUPS:
        raise ValueError(f"Experiment 6 supports groups {SUPPORTED_GROUPS}, got {groups}")
    layers = model.model.layers
    dense_parameters = 0
    grouped_parameters = 0
    for layer_index, layer in enumerate(layers):
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(attention, name)
            if isinstance(projection, GroupedLinear):
                if projection.groups != groups:
                    raise ValueError(
                        f"layer {layer_index} {name} already uses {projection.groups} groups")
                grouped_parameters += projection.weight.numel()
                if projection.bias is not None:
                    grouped_parameters += projection.bias.numel()
                dense_parameters += projection.in_features * projection.out_features
                if projection.bias is not None:
                    dense_parameters += projection.out_features
                continue
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"layer {layer_index} {name} must be nn.Linear, got {type(projection).__name__}")
            dense_parameters += sum(parameter.numel() for parameter in projection.parameters())
            replacement = GroupedLinear.from_dense(projection, groups)
            setattr(attention, name, replacement)
            grouped_parameters += sum(parameter.numel() for parameter in replacement.parameters())

    model.config.experiment6_qkv_groups = groups
    return GroupedQKVReport(
        groups=groups,
        layers=len(layers),
        dense_qkv_parameters=dense_parameters,
        grouped_qkv_parameters=grouped_parameters,
    )


class Experiment6BaselineForCausalLM(Qwen3ForCausalLM):
    """Stock residual Qwen3 with opt-in grouped Q/K/V (G4/G8)."""

    def __init__(self, config):
        super().__init__(config)
        groups = getattr(config, "experiment6_qkv_groups", None)
        if groups is None:
            raise ValueError("Experiment6BaselineForCausalLM requires experiment6_qkv_groups")
        self.experiment6_qkv_report = apply_chunk_restricted_qkv(self, int(groups))


class Experiment6MHARForCausalLM(Qwen3AttnResForCausalLM):
    """Existing full-MH model with opt-in grouped Q/K/V (C4/C8)."""

    def __init__(self, config):
        if getattr(config, "attnres_mode", None) != "full_mh":
            raise ValueError("Experiment6MHARForCausalLM requires attnres_mode='full_mh'")
        groups = getattr(config, "experiment6_qkv_groups", None)
        if groups is None or int(groups) != int(config.attnres_num_heads):
            raise ValueError(
                "coupled MHAR requires experiment6_qkv_groups == attnres_num_heads")
        super().__init__(config)
        self.experiment6_qkv_report = apply_chunk_restricted_qkv(self, int(groups))


def experiment6_parameter_report(model: nn.Module) -> dict:
    """Return explicit total and QKV parameter counts for run manifests."""

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    qkv = 0
    output = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        qkv += sum(parameter.numel() for name in ("q_proj", "k_proj", "v_proj")
                   for parameter in getattr(attention, name).parameters())
        output += sum(parameter.numel() for parameter in attention.o_proj.parameters())
    return {
        "trainable_parameters": trainable,
        "qkv_parameters": qkv,
        "qkv_macs_per_token": qkv,
        "attention_output_parameters": output,
        "attention_output_macs_per_token": output,
        "qkv_groups": getattr(model.config, "experiment6_qkv_groups", None),
    }
