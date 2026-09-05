"""Experiment 11 dense-Q soft MHAR query specialization.

The ordinary dense Q parameter is preserved.  A fixed, non-parameterized mask
scales cross-chunk input coordinates for selected query heads before projection.
K, V, Q normalization, attention, and W_O remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.attention_residuals.modeling_qwen3_attnres import Qwen3AttnResForCausalLM


SOFT_Q_GROUPS = 8


@dataclass(frozen=True)
class Experiment11Run:
    run_id: str
    family: str
    lambda_value: float


RUNS = (
    Experiment11Run("s2q8-l000", "s2q8", 0.00),
    Experiment11Run("s2q8-l010", "s2q8", 0.10),
    Experiment11Run("s2q8-l025", "s2q8", 0.25),
    Experiment11Run("s2q8-l050", "s2q8", 0.50),
    Experiment11Run("gslq8-l000", "gslq8", 0.00),
    Experiment11Run("gslq8-l010", "gslq8", 0.10),
    Experiment11Run("gslq8-l025", "gslq8", 0.25),
    Experiment11Run("gslq8-l050", "gslq8", 0.50),
    Experiment11Run("m8-l100", "m8", 1.00),
)
RUN_BY_ID = {row.run_id: row for row in RUNS}


def run_spec(run_id: str) -> Experiment11Run:
    try:
        return RUN_BY_ID[run_id]
    except KeyError as error:
        raise ValueError(f"unknown Experiment 11 run ID: {run_id}") from error


class SoftSpecializedQueryLinear(nn.Linear):
    """Dense query projection with fixed per-row cross-chunk input gains.

    Trainable parameter names and shapes are identical to ``nn.Linear``.  The
    derived mask is a persistent, non-parameter buffer so meta-device
    ``from_pretrained`` loading cannot leave an uninitialized mask behind.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        family: str,
        lambda_value: float,
        groups: int = SOFT_Q_GROUPS,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        if family not in {"s2q8", "gslq8", "m8"}:
            raise ValueError(f"unsupported Experiment 11 family: {family}")
        if groups != SOFT_Q_GROUPS:
            raise ValueError(f"Experiment 11 requires exactly {SOFT_Q_GROUPS} groups")
        if not 0.0 <= float(lambda_value) <= 1.0:
            raise ValueError("lambda must lie in [0, 1]")
        if in_features % groups or out_features % (2 * groups):
            raise ValueError("soft Q requires 8 chunks and two query heads per group")
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.family = family
        self.lambda_value = float(lambda_value)
        self.groups = int(groups)
        self.in_features_per_group = in_features // groups
        self.head_dim = out_features // (2 * groups)
        self.register_buffer(
            "_experiment11_mask", self._make_mask(device=device, dtype=dtype),
            persistent=True,
        )

    @classmethod
    def from_dense(
        cls,
        dense: nn.Linear,
        *,
        family: str,
        lambda_value: float,
        groups: int = SOFT_Q_GROUPS,
    ) -> "SoftSpecializedQueryLinear":
        result = cls(
            dense.in_features,
            dense.out_features,
            family=family,
            lambda_value=lambda_value,
            groups=groups,
            bias=dense.bias is not None,
            device=dense.weight.device,
            dtype=dense.weight.dtype,
        )
        if not dense.weight.is_meta:
            with torch.no_grad():
                result.weight.copy_(dense.weight)
                if dense.bias is not None:
                    result.bias.copy_(dense.bias)
        return result

    def specialized_head_positions(self) -> tuple[int, ...]:
        if self.family == "s2q8":
            return (0, 1)
        if self.family == "gslq8":
            return (0,)
        return ()

    def diagnostic_head_positions(self) -> tuple[int, ...]:
        """Head slots used for effective-softness comparisons with M8."""

        return (0, 1) if self.family in {"s2q8", "m8"} else (0,)

    def _make_mask(self, *, device=None, dtype=None) -> torch.Tensor:
        mask = torch.ones(
            self.groups, 2, 1, self.in_features,
            device=device, dtype=dtype or torch.get_default_dtype(),
        )
        if self.family == "m8":
            return mask
        for group in range(self.groups):
            start = group * self.in_features_per_group
            stop = start + self.in_features_per_group
            for head in self.specialized_head_positions():
                mask[group, head].fill_(self.lambda_value)
                mask[group, head, :, start:stop] = 1
        return mask

    def masked_weight(self) -> torch.Tensor:
        return (
            self.weight.view(self.groups, 2, self.head_dim, self.in_features)
            * self._experiment11_mask
        ).reshape(self.out_features, self.in_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {inputs.shape[-1]}")
        return F.linear(inputs, self.masked_weight(), self.bias)

    def local_cross_components(
        self, inputs: torch.Tensor, *, fp32: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nominal local and effective cross contributions per Q head.

        The decomposition is defined relative to each GQA group's matching
        1/8-width chunk.  It is also meaningful for the all-global M8 endpoint.
        """

        if fp32:
            inputs, weight = inputs.float(), self.weight.float()
        else:
            weight = self.weight
        weight = weight.view(
            self.groups, 2, self.head_dim, self.in_features)
        local_rows, cross_rows = [], []
        for group in range(self.groups):
            start = group * self.in_features_per_group
            stop = start + self.in_features_per_group
            local_rows.append(torch.einsum(
                "...i,hoi->...ho", inputs[..., start:stop],
                weight[group, :, :, start:stop],
            ))
            left = torch.einsum(
                "...i,hoi->...ho", inputs[..., :start], weight[group, :, :, :start]
            ) if start else inputs.new_zeros(*inputs.shape[:-1], 2, self.head_dim)
            right = torch.einsum(
                "...i,hoi->...ho", inputs[..., stop:], weight[group, :, :, stop:]
            ) if stop < self.in_features else inputs.new_zeros(
                *inputs.shape[:-1], 2, self.head_dim)
            cross = left + right
            scales = []
            for head in range(2):
                scales.append(
                    self.lambda_value
                    if head in self.specialized_head_positions()
                    else 1.0
                )
            cross_rows.append(
                cross * torch.as_tensor(
                    scales, device=cross.device, dtype=cross.dtype
                ).view(*([1] * (cross.ndim - 2)), 2, 1)
            )
        return torch.stack(local_rows, dim=-3), torch.stack(cross_rows, dim=-3)


def apply_soft_specialized_query(model: nn.Module, run_id: str) -> None:
    row = run_spec(run_id)
    config = model.config
    if int(config.num_key_value_heads) != SOFT_Q_GROUPS:
        raise ValueError("Experiment 11 requires 8 KV heads")
    if int(config.num_attention_heads) != 2 * SOFT_Q_GROUPS:
        raise ValueError("Experiment 11 requires 16 Q heads")
    if int(config.hidden_size) % SOFT_Q_GROUPS:
        raise ValueError("hidden width must divide into eight chunks")
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        query = attention.q_proj
        if isinstance(query, SoftSpecializedQueryLinear):
            observed = (query.family, query.lambda_value, query.groups)
            expected = (row.family, row.lambda_value, SOFT_Q_GROUPS)
            if observed != expected:
                raise ValueError(
                    f"layer {layer_index} soft-Q identity mismatch: "
                    f"expected {expected}, got {observed}")
        elif isinstance(query, nn.Linear):
            attention.q_proj = SoftSpecializedQueryLinear.from_dense(
                query,
                family=row.family,
                lambda_value=row.lambda_value,
            )
        else:
            raise TypeError(
                f"layer {layer_index} q_proj must be nn.Linear, "
                f"got {type(query).__name__}")
        for name in ("k_proj", "v_proj", "o_proj"):
            if not isinstance(getattr(attention, name), nn.Linear):
                raise TypeError(f"Experiment 11 requires dense {name}")
    config.experiment11_run_id = row.run_id
    config.experiment11_family = row.family
    config.experiment11_lambda = row.lambda_value
    config.experiment11_soft_q_groups = SOFT_Q_GROUPS
    config.experiment11_specialized_head_positions = list(
        model.model.layers[0].self_attn.q_proj.specialized_head_positions())


class Experiment11MHARForCausalLM(Qwen3AttnResForCausalLM):
    """MHAR-8 model with a frozen Experiment 11 dense-Q input bias."""

    def __init__(self, config):
        run_id = getattr(config, "experiment11_run_id", None)
        if run_id is None:
            raise ValueError("Experiment11MHARForCausalLM requires a frozen run ID")
        if getattr(config, "attnres_mode", None) != "full_mh":
            raise ValueError("Experiment 11 requires attnres_mode='full_mh'")
        if int(getattr(config, "attnres_num_heads", 0)) != SOFT_Q_GROUPS:
            raise ValueError("Experiment 11 requires MHAR-8")
        super().__init__(config)
        apply_soft_specialized_query(self, run_id)


def experiment11_parameter_report(model: nn.Module) -> dict:
    """Report matched physical parameters and dense projection MAC proxies."""

    layers = list(model.model.layers)
    q_modules = [layer.self_attn.q_proj for layer in layers]
    if not all(isinstance(module, SoftSpecializedQueryLinear) for module in q_modules):
        raise TypeError("Experiment 11 requires SoftSpecializedQueryLinear in every layer")
    report = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        "experiment11_run_id": model.config.experiment11_run_id,
        "family": model.config.experiment11_family,
        "lambda": float(model.config.experiment11_lambda),
        "soft_q_groups": SOFT_Q_GROUPS,
        "mask_parameters": 0,
    }
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        count = sum(
            parameter.numel()
            for layer in layers
            for parameter in getattr(layer.self_attn, name).parameters()
        )
        key = name.removesuffix("_proj")
        report[f"{key}_parameters"] = count
        report[f"{key}_macs_per_token"] = count
    report["qkv_parameters"] = sum(
        report[f"{key}_parameters"] for key in ("q", "k", "v"))
    report["qkv_macs_per_token"] = report["qkv_parameters"]
    return report


def weight_specialization_metrics(model: nn.Module) -> dict:
    """Return FP32 layer/group/head weight-space compensation diagnostics."""

    rows = []
    eps = 1e-12
    for layer_index, layer in enumerate(model.model.layers):
        query = layer.self_attn.q_proj
        if not isinstance(query, SoftSpecializedQueryLinear):
            raise TypeError("weight metrics require Experiment 11 soft Q")
        weight = query.weight.detach().float().view(
            query.groups, 2, query.head_dim, query.in_features)
        for group in range(query.groups):
            start = group * query.in_features_per_group
            stop = start + query.in_features_per_group
            for head in query.diagnostic_head_positions():
                local = weight[group, head, :, start:stop]
                cross = torch.cat(
                    (weight[group, head, :, :start], weight[group, head, :, stop:]),
                    dim=-1,
                )
                local_rms = float(local.square().mean().sqrt())
                cross_rms = float(cross.square().mean().sqrt())
                scale = (
                    query.lambda_value
                    if head in query.specialized_head_positions()
                    else 1.0
                )
                rows.append({
                    "layer": layer_index,
                    "group": group,
                    "head_in_group": head,
                    "local_weight_rms": local_rms,
                    "cross_weight_rms": cross_rms,
                    "cross_scale": scale,
                    "r_weight": scale * cross_rms / max(local_rms, eps),
                })
    return {"epsilon": eps, "rows": rows}
