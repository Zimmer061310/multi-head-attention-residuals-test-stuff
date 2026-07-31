"""
Qwen3 with Block Attention Residuals (AttnRes).

Replaces standard additive residual connections with softmax attention over
previous block representations, as described in:
  "Attention Residuals" (Kimi Team, arXiv:2603.15031)

Init follows the reference implementation (Kimi, arXiv:2603.15031): all
pseudo-query vectors are zero-initialized, so every depth softmax starts as
a uniform average over its sources; the proj weights then learn selective
routing during training.  (An earlier recency-bias identity-init variant
was removed — see git history of f75df9c.)
"""

import os
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

# Re-use Qwen3 components directly from the installed transformers package.
# We only override DecoderLayer and Model; everything else is unchanged.
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    Qwen3PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

# Fused Triton kernels for full_mh routing (optional; enabled via
# enable_fused_mhar() or ATTNRES_FUSED=1). Falls back to the eager path.
try:
    import mhar_triton as _mhar_fused
    if not _mhar_fused.HAS_TRITON:
        _mhar_fused = None
except ImportError:
    _mhar_fused = None


def enable_fused_mhar(enabled: bool = True):
    """Route full_mh depth routing through fused Triton kernels (fwd+bwd)."""
    if _mhar_fused is None:
        raise RuntimeError("mhar_triton unavailable (triton not installed?)")
    _mhar_fused.enable_fused_mhar(enabled)


# ---------------------------------------------------------------------------
# V-stream decoupled attention
# ---------------------------------------------------------------------------

class Qwen3AttnResAttention(Qwen3Attention):
    """Qwen3Attention extended to accept separate hidden states for value projection.

    When ``value_hidden_states`` is provided, V is projected from it instead of
    from ``hidden_states`` (which is still used for Q and K).  This enables
    per-stream delta routing: Q/K attend from one depth-aggregated input while
    V retrieves content from a different depth-aggregated input.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        value_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        # V from separate input if provided, else from hidden_states
        v_input = value_hidden_states if value_hidden_states is not None else hidden_states
        value_states = self.v_proj(v_input).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3HeadwiseAttention(Qwen3Attention):
    """Qwen3Attention where each GQA head group reads its own routed input.

    ``hidden_states`` is [B, T, G, D] — one depth-routed residual mix per KV
    head group (G = num_key_value_heads). Group g's query heads, key and value
    are all projected from slice g, so each head group attends from (and
    retrieves content out of) its own view of the depth history. FLOP cost is
    identical to standard attention: every projection weight row is applied to
    exactly one input. With all G slices equal this reduces exactly to
    Qwen3Attention on the shared slice.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, G, D]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t, g, d = hidden_states.shape
        n_q = self.config.num_attention_heads
        n_kv = self.config.num_key_value_heads
        if g != n_kv:
            raise ValueError(f"expected {n_kv} routed groups, got {g}")

        # Group the projection weights by KV head: query heads g*(n_q/n_kv) ..
        # (g+1)*(n_q/n_kv)-1 belong to KV head g (repeat_kv ordering).
        q_w = self.q_proj.weight.view(n_kv, (n_q // n_kv) * self.head_dim, d)
        k_w = self.k_proj.weight.view(n_kv, self.head_dim, d)
        v_w = self.v_proj.weight.view(n_kv, self.head_dim, d)

        query_states = torch.einsum("g e d, b t g d -> b t g e", q_w, hidden_states)
        query_states = query_states.reshape(b, t, n_q, self.head_dim)
        key_states = torch.einsum("g e d, b t g d -> b t g e", k_w, hidden_states)
        value_states = torch.einsum("g e d, b t g d -> b t g e", v_w, hidden_states)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(b, t, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DepthAttention(Qwen3Attention):
    """Depth-Attention: cross-layer value mixing (Zeng et al., arXiv:2606.05014).

    Before normal self-attention, the layer's own per-head query attends over the
    keys of a strided set of earlier layers at the SAME token position, and mixes
    those layers' (depth-mixed) values into the current value:

        a_{l,j}^t = softmax_{j in S(l)}( q_l^t . k_j^t / sqrt(d) )      (Eq 2)
        v~_l^t    = a_{l,l} v_l^t + sum_{j<l in S(l)} a_{l,j} v~_j^t    (Eq 3)
        O_l       = CausalAttn(Q_l, K_l, V~_l)                          (Eq 4)

    Reuses q_proj/k_proj/v_proj (no new params). GQA: queries are averaged within
    each group so depth-attention runs at KV-head resolution. The depth dot-product
    is RoPE-invariant (same position t), so it is computed pre-RoPE. The mixed value
    replaces V in self-attention (and in the KV cache, matching the paper).

    Returns (attn_output, k_raw, v_mixed); the layer appends (k_raw, v_mixed) to the
    depth cache for use by later layers.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        depth_cache: Optional[list] = None,
        depth_stride: int = 1,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,Hq,T,d]
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,Hkv,T,d]
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)               # [B,Hkv,T,d]

        B, Hq, T, d = q.shape
        Hkv = k.shape[1]
        g = Hq // Hkv

        # ---- Depth-attention value mixing (Eq 2-3), pre-RoPE (RoPE-invariant at same t) ----
        depth_cache = depth_cache or []
        ell = len(depth_cache)                       # this layer's depth index
        q_bar = q.view(B, Hkv, g, T, d).mean(dim=2)  # GQA query averaging -> [B,Hkv,T,d]

        src_keys = [k]   # own key (j = ell)
        src_vals = [v]   # own RAW value v_l (Eq 3 uses v_l, not v~_l, for the own term)
        j = ell - depth_stride
        while j >= 0:
            kj, vj = depth_cache[j]
            src_keys.append(kj)
            src_vals.append(vj)                      # earlier layers contribute their depth-mixed v~_j
            j -= depth_stride

        if len(src_keys) == 1:
            v_mixed = v                              # layer 0 / no strided source -> no mixing
        else:
            Ks = torch.stack(src_keys, dim=0)        # [N,B,Hkv,T,d]
            Vs = torch.stack(src_vals, dim=0)        # [N,B,Hkv,T,d]
            logits = (q_bar.unsqueeze(0) * Ks).sum(-1) * self.scaling  # [N,B,Hkv,T]
            alpha = logits.softmax(dim=0)
            v_mixed = (alpha.unsqueeze(-1) * Vs).sum(dim=0)            # [B,Hkv,T,d]

        # ---- Standard causal self-attention with the mixed value (Eq 4) ----
        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(k_rot, v_mixed, self.layer_idx, cache_kwargs)
        else:
            key_states, value_states = k_rot, v_mixed

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self, q_rot, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # cache raw (pre-RoPE) key for future depth scoring, and the mixed value (Eq 3 recursion)
        return attn_output, k, v_mixed


# ---------------------------------------------------------------------------
# Config extension
# ---------------------------------------------------------------------------

class Qwen3AttnResConfig(Qwen3Config):
    """Qwen3Config extended with AttnRes hyper-parameters."""

    model_type = "qwen3_attnres"

    def __init__(self, attnres_num_blocks: int = 8,
                 attnres_mode: str = "block",
                 attnres_gate_type: str = "bias",
                 **kwargs):
        # Remove legacy keys if present (from old checkpoints)
        kwargs.pop("attnres_init_bias", None)
        kwargs.pop("attnres_gate_init", None)
        super().__init__(**kwargs)
        self.attnres_num_blocks = attnres_num_blocks
        # "block" = Block AttnRes (grouped), "full" = Full AttnRes (per-sublayer)
        # "delta" = Delta AttnRes (attend over sublayer outputs, not cumulative states)
        # "delta_block" = Delta + Block: attend over block-aggregated deltas (best of both)
        # "delta_v" = Delta with V-stream decoupling: V gets independent depth routing
        self.attnres_mode = attnres_mode
        # Null source for identity init (for fine-tuning pretrained models)
        self.attnres_use_null_source = kwargs.pop("attnres_use_null_source", False)
        # Gate type: "bias" (recency bias on softmax logit),
        #            "sigmoid_scalar" (scalar sigmoid gate between residual & attnres),
        #            "sigmoid_vector" (input-dependent per-dim sigmoid gate)
        self.attnres_gate_type = attnres_gate_type
        # Routing heads for "full_mh": hidden_size must be divisible by this
        self.attnres_num_heads = kwargs.pop("attnres_num_heads", 8)


# ---------------------------------------------------------------------------
# Core Block-AttnRes operation
# ---------------------------------------------------------------------------

def _block_attn_res_kernel(
    V: torch.Tensor,              # pre-stacked sources [N+1, B, T, D]
    query: torch.Tensor,          # (D,)
    norm: Qwen3RMSNorm,
) -> torch.Tensor:
    """Compiled inner kernel for block_attn_res (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", weights, V)


def _block_attn_res_kernel_with_entropy(
    V: torch.Tensor,
    query: torch.Tensor,
    norm: Qwen3RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled inner kernel for block_attn_res with entropy (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    h = torch.einsum("n b t, n b t d -> b t d", weights, V)
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
    return h, entropy


def _mh_block_attn_res_kernel(
    V: torch.Tensor,              # pre-stacked sources [N, B, T, D]
    query: torch.Tensor,          # (H, D/H) — per-head pseudo-queries
    norm: Qwen3RMSNorm,
) -> torch.Tensor:
    """Compiled inner kernel for mh_block_attn_res (no Python list ops)."""
    K = norm(V)  # full-D RMSNorm, same key normalization as single-head
    n, b, t, d = V.shape
    h, dh = query.shape
    logits = torch.einsum("h k, n b t h k -> n b t h", query, K.view(n, b, t, h, dh))
    weights = logits.softmax(dim=0)
    out = torch.einsum("n b t h, n b t h k -> b t h k", weights, V.view(n, b, t, h, dh))
    return out.reshape(b, t, d)


def _hw_block_attn_res_kernel(
    V: torch.Tensor,              # pre-stacked sources [N, B, T, D]
    query: torch.Tensor,          # (G, D) — per-head-group pseudo-queries
    norm: Qwen3RMSNorm,
) -> torch.Tensor:
    """Compiled inner kernel for hw_block_attn_res (no Python list ops).

    Unlike the mh kernel (which scores each query against a D/H slice of the
    sources), every group scores the FULL normed source with its own (D,)
    query and mixes the full source — groups differ only in routing, not in
    which dims they see. Output is [B, T, G, D]: one routed mix per group.
    """
    K = norm(V)
    logits = torch.einsum("g d, n b t d -> n b t g", query, K)
    weights = logits.softmax(dim=0)
    return torch.einsum("n b t g, n b t d -> b t g d", weights, V)


# Compiled versions (created lazily via enable_compile())
_compiled_block_kernel = None
_compiled_block_kernel_entropy = None
_compiled_delta_kernel = None
_compiled_delta_kernel_entropy = None
_compiled_mh_block_kernel = None
_compiled_hw_block_kernel = None


def enable_compile():
    """Compile the AttnRes kernels with torch.compile(dynamic=True)."""
    global _compiled_block_kernel, _compiled_block_kernel_entropy
    global _compiled_delta_kernel, _compiled_delta_kernel_entropy
    global _compiled_mh_block_kernel, _compiled_hw_block_kernel
    _compiled_block_kernel = torch.compile(
        _block_attn_res_kernel, dynamic=True)
    _compiled_block_kernel_entropy = torch.compile(
        _block_attn_res_kernel_with_entropy, dynamic=True)
    _compiled_delta_kernel = torch.compile(
        _delta_attn_res_kernel, dynamic=True)
    _compiled_delta_kernel_entropy = torch.compile(
        _delta_attn_res_kernel_with_entropy, dynamic=True)
    _compiled_mh_block_kernel = torch.compile(
        _mh_block_attn_res_kernel, dynamic=True)
    _compiled_hw_block_kernel = torch.compile(
        _hw_block_attn_res_kernel, dynamic=True)


def block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, T, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, T, D]
    proj: nn.Linear,              # learned pseudo-query weight  (d,)
    norm: Qwen3RMSNorm,           # RMSNorm applied to keys before scoring
    return_entropy: bool = False, # if True, also return mean entropy of softmax weights
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Attend over all block representations + the current partial block.

    Returns a [B, T, D] tensor — the attended aggregation of depth history.
    If return_entropy=True, also returns a scalar entropy value.
    """
    # Stack outside compiled region so torch.compile sees a tensor, not a list
    V = torch.stack(blocks + [partial_block], dim=0)
    query = proj.weight.view(-1)

    if return_entropy:
        kernel = _compiled_block_kernel_entropy or _block_attn_res_kernel_with_entropy
        return kernel(V, query, norm)

    kernel = _compiled_block_kernel or _block_attn_res_kernel
    return kernel(V, query, norm)


def mh_block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, T, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, T, D]
    proj: nn.Linear,              # learned pseudo-query weight  (d,) viewed as (H, d/H)
    norm: Qwen3RMSNorm,           # RMSNorm applied to keys before scoring
    num_heads: int,
) -> torch.Tensor:
    """
    Multi-head block_attn_res: the hidden dim is split into num_heads
    subspaces, each with its own (D/H,) pseudo-query and its own softmax
    over depth sources — different subspaces can route to different layers.
    Same parameter count as block_attn_res (the (D,) query is reshaped).
    """
    if _mhar_fused is not None:
        mctx = _mhar_fused.current_context()
        if mctx is not None:
            return _mhar_fused.route(mctx, blocks, partial_block, proj, norm, num_heads)

    V = torch.stack(blocks + [partial_block], dim=0)
    query = proj.weight.view(num_heads, -1)

    kernel = _compiled_mh_block_kernel or _mh_block_attn_res_kernel
    return kernel(V, query, norm)


def hw_block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, T, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, T, D]
    proj: nn.Linear,              # per-group pseudo-queries, weight (G, D)
    norm: Qwen3RMSNorm,           # RMSNorm applied to keys before scoring
) -> torch.Tensor:
    """
    Head-group-wise block_attn_res: each GQA head group has its own FULL-D
    pseudo-query and its own softmax over depth sources. Returns [B, T, G, D]
    — one routed mix of the depth history per group, consumed by
    Qwen3HeadwiseAttention. Adds (G-1)*D params per call site vs block_attn_res.
    """
    V = torch.stack(blocks + [partial_block], dim=0)
    query = proj.weight  # (G, D)

    kernel = _compiled_hw_block_kernel or _hw_block_attn_res_kernel
    return kernel(V, query, norm)


def _delta_attn_res_kernel(
    V: torch.Tensor,              # pre-stacked sources [N, B, T, D]
    partial_block: torch.Tensor,  # [B, T, D]
    query: torch.Tensor,          # (D,)
    norm: Qwen3RMSNorm,
) -> torch.Tensor:
    """Compiled inner kernel for delta_attn_res (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    selected = torch.einsum("n b t, n b t d -> b t d", weights, V)
    return partial_block + selected


def _delta_attn_res_kernel_with_entropy(
    V: torch.Tensor,
    partial_block: torch.Tensor,
    query: torch.Tensor,
    norm: Qwen3RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled inner kernel for delta_attn_res with entropy (no Python list ops)."""
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", query, K)
    weights = logits.softmax(dim=0)
    selected = torch.einsum("n b t, n b t d -> b t d", weights, V)
    h = partial_block + selected
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
    return h, entropy


def delta_attn_res(
    deltas: list[torch.Tensor],   # previous sublayer outputs (deltas)  [B, T, D] each
    partial_block: torch.Tensor,  # current residual stream  [B, T, D]
    proj: nn.Linear,
    norm: Qwen3RMSNorm,
    null_source: nn.Parameter | None = None,  # learnable null token for identity init
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Attend over previous sublayer deltas and add selected information to
    the current residual stream.

    If null_source is provided, it's prepended as source 0. When softmax puts
    all weight on null_source (zeros), the output is partial_block (identity).
    This enables exact identity initialization for fine-tuning pretrained models.

        h = partial_block + weighted_sum([null_source] + deltas)
        init: null_source=0, proj=0 → uniform weights → Σα·v ≈ 0 → h ≈ partial_block
    """
    if not deltas and null_source is None:
        if return_entropy:
            return partial_block, torch.tensor(0.0, device=partial_block.device)
        return partial_block

    # Build source list outside compiled region
    sources = list(deltas)
    if null_source is not None:
        null_expanded = null_source.unsqueeze(0).unsqueeze(0).expand_as(partial_block)
        sources = [null_expanded] + sources

    if not sources:
        if return_entropy:
            return partial_block, torch.tensor(0.0, device=partial_block.device)
        return partial_block

    # Stack outside compiled region
    V = torch.stack(sources, dim=0)
    query = proj.weight.view(-1)

    if return_entropy:
        kernel = _compiled_delta_kernel_entropy or _delta_attn_res_kernel_with_entropy
        return kernel(V, partial_block, query, norm)

    kernel = _compiled_delta_kernel or _delta_attn_res_kernel
    return kernel(V, partial_block, query, norm)


def gated_delta_attn_res(
    deltas: list[torch.Tensor],   # previous sublayer outputs  [B, T, D] each
    partial_block: torch.Tensor,  # current residual stream  [B, T, D]
    proj: nn.Linear,              # learned query  (d,)
    norm: Qwen3RMSNorm,
    gate_proj: nn.Linear,         # gate projection (d -> d), produces per-source per-dim gate
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Gated Attention Residuals with pre-softmax source gating.

    1. Gate each source: V_gated = sigmoid(gate_proj(partial)) * V
       → filter which source dimensions are useful before routing
    2. Softmax attention over gated sources
    3. Add selected to residual stream

    The gate sees the current hidden state and decides which source
    dimensions to let through, THEN softmax selects among filtered sources.
    """
    if not deltas:
        if return_entropy:
            return partial_block, torch.tensor(0.0, device=partial_block.device)
        return partial_block

    # Stack deltas: (N, B, T, D)
    V = torch.stack(deltas, dim=0)

    # Pre-softmax gate: filter sources based on current hidden state
    # gate: (B, T, D) → broadcast over N sources
    gate = torch.sigmoid(gate_proj(partial_block))  # (B, T, D)
    V_gated = V * gate.unsqueeze(0)                 # (N, B, T, D)

    # Keys from gated values
    K = norm(V_gated)

    # Query
    query = proj.weight.view(-1)                              # (D,)
    logits = torch.einsum("d, n b t d -> n b t", query, K)   # (N, B, T)

    # Softmax over filtered sources
    weights = logits.softmax(dim=0)                           # (N, B, T)

    # Weighted sum of gated values
    selected = torch.einsum("n b t, n b t d -> b t d", weights, V_gated)

    # Add to residual stream
    h = partial_block + selected

    if return_entropy:
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
        return h, entropy

    return h


# ---------------------------------------------------------------------------
# Modified decoder layer
# ---------------------------------------------------------------------------

class Qwen3AttnResDecoderLayer(GradientCheckpointingLayer):
    """
    Qwen3 decoder layer with Block AttnRes depth attention.

    Pseudo-queries are zero-initialized (Kimi, arXiv:2603.15031), so at init
    block_attn_res returns the uniform average of its sources; proj weights
    learn selective routing during training.

    Forward:
        h = block_attn_res(blocks, partial_block)   # uniform avg at init
        attn_out = self_attn(layernorm(h))
        partial_block = partial_block + attn_out     # standard residual add
    """

    def __init__(self, config: Qwen3AttnResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # AttnRes mode
        self.attnres_mode = getattr(config, "attnres_mode", "block")
        self.attnres_num_heads = getattr(config, "attnres_num_heads", 8)
        if self.attnres_mode in ("full_mh", "full_hw", "block_mh") and config.hidden_size % self.attnres_num_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} not divisible by attnres_num_heads {self.attnres_num_heads}")
        if self.attnres_mode == "full_hw" and getattr(config, "attnres_gate_type", "bias") != "bias":
            raise NotImplementedError(
                "full_hw routes per head group before the attn sublayer; only the identity 'bias' gate is supported there")

        # Use V-stream decoupled attention for delta_v and full_v modes
        if self.attnres_mode in ("delta_v", "full_v", "full_split_shared_v"):
            self.self_attn = Qwen3AttnResAttention(config=config, layer_idx=layer_idx)
        elif self.attnres_mode == "full_hw":
            self.self_attn = Qwen3HeadwiseAttention(config=config, layer_idx=layer_idx)
        elif self.attnres_mode == "depth_attn":
            self.self_attn = Qwen3DepthAttention(config=config, layer_idx=layer_idx)
            # Sparse depth sources: layer l mixes layers l, l-s, l-2s, ...  (paper default s=L/2)
            self.depth_stride = max(1, config.num_hidden_layers // 2)
        else:
            self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # Hyper-Connections (Zhu et al., 2024): n parallel residual streams with
        # per-sublayer learned depth (input) and width (output) connections.
        if self.attnres_mode == "hyper_connection":
            n = getattr(config, "attnres_hyper_n", 4)
            self.hyper_n = n
            def _hc_params():
                # alpha: depth-mix over streams (softmax); A: stream-mixing (~identity);
                # beta: width connection (output mainly to stream 0). Small noise breaks
                # stream symmetry so all streams are learnable; init ~ standard residual.
                alpha = nn.Parameter(0.02 * torch.randn(n))
                A = nn.Parameter(torch.eye(n) + 0.02 * torch.randn(n, n))
                beta0 = torch.zeros(n); beta0[0] = 1.0
                beta = nn.Parameter(beta0 + 0.02 * torch.randn(n))
                return alpha, A, beta
            self.hc_alpha_attn, self.hc_A_attn, self.hc_beta_attn = _hc_params()
            self.hc_alpha_mlp,  self.hc_A_mlp,  self.hc_beta_mlp  = _hc_params()

        # AttnRes components — one (proj, norm) per sublayer for Q/K routing.
        # full_hw: one full-D pseudo-query per GQA head group (G, D); the attn
        # sublayer routes a separate depth mix for each group.
        attn_routes = config.num_key_value_heads if self.attnres_mode == "full_hw" else 1
        self.attn_res_proj = nn.Linear(config.hidden_size, attn_routes, bias=False)
        self.attn_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # V-stream: independent routing for value projection
        if self.attnres_mode in ("delta_v", "full_v", "block_v", "delta_block_v", "full_split_shared_v"):
            self.v_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.v_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gate type determines how AttnRes output is mixed with residual stream
        self.gate_type = getattr(config, "attnres_gate_type", "bias")

        if self.gate_type == "sigmoid_scalar":
            # Scalar sigmoid gate: sigmoid(-2) ≈ 0.12 → small initial mixing
            self.attn_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
            self.mlp_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
        elif self.gate_type == "sigmoid_vector":
            # Input-dependent vector gate: per-dim, per-token gating
            self.attn_res_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.attn_res_gate_proj.weight)
            nn.init.constant_(self.attn_res_gate_proj.bias, -2.0)
            self.mlp_res_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.mlp_res_gate_proj.weight)
            nn.init.constant_(self.mlp_res_gate_proj.bias, -2.0)
        elif self.gate_type == "learnable_alpha":
            # Simple learnable scalar: h = (1-α)*partial + α*attnres, init α=0
            self.attn_res_alpha = nn.Parameter(torch.tensor(0.0))
            self.mlp_res_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            # Default "bias": no gate, AttnRes output used directly
            pass

        # Null source for identity init (fine-tuning)
        self.use_null_source = getattr(config, "attnres_use_null_source", False)
        if self.use_null_source:
            # Zero-init: softmax gives uniform weight, null contributes zeros → h ≈ partial
            self.attn_null_source = nn.Parameter(torch.zeros(config.hidden_size))
            self.mlp_null_source = nn.Parameter(torch.zeros(config.hidden_size))

        # Pre-softmax gate for "pre_gated" mode
        if self.attnres_mode == "pre_gated":
            self.attn_pre_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.attn_pre_gate.weight)
            nn.init.constant_(self.attn_pre_gate.bias, 0.0)  # sigmoid(0)=0.5 → pass half
            self.mlp_pre_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.mlp_pre_gate.weight)
            nn.init.constant_(self.mlp_pre_gate.bias, 0.0)

        # Block boundary: how many transformer layers per block (used in block mode)
        num_layers = config.num_hidden_layers
        num_blocks = getattr(config, "attnres_num_blocks", 8)
        self.layers_per_block = max(1, (num_layers + num_blocks - 1) // num_blocks)

    @property
    def is_block_boundary(self) -> bool:
        """True when this layer is the last in its block (0-indexed)."""
        return (self.layer_idx) % self.layers_per_block == 0

    @property
    def is_new_block_start(self) -> bool:
        """True when this is the first layer of a new block (including block 0)."""
        return self.layer_idx % self.layers_per_block == 0

    def _apply_gate(self, hidden_states, h_attn, sublayer: str):
        """Apply gating between residual stream and AttnRes output."""
        if self.gate_type == "sigmoid_scalar":
            logit = self.attn_res_gate_logit if sublayer == "attn" else self.mlp_res_gate_logit
            gate = torch.sigmoid(logit)
            return (1 - gate) * hidden_states + gate * h_attn
        elif self.gate_type == "sigmoid_vector":
            gate_proj = self.attn_res_gate_proj if sublayer == "attn" else self.mlp_res_gate_proj
            gate = torch.sigmoid(gate_proj(hidden_states))  # (B, T, D)
            return (1 - gate) * hidden_states + gate * h_attn
        elif self.gate_type == "learnable_alpha":
            alpha = self.attn_res_alpha if sublayer == "attn" else self.mlp_res_alpha
            return (1 - alpha) * hidden_states + alpha * h_attn
        else:
            # "bias" mode: no gate, AttnRes output used directly
            return h_attn

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        entropy_accum = kwargs.pop("entropy_accum", None)

        if self.attnres_mode == "hyper_connection":
            # ---- Hyper-Connections (Zhu et al., 2024) ----
            # partial_block carries the n streams H: [B, T, n, D]. Each sublayer
            # reads a learned depth-mix as input and writes its output back via a
            # learned width-connection plus stream mixing.
            H = partial_block
            h_in = torch.einsum("r, b t r d -> b t d", self.hc_alpha_attn.softmax(0), H)
            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_in),
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )
            H = torch.einsum("ji, b t i d -> b t j d", self.hc_A_attn, H) \
                + torch.einsum("j, b t d -> b t j d", self.hc_beta_attn, attn_out)
            h_in = torch.einsum("r, b t r d -> b t d", self.hc_alpha_mlp.softmax(0), H)
            mlp_out = self.mlp(self.post_attention_layernorm(h_in))
            H = torch.einsum("ji, b t i d -> b t j d", self.hc_A_mlp, H) \
                + torch.einsum("j, b t d -> b t j d", self.hc_beta_mlp, mlp_out)
            return blocks, H

        if self.attnres_mode == "depth_attn":
            # ---- Depth-Attention (arXiv:2606.05014) ----
            # Standard residual stream + standard MLP; the only change is that the
            # attention's value is depth-mixed across earlier layers (Eq 2-4).
            # `blocks` carries the depth cache: a list of (k_raw_j, v_mixed_j).
            depth_cache = blocks
            residual = partial_block
            attn_out, k_l, v_mixed_l = self.self_attn(
                hidden_states=self.input_layernorm(partial_block),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                depth_cache=depth_cache,
                depth_stride=self.depth_stride,
            )
            partial_block = residual + attn_out
            depth_cache = depth_cache + [(k_l, v_mixed_l)]

            # MLP sublayer (standard)
            mlp_out = self.mlp(self.post_attention_layernorm(partial_block))
            partial_block = partial_block + mlp_out

            return depth_cache, partial_block

        if self.attnres_mode == "pre_gated":
            # ---- Pre-gated delta: gate sources BEFORE softmax ----
            # Same source collection as delta mode, but uses gated_delta_attn_res

            # Attention sublayer
            h = gated_delta_attn_res(blocks, partial_block,
                                     self.attn_res_proj, self.attn_res_norm, self.attn_pre_gate)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            h = gated_delta_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm, self.mlp_pre_gate)

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "first_layer":
            # ---- First Layer Residual: softmax attention over [partial, mlp_out_0] ----
            # Like a 2-source AttnRes with learned query
            if len(blocks) >= 3:
                first_layer_mlp = blocks[2]  # mlp_out_0
            else:
                first_layer_mlp = None

            # Attention sublayer
            if first_layer_mlp is not None:
                V = torch.stack([partial_block, first_layer_mlp], dim=0)  # (2, B, T, D)
                K = self.attn_res_norm(V)
                query = self.attn_res_proj.weight.view(-1)
                logits = torch.einsum("d, n b t d -> n b t", query, K)  # (2, B, T)
                weights = logits.softmax(dim=0)
                h = torch.einsum("n b t, n b t d -> b t d", weights, V)
            else:
                h = partial_block

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            if first_layer_mlp is not None:
                V = torch.stack([partial_block, first_layer_mlp], dim=0)
                K = self.mlp_res_norm(V)
                query = self.mlp_res_proj.weight.view(-1)
                logits = torch.einsum("d, n b t d -> n b t", query, K)
                weights = logits.softmax(dim=0)
                h = torch.einsum("n b t, n b t d -> b t d", weights, V)
            else:
                h = partial_block

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "delta_block":
            # ---- Delta-Block mode: delta_attn_res with block-aggregated sources ----
            # blocks = completed block delta sums (managed by model forward)
            # Layer does NOT modify blocks; model forward handles accumulation.
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.attn_res_proj, self.attn_res_norm, attn_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )


            if self.is_block_boundary:
                if not blocks:
                    blocks = blocks + [partial_block]
                else:
                    blocks = blocks + [partial_block - blocks[-1]]

            partial_block = partial_block + attn_out

            # MLP sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.mlp_res_proj, self.mlp_res_norm, mlp_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            # blocks unchanged — model forward handles block delta accumulation
            return blocks, partial_block

        if self.attnres_mode == "delta_block_v":
            # ---- Delta-Block-V mode: delta_block + per-stream V routing ----
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer — two delta_attn_res calls (QK and V)
            h_qk = delta_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm, attn_null)
            h_v = delta_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm, attn_null)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out

            # MLP sublayer — single routing
            h_attn = delta_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            # blocks unchanged — model forward handles block delta accumulation
            return blocks, partial_block

        if self.attnres_mode == "delta":
            # ---- Delta mode: attend over previous sublayer outputs (deltas) ----
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.attn_res_proj, self.attn_res_norm, attn_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if not blocks:
                blocks = blocks + [partial_block]
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            if entropy_accum is not None:
                h_attn, ent = delta_attn_res(blocks, partial_block,
                                             self.mlp_res_proj, self.mlp_res_norm, mlp_null,
                                             return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = delta_attn_res(blocks, partial_block,
                                        self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "delta_v":
            # ---- Delta-V mode: V-stream gets independent depth routing ----
            # Q/K use attn_res_proj routing; V uses v_res_proj routing.
            # MLP sublayer is unchanged from delta mode.
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer — two parallel delta_attn_res calls
            h_qk = delta_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm, attn_null)
            h_v = delta_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm, attn_null)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer (same as delta — no V-stream decoupling needed)
            h_attn = delta_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "full_v":
            # ---- Full-V mode: cumulative states + per-stream V routing ----
            # Like full mode but with independent V-stream depth routing.

            # Attention sublayer — two block_attn_res calls (QK and V)
            h_qk = block_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm)
            h_v = block_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [partial_block]  # cumulative state after attn

            # MLP sublayer — single routing (same as full mode)
            h_attn = block_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [partial_block]  # cumulative state after mlp

            return blocks, partial_block

        if self.attnres_mode == "full_split":
            # ---- Full-split mode: two independent residual towers ----
            # Same per-sublayer replacement routing as `full`, but attn and MLP
            # each keep their OWN source list and partial carrier. Attn routing
            # pools only over past attn outputs; MLP routing only over past MLP
            # outputs. The two streams never mix until the final routing in the
            # model forward.
            #   state: blocks = (attn_blocks, mlp_blocks)
            #          partial_block = (partial_attn, partial_mlp)
            attn_blocks, mlp_blocks = blocks
            partial_a, partial_m = partial_block

            # Attention tower
            h_a = block_attn_res(attn_blocks, partial_a,
                                 self.attn_res_proj, self.attn_res_norm)
            h_a = self._apply_gate(partial_a, h_a, "attn")
            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_a),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            attn_blocks = attn_blocks + [partial_a]
            partial_a = attn_out

            # MLP tower
            h_m = block_attn_res(mlp_blocks, partial_m,
                                 self.mlp_res_proj, self.mlp_res_norm)
            h_m = self._apply_gate(partial_m, h_m, "mlp")
            mlp_out = self.mlp(self.post_attention_layernorm(h_m))
            mlp_blocks = mlp_blocks + [partial_m]
            partial_m = mlp_out

            return (attn_blocks, mlp_blocks), (partial_a, partial_m)

        if self.attnres_mode == "full_split_shared":
            # ---- Full-split-shared: split routing pools, ONE shared stream ----
            # Like `delta` (additive routing over a cumulative residual stream),
            # but the routing pool is split by type: attn routing pools only over
            # past attn outputs, MLP routing only over past MLP outputs. Crucially
            # there is ONE shared cumulative `partial_block`, so attn and MLP still
            # couple through the stream (unlike `full_split`'s severed towers).
            #   state: blocks = (attn_blocks, mlp_blocks); partial_block = tensor
            attn_blocks, mlp_blocks = blocks
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer — additive routing over attn history only
            h = delta_attn_res(attn_blocks, partial_block,
                               self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h, "attn")
            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            attn_blocks = attn_blocks + [attn_out]

            # MLP sublayer — additive routing over MLP history only
            h = delta_attn_res(mlp_blocks, partial_block,
                               self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h, "mlp")
            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            mlp_blocks = mlp_blocks + [mlp_out]

            return (attn_blocks, mlp_blocks), partial_block

        if self.attnres_mode == "full_split_shared_v":
            # ---- full_split_shared + VALUE RESIDUAL ----
            # Same as full_split_shared (split routing pools, one shared cumulative
            # stream), but the attention's V projection gets its OWN depth-routing:
            # Q/K read one depth-aggregated input (attn_res_proj), V reads a
            # separately-routed input (v_res_proj) — both over the attn history.
            attn_blocks, mlp_blocks = blocks
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer — Q/K and V route over attn history independently
            h_qk = delta_attn_res(attn_blocks, partial_block,
                                  self.attn_res_proj, self.attn_res_norm, attn_null)
            h_v = delta_attn_res(attn_blocks, partial_block,
                                 self.v_res_proj, self.v_res_norm, attn_null)
            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")
            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out
            attn_blocks = attn_blocks + [attn_out]

            # MLP sublayer — additive routing over MLP history only (unchanged)
            h = delta_attn_res(mlp_blocks, partial_block,
                               self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h, "mlp")
            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            mlp_blocks = mlp_blocks + [mlp_out]

            return (attn_blocks, mlp_blocks), partial_block

        if self.attnres_mode == "block":
            # ---- Block mode (Kimi-faithful) ----
            # block_attn_res sees old partial BEFORE boundary reset.
            # At new-block start: store old partial in blocks, reset partial.
            # partial_block tracks intra-block accumulation only.

            # Attention sublayer — uses old partial
            if entropy_accum is not None:
                h_attn, ent = block_attn_res(blocks, partial_block,
                                             self.attn_res_proj, self.attn_res_norm, return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = block_attn_res(blocks, partial_block,
                                        self.attn_res_proj, self.attn_res_norm)
            h = self._apply_gate(partial_block, h_attn, "attn")

            # New block boundary: store old partial, reset
            if self.is_new_block_start:
                blocks = blocks + [partial_block]
                partial_block = torch.zeros_like(partial_block)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out

            # MLP sublayer
            if entropy_accum is not None:
                h_attn, ent = block_attn_res(blocks, partial_block,
                                             self.mlp_res_proj, self.mlp_res_norm, return_entropy=True)
                entropy_accum.append(ent)
            else:
                h_attn = block_attn_res(blocks, partial_block,
                                        self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            return blocks, partial_block

        if self.attnres_mode == "block_mh":
            # ---- Multi-head block mode: block-grouped sources (few) + per-subspace routing ----
            # Identical to "block" (Kimi-faithful block routing) but the (D,) query is
            # reshaped into H heads, so each d/H subspace routes independently over the
            # cheap block source list. Parameter-free vs single-head block; far fewer
            # sources than full_mh, so much closer to baseline throughput.
            h_attn = mh_block_attn_res(blocks, partial_block,
                                       self.attn_res_proj, self.attn_res_norm,
                                       self.attnres_num_heads)
            h = self._apply_gate(partial_block, h_attn, "attn")

            # New block boundary: store old partial, reset
            if self.is_new_block_start:
                blocks = blocks + [partial_block]
                partial_block = torch.zeros_like(partial_block)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out

            # MLP sublayer
            h_attn = mh_block_attn_res(blocks, partial_block,
                                       self.mlp_res_proj, self.mlp_res_norm,
                                       self.attnres_num_heads)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            return blocks, partial_block

        if self.attnres_mode == "block_v":
            # ---- Block-V mode: block mode + per-stream V routing ----

            # Attention sublayer — two block_attn_res calls (QK and V)
            h_qk = block_attn_res(blocks, partial_block,
                                   self.attn_res_proj, self.attn_res_norm)
            h_v = block_attn_res(blocks, partial_block,
                                  self.v_res_proj, self.v_res_norm)

            h_qk = self._apply_gate(partial_block, h_qk, "attn")
            h_v_gated = self._apply_gate(partial_block, h_v, "attn")

            # New block boundary: store old partial, reset
            if self.is_new_block_start:
                blocks = blocks + [partial_block]
                partial_block = torch.zeros_like(partial_block)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_qk),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                value_hidden_states=self.input_layernorm(h_v_gated),
            )
            partial_block = partial_block + attn_out

            # MLP sublayer — single routing (same as block mode)
            h_attn = block_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

            return blocks, partial_block

        if self.attnres_mode == "delta_replace":
            # ---- Delta Replace mode: replacement routing + delta sources ----
            # Uses block_attn_res (h = weighted_sum) with delta sources (attn_out, mlp_out).

            # Attention sublayer
            h_attn = block_attn_res(blocks, partial_block,
                                    self.attn_res_proj, self.attn_res_norm)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [attn_out]

            # MLP sublayer
            h_attn = block_attn_res(blocks, partial_block,
                                    self.mlp_res_proj, self.mlp_res_norm)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [mlp_out]

            return blocks, partial_block

        if self.attnres_mode == "full_additive":
            # ---- Full Additive mode: additive routing + cumulative sources ----
            # Uses delta_attn_res (h = partial + weighted_sum) with cumulative
            # partial_block sources (like full mode).
            attn_null = self.attn_null_source if self.use_null_source else None
            mlp_null = self.mlp_null_source if self.use_null_source else None

            # Attention sublayer
            h_attn = delta_attn_res(blocks, partial_block,
                                    self.attn_res_proj, self.attn_res_norm, attn_null)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            partial_block = partial_block + attn_out
            blocks = blocks + [partial_block]  # cumulative state

            # MLP sublayer
            h_attn = delta_attn_res(blocks, partial_block,
                                    self.mlp_res_proj, self.mlp_res_norm, mlp_null)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out
            blocks = blocks + [partial_block]  # cumulative state

            return blocks, partial_block

        if self.attnres_mode == "full_hw":
            # ---- Full-HW mode: head-group-wise depth routing ----
            # Each GQA head group has its own full-D pseudo-query and softmax
            # over sources; group g's Q/K/V are projected from its own routed
            # mix, so head groups specialize over depth. MLP sublayer and
            # final routing stay identical to full_mh (subspace multi-head),
            # isolating the attention-side change.
            h_grp = hw_block_attn_res(blocks, partial_block,
                                      self.attn_res_proj, self.attn_res_norm)

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h_grp),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            blocks = blocks + [partial_block]
            partial_block = attn_out

            # MLP sublayer — same as full_mh
            h_attn = mh_block_attn_res(blocks, partial_block,
                                       self.mlp_res_proj, self.mlp_res_norm,
                                       self.attnres_num_heads)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            blocks = blocks + [partial_block]
            partial_block = mlp_out

            return blocks, partial_block

        if self.attnres_mode == "full_mh":
            # ---- Full-MH mode: full mode with multi-head depth routing ----
            # Hidden dim split into H subspaces; each head gets its own query
            # and softmax over sources, so subspaces route independently.
            h_attn = mh_block_attn_res(blocks, partial_block,
                                       self.attn_res_proj, self.attn_res_norm,
                                       self.attnres_num_heads)
            h = self._apply_gate(partial_block, h_attn, "attn")

            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(h),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            blocks = blocks + [partial_block]
            partial_block = attn_out

            # MLP sublayer
            h_attn = mh_block_attn_res(blocks, partial_block,
                                       self.mlp_res_proj, self.mlp_res_norm,
                                       self.attnres_num_heads)
            h = self._apply_gate(partial_block, h_attn, "mlp")

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            blocks = blocks + [partial_block]
            partial_block = mlp_out

            return blocks, partial_block

        # ---- Full mode (delta sources + replacement routing + final routing) ----

        # Attention sublayer
        h_attn = block_attn_res(blocks, partial_block,
                                self.attn_res_proj, self.attn_res_norm)
        h = self._apply_gate(partial_block, h_attn, "attn")

        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(h),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        blocks = blocks + [partial_block]
        partial_block = attn_out

        # MLP sublayer
        h_attn = block_attn_res(blocks, partial_block,
                                self.mlp_res_proj, self.mlp_res_norm)
        h = self._apply_gate(partial_block, h_attn, "mlp")

        mlp_out = self.mlp(self.post_attention_layernorm(h))
        blocks = blocks + [partial_block]
        partial_block = mlp_out

        return blocks, partial_block


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class Qwen3AttnResModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Block AttnRes via recency-biased depth attention."""

    config_class = Qwen3AttnResConfig

    def _init_weights(self, module):
        """Override to preserve AttnRes initialization."""
        super()._init_weights(module)
        if isinstance(module, Qwen3AttnResDecoderLayer):
            gate_type = getattr(self.config, "attnres_gate_type", "bias")
            # Guard every bare-scalar init: transformers 5.x re-applies
            # _init_weights over bare Parameters after checkpoint loading on the
            # direct-class from_pretrained path (submodules are marked loaded and
            # skipped; bare Parameters are not), silently resetting trained gates.
            def _fresh(p):
                return not getattr(p, "_is_hf_initialized", False)
            # Kimi (arXiv:2603.15031): pseudo-query vectors MUST be zero-init so
            # routing starts as a uniform average over sources; random init makes
            # the depth softmax arbitrarily peaked at step 0 (training volatility).
            # Runs after the generic Linear init above, which would otherwise
            # leave these at normal(0, initializer_range).
            # ATTNRES_RANDOM_QUERY_INIT=1 keeps the legacy random init (control
            # experiments only — reproduces the pre-2026-07-19 behavior).
            if os.environ.get("ATTNRES_RANDOM_QUERY_INIT") != "1":
                for lin in (module.attn_res_proj, module.mlp_res_proj,
                            getattr(module, "v_res_proj", None)):
                    if lin is not None and _fresh(lin.weight):
                        lin.weight.data.zero_()
            if gate_type == "sigmoid_scalar":
                if _fresh(module.attn_res_gate_logit):
                    module.attn_res_gate_logit.data.fill_(-2.0)
                if _fresh(module.mlp_res_gate_logit):
                    module.mlp_res_gate_logit.data.fill_(-2.0)
            elif gate_type == "sigmoid_vector":
                nn.init.zeros_(module.attn_res_gate_proj.weight)
                nn.init.constant_(module.attn_res_gate_proj.bias, -2.0)
                nn.init.zeros_(module.mlp_res_gate_proj.weight)
                nn.init.constant_(module.mlp_res_gate_proj.bias, -2.0)
            elif gate_type == "learnable_alpha":
                if _fresh(module.attn_res_alpha):
                    module.attn_res_alpha.data.fill_(0.0)
                if _fresh(module.mlp_res_alpha):
                    module.mlp_res_alpha.data.fill_(0.0)
        if isinstance(module, Qwen3AttnResModel) and os.environ.get("ATTNRES_RANDOM_QUERY_INIT") != "1":
            final_proj = getattr(module, "final_res_proj", None)
            if final_proj is not None and not getattr(final_proj.weight, "_is_hf_initialized", False):
                final_proj.weight.data.zero_()

    def __init__(self, config: Qwen3AttnResConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3AttnResDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Final block_attn_res: produces effective hidden state after all layers
        # by routing over all sources + last partial. Needed for any mode where
        # partial_block doesn't carry the full cumulative state.
        self._attnres_mode = getattr(config, "attnres_mode", "block")
        self._needs_final_routing = self._attnres_mode in (
            "block", "block_v", "full", "full_mh", "full_hw", "full_split", "delta_centered_reset",
        )
        if self._needs_final_routing:
            self.final_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.final_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Block AttnRes state: list of completed block tensors + current partial
        # The token embedding acts as the first "block" (block 0).
        attnres_mode = getattr(self.config, "attnres_mode", "block")
        is_delta_block = attnres_mode in ("delta_block", "delta_block_v")

        if attnres_mode == "full_split":
            # full_split: two independent residual towers.
            #   blocks = (attn_blocks, mlp_blocks); partial = (partial_attn, partial_mlp)
            blocks = ([], [])
            partial_block = (inputs_embeds, inputs_embeds)
        elif attnres_mode in ("full_split_shared", "full_split_shared_v"):
            # full_split_shared(_v): split routing pools, ONE shared cumulative stream.
            #   blocks = (attn_blocks, mlp_blocks); partial_block = tensor (cumulative)
            blocks = ([], [])
            partial_block = inputs_embeds
        elif is_delta_block:
            # delta_block: sources are embedding + block-aggregated deltas
            blocks: list[torch.Tensor] = []
            block_start_partial: torch.Tensor = torch.zeros_like(inputs_embeds)
            partial_block: torch.Tensor = inputs_embeds
        elif attnres_mode == "hyper_connection":
            # hyper-connection: replicate embedding into n parallel streams [B,T,n,D]
            blocks: list[torch.Tensor] = []
            n_hc = getattr(self.config, "attnres_hyper_n", 4)
            partial_block: torch.Tensor = inputs_embeds.unsqueeze(2).expand(-1, -1, n_hc, -1).contiguous()
        else:
            # block/full modes: layer 0 stores embed at boundary, so start empty
            blocks: list[torch.Tensor] = []
            partial_block: torch.Tensor = inputs_embeds

        # Entropy accumulation for auxiliary loss
        entropy_lambda = kwargs.pop("entropy_lambda", 0.0)
        entropy_accum = [] if entropy_lambda > 0 else None

        # Fused Triton routing (full_mh): one shared source buffer for the whole
        # forward instead of per-sublayer torch.stack. Not used under gradient
        # checkpointing (recomputation would re-append rows out of order).
        _mhar_ctx = None
        if (
            attnres_mode == "full_mh"
            and _mhar_fused is not None
            and _mhar_fused.fused_mhar_enabled()
            and inputs_embeds.is_cuda
            and not (self.gradient_checkpointing and self.training)
        ):
            _mhar_ctx = _mhar_fused.MHARFusedContext(
                2 * len(self.layers) + 1, inputs_embeds)
            _mhar_fused.set_context(_mhar_ctx)
        try:
            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    blocks, partial_block = self._gradient_checkpointing_func(
                        layer.__call__,
                        blocks,
                        partial_block,
                        causal_mask_mapping[layer.attention_type],
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    blocks, partial_block = layer(
                        blocks=blocks,
                        partial_block=partial_block,
                        attention_mask=causal_mask_mapping[layer.attention_type],
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        entropy_accum=entropy_accum,
                    )

                # delta_block: accumulate block deltas at block boundaries
                # if is_delta_block and layer.is_block_boundary:
                #     block_delta = partial_block - block_start_partial
                #     blocks = blocks + [block_delta]
                #     block_start_partial = partial_block

            # Final routing: produce effective hidden state from all sources.
            # Needed for modes where partial_block doesn't carry the full state.
            if attnres_mode == "full_split":
                # Combine the two towers: route over all attn + mlp sources at once.
                attn_blocks, mlp_blocks = blocks
                partial_a, partial_m = partial_block
                all_sources = attn_blocks + mlp_blocks + [partial_a]
                partial_block = block_attn_res(
                    all_sources, partial_m,
                    self.final_res_proj, self.final_res_norm,
                )
            elif attnres_mode in ("full_mh", "full_hw"):
                partial_block = mh_block_attn_res(
                    blocks, partial_block,
                    self.final_res_proj, self.final_res_norm,
                    getattr(self.config, "attnres_num_heads", 8),
                )
            elif self._needs_final_routing:
                partial_block = block_attn_res(
                    blocks, partial_block,
                    self.final_res_proj, self.final_res_norm,
                )
            elif attnres_mode == "hyper_connection":
                # combine the n streams into the final representation
                partial_block = partial_block.mean(dim=2)
        finally:
            if _mhar_ctx is not None:
                _mhar_fused.set_context(None)

        hidden_states = self.norm(partial_block)

        # Compute mean entropy across all sublayers
        attnres_entropy = None
        if entropy_accum:
            attnres_entropy = torch.stack(entropy_accum).mean()

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        # Attach entropy as extra attribute
        out.attnres_entropy = attnres_entropy
        return out


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class Qwen3AttnResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal LM with Block AttnRes residuals."""

    config_class = Qwen3AttnResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3AttnResConfig):
        super().__init__(config)
        self.model = Qwen3AttnResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            # Entropy bonus: encourage diverse cross-layer attention
            entropy_lambda = kwargs.get("entropy_lambda", 0.0)
            attnres_entropy = getattr(outputs, "attnres_entropy", None)
            if entropy_lambda > 0 and attnres_entropy is not None:
                # Subtract entropy (negative because we want to maximize it)
                loss = loss - entropy_lambda * attnres_entropy

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

