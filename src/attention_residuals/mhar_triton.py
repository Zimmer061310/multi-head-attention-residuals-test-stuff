"""
Fused Triton kernels for MHAR (full_mh) multi-head depth routing.

The eager path in modeling_qwen3_attnres.mh_block_attn_res does, per sublayer s
with N_s = s+1 depth sources:

    V = torch.stack(blocks + [partial])          # O(N) full copy
    K = RMSNorm(V)                               # materializes K [N,B,T,D]
    logits = einsum("hk,nbthk->nbth", q, K)
    w = logits.softmax(0)
    out = einsum("nbth,nbthk->bthk", w, V)

Summed over the 2L+1 sublayers this is O(L^2 * B*T*D) memory traffic, done in
~6 separate kernels forward + a longer chain backward, and it keeps every
stacked V alive for backward (the dominant activation-memory cost of MHAR).

This module replaces it with:

  * one persistent source buffer per model forward ([2L+1, B, T, D], each new
    residual state written exactly once — no per-sublayer stacking),
  * one Triton kernel per routing call forward (online softmax over sources,
    RMSNorm fused, nothing materialized except the routing weights
    [N, B, T, H] which are ~D/H smaller than V),
  * one Triton kernel per routing call backward (recomputes the norm from the
    buffer, produces dV for all sources and accumulates it into a shared fp32
    gradient buffer in-place; per-source grads are handed back to autograd
    only for the single row owned by this call's `partial` input).

The in-place dbuf accumulation is safe because autograd runs the routing
calls' backwards in strictly reverse forward order: every later call is a
graph descendant of this call's output (out_s feeds partial_{s+1} feeds
call s+1), so by the time call s's backward runs, all contributions from
calls s' > s to rows 0..N_s-1 are already in dbuf. This is asserted at
runtime (`begin_backward`).

Enable with ATTNRES_FUSED=1 or programmatically via enable_fused_mhar().
Optional ATTNRES_FUSED_CHECK=1 cross-checks every fused forward against the
eager reference (slow; for debugging).
"""

import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only envs
    HAS_TRITON = False

_ENABLED = os.environ.get("ATTNRES_FUSED", "0") == "1"
_CHECK = os.environ.get("ATTNRES_FUSED_CHECK", "0") == "1"
_CURRENT_CTX = None


def enable_fused_mhar(enabled: bool = True):
    global _ENABLED
    _ENABLED = enabled


def fused_mhar_enabled() -> bool:
    return _ENABLED and HAS_TRITON


def current_context():
    return _CURRENT_CTX


def set_context(ctx):
    global _CURRENT_CTX
    _CURRENT_CTX = ctx


if HAS_TRITON:

    @triton.jit
    def _mhar_fwd_kernel(
        V,            # bf16/fp32 [>=N, BT, D] source buffer (row-contiguous)
        OUT,          # [BT, D] routed mix, dtype of V
        W,            # fp32 [N, BT, H] routing weights (written)
        Q,            # (D,) per-head pseudo-queries, head h owns [h*K, (h+1)*K)
        G,            # (D,) RMSNorm weight
        PARTIAL,      # [BT, D] residual stream; added to the mix iff ADD_PARTIAL
        N, BT, D, H, K,
        stride_vn,    # row stride of V (= BT*D)
        EPS, INV_D,
        ADD_PARTIAL: tl.constexpr,   # delta modes: out = partial + mix
        HB: tl.constexpr, KB: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs_h = tl.arange(0, HB)
        offs_k = tl.arange(0, KB)
        hmask = offs_h < H
        mask2 = hmask[:, None] & (offs_k[None, :] < K)
        offd = offs_h[:, None] * K + offs_k[None, :]

        q = tl.load(Q + offd, mask=mask2, other=0.0).to(tl.float32)
        g = tl.load(G + offd, mask=mask2, other=0.0).to(tl.float32)

        vbase = V + pid * D
        wbase = W + pid * H + offs_h
        m = tl.full((HB,), -1e30, tl.float32)
        l = tl.zeros((HB,), tl.float32)
        acc = tl.zeros((HB, KB), tl.float32)
        for n in range(N):
            n64 = n.to(tl.int64)
            v = tl.load(vbase + n64 * stride_vn + offd, mask=mask2, other=0.0).to(tl.float32)
            ss = tl.sum(tl.sum(v * v, axis=1), axis=0) * INV_D
            r = tl.rsqrt(ss + EPS)
            # match HF RMSNorm: normalize in fp32, round to input dtype, then * weight
            khat = (v * r).to(V.dtype.element_ty).to(tl.float32) * g
            logit = tl.sum(q * khat, axis=1)
            logit = tl.where(hmask, logit, -1e30)
            tl.store(wbase + n64 * BT * H, logit, mask=hmask)  # raw logit; normalized below
            mn = tl.maximum(m, logit)
            alpha = tl.exp(m - mn)
            p = tl.exp(logit - mn)
            l = l * alpha + p
            acc = acc * alpha[:, None] + p[:, None] * v
            m = mn
        out = acc / l[:, None]
        if ADD_PARTIAL:
            pb = tl.load(PARTIAL + pid * D + offd, mask=mask2, other=0.0).to(tl.float32)
            out = out + pb
        tl.store(OUT + pid * D + offd, out.to(OUT.dtype.element_ty), mask=mask2)
        # exact softmax weights from the raw logits (tiny: N*H per position)
        for n in range(N):
            n64 = n.to(tl.int64)
            lg = tl.load(wbase + n64 * BT * H, mask=hmask, other=0.0)
            tl.store(wbase + n64 * BT * H, tl.exp(lg - m) / l, mask=hmask)

    @triton.jit
    def _mhar_bwd_kernel(
        V,            # [>=N, BT, D] source buffer
        W,            # fp32 [N, BT, H] routing weights from forward
        DOUT,         # [BT, D]
        DBUF,         # fp32 [>=N, BT, D] source-grad accumulator
        DQP,          # fp32 [BT, D] per-position dq partials (summed on host)
        DGP,          # fp32 [BT, D] per-position dg partials
        Q, G,
        N, BT, D, H, K,
        stride_vn,
        EPS, INV_D,
        ACCUM: tl.constexpr,   # rmw dbuf (later calls) vs pure store (first bwd call)
        HB: tl.constexpr, KB: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs_h = tl.arange(0, HB)
        offs_k = tl.arange(0, KB)
        hmask = offs_h < H
        mask2 = hmask[:, None] & (offs_k[None, :] < K)
        offd = offs_h[:, None] * K + offs_k[None, :]

        q = tl.load(Q + offd, mask=mask2, other=0.0).to(tl.float32)
        g = tl.load(G + offd, mask=mask2, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + pid * D + offd, mask=mask2, other=0.0).to(tl.float32)

        vbase = V + pid * D
        wbase = W + pid * H + offs_h
        dbase = DBUF + pid * D + offd

        # pass 1: S_h = sum_n w * <dout_h, v_h>  (softmax-backward coupling term)
        S = tl.zeros((HB,), tl.float32)
        for n in range(N):
            n64 = n.to(tl.int64)
            v = tl.load(vbase + n64 * stride_vn + offd, mask=mask2, other=0.0).to(tl.float32)
            w = tl.load(wbase + n64 * BT * H, mask=hmask, other=0.0)
            S += w * tl.sum(dout * v, axis=1)

        # pass 2: per-source grads + dq/dg accumulation
        dq_acc = tl.zeros((HB, KB), tl.float32)
        dg_acc = tl.zeros((HB, KB), tl.float32)
        for n in range(N):
            n64 = n.to(tl.int64)
            v = tl.load(vbase + n64 * stride_vn + offd, mask=mask2, other=0.0).to(tl.float32)
            w = tl.load(wbase + n64 * BT * H, mask=hmask, other=0.0)
            dw = tl.sum(dout * v, axis=1)
            dl = w * (dw - S)                      # dlogits
            ss = tl.sum(tl.sum(v * v, axis=1), axis=0) * INV_D
            r = tl.rsqrt(ss + EPS)
            khat_pre = (v * r).to(V.dtype.element_ty).to(tl.float32)
            dq_acc += dl[:, None] * (khat_pre * g)
            dK = dl[:, None] * q
            dg_acc += dK * khat_pre
            dkp = dK * g                            # grad wrt v*r (norm cast treated as identity)
            srow = tl.sum(tl.sum(dkp * v, axis=1), axis=0)
            dv = r * dkp - (r * r * r * INV_D) * srow * v + w[:, None] * dout
            ptr = dbase + n64 * stride_vn
            if ACCUM:
                dv += tl.load(ptr, mask=mask2, other=0.0)
            tl.store(ptr, dv, mask=mask2)
        tl.store(DQP + pid * D + offd, dq_acc, mask=mask2)
        tl.store(DGP + pid * D + offd, dg_acc, mask=mask2)


def _next_pow2(x: int) -> int:
    return max(1, 1 << (x - 1).bit_length())


def _launch_cfg(H: int, K: int):
    HB, KB = _next_pow2(H), _next_pow2(K)
    tile = HB * KB
    if tile > 8192:
        raise RuntimeError(
            f"MHAR fused: routing tile {HB}x{KB} too large for a register-resident "
            f"kernel (H={H}, D/H={K}); use the eager path for this shape")
    num_warps = 8 if tile >= 4096 else (4 if tile >= 1024 else 2)
    return HB, KB, num_warps


def _check_dims(D: int, num_heads: int):
    if num_heads < 1 or D % num_heads != 0:
        raise RuntimeError(
            f"MHAR fused: hidden size {D} not divisible by routing heads {num_heads}")


class MHARFusedContext:
    """Per-model-forward state: the shared source buffer and (in backward)
    the shared fp32 source-grad accumulator."""

    def __init__(self, n_max: int, ref: torch.Tensor):
        B, T, D = ref.shape
        self.B, self.T, self.D = B, T, D
        self.BT = B * T
        self.n_max = n_max
        self.buf = torch.empty((n_max, B, T, D), device=ref.device, dtype=ref.dtype)
        self.n = 0
        self.dbuf = None
        self._bwd_expect = None

    def append(self, partial: torch.Tensor) -> int:
        if partial.shape != (self.B, self.T, self.D):
            raise RuntimeError(
                f"MHAR fused: source shape {tuple(partial.shape)} != {(self.B, self.T, self.D)}")
        if self.n >= self.n_max:
            raise RuntimeError("MHAR fused: more routing calls than n_max — context desync")
        row = self.n
        with torch.no_grad():
            self.buf[row].copy_(partial)
        self.n += 1
        return row

    def begin_backward(self, N: int) -> bool:
        """Returns True if this is the first backward call (dbuf uninitialized:
        the kernel pure-stores instead of read-modify-write)."""
        first = self.dbuf is None
        if first:
            if N != self.n:
                raise RuntimeError(
                    f"MHAR fused: first backward call has N={N}, expected {self.n}. "
                    "Backward must run in reverse forward order — disable fused mode "
                    "(e.g. under gradient checkpointing / partial backward).")
            self.dbuf = torch.empty(
                (self.n, self.B, self.T, self.D), device=self.buf.device, dtype=torch.float32)
            self._bwd_expect = self.n
        if N != self._bwd_expect:
            raise RuntimeError(
                f"MHAR fused: backward order violation (got N={N}, expected {self._bwd_expect}). "
                "Disable fused mode for this use case.")
        self._bwd_expect = N - 1
        return first


class _FusedMHARFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, q_weight, g_weight, mctx, num_heads, eps):
        row = mctx.append(partial.contiguous())
        N = row + 1
        D, BT, H = mctx.D, mctx.BT, num_heads
        _check_dims(D, H)
        K = D // H
        out = torch.empty_like(partial, memory_format=torch.contiguous_format)
        w = torch.empty((N, mctx.BT, H), device=partial.device, dtype=torch.float32)
        HB, KB, num_warps = _launch_cfg(H, K)
        _mhar_fwd_kernel[(BT,)](
            mctx.buf, out, w, q_weight, g_weight, out,  # PARTIAL unused (dummy ptr)
            N, BT, D, H, K, mctx.BT * D, eps, 1.0 / D,
            ADD_PARTIAL=False, HB=HB, KB=KB, num_warps=num_warps,
        )
        # mctx.buf is deliberately NOT in save_for_backward: later routing calls
        # append rows in place (bumping its version), but rows [0, N) that this
        # call reads are immutable once written.
        ctx.save_for_backward(w, q_weight, g_weight)
        ctx.mctx = mctx
        ctx.meta = (N, H, K, eps)
        ctx.q_shape = q_weight.shape
        return out

    @staticmethod
    def backward(ctx, dout):
        w, q_weight, g_weight = ctx.saved_tensors
        mctx = ctx.mctx
        N, H, K, eps = ctx.meta
        first = mctx.begin_backward(N)
        D, BT = mctx.D, mctx.BT
        dout = dout.contiguous()
        dqp = torch.empty((BT, D), device=dout.device, dtype=torch.float32)
        dgp = torch.empty((BT, D), device=dout.device, dtype=torch.float32)
        HB, KB, num_warps = _launch_cfg(H, K)
        _mhar_bwd_kernel[(BT,)](
            mctx.buf, w, dout, mctx.dbuf, dqp, dgp, q_weight, g_weight,
            N, BT, D, H, K, mctx.BT * D, eps, 1.0 / D,
            ACCUM=not first, HB=HB, KB=KB, num_warps=num_warps,
        )
        dq = dqp.sum(dim=0).to(q_weight.dtype).view(ctx.q_shape)
        dg = dgp.sum(dim=0).to(g_weight.dtype)
        # this call's `partial` is source row N-1; all contributions from later
        # calls are already accumulated (backward runs in reverse forward order)
        dpartial = mctx.dbuf[N - 1].to(dout.dtype)
        return dpartial, dq, dg, None, None, None


class _FusedDeltaMHFn(torch.autograd.Function):
    """Self-contained fused mh_delta_attn_res: out = partial + routed mix over
    [null?] + delta sources. No cross-call state (per-call stacked V, per-call
    dV), so it composes with FSDP and gradient checkpointing; source counts in
    delta modes are small (num_blocks+1), so the per-call stack is cheap."""

    @staticmethod
    def forward(ctx, partial, q_weight, g_weight, null_source, num_heads, eps, *deltas):
        B, T, D = partial.shape
        BT = B * T
        srcs = list(deltas)
        if null_source is not None:
            srcs = [null_source.unsqueeze(0).unsqueeze(0).expand(B, T, D)] + srcs
        V = torch.stack(srcs, dim=0)
        N = V.shape[0]
        H = num_heads
        _check_dims(D, H)
        K = D // H
        partial = partial.contiguous()
        out = torch.empty_like(partial, memory_format=torch.contiguous_format)
        w = torch.empty((N, BT, H), device=partial.device, dtype=torch.float32)
        HB, KB, num_warps = _launch_cfg(H, K)
        _mhar_fwd_kernel[(BT,)](
            V, out, w, q_weight, g_weight, partial,
            N, BT, D, H, K, BT * D, eps, 1.0 / D,
            ADD_PARTIAL=True, HB=HB, KB=KB, num_warps=num_warps,
        )
        ctx.save_for_backward(V, w, q_weight, g_weight)
        ctx.meta = (N, H, K, eps, null_source is not None, B, T, D)
        ctx.q_shape = q_weight.shape
        return out

    @staticmethod
    def backward(ctx, dout):
        V, w, q_weight, g_weight = ctx.saved_tensors
        N, H, K, eps, has_null, B, T, D = ctx.meta
        BT = B * T
        dout = dout.contiguous()
        dV = torch.empty((N, B, T, D), device=dout.device, dtype=torch.float32)
        dqp = torch.empty((BT, D), device=dout.device, dtype=torch.float32)
        dgp = torch.empty((BT, D), device=dout.device, dtype=torch.float32)
        HB, KB, num_warps = _launch_cfg(H, K)
        _mhar_bwd_kernel[(BT,)](
            V, w, dout, dV, dqp, dgp, q_weight, g_weight,
            N, BT, D, H, K, BT * D, eps, 1.0 / D,
            ACCUM=False, HB=HB, KB=KB, num_warps=num_warps,
        )
        dq = dqp.sum(dim=0).to(q_weight.dtype).view(ctx.q_shape)
        dg = dgp.sum(dim=0).to(g_weight.dtype)
        i0 = 0
        dnull = None
        if has_null:
            dnull = dV[0].sum(dim=(0, 1)).to(q_weight.dtype)
            i0 = 1
        ddeltas = tuple(dV[i].to(dout.dtype) for i in range(i0, N))
        # out = partial + mix, so partial's grad is dout itself
        return (dout, dq, dg, dnull, None, None) + ddeltas


def delta_route(deltas, partial_block, proj, norm, num_heads, null_source):
    """Fused replacement for mh_delta_attn_res (delta_mh mode)."""
    return _FusedDeltaMHFn.apply(
        partial_block, proj.weight, norm.weight, null_source, num_heads,
        norm.variance_epsilon, *deltas)


def route(mctx, blocks, partial_block, proj, norm, num_heads):
    """Fused replacement for mh_block_attn_res under an active context."""
    if len(blocks) != mctx.n:
        raise RuntimeError(
            f"MHAR fused: context desync (len(blocks)={len(blocks)}, buffered={mctx.n})")
    out = _FusedMHARFn.apply(
        partial_block, proj.weight, norm.weight, mctx, num_heads, norm.variance_epsilon)
    if _CHECK:
        ref = _eager_reference(
            mctx.buf[:mctx.n], proj.weight.view(num_heads, -1), norm.weight,
            norm.variance_epsilon, num_heads)
        err = (out.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        if err > 3e-2 * max(scale, 1.0):
            raise RuntimeError(f"MHAR fused CHECK failed: max abs err {err:.4e} (scale {scale:.3e})")
    return out


def _eager_reference(V, query, g, eps, num_heads):
    """Mirror of modeling_qwen3_attnres._mh_block_attn_res_kernel (fp32 math)."""
    x32 = V.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    khat = (x32 * torch.rsqrt(var + eps)).to(V.dtype)
    Kn = (g * khat).float()
    n, b, t, d = V.shape
    h, dh = query.shape
    logits = torch.einsum("hk,nbthk->nbth", query.float(), Kn.view(n, b, t, h, dh))
    weights = logits.softmax(dim=0)
    out = torch.einsum("nbth,nbthk->bthk", weights, x32.view(n, b, t, h, dh))
    return out.reshape(b, t, d)
