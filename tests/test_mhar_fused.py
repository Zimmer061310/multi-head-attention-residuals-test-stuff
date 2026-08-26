"""
Correctness tests + microbenchmark for the fused Triton MHAR (full_mh) kernels.

Correctness strategy: run the SAME routing-call chain (each call's output feeds
the next call's partial through a small linear map, sources accumulate exactly
like full_mh's blocks list) through the eager path and the fused path, then
compare loss and every gradient (embedding, per-call q / norm-weight / mix
weight). fp32 chains must match tightly (same math, different reduction
order); bf16 chains loosely (both paths round differently).

Usage (on a GPU node):
    python3 -m tests.test_mhar_fused            # correctness
    python3 -m tests.test_mhar_fused --bench    # + routing-chain microbenchmark
"""

import argparse
import sys
import time

import torch
import torch.nn as nn

from src.attention_residuals import mhar_triton
from src.attention_residuals.modeling_qwen3_attnres import mh_block_attn_res, enable_compile
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

DEV = "cuda"


def build_chain_modules(D, S, dtype, seed):
    torch.manual_seed(seed)
    projs, norms, mixes = [], [], []
    for _ in range(S):
        pj = nn.Linear(D, 1, bias=False).to(DEV, dtype)
        nm = Qwen3RMSNorm(D, eps=1e-6).to(DEV, dtype)
        mx = nn.Linear(D, D, bias=False).to(DEV, dtype)
        with torch.no_grad():
            pj.weight.normal_(0, 4.0 * D ** -0.5)   # sizeable logits -> peaked softmax
            nm.weight.uniform_(0.8, 1.2)            # non-trivial dg
            mx.weight.normal_(0, D ** -0.5)
        projs.append(pj); norms.append(nm); mixes.append(mx)
    return projs, norms, mixes


def run_chain(D, H, S, B, T, dtype, fused, seed=0):
    projs, norms, mixes = build_chain_modules(D, S, dtype, seed)
    torch.manual_seed(seed + 1)
    emb = torch.randn(B, T, D, device=DEV, dtype=dtype) * 3.0
    emb.requires_grad_(True)

    mctx = None
    if fused:
        mctx = mhar_triton.MHARFusedContext(S, emb)
        mhar_triton.set_context(mctx)
    try:
        blocks, partial = [], emb
        outs = []
        for s in range(S):
            h = mh_block_attn_res(blocks, partial, projs[s], norms[s], H)
            outs.append(h)
            blocks = blocks + [partial]
            partial = mixes[s](h)
        loss = torch.stack([o.float().pow(2).mean() for o in outs]).sum() \
            + partial.float().pow(2).mean()
        loss.backward()
    finally:
        if fused:
            mhar_triton.set_context(None)

    grads = {"emb": emb.grad.detach().float().clone()}
    for s in range(S):
        grads[f"q{s}"] = projs[s].weight.grad.detach().float().clone()
        grads[f"g{s}"] = norms[s].weight.grad.detach().float().clone()
        grads[f"m{s}"] = mixes[s].weight.grad.detach().float().clone()
    return loss.item(), grads


def rel_err(a, b):
    denom = b.abs().max().item()
    return (a - b).abs().max().item() / max(denom, 1e-12)


def check_chain(D, H, S, B, T, dtype, tol):
    l_ref, g_ref = run_chain(D, H, S, B, T, dtype, fused=False)
    l_fus, g_fus = run_chain(D, H, S, B, T, dtype, fused=True)
    loss_err = abs(l_fus - l_ref) / max(abs(l_ref), 1e-12)
    worst_name, worst = "", 0.0
    for k in g_ref:
        e = rel_err(g_fus[k], g_ref[k])
        if e > worst:
            worst_name, worst = k, e
    status = "OK " if (loss_err < tol and worst < tol) else "FAIL"
    print(f"[{status}] D={D:5d} H={H:2d} S={S:2d} B={B} T={T:4d} {str(dtype):15s} "
          f"loss_rel={loss_err:.2e} worst_grad_rel={worst:.2e} ({worst_name})")
    return status == "OK "


def check_no_grad(D, H, S, B, T, dtype):
    projs, norms, mixes = build_chain_modules(D, S, dtype, 0)
    emb = torch.randn(B, T, D, device=DEV, dtype=dtype)
    mctx = mhar_triton.MHARFusedContext(S, emb)
    mhar_triton.set_context(mctx)
    try:
        with torch.no_grad():
            blocks, partial = [], emb
            for s in range(S):
                h = mh_block_attn_res(blocks, partial, projs[s], norms[s], H)
                blocks = blocks + [partial]
                partial = mixes[s](h)
        ok = torch.isfinite(partial).all().item()
    finally:
        mhar_triton.set_context(None)
    print(f"[{'OK ' if ok else 'FAIL'}] no_grad chain D={D} H={H} S={S}")
    return ok


# ---------------------------------------------------------------------------
# Microbenchmark: pure routing-chain cost (partial_{s+1} = out_s), the exact
# access pattern of a full_mh forward+backward minus attention/MLP compute.
# ---------------------------------------------------------------------------

def bench_chain(D, H, S, B, T, mode, iters=10):
    dtype = torch.bfloat16
    torch.manual_seed(0)
    proj = nn.Linear(D, 1, bias=False).to(DEV, dtype)
    norm = Qwen3RMSNorm(D, eps=1e-6).to(DEV, dtype)
    emb = torch.randn(B, T, D, device=DEV, dtype=dtype)

    def one_pass():
        e = emb.clone().requires_grad_(True)
        mctx = None
        if mode == "fused":
            mctx = mhar_triton.MHARFusedContext(S, e)
            mhar_triton.set_context(mctx)
        try:
            blocks, partial = [], e
            for _ in range(S):
                h = mh_block_attn_res(blocks, partial, proj, norm, H)
                blocks = blocks + [partial]
                partial = h
            partial.float().pow(2).mean().backward()
        finally:
            if mode == "fused":
                mhar_triton.set_context(None)

    tag = f"{mode:8s} D={D} H={H} S={S} B={B} T={T}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(3):
            one_pass()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            one_pass()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {tag}: OOM (>{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
        return None
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9
    gb = sum(range(1, S + 1)) * B * T * D * 2 / 1e9
    print(f"  {tag}: {ms:8.2f} ms/chain, peak {peak:5.1f} GB (V once = {gb:.1f} GB)")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    assert mhar_triton.HAS_TRITON, "triton required"
    torch.manual_seed(0)

    print("== fp32 chains (tight: same math, different reduction order) ==")
    ok = True
    ok &= check_chain(512, 4, 25, 2, 64, torch.float32, 2e-4)
    ok &= check_chain(1024, 8, 49, 2, 32, torch.float32, 2e-4)
    ok &= check_chain(1280, 8, 16, 2, 48, torch.float32, 2e-4)   # K=160 non-pow2
    ok &= check_chain(768, 12, 8, 2, 24, torch.float32, 2e-4)    # H=12
    ok &= check_chain(512, 1, 8, 2, 24, torch.float32, 2e-4)     # single head
    ok &= check_chain(256, 16, 3, 1, 7, torch.float32, 2e-4)     # tiny/odd T
    ok &= check_chain(512, 4, 1, 2, 16, torch.float32, 2e-4)     # N=1 edge

    print("== bf16 chains (loose: both paths round differently) ==")
    ok &= check_chain(512, 4, 25, 2, 64, torch.bfloat16, 4e-2)
    ok &= check_chain(1024, 8, 49, 2, 32, torch.bfloat16, 4e-2)
    ok &= check_chain(1280, 8, 16, 2, 48, torch.bfloat16, 4e-2)

    print("== no_grad path ==")
    ok &= check_no_grad(512, 4, 25, 2, 64, torch.bfloat16)

    if not ok:
        print("CORRECTNESS FAILURES — do not use --fused")
        sys.exit(1)
    print("all correctness checks passed")

    if args.bench:
        print("== routing-chain microbenchmark (fwd+bwd, bf16) ==")
        # reduced batch so the eager path fits in 80 GB (it retains stacked V
        # + fp32 norm intermediates per call); ratios scale ~linearly in B
        small = [  # (D, H, S, B, T): 100M / 350M / 1B depth+width, reduced B
            (512, 4, 25, 2, 2048),
            (1024, 8, 49, 1, 1024),
            (1280, 8, 73, 1, 1024),
        ]
        full = [  # the actual per-GPU training microbatch shapes
            (512, 4, 25, 8, 2048),
            (1024, 8, 49, 4, 1024),
            (1280, 8, 73, 2, 1024),
        ]
        eager_ms = [bench_chain(*c, "eager") for c in small]
        enable_compile()  # sticky: switches the eager path to torch.compile kernels
        compiled_ms = [bench_chain(*c, "eager") for c in small]
        fused_ms = [bench_chain(*c, "fused") for c in small]
        for c, e, co, f in zip(small, eager_ms, compiled_ms, fused_ms):
            if e and co and f:
                print(f"  -> {c}: fused vs eager {e / f:.2f}x, vs compiled {co / f:.2f}x")
        print("  -- full training microbatch shapes (fused + compiled-eager) --")
        for c in full:
            bench_chain(*c, "eager")
            bench_chain(*c, "fused")


if __name__ == "__main__":
    main()
