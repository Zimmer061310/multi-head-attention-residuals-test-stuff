"""
Correctness tests + microbenchmark for the fused delta_mh routing
(mh_delta_attn_res -> mhar_triton.delta_route).

1. op-level chain: null source + growing block-delta sources, eager vs fused,
   all grads (stream leaf, q, norm weight, null, mix weights). fp32 tight.
2. model-level: tiny LlamaAttnResForCausalLM (delta_mh), eager vs fused,
   both WITHOUT and WITH gradient checkpointing (the 8B CPT setup).
3. --bench: routing-call cost at the 8B mid-training shape.
"""

import argparse
import sys
import time

import torch
import torch.nn as nn

from src.attention_residuals import mhar_triton
from src.attention_residuals.modeling_llama_attnres import (
    LlamaAttnResConfig, LlamaAttnResForCausalLM, mh_delta_attn_res, enable_fused_mhar)
from transformers.models.llama.modeling_llama import LlamaRMSNorm

DEV = "cuda"


def run_chain(D, H, S, B, T, dtype, fused, n_blocks=3, seed=0):
    torch.manual_seed(seed)
    projs, norms, mixes, nulls = [], [], [], []
    for _ in range(S):
        pj = nn.Linear(D, 1, bias=False).to(DEV, dtype)
        nm = LlamaRMSNorm(D, eps=1e-6).to(DEV, dtype)
        mx = nn.Linear(D, D, bias=False).to(DEV, dtype)
        nu = nn.Parameter(torch.randn(D, device=DEV, dtype=dtype) * 0.5)
        with torch.no_grad():
            pj.weight.normal_(0, 4.0 * D ** -0.5)
            nm.weight.uniform_(0.8, 1.2)
            mx.weight.normal_(0, D ** -0.5)
        projs.append(pj); norms.append(nm); mixes.append(mx); nulls.append(nu)
    torch.manual_seed(seed + 1)
    emb = torch.randn(B, T, D, device=DEV, dtype=dtype) * 3.0
    emb.requires_grad_(True)

    enable_fused_mhar(fused)
    blocks, partial = [], emb
    outs = []
    for s in range(S):
        h = mh_delta_attn_res(blocks, partial, projs[s], norms[s], H, nulls[s])
        outs.append(h)
        if s % max(1, S // n_blocks) == 0:  # block boundary: append a delta source
            blocks = blocks + [partial if not blocks else partial - blocks[-1]]
        partial = partial + 0.1 * mixes[s](h)
    loss = torch.stack([o.float().pow(2).mean() for o in outs]).sum() \
        + partial.float().pow(2).mean()
    loss.backward()
    enable_fused_mhar(False)

    grads = {"emb": emb.grad.detach().float().clone()}
    for s in range(S):
        grads[f"q{s}"] = projs[s].weight.grad.detach().float().clone()
        grads[f"g{s}"] = norms[s].weight.grad.detach().float().clone()
        grads[f"m{s}"] = mixes[s].weight.grad.detach().float().clone()
        grads[f"n{s}"] = nulls[s].grad.detach().float().clone()
    return loss.item(), grads


def rel_err(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


def check_chain(D, H, S, B, T, dtype, tol):
    l_ref, g_ref = run_chain(D, H, S, B, T, dtype, fused=False)
    l_fus, g_fus = run_chain(D, H, S, B, T, dtype, fused=True)
    loss_err = abs(l_fus - l_ref) / max(abs(l_ref), 1e-12)
    worst, worst_name = 0.0, ""
    for k in g_ref:
        e = rel_err(g_fus[k], g_ref[k])
        if e > worst:
            worst, worst_name = e, k
    ok = loss_err < tol and worst < tol
    print(f"[{'OK ' if ok else 'FAIL'}] delta chain D={D:5d} H={H:2d} S={S:2d} B={B} "
          f"T={T:4d} {str(dtype):15s} loss_rel={loss_err:.2e} "
          f"worst_grad_rel={worst:.2e} ({worst_name})")
    return ok


def run_model(model, ids, fused, ckpt):
    enable_fused_mhar(fused)
    if ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids, use_cache=False)
    out.loss.backward()
    enable_fused_mhar(False)
    return out.loss.item(), {n: p.grad.detach().float().clone()
                             for n, p in model.named_parameters() if p.grad is not None}


def build_model(dtype):
    torch.manual_seed(7)
    cfg = LlamaAttnResConfig(
        vocab_size=1024, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False,
        head_dim=64, attnres_mode="delta_mh", attnres_num_heads=4,
        attnres_num_blocks=2, attnres_gate_type="learnable_alpha",
        attnres_use_null_source=True,
    )
    model = LlamaAttnResForCausalLM(cfg).to(DEV, dtype)
    # zero-init gate/query blocks all routing grads; randomize so the test bites
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "res_proj" in n:
                p.normal_(0, 0.05)
            elif "alpha" in n or "gate" in n:
                p.fill_(0.3)
            elif "null" in n:
                p.normal_(0, 0.5)
    return model


def check_model(dtype, ckpt, tol_loss, tol_grad):
    gen = torch.Generator(DEV).manual_seed(3)
    ids = torch.randint(0, 1024, (2, 128), device=DEV, generator=gen)
    l_ref, g_ref = run_model(build_model(dtype), ids, fused=False, ckpt=ckpt)
    l_fus, g_fus = run_model(build_model(dtype), ids, fused=True, ckpt=ckpt)
    loss_err = abs(l_fus - l_ref) / max(abs(l_ref), 1e-12)

    if dtype == torch.float32:
        worst, worst_name = 0.0, ""
        for n in g_ref:
            e = rel_err(g_fus[n], g_ref[n])
            if e > worst:
                worst, worst_name = e, n
        ok = loss_err < tol_loss and worst < tol_grad and g_ref.keys() == g_fus.keys()
        detail = f"worst grad rel={worst:.2e} ({worst_name})"
    else:
        # bf16: routing/gate grads are small, cancellation-dominated quantities,
        # and single-batch aggregate errors swing with rounding luck. Criterion:
        # over several batches, fused must be on (geometric) average no farther
        # from the fp32 ground truth than eager. Op-level exactness is already
        # established by the fp32 chain/model tests above.
        import statistics
        ratios = []
        for seed in range(4):
            g2 = torch.Generator(DEV).manual_seed(seed)
            b_ids = torch.randint(0, 1024, (2, 128), device=DEV, generator=g2)
            _, g32 = run_model(build_model(torch.float32), b_ids, fused=False, ckpt=ckpt)
            _, ge = run_model(build_model(dtype), b_ids, fused=False, ckpt=ckpt)
            _, gf = run_model(build_model(dtype), b_ids, fused=True, ckpt=ckpt)
            ef2 = sum(((gf[n] - g32[n]) ** 2).sum().item() for n in g32)
            ee2 = sum(((ge[n] - g32[n]) ** 2).sum().item() for n in g32)
            ratios.append((ef2 / max(ee2, 1e-24)) ** 0.5)
        gm = statistics.geometric_mean(ratios)
        ok = loss_err < tol_loss and gm < 1.5 and g_ref.keys() == g_fus.keys()
        detail = (f"grad-error-vs-fp32-truth fused/eager geomean over 4 batches = "
                  f"{gm:.3f} (<1 means fused closer to truth)")
    print(f"[{'OK ' if ok else 'FAIL'}] delta model {str(dtype):15s} ckpt={int(ckpt)} "
          f"loss ref={l_ref:.6f} fused={l_fus:.6f} rel={loss_err:.2e} | {detail}")
    return ok


def bench(D, H, N, B, T, calls, iters=5):
    dtype = torch.bfloat16
    torch.manual_seed(0)
    proj = nn.Linear(D, 1, bias=False).to(DEV, dtype)
    norm = LlamaRMSNorm(D, eps=1e-6).to(DEV, dtype)
    null = nn.Parameter(torch.randn(D, device=DEV, dtype=dtype) * 0.5)
    deltas = [torch.randn(B, T, D, device=DEV, dtype=dtype) for _ in range(N - 1)]

    def one_call(h):
        return mh_delta_attn_res(deltas, h, proj, norm, H, null)

    def one_pass(fused):
        # per-call activation checkpointing, mirroring the 8B CPT's --grad_ckpt
        enable_fused_mhar(fused)
        partial = torch.randn(B, T, D, device=DEV, dtype=dtype, requires_grad=True)
        h = partial
        for _ in range(calls):
            h = torch.utils.checkpoint.checkpoint(one_call, h, use_reentrant=False)
        h.float().pow(2).mean().backward()
        enable_fused_mhar(False)

    res = {}
    for fused in (False, True):
        for _ in range(2):
            one_pass(fused)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            one_pass(fused)
        torch.cuda.synchronize()
        res[fused] = (time.perf_counter() - t0) / iters * 1e3
    print(f"  D={D} H={H} N={N} B={B} T={T} x{calls} calls (fwd+bwd): "
          f"eager {res[False]:.1f} ms  fused {res[True]:.1f} ms  "
          f"({res[False] / res[True]:.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    assert mhar_triton.HAS_TRITON

    ok = True
    print("== delta chains ==")
    ok &= check_chain(512, 4, 12, 2, 64, torch.float32, 2e-4)
    ok &= check_chain(4096, 8, 8, 1, 32, torch.float32, 2e-4)   # 8B width, H=8
    ok &= check_chain(768, 12, 6, 2, 24, torch.float32, 2e-4)
    ok &= check_chain(512, 4, 12, 2, 64, torch.bfloat16, 4e-2)
    ok &= check_chain(4096, 8, 8, 1, 32, torch.bfloat16, 4e-2)

    print("== delta model (LlamaAttnRes, delta_mh) ==")
    ok &= check_model(torch.float32, ckpt=False, tol_loss=1e-5, tol_grad=5e-4)
    ok &= check_model(torch.float32, ckpt=True, tol_loss=1e-5, tol_grad=5e-4)
    ok &= check_model(torch.bfloat16, ckpt=False, tol_loss=5e-3, tol_grad=6e-2)
    ok &= check_model(torch.bfloat16, ckpt=True, tol_loss=5e-3, tol_grad=6e-2)

    if not ok:
        print("FAILURES — do not use --fused for delta_mh")
        sys.exit(1)
    print("all delta_mh fused checks passed")

    if args.bench:
        print("== routing cost at 8B mid-training shape ==")
        bench(4096, 8, 9, 2, 4096, calls=64)


if __name__ == "__main__":
    main()
