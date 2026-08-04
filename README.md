# Multi-Head Attention Residuals (MHAR)

Paper and reference implementation for **Multi-Head Attention Residuals**.

[📄 arXiv:2607.27230](https://arxiv.org/abs/2607.27230) · [📝 Blog post](https://wdlctc.github.io/multi-head-attention-residuals.html) · [PDF](paper/mhar_arxiv_v1.pdf)

Cheng Luo, Zefan Cai, Junjie Hu

## TL;DR

Attention residuals let every sublayer attend over the depth history through a learned
softmax — but that read uses a **single query shared across the entire width**, forcing
every feature subspace to read the depth history through one distribution. The cost of
this forced compromise grows with how much the subspaces disagree about which layers to
read, and disagreement grows with model width.

**MHAR** reshapes the routing query into `H` per-subspace heads, each with its own softmax
over the depth history:

- the read becomes block-diagonal,
- the reshape adds **zero parameters** and negligible compute,
- `H = 1` recovers attention residuals exactly.

Trained from scratch, MHAR improves validation loss over a standard Transformer at
100M, 350M, and 1B (−0.061, −0.149, −0.140), the best of four methods in every setting,
with the gain increasing toward larger scales. Validation loss is U-shaped in `H` with a
flat optimum at `H = 4`–`8`; we adopt `H = 8` for large models. Fused Triton routing
kernels raise attention-residual training throughput from 0.2–0.5× to 0.55–0.88× of the
baseline at near-baseline peak memory, and an identity-preserving conversion (delta
attention residuals) supports 8B mid-training: **+3.2 GSM8K, +3.1 GPQA**.

## Repository layout

```
paper/                              arXiv source + PDF
Attention-Residuals/
  modeling_qwen3_attnres.py         Qwen3-style architecture with all attention-residual
                                    modes (baseline / full / full_mh = MHAR / delta / ...)
  modeling_llama_attnres.py         Llama-style architecture for converting pretrained
                                    checkpoints (8B mid-training via delta_mh)
  mhar_triton.py                    fused Triton routing kernels
train_scratch.py                    from-scratch pretraining entry point
train_cpt.py                        continued-pretraining (conversion) entry point
test_mhar_fused.py                  fused-vs-eager correctness tests
test_mhar_fused_delta.py            fused-vs-eager tests for the delta variant
```

## Quick start

From-scratch pretraining with MHAR (`full_mh`, 8 routing heads):

```bash
torchrun --nproc_per_node=8 train_scratch.py \
    --mode full_mh --attnres_heads 8
```

Single-head attention residuals (the `H = 1` special case):

```bash
torchrun --nproc_per_node=8 train_scratch.py --mode full
```

Standard Transformer baseline:

```bash
torchrun --nproc_per_node=8 train_scratch.py --mode baseline
```

Convert a pretrained checkpoint with the identity-preserving delta variant
(zero disruption at step 0), then continue pretraining:

```bash
torchrun --nproc_per_node=8 train_cpt.py \
    --mode delta_mh --attnres_heads 8
```

Verify the fused Triton kernels against the eager path:

```bash
python test_mhar_fused.py
python test_mhar_fused_delta.py
```

Requirements: PyTorch ≥ 2.4, `transformers`, `triton`, `datasets` (and `wandb` if you
pass `--wandb_entity`). Data defaults to `HuggingFaceFW/fineweb-edu`; pass
`--data_files 'path/*.parquet'` to train on local parquet shards.

## Citation

```bibtex
@article{luo2026mhar,
  title={Multi-Head Attention Residuals},
  author={Cheng Luo and Zefan Cai and Junjie Hu},
  journal={arXiv preprint arXiv:2607.27230},
  year={2026}
}
```

## Related work by the authors

- [Delta Attention Residuals](https://github.com/wdlctc/delta-attention-residuals-code) —
  additive (identity-preserving) routing over sublayer deltas; the conversion technique
  used for 8B mid-training in this paper.
- [Open Attention Residuals](https://github.com/wdlctc/open-attention-residuals) — open
  implementation of attention residuals (Kimi Team, arXiv:2603.15031).
