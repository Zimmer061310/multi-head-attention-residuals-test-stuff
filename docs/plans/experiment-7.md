# Experiment 7 — Local-Q / Global-KV Coupling

## Status

Implementation complete and locally verified. No GPU run has started.
The first screen is single-seed step-2,000 screening only; it is not a
multi-seed confirmation experiment.

Implementation lives in `src/experiments/experiment7_local_q.py` and is
activated only by the frozen Experiment 7 command-line identities. The legacy
Qwen3/MHAR classes and Experiment 6 classes are unchanged. Training, fixed-set
evaluation, paired analysis, three-GPU scheduling, result backup, and
success-only shutdown are covered by the Experiment 7 scripts and runbook.

## Goal

Test a weaker and more targeted version of Experiment 6:

> Let each MHAR chunk specialize the query heads assigned to it, while keys
> and values remain globally informed by the full routed representation.

Experiment 6 made all three projections local:

\[
Q,K,V \leftarrow \text{one chunk}.
\]

That restriction was strongly harmful, although the negative interaction
terms showed that MHAR partially mitigated its cost. Experiment 7 asks whether
only query construction can be localized:

\[
\boxed{
Q \leftarrow \text{local MHAR chunk},\qquad
K,V \leftarrow \text{full routed residual}
}
\]

The intended claim is narrow: MHAR chunks may help determine what an attention
group asks for without being self-contained key/value feature spaces.

## Fixed 1B backbone

Use the same matched setup as Experiments 2 and 6:

- hidden width 1,280 and 36 layers;
- 16 query heads, 8 KV heads, head dimension 80;
- FFN width 5,120;
- sequence length 1,024 and global batch 32;
- FineWeb-Edu `sample-10BT` at pinned revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- Qwen3 tokenizer at pinned revision
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- AdamW, bf16, peak LR 5e-4, minimum LR 5e-5, and 1,000-step warmup;
- the original 20,000-step schedule, stopped atomically at step 2,000;
- seed 42 and the same packed data order;
- the Experiment 2 Stage B fixed evaluation artifact with SHA-256
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`;
- dense K, dense V, dense output projection, normalization, RoPE, GQA, FFN,
  tokenizer, optimizer, schedule, and evaluation procedure.

No multi-seed run or continuation beyond step 2,000 is authorized in the
screening stage.

## Model names and run matrix

The required new models are:

| ID | Residual routing | Q input | K/V input | Purpose |
|---|---|---|---|---|
| **LQ4** | MHAR-4, 4 × 320 | 4 local groups | dense/global | proposed H4 model |
| **LQ8** | MHAR-8, 8 × 160 | 8 local groups | dense/global | proposed H8 model |

Reuse the accepted controls on the identical fixed artifact:

| ID | Residual routing | Q/K/V input | Status |
|---|---|---|---|
| **M4** | MHAR-4 | dense/global | accepted step-2,000 result |
| **M8** | MHAR-8 | dense/global | accepted step-2,000 result |
| **B** | ordinary residual | dense/global | accepted Experiment 6 result |

Optional non-MHAR local-Q controls are:

| ID | Residual routing | Q input | K/V input | Purpose |
|---|---|---|---|---|
| **BLQ4** | ordinary residual | 4 local groups | dense/global | isolate generic local-Q cost |
| **BLQ8** | ordinary residual | 8 local groups | dense/global | isolate generic local-Q cost |

Thus the minimum screen has two new runs, **LQ4** and **LQ8**. Including both
optional controls makes four new runs. The labels `Q4`, `Q8`, `BQ4`, and `BQ8`
must not be used in manifests, W&B names, figures, or reports.

## Architecture

Let the MHAR output be

\[
x_{\mathrm{MHAR}}=[c_0;c_1;\ldots;c_{G-1}],
\qquad c_g\in\mathbb{R}^{1280/G}.
\]

Only the query projection is block diagonal:

\[
W_Q=\operatorname{diag}
\left(W_Q^{(0)},\ldots,W_Q^{(G-1)}\right).
\]

Keys and values remain ordinary dense projections:

\[
K=x_{\mathrm{MHAR}}W_K,\qquad
V=x_{\mathrm{MHAR}}W_V.
\]

Normal Q/K normalization, RoPE, grouped-query attention, causal masking, KV
caching, and dense \(W_O\) follow unchanged.

### LQ4 mapping

Each 320-dimensional chunk generates four query heads:

```text
c0 -> Q0-Q3
c1 -> Q4-Q7
c2 -> Q8-Q11
c3 -> Q12-Q15
```

All eight K heads and all eight V heads read the complete 1,280-dimensional
routed representation.

### LQ8 mapping

Each 160-dimensional chunk generates two query heads:

```text
c0 -> Q0,Q1
c1 -> Q2,Q3
...
c7 -> Q14,Q15
```

This gives one MHAR chunk per GQA query group while its shared K/V head remains
globally informed.

## Initialization and parameter counts

Initialize the corresponding dense model first, then replace only `q_proj`
with its retained diagonal blocks. Every retained Q weight and every non-Q
weight must exactly match the paired dense model at initialization. `k_proj`,
`v_proj`, MHAR routers, and `o_proj` remain untouched.

Because Qwen3 normalizes Q after projection, this design also avoids the
unnormalized local-V scale reduction that may have contributed to Experiment
6's early penalty.

Dense QKV has 3,276,800 weights per layer. Local-Q/global-KV retains:

| Design | Local Q | Dense K+V | Total QKV/layer | Dense fraction |
|---|---:|---:|---:|---:|
| LQ4/BLQ4 | 409,600 | 1,638,400 | 2,048,000 | 62.5% |
| LQ8/BLQ8 | 204,800 | 1,638,400 | 1,843,200 | 56.25% |

At the current 36-layer model shape, expected total trainable parameters are:

- LQ4: 1,035,263,360;
- BLQ4: 1,035,076,480;
- LQ8: 1,027,890,560;
- BLQ8: 1,027,703,680.

These runs are not parameter matched to their dense controls. Parameter count
and QKV MAC/token must therefore accompany every quality comparison.

## Primary comparisons

Does Local-Q / Global-KV help or hurt MHAR?

\[
\boxed{LQ4-M4},\qquad \boxed{LQ8-M8}.
\]

Negative delta NLL favors Local-Q coupling.

If BLQ4 and BLQ8 are included, estimate the generic local-Q cost:

\[
BLQ4-B,\qquad BLQ8-B,
\]

and the MHAR-specific interactions:

\[
(LQ4-M4)-(BLQ4-B),
\]

\[
(LQ8-M8)-(BLQ8-B).
\]

Negative interaction means MHAR adapts better to local queries than an
ordinary residual model does.

For diagnosis only, also report `LQ4-C4` and `LQ8-C8`. These are not primary
causal contrasts because Experiment 7 restores dense K/V parameters absent in
Experiment 6.

## Training and evaluation protocol

Train every new model from scratch with matched seed, packed data order,
optimizer, scheduler, and initialization procedure. Save rolling atomic
checkpoints every 100 steps and protect steps 500, 1,000, 1,500, and 2,000.

At step 2,000, pause and evaluate all new checkpoints on both fixed splits.
Report:

- held-out token-weighted NLL and perplexity;
- paired per-sequence bootstrap 95% confidence intervals for all frozen
  contrasts;
- trainable parameters and QKV MAC/token;
- measured training and inference throughput;
- peak GPU memory and elapsed wall time;
- W&B and checkpoint links.

Do not automatically continue training. Step 2,000 is an early screen for an
obviously harmful design, not evidence sufficient to declare a winner.

## Interpretation

1. **LQ4/LQ8 approximately match M4/M8:** local queries preserve MHAR quality
   with fewer Q parameters; longer training becomes worth reviewing.
2. **LQ4/LQ8 beat M4/M8:** strong single-seed evidence that MHAR chunk identity
   is useful specifically for query specialization.
3. **They lose, but far less than C4/C8:** Experiment 6 failed mainly because K
   and V were localized; Local-Q may remain viable but still requires review.
4. **They lose almost as badly as C4/C8:** even query construction needs broad
   cross-chunk access, so hard chunk-to-attention coupling should stop.
5. **Optional interaction is negative:** MHAR representations adapt to local Q
   better than ordinary residual representations, even if the absolute model
   does not beat M4/M8.

The frozen catastrophic rule from Experiment 6 remains: non-finite loss,
non-finite held-out NLL, incomplete checkpoint, or a primary confirmation
penalty above +0.5 NLL rejects the design. Any other result is reported for
review without automatic continuation.

## Isolation safeguards for implementation

- Implement Experiment 7 in a separate opt-in module; do not modify MHAR
  routing kernels or Experiment 1-6 result paths.
- Replace `q_proj` only. Assert that `k_proj`, `v_proj`, and `o_proj` remain
  dense and initialization-identical to the paired control.
- Require the exact labels `lq4`, `lq8`, `blq4`, and `blq8` in configuration,
  manifests, checkpoint identity, and W&B identity.
- Unit-test equivalence to an explicitly masked dense Q matrix, forbidden
  cross-group Q influence, matched initialization, parameter counts, forward
  pass, checkpoint round trip, and unchanged legacy dense paths.
- Fail closed on artifact/hash mismatch, variant mismatch, incomplete
  checkpoint, failed evaluation, failed backup, or live GPU processes at
  shutdown.

## Core question

\[
\boxed{
\text{Can MHAR chunks specialize what attention asks for while K/V remain
globally informed?}
}
\]
