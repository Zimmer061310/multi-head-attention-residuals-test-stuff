# Experiment 8 — Hybrid-Q8 / Global-KV

## Status

Preregistered implementation screen. Code and run infrastructure are prepared,
but no GPU run is authorized by this document alone. The first screen is seed
42 through the atomic step-2,000 checkpoint. It does not authorize continuation
or multi-seed training.

## Question

Can each H8/GQA group benefit from heterogeneous query access—one query head
specialized to its MHAR chunk and one query head retaining full residual
access—while K, V, and W_O remain global/dense?

For group `g`, standard Qwen3 GQA assigns query heads `2g` and `2g+1` to KV
head `g`. Experiment 8 freezes the following ordering:

```text
Q[2g]   = local(c_g)       # even head
Q[2g+1] = global(x)        # odd head
K[g]    = global(x)
V[g]    = global(x)
```

The query outputs are interleaved in ordinary Qwen3 head order before normal
Q normalization, RoPE, GQA attention, concatenation, and dense W_O.

## Why the baseline hybrid control is required

HQ8 alone cannot distinguish an MHAR-specific benefit from generic structured
query sparsity, parameter reduction, or regularization. Therefore train two new
models:

| ID | Residual routing | Queries per GQA group | K/V | Status |
|---|---|---|---|---|
| M8 | MHAR-8 | global + global | global | accepted result |
| HQ8 | MHAR-8 | local + global | global | new run |
| LQ8 | MHAR-8 | local + local | global | accepted result |
| B | ordinary | global + global | global | accepted result |
| BHQ8 | ordinary | local + global | global | new run |
| BLQ8 | ordinary | local + local | global | accepted result |

M8/LQ8 and B/BLQ8 are reused only because their per-sequence results use the
same seed-42 step-2,000 milestone and identical fixed artifact.

## Fixed backbone and training recipe

- hidden width 1,280, 36 layers, and FFN width 5,120;
- 16 Q heads, 8 KV heads, head dimension 80;
- sequence length 1,024 and global batch 32;
- FineWeb-Edu `sample-10BT`, pinned revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- Qwen3 tokenizer, pinned revision
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- AdamW, bf16, LR 5e-4 to 5e-5, 1,000-step warmup;
- original 20,000-step schedule, atomically stopped at step 2,000;
- seed 42, identical packed-data order, and matched dense initialization;
- atomic checkpoint every 100 steps, retaining the latest state and protected
  step 2,000 without a duplicate final checkpoint;
- document-disjoint fixed artifact SHA-256
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`.

## Initialization and physical parameters

Instantiate the matched dense model first. For every GQA pair, copy the even Q
head's matching 160-coordinate diagonal block into the local projection and
copy the odd Q head's complete dense row block into the global projection.
Every retained Q value and every K/V/O/non-attention value must match the dense
initialization exactly. Disallowed local-Q weights must not exist as trainable
parameters.

Per layer:

- local Q: `8 × 160 × 80 = 102,400`;
- global Q: `8 × 1,280 × 80 = 819,200`;
- total Q: `921,600`;
- dense K+V: `1,638,400`;
- total QKV: `2,560,000`, or 78.125% of dense QKV.

Expected full-model trainable parameters are 1,053,695,360 for HQ8 and
1,053,508,480 for BHQ8. Report physical parameter counts, QKV projection MACs,
measured throughput, peak memory, and wall time. Projection MAC counts are not
whole-model FLOPs or guaranteed hardware speedups.

## Frozen comparisons

Use token-weighted held-out NLL and 10,000 paired per-sequence bootstrap
resamples on discovery and confirmation. Define delta as first model minus
second model, so negative values favor the first model.

Primary:

```text
HQ8 - M8
(HQ8 - M8) - (BHQ8 - B)
```

Control:

```text
BHQ8 - B
```

Endpoint and curve-shape diagnostics:

```text
HQ8 - LQ8
BHQ8 - BLQ8
2*HQ8 - M8 - LQ8
2*BHQ8 - B - BLQ8
```

The interaction is required for an MHAR-specific interpretation. The endpoint
contrasts describe the 0%→50%→100% local-Q curve but do not replace the primary
HQ8−M8 comparison.

## Frozen interpretation

- HQ8−M8 confirmation CI wholly below zero: within-seed evidence that hybrid
  queries improve the step-2,000 MHAR screen.
- HQ8−M8 CI wholly inside ±0.005 NLL: within-seed practical match with fewer Q
  parameters; this is not across-seed equivalence.
- M8 < HQ8 < LQ8 on both splits: query locality is progressively harmful.
- Negative interaction: MHAR tolerates or benefits from hybrid Q more than the
  ordinary-residual control does.
- HQ8 improvement without a favorable interaction: structured Q restriction
  may help generically; do not attribute the gain specifically to MHAR chunks.

A primary confirmation penalty above +0.5 NLL, non-finite values, or an
incomplete checkpoint is catastrophic. Otherwise report for review. There is
no automatic continuation, H4 expansion, or multi-seed run.

## Isolation and completion safeguards

- Implementation is opt-in under Experiment 8 identities only.
- Legacy MHAR, Experiment 6, and Experiment 7 modules remain unchanged.
- K, V, and W_O must remain ordinary dense `nn.Linear` modules.
- Validate GQA ordering, forbidden local cross-chunk influence, global-head
  access, exact retained initialization, parameter counts, forward behavior,
  checkpoint round-trip, and unchanged legacy tests.
- Evaluation fails closed on recipe, source, variant, artifact, checkpoint, or
  head-order mismatch.
- Compact results are committed and pushed before shutdown. Incomplete work,
  failed backup, or active GPU processes blocks shutdown.
