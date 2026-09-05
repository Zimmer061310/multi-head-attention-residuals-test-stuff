# Experiment 11 — Soft MHAR Query Specialization

## Status and authorization

This document froze the scientific plan before implementation. Architecture
and workflow implementation were subsequently authorized and are now present
as opt-in Experiment 11 code. GPU rental, training, continuation beyond step
2,000, and multi-seed work remain unauthorized and require separate explicit
authorization.

The first stage is a seed-42, step-2,000 screening experiment. Its purpose is
to determine whether a fixed cross-chunk input bias produces a useful
intermediate degree of query specialization. It is not a convergence or
multi-seed claim.

## Motivation

Experiments 8–10 establish that the hard-local heads inside trained HQ8 are not
dead computation: removing them hurts, all eight groups contribute, and every
one-group wrong-chunk substitution hurts on confirmation. They do not show
that a hard global/local division is optimal.

Experiment 11 changes the central variable from the number of local heads to
the strength of the specialization bias between an MHAR chunk and its query
heads.

## Fixed backbone and training recipe

Use the same 1B MHAR-8 setup as the accepted M8/HQ8 experiments:

- hidden width 1,280, 36 layers, and FFN width 5,120;
- 16 Q heads, 8 KV heads, and head dimension 80;
- eight contiguous 160-dimensional MHAR chunks;
- sequence length 1,024 and global batch 32;
- FineWeb-Edu `sample-10BT`, pinned revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- Qwen3 tokenizer, pinned revision
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- AdamW, bf16, LR 5e-4 to 5e-5, and 1,000-step warmup;
- original 20,000-step schedule, atomically stopped at step 2,000;
- seed 42, identical packed-data order, and identical dense initialization;
- dense global K, V, and W_O in every condition;
- the same document-disjoint fixed artifact used by Experiments 8–10, SHA-256
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`.

No result from a physically sparse LQ8 or HQ8 checkpoint is reused as a
training endpoint. Their parameterization and optimizer state differ from the
dense masked formulation below.

## Soft-local query definition

Let the routed residual be `x = [c_0; ...; c_7]`. For target GQA group `g`,
define the fixed input mask

```text
m_g(d) = 1       if d belongs to c_g
         lambda  otherwise.
```

For a soft-local query head:

```text
Q[g,h] = (m_g * x) W_Q[g,h]
       = c_g W_L[g,h] + lambda x_not_g W_X[g,h].
```

`lambda` is fixed for the entire run and is not learned. All dense Q weights
physically exist at every lambda. The implementation must apply the mask before
the Q projection; it must not mask an already-computed query output.

This parameterization has an important interpretation limit: for every
`lambda > 0`, the optimizer can increase the cross-chunk weights and partially
cancel the input scaling. Lambda is therefore an optimization bias or
structural prior, not a guaranteed final mixture fraction. The effective
specialization measurements below are required, not optional diagnostics.

## Architecture families

### S2Q8: soft-local plus soft-local

Both query heads in each GQA group receive the same chunk-biased input:

```text
Q[2g]   = soft_local(x, c_g, lambda)
Q[2g+1] = soft_local(x, c_g, lambda)
K[g], V[g] = dense_global(x)
```

At `lambda=0`, this is functionally hard-local LQ8 with dormant dense cross
weights. At `lambda=1`, it is ordinary dense M8.

### GSLQ8: global plus soft-local

Preserve the Experiment 8 ordering: the even query head is specialized and the
odd query head is global.

```text
Q[2g]   = soft_local(x, c_g, lambda)
Q[2g+1] = dense_global(x)
K[g], V[g] = dense_global(x)
```

At `lambda=0`, this is functionally HQ8 with dormant dense cross weights. At
`lambda=1`, it is ordinary dense M8.

## Frozen nine-run matrix

| Run ID | Family | Lambda | Queries per GQA group |
|---|---|---:|---|
| `s2q8-l000` | S2Q8 | 0.00 | soft-local + soft-local |
| `s2q8-l010` | S2Q8 | 0.10 | soft-local + soft-local |
| `s2q8-l025` | S2Q8 | 0.25 | soft-local + soft-local |
| `s2q8-l050` | S2Q8 | 0.50 | soft-local + soft-local |
| `gslq8-l000` | GSLQ8 | 0.00 | soft-local + global |
| `gslq8-l010` | GSLQ8 | 0.10 | soft-local + global |
| `gslq8-l025` | GSLQ8 | 0.25 | soft-local + global |
| `gslq8-l050` | GSLQ8 | 0.50 | soft-local + global |
| `m8-l100` | shared endpoint | 1.00 | global + global |

The lambda-1 endpoint is trained once because both families are exactly the
same architecture and computation at that endpoint. The controller must reject
duplicate or missing run IDs.

## Matched initialization and parameterization

Instantiate the same dense M8 model for every run from the same seed. Install
only a non-parameterized fixed input mask around the relevant dense Q heads.
Every Q/K/V/O, MHAR, FFN, embedding, normalization, and output value must be
bitwise identical before applying the run-specific mask.

All nine runs have the same trainable parameter count and dense QKV projection
MAC count. The mask may add small elementwise overhead. This experiment tests
specialization, not computational savings, and no speedup claim is permitted.

At lambda zero, inactive cross-block parameters remain in the optimizer. Their
function is exactly zero, even if decoupled weight decay changes their stored
values. The manifest must record this behavior.

## Effective-specialization measurements

For layer, group, and relevant query head, partition the dense query weight into
its local and cross blocks. Record the following in FP32.

### Weight-space ratio

```text
R_weight = lambda * RMS(W_X) / RMS(W_L)
```

where `RMS(W)` is the Frobenius norm divided by the square root of the number
of entries. This prevents the 1,120-dimensional cross block from appearing
large solely because it contains seven times as many input columns.

### Activation-space ratio

On fixed examples, form the pre-Q-normalization components

```text
q_L = c_g W_L
q_X = lambda x_not_g W_X
R_act = mean(||q_X||_2) / mean(||q_L||_2).
```

Use the ratio of aggregate token-weighted means, not the mean of unstable
per-token ratios. Record the complete layer-by-group-by-head tensors.

### Query-direction change

Use the model's actual per-head Q normalization for both the full query and the
local-only counterfactual:

```text
q_full_norm  = QNorm(q_L + q_X)
q_local_norm = QNorm(q_L)
theta = mean(acos(clamp(cos(q_full_norm, q_local_norm), -1, 1))).
```

Calculate norms, cosine similarity, clamping, and angles in FP32 with an
explicit epsilon. Also record the pre-normalization cosine between `q_L` and
`q_X`. Theta is the primary effective-specialization diagnostic because it
measures how much cross-chunk access changes the normalized query direction.

For the shared M8 endpoint, compute the same decomposition relative to each
head's nominal GQA chunk. For GSLQ8 comparisons, use the even soft-head slots;
for S2Q8 comparisons, use both query-head slots.

## Measurement schedule

- Atomic resumable training checkpoint every 100 optimizer steps, retaining a
  rolling latest checkpoint and an immutable complete step-2,000 checkpoint.
- Record weight metrics every 100 steps.
- At steps 0, 500, 1,000, 1,500, and 2,000, pause at an atomic boundary and
  compute discovery-probe `R_act`, theta, and component cosine statistics.
- Do not inspect confirmation examples during training or candidate selection.
- Held-out discovery and confirmation NLL use token-weighted aggregation and
  preserve per-sequence losses for paired inference.

The discovery probe is fixed before launch. Its sequence IDs and artifact hash
must be recorded in the protocol. Intermediate activation artifacts are saved
without retaining every historical model checkpoint.

## Frozen selection procedure

At step 2,000:

1. Evaluate all nine runs on discovery only.
2. Within S2Q8, select the minimum-discovery-NLL candidate from
   `{0.10, 0.25, 0.50}`.
3. Within GSLQ8, select the minimum-discovery-NLL candidate from
   `{0.10, 0.25, 0.50}`.
4. Break an exact numerical tie by choosing the smaller lambda.
5. Write an immutable selection manifest containing both run IDs, lambda
   values, checkpoint hashes, discovery-result hashes, and source commit.
6. Verify the selection-manifest hash before opening confirmation.
7. Evaluate all nine runs on confirmation for the preregistered curve, but do
   not change either selected candidate.

## Primary confirmation comparisons

Negative delta NLL favors the first named model. Use 10,000 paired
per-sequence bootstrap resamples with frozen seeds.

For each selected family candidate:

```text
selected intermediate - family lambda-0 endpoint
selected intermediate - shared M8 lambda-1 endpoint
```

Then compare the independently selected families:

```text
selected S2Q8 - selected GSLQ8
```

Report the complete discovery and confirmation curves even though only the two
frozen intermediate candidates receive selected-candidate interpretation.

## Effective-softness confirmation

For each selected candidate, compare its aggregate activation metrics with the
corresponding head slots in M8 using paired fixed-example resampling.

A selected candidate demonstrates an intermediate effective state only if:

- `R_act > 0` and `theta > 0` as finite point estimates;
- the 95% interval for `R_act(selected) - R_act(M8)` is wholly below zero; and
- the 95% interval for `theta(selected) - theta(M8)` is wholly below zero.

The complete maps and trajectories remain primary evidence. `R_weight` is a
supporting compensation diagnostic and cannot replace activation measurements.

## Frozen result classification

For a family, record `soft_specialization_supported` only if its selected
intermediate candidate:

1. has confirmation delta NLL below zero against both its lambda-0 endpoint
   and shared M8;
2. has both paired 95% NLL intervals wholly below zero; and
3. passes the effective-softness confirmation above.

If the NLL requirements pass but effective softness does not, record
`performance_gain_without_demonstrated_softness`. Otherwise record
`no_confirmed_soft_specialization_advantage`.

These are within-seed step-2,000 classifications. They do not authorize
continuation, a 20,000-step claim, or multi-seed replication.

## Required figures

Generate reproducibly from committed JSON/CSV artifacts, not manually in the
W&B interface:

1. discovery and confirmation NLL versus lambda for both families;
2. selected confirmation contrasts with paired 95% intervals;
3. aggregate `R_weight`, `R_act`, and theta versus lambda;
4. step-wise compensation trajectories;
5. layer-by-group effective-specialization heatmaps for both selected models;
6. parameter, projection-MAC, throughput, peak-memory, and wall-time table.

Save 300-DPI PNG and vector PDF versions. W&B stores metrics, tables, and links;
the repository contains the reproducible publication figures.

## Implementation and failure safeguards

- Implement Experiment 11 as opt-in experiment-only modules and configuration.
- Do not change legacy MHAR or Experiments 1–10 behavior.
- Validate masks, head ordering, lambda endpoints, identical initialization,
  gradients, Q normalization placement, checkpoint round-trip, and metrics on
  a small CPU model before any GPU launch.
- Require a clean pinned source commit, frozen run-matrix hash, dependency
  snapshot, fixed artifact hash, W&B identity, adequate disk, and idle assigned
  GPUs during preflight.
- Never silently retry or overwrite a failed scientific run.
- Refuse confirmation if no valid frozen selection manifest exists.
- Refuse analysis on missing conditions, non-finite values, inconsistent
  checkpoints, wrong data position, dirty scientific code, or hash mismatch.
- Write explicit failure and completion markers.
- Commit and push compact results, figures, manifests, and `FINAL_REPORT.md`
  before any success-only shutdown.
- Verify remote publication and GPU idleness before shutdown.

## Interpretation limits

- one seed and one early checkpoint;
- lambda is a training bias, not a guaranteed information percentage;
- activation summaries describe the fixed probe distribution;
- paired sequence intervals do not quantify training-seed uncertainty;
- dense parameter matching deliberately removes the parameter/computation
  advantage of physically sparse local Q;
- a winning screen motivates longer and multi-seed evaluation but does not
  establish a final architecture.
