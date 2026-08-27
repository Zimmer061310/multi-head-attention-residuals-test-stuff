# Experiment 3 — Training-Time Learnability of MHAR Boundaries

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED
- Version Label: experiment_3_master_v1

## Status

This is a preregistration for the evidence required **before** implementing a
mechanism that learns MHAR routing-group boundaries during training.

Experiment 3 does not introduce a learnable-boundary architecture. It tests
whether a useful training-time boundary signal exists, persists long enough to
use, and improves future optimization when followed.

## Evidence already established

Experiment 2 established two facts that motivate, but do not answer, this
question:

1. Frozen partition choice changes NLL.
2. Boundary preferences measured in a trained H16 representation did not
   transfer reliably to separately trained mixed-width models. At the seed-42
   step-2,000 screen, the frozen `k=4 worst` partition outperformed the frozen
   `k=4 best` partition after matched training.

The second result is consistent with boundary preferences being co-adapted to
the representation of each model. It does **not** show that those preferences
can be measured or acted on during training.

## Research question

Can a model-local measurement made at training step \(t\) identify a boundary
change that remains useful long enough to improve the model's subsequent
training trajectory?

## Core hypothesis

For each training seed, the current representation induces a non-random,
short-term-stable ranking over local MHAR boundary changes. Selecting a change
from that ranking will lead to lower future NLL than selecting a random or
predicted-bad change.

The hypothesis is model-local. It does not require the same boundary indices to
win across seeds.

## Fixed model and boundary vocabulary

The primary signal experiment uses the 1B native H16 MHAR model:

- hidden size 1,280;
- 36 layers;
- 16 attention heads and 8 KV heads;
- FFN width 5,120;
- 16 MHAR routing groups of width 80;
- sequence length 1,024;
- global batch 32;
- the matched FineWeb-Edu training recipe from Experiment 2.

Write the native H16 partition as

\[
[h_0]|[h_1]|\cdots|[h_{15}],\qquad h_i\in\mathbb R^{80}.
\]

There are 15 removable adjacent boundaries

\[
e_i=(h_i,h_{i+1}),\qquad i=0,\ldots,14.
\]

Removing exactly one boundary produces 15 candidate partitions. Every
candidate has exactly one 160-dimensional group, fourteen 80-dimensional
groups, and fifteen routing softmaxes. Candidate-to-candidate comparisons are
therefore matched in width composition, parameter count, and router count.

The same candidate partition is applied globally at all 73 routing sites.
Site-specific learning is outside this experiment.

## Boundary score

At checkpoint \(M_s(t)\) for seed \(s\), define

\[
\Delta_i^{(s)}(t)
=
NLL(M_s(t);\text{remove }e_i)
-
NLL(M_s(t);\text{native H16}).
\]

Lower is better:

- \(\Delta_i<0\): the frozen removal improves current NLL;
- \(\Delta_i>0\): the frozen removal damages current NLL;
- smaller \(\Delta_i\): a more preferred removal relative to other candidates.

The score is a model-local probe. It is not yet a learned objective and is not
called a causal boundary effect.

## Fixed evaluation data

Before evaluating any candidate, materialize a new Experiment 3 artifact with:

- 512 discovery sequences;
- 512 confirmation sequences;
- sequence length 1,024;
- document-disjoint discovery and confirmation sets;
- source documents excluded from training data and previous Experiment 2
  evaluation artifacts.

Freeze the source shard identities, tokenizer revision, document IDs, tensor
shapes, and SHA-256. Every checkpoint, seed, partition, and branch must see the
same tokens in the same order. Primary metrics are token-weighted NLL and
per-sequence NLL for paired bootstrap inference.

Confirmation data may validate a frozen analysis, but it may not select a
boundary or alter a branch.

## Subexperiments

Experiment 3 is divided into five plans:

1. [Experiment 3A — Boundary Signal Existence](experiment-3a-boundary-signal.md)
2. [Experiment 3B — Short-Term Temporal Stability](experiment-3b-temporal-stability.md)
3. [Experiment 3C — Actionability by Branched Training](experiment-3c-actionability.md)
4. [Experiment 3D — Within-Seed Cross-Seed Replication](experiment-3d-cross-seed.md)
5. [Experiment 3E — Boundary-Landscape Smoothness](experiment-3e-landscape.md)

## Locked execution sequence

1. Freeze the Experiment 3 artifact and candidate enumeration.
2. Run parity tests for native H16 and all single-boundary partitions.
3. Run 3A and 3B for seed 42 at steps 1,000, 1,500, 2,000, and 3,000.
4. If the 3A signal gate and 3B stability gate pass, run the seed-42 3C
   branched-training test.
5. If the seed-42 actionability direction is positive, repeat the complete
   measurement-and-branch procedure for seeds 43 and 44 under 3D.
6. Run 3E to determine whether the repeatable landscape is smooth enough for a
   soft/differentiable parameterization or requires discrete restructuring.
7. Apply the final go/no-go rule before designing a learnable architecture.

Skipping a failed gate requires a new explicitly labeled exploratory plan. It
may not be treated as part of this preregistration.

## Minimum proof chain and gates

### Gate 1 — A measurable preference exists

At the preregistered checkpoint, both must hold:

1. discovery and confirmation candidate rankings have Spearman
   \(\rho\ge 0.5\); and
2. the discovery-selected best-versus-worst contrast has a paired confirmation
   95% bootstrap interval excluding zero in the predicted direction.

### Gate 2 — The preference has short-term stability

For adjacent checkpoint pairs separated by 500 or 1,000 steps:

- every primary adjacent-time Spearman correlation must be positive; and
- the median adjacent-time Spearman correlation must be at least 0.5.

### Gate 3 — The preference is actionable

After branching from the same checkpoint and continuing for the fixed horizon,
the predicted-good branch must beat the random branch in the preregistered
direction. A paired 95% interval excluding zero is the minimum positive result.

Predicted-good versus predicted-bad is the matched diagnostic. Predicted-good
versus unchanged H16 is a practical but router-count-unmatched comparison.

### Gate 4 — The procedure replicates within seeds

Across seeds 42, 43, and 44:

- at least two seeds must pass the signal and stability gates;
- at least two seeds must show predicted-good lower than random at the future
  endpoint; and
- the across-seed mean good-minus-random effect must be negative.

With only three seeds, report every seed and do not use asymptotic seed-level
significance claims.

## Final decision

Proceed to design online learnable MHAR boundaries only if Gates 1–4 pass.

Then use Experiment 3E to select the mechanism family:

| Observed landscape | Mechanism justified for the next experiment |
|---|---|
| repeatable, temporally stable, locally smooth | soft or differentiable boundaries |
| repeatable and stable, but jagged/discrete | periodic discrete search, straight-through, Gumbel, or bandit selection |
| measurable but not actionable | improve the scoring objective; do not learn boundaries yet |
| unstable or non-replicating | stop the learnable-boundary program at this scale |

## Common controls

All subexperiments must preserve:

- model weights at the measured checkpoint;
- query coefficients attached to their original residual coordinates;
- full-width RMS normalization before grouping;
- residual coordinate order;
- the same global partition at all routing sites;
- checkpoint, optimizer, scheduler, RNG, data-position, source-code, and artifact
  hashes;
- bf16 evaluation with identical batching;
- discovery/confirmation separation;
- resumable JSONL results and immutable manifests;
- W&B project `MHAR Stuff` with a dedicated Experiment 3 group.

## Interpretation boundaries

Passing Experiment 3 would establish that a model-local boundary probe can
guide short-horizon training. It would not establish:

- universal semantic boundary coordinates;
- a final learnable-boundary parameterization;
- site-specific or layer-specific optimal partitions;
- long-horizon gains at 20,000 steps;
- improved wall-clock efficiency.

Those require later experiments after the training-time signal is proven.
