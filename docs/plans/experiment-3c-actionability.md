# Experiment 3C — Actionability by Branched Training

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED
- Version Label: experiment_3c_v1

## Question

Does following a boundary preference measured at step 1,500 improve NLL after
another 500 training steps?

## Authorization gate

Run this experiment only after seed 42 passes the Experiment 3A signal gate and
Experiment 3B temporal-stability gate. The branch selection is generated from
step-1,500 discovery scores only.

## Shared branch source

Start every branch from the exact same seed-42 native H16 step-1,500 checkpoint,
including:

- model weights;
- AdamW state;
- scheduler state;
- RNG state;
- packed-data iterator position;
- training recipe and source commit.

Continue every branch through the same examples, in the same order, to atomic
step 2,000. Branches differ only in the static global MHAR partition.

## Frozen branch selection

Before starting any branch, write one immutable manifest with four roles:

| Branch | Selection rule | Routing groups |
|---|---|---:|
| predicted-good | lowest step-1,500 discovery \(\Delta_i\) | 15 |
| predicted-bad | highest step-1,500 discovery \(\Delta_i\) | 15 |
| random | frozen uniform draw from boundaries ranked 5–11, excluding good/bad | 15 |
| unchanged | native H16 partition | 16 |

The random seed must be fixed before reading step-1,500 scores. Drawing from the
middle ranks prevents an accidental duplicate of the intentionally good or bad
control while remaining independent of boundary identity.

The good, bad, and random branches are exactly matched in group-width
composition, parameter count, and router count. The unchanged branch answers a
practical question but is not router-count matched.

## Training controls

- Use the same future FineWeb-Edu token stream for all four branches.
- Preserve the original 20,000-step learning-rate schedule; do not restart or
  rewarm it.
- Do not reset optimizer moments when changing the partition because parameter
  identities and shapes are unchanged.
- Use separate output directories and W&B run IDs.
- Save an atomic step-2,000 checkpoint and complete run identity for every
  branch.
- Do not continue beyond step 2,000 under this plan.

## Evaluation

At step 2,000 evaluate all four trained branches on the fixed Experiment 3
discovery and confirmation sets. Candidate selection remains frozen from the
step-1,500 discovery probe.

Primary endpoint:

\[
NLL_{good}(2000)-NLL_{random}(2000)
\]

Secondary endpoints:

\[
NLL_{good}-NLL_{bad},\qquad
NLL_{good}-NLL_{unchanged}.
\]

Also report training loss over the 500-step branch horizon, but do not substitute
training loss for fixed-data NLL.

## Statistical analysis

For every endpoint report:

- token-weighted NLL difference;
- per-sequence paired-bootstrap 95% interval with 10,000 samples;
- discovery and confirmation estimates separately;
- the complete four-branch ranking.

Do not perform a post-hoc search over branch horizon, checkpoint, or random
control.

## Decision rule

Minimum actionable result:

\[
NLL_{good}<NLL_{random}
\]

on confirmation with the paired 95% interval entirely below zero.

Strong result:

\[
NLL_{good}<NLL_{random}<NLL_{bad}
\]

and predicted-good also beats unchanged H16.

If good beats bad but not random, label the signal directionally informative but
not yet operationally actionable. If good fails to beat both random and bad, do
not proceed to a learnable-boundary architecture.

## Optional preregistered horizon replication

Only after the 500-step result is frozen, an explicitly pre-authorized secondary
replication may restart the same four branches from step 1,500 and continue to
step 2,500. It must use the same selection manifest and may not replace the
primary 500-step endpoint.

## Outputs

- `branch_selection_seed42.json`;
- four atomic checkpoints and run manifests;
- `branch_training_metrics.jsonl`;
- `actionability_results.json`;
- `actionability_contrasts.csv`;
- `fig_actionability_future_nll.png/.pdf`;
- `fig_actionability_training_curves.png/.pdf`;
- four training runs and one analysis run in W&B project `MHAR Stuff`.

## Interpretation

A pass establishes that the measured boundary signal can guide a short-horizon
training intervention for seed 42. It does not establish cross-seed reliability;
that is Experiment 3D.
