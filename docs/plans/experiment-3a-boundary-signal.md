# Experiment 3A — Boundary Signal Existence During Training

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED
- Version Label: experiment_3a_v1

## Question

At a fixed training step, do local single-boundary removals have a repeatable,
non-random ordering, or are their measured NLL differences evaluation noise?

## Hypothesis

The 15 single-boundary removals of a native H16 checkpoint will produce a
structured ranking that replicates on document-disjoint confirmation data.

## Checkpoints

Use seed 42 H16 checkpoints at steps:

\[
t\in\{1000,1500,2000,3000\}.
\]

Step 1,500 is the primary preregistered checkpoint because it permits a matched
500-step actionability branch to step 2,000. The other checkpoints are required
for Experiment 3B.

If an existing checkpoint lacks complete optimizer, scheduler, RNG, and packed
data-position state, it may be used for frozen 3A evaluation but not as the
source of a 3C training branch.

## Candidate set

For each checkpoint evaluate:

- native H16 once;
- remove boundary 0;
- remove boundary 1;
- ...;
- remove boundary 14.

Every removal creates one adjacent 160-dimensional group and fourteen
80-dimensional groups. No coordinate permutation or non-adjacent grouping is
allowed.

## Evaluation procedure

1. Verify checkpoint and fixed-artifact hashes.
2. Verify eager native-H16 parity against the fused native path.
3. Evaluate native H16 on discovery.
4. Evaluate all 15 candidates on the same discovery sequences.
5. Save token-weighted and per-sequence NLL.
6. Freeze the discovery ranking, best boundary, worst boundary, and top-three
   set in an immutable selection manifest.
7. Evaluate the frozen complete candidate set on confirmation. The full set is
   retained so discovery/confirmation rank reliability can be estimated; the
   confirmation result may not revise the selection manifest.

## Primary estimands

For each boundary \(i\):

\[
\Delta_i(t)=NLL_i(t)-NLL_{native}(t).
\]

Report:

- all 15 \(\Delta_i(t)\) values;
- discovery/confirmation Spearman correlation;
- discovery-selected best-minus-worst NLL on confirmation;
- paired sequence-bootstrap 95% interval for that fixed contrast;
- top-three overlap between discovery and confirmation;
- median and range of candidate \(\Delta NLL\).

Use 10,000 paired bootstrap samples with a frozen bootstrap seed. Bootstrap
sequences or documents, never individual tokens.

## Decision rule

Boundary preference is present at step 1,500 only if both hold:

1. discovery/confirmation Spearman \(\rho\ge0.5\); and
2. the discovery-selected best-minus-worst confirmation interval is entirely
   below zero.

Report results at the other checkpoints without changing this primary gate.
If the primary gate fails, Experiment 3C is not authorized by this plan.

## Required controls

- Duplicate native evaluation must reproduce exactly or within a frozen
  numerical tolerance established by the parity test.
- Candidate iteration order must be permuted by a frozen seed so temperature or
  process drift cannot align with boundary index.
- Reevaluate native H16 at the end of the run as a drift sentinel.
- Record GPU model, dtype, batch size, CUDA/PyTorch versions, elapsed time, and
  peak memory.

## Outputs

- `signal_results.jsonl`: native and all candidate measurements;
- `signal_run_manifest.json`: immutable identities and hashes;
- `signal_selection_step-1500.json`: discovery-frozen best/worst/top-three;
- `signal_summary.json`: reliability and paired inference;
- `boundary_signal_map.csv`: checkpoint-by-boundary score matrix;
- `fig_boundary_signal_step1500.png/.pdf`: discovery and confirmation scores;
- W&B table containing every candidate, not only winners.

## Interpretation

A pass establishes a measurable model-local frozen preference at a training
checkpoint. It does not show that the preference persists or improves future
training; those are Experiments 3B and 3C.
