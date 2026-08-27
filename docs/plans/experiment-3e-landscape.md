# Experiment 3E — Local Boundary-Landscape Smoothness

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED
- Version Label: experiment_3e_v1

## Question

Does moving a boundary by a small coordinate distance change NLL gradually, or
is the repeatable landscape too jagged for a soft/differentiable boundary?

## Role in Experiment 3

This experiment selects a mechanism family only after a useful boundary signal
has been established. Smoothness alone does not justify boundary learning.

## Source model

Use the seed-42 native H8 step-2,000 checkpoint because H8 is the strongest
matched baseline at the completed Stage B milestone and has seven internal
boundaries that can be moved while preserving eight total routing groups.

The native boundaries are:

\[
b_j=160j,\qquad j=1,\ldots,7.
\]

## General contiguous-router requirement

The current singleton/doubleton 80-dimensional router is insufficient for this
test. Implement a separate eager-only general contiguous router that accepts
ordered segment endpoints at arbitrary coordinate indices.

It must preserve:

- full-width RMS normalization before segmentation;
- each query coefficient's original coordinate attachment;
- original residual coordinate order;
- one depth softmax per contiguous segment;
- scatter-free ordered output concatenation.

Before scientific evaluation, the uniform endpoints
`[0,160,320,...,1280]` must reproduce native H8 logits and gradients within a
frozen tolerance. Do not modify the fused Triton kernel for this first probe.

## Boundary-movement grid

Move exactly one H8 boundary at a time while holding the other six fixed.
For every boundary \(b_j\), evaluate:

\[
\delta\in\{-40,-30,-20,-10,0,10,20,30,40\}
\]

dimensions, so the tested location is \(b_j+\delta\).

All resulting neighboring groups remain at least 120 dimensions wide. Every
candidate retains eight routing groups and all 1,280 coordinates. The complete
set contains 56 non-native candidates plus one shared native H8 baseline:

\[
7\times8+1=57.
\]

Apply the moved boundary globally at all 73 routing sites.

## Metric

For boundary \(j\) and offset \(\delta\), define

\[
\Delta NLL_j(\delta)
=NLL(b_j+\delta)-NLL(b_j).
\]

Evaluate the complete grid on the frozen Experiment 3 discovery and
confirmation sets. Randomize candidate execution order with a frozen seed and
reevaluate native H8 at the end as a drift sentinel.

## Smoothness diagnostics

For each boundary and split report:

1. the complete offset curve;
2. adjacent 10-dimensional first differences;
3. discrete second differences;
4. a quadratic fit and leave-one-offset-out prediction \(R^2\);
5. the discovery/confirmation correlation of the eight nonzero offsets;
6. the offset of the observed local minimum.

Define normalized roughness:

\[
Q_j=
\frac{\operatorname{median}_m
|L_{m+1}-2L_m+L_{m-1}|}
{\max_m L_m-\min_m L_m+10^{-12}}.
\]

This statistic is descriptive and must be reported with the raw curves.

## Mechanism-selection rule

Classify the landscape as **soft-learning compatible** only if:

- the frozen offset curves replicate across discovery and confirmation with
  median within-boundary Spearman at least 0.5;
- the median leave-one-offset-out quadratic \(R^2\) is at least 0.5; and
- the median normalized roughness \(Q_j\) is at most 0.25.

If curves replicate but either shape threshold fails, prefer a discrete method:
periodic search, straight-through/Gumbel selection, or a bandit-style update.

If the curves do not replicate or their best-versus-worst paired interval does
not exclude zero, conclude that this grid provides no usable landscape evidence.

The thresholds are operational heuristics for mechanism selection, not claims
of mathematical differentiability.

## Outputs

- `boundary_move_results.jsonl` for all 57 choices;
- `boundary_move_manifest.json` with checkpoint, artifact, code, and grid hashes;
- `boundary_landscape_metrics.csv`;
- `boundary_landscape_summary.json` with the mechanism classification;
- `fig_boundary_landscape_small_multiples.png/.pdf` with one panel per boundary;
- `fig_boundary_landscape_roughness.png/.pdf`;
- W&B table with exact boundary locations, offsets, and NLL.

## Interpretation boundaries

A smooth repeatable curve makes a soft parameterization plausible; it does not
prove that gradient descent through such a parameterization will work. A jagged
curve does not reject boundary learning, but it directs the next experiment
toward discrete periodic restructuring rather than continuous boundary motion.
