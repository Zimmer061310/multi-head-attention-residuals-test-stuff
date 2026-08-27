# Experiment 3B — Short-Term Temporal Stability of Boundary Preference

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED
- Version Label: experiment_3b_v1

## Question

Does a boundary removal ranked as good at step \(t\) remain relatively good
after another 500–1,000 training steps?

## Hypothesis

Boundary-score rankings evolve gradually enough that a measurement can guide a
short-horizon training decision.

## Inputs

Use the complete 15-boundary score vectors produced by Experiment 3A at:

\[
t\in\{1000,1500,2000,3000\}.
\]

The score at each time is measured relative to that checkpoint's own native
H16 baseline. Never correlate raw candidate NLL across checkpoints.

## Primary comparisons

The primary adjacent-time pairs are:

\[
(1000,1500),\quad(1500,2000),\quad(2000,3000).
\]

For each pair \((t,u)\), compute:

1. Spearman rank correlation
   \(\rho(\Delta(t),\Delta(u))\);
2. Pearson correlation as a scale-sensitive secondary metric;
3. top-three Jaccard overlap;
4. sign agreement relative to the within-checkpoint median;
5. future regret of the boundary selected at \(t\):

\[
R_{t\rightarrow u}
=
\Delta_{i^*(t)}(u)-\min_i\Delta_i(u),
\qquad
i^*(t)=\arg\min_i\Delta_i(t).
\]

Low regret means the earlier choice remains close to the later optimum even if
the exact winner changes.

## Confirmation discipline

Run the analysis separately on discovery and confirmation scores. Discovery is
primary for selection and actionability. Confirmation estimates whether the
temporal pattern is data-split dependent.

Do not average discovery and confirmation NLL or use confirmation to replace a
discovery-selected boundary.

## Stability gate

Short-term stability passes for seed 42 only if:

- all three primary discovery Spearman correlations are positive;
- their median is at least 0.5; and
- the corresponding confirmation correlations have the same sign in at least
  two of the three intervals.

The exact correlations, overlaps, and regrets must be reported even if the gate
fails. No threshold may be changed after viewing the results.

## Secondary timescale analysis

Report, but do not use for the primary gate:

- \((1000,2000)\);
- \((1000,3000)\);
- \((1500,3000)\).

These distinguish a useful short-lived signal from a universal fixed-boundary
claim. Decay over longer intervals is compatible with online learning.

## Outputs

- `temporal_correlations.csv` with split, seed, time pair, correlation, and
  overlap metrics;
- `temporal_regret.csv` with the selected boundary and future regret;
- `temporal_summary.json` with the frozen gate result;
- `fig_boundary_score_trajectories.png/.pdf` showing all 15 score trajectories;
- `fig_temporal_stability_matrix.png/.pdf` showing rank correlations by time;
- W&B tables and figures linked to the same Experiment 3 group.

## Interpretation

A pass means the model-local ranking is stable over the horizon needed to make
a boundary update. It does not show that following the ranking changes the
future training outcome. That claim is reserved for Experiment 3C.
