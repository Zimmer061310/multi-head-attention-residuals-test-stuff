# Experiment 3D — Cross-Seed Replication Without Universal Boundary IDs

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED
- Version Label: experiment_3d_v1

## Question

Does the same measurement-and-selection procedure identify a useful local
boundary change independently within multiple training seeds?

## Non-hypothesis

This experiment does **not** test whether different seeds choose the same
boundary index. Boundary-ID overlap is descriptive only and cannot determine
success or failure.

## Seeds and matched training

Use seeds:

\[
42,\quad43,\quad44.
\]

For each seed, train the same native H16 architecture under the same recipe and
preserve atomic checkpoints at steps 1,000, 1,500, 2,000, and 3,000. Corpus
shards and global recipe remain fixed, while initialization and stochastic RNG
streams are seed-specific.

## Within-seed procedure

For each seed independently:

1. evaluate all 15 single-boundary removals at all four checkpoints;
2. compute Experiment 3A signal reliability;
3. compute Experiment 3B adjacent-time stability;
4. select predicted-good, predicted-bad, and random controls from that seed's
   step-1,500 discovery scores;
5. branch the exact step-1,500 state and continue each branch to step 2,000;
6. evaluate future branch NLL on the unchanged fixed artifact.

Never use another seed's score, winning boundary, or branch outcome to select a
boundary for the current seed.

## Primary seed-level outcomes

For seed \(s\), record:

- Experiment 3A pass/fail and discovery/confirmation Spearman;
- median adjacent-time Spearman from Experiment 3B;
- selected good, random, and bad boundary IDs;
- future \(NLL_{good}-NLL_{random}\);
- future \(NLL_{good}-NLL_{bad}\);
- future \(NLL_{good}-NLL_{unchanged}\);
- paired sequence-bootstrap intervals.

## Across-seed analysis

Report the three seed rows without pooling them away. Then report:

- mean and median of each seed-level future-NLL contrast;
- number of seeds with the predicted direction;
- range across seeds;
- descriptive boundary-ID overlap;
- a nested bootstrap that resamples seeds first and sequences within seed,
  clearly labeled as unstable with only three seeds.

Do not claim universal boundaries from boundary-ID frequency. Do not treat the
three seeds as hundreds of independent observations by pooling sequences.

## Replication gate

The model-local learning premise passes only if:

1. at least two of three seeds pass the 3A signal gate;
2. at least two of three seeds pass the 3B stability gate;
3. predicted-good beats random at the future endpoint in at least two seeds;
4. the mean seed-level good-minus-random NLL is negative.

Strong replication additionally requires predicted-good to beat predicted-bad
in all three seeds.

## Failure interpretations

- Different winning IDs with successful within-seed prediction: expected and
  supportive of model-local learning.
- Stable signals but failed actionability: the frozen NLL probe is descriptive
  but not a sufficient control objective.
- One successful seed only: insufficient evidence for an architectural
  mechanism.
- No stable within-seed signal: online learning would chase noise at this
  timescale.

## Outputs

- one immutable probe and branch-selection manifest per seed;
- `cross_seed_summary.csv` with all seed-level metrics;
- `cross_seed_summary.json` with the replication gate;
- `fig_cross_seed_actionability.png/.pdf`;
- `fig_cross_seed_boundary_ids.png/.pdf`, explicitly labeled descriptive;
- W&B group containing all seed-specific probe, branch, and analysis runs.

## Interpretation

A pass supports a procedure that learns where **this model instance** wants its
boundaries. It does not support memorizing a universal coordinate partition.
