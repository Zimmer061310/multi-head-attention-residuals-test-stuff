# Experiment 10 — Per-Group Local/Global Query Contribution

## Question

Inside the accepted Experiment 8 HQ8 model, is the local-query contribution
distributed across all eight MHAR chunk/GQA groups or concentrated in a few
groups?

This is a frozen causal-intervention experiment. It performs no training and
does not modify the accepted checkpoint or fixed evaluation artifact.

## Frozen starting point

- model: accepted Experiment 8 HQ8, seed 42, atomic step 2,000
- checkpoint content SHA-256:
  `74cff0ab19409dac9f6104e8986e4890c0837bc730d8ac233ae18011dbc58333`
- fixed document-disjoint artifact SHA-256:
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`
- reference condition must exactly reproduce the accepted Experiment 8 HQ8
  aggregate and per-sequence NLLs on discovery and confirmation

For each GQA group `g`, even query head `2g` is local and odd query head
`2g+1` is global. All ablations zero complete attention-head outputs
immediately before dense `W_O`, in every layer.

## Experiment 10A — single local-head ablations

Evaluate `remove-local-g0` through `remove-local-g7`. Define

`D_L[g] = NLL(remove-local-g) - NLL(HQ8)`.

Positive values mean that the trained local head contributes useful
computation. Every contrast receives a paired 10,000-sample per-sequence
bootstrap interval on both fixed splits.

## Experiment 10B — matched global-head ablations

Evaluate `remove-global-g0` through `remove-global-g7` and define

`D_G[g] = NLL(remove-global-g) - NLL(HQ8)`.

Report the two eight-element vectors and their discovery/confirmation rank
correlation. The correlation is descriptive because there are only eight
groups.

## Experiment 10C — whole-group ablations

Evaluate `remove-group-g0` through `remove-group-g7`, zeroing both query-head
outputs in one GQA group. Define `D_GL[g]` and the interaction

`I[g] = D_GL[g] - D_L[g] - D_G[g]`.

Positive `I[g]` means joint removal is more damaging than the sum of the two
single removals; negative means subadditivity or redundancy. Report paired
bootstrap intervals for `I[g]`.

## Frozen descriptive classification

A local group is called useful only if its discovery delta is positive, its
confirmation delta is at least 0.001 NLL, and its confirmation paired-bootstrap
95% CI lower bound is above zero.

Among positive confirmation `D_L` values, define the top-two share as the sum
of the two largest divided by the total.

- `distributed`: at least six useful groups and top-two share at most 0.50
- `concentrated`: at most four useful groups or top-two share at least 0.60
- otherwise: `mixed_or_inconclusive`

These thresholds summarize the map; the full vectors and intervals remain the
primary result.

Also compare the sum of single-local damages with Experiment 9's accepted
all-local damage. Their difference is a descriptive collective-dependence
diagnostic, not an additivity assumption.

## Frozen Experiment 10D gate — one-group alignment

Run 10D only if at least one local group passes the useful-group rule above.
For every target group, exhaust all seven incorrect source chunks while leaving
the other seven local assignments aligned. This produces 56 conditions with no
sampling discretion.

For each target group report the mean substituted-minus-aligned NLL, a
two-stage bootstrap over its seven substitutions and held-out sequences, and
the fraction of substitutions that hurt. This identifies which local heads
depend most strongly on their exact MHAR chunk.

## Limits

This is one trained seed at one checkpoint. The interventions are
off-distribution and do not show how a retrained heterogeneous architecture
would perform. Sequence bootstrap intervals do not include training-seed
uncertainty. Group indices are coordinates inside this specific trained model,
not universal MHAR locations.
