# Experiment 4 — Short-Horizon Boundary Actionability

## Question

Does the boundary ranked best by a frozen step-1,500 probe produce a measurable
optimization advantage over the boundary ranked worst when both are trained on
the same next 500 steps?

This is a new experiment authorized after Experiment 3. It does not retroactively
change the Experiment 3 temporal-stability gate or its negative conclusion.

## Frozen source and branches

Use the native-H16 seed-43 step-1,500 checkpoint. Its step-1,500 discovery probe
froze the following choices before any branch training:

| Label | Branch role | Routing intervention |
|---|---|---|
| A | predicted-good | remove boundary 03 (`h3` and `h4` share one router) |
| B | predicted-bad | remove boundary 13 (`h13` and `h14` share one router) |
| C | unchanged | native H16 |

The seed-43 probe had discovery-confirmation Spearman 0.9393 and a confirmation
A-minus-B NLL of -0.01106 (95% CI [-0.01231, -0.00978]). This information is
used only to freeze A and B.

Create three immutable, content-identical copies of the complete step-1,500
checkpoint. Each copy must contain identical model weights, AdamW state,
scheduler state, RNG state, data-stream position, and training identity. Record
and verify one content-tree SHA-256 for all three copies before launch.

## Training protocol

- Run A, B, and C simultaneously on three RTX 5090 GPUs.
- Restore every branch from its own identical step-1,500 copy.
- Preserve the original seed-43 FineWeb-Edu recipe and 20,000-step LR schedule.
- Do not reset optimizer moments, scheduler, RNG, or data position.
- Feed all branches the same examples in the same order.
- Change only the static global routing partition.
- Train exactly 500 optimizer steps, stopping atomically at step 2,000.
- Log training loss every 10 optimizer steps, yielding 50 paired observations.
- Save a complete step-2,000 checkpoint for each branch.
- Do not continue beyond step 2,000 under this protocol.

## Frozen analysis

At every logged step, compute:

\[
\Delta_{A-B}(t)=L_A(t)-L_B(t),
\quad
\Delta_{A-C}(t)=L_A(t)-L_C(t),
\quad
\Delta_{C-B}(t)=L_C(t)-L_B(t).
\]

Report the raw paired loss at all 50 points and predeclared summaries at +100,
+200, +300, +400, and +500 steps. For each horizon, report:

1. the endpoint loss difference;
2. the cumulative paired mean from step 1,510 through that horizon;
3. a paired moving-block-bootstrap 95% interval for the cumulative A-minus-B
   mean, using 50-step blocks and 10,000 resamples;
4. the trailing-100-step paired mean.

The +100-step summary is the early-adaptation diagnostic. The +500-step
cumulative mean is the primary endpoint. Horizons may not be selected post hoc.

## Decision rule

Clear short-horizon actionability is present if the cumulative A-minus-B mean is
negative and its paired 95% interval is entirely below zero at either the
predeclared +100-step diagnostic or the +500-step primary endpoint.

Directional but uncertain actionability means the paired mean is negative but
the interval includes zero. No actionability means A and B are practically
indistinguishable or A is worse at both +100 and +500.

- Clear or consistently directional A < B supports investigating continuously
  learnable/adaptive grouping.
- A approximately equal to B by +100 and through +500 indicates rapid
  co-adaptation and argues against the boundary-learning direction.

C is contextual: it shows whether either intervention improves on leaving H16
unchanged. It is not router-count matched to A and B.

## Outputs

- frozen branch manifest and checkpoint-copy hashes;
- three W&B training runs in project `MHAR Stuff`;
- three `training_metrics.jsonl` files;
- `paired_losses.csv` with all 50 points;
- `short_horizon_summary.json`;
- `fig_short_horizon_losses.png` and `fig_short_horizon_delta.png`;
- one W&B analysis run;
- three complete atomic step-2,000 checkpoints.

## Interpretation limit

A positive result proves only that a locally measured boundary preference can
predict a short-horizon training advantage for this seed and checkpoint. It does
not prove that one fixed boundary remains optimal, that boundary IDs transfer
across seeds, or that a particular adaptive mechanism will succeed.
