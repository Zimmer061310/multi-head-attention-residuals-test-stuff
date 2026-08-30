# Experiment 4 — Short-Horizon Actionability: Final Report

## Outcome

All three seed-43 branches completed at step 2,000. There is a small early
advantage for the discovery-selected good boundary over the bad boundary, but
no clearly sustained advantage across the full 500-step horizon. The unchanged
H16 control has the lowest mean logged training loss.

The unchanged frozen decision is `investigate_adaptive_grouping`, because the
predeclared +100-step cumulative A-minus-B bootstrap interval is below zero.
This supports investigating a transient, within-seed signal, not claiming that
changing boundaries already improves over native H16 or that an adaptive
mechanism will work. No further training or new experiment was launched.

## Branches and provenance

| Label | Intervention | Groups | W&B |
|---|---|---:|---|
| A | remove-03: merge atoms 3 and 4 | 15 | [training A](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/l7zsg219) |
| B | remove-13: merge atoms 13 and 14 | 15 | [training B](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/th9sjox7) |
| C | native H16, unchanged | 16 | [training C](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/0bu9k9sc) |

- Parent: seed 43, native H16, step 1,500.
- Parent checkpoint content SHA-256: `ece1370fb201f3ba661424bdb7126a4b1d59221bccba6e46332372e63c9faaeb`.
- Frozen selection manifest SHA-256: `ddb04b9ec66baa3627df8c19fbf4333b0b338e13ed647bb6f328907c05e76ed0`.
- Training source: `a6cdf76762d15ab43a41e89827066b568bdd457e`.
- Three identical starting states; original optimizer, scheduler, RNG, and
  packed-data position restored. Original 20,000-step learning-rate schedule
  retained, with a stop at step 2,000.
- All three branches used the eager implementation, including C; the parent
  was originally trained with fused routing.
- 50 logged optimizer-step losses per branch, at 1,510, 1,520, ..., 2,000.
  These are the losses of the logged optimizer steps, not averages over all
  intervening unlogged steps and not fixed held-out NLL.
- W&B records all three runs as finished with `completed_steps=2000`,
  `latest_checkpoint_step=2000`, and final checkpoint paths.

## Predeclared comparisons

All values below are training NLL differences in nats per token. Negative A-B
favors A. Intervals reproduce the frozen 10,000-resample moving-block bootstrap
with blocks of five logged observations (50 optimizer steps).

| Horizon | Endpoint A-B | Cumulative mean A-B | Cumulative 95% interval | Trailing-100 mean A-B |
|---|---:|---:|---|---:|
| +100 / step 1,600 | -0.002043724 | -0.001747179 | [-0.001925945, -0.000431252] | -0.001747179 |
| +200 / step 1,700 | +0.002378464 | -0.000865531 | [-0.001289868, -0.000140524] | +0.000016117 |
| +300 / step 1,800 | +0.007855415 | -0.000188446 | [-0.000865795, +0.000484150] | +0.001165724 |
| +400 / step 1,900 | +0.000507355 | -0.000378335 | [-0.000817838, +0.000430650] | -0.000948000 |
| +500 / step 2,000 | +0.003745079 | -0.000228777 | [-0.000671293, +0.000352318] | +0.000369453 |

At +100, A is better than B on average. By the second 100-step window their
mean difference is close to zero. The last-100 mean and the single step-2,000
point favor B; the full-horizon interval includes zero. The positive early
result should therefore not be described as a durable advantage.

## Unchanged-control context

| Logged window | Mean A-C | Mean C-B |
|---|---:|---:|
| First 100 optimizer steps | +0.000940371 | -0.002687550 |
| Full 500 optimizer steps | +0.000446291 | -0.000675068 |

Mean losses across all 50 logged points rank **C < A < B**:

| Branch | Mean logged loss | Single step-2,000 loss |
|---|---:|---:|
| A | 4.392125168 | 4.452830315 |
| B | 4.392353945 | 4.449085236 |
| C | 4.391678877 | 4.455806255 |

No superiority over unchanged H16 is established. C is contextual, not matched
to A and B in router count. Confidence intervals reported above concern A-B;
the control means are descriptive.

## Limitations

- One seed, one parent checkpoint, one selected best/worst pair, and no random
  boundary branch: this is not cross-seed evidence or a complete test of an
  online selection policy.
- The early interval uses only ten logged points and a block bootstrap on an
  evolving training trajectory. Treat it as the specified within-run
  diagnostic, not a population-level guarantee.
- A full-horizon interval crossing zero does not establish equivalence.
  No practical-equivalence margin was preregistered.
- Comparing a frozen validation-probe gap directly with a later training-loss
  gap does not estimate a precise adaptation percentage.
- Experiment 3's failed temporal gate remains unchanged. This independently
  authorized Experiment 4 does not retrospectively turn Experiment 3C into a
  completed or successful experiment.

## Operations, recovery, and shutdown

The first startup attempt failed before optimizer updates because Transformers
attempted an unnecessary Hugging Face API request. Its logs/manifests were
archived remotely, not deleted. The replacement launch set offline tokenizer
loading, preserved all scientific inputs, and used new W&B run IDs. This
non-scientific fix was committed and pushed as `a6cdf76`.

The replacement launch began around 2026-08-30 00:16 CST. W&B records the final
A/B losses around 05:20 CST and successful analysis at 05:21 CST. Total elapsed
time to analysis was approximately 5 hours 5 minutes. The success-only
controller was configured to shut down after 70 minutes, around 06:32 CST.

At final inspection, SSH closed connections in three attempts, including one
outside the local sandbox. This is consistent with the scheduled shutdown,
but the provider's power/billing state was not independently verified. The
checkpoint files could not be inspected directly after shutdown and remain on
the server; W&B records their successful save paths.

The live check ran after the shutdown grace. Compact artifacts were therefore
recovered from W&B instead of copied over SSH:

1. Original 50-row paired table and both figures downloaded and SHA-256 checked.
2. Every table loss checked against each branch's complete W&B loss history;
   identical API pagination duplicates were checked before deduplication.
3. Logged data counters equal `step * 32` at every observation for all branches.
4. Frozen horizon summaries recomputed locally; logged early/full means and
   decision reproduced (floating-point serialization tolerance `1e-15`).
5. Selection manifest deterministically reconstructed; its byte-level SHA-256
   exactly matches the hash recorded in all three training configs.

No scientific result, input, threshold, or remote accepted output was changed.
See `RECOVERY_PROVENANCE.json` for the distinction between original downloads
and reconstructed artifacts. Seven focused tests passed during recovery.

## Artifacts

- [W&B analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/42j2s8ip)
- `paired_losses.csv`: all 50 paired observations.
- `short_horizon_summary.json`: reproduced frozen numerical analysis.
- `branch_selection_seed43.json`: hash-verified reconstruction of frozen manifest.
- `wandb_original/`: byte-verified table and figure downloads.
- `wandb_history_A.json`, `wandb_history_B.json`, `wandb_history_C.json`.
- `wandb_run_metadata.json`: recorded configs and completion summaries.

![Paired training loss differences](fig_short_horizon_delta.png)
