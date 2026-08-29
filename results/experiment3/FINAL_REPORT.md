# Experiment 3 final report: boundary learnability prerequisites

## Outcome

The experiment found a strong, reproducible **within-checkpoint boundary
preference**, but not a sufficiently stable **across-checkpoint ranking**.

- All three seeds passed the frozen boundary-signal gate at step 1,500.
- Zero of three seeds passed the frozen temporal-stability gate.
- Experiment 3C actionability branching was therefore not authorized.
- The formal Experiment 3D replication gate was not evaluated because it
  requires the actionability stage.

At this scale and under this protocol, a boundary learner would have a real
instantaneous objective, but the preferred action changes too quickly or too
irregularly for the preregistered online-learning justification.

## Frozen gate results

| Seed | Signal Spearman | Best | Worst | Confirmation best-worst delta NLL | 95% CI | Signal | Adjacent temporal Spearman | Median | Temporal |
|---:|---:|---|---|---:|---|---|---|---:|---|
| 42 | 0.9786 | `remove-06` | `remove-01` | -0.027385 | [-0.030164, -0.024788] | pass | 0.4679, 0.4321, 0.3821 | 0.4321 | fail |
| 43 | 0.9393 | `remove-03` | `remove-13` | -0.011057 | [-0.012314, -0.009783] | pass | 0.6179, 0.6000, -0.1214 | 0.6000 | fail |
| 44 | 0.9286 | `remove-03` | `remove-13` | -0.007270 | [-0.008352, -0.006204] | pass | 0.3750, 0.3607, 0.3250 | 0.3607 | fail |

All three confirmation sequences agreed with the sign of the corresponding
discovery temporal correlations. Seed 43 nevertheless failed because its
2,000-to-3,000 discovery correlation was negative; seeds 42 and 44 failed the
minimum median-correlation threshold.

## Interpretation

The result separates three claims that should not be conflated:

1. **Partition choice matters:** already established by Experiments 1 and 2.
2. **A local training-time signal exists:** supported in all three seeds here.
3. **The signal is stable enough to drive online restructuring:** not supported.

Seeds 43 and 44 independently selected the same step-1,500 best and worst
boundary removals, but this does not establish universal coordinates. The
temporal failures show that a high-quality snapshot ranking can still age
poorly. The completed H8 landscape control remains locally smooth at its frozen
checkpoint, so spatial smoothness and temporal stability are distinct
properties.

## Step-1,500 rankings

Best to worst, discovery frozen first and confirmation second:

- Seed 42 discovery: `06, 08, 05, 07, 09, 11, 10, 14, 12, 03, 13, 04, 02, 00, 01`.
- Seed 42 confirmation: `06, 08, 07, 05, 11, 09, 12, 10, 14, 03, 13, 04, 00, 02, 01`.
- Seed 43 discovery: `03, 04, 10, 11, 02, 06, 09, 05, 01, 08, 07, 00, 12, 14, 13`.
- Seed 43 confirmation: `03, 04, 10, 09, 05, 11, 06, 02, 01, 08, 07, 00, 14, 12, 13`.
- Seed 44 discovery: `03, 11, 07, 06, 04, 08, 00, 02, 09, 14, 05, 12, 01, 10, 13`.
- Seed 44 confirmation: `03, 11, 06, 07, 04, 08, 09, 14, 05, 02, 00, 12, 10, 13, 01`.

Each number denotes the left atomic-block index of the removed H16 boundary.

## W&B analysis runs

- Seed 42: [signal](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/5u6yzo8x), [temporal](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/imqe8s29).
- Seed 43: [signal](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/2v03rk9q), [temporal](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/5xon3hn8).
- Seed 44: [signal](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/m74ipofe), [temporal](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/ihol4try).
- H8 landscape: [analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/s6igfygd).

Final step-3,000 split runs:

- Seed 43: [discovery](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/6eco0u1t), [confirmation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/6m9qste3).
- Seed 44: [discovery](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/74d5ytq1), [confirmation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/s2ozkusl).

## Integrity and operational deviations

- Fixed artifact SHA-256:
  `1a239dfc65c3b4f9184ccbdc28e9165b4bc452312a2d7360d6b7519fafb1a5af`.
- Locked training commit:
  `81ff30572d5dd5dadba715290897d6b10aa58587`.
- All reported final measurements use the immutable step-3,000 checkpoints.
- The external stop watcher failed to terminate all three trainers promptly.
  Seed 42 reached approximately step 3,020; seeds 43 and 44 reached
  approximately step 3,030. Each process was manually stopped only after its
  complete step-3,000 checkpoint manifest, state, model, step, and seed were
  verified. No protected checkpoint was replaced or modified.
- A NumPy integer JSON-serialization defect in temporal reporting was fixed in
  commit `4b6c068`; the fix changed serialization only and all frozen inputs and
  computed statistics were preserved.
