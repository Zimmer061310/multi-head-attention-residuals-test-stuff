# Experiment 2 step-2,000 final report

## Outcome

The constrained additive boundary score transfers beyond its fitted `k=4`
population. Every tested merge count has a positive rank correlation between
the frozen additive score and measured confirmation-set delta NLL. The
relationship is strongest near `k=4` and remains useful at the endpoints.

All evaluated mixed-width interventions still have positive delta NLL versus
native H16. This result establishes predictable relative partition quality for
the frozen checkpoint; it does not show that a mixed-width model trained from
scratch would outperform native H16.

## Registered k=3 and k=5 transfer

| k | Uniform n | Spearman | Target top mean delta NLL | Middle | Bottom | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 3 | 30 | 0.920801 | 0.316325 | 0.392659 | 0.469295 | pass |
| 5 | 30 | 0.930590 | 1.019556 | 1.172641 | 1.452828 | pass |

Both preregistered directional gates passed: uniform-sample correlation was
positive, and predicted-top candidates had lower mean delta NLL than both the
middle and bottom strata.

## Conditional sequential follow-up

This stage reused the fixed confirmation split after the `k=3/5` gate was
observed. It is sequential follow-up evidence, not another untouched
confirmation experiment.

| k | Rank sample | Evaluated candidates | Spearman | Best measured delta NLL | Worst measured delta NLL |
|---:|:---|---:|---:|---:|---:|
| 1 | exhaustive | 15 | 0.614286 | 0.049573 | 0.081689 |
| 2 | uniform | 43 unique, 30 uniform | 0.836263 | 0.139690 | 0.231398 |
| 6 | uniform | 48 unique, 30 uniform | 0.880756 | 1.619806 | 2.236833 |
| 7 | exhaustive | 36 | 0.761133 | 2.533600 | 3.148599 |

Targeted sequential strata remained correctly ordered:

| k | Top mean delta NLL | Middle | Bottom | Top minus middle | Top minus bottom |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.159192 | 0.192414 | 0.220576 | -0.033223 | -0.061384 |
| 6 | 1.694386 | 1.820122 | 2.190767 | -0.125737 | -0.496381 |

## Best measured boundary sets

| k | Removed H16 boundaries | Predicted rank | Delta NLL |
|---:|:---|---:|---:|
| 1 | `[14]` | 3 | 0.049573 |
| 2 | `[6, 14]` | 1 | 0.139690 |
| 3 | `[6, 8, 14]` | targeted top | 0.295046 |
| 5 | `[2, 6, 9, 11, 14]` | targeted top | 0.975182 |
| 6 | `[0, 2, 6, 9, 11, 14]` | 5 | 1.619806 |
| 7 | `[0, 2, 5, 7, 9, 11, 14]` | 4 | 2.533600 |

Boundary 14 appears in every measured winner and boundary 6 appears in the
winner for `k=2,3,5,6`. This agrees with the fitted leading relative boundary
ordering `(6,7)`, `(5,6)`, `(14,15)`, `(13,14)`. These are descriptive frozen
checkpoint regularities, not independently identified causal boundary effects.

## Model diagnostics

- Constrained additive `k=4` five-fold CV: R2 `0.932520`, Spearman `0.971511`.
- Prediction-only ridge nested CV: R2 `0.998096`, Spearman `0.999018`.
- The ridge result shows secondary predictable combinatorial structure, but its
  aliased interaction coefficients were not interpreted or transferred.

## W&B runs

- [Boundary model fit](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/xibx5z5g)
- [k=3 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/ew5kihwr)
- [k=5 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/z64ynlsi)
- [Registered transfer analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/6oq614na)
- [Conditional manifest preparation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/mmqqqjkj)
- [k=1 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/vs0b86te)
- [k=2 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/1p6w116o)
- [k=6 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/87g16hn2)
- [k=7 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/2mcffrew)
- [Conditional follow-up analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/g7a0tv6g)

## Immutable provenance

- Checkpoint: step 2,000 H16 checkpoint.
- Fixed evaluation artifact SHA-256:
  `80276b413b27c2da2ed7bc3b1121536f9bc7763b3cc99c3f86e49a83e5705cb3`.
- Primary transfer implementation commit:
  `cfc73eee198fef959aebadbffcc7ba3e214b7aa9`.
- Conditional follow-up implementation commit:
  `d2ec6a6b7f36c20573cce7e1bd522ff429a696d9`.
