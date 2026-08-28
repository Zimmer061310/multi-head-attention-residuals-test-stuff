# Experiment 3 seed-42 checkpoint report

## Decision

The frozen boundary signal exists at step 1,500, but it does not satisfy the
preregistered short-term temporal-stability gate. Experiment 3C actionability
branching is therefore not authorized by the current plan.

## Boundary-signal gate

- discovery/confirmation Spearman: `0.9785714286`;
- discovery-selected best removal: `remove-06`;
- discovery-selected worst removal: `remove-01`;
- confirmation best-minus-worst mean delta NLL: `-0.0273847682`;
- paired-bootstrap 95% CI: `[-0.0301638782, -0.0247883688]`;
- gate: **pass**.

W&B: <https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/5u6yzo8x>

## Temporal-stability gate

Adjacent discovery Spearman values for steps 1,000 to 1,500, 1,500 to 2,000,
and 2,000 to 3,000 were respectively:

- `0.4678571429`;
- `0.4321428571`;
- `0.3821428571`.

The median was `0.4321428571`, below the frozen minimum of `0.5`. All three
confirmation comparisons had the same correlation sign, but that does not
override the median-rank-stability requirement. Gate: **fail**.

W&B: <https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/imqe8s29>

## H8 boundary landscape

The separate mechanism-selection control was replicable and locally smooth:

- median discovery/confirmation Spearman: `1.0`;
- median quadratic LOOCV R-squared: `0.9548871686`;
- median normalized roughness: `0.1108531164`;
- confirmation best-minus-worst mean delta NLL: `-0.0865728529`;
- paired-bootstrap 95% CI: `[-0.0897100431, -0.0835099603]`.

W&B: <https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/s6igfygd>

## Operational notes

- All analyses use the immutable step-3,000 checkpoint and fixed evaluation
  artifact SHA-256
  `1a239dfc65c3b4f9184ccbdc28e9165b4bc452312a2d7360d6b7519fafb1a5af`.
- The seed-42 stop watcher exited late. Training reached approximately step
  3,020 before manual termination; the saved step-3,000 checkpoint was not
  modified or replaced.
- Temporal reporting initially exposed a NumPy integer JSON-serialization bug.
  The serialization-only fix was committed separately, tested, and the frozen
  analysis was rerun without changing inputs or statistics.
