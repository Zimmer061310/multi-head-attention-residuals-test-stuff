# Experiment 8 — Hybrid-Q8 / Global-KV

## Result

Frozen decision: `eligible_for_review`. Within-seed assessment: `within_seed_practical_match`.

This is a seed-42 step-2,000 screen, not a convergence or multi-seed claim.
Negative delta NLL favors the first named model.

## Paired fixed-validation contrasts

| Contrast | Role | Split | Delta NLL | 95% CI |
|---|---|---|---:|---:|
| hq8-minus-m8 | primary | discovery | -0.000241 | [-0.002690, +0.002265] |
| hq8-minus-m8 | primary | confirmation | -0.000671 | [-0.002908, +0.001551] |
| bhq8-minus-b | control | discovery | +0.020296 | [+0.017682, +0.022925] |
| bhq8-minus-b | control | confirmation | +0.021492 | [+0.019057, +0.023976] |
| h8-interaction | primary_interaction | discovery | -0.020537 | [-0.023900, -0.017188] |
| h8-interaction | primary_interaction | confirmation | -0.022163 | [-0.025593, -0.018753] |
| hq8-minus-lq8 | endpoint_diagnostic | discovery | -0.058302 | [-0.061449, -0.055317] |
| hq8-minus-lq8 | endpoint_diagnostic | confirmation | -0.054724 | [-0.057442, -0.052069] |
| bhq8-minus-blq8 | endpoint_control_diagnostic | discovery | -0.039475 | [-0.042196, -0.036792] |
| bhq8-minus-blq8 | endpoint_control_diagnostic | confirmation | -0.037657 | [-0.040876, -0.034741] |
| mhar-midpoint-curvature | shape_diagnostic | discovery | -0.058542 | [-0.062943, -0.054249] |
| mhar-midpoint-curvature | shape_diagnostic | confirmation | -0.055395 | [-0.059598, -0.051198] |
| baseline-midpoint-curvature | shape_control_diagnostic | discovery | -0.019179 | [-0.023651, -0.014755] |
| baseline-midpoint-curvature | shape_control_diagnostic | confirmation | -0.016165 | [-0.020647, -0.011792] |

## New run links

- HQ8: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/dwied13n
- BHQ8: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/01qke5pe
- Analysis: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/8r9kzimc

## Interpretation limits

- single seed at step 2000; reject catastrophic designs and report within-seed evidence only; do not declare convergence or a multi-seed winner
- The paired intervals quantify held-out sequence sampling uncertainty, not training-seed uncertainty.
- Parameter/MAC reports cover physical projection structure; they are not measured whole-model speedups.
- No continuation or multi-seed experiment is authorized by this result.
