# Experiment 9 — HQ8 Local-vs-Global Head Contribution

## Frozen result

Experiment 9A local-contribution gate: `pass`.
The unchanged condition exactly reproduced the accepted Experiment 8 HQ8 result.
Positive delta NLL means the removed heads helped; negative means the ablation improved NLL.

## Structured contribution ablations

| Ablation | Split | Delta NLL | Paired 95% CI | Balanced-mask percentile |
|---|---|---:|---:|---:|
| zero local | discovery | +0.174576 | [+0.169265, +0.179956] | 0.0% |
| zero local | confirmation | +0.165466 | [+0.161240, +0.169977] | 0.0% |
| zero global | discovery | +1.515075 | [+1.494105, +1.536195] | 100.0% |
| zero global | confirmation | +1.494456 | [+1.472052, +1.516987] | 100.0% |

The 70 exhaustive balanced masks remove four local and four global heads, with exactly one removed head per GQA group.
Their confirmation delta-NLL median is +0.489875 (IQR [+0.441395, +0.556089]).

## Local versus global population

Positive values mean removing local heads hurts more than removing global heads.

| Split | Local-damage minus global-damage | Paired 95% CI |
|---|---:|---:|
| discovery | -1.340499 | [-1.359873, -1.321467] |
| confirmation | -1.328989 | [-1.349194, -1.308570] |

## Local-chunk alignment

Alignment evidence: `pass`.

| Split | Mean derangement delta NLL | Two-stage 95% CI | Positive fraction |
|---|---:|---:|---:|
| discovery | +0.070537 | [+0.068522, +0.072620] | 100.0% |
| confirmation | +0.067644 | [+0.065945, +0.069380] | 100.0% |

## Figures

- [Structured contribution ablations against 70 matched masks](fig_head_contribution.pdf)
- [Frozen local-chunk derangements](fig_local_chunk_alignment.pdf)

## W&B

- 9A analysis: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/7q07hopp
- 9B analysis: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/6pe9o9po

## Interpretation limits

- This is one trained seed at one checkpoint.
- Head zeroing is an off-distribution intervention, not retraining without those heads.
- Confidence intervals quantify held-out sequence sampling; they do not quantify seed uncertainty.
- Experiment 9B uses 32 frozen sampled derangements, not all 14,833 derangements.
- This result does not authorize additional training or architecture experiments.

Figures are generated reproducibly by `figures/gen_fig_experiment9_head_contribution.py`.
