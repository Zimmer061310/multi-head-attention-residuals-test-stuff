# Experiment 10 — Per-Group Local/Global Query Contribution

## Frozen result

The unchanged condition exactly reproduced the accepted Experiment 8 HQ8 result.
Positive delta NLL means the removed head or group helped the trained model.
Local contribution classification: `distributed`.
Useful local groups: [0, 1, 2, 3, 4, 5, 6, 7].

## Per-group contribution map

| Group | Split | D_L | D_G | D_GL | Interaction |
|---:|---|---:|---:|---:|---:|
| 0 | discovery | +0.002163 | +0.055098 | +0.066011 | +0.008750 |
| 0 | confirmation | +0.002117 | +0.053525 | +0.064848 | +0.009205 |
| 1 | discovery | +0.006256 | +0.032437 | +0.053990 | +0.015297 |
| 1 | confirmation | +0.006968 | +0.031885 | +0.054539 | +0.015686 |
| 2 | discovery | +0.007067 | +0.055860 | +0.067887 | +0.004961 |
| 2 | confirmation | +0.007177 | +0.053840 | +0.065596 | +0.004579 |
| 3 | discovery | +0.002852 | +0.023844 | +0.031992 | +0.005296 |
| 3 | confirmation | +0.002535 | +0.021498 | +0.029033 | +0.005000 |
| 4 | discovery | +0.013545 | +0.057600 | +0.063181 | -0.007964 |
| 4 | confirmation | +0.011945 | +0.050249 | +0.056645 | -0.005549 |
| 5 | discovery | +0.007738 | +0.062677 | +0.076843 | +0.006428 |
| 5 | confirmation | +0.007684 | +0.053466 | +0.072092 | +0.010942 |
| 6 | discovery | +0.007903 | +0.066649 | +0.074895 | +0.000343 |
| 6 | confirmation | +0.007342 | +0.055204 | +0.065301 | +0.002755 |
| 7 | discovery | +0.003412 | +0.065336 | +0.069278 | +0.000530 |
| 7 | confirmation | +0.003061 | +0.062250 | +0.067065 | +0.001755 |

## Distribution diagnostics

Confirmation positive top-two local-damage share: 40.2%.
Discovery D_L/D_G Spearman: +0.571.
Confirmation D_L/D_G Spearman: +0.024.

## Collective local-head diagnostic

| Split | All-local damage | Sum of single-local damages | Collective gap | 95% CI |
|---|---:|---:|---:|---:|
| discovery | +0.174576 | +0.050936 | +0.123640 | [+0.120218, +0.127307] |
| confirmation | +0.165466 | +0.048829 | +0.116638 | [+0.113928, +0.119382] |

## One-group alignment

| Target group | Split | Mean wrong-chunk delta NLL | Two-stage 95% CI | Positive fraction |
|---:|---|---:|---:|---:|
| 0 | discovery | +0.002657 | [+0.002361, +0.002948] | 100.0% |
| 0 | confirmation | +0.002443 | [+0.002175, +0.002711] | 100.0% |
| 1 | discovery | +0.007610 | [+0.007172, +0.008063] | 100.0% |
| 1 | confirmation | +0.008012 | [+0.007557, +0.008462] | 100.0% |
| 2 | discovery | +0.006091 | [+0.005692, +0.006499] | 100.0% |
| 2 | confirmation | +0.006093 | [+0.005735, +0.006460] | 100.0% |
| 3 | discovery | +0.003357 | [+0.003083, +0.003632] | 100.0% |
| 3 | confirmation | +0.003163 | [+0.002864, +0.003460] | 100.0% |
| 4 | discovery | +0.014016 | [+0.012829, +0.015254] | 100.0% |
| 4 | confirmation | +0.012006 | [+0.011131, +0.012917] | 100.0% |
| 5 | discovery | +0.006426 | [+0.005969, +0.006885] | 100.0% |
| 5 | confirmation | +0.006820 | [+0.006341, +0.007322] | 100.0% |
| 6 | discovery | +0.002305 | [+0.001967, +0.002636] | 100.0% |
| 6 | confirmation | +0.002168 | [+0.001899, +0.002442] | 100.0% |
| 7 | discovery | +0.003156 | [+0.002837, +0.003488] | 100.0% |
| 7 | confirmation | +0.003162 | [+0.002804, +0.003531] | 100.0% |

## Figures

- [Per-group local/global/whole-group contribution](fig_group_contribution.pdf)
- [Local-head distribution and interaction](fig_local_distribution.pdf)
- [One-group local-chunk alignment map](fig_group_alignment.pdf)

## W&B

- 10ABC analysis: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/uyzpoa2t
- 10D analysis: https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/aklimpg4

## Interpretation limits

- single trained seed and one checkpoint
- head zeroing and chunk substitution are off-distribution interventions and do not equal retraining
- paired sequence confidence intervals do not quantify training-seed uncertainty
- single-head damages need not add to the all-local population damage because the network is nonlinear
- per-group labels refer to fixed HQ8 GQA positions and are not universal head coordinates

Figures are generated reproducibly by `figures/gen_fig_experiment10_group_contribution.py`.
