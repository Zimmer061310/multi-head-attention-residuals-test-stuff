# Experiment 2 Stage B screening: step 2000

## Decision

The preregistered gate returned `stop_at_2000`. Discovery and confirmation
rankings were identical (Spearman rho = 1.000), and multiple paired-bootstrap
contrasts were decisive. The result is therefore not classified as chaotic and
training must not continue to step 5000.

Evaluation used the immutable fixed artifact with SHA-256
`29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`.

## Confirmation ranking

Lower NLL is better. Confidence intervals are paired sequence-bootstrap 95%
intervals for delta NLL relative to H8, using 10,000 bootstrap samples.

| Rank | Variant | NLL | Delta NLL vs H8 | 95% CI |
| ---: | --- | ---: | ---: | --- |
| 1 | H8 | 4.137027 | 0.000000 | reference |
| 2 | H4 | 4.137516 | +0.000489 | [-0.001724, +0.002673] |
| 3 | Mixed k=5 | 4.179216 | +0.042189 | [+0.040083, +0.044265] |
| 4 | Mixed k=4 worst | 4.195826 | +0.058799 | [+0.056390, +0.061223] |
| 5 | Mixed k=4 best | 4.214440 | +0.077413 | [+0.074698, +0.080172] |
| 6 | Mixed k=3 | 4.230281 | +0.093254 | [+0.090252, +0.096276] |
| 7 | Mixed k=2 | 4.238565 | +0.101538 | [+0.098348, +0.104757] |
| 8 | H16 | 4.248184 | +0.111157 | [+0.107941, +0.114385] |

## Main findings

1. H8 is the numerical winner, while H4 is statistically tied with H8 at this
   milestone because its confidence interval includes zero.
2. Every mixed-width model is decisively worse than H8.
3. More merging helps within the mixed family: k=5 beats k=4, k=3, and k=2.
4. The frozen boundary model did not transfer to from-scratch k=4 training.
   The partition labeled `mixed-k4-worst` beats `mixed-k4-best` by 0.018614
   NLL, with a paired 95% CI of [0.016633, 0.020598] for best minus worst.
5. All mixed variants still decisively beat H16, so the broad benefit of
   reducing fine-grained H16 routing survives, but the selected heterogeneous
   boundary locations do not beat uniform H8 or H4 routing.

## W&B

- [Combined analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/s8tlkvxm)
- [H16 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/mwuvk6ee)
- [H8 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/rkvwhctm)
- [H4 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/bp4b3c13)
- [Mixed k=2 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/jb44s1c4)
- [Mixed k=3 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/1yw6dzc7)
- [Mixed k=4 best evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/u0wuclez)
- [Mixed k=4 worst evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/c6z7a3x9)
- [Mixed k=5 evaluation](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/24p06syz)

Raw paired results, run manifests, logs, the ranking CSV, and publication PNG
and PDF are preserved alongside this report.
