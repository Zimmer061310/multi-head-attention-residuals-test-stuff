# Experiment 2 step-2,000 partition-choice map

This directory contains the direct map from every evaluated routing choice to its
discovery-set NLL change relative to native H16.

- Choice 1 is native H16 and has `delta_nll_vs_native_h16 = 0`.
- Choices 2-496 are all 495 preregistered mixed-width partitions, sorted from
  lowest to highest discovery-set delta NLL.
- Each mixed choice removes four non-overlapping adjacent H16 boundaries. In the
  lower panel of the figure, a blue cell at row `i-i+1` means those two adjacent
  80-dimensional H16 atoms share one 160-dimensional router.
- `partition_choice_map.csv` gives the exact partition ID, NLL, delta NLL, and
  merged-boundary list for every plotted column.

Source discovery results:

- W&B run: <https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/bd3zn69v>
- SHA-256: `763aeb36b6bdad064333d48dcad3818f14b109695ebe39e23f23e261b3eb2bcc`
- Native H16 NLL: `4.36591859810047`
- Best mixed delta NLL: `+0.628320705156522`
- Median mixed delta NLL: `+0.780195044632182`
- Worst mixed delta NLL: `+1.013156819087203`

The map is descriptive for this frozen step-2,000 H16 checkpoint. It does not
establish the performance of an H8 or mixed-width model trained from scratch.

Regenerate it by running `experiment2_mixed_width.py analyze` on the immutable
`discovery_results.jsonl`; the analysis command writes both the PNG/PDF figure
and the 496-row CSV automatically.
