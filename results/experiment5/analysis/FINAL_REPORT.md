# Experiment 5: fixed-validation washout

inconclusive_or_nonmonotonic

Single seed; reused held-out data; pointwise paired-sequence bootstrap, not training-seed uncertainty. CI crossing zero is not equivalence. No new experiment is automatically authorized.

All NLLs below use the same 512 confirmation sequences. Negative differences favor A.

| Added steps | A | B | C | A−B | A−B 95% CI | A−C |
|---:|---:|---:|---:|---:|:---|---:|
| 0 | 4.566359 | 4.577415 | 4.542212 | -0.011057 | [-0.012314, -0.009783] | +0.024147 |
| 1 | 4.562810 | 4.571514 | 4.548150 | -0.008704 | [-0.009517, -0.007880] | +0.014660 |
| 2 | 4.568303 | 4.577955 | 4.549323 | -0.009651 | [-0.010542, -0.008771] | +0.018980 |
| 5 | 4.574961 | 4.584848 | 4.564402 | -0.009886 | [-0.010635, -0.009140] | +0.010559 |
| 10 | 4.566772 | 4.573697 | 4.560174 | -0.006925 | [-0.007630, -0.006230] | +0.006598 |
| 20 | 4.541813 | 4.545156 | 4.539615 | -0.003343 | [-0.003949, -0.002730] | +0.002198 |
| 50 | 4.497404 | 4.496013 | 4.495748 | +0.001392 | [+0.000901, +0.001872] | +0.001656 |
| 100 | 4.441992 | 4.443945 | 4.441740 | -0.001953 | [-0.002360, -0.001538] | +0.000252 |

A/B use 15 groups; C uses 16. All use eager routing; parent training used fused routing.
Snapshots were evaluated in separate processes after uninterrupted training to step1600.
Original optimizer/scheduler/RNG/data position and the 20,000-step LR schedule were restored.
No Experiment 3 gate or accepted earlier result was changed.

[W&B fixed-evaluation analysis](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/52j6bu4r)

- [predicted-good training](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/v4t00sfq)
- [predicted-bad training](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/t03sqdoy)
- [unchanged training](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/xrp2boi5)
