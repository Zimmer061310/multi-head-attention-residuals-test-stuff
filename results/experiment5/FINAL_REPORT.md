# Experiment 5 — fixed-validation boundary washout

Completed 2026-08-30, approximately 14:00 CST. All three branches stopped at
step1600. This is fixed-held-out NLL, not the training-batch loss used in
Experiment 4. Validation assessment: **share with the limitations below**.

## Main finding

The initial best-merge advantage is real and survives the first20 updates, but
does not remain consistently favorable: A beats B through+20, B beats A at+50,
and A beats B again at+100 with a much smaller gap. Both fixed splits show this
sign pattern. A never has lower measured NLL than unchanged C at any sampled
offset. Thus this run shows a transient advantage over the deliberately bad
merge, not an improvement over leaving the model unchanged.

The frozen decision remains `inconclusive_or_nonmonotonic`. No sampled A−B
interval lies wholly within the preregistered practical-equivalence margin
±0.001 NLL, so no sustained practical-washout time was identified. Nor does the
gap favor A at both+50 and+100. We cannot report a monotonic washout, exact
disappearance time, or a demonstrated benefit from adaptive boundaries.

## Primary confirmation measurements

A=remove-03, B=remove-13, C=native H16. All entries use the same512 confirmation
sequences (523,776 scored next-token positions per measurement). Units are
nats/token; negative differences favor A. Absolute step=1500+added steps.

| Added steps | A NLL | B NLL | C NLL | A−B | A−C |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.566359 | 4.577415 | 4.542212 | −0.011057 | +0.024147 |
| 1 | 4.562810 | 4.571514 | 4.548150 | −0.008704 | +0.014660 |
| 2 | 4.568303 | 4.577955 | 4.549323 | −0.009651 | +0.018980 |
| 5 | 4.574961 | 4.584848 | 4.564402 | −0.009886 | +0.010559 |
| 10 | 4.566772 | 4.573697 | 4.560174 | −0.006925 | +0.006598 |
| 20 | 4.541813 | 4.545156 | 4.539615 | −0.003343 | +0.002198 |
| 50 | 4.497404 | 4.496013 | 4.495748 | +0.001392 | +0.001656 |
| 100 | 4.441992 | 4.443945 | 4.441740 | −0.001953 | +0.000252 |

| Added steps | A−B paired95% CI | A−C paired95% CI |
|---:|:---|:---|
| 0 | [−0.012314, −0.009783] | [+0.023038, +0.025313] |
| 1 | [−0.009517, −0.007880] | [+0.014000, +0.015340] |
| 2 | [−0.010542, −0.008771] | [+0.018137, +0.019878] |
| 5 | [−0.010635, −0.009140] | [+0.009982, +0.011137] |
| 10 | [−0.007630, −0.006230] | [+0.006039, +0.007187] |
| 20 | [−0.003949, −0.002730] | [+0.001742, +0.002666] |
| 50 | [+0.000901, +0.001872] | [+0.001285, +0.002030] |
| 100 | [−0.002360, −0.001538] | [−0.000084, +0.000589] |

At+20 the A−B gap retains30.2% of its baseline magnitude; at+100 it retains17.7%,
with a reversal at+50 in between. The+100 A−C interval includes zero: there is
no demonstrated advantage over C, rather than proof of identical models.

Secondary discovery A−B at+20/+50/+100 is −0.003173/+0.001004/−0.001430.
At+100 discovery A−C is+0.000576 [95% CI +0.000216,+0.000921]. Full precision,
both splits and all pointwise intervals are preserved in
`analysis/fixed_eval_losses.csv` and `analysis/washout_summary.json`.

## Verification and scope

- The24 measurement files cover all3 branches ×8 offsets, each with both fixed
  splits. Source, branch, artifact, precision, token counts and aggregate
  consistency were checked. All36 compact server files matched local hashes;
  the five previously accepted baseline/manifest/gate files matched byte-for-byte.
- Step0 reproduced Experiment3 with zero aggregate or per-sequence error for
  all six branch/split combinations. The per-sequence bootstrap and frozen
  decision were recomputed locally and matched the saved summary exactly.
- All21 requested trained snapshots plus all3 complete final step1600 states
  were validated on the server. Each branch restored the exact seed43 step1500
  optimizer, scheduler, RNG and data position48000, then trained uninterrupted
  for100 updates. Evaluation ran in separate processes after training.
- The original20,000-step LR schedule was retained. A/B both have15 routing
  groups with the same width composition; C has16. All three use eager routing;
  the parent was originally trained with fused routing.
- Intervals use10,000 paired sequence-bootstrap resamples, seed20260830. They
  are pointwise, not simultaneous or training-seed confidence intervals.
- This is one seed, one selected best/worst pair and reused held-out data from
  Experiment3. It is not fresh external validation or a test of an online
  adaptive mechanism. Unsampled times and other boundary choices are unknown.
- The figure was visually checked: signs, units, intervals and the±0.001 band
  agree with the data. Its x-axis is explicitly symlog, not linear; connecting
  lines do not establish behavior at unsampled updates. Panels auto-scale
  separately, so compare numerical values rather than pixel heights.

The validation pass prevents two unsupported conclusions: "the gap vanished
within20 updates" and "adaptive grouping improves training." Neither follows
from these measurements. No new experiment is automatically authorized.

## Provenance and operational deviation

Execution remained at `8e763c25559ef76b45549049f7443925d66460e7` in
`/root/mhar-experiment5-run`. The successful controller ran approximately
12:33–14:00 CST (about87 minutes including step0, training, checkpoint I/O,
evaluation and upload).

- Parent content SHA-256:
  `ece1370fb201f3ba661424bdb7126a4b1d59221bccba6e46332372e63c9faaeb`.
- Fixed artifact SHA-256:
  `1a239dfc65c3b4f9184ccbdc28e9165b4bc452312a2d7360d6b7519fafb1a5af`.
- Branch manifest SHA-256:
  `70200b6cc0a3e10b7ad9bd5724518296024b080f6a22872f7db3bf6c9d248e24`.
- Final summary SHA-256:
  `1a0d319e17d7b7fcbfb26c5c9e8161a32e9533a378f2e502afae5b94cfa05f7a`.

The initial39fc117 attempt failed before any measured loss or optimizer update
because a file-hashing helper did not accept CLI string paths. Its logs and
manifest remain archived under
`/root/autodl-tmp/experiment5-startup-failed-39fc117` and its sibling controller
log. The8e763c2 fix only normalized paths and added a regression test, preserving
all scientific inputs and tolerances. No accepted result or checkpoint was
replaced. There were no training overshoots in this successful Exp5 run.

All38 focused tests passed again after final artifact backup. Verification uses:

```sh
python3 -m unittest tests.test_experiment5_controller tests.test_experiment5_washout tests.test_train_resume tests.test_experiment3_actionability tests.test_experiment4_short_horizon tests.test_experiment3_signal tests.test_experiment3_router
```

The server-generated `analysis/FINAL_REPORT.md`, measurements and figures are
preserved unchanged. This document adds human-reviewed interpretation and QA.
Shutdown remains contingent on verified backup/push acknowledgment, followed
by the controller's10-minute grace and final completion/GPU-idleness checks.
SSH unreachability alone is not independent confirmation of provider billing.

## W&B

- [Fixed-validation analysis and figure](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/52j6bu4r)
- [A — predicted-good training](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/v4t00sfq)
- [B — predicted-bad training](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/t03sqdoy)
- [C — unchanged training](https://wandb.ai/zimmer061310-ena/MHAR%20stuff/runs/xrp2boi5)
