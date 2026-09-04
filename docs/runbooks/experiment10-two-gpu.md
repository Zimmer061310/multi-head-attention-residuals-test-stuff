# Experiment 10 two-GPU frozen-evaluation runbook

Experiment 10 performs no training. It evaluates the accepted Experiment 8
HQ8 step-2,000 checkpoint under frozen per-group interventions.

## GPU schedule

- Phase 10ABC: 25 conditions split deterministically across GPUs 0 and 1.
- Phase 10D: 56 exhaustive one-group chunk substitutions split across the same
  GPUs, only if at least one local group passes the preregistered usefulness
  rule.
- Each worker loads one HQ8 model and evaluates its assigned conditions
  sequentially. Conditions are never duplicated.

Expected wall time on two RTX 5090s is roughly 20–30 minutes for 10ABC and
55–70 additional minutes if 10D runs, plus analysis, backup, and the ten-minute
shutdown grace: about 1.5–2 hours end to end when the gate passes. This estimate
uses the Experiment 9 fixed-evaluation timings and is not a guarantee.

## Frozen inputs

- checkpoint:
  `/root/autodl-tmp/experiment8/screening/hq8/step-2000`
- checkpoint content SHA-256:
  `74cff0ab19409dac9f6104e8986e4890c0837bc730d8ac233ae18011dbc58333`
- artifact:
  `/root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt`
- artifact SHA-256:
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`

## Before launch

1. Start the disk-preserving server containing the accepted checkpoint.
2. Use a clean checkout pinned to the reviewed Experiment 10 setup commit.
3. Verify both GPUs, the pinned Python environment, W&B login, GitHub push
   access, checkpoint/artifact hashes, and focused tests.
4. Confirm `/root/autodl-tmp/experiment10` does not contain an unreviewed prior
   attempt. Any `FAILED.json` blocks continuation and shutdown.

## Launch

```bash
screen -L -Logfile /root/autodl-tmp/experiment10-controller.log \
  -dmS mhar-exp10-controller \
  /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  scripts/evaluate/run_experiment10_controller.py \
  --repo /root/mhar-experiment10-run \
  --python /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  --output-root /root/autodl-tmp/experiment10 \
  --checkpoint /root/autodl-tmp/experiment8/screening/hq8/step-2000 \
  --artifact /root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt \
  --controller-commit REVIEWED_COMMIT \
  --result-branch codex/experiment-10-per-group-contribution
```

The controller first requires exact Experiment 8 HQ8 reproduction. It runs
10ABC, applies the frozen 10D gate, uploads analysis tables to W&B, creates an
independent report and figures, copies compact results into
`results/experiment10`, commits and verifies the remote experiment branch, then
waits ten minutes and shuts down only if GPUs are idle. Missing results, dirty
code, mismatched hashes, a failed push, or active GPU processes fail closed.
