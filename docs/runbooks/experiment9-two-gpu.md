# Experiment 9 two-GPU frozen-evaluation runbook

Experiment 9 performs no training. It evaluates the accepted Experiment 8 HQ8
step-2,000 checkpoint under frozen causal interventions.

## GPU schedule

- Phase 9A: 73 conditions split deterministically across GPUs 0 and 1.
- Phase 9B: 32 frozen derangements split across the same two GPUs, only if the
  preregistered 9A local-contribution gate passes.
- Each worker loads one HQ8 model and processes its assigned conditions
  sequentially. There is no duplicate condition and neither GPU waits during an
  active phase.

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

1. Start the disk-preserving clone that contains the accepted checkpoint.
2. Use a clean checkout pinned to the reviewed Experiment 9 commit.
3. Verify both RTX 5090 GPUs, the pinned Python environment, W&B login, GitHub
   push access, the checkpoint hash, artifact hash, and focused tests.
4. Confirm `/root/autodl-tmp/experiment9` does not contain an unreviewed prior
   attempt. `FAILED.json` always blocks automatic continuation and shutdown.

## Launch

```bash
screen -L -Logfile /root/autodl-tmp/experiment9-controller.log \
  -dmS mhar-exp9-controller \
  /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  scripts/evaluate/run_experiment9_controller.py \
  --repo /root/mhar-experiment9-run \
  --python /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  --output-root /root/autodl-tmp/experiment9 \
  --checkpoint /root/autodl-tmp/experiment8/screening/hq8/step-2000 \
  --artifact /root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt \
  --controller-commit REVIEWED_COMMIT \
  --result-branch codex/experiment-9-head-contribution
```

The controller first requires exact reproduction of Experiment 8 HQ8. It then
analyzes 9A and conditionally runs 9B, uploads compact tables to W&B, copies all
per-sequence results, publication figures, and manifests into
`results/experiment9`, commits and
directly verifies the remote branch, waits ten minutes, checks GPU idleness, and
shuts down. Any failed prerequisite or incomplete result writes `FAILED.json`
and refuses shutdown.
