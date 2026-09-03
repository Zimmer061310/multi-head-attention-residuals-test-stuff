# Experiment 8 two-GPU runbook

This run trains only HQ8 and BHQ8. Accepted M8, LQ8, B, and BLQ8 fixed-set
results are reused. It never trains beyond step 2,000 and never launches
multi-seed work.

## GPU assignment

- GPU 0: HQ8
- GPU 1: BHQ8

Both start immediately. Each writes an atomic resumable checkpoint every 100
steps, retains only the latest state plus protected step 2,000, and reuses the
terminal milestone as the final checkpoint. A two-GPU machine avoids both a
waiting task and an idle rented GPU.

## Before launch

Verify all of the following:

1. The checkout is clean and pinned to the reviewed Experiment 8 commit.
2. The Python environment passes the focused Experiment 6–8 and training tests.
3. The fixed artifact hash equals
   `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`.
4. The two pinned FineWeb-Edu shards, tokenizer, W&B login, GitHub push access,
   free disk space, and both RTX 5090 GPUs are available.
5. No Experiment 8 output or screen already exists. Existing partial output
   requires explicit review; the launcher never silently restarts it.

## Launch

```bash
MHAR_PYTHON_BIN=/root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  scripts/train/launch_experiment8_2gpu.sh
```

Start the controller in a persistent screen with the exact reviewed commit for
both `--controller-commit` and `--training-commit`:

```bash
screen -L -Logfile /root/autodl-tmp/experiment8-controller.log \
  -dmS mhar-exp8-controller \
  /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  scripts/evaluate/run_experiment8_controller.py \
  --repo /root/mhar-experiment8-run \
  --python /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  --output-root /root/autodl-tmp/experiment8/screening \
  --artifact /root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt \
  --controller-commit REVIEWED_COMMIT \
  --training-commit REVIEWED_COMMIT \
  --result-branch codex/experiment-8-hybrid-q
```

The controller validates both atomic checkpoints, evaluates both fixed splits
in parallel, computes frozen paired contrasts, copies compact artifacts into
`results/experiment8/step-2000`, commits and verifies the remote push, waits ten
minutes, confirms GPU idleness, and shuts down. Any failed gate writes
`controller/FAILED.json` and blocks shutdown.
