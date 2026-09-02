# Experiment 7 three-GPU runbook

This run trains the four new seed-42 models only. Accepted B, M4, M8, C4 and
C8 fixed-evaluation results are reused. It never continues beyond step 2000.
Training writes an atomic resumable checkpoint every 100 steps and retains the
latest state plus step 2000, which is the only evaluation milestone.

## GPU assignment

- GPU 0: LQ4
- GPU 1: LQ8
- GPU 2: BLQ4, then BLQ8

This is the cost-efficient assignment: LQ4/LQ8 are expected to be the critical
path, while both baseline-residual controls should finish sequentially before
the MHAR runs.

## Launch

From a clean pinned checkout with the verified Python environment, fixed
artifact, local FineWeb-Edu shards, and W&B credentials:

```bash
MHAR_PYTHON_BIN=/root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  scripts/train/launch_experiment7_3gpu.sh
```

Start `scripts/evaluate/run_experiment7_controller.py` in a persistent screen,
passing the exact controller/training commit, fixed artifact, output root, and
result branch. The controller validates four atomic step-2000 checkpoints,
evaluates on the frozen artifact, computes paired contrasts, pushes compact
results, waits ten minutes, checks GPU idleness, and shuts down. Any incomplete
checkpoint, mismatched identity, failed evaluation, failed push, or active GPU
process blocks shutdown.
