# Experiment 11 multi-GPU screening runbook

Experiment 11 trains nine independent seed-42 models to step 2,000. It is a
screen for soft query specialization, not a 20,000-step or multi-seed claim.
No GPU run is authorized merely by this runbook or the implementation commit.

## Run matrix and scheduling

The immutable run matrix is in `configs/experiment11/protocol.json` and has
SHA-256 `10f6d106d577b918f98c70fd0b3ff838ead9042c7c03978ce61f5ab96f7eb654`
over its canonical `runs` array.

- S2Q8: lambda 0, 0.1, 0.25, and 0.5.
- GSLQ8: lambda 0, 0.1, 0.25, and 0.5.
- Shared dense M8 endpoint: lambda 1.

One model uses one GPU. The controller accepts 1–9 GPU IDs and dynamically
fills them. Nine GPUs minimize training wall time; five GPUs require two waves;
three GPUs require three waves. The model recipe and global batch remain the
same at every GPU count.

For the authorized six-plus-two GPU deployment, use
`scripts/evaluate/run_experiment11_split_worker.py` on each host. Assign seven
runs to the six-GPU host and two runs to the two-GPU host. The six-GPU worker
rotates its seventh run at atomic boundaries, stops at every frozen probe
milestone, and serializes large checkpoint writes through
`MHAR_CHECKPOINT_LOCK`. This changes scheduling only: optimizer, scheduler,
RNG, data position, W&B identity, and final step remain exact across resumes.
Each worker refuses dirty source, identity-mismatched checkpoints, missing
probe outputs, duplicate run IDs, active GPUs, or a changed artifact hash.

The controller pauses every model at steps 500, 1,000, 1,500, and 2,000. It
measures the fixed 32-sequence discovery probe at step 0 and each pause. The
step-0 probe runs inside the exact initialized training process before the
first optimizer update.

## Frozen inputs

- evaluation artifact:
  `/root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt`
- artifact SHA-256:
  `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`
- source data: FineWeb-Edu `sample-10BT`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- tokenizer: Qwen3-0.6B, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`

The launch checkout must be clean and pinned to the reviewed implementation
commit. Use the existing Stage-B Python environment or reconstruct it with the
repository bootstrap. Do not copy a virtual environment between incompatible
system images.

## Storage

The controller defaults to requiring at least 110 GB free before launch. Nine
simultaneous optimizer-bearing checkpoints are substantially larger than the
compact result bundle. Do not reduce this guard without measuring the actual
checkpoint size and revising the reviewed runbook. The controller retains the
nine complete step-2,000 states; pruning them is not part of automatic closure.

## Launch

After a separate explicit GPU authorization, use a persistent screen:

```bash
screen -L -Logfile /root/autodl-tmp/experiment11-controller.log \
  -dmS mhar-exp11-controller \
  /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  scripts/evaluate/run_experiment11_controller.py \
  --repo /root/mhar-experiment11-run \
  --python /root/autodl-tmp/venvs/mhar-stage-b/bin/python \
  --output-root /root/autodl-tmp/experiment11 \
  --artifact /root/autodl-tmp/experiment2/stage-b-screening/fixed_eval.pt \
  --controller-commit REVIEWED_IMPLEMENTATION_COMMIT \
  --result-branch codex/experiment-11-soft-specialization \
  --gpu-ids 0,1,2,3,4
```

Add `--shutdown-on-success` only when the user explicitly authorizes automatic
shutdown for that server. Without the flag, successful publication leaves the
server running for review.

## Frozen gate order

1. Preflight validates clean source, source commit, artifact hash, GPU idleness,
   available GPU IDs, dependency snapshot, and disk space.
2. Train all nine models through each atomic milestone and run discovery-only
   effective-softness probes.
3. Evaluate all nine step-2,000 checkpoints on discovery.
4. Select one intermediate lambda within each family and write the immutable,
   content-hashed selection manifest.
5. Only then evaluate confirmation for all nine runs and probe the two selected
   models plus M8 on confirmation.
6. Compute paired sequence bootstrap intervals, frozen classifications, tables,
   PNG/PDF figures, and `FINAL_REPORT.md`.
7. Copy compact artifacts to `results/experiment11`, commit, push, and directly
   verify the remote branch.
8. If explicitly enabled, wait ten minutes and shut down only with idle GPUs.

Missing checkpoints, missing probes, non-finite NLL values, dirty source, hash
mismatches, changed discovery inputs, missing confirmation selection, failed
publication, or active GPU processes stop the workflow and write `FAILED.json`.
There is no automatic scientific retry and no continuation past step 2,000.

## Expected outputs

- `selection_manifest.json`
- 18 fixed-evaluation result files (nine runs by two splits)
- discovery probe trajectories and selected confirmation probes
- `analysis/summary.json`, `nll_curves.csv`, and
  `softness_trajectories.csv`
- six figures in 300-DPI PNG and vector PDF
- `results/experiment11/FINAL_REPORT.md`

W&B tracks the nine training runs and the analysis run. The repository figures
are regenerated from committed JSON/CSV, not manually exported from W&B.
