# Experiment 3 GPU Runbook

This runbook prepares and executes the preregistered training-time boundary
learnability tests. The repository is fully prepared, but none of the commands
below are executed automatically on a GPU server.

## Locked inputs

- Training: FineWeb-Edu sample/10BT shards `000_00000.parquet` and
  `001_00000.parquet` at the pinned dataset commit.
- Experiment 3 evaluation: shard `003_00000.parquet`, 2,152,437,524 bytes,
  SHA-256 `22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb`.
- Tokenizer: `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Fixed artifact: 512 discovery and 512 confirmation sequences, 1,024 tokens
  each, seed `20260828`, document-disjoint split construction.
- W&B project: `MHAR Stuff`; group prefix
  `mhar-exp3-boundary-learnability`.

The evaluation shard is distinct from training shards `000–001` and the
Experiment 2 evaluation shard `002`. The generated artifact receives its own
SHA-256 at materialization time; the preflight records and enforces it on every
evaluation invocation.

## 1. Prepare a fresh server

Clone the repository, check out the committed Experiment 3 revision, then run:

```bash
MHAR_MIN_GPUS=1 scripts/setup/bootstrap_experiment3_server.sh
```

The bootstrap installs the exact recorded CUDA/Python stack, checks GitHub and
W&B authentication, downloads and hashes shard `003`, materializes the fixed
artifact, runs CPU/synthetic tests, and validates the visible GPUs. It ends
without launching training or evaluation.

For a server that already has the environment:

```bash
scripts/setup/prepare_experiment3_eval_artifact.sh
python scripts/setup/preflight_experiment3_server.py \
  --artifact /root/autodl-tmp/experiment3/fixed_eval.pt
```

Record the printed artifact SHA-256 in the run ledger before the first probe.

## 2. Train native H16 probe checkpoints

On three GPUs, train independent seeds 42, 43, and 44 to step 3,000 while
retaining steps 1,000, 1,500, 2,000, and 3,000:

```bash
scripts/train/launch_experiment3_h16_probes_3gpu.sh
```

Existing exact checkpoints can be reused only when their training manifests
match the architecture, optimizer, scheduler, data, tokenizer, seed, and source
revision. Resume one interrupted seed by passing its atomic checkpoint to
`run_experiment3_h16_probe_seed.sh`.

When the controllers and the locked training source are separate worktrees,
keep evaluation on the Experiment 3 commit and launch training with:

```bash
MHAR_CONTROLLER_REPO_DIR=/root/mhar-experiment \
MHAR_TRAINING_REPO_DIR=/root/mhar-training-81ff305 \
MHAR_RESUME_SEED42=/root/autodl-tmp/experiment2/stage-b-screening/h16/step-2000 \
  scripts/train/launch_experiment3_h16_probes_3gpu.sh
```

The per-seed resume variables are `MHAR_RESUME_SEED42`,
`MHAR_RESUME_SEED43`, and `MHAR_RESUME_SEED44`. An unset variable starts that
seed from scratch. This preserves the source revision recorded in reused
training manifests while allowing the newer controllers to analyze results.
Experiment 3C branches must instead run from the Experiment 3 controller
worktree because exact branch-state support was added there; its parent
validation deliberately compares the locked scientific recipe rather than the
controller commit.

## 3. Run 3A and 3B for one seed

```bash
MHAR_SEED=42 scripts/evaluate/run_experiment3_seed_probes.sh
```

Repeat for seeds 43 and 44 only according to the locked gate sequence. Each
candidate evaluation is resumable through immutable JSONL and run manifests.
The controller uploads raw results, tables, gate summaries, PNG/PDF figures,
and captions to W&B.

After seed 42 passes actionability, the seed-43/44 actionability controller
verifies the frozen seed-42 summary and then permits their diagnostic branches
even if a local signal/stability gate failed. This avoids selecting only
favorable seeds; their local failures still count against the replication gate.

## 4. Freeze and train 3C branches

First freeze the seed-local branch selection:

```bash
MHAR_SEED=42 scripts/evaluate/run_experiment3_actionability.sh
```

The command refuses selection unless 3A and 3B passed. Then train the four
matched branches from the same step-1,500 checkpoint:

```bash
MHAR_BRANCH_MANIFEST=/root/autodl-tmp/experiment3/results/seed-42/actionability/branch_selection.json \
MHAR_PARENT_CHECKPOINT=/root/autodl-tmp/experiment3/checkpoints/h16/seed-42/step-1500 \
scripts/train/launch_experiment3_actionability_4gpu.sh
```

After all branches atomically reach step 2,000, evaluate and analyze them:

```bash
MHAR_SEED=42 MHAR_EVALUATE_BRANCHES=1 \
  scripts/evaluate/run_experiment3_actionability.sh
```

The primary contrast is predicted-good minus random on confirmation sequences.
Do not use confirmation data to change the frozen branch choices.

## 5. Cross-seed replication

After all three seed bundles contain signal, temporal, selection, branch
evaluation, and actionability summaries:

```bash
scripts/evaluate/run_experiment3_cross_seed.sh
```

This reports every seed, nested seed/sequence bootstrap summaries, and the
locked replication gate. It does not require the same boundary ID across seeds.

## 6. Landscape mechanism diagnostic

Point the landscape controller at the seed-42 native H8 step-2,000 checkpoint
trained under the same recipe:

```bash
MHAR_CHECKPOINT=/path/to/h8/step-2000 \
  scripts/evaluate/run_experiment3_landscape.sh
```

It evaluates native H8 plus 56 one-boundary displacement candidates on both
splits, then classifies the landscape as soft-learning-compatible,
repeatable-but-discrete, or insufficient evidence.

## Outputs to preserve

Preserve and upload:

- repository commit and dirty-state check;
- source shard identity and fixed artifact plus sidecar SHA-256;
- every checkpoint `training_manifest.json` and model SHA-256;
- candidate-order manifests and resumable JSONL files;
- frozen selection and branch manifests;
- discovery and confirmation token-weighted and per-sequence NLL;
- Spearman, top-overlap, future regret, paired bootstrap intervals, and gates;
- PNG and vector PDF figures with captions;
- all W&B run URLs and artifact versions.

No learnable-boundary architecture should be implemented unless the final
proof-chain gate in `docs/plans/experiment-3.md` passes.
