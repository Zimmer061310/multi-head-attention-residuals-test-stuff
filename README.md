# Multi-Head Attention Residuals (MHAR)

Paper and reference implementation for **Multi-Head Attention Residuals**.

[📄 arXiv:2607.27230](https://arxiv.org/abs/2607.27230) · [📝 Blog post](https://wdlctc.github.io/multi-head-attention-residuals.html) · [PDF](paper/mhar_arxiv_v1.pdf)

Cheng Luo, Zefan Cai, Junjie Hu

## TL;DR

Attention residuals let every sublayer attend over the depth history through a learned
softmax — but that read uses a **single query shared across the entire width**, forcing
every feature subspace to read the depth history through one distribution. The cost of
this forced compromise grows with how much the subspaces disagree about which layers to
read, and disagreement grows with model width.

**MHAR** reshapes the routing query into `H` per-subspace heads, each with its own softmax
over the depth history:

- the read becomes block-diagonal,
- the reshape adds **zero parameters** and negligible compute,
- `H = 1` recovers attention residuals exactly.

Trained from scratch, MHAR improves validation loss over a standard Transformer at
100M, 350M, and 1B (−0.061, −0.149, −0.140), the best of four methods in every setting,
with the gain increasing toward larger scales. Validation loss is U-shaped in `H` with a
flat optimum at `H = 4`–`8`; we adopt `H = 8` for large models. Fused Triton routing
kernels raise attention-residual training throughput from 0.2–0.5× to 0.55–0.88× of the
baseline at near-baseline peak memory, and an identity-preserving conversion (delta
attention residuals) supports 8B mid-training: **+3.2 GSM8K, +3.1 GPQA**.

## Repository layout

```
src/attention_residuals/            model definitions and routing kernels
src/training/                       from-scratch and continued-pretraining entry points
src/experiments/                    frozen Experiment 1 and 2 evaluation workflows
scripts/setup/                      fresh-server bootstrap and validation
scripts/train/                      single-run and eight-GPU launchers
scripts/evaluate/                   milestone and post-training controllers
configs/                            frozen experiment and environment manifests
docs/plans/                         experiment plans and analysis specifications
requirements/                       pinned Python environments
tests/                              correctness, parity, and workflow tests
results/                            checked-in experiment artifacts
paper/                              arXiv source and PDF
```

## Quick start

From-scratch pretraining with MHAR (`full_mh`, 8 routing heads):

```bash
torchrun --nproc_per_node=8 --module src.training.train_scratch \
    --mode full_mh --attnres_heads 8
```

Single-head attention residuals (the `H = 1` special case):

```bash
torchrun --nproc_per_node=8 --module src.training.train_scratch --mode full
```

Standard Transformer baseline:

```bash
torchrun --nproc_per_node=8 --module src.training.train_scratch --mode baseline
```

Convert a pretrained checkpoint with the identity-preserving delta variant
(zero disruption at step 0), then continue pretraining:

```bash
torchrun --nproc_per_node=8 --module src.training.train_cpt \
    --mode delta_mh --attnres_heads 8
```

Verify the fused Triton kernels against the eager path:

```bash
python -m tests.test_mhar_fused
python -m tests.test_mhar_fused_delta
```

## Experiment 1: exhaustive residual partitions

The frozen H=4 compatibility experiment evaluates all 105 pairings of eight
160-dimensional primitive blocks into four routing groups.  The same partition
is applied at all 73 routing sites, and every candidate sees the same fixed
tokens.  The full preregistration is in
[`docs/plans/experiment-1.md`](docs/plans/experiment-1.md).

### Train the preregistered 1B H=4 checkpoint

The checked-in launcher fixes the FineWeb-Edu substitution and every planned
training-scale field: 1.0795B parameters, width 1280, 36 layers, 16 attention
heads, 8 KV heads, FFN 5120, MHAR H=4, sequence length 1024, global batch 32,
20,000 AdamW steps, peak/minimum LR `5e-4`/`5e-5`, 1,000 warmup steps, bf16,
and seed 42. FineWeb-Edu's official `sample-10BT` configuration and the Qwen3
tokenizer are pinned to immutable Hub commits in the script. The server run
uses two local parquet shards (about 1.4B tokens, versus 655.36M consumed) so
training does not depend on a six-day network stream. Every matched shard's
absolute path, byte size, and SHA-256 are frozen in the run identity.

```bash
MHAR_PYTHON_BIN=/path/to/python \
MHAR_OUTPUT_DIR=/fast-disk/experiment1/checkpoint-1b-h4-fineweb-edu \
MHAR_DATA_FILES='/fast-disk/fineweb-edu-sample-10BT/train/*.parquet' \
./scripts/train/run_experiment1_train_1b_h4.sh
```

The single-5090 execution uses per-device batch 4 and accumulation 8 to retain
global batch 32. Fused MHAR and activation checkpointing are systems-only
choices. Every 500 steps, the runner atomically saves model weights, AdamW
moments, scheduler, RNG state, packed-data position, W&B run ID, and the full
immutable run identity. It retains the newest two step checkpoints.

Resume without changing the schedule or data position:

```bash
MHAR_PYTHON_BIN=/path/to/python \
MHAR_OUTPUT_DIR=/fast-disk/experiment1/checkpoint-1b-h4-fineweb-edu \
./scripts/train/run_experiment1_train_1b_h4.sh \
  /fast-disk/experiment1/checkpoint-1b-h4-fineweb-edu/step-500
```

Training logs to W&B project `MHAR Stuff`, group
`mhar-exp1-1b-h4-fineweb-edu`. W&B is required: the process aborts instead of
silently running untracked.

### Train the matched 1B H=8 architectural control

Experiment 2 Stage B uses a separately trained uniform H=8 model. Its launcher
matches the H=16 recipe in architecture, FineWeb-Edu shards and revisions,
tokenizer, seed, optimizer, schedule, batch, precision, and total steps; only
`attnres_heads` and the run/output identity differ. The 2,000, 5,000, 10,000,
and 20,000 step checkpoints are protected from ordinary two-checkpoint
rotation.

```bash
MHAR_PYTHON_BIN=/path/to/python \
MHAR_OUTPUT_DIR=/fast-disk/experiment2/checkpoint-1b-h8-fineweb-edu \
MHAR_DATA_FILES='/fast-disk/fineweb-edu-sample-10BT/train/*.parquet' \
./scripts/train/run_experiment2_train_1b_h8.sh
```

The run logs to W&B project `MHAR Stuff`, group
`mhar-exp2-stage-b-1b-h8-fineweb-edu`, and can be resumed from any protected
milestone by passing its checkpoint directory as the sole argument.

The full eight-model Stage B screen is frozen in
`configs/experiment2/stage-b-screening.json`. Launch any model on one GPU with:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train/run_experiment2_stage_b_screen.sh mixed-k4-best
```

Valid names are `h16`, `h8`, `h4`, `mixed-k2`, `mixed-k3`, `mixed-k4-best`,
`mixed-k5`, and `mixed-k4-worst`. On a multi-GPU host, assign one different
model to each GPU; this preserves the single-GPU global-batch-32 recipe and is
the preferred parallelization strategy for the screening matrix.

### Fresh 8-GPU server setup

Do not copy a Python environment from a machine with a different CPU or CUDA
image. The successful RTX 5090 environment is reconstructed from wheels using
`requirements/stage-b.txt`; its exact W&B-recorded versions and two frozen
FineWeb-Edu shard identities are in `configs/environment/stage-b-server.json`.

After cloning the repository on the new server, run interactively:

```bash
MHAR_BOOTSTRAP_PYTHON=python3.12 ./scripts/setup/bootstrap_stage_b_server.sh
```

The bootstrap installs one shared virtual environment, authenticates GitHub
and W&B without storing credentials in the repository, downloads/resumes the
two 4.3 GB total dataset shards, verifies both SHA-256 hashes, checks at least
eight bf16 GPUs and 700 GiB free disk, and runs the correctness preflight.

Launch one model per GPU across all eight GPUs:

```bash
./scripts/train/launch_experiment2_stage_b_8gpu.sh
```

The launcher refuses pre-existing screens or output manifests. Resumption is
therefore always explicit and cannot accidentally overwrite a run.

Install the pinned experiment runtime and run the CPU correctness suite:

```bash
python3 -m pip install -r requirements/experiment1.txt
MHAR_RUN_MODEL_INTEGRATION=1 python3 -m unittest -v \
  tests.test_experiment1_partitions tests.test_experiment1_runner tests.test_train_resume
```

Materialize non-overlapping discovery and confirmation tensors from an explicit
dataset revision or local parquet manifest:

```bash
python3 -m src.experiments.experiment1_partition_compatibility materialize \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-revision <immutable-revision> \
  --tokenizer Qwen/Qwen3-0.6B \
  --tokenizer-revision <immutable-revision> \
  --seq-len 1024 \
  --discovery-sequences 512 \
  --confirmation-sequences 512 \
  --output output/experiment1/fixed_eval.pt
```

Evaluate all 105 partitions on discovery, rank them, evaluate only the selected
reference/best/worst candidates on confirmation, then write the final report:

```bash
python3 -m src.experiments.experiment1_partition_compatibility evaluate \
  --checkpoint <1b-h4-full_mh-checkpoint> \
  --artifact output/experiment1/fixed_eval.pt \
  --output-dir output/experiment1/run \
  --split discovery --device cuda --dtype bf16 --batch-size 1 \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4

python3 -m src.experiments.experiment1_partition_compatibility analyze \
  --discovery-results output/experiment1/run/discovery_results.jsonl \
  --output-dir output/experiment1/run/analysis \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4

python3 -m src.experiments.experiment1_partition_compatibility evaluate \
  --checkpoint <1b-h4-full_mh-checkpoint> \
  --artifact output/experiment1/fixed_eval.pt \
  --output-dir output/experiment1/run \
  --split confirmation \
  --discovery-results output/experiment1/run/discovery_results.jsonl \
  --device cuda --dtype bf16 --batch-size 1 \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4

python3 -m src.experiments.experiment1_partition_compatibility analyze \
  --discovery-results output/experiment1/run/discovery_results.jsonl \
  --confirmation-results output/experiment1/run/confirmation_results.jsonl \
  --output-dir output/experiment1/run/final-analysis \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4
```

Results are appended and synced after every partition, so an interrupted run
can resume in the same output directory.  The run manifest rejects changes to
the checkpoint hash, fixed-data hash, software commit, dtype, or partition set.
W&B receives separate smoke/discovery/confirmation/analysis jobs under one
group, plus the fixed-token artifact, raw JSONL/manifests, ranked table, and
final report artifact. The checkpoint itself is not uploaded; its SHA-256 is
recorded.

To queue the complete workflow behind the exact 1B H=4 training launcher, run:

```bash
MHAR_PYTHON_BIN=/root/autodl-tmp/mhar-venv/bin/python \
  ./scripts/evaluate/run_experiment1_after_training.sh
```

For staged checkpoint inspection, run the milestone controller from a separate
screen session. It waits for an atomically completed checkpoint, interrupts the
training screen, runs all 105 discovery partitions, selects and evaluates the
reference/best/worst partitions on the untouched confirmation set, and then
exits so the result can be reviewed before training resumes:

```bash
MHAR_PYTHON_BIN=/root/autodl-tmp/mhar-venv/bin/python \
MHAR_REPO_DIR=/root/mhar-experiment \
./scripts/evaluate/run_experiment1_milestone.sh 2000
```

The preregistered milestones are 2,000, 5,000, 10,000, and 20,000 steps. Resume
from the reviewed checkpoint with `scripts/train/run_experiment1_train_1b_h4.sh <checkpoint>`
before starting the next milestone controller. Each milestone uses a distinct
W&B group and output directory; the fixed discovery/confirmation artifact is
shared unchanged across milestones.

The queue materializes document-disjoint discovery and confirmation tensors
from the reserved local FineWeb-Edu shard, records the source shard SHA-256,
waits for the atomically published final checkpoint, runs a three-partition
production smoke, then executes discovery, analysis, confirmation, and final
analysis. It exits rather than silently waiting if training stops without a
final checkpoint.

The analysis exports 300-dpi PNG and vector PDF versions of:

- `fig_nll_vs_distance`: all partitions against mean coordinate distance, with
  distance-bin medians and reference/best/worst markers;
- `fig_nll_by_retention`: the complete loss distribution at 0%, 25%, 50%, and
  100% original-pair retention;
- `fig_partition_ranking`: the full discovery ranking from 1 through 105; and
- `fig_confirmation`: selected discovery extrema on the untouched set, when
  confirmation results are supplied.

Requirements: PyTorch ≥ 2.4, `transformers`, `triton`, `datasets`, `matplotlib`,
and `wandb`. Data defaults to `HuggingFaceFW/fineweb-edu`; pass
`--data-files 'path/*.parquet'` to use local parquet shards.

## Experiment 2 boundary contribution and transfer

Fit the identifiable centered, sum-to-zero additive boundary model on the
complete 495-partition `k=4` discovery result. This also runs the nested-CV
prediction-only ridge diagnostic and freezes separate targeted and uniform
selection manifests for `k=3` and `k=5`:

```bash
python3 -m src.experiments.experiment2_boundary_contribution fit \
  --discovery-results <step-2000-discovery-results.jsonl> \
  --output-dir <boundary-model-output> \
  --seed 20260826 --uniform-size 30 \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp2-1b-h16-step-2000
```

Evaluate each frozen manifest with `python3 -m src.experiments.experiment2_mixed_width evaluate` using
`--split confirmation` and the unchanged checkpoint/fixed-token artifact. Then
analyze transfer without combining the targeted and uniform sampling purposes:

```bash
python3 -m src.experiments.experiment2_boundary_contribution analyze-transfer \
  --k3-results <k3-confirmation-results.jsonl> \
  --k3-selection <boundary-model-output/k3_selection.json> \
  --k5-results <k5-confirmation-results.jsonl> \
  --k5-selection <boundary-model-output/k5_selection.json> \
  --output-dir <transfer-analysis-output> \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp2-1b-h16-step-2000
```

The additive coefficients rank relative boundary-removal damage within a fixed
merge count. They do not imply improvement over native H16. Ridge interactions
are retained only as a within-`k=4` prediction diagnostic and are never
transferred across merge counts.

If the frozen `k=3` or `k=5` directional gate passes, prepare the corresponding
pre-registered continuation (`k=1/2` for `k=3`; `k=6/7` for `k=5`):

```bash
python3 -m src.experiments.experiment2_conditional_followup prepare \
  --discovery-results <step-2000-discovery-results.jsonl> \
  --boundary-effects <boundary-model-output/boundary_effects.csv> \
  --transfer-summary <transfer-analysis-output/transfer_summary.json> \
  --output-dir <conditional-followup-manifests> \
  --wandb-mode online --wandb-project 'MHAR Stuff'
```

Evaluate every manifest listed by `followup_gate.json` with
`python3 -m src.experiments.experiment2_mixed_width evaluate --split confirmation`. Then pass those
result/selection paths to `python3 -m src.experiments.experiment2_conditional_followup analyze`.
`k=1` and `k=7` are exhaustive; `k=2` and `k=6` use the frozen targeted plus
uniform sampling design. This stage reuses the confirmation split and is
therefore labeled a sequential follow-up, not a new untouched confirmation.

The checked-in sequential runner performs every eligible evaluation and the
final analysis without changing the frozen gate:

```bash
python3 -m src.experiments.experiment2_conditional_followup run \
  --gate-manifest <conditional-followup-manifests/followup_gate.json> \
  --manifests-dir <conditional-followup-manifests> \
  --checkpoint <step-2000-checkpoint> --artifact <fixed-eval.pt> \
  --output-dir <conditional-followup-results> \
  --wandb-mode online --wandb-project 'MHAR Stuff'
```

## Experiment 3: training-time boundary learnability

Experiment 3 deliberately precedes any learnable-boundary architecture. It
tests whether model-local boundary preferences exist during training, remain
stable over a short horizon, and improve future optimization when followed.
The master preregistration and executable subplans are:

- [`docs/plans/experiment-3.md`](docs/plans/experiment-3.md)
- [`docs/plans/experiment-3a-boundary-signal.md`](docs/plans/experiment-3a-boundary-signal.md)
- [`docs/plans/experiment-3b-temporal-stability.md`](docs/plans/experiment-3b-temporal-stability.md)
- [`docs/plans/experiment-3c-actionability.md`](docs/plans/experiment-3c-actionability.md)
- [`docs/plans/experiment-3d-cross-seed.md`](docs/plans/experiment-3d-cross-seed.md)
- [`docs/plans/experiment-3e-landscape.md`](docs/plans/experiment-3e-landscape.md)
- [`docs/runbooks/experiment-3.md`](docs/runbooks/experiment-3.md) — exact GPU setup and execution commands

The proof chain is locked: signal existence, short-term temporal stability,
branched-training actionability, and within-seed replication across three
seeds. Boundary-ID agreement across seeds is not required. A separate local
movement grid determines whether a later learner should use soft boundaries or
discrete periodic restructuring.

## Citation

```bibtex
@article{luo2026mhar,
  title={Multi-Head Attention Residuals},
  author={Cheng Luo and Zefan Cai and Junjie Hu},
  journal={arXiv preprint arXiv:2607.27230},
  year={2026}
}
```

## Related work by the authors

- [Delta Attention Residuals](https://github.com/wdlctc/delta-attention-residuals-code) —
  additive (identity-preserving) routing over sublayer deltas; the conversion technique
  used for 8B mid-training in this paper.
- [Open Attention Residuals](https://github.com/wdlctc/open-attention-residuals) — open
  implementation of attention residuals (Kimi Team, arXiv:2603.15031).
