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
paper/                              arXiv source + PDF
Attention-Residuals/
  modeling_qwen3_attnres.py         Qwen3-style architecture with all attention-residual
                                    modes (baseline / full / full_mh = MHAR / delta / ...)
  modeling_llama_attnres.py         Llama-style architecture for converting pretrained
                                    checkpoints (8B mid-training via delta_mh)
  mhar_triton.py                    fused Triton routing kernels
train_scratch.py                    from-scratch pretraining entry point
train_cpt.py                        continued-pretraining (conversion) entry point
test_mhar_fused.py                  fused-vs-eager correctness tests
test_mhar_fused_delta.py            fused-vs-eager tests for the delta variant
```

## Quick start

From-scratch pretraining with MHAR (`full_mh`, 8 routing heads):

```bash
torchrun --nproc_per_node=8 train_scratch.py \
    --mode full_mh --attnres_heads 8
```

Single-head attention residuals (the `H = 1` special case):

```bash
torchrun --nproc_per_node=8 train_scratch.py --mode full
```

Standard Transformer baseline:

```bash
torchrun --nproc_per_node=8 train_scratch.py --mode baseline
```

Convert a pretrained checkpoint with the identity-preserving delta variant
(zero disruption at step 0), then continue pretraining:

```bash
torchrun --nproc_per_node=8 train_cpt.py \
    --mode delta_mh --attnres_heads 8
```

Verify the fused Triton kernels against the eager path:

```bash
python test_mhar_fused.py
python test_mhar_fused_delta.py
```

## Experiment 1: exhaustive residual partitions

The frozen H=4 compatibility experiment evaluates all 105 pairings of eight
160-dimensional primitive blocks into four routing groups.  The same partition
is applied at all 73 routing sites, and every candidate sees the same fixed
tokens.  The full preregistration is in
[`experiment 1 plan.md`](experiment%201%20plan.md).

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
./run_experiment1_train_1b_h4.sh
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
./run_experiment1_train_1b_h4.sh \
  /fast-disk/experiment1/checkpoint-1b-h4-fineweb-edu/step-500
```

Training logs to W&B project `MHAR Stuff`, group
`mhar-exp1-1b-h4-fineweb-edu`. W&B is required: the process aborts instead of
silently running untracked.

Install the pinned experiment runtime and run the CPU correctness suite:

```bash
python3 -m pip install -r requirements-experiment1.txt
MHAR_RUN_MODEL_INTEGRATION=1 python3 -m unittest -v \
  test_experiment1_partitions.py test_experiment1_runner.py test_train_resume.py
```

Materialize non-overlapping discovery and confirmation tensors from an explicit
dataset revision or local parquet manifest:

```bash
python3 experiment1_partition_compatibility.py materialize \
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
python3 experiment1_partition_compatibility.py evaluate \
  --checkpoint <1b-h4-full_mh-checkpoint> \
  --artifact output/experiment1/fixed_eval.pt \
  --output-dir output/experiment1/run \
  --split discovery --device cuda --dtype bf16 --batch-size 1 \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4

python3 experiment1_partition_compatibility.py analyze \
  --discovery-results output/experiment1/run/discovery_results.jsonl \
  --output-dir output/experiment1/run/analysis \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4

python3 experiment1_partition_compatibility.py evaluate \
  --checkpoint <1b-h4-full_mh-checkpoint> \
  --artifact output/experiment1/fixed_eval.pt \
  --output-dir output/experiment1/run \
  --split confirmation \
  --discovery-results output/experiment1/run/discovery_results.jsonl \
  --device cuda --dtype bf16 --batch-size 1 \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp1-1b-h4

python3 experiment1_partition_compatibility.py analyze \
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
  ./run_experiment1_after_training.sh
```

For staged checkpoint inspection, run the milestone controller from a separate
screen session. It waits for an atomically completed checkpoint, interrupts the
training screen, runs all 105 discovery partitions, selects and evaluates the
reference/best/worst partitions on the untouched confirmation set, and then
exits so the result can be reviewed before training resumes:

```bash
MHAR_PYTHON_BIN=/root/autodl-tmp/mhar-venv/bin/python \
MHAR_REPO_DIR=/root/mhar-experiment \
./run_experiment1_milestone.sh 2000
```

The preregistered milestones are 2,000, 5,000, 10,000, and 20,000 steps. Resume
from the reviewed checkpoint with `run_experiment1_train_1b_h4.sh <checkpoint>`
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
python3 experiment2_boundary_contribution.py fit \
  --discovery-results <step-2000-discovery-results.jsonl> \
  --output-dir <boundary-model-output> \
  --seed 20260826 --uniform-size 30 \
  --wandb-mode online --wandb-project 'MHAR Stuff' \
  --wandb-group mhar-exp2-1b-h16-step-2000
```

Evaluate each frozen manifest with `experiment2_mixed_width.py evaluate` using
`--split confirmation` and the unchanged checkpoint/fixed-token artifact. Then
analyze transfer without combining the targeted and uniform sampling purposes:

```bash
python3 experiment2_boundary_contribution.py analyze-transfer \
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
python3 experiment2_conditional_followup.py prepare \
  --discovery-results <step-2000-discovery-results.jsonl> \
  --boundary-effects <boundary-model-output/boundary_effects.csv> \
  --transfer-summary <transfer-analysis-output/transfer_summary.json> \
  --output-dir <conditional-followup-manifests> \
  --wandb-mode online --wandb-project 'MHAR Stuff'
```

Evaluate every manifest listed by `followup_gate.json` with
`experiment2_mixed_width.py evaluate --split confirmation`. Then pass those
result/selection paths to `experiment2_conditional_followup.py analyze`.
`k=1` and `k=7` are exhaustive; `k=2` and `k=6` use the frozen targeted plus
uniform sampling design. This stage reuses the confirmation split and is
therefore labeled a sequential follow-up, not a new untouched confirmation.

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
