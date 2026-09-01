# Experiment 6 — Coupled MHAR Chunks to Attention Groups

## Status

Implementation complete; no GPU run has started. The first screen is frozen to
seed 42 and step 2,000. It is not a multi-seed confirmation experiment.

## Question

Does preserving each depth-routed MHAR coordinate chunk through Q/K/V
projection improve learning relative to immediately remixing the concatenated
routed representation with dense Q/K/V?

The intervention is block-diagonal Q/K/V connectivity. Attention itself,
Q/K normalization, RoPE, GQA repetition, and the dense output projection stay
unchanged.

## Fixed 1B backbone

All runs use the prior matched recipe:

- width 1,280 and 36 layers;
- 16 query heads, 8 KV heads, head dimension 80;
- FFN width 5,120;
- sequence length 1,024 and global batch 32;
- FineWeb-Edu `sample-10BT` at pinned revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- Qwen3 tokenizer at pinned revision
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- AdamW, bf16, peak LR 5e-4, minimum LR 5e-5, and 1,000-step warmup;
- the original 20,000-step schedule, stopped atomically at step 2,000;
- atomic checkpoints every 100 steps, retaining the latest two plus protected
  500/1,000/1,500/2,000 milestones for balance-loss recovery;
- seed 42 and the same packed data order;
- the Experiment 2 Stage B document-disjoint fixed evaluation artifact with
  SHA-256 `29e545dd9399e9eaea6f5abf38f20ef76ba232466a01b6ddf13c3a6a287a3691`.

## Frozen run matrix

| ID | Residual routing | QKV input | Status |
|---|---|---|---|
| B | ordinary residual | dense | new run |
| M4 | MHAR 4 × 320 | dense | reuse accepted seed-42 step-2,000 result |
| C4 | MHAR 4 × 320 | four restricted groups | new run |
| G4 | ordinary residual | four restricted groups | new run |
| M8 | MHAR 8 × 160 | dense | reuse accepted seed-42 step-2,000 result |
| C8 | MHAR 8 × 160 | eight restricted groups | new run |
| G8 | ordinary residual | eight restricted groups | new run |

M4 and M8 are the prior Stage B H4 and H8 runs. Their accepted per-sequence
results use the exact fixed artifact required here. They are reused only after
dense-path regression tests pass. Their model checkpoints are still required
if later continuation beyond step 2,000 is authorized.

## Connectivity and initialization

For four groups, each 320-dimensional input chunk feeds four Q heads and two KV
heads. For eight groups, each 160-dimensional chunk feeds two Q heads and one
KV head. Grouped Q/K/V weights are stored without disallowed parameters:

\[
W = \operatorname{diag}(W^{(0)},\ldots,W^{(G-1)}).
\]

The allowed blocks are copied from a normally initialized dense model. Thus B
and G share every initial value that exists in G, and M and C share every
initial value that exists in C. All non-QKV weights, including MHAR routers and
dense \(W_O\), match within each pair.

The restricted models are not parameter-matched to dense models. At the real
1B shape, dense QKV has 3,276,800 weights per layer, four-group QKV has 819,200,
and eight-group QKV has 409,600. Report trainable parameters, QKV MACs per
token, throughput, memory, and total time alongside quality.

## Primary outcomes

Use held-out token-weighted NLL and paired per-sequence bootstrap intervals:

\[
C4-M4,\qquad C8-M8,
\]

\[
G4-B,\qquad G8-B,
\]

and the difference-in-differences interactions:

\[
(C4-M4)-(G4-B),
\]

\[
(C8-M8)-(G8-B).
\]

Negative values favor coupling or a beneficial MHAR interaction.

## Screening interpretation

This is one seed at an early checkpoint. It can reject catastrophic designs or
justify a longer test, but it cannot establish a winner. Before results are
opened, a primary restriction contrast above +0.5 NLL is labeled catastrophic;
non-finite loss, non-finite held-out NLL, or an incomplete checkpoint is also
catastrophic. No automatic continuation and no multi-seed run are authorized.

## Isolation safeguards

- The MHAR routing kernels and Experiment 1–5 partition functions are unchanged.
- Grouped QKV is enabled only by `--qkv_groups 4|8`.
- The option is legal only for ordinary baseline or matching `full_mh`.
- Without that option, the legacy training identity omits every Experiment 6
  field, preserving old exact-resume compatibility.
- Unit tests compare grouped projection with an explicit masked dense matrix,
  check matched initialization, preserve dense `o_proj`, check parameter
  counts, and exercise a complete forward pass.
- All existing relevant regression tests must pass before GPU launch.

## Commands

Launch the five new runs on five GPUs:

```bash
scripts/train/launch_experiment6_5gpu.sh
```

Run and evaluate one variant:

```bash
scripts/train/run_experiment6_screen.sh c8
scripts/evaluate/run_experiment6_screen.sh c8
```

After all five results exist, combine them with M4/M8:

```bash
python3 -m src.experiments.experiment6_screening analyze \
  --results-root /root/autodl-tmp/experiment6/screening/results \
  --output-dir /root/autodl-tmp/experiment6/screening/analysis
```
