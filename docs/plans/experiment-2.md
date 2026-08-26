
# Experiment 2 — Mixed-Width Contiguous MHAR Boundary Search

## Objective

Build on Experiment 1 and test a new hypothesis:

> MHAR should not necessarily divide the residual stream into equal-width routing heads. Some boundaries may cut through internally compatible residual structure, while other boundaries may be useful and should remain.

The goal is **not** to reorder residual coordinates and not to regroup distant pieces.

Coordinate order must remain completely unchanged.

We only change:

> **where the boundaries between MHAR routing heads are placed.**

---

# Background

For the 1B MHAR model:

\[
D=1280
\]

The two important uniform configurations are:

### H=16

\[
1280/16=80
\]

so the residual is divided as:

```text
[h0][h1][h2][h3]...[h15]
```

where every primitive region has width 80.

### H=8

Adjacent H=16 regions are merged:

```text
[h0+h1][h2+h3][h4+h5]...[h14+h15]
```

Each routing head has width 160.

The MHAR paper reports the U-shaped behavior where H=8 performs substantially better than H=16 at 1B.

This suggests that H=16 may over-split the residual stream.

However, uniform H=8 makes a stronger assumption:

> Every second H=16 boundary should be removed, uniformly across the entire residual width.

Experiment 2 tests whether this assumption is unnecessarily rigid.

---

# Main Hypothesis

Different regions of the residual stream may require different routing granularity.

Some neighboring 80-dimensional H=16 regions may be better treated as one 160-dimensional routing region:

\[
[h_i+h_{i+1}]
\]

while other neighboring regions may benefit from remaining independent:

\[
[h_j][h_{j+1}].
\]

Therefore, a mixed-width partition such as:

```text
[h0+h1]
[h2]
[h3]
[h4+h5]
[h6]
[h7]
[h8+h9]
[h10]
[h11]
[h12+h13]
[h14]
[h15]
```

may outperform a uniformly split configuration.

This keeps:

- coordinate order unchanged;
- total residual width unchanged;
- the same underlying residual representation;
- contiguous routing regions only.

It changes only which H=16 boundaries are removed.

---

# Terminology

Do not call this architecture ordinary `H=12`.

It has 12 routing groups in the example above, but they have unequal widths.

Use:

> **mixed-width MHAR**

or:

> **mixed-width contiguous partition**

For the half-H8 / half-H16 condition:

- 4 routing groups have width 160;
- 8 routing groups have width 80;
- total routing groups = 12;
- half of the 1280 residual dimensions belong to H=8-sized groups;
- half belong to H=16-sized groups.

---

# Atomic Representation

Treat the H=16 partition as the atomic coordinate segmentation:

\[
h_0,h_1,\ldots,h_{15}
\]

with:

\[
h_i\in\mathbb{R}^{80}.
\]

Their order may never change.

Allowed operation:

\[
[h_i][h_{i+1}]
\rightarrow
[h_i+h_{i+1}]
\]

Not allowed:

```text
[h0+h7]
[h3+h12]
```

or any coordinate permutation.

Only adjacent atomic regions may merge.

---

# Primary Experiment

## Half-H8 / Half-H16 Partition

Start from 16 atomic H=16 regions.

Merge exactly four non-overlapping adjacent pairs.

This creates:

\[
4\times160 + 8\times80 = 1280
\]

with:

\[
4+8=12
\]

routing groups.

Example:

```text
[h0+h1]
[h2]
[h3]
[h4+h5]
[h6]
[h7]
[h8+h9]
[h10]
[h11]
[h12+h13]
[h14]
[h15]
```

The experiment asks:

> With exactly the same number of merged and unmerged dimensions, does the location of the four removed boundaries affect loss?

---

# Exhaustive Search Space

For 16 ordered atomic regions, choose four non-overlapping adjacent pairs to merge.

The number of valid configurations is:

\[
\binom{16-4}{4}
=
\binom{12}{4}
=
495.
\]

Therefore:

> **Enumerate and evaluate all 495 possible half-H8 / half-H16 boundary patterns.**

Do not randomly sample them.

Every configuration must contain:

- exactly four 160-d groups;
- exactly eight 80-d groups;
- exactly twelve routing softmaxes;
- exactly the same coordinate order.

The only independent variable is:

\[
\boxed{\text{which four H=16 boundaries are removed}}
\]

---

# Why This Experiment Matters

Experiment 1 found preliminary evidence in the seed-42, step-2000 H=4
checkpoint that routing-group composition matters.  Treat that result as
preliminary until the preregistered independent repetitions are complete.

Experiment 2 asks a more architectural question:

> Can we improve MHAR while preserving coordinate locality simply by choosing better contiguous boundaries?

This separates two concepts:

### Uniform head count

Current MHAR chooses one width globally:

\[
80,80,80,\ldots
\]

or:

\[
160,160,160,\ldots
\]

### Adaptive local granularity

Experiment 2 allows:

\[
160,80,80,160,80,80,\ldots
\]

Different residual regions can therefore use different routing granularity.

---

# Important Implementation Requirement

Do not reorder coordinates.

For each routing group, keep each query coefficient attached to its original residual coordinate.

For a merged group:

\[
G=[h_i,h_{i+1}]
\]

construct the routing score from the corresponding query and normalized key coordinates:

\[
q_G=[q_i,q_{i+1}]
\]

\[
K_G=[K_i,K_{i+1}]
\]

then compute one depth softmax:

\[
\alpha_G
=
\operatorname{softmax}_{depth}
(q_G^\top K_G).
\]

Use that same depth distribution to mix the corresponding value coordinates.

For an unmerged atomic region:

\[
G=[h_i]
\]

retain its independent 80-dimensional depth softmax.

After routing, place every output coordinate back in its original location.

The residual coordinate order must be identical before and after routing.

---

# Kernel Implementation

The current MHAR implementation assumes equal-size contiguous heads through `view(...)`.

Mixed-width routing cannot use that assumption.

Implement a separate eager reference path first, for example:

```text
mixed_width_mhar_eager(...)
```

Inputs should include:

- source tensor;
- routing query;
- RMSNorm;
- ordered segment boundaries or segment widths.

Example configuration:

```python
segment_widths = [
    160,
    80,
    80,
    160,
    80,
    80,
    160,
    80,
    80,
    160,
    80,
    80,
]
```

or preferably explicit atomic intervals.

Do not optimize the Triton kernel until correctness is established.

---

# Required Parity Tests

## Pure H=16

Configuration:

```text
80 × 16
```

The mixed-width eager implementation must reproduce ordinary MHAR H=16 numerically.

## Pure H=8

Configuration:

```text
160 × 8
```

The mixed-width eager implementation must reproduce ordinary MHAR H=8 numerically.

Check:

\[
\max |y_{\mathrm{new}}-y_{\mathrm{existing}}|
\]

within normal floating-point tolerance.

Only proceed after both endpoint configurations pass.

Also require:

- forward and gradient parity for query, RMSNorm parameters, and source values;
- complete-model forward parity at the pure H=16 endpoint;
- one hand-computed mixed-width case containing both 80-d and 160-d groups.

---

# Evaluation Design

Materialize two new, fixed, document-disjoint token artifacts specifically for
Experiment 2:

- **discovery:** every one of the 495 configurations sees exactly these tokens;
- **confirmation:** remains untouched until discovery ranking and candidate
  selection have been frozen.

Do not advance a streaming validation iterator separately for different
partitions.  Do not reuse an already-opened Experiment 1 confirmation artifact
as the Experiment 2 untouched confirmation set.

Primary metric:

\[
\text{token-weighted NLL}
\]

Secondary metric:

\[
PPL=e^{NLL}.
\]

For each of the 495 configurations record:

```text
configuration_id
merged_pairs
segment_widths
number_of_routing_groups
NLL
ΔNLL_vs_H16
PPL
elapsed_seconds
tokens_per_second
peak_GPU_memory
checkpoint_SHA256
evaluation_artifact_SHA256
git_commit
```

The frozen-search delta is defined only against the native checkpoint:

\[
\Delta NLL(P)
=
NLL(\text{H16 checkpoint with }P)
-
NLL(\text{original H16 computation}).
\]

A pure H=8 segmentation may be evaluated on the same tensors as an operator
endpoint, but it must be labelled **frozen H16→H8 conversion**.  It is not an
architectural H=8 control.

---

# Critical Experimental Distinction

There are two stages.

## Stage A — Frozen Boundary Search

Train or load a checkpoint whose native routing configuration is H=16.  The
sixteen trained 80-dimensional routing regions are the atomic units of this
experiment.  The existing Experiment 1 H=4 checkpoint is not valid for this
question because its 80-dimensional slices were not independently routed
during training.

Use the already-preregistered 1B training scale:

- `hidden_size=1280`, `num_hidden_layers=36`;
- 16 attention heads and 8 KV heads;
- FFN width 5120;
- sequence length 1024, global batch 32;
- FineWeb-Edu with pinned dataset and tokenizer revisions;
- AdamW, peak learning rate \(5\times10^{-4}\), bf16;
- 20,000 training steps, with the same schedule and data order as the matched
  controls.

For a pilot analysis, a preregistered milestone checkpoint may be used, but the
milestone must be stated in every result and must not be presented as the final
20,000-step architectural result.

The frozen search uses four preregistered checkpoint gates:

\[
2{,}000,\quad 5{,}000,\quad 10{,}000,\quad 20{,}000\text{ steps}.
\]

At each gate, wait for the atomic checkpoint, pause training, and run the full
495-partition discovery plus untouched confirmation workflow.  Report and
review the completed milestone before resuming toward the next gate.  Never
resume automatically from 2,000 to 5,000, from 5,000 to 10,000, or from 10,000
to 20,000.

Apply one partition \(P\) globally and identically at all 73 routing sites:

\[
P_1=P_2=\cdots=P_{73}=P.
\]

Evaluate the original native H=16 computation once, then enumerate all 495
mixed-width partitions on discovery data.

Purpose:

> Determine whether different choices of contiguous boundaries produce different routing quality and identify promising boundary patterns.

This is a discovery experiment.

After discovery, freeze a confirmation set containing:

- original H=16;
- the best five discovery partitions;
- a median partition;
- the worst partition;
- any additional structurally diverse candidate selected by a rule written
  before confirmation is opened.

Evaluate this frozen selection once on the untouched confirmation artifact.

Do not compare a frozen mixed-width winner directly with a separately trained
H=8 model.  Stage A asks only where an H16-trained model tolerates or benefits
from removing boundaries.

---

## Stage B — Architectural Confirmation

If Stage A finds promising mixed-width patterns that survive untouched
confirmation, freeze one to three candidates before training.  Train from
scratch:

- uniform H=8;
- uniform H=16;
- each selected mixed-width partition.

Use the same:

- 1B architecture;
- training data;
- optimizer;
- schedule;
- training steps;
- model width;
- evaluation protocol;
- corpus order and initialization protocol;

as the H=8 and H=16 controls.

The first matched architectural control is uniform H=8. Train it from scratch
with seed 42 and the exact H=16 FineWeb-Edu recipe; only `attnres_num_heads`
changes from 16 to 8. Preserve checkpoints at 2,000, 5,000, 10,000, and 20,000
steps so H=8 and H=16 can be compared at equal training progress. The checked-in
launcher is `scripts/train/run_experiment2_train_1b_h8.sh`, and its W&B group is
`mhar-exp2-stage-b-1b-h8-fineweb-edu`.

### Frozen seven-run screening matrix

For the seed-42 architectural screen, train these seven models from scratch:

| Run | Removed H16 boundaries | Routing groups | Role |
|---|---:|---:|---|
| H16 | none | 16 | fine-grained baseline |
| H8 | all alternating boundaries | 8 | strong MHAR baseline |
| mixed k=2 | `[6,14]` | 14 | light merging |
| mixed k=3 | `[6,8,14]` | 13 | medium-light merging |
| mixed k=4 best | `[2,6,9,14]` | 12 | midpoint |
| mixed k=5 | `[2,6,9,11,14]` | 11 | heavier merging |
| mixed k=4 worst | `[0,4,8,10]` | 12 | bad-partition control |

The k=4 worst choice is frozen as the maximum discovery NLL among all 495
step-2,000 k=4 choices, with ties resolved by partition id. The best and worst
k=4 models have the same eight 80-dimensional singleton groups and four
160-dimensional doubleton groups. Only the four removed boundary locations
differ. The machine-readable preregistration is
`configs/experiment2/stage-b-screening.json`.

All seven runs share one W&B group and retain the same 2,000, 5,000, 10,000,
and 20,000 step checkpoints. Uniform H16/H8 use the existing fused routing
kernel. Mixed-width routing currently uses the correctness-tested eager path;
therefore mathematical routing cost is controlled, but measured training
throughput is not yet a systems control. Do not compare runtime until a fused
mixed-width kernel passes forward, gradient, and optimizer-step parity tests.

Use at least three independent seeds per architecture.  Report both the
per-seed results and the across-seed mean with uncertainty.  Because the
architectures execute different numbers of routing softmaxes, also report
tokens/second, peak memory, and total training time; parameters are matched but
systems cost is not exactly identical.

Then compare:

\[
L_{H8}
\]

\[
L_{H16}
\]

\[
L_{\mathrm{mixed}}.
\]

This is the experiment that determines whether mixed-width MHAR itself is superior.

---

# Three Primary Outcome Cases

The following cases apply only to the separately trained Stage B models.  They
must not be inferred from frozen Stage A conversions.

## Case 1 — Mixed-width partition is worse than H=16

\[
L_{\mathrm{mixed}}>L_{H16}>L_{H8}.
\]

Interpretation:

> This particular boundary pattern is bad.

It does not reject the overall mixed-width hypothesis.

Because there are many possible cutting patterns, test the complete search space and determine whether other boundary choices perform better.

---

# Case 2 — Mixed-width partition falls between H=16 and H=8

\[
L_{H8}
<
L_{\mathrm{mixed}}
<
L_{H16}.
\]

Interpretation:

> Selectively removing some H=16 boundaries improves routing compared with full H=16 over-splitting.

This establishes that some boundaries are more useful to remove than leaving the residual uniformly H=16.

However, it does not yet show that mixed-width routing is superior to ordinary H=8.

This result supports:

> Different locations along residual width may prefer different routing granularity.

but uniform H=8 remains the stronger architecture.

---

# Case 3 — Mixed-width partition beats H=8

\[
L_{\mathrm{mixed}}
<
L_{H8}
<
L_{H16}.
\]

This is the target result.

Interpretation:

> A non-uniform contiguous residual partition can outperform both uniform H=8 and uniform H=16 routing.

This provides strong evidence that:

\[
\boxed{
\text{optimal MHAR boundaries need not be equally spaced}
}
\]

and:

\[
\boxed{
\text{boundary location is an architectural variable}
}
\]

This supports the hypothesis that the residual stream contains regions with different optimal routing granularity.

Do **not** claim yet that:

- individual latent features were identified;
- feature boundaries were directly discovered;
- the optimal boundaries can already be learned automatically.

The correct next research question becomes:

> How can MHAR learn these boundary locations automatically during training?

---

# Additional Possible Result

There may be substantial variation among the 495 configurations even if none beats H=8.

Example:

```text
best mixed:   2.77
median mixed: 2.81
worst mixed:  2.87
H8:           2.75
H16:          2.83
```

This would still be important.

It would show:

\[
\boxed{
\text{with exactly the same width composition and router count,
boundary placement alone changes performance}
}
\]

That directly validates boundary location as a meaningful variable.

---

# Extended Experiment: Full H16 → H8 Boundary-Removal Path

After the primary four-merge experiment, optionally evaluate different numbers of removed boundaries.

Let:

\[
k=\text{number of non-overlapping adjacent H16 pairs merged}.
\]

Then:

\[
H_{\mathrm{routing}}=16-k.
\]

The configurations are:

| Merged pairs \(k\) | Width composition | Routing groups |
|---:|---|---:|
| 0 | 16 × 80 | 16 |
| 1 | 1 × 160 + 14 × 80 | 15 |
| 2 | 2 × 160 + 12 × 80 | 14 |
| 3 | 3 × 160 + 10 × 80 | 13 |
| 4 | 4 × 160 + 8 × 80 | 12 |
| 5 | 5 × 160 + 6 × 80 | 11 |
| 6 | 6 × 160 + 4 × 80 | 10 |
| 7 | 7 × 160 + 2 × 80 | 9 |
| 8 | 8 × 160 | 8 |

For a given \(k\), the number of valid non-overlapping adjacent-merge configurations is:

\[
N_k=\binom{16-k}{k}.
\]

The primary experiment is:

\[
k=4,
\qquad
N_4=495.
\]

The endpoints are:

\[
k=0\Rightarrow H=16
\]

and:

\[
k=8\Rightarrow H=8.
\]

This extended analysis can reveal whether loss improves smoothly as selected boundaries are removed or whether the **specific locations** dominate over the raw number of heads.

---

# Main Analyses

For the 495 primary configurations produce:

1. Full NLL distribution.
2. Best mixed-width partition.
3. Worst mixed-width partition.
4. Median partition.
5. Difference between best and worst.
6. Difference between best mixed partition and H=16.
7. Descriptive difference between the best mixed partition and the frozen
   H16→H8 operator endpoint, clearly labelled as a conversion rather than a
   trained H=8 comparison.
8. Frequency with which each H=16 boundary is removed among the top-performing configurations.
9. Frequency with which each boundary remains among the top-performing configurations.
10. A boundary importance map across the 1280-dimensional residual axis.

A useful descriptive boundary association score is:

\[
S_i
=
E[L\mid\text{boundary }i\text{ kept}]
-
E[L\mid\text{boundary }i\text{ removed}].
\]

Interpretation:

- large positive \(S_i\): removing the boundary tends to help;
- negative \(S_i\): preserving the boundary tends to help.

Because non-overlapping merges make neighboring boundary choices dependent,
this score is not a causal boundary effect.  Adjust top-partition frequencies
for each boundary's null inclusion probability and support the descriptive map
with either matched-swap contrasts or a constrained fixed-\(k\) indicator
regression.  Use paired document- or sequence-level bootstrap intervals on the
untouched confirmation set.

---

# Success Criteria

Experiment 2 has multiple levels of success.

### Minimum positive result

Different mixed-width boundary configurations produce different discovery NLL
despite having identical width composition and routing-group count, and the
preselected contrast survives untouched confirmation with a paired confidence
interval excluding zero.

This proves:

> Boundary placement matters.

### Strong result

The preselected mixed-width configurations outperform the native frozen H=16
computation on untouched confirmation.

This proves:

> Selective merging is better than uniform over-splitting.

### Target result

A mixed-width configuration trained and evaluated under matched conditions
outperforms H=8 across the preregistered training seeds.

This establishes:

> Equal-width routing heads are a limitation of current MHAR.

---

# Final Research Progression

Experiment 1 provisionally found:

\[
\boxed{\text{which residual subspaces share a router matters}}
\]

Experiment 2 asks:

\[
\boxed{\text{can better contiguous boundary placement improve MHAR?}}
\]

If Experiment 2 succeeds, Experiment 3 becomes:

\[
\boxed{\text{can the model learn its own routing-head boundaries?}}
\]

The eventual architecture should preserve residual coordinate order while learning where routing groups begin and end.

---

# Locked Execution Sequence

1. Train or load the matched native H=16 checkpoint.
2. Implement and parity-test the mixed-width eager router.
3. Enumerate all 495 valid \(k=4\) partitions.
4. Apply each partition globally at all 73 routing sites.
5. Search on fixed Experiment 2 discovery tokens using native H=16 as the only
   clean frozen baseline.
6. Freeze the candidate-selection manifest before opening confirmation data.
7. Confirm the selected partitions once on untouched tokens.
8. Freeze one to three confirmed mixed-width candidates.
9. Train H=8, H=16, and the selected mixed architectures from scratch with the
   matched recipe and at least three seeds.
10. Compare final token-weighted NLL and systems cost.

The claim boundary is fixed:

- **Frozen H16 search:** where can an H16-trained model tolerate or benefit
  from removing boundaries?
- **Matched retraining:** is mixed-width MHAR a better architecture than
  equal-width MHAR?

All stages must write resumable JSONL results and immutable manifests, log to
the W&B project `MHAR Stuff`, record code/checkpoint/data hashes, and preserve
the discovery/confirmation separation.
