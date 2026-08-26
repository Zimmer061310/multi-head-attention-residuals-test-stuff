# Exhaustive Residual Partition Compatibility in MHAR

## 1. Research question

For a fixed four-head MHAR routing capacity, does changing which residual
subspaces are paired to share one depth softmax affect language-model loss?

The experiment keeps fixed:

- the trained model and every parameter;
- the source history available to each routing site;
- the total residual width;
- the number and width of routing heads;
- the number of depth softmaxes;
- the mathematical routing operation; and
- the evaluation tokens.

It changes only:

> which two 160-dimensional residual subspaces share one depth softmax.

The primary experiment uses a **1B model trained with MHAR \(H=4\)** and
exhaustively evaluates all 105 possible pairings of eight primitive residual
subspaces into four routing groups.

---

## 2. Hypothesis and terminology

### 2.1 Latent routing-group coherence

The motivating hypothesis is that residual subspaces differ in the depth
history they would prefer to read. Subspaces forced to share one softmax are a
better routing group when their preferred depth decisions are compatible.

Call this unobserved property **latent routing-group coherence**.

This does not mean human-interpretable semantic purity. The experiment does not
claim to identify concepts, syntax, entities, or reasoning features.

### 2.2 Directly manipulated and measured structure

The experiment does not observe latent routing-group coherence directly. It
manipulates the partition and computes two structural descriptors:

1. **Original-pair retention**: how many pairs from the trained contiguous
   partition remain intact.
2. **Coordinate distance**: how far apart the paired primitive blocks are in
   the original residual-coordinate order.

Loss determines whether one partition works better than another. These
structural descriptors test whether any loss effect is specifically related to
the trained contiguous partition or to coordinate locality.

### 2.3 Hypotheses

Primary hypothesis:

\[
\exists P_a,P_b:\quad \operatorname{NLL}(P_a)\ne
\operatorname{NLL}(P_b).
\]

That is, partition membership affects frozen-model performance.

Locality hypothesis:

\[
D(P)\uparrow \quad\Longrightarrow\quad \operatorname{NLL}(P)\uparrow.
\]

Null result:

The 105 partitions produce no practically detectable NLL differences on the
fixed evaluation set.

---

## 3. Actual 1B MHAR setup

Use the paper's 1B Qwen3-style configuration:

- model width \(d_{\text{model}}=1280\);
- 36 Transformer layers;
- 16 ordinary attention heads;
- 8 KV heads;
- FFN width 5120;
- sequence length 1024;
- global training batch 32;
- 20,000 training steps;
- AdamW;
- peak learning rate \(5\times10^{-4}\);
- bfloat16; and
- MHAR \(H=4\) for the primary checkpoint.

The primary checkpoint must have been trained with four routing heads. Its
ordinary routing partition is four contiguous 320-dimensional slices.

MHAR routes twice in each of the 36 layers—once before attention and once
before the MLP—and routes once more after the final layer:

\[
36\times2+1=73
\]

routing sites.

The primary intervention applies the same candidate partition at all 73 sites.

### 3.1 Checkpoint requirement

Before evaluation, record:

- checkpoint path and SHA-256 hash;
- architecture/configuration dump;
- training step;
- tokenizer revision;
- dataset revision or manifest; and
- software commit and dependency versions.

If a compatible trained \(H=4\) checkpoint is unavailable, it must be trained
with the fixed recipe before this experiment begins. Do not substitute an
\(H=8\) checkpoint in the primary experiment.

---

## 4. Partition space

### 4.1 Primitive residual blocks

Split the 1280-dimensional residual coordinate axis into eight contiguous
primitive blocks:

\[
x=[h_0,h_1,\ldots,h_7],\qquad h_i\in\mathbb{R}^{160}.
\]

The four-head checkpoint's trained contiguous partition is:

\[
P_0=\{(0,1),(2,3),(4,5),(6,7)\}.
\]

Each candidate partition pairs the eight primitive blocks into four unordered
pairs. Every final routing group therefore has width 320.

### 4.2 Number of partitions

The number of unordered perfect matchings of eight objects is:

\[
|\mathcal P|=\frac{8!}{2^4 4!}=105.
\]

Generate all 105 partitions deterministically and evaluate every one. Random
partition sampling is not part of the primary experiment.

Canonicalize each partition by:

1. sorting the two indices inside each pair; and
2. lexicographically sorting the four pairs.

Assert that generation produces exactly 105 unique canonical partitions.

Examples:

\[
P_0=\{(0,1),(2,3),(4,5),(6,7)\},
\]

\[
P_1=\{(0,2),(1,3),(4,6),(5,7)\},
\]

\[
P_2=\{(0,7),(1,6),(2,5),(3,4)\}.
\]

---

## 5. Partition descriptors

These descriptors are computed before looking at evaluation loss.

### 5.1 Original-pair retention

Let

\[
P_0=\{(0,1),(2,3),(4,5),(6,7)\}.
\]

Define:

\[
R(P)=\frac{|P\cap P_0|}{4}.
\]

The exhaustive partition set has the following distribution:

| Original pairs retained | Retention | Number of partitions |
|---:|---:|---:|
| 4 | 100% | 1 |
| 2 | 50% | 12 |
| 1 | 25% | 32 |
| 0 | 0% | 60 |

There is no 75% condition: preserving three original pairs forces the final two
blocks to form the fourth original pair.

### 5.2 Coordinate distance

For partition \(P\), define mean within-pair coordinate distance:

\[
D(P)=\frac{1}{4}\sum_{(i,j)\in P}|i-j|.
\]

The reference partition has \(D(P_0)=1\). The maximally separated example

\[
\{(0,7),(1,6),(2,5),(3,4)\}
\]

has \(D=4\).

Across all 105 partitions:

| Total pair distance | Mean distance \(D(P)\) | Number of partitions |
|---:|---:|---:|
| 4 | 1.0 | 1 |
| 6 | 1.5 | 6 |
| 8 | 2.0 | 12 |
| 10 | 2.5 | 20 |
| 12 | 3.0 | 24 |
| 14 | 3.5 | 18 |
| 16 | 4.0 | 24 |

Coordinate distance is an intervention descriptor, not a direct measurement of
latent features.

---

## 6. Arbitrary-group MHAR operation

### 6.1 Required semantics

At one routing site, let the source tensor be:

\[
V\in\mathbb{R}^{N\times B\times T\times1280}.
\]

First apply the existing full-width RMSNorm exactly as ordinary MHAR does:

\[
K=\operatorname{RMSNorm}(V).
\]

Conceptually split \(K\), \(V\), and the learned routing query \(q\) into the
same eight coordinate-attached 160-dimensional blocks:

\[
K=[K^{(0)},\ldots,K^{(7)}],
\]

\[
V=[V^{(0)},\ldots,V^{(7)}],
\]

\[
q=[q^{(0)},\ldots,q^{(7)}].
\]

For a candidate group \(G=(a,b)\), gather:

\[
K_G=[K^{(a)},K^{(b)}],\qquad
V_G=[V^{(a)},V^{(b)}],\qquad
q_G=[q^{(a)},q^{(b)}].
\]

The query coefficients remain attached to their original residual coordinates.

Compute one depth distribution for the pair:

\[
\alpha_s^G=operatorname{softmax}_s(q_G^\top K_{s,G}),
\]

then route:

\[
y_G=\sum_s \alpha_s^G V_{s,G}.
\]

Finally scatter the two 160-dimensional outputs back to their original residual
coordinates. No coordinate permutation persists beyond the routing operation.

Thus only the relationship "these two primitive blocks share a depth softmax"
changes.

### 6.2 Implementation rule

Implement a separate eager reference operation, for example:

```python
arbitrary_group_mhar_eager(V, proj, norm, partition)
```

Do not modify or use the fused Triton routing path for Experiment 1. The current
fused path assumes contiguous heads through reshape/view operations. Every one
of the 105 candidates, including the reference, must use the same eager
arbitrary-group implementation during the experiment.

---

## 7. Required correctness gates

No experimental run begins until all gates pass.

### 7.1 Partition-generation test

- exactly 105 partitions are generated;
- every partition contains four disjoint pairs;
- every index 0 through 7 appears exactly once;
- canonical representations are unique; and
- the retention and coordinate-distance counts match the tables above.

### 7.2 Reference identity test

For

\[
P_0=\{(0,1),(2,3),(4,5),(6,7)\},
\]

the new arbitrary-group eager operation must reproduce the existing ordinary
eager MHAR operation with \(H=4\) to an explicitly recorded numerical tolerance:

\[
\max |y_{\text{arbitrary}}-y_{\text{ordinary}}|\le \epsilon.
\]

Test representative source counts, batch sizes, sequence lengths, dtypes, and
random seeds. Record maximum absolute and relative errors.

Backward/gradient parity is required before using the operation for training,
but forward parity is the required gate for this frozen inference experiment.

### 7.3 Representation-invariance test

Reordering the four pairs, or reversing the two members of a pair while keeping
coordinates and query coefficients attached, must not change the scattered
output beyond numerical tolerance. This verifies that the implementation
represents unordered partitions rather than ordered head labels.

### 7.4 End-to-end reference parity

Run the complete model on fixed test tokens with ordinary \(H=4\) routing and
with arbitrary-group routing under \(P_0\). Logit and NLL differences must be
within the preregistered numerical tolerance.

---

## 8. Fixed paired evaluation data

Do not use the repository's advancing streaming-validation iterator unchanged.

Materialize two immutable, non-overlapping evaluation artifacts:

1. **discovery set**: used to evaluate all 105 partitions and select the
   reference, best, and worst candidates;
2. **confirmation set**: opened only after the discovery ranking is frozen and
   used to re-evaluate the selected candidates.

For each artifact, record:

- source dataset and revision;
- document/sample selection procedure;
- tokenizer and revision;
- sequence length and masking rules;
- total sequences and predicted tokens;
- serialized artifact path; and
- SHA-256 hash.

Every partition must see the exact same discovery tokens in the exact same
order. Evaluation uses `model.eval()`, no gradients, no dropout, and otherwise
identical inference settings.

### 8.1 Primary metric

Use token-weighted negative log-likelihood:

\[
\operatorname{NLL}(P)=
\frac{\sum_t \ell_t(P)}{\text{number of predicted non-masked tokens}}.
\]

Do not average already-averaged batch losses when batches can contain different
numbers of valid tokens.

Perplexity is secondary and derived only as:

\[
\operatorname{PPL}(P)=\exp(\operatorname{NLL}(P)).
\]

The primary comparison is the paired difference from the reference partition:

\[
\Delta\operatorname{NLL}(P)=
\operatorname{NLL}(P)-\operatorname{NLL}(P_0).
\]

---

## 9. Primary experiment procedure

1. Load and verify the trained 1B \(H=4\) checkpoint.
2. Load the hashed fixed discovery set.
3. Generate and canonicalize all 105 partitions.
4. Verify all correctness gates.
5. For each partition \(P\):
   - apply \(P\) consistently at all 73 MHAR routing sites;
   - run the same eager arbitrary-group operation;
   - evaluate the exact same discovery tokens;
   - record total NLL sum, valid-token count, token-weighted NLL, and PPL;
   - record \(R(P)\), \(D(P)\), and the canonical partition.
6. Freeze the complete discovery result table and its hash.
7. Select the reference, discovery-best, and discovery-worst partitions.
8. Evaluate those selected partitions on the untouched confirmation set.

Evaluation order should be deterministic. If hardware drift is a concern,
interleave periodic reruns of \(P_0\) as a systems-stability check; these reruns
must not replace the fixed paired-token design.

---

## 10. Required analysis

### 10.1 Complete partition distribution

Report:

- minimum, maximum, mean, median, and spread of NLL over all 105 partitions;
- every partition's \(\Delta\operatorname{NLL}\);
- the full ranked partition table; and
- the reference partition's rank among all 105.

### 10.2 Original-pair retention

For each available retention level \(R\in\{0,0.25,0.5,1\}\), report the number of
partitions and the distribution of \(\Delta\operatorname{NLL}\).

The 100% condition contains one partition, so do not present it as having
between-partition variance.

### 10.3 Coordinate distance

For each \(D\in\{1,1.5,2,2.5,3,3.5,4\}\), report partition count and the
distribution of \(\Delta\operatorname{NLL}\).

Report at least:

- Spearman association between \(D(P)\) and NLL;
- a clearly specified trend estimate; and
- the raw partition-level points so that non-monotonic structure is visible.

Do not reduce the analysis to distance-bin averages alone.

### 10.4 Best and worst structures

Report:

- discovery-best partition;
- discovery-worst partition;
- reference partition;
- their discovery NLLs and ranks; and
- their untouched confirmation-set NLLs.

The confirmation set determines whether the selected best/worst ordering
replicates after selection.

---

## 11. Interpretation rules

### Outcome A: strong monotonic locality effect

If larger \(D(P)\) systematically predicts larger NLL, conclude:

> MHAR performance is sensitive to residual-subspace grouping, and
> coordinate-local primitive subspaces are systematically more compatible under
> a shared depth softmax in this trained H=4 model.

This supports grouping quality as a relevant mechanism. It does not establish
human semantic purity or a universal coordinate law.

### Outcome B: partition differences without a locality trend

If NLL differs materially across partitions but does not track \(D(P)\),
conclude:

> Pair membership matters, but simple coordinate locality does not explain the
> good partitions.

This motivates searching for a hidden compatibility structure rather than
assuming nearby coordinates belong together.

### Outcome C: all partitions are effectively identical

Conclude:

> This experiment finds no practically detectable effect of pair membership at
> the tested 160-to-320-dimensional granularity for this checkpoint and
> evaluation distribution.

This weakens the motivation for learned four-head grouping in this setup, but
does not prove that grouping can never matter under other training conditions or
granularities.

### Outcome D: an alternative beats the contiguous reference

If an alternative partition beats \(P_0\) on discovery and confirms on the
untouched set, conclude:

> The trained contiguous H=4 partition is not uniquely optimal under the frozen
> intervention, providing direct motivation to search or learn partitions.

Do not claim a winning alternative before confirmation.

### General claim boundary

A loss effect establishes frozen-model partition dependence. It does not by
itself reveal the model's latent features or distinguish router mismatch from
broader backbone co-adaptation. Those mechanism questions belong to follow-up
experiments.

---

## 12. Required output artifacts

Save:

- experiment configuration;
- checkpoint/config/tokenizer identifiers and hashes;
- discovery and confirmation dataset manifests and hashes;
- source-control commit hash;
- all 105 canonical partitions;
- correctness-test results and numerical tolerances;
- raw per-partition total NLL and valid-token counts;
- the complete ranked result table;
- retention and coordinate-distance summaries;
- selected reference/best/worst confirmation results; and
- failure logs for any incomplete run.

Never silently resume with a different checkpoint, dataset artifact, partition
list, numerical mode, or software revision.

---

## 13. Follow-up experiments, not prerequisites

These are valuable only after the primary exhaustive experiment is complete.

### 13.1 H8-to-H4 compatibility experiment

Load a model trained with MHAR \(H=8\). Treat its eight learned 160-dimensional
routing slices as primitive blocks and merge them into four shared-softmax groups
under all 105 pairings.

This asks:

> Which independently trained H=8 subspaces remain compatible when forced to
> share four routers?

It is not the primary frozen-partition test because every candidate, including
the coordinate-local merge, changes the checkpoint's trained \(H=8\)
computation to \(H=4\). Label results explicitly as **H8-to-H4 compatibility**.

### 13.2 Routing-site localization

If the global intervention has an effect, localize it with preregistered
partitions at:

- attention-routing sites only;
- MLP-routing sites only;
- early, middle, and late depth bands; and
- final routing only.

### 13.3 Router-only recalibration

Freeze the backbone, recalibrate only routing-query parameters on a separate
training/calibration split, and evaluate on untouched data. This tests whether
the zero-shot partition penalty primarily reflects router mismatch.

### 13.4 Checkpoint evolution

Repeat selected partitions at early, middle, and final checkpoints to determine
when partition sensitivity emerges during training.

### 13.5 Learned grouping

Only after a reproducible partition effect is established should learned
projection, clustering, SAE-derived grouping, or differentiable grouping be
compared against fixed and parameter-matched controls.

---

## 14. Final decision criterion

Experiment 1 answers:

> At fixed \(H=4\) routing capacity, does the pairing of 160-dimensional
> residual subspaces into shared 320-dimensional depth-routing groups affect
> frozen-model language loss?

The result supports a learned-grouping research direction if either:

1. partition membership produces a reproducible NLL spread; or
2. a non-reference partition beats the contiguous partition and confirms on
   untouched data.

A locality trend determines whether simple coordinate locality explains that
effect. It is not required for grouping membership itself to matter.
