# 04 — Readout, Evaluator Architectures, CPU Deployment, and Training Control

Covered axes: **A17, A18, A19, A20, A23, A24**.  A24 is described in the
control document; this document defines its training/evaluator dependencies.

The central attribution rule is simple: **freeze the search controller when an
evaluator is compared, and freeze the evaluator when a controller is compared.**

---

## A17 — B13 finite-N curvature readout

### Question

Can a finite-sample curvature correction improve the full root policy target
without changing the selected action?

### Current evidence

The current trained-network replay result supports a decision-neutral
KL-to-oracle improvement.  It does not support a play-strength or online score
claim.

### Three roles must remain separate

1. post-hoc final policy readout;
2. live selection shaping;
3. self-play training target.

The first is `MECHANISM_VALID`; the other two are new experiments.

### Repository reuse

Use the pinned `quartz/phase15_one_loop.py` and B13 Phase-15 operator for
claim-bearing replay.  The foundry class is only a common proposal/readout
interface.

### Skeletons

- Rust: `A17B13CurvatureReadout`
- Python: `A17B13CurvatureReadout`

### Tests

- exact decision neutrality when claimed;
- policy normalization and zero-count behavior;
- oracle KL, entropy, rare-action mass, and gradient effects;
- separate training runs for target shaping.

---

## A18 — Diffusion-regularized deterministic evaluator

### Question

Does denoising auxiliary learning improve a direct policy-value evaluator while
preserving deterministic, low-latency MCTS inference?

### Architecture contract

```text
clean board → shared deterministic encoder → policy/value heads
shared latent or masked board → corruption → denoising auxiliary loss
MCTS inference → clean direct policy/value path only
```

No random VAE sample, random timestep, denoising loop, reconstruction, or
state-generation call is allowed in leaf evaluation.

### Variants

1. parameter/FLOP-matched plain U-Net/local-global evaluator;
2. latent Gaussian denoising auxiliary;
3. masked categorical board denoising;
4. policy-map denoising auxiliary;
5. optional offline policy/value-conditioned position generator, kept outside
   the evaluator benchmark.

### Skeletons

- Rust: `A18DiffusionRegularizedEvaluator` is a deployment/config contract.
- Python: `A18DiffusionRegularizedEvaluator` defines loss/inference contracts.

### Required corrections in real model code

- preserve 15×15 spatial alignment via pad/crop or explicit interpolation;
- spatial policy head, not a giant flattened reconstruction MLP;
- global/multiscale value pooling;
- game-level train/validation split;
- replay buffer across iterations;
- single batched inference server for multiple self-play workers.

### Evaluation

Policy target KL/top-k, value calibration, batch-1 and search-batch latency,
positions/s, MCTS nodes/s, fixed-evaluation Elo, and fixed-wall-clock Elo.

---

## A19 — RW-ResT-AZ-40/144-Lite evaluator

### Question

Can a sparse randomly wired local backbone plus a small number of true global
mixing blocks improve board-size transfer or strength-per-evaluation?

### Proposed architecture

```text
40 random nodes, 144 channels
2 cells × 20 nodes
one-conv pre-activation residual node operation
WS-dominant degree-capped DAG with mandatory chain
soft gates during training, dataset-average static pruning for deployment
2 axial/global-token attention blocks
spatial policy conv + separate pass logit
global mean + max (+ token) value head
```

This is intentionally lighter than forty two-convolution BasicBlocks.  Random
wiring connects feature blocks, not distant board cells; global attention is
therefore a separate, measured component.

### Skeletons

- Rust: `A19RwRestLiteEvaluator` captures runtime/deployment metadata.
- Python: `A19RwRestLiteEvaluator` captures the architecture experiment.

### Experiment design

- screen several graph seeds on a fixed replay proxy;
- full self-play only for shortlisted graphs;
- ResNet-FCN, static WS Lite, learnable static weights, soft routing,
  attention-only, combined, and heavier upper-bound model;
- same parameter/FLOP, same eval count, and same wall-clock views;
- held-out board-size test separate from mixed-size training.

---

## A20 — Regret/instability state archive

### Question

Can training compute be redirected toward positions where extra search or a
better evaluator would have reduced decision regret?

### Archive record

```text
position identity and serialized state reference
checkpoint/evaluator identity
reason: regret, H1 instability, omission, late flip, hidden discovery,
        prior-search disagreement, rare regime, tactical failure
priority and importance weight
search trace and oracle/reference provenance
deduplication group
```

### Operations

`ARCHIVE_STATE` is emitted during trace/search collection.  Training later
chooses restart, reanalysis, extra search, or ordinary replay.  These are
inter-position control actions and must be budgeted separately from per-move
search.

### Skeletons

- Rust: `A20RegretStateArchive` emits a compact priority event.
- Python: `ArchiveRecord` and `A20RegretStateArchive` define the durable record.

### Tests

- deduplication by state/position group;
- bounded archive size and deterministic eviction;
- mixture with uniform replay;
- importance-weight/ESS reporting;
- fixed total NN evaluations and learner updates;
- standard self-play, uniform restart, Go-Exploit-like archive, and regret
  archive comparators.

---

## A23 — CPU incremental pattern student

### Question

Can a game-specialized incremental evaluator beat a generic small CNN at
fixed CPU move time while remaining faithful to the neural teacher?

### Architecture lane

For Gomoku/Renju, the initial target is a Rapfi/NNUE-like line-pattern or codebook
student with:

- local pattern indexing;
- make/unmake incremental updates;
- int8/int16 accumulation;
- SIMD-friendly contiguous tables;
- policy/value distillation;
- explicit legality/tactical features.

A generic quantized ConvNet remains a separate comparator.  Do not call a
full-board small ResNet “incremental.”

### Skeletons

- Rust: `A23CpuIncrementalPatternStudent` defines inference/update contracts.
- Python: `A23CpuIncrementalPatternStudent` defines distillation/deployment
  metadata.

### Tests

- full recomputation equals incremental update after random make/unmake traces;
- quantization error and teacher calibration;
- batch-1 latency, cache update latency, and nodes/s;
- fixed-time PUCT/PVS comparison using the same evaluator;
- external CPU engine/tactical suite where licensing and reproducibility allow.

---

## A24 training data for the learned budget gate

A24 uses counterfactual or budget-ladder labels:

```text
state features at budget b
chosen action at b
deep-reference action/value
incremental time/evaluations to next budget
realized regret reduction
```

Group train/test splits by game state and self-play game.  A model that sees a
deeper checkpoint of the same root in training and an earlier checkpoint in
test is leaking the answer.

---

## Representation campaign order

```text
A17 trace-only readout
    → deterministic evaluator baseline cleanup
    → A18 auxiliary-loss ablations
    → A19 graph-seed proxy and shortlisted self-play
    → A23 CPU student distillation/deployment
    → A20 training archive
    → A24 budget gate on frozen planner/evaluator
```
