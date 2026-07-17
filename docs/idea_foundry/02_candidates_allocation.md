# 02 — Candidate Coverage, Widening, Proof, Geometry, and Allocation

This document covers axes that prevent a root controller from becoming very
confident about the wrong **visible** action set.

Covered axes: **A06, A07, A08, A12, A25, A26**, with A04 as the common
allocation consumer.

## Shared distinction: omission versus ranking

Every candidate experiment must report two errors separately:

```text
R_omit = P(best reference action is not visible)
R_rank = P(best visible action is misranked | reference best is visible).
```

- `WIDEN` targets omission risk.
- `SAMPLE`/`CHALLENGE` target ranking risk.
- `PROVE` targets tactical exactness.
- `STOP` is illegal while a calibrated omission guard fails.

This distinction is reflected in `RootObservation.candidate_omission_bound` and
in the foundry `MetaAction` enum.

---

## A06 — Gumbel candidate selection and Sequential Halving

### Question

Can a low-budget search cover more plausible root actions and allocate a fixed
budget among them more efficiently than ordinary PUCT visits?

### Repository reuse

The repo already contains `src/mcts/policy/gumbel_sh.rs`.  The foundry axis
should wrap and instrument that implementation instead of creating a second
Gumbel sampler.

### Root contract

```text
candidate set = Gumbel top-m without replacement
                ∪ high-uncertainty reserve
                ∪ tactical sentinels
round allocation = resumable Sequential Halving bracket
```

Training/evaluation randomness must be controlled explicitly.  Deterministic
evaluation uses pinned Gumbel samples or a deterministic candidate mode.

### Skeletons

- Rust: `A06GumbelSequentialHalving`
- Python: `A06GumbelSequentialHalving`

### First tests

- no duplicate candidate IDs;
- reproducibility under pinned seed;
- exact budget conservation across rounds;
- low-prior hidden-best fixture;
- compare top-1 and top-k recall against prior top-k and ordinary PUCT.

---

## A07 — Residual partition-mass widening

### Question

How much regularized posterior mass may still lie outside the live candidate
set, and is that mass large enough to justify `WIDEN`?

### Core calculation

For a static anchor prior and score `s_a`, define

```text
Z_L   = Σ_{a in live} pi0(a) exp(s_a / tau)
Z_out = Σ_{a outside} pi0(a) exp(s_a / tau).
```

If only upper scores are available outside the live set, use
`Z_out_upper`.  The conditional truncation error obeys

```text
TV(full posterior, live-conditional posterior) = Z_out / (Z_L + Z_out),
```

and is bounded by replacing `Z_out` with `Z_out_upper`.

### Hard part

The formula is cheap; calibrated upper scores for unmaterialized actions are
not.  Initial sources:

- direct network prior plus value-head uncertainty;
- cheap tactical features;
- action-embedding nearest-neighbour bounds;
- global bounded-value fallback, reported as loose.

### Skeletons

- Rust: `A07ResidualEvidenceWidening`
- Python: `A07ResidualEvidenceWidening`

### Tests

- exact enumerated roots where all outside scores are known;
- bound coverage and tightness;
- zero/live/full edge cases;
- compare standard progressive widening, Gumbel-SH, and residual-mass widening
  at identical evaluation and wall-clock budgets.

### Non-claim

This is not a claim to run the full nested-sampling evidence algorithm inside
MCTS.  It borrows residual-mass reasoning for a finite-action widening error.

---

## A08 — Tactical sentinel and proof backend

### Question

Can cheap exact or shallow tactical checks protect forced actions from a
miscalibrated neural prior or statistical stop rule?

### Architecture

The generic foundry contract emits `PROVE(edge, budget)`.  Game-specific logic
stays behind a trait/adapter and does not contaminate the generic controller.

Potential Gomoku/Renju operations:

```text
immediate win
mandatory immediate block
open-four / double-threat candidates
forbidden-move legality
bounded threat-space search
```

Chess and Go require different adapters.  A sentinel flag may force inclusion;
it may only force final selection when the underlying routine returns an exact
proof under the game rules.

### Skeletons

- Rust: `A08TacticalProofBackend`
- Python: `A08TacticalProofBackend`

### Tests

- solved tactical corpus;
- false-positive and false-negative rates;
- proof result dominates score only when `proven=true`;
- proof cost included in the proposal cost vector;
- generic controller tests use mock proof backends.

---

## A12 — JSD-preconditioned locally balanced root sampler

### Question

Can sibling successor policies define a useful local geometry for moving
search effort among root actions without treating JSD itself as an energy
bonus?

### Required separation

```text
target density μ(a) ← anchor prior and Q/value evidence
local geometry K_ab ← sibling policy-value JSD
```

JSD is not directly subtracted from PUCT.  For sibling action successors
`x_a`, create compatible policy-value representations `rho_a`, then

```text
d_ab = sqrt(JSD(rho_a, rho_b))
K_ab = exp(-d_ab² / (2 ell²)) on a symmetric neighbour graph
R_ab = K_ab * g(μ(b)/μ(a))
```

with a balancing function such as square-root or Barker.  At a frozen snapshot
this is a locally balanced discrete transition.  During MCTS it is an adaptive,
non-stationary sampler; do not claim exact stationary MCMC.

### Support handling

Compare:

1. common-legal-support policy renormalization;
2. fixed action-space unmasked logits plus explicit legality channel.

The first is the safe default.

### Skeletons

- Rust: `A12JsdLocallyBalancedSampler`
- Python: `A12JsdLocallyBalancedSampler`

### First implementation

Root-only trace replay with cached child network outputs.  Never reevaluate all
siblings in every selection call.

### Baselines

prior sampling, Boltzmann, Gumbel, board Hamming kernel, latent cosine kernel,
and PTSA-style abstraction.

---

## A25 — MENTS / soft-backup ablation

### Question

Does a maximum-entropy/soft backup improve sample efficiency in selected
uncertain regimes without changing the original game objective or blurring
forced tactics?

### Scope

`DORMANT` and opt-in.  Apply root-only or shallow-depth soft backup first.
Temperature must decay or be selected by a target-entropy contract.

### Skeletons

- Rust: `A25MentsSoftBackup`
- Python: `A25MentsSoftBackup`

### Tests

- temperature → 0 recovers hard/max backup;
- value sign and player-to-move conventions;
- tactical suite degradation;
- compare against existing `src/mcts/policy/ments.rs` rather than duplicating
  its math.

### Non-claim

A maximum-entropy optimum can differ from the reward-optimal action.  MENTS is
an experimental allocator/backup, not a universal objective replacement.

---

## A26 — Exact nested-contour validation lab

### Question

Can a precisely defined small discrete trajectory model maintain separated
modes and estimate its target mass accurately enough to justify a later
approximate live-set backend?

### Isolation rule

This is an `ANALYSIS_ONLY` backend.  It does not control the production MCTS.
It needs an explicit:

- trajectory space;
- reference prior/proposal;
- score/likelihood;
- constrained resampling kernel;
- enumerated or independently computed ground truth.

### Skeletons

- Rust: `A26NestedContourExactLab` (contract only)
- Python: `A26NestedContourExactLab`

### Test bank

- small finite action/trajectory spaces;
- disconnected modes with known masses;
- plateaus;
- smallest-mode survival;
- evidence/posterior error;
- compare SMC and stratified enumeration.

Only after this lab is sound may A11 borrow a live-set heuristic.  A11 must not
inherit an “exact evidence” claim.

---

## Candidate campaign matrix

Use a shared bank with at least:

```text
prior correct
prior badly wrong
best action in low-prior tail
near tie
diffuse many-near-optimal
forced win
forced block
multimodal value
high branching
unmaterialized best
correlated/transposed subtrees
```

Report discovery probability, discovery latency, omission/ranking regret,
false-stop rate, NN evaluations, CPU/GPU time, and candidate resurrection rate.
