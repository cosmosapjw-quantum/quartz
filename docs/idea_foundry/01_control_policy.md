# 01 — Control, Risk, Policy Improvement, and Counterfactual Learning

This document specifies the foundry axes that decide **whether to continue,
what computation to buy, and whether the root policy may be changed**.  These
modules sit above the current `SearchPolicy` interface; they must preserve the
existing `observe → immutable cache publish → O(1) hot-path read` contract.

Covered axes: **A01, A02, A03, A04, A05, A09, A10, A24**.

## Shared repository design

### Current insertion points

- `src/mcts/policy/trait_def.rs`
  - `SearchSnapshot` is the periodic root snapshot.
  - `EdgeView` is the read-only edge surface.
  - `SearchPolicy::observe` is the heavy-compute boundary.
  - `SearchPolicy::score_adjustment` must stay O(1).
  - `SearchPolicy::should_halt` may only use fresh evidence.
- `src/mcts/policy/cache.rs`
  - `PolicyCachePublisher` is the approved lock-free publish mechanism.
  - arrays are indexed by `edge_pos`, never by game action identifiers.
- `quartz/phase15_ablation.py`
  - trace-generating search semantics and post-hoc readout semantics are kept
    separate through `search_relevant_signature`.
- `scripts/phase15_online_ablation.py`
  - resident root continuation is the first practical host for online control.
- `quartz/idea_foundry/contracts.py` and `src/mcts/foundry/types.rs`
  - new modules emit `MetaProposal`; they do not mutate PUCT independently.

### Missing fields

Do not overload the present `SearchSnapshot` fields.  Add a schema-bumped
companion (`FoundryRootExtras`) for entropy/margin slopes, H1 stability,
omission risk, revision count, evaluator identity, and runtime state.  Until
that lands, Python trace replay is authoritative for these features.

---

## A01 — Calibrated stop council

### Question

Can the controller estimate the probability that the currently committed root
action disagrees with a held-out deeper-search action, while avoiding the
known failure modes of any single H1 or `P_flip` threshold?

### Existing evidence and scope

H1 has substantially better calibration than the incumbent `P_flip` on the
current Stage-7 trace bank, but neither signal has earned a low-budget online
saving claim.  The council therefore starts as **trace-only calibration**, not
as a new live stop rule.  The uploaded reviews consistently call for a
multi-signal risk model rather than letting one miscalibrated statistic control
halt, widening, and refresh simultaneously.

### Inputs

```text
H1 argmax stability
P_flip
top-2 margin and slope
root entropy and slope
best/runner interval overlap
candidate omission bound
revision count
evaluator strength/version
budget and continuation mode
```

### Output

`MetaAction::Stop` only when:

1. the upper confidence bound on wrong-decision risk is below the configured
   risk allowance;
2. the cache/snapshot is fresh;
3. tactical and candidate-coverage guards pass; and
4. the arbiter sees no positive-LCB non-stop proposal.

### Skeletons

- Rust: `A01StopCouncil` in `src/mcts/foundry/control.rs`
- Python: `A01StopCouncil` in `quartz/idea_foundry/control.py`

### First tests

1. Position-grouped calibration split; no checkpoint from the same position
   group on both sides.
2. H1-only, `P_flip`-only, margin-only, logistic council, and isotonic council.
3. ECE, Brier, log loss, selective risk, and realized budget.
4. Stratify continuation and restart traces; never pool them.

### Promotion

Promote from `SHADOW` only if the council beats the best scalar baseline on a
held-out position group and provides a non-inferior matched-budget stop policy.

---

## A02 — Static-anchor regularized policy improvement

### Question

Can search use current value evidence without recursively mutating the network
prior and amplifying its own selection bias?

### Operator

For a frozen root anchor prior `pi0` and a chosen robust score `s_a`, compute a
temporary root policy

```text
q_tau(a) ∝ pi0(a) exp(s_a / tau).
```

The next checkpoint always recomputes from `pi0`; `q_tau` never becomes the new
anchor.  Candidate-tail mass must be explicitly preserved when only a visible
subset is transformed.

### Inputs

- frozen network prior per legal action;
- completed-backup mean and uncertainty bounds;
- visible/live candidate mask;
- KL or displacement allowance;
- optional safe-step search over temperature.

### Outputs

A post-hoc or cache-published `effective_prior` vector.  It is not a persistent
`refresh_prior` mutation.

### Skeletons

- Rust: `A02StaticAnchorRpo`
- Python: `A02StaticAnchorRPO`

### First tests

- identity when all action scores are equal;
- zero-prior floor and exact normalization;
- anchor immutability across checkpoints;
- explicit preservation of unmaterialized mass;
- compare static anchor, legacy cumulative refresh, Gumbel, and exact RPO on
  identical trace prefixes.

### Promotion

Requires lower oracle regret or KL-to-oracle without worse hidden-action recall
under matched evaluations and matched wall-clock.

---

## A03 — Uncertainty decomposition

### Question

Which part of root uncertainty comes from completed Monte-Carlo backups, neural
model epistemic error, within-search drift, and systematic evaluator bias?

### Contract

```text
r_total = r_mc + r_epistemic + r_drift + r_bias
```

is the default.  Root-sum-square is an ablation that may only be promoted after
covariance calibration.  Virtual visits and unfinished trajectories are never
used as completed evidence.

### Proposed data sources

- `r_mc`: Welford/anytime bounded-backup statistic;
- `r_epistemic`: shared trunk with several cheap value heads or calibrated
  checkpoint ensemble;
- `r_drift`: recent block mean against earlier block mean;
- `r_bias`: held-out deeper search, exact tactical labels, or learned residual
  envelope.

### Skeletons

- Rust: `A03UncertaintyDecomposition`
- Python: `A03UncertaintyDecomposition`

### First tests

Inject constant, pattern-dependent, heteroscedastic, and correlated errors in a
synthetic root.  Report interval coverage separately for every channel and for
the combined interval.  Do not call a Welford dispersion estimator “NN
posterior uncertainty.”

---

## A04 — KG/VOC computation allocator

### Question

Which visible arm or computation should receive the next measured-cost batch?

### Current disposition

The existing low-budget KG **stop** claim is closed for the tested Gomoku7
regime.  This does not close KG as an **allocation** feature.  The foundry
skeleton therefore emits `SAMPLE` or `CHALLENGE`; it does not halt.

### Target utility

```text
net_value(c) = LCB(expected root simple-regret reduction | c)
               - measured compute cost(c).
```

The initial approximation may use a top-m Gaussian/quadrature KG plus a learned
residual.  Telemetry must label the value as proxy until calibrated against
counterfactual branches.

### Skeletons

- Rust: `A04KgVocAllocator`
- Python: `A04KgVocAllocator`

### Tests

- monotonicity under increased challenger uncertainty;
- no false “stop” output;
- actual per-action latency rather than a universal cost constant;
- compare allocation gain with uniform, PUCT, UCB, Gumbel-SH, and forked-oracle
  best action.

---

## A05 — Forked counterfactual meta-action teacher

### Question

What would have happened if, from exactly the same root snapshot, the system
had stopped, sampled the incumbent, challenged the runner-up, widened, proved a
tactical move, or changed scheduler settings?

### Required engine work

The current resident-session interface can continue a root but cannot yet clone
one immutable search state into several deterministic branches.  The production
implementation should add:

```text
serialize/freeze root snapshot
fork deterministic RNG streams
resume branch with one explicit MetaAction
return action, final policy, realized visits/evals/ms, and failure status
```

State identity must include root hash, evaluator version, candidate-generation
epoch, TT identity policy, and cache schema.

### Labels

```text
y_c = decision_loss(STOP) - decision_loss(c)
      over realized cost(c)
```

Store raw numerator and cost vector; do not retain only a ratio.

### Skeletons

- Rust: `A05CounterfactualMetaTeacher` documents the executor seam.
- Python: `A05CounterfactualMetaTeacher` builds `CounterfactualLabel` records.

### First implementation slice

Only `STOP`, `SAMPLE(best, k)`, `SAMPLE(challenger, k)`, and `WIDEN(k)`.
Add proof and systems actions after deterministic replay is pinned.

---

## A09 — H3 entropy–margin change-point router

### Question

Do increasing root entropy and shrinking top-two margin identify a regime where
extra exploration/depth is valuable?

### Current disposition

The previous binary gate fired zero times at its default floors.  This axis
starts with continuous features and change-point labels, not another arbitrary
threshold.

### Skeletons

- Rust: `A09H3ChangePointRouter`
- Python: `A09H3ChangePointRouter`

### Tests

- threshold-free precision/recall against forked-VOC hard-state labels;
- percentile and logistic change scores;
- false-trigger cost on easy positions;
- compare `DEEPEN`, `WIDEN`, and no-op branches instead of presuming a burst is
  the right response.

---

## A10 — Prior-refresh specialist

### Question

Is cumulative prior mutation useful in a narrowly defined weak-evaluator or
out-of-distribution regime even though it is unsafe as the default?

### Guarded role

`DORMANT` conditional expert.  It may run only in a preregistered slice with:

- frozen baseline and identical candidate coverage;
- explicit evaluator quality/OOD signal;
- bounded refresh displacement;
- tail-mass preservation;
- rollback to anchor after every root.

### Skeletons

- Rust: `A10PriorRefreshSpecialist`
- Python: `A10PriorRefreshSpecialist`

A positive narrow result must not change the no-refresh default.

---

## A24 — Learned state-dependent budget gate

### Question

Can a small model select one of a discrete set of planning budgets on top of a
frozen planner and transfer across checkpoint strength or games?

### Inputs

Use only cheap features available before buying the next budget tier: direct
policy entropy/margin, network uncertainty, current root statistics, elapsed
cost, and tactical flags.  Do not give the gate oracle/deeper-search features at
inference.

### Outputs

A `SAMPLE` continuation amount or `STOP`, represented as an explicit proposal.
The model is not allowed to change PUCT and budget simultaneously in the first
ablation.

### Skeletons

- Rust: `A24LearnedBudgetGate`
- Python: `A24LearnedBudgetGate`

### Evaluation

- fixed-budget grid;
- hand-built entropy/margin gate;
- A01 calibrated stop council;
- learned budget model;
- cross-game and cross-checkpoint holdout;
- decision quality per wall-clock, not only average visits.

---

## Group integration order

```text
Trace features (A03, A09)
    → risk calibration (A01)
    → static policy readout (A02)
    → counterfactual labels (A05)
    → allocation model (A04)
    → discrete budget gate (A24)
    → conditional legacy expert audit (A10)
```

No combined live controller is enabled until each proposal has a calibrated
cost and an independently measured causal label.
