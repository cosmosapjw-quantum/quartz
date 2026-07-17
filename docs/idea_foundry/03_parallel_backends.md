# 03 — Parallel Execution, Particle Search, Path Diversity, and Graph Sharing

This document covers axes whose main effect is **how computation is executed or
shared**, rather than which root action currently looks best.

Covered axes: **A11, A13, A14, A15, A16**.

## Shared runtime principles

1. Completed backups are evidence; pending work is scheduling state.
2. Simulation count, NN evaluation count, and wall-clock are distinct budgets.
3. Throughput gains are not decision-quality gains until paired search results
   are measured.
4. CPU, CUDA, and ROCm service curves are distinct artifacts.
5. Fixed-budget controller comparisons pin threads, batch policy, and runtime
   contract unless the scheduler itself is the tested factor.

The existing repo already exposes phase timings, virtual-loss telemetry,
root-continuation sessions, TT contention counters, and a service-curve lab.
The foundry should extend those paths instead of inventing separate profilers.

---

## A11 — Dynamic live-set particle root search

### Question

Can independent or weakly coupled particle groups preserve several root modes
and dynamically reallocate parallel search to actions with high decision value?

### States

```text
ACTIVE       normal allocation
HIBERNATING  small resurrection quota
FROZEN       no current quota, but restorable on a regime change
PROVEN       exact tactical result
```

### Statistics

Per candidate/group:

- independent-group return summaries;
- robust mean/median-of-means;
- probability-of-best proxy;
- uncertainty and cost;
- unique trajectory and duplicate-evaluation rates;
- mode/cluster identity.

### Actions

`SAMPLE`, `RESAMPLE_MODE`, `WIDEN`, and `STOP` proposals.  Permanent pruning is
not the first implementation.

### Skeletons

- Rust: `A11DynamicLiveSetParticles`
- Python: `A11DynamicLiveSetParticles`

### Engine design

Use a root-level backend first.  Each particle trajectory still follows legal
game transitions and uses the current evaluator.  Keep group statistics
separate from shared state/evaluation caches.  Integrate deeper-node particles
only after root attribution is clear.

### Comparators

Gumbel-SH, current tree-parallel MCTS, WU-UCT-style pending correction, SMC
policy improvement, and PMCTS where reproducible.

---

## A13 — Pending-flow / WU-UCT correction

### Question

Can explicitly tracking unobserved in-flight work reduce redundant selection or
bias without the pessimism of a fixed virtual value?

### Required state

```text
N_observed(edge)
N_pending(edge)
virtual_value(edge) only if the tested mode uses it
pending age / queue state
```

Pending counts shape selection but never enter confidence intervals as if they
were completed outcomes.

### Skeletons

- Rust: `A13PendingFlowWuUct`
- Python: `A13PendingFlowWuUct`

### Repository seam

Extend the existing `parallel::VlMode`/selection telemetry and the
`pending_flow_lab.py` bridge.  Do not introduce an unrelated parallel search
loop.

### Tests

- deterministic delayed evaluator;
- pending reservation accounting under cancellation/failure;
- fixed virtual loss, current adaptive split VL, pending-count-only, and hybrid;
- quality, duplicate rate, average virtual pessimism, p99 selection latency.

---

## A14 — Whole-path semantic LSH diversity

### Question

After edge-level contention is controlled, are parallel workers still spending
large batches on semantically similar root-to-leaf paths?

### Distinct claim

This is not the rejected claim that MinHash should beat adaptive virtual loss
on immediate edge duplicates.  It targets different edges or transposed states
whose path shingles remain highly overlapping.

### Trace representation

```text
shingle = (state hash, action id, depth bucket, optional tactical motif)
path signature = MinHash or another fixed-size sketch
similarity = estimated Jaccard / calibrated semantic overlap
```

Online repulsion is gated by high thread/inflight count, low residual edge
duplication, and high measured semantic overlap.

### Skeletons

- Rust: `A14SemanticPathLsh`
- Python: `A14SemanticPathLsh`

### First stage

Shadow telemetry only.  Test whether overlap predicts duplicated NN leaf
states, low batch diversity, or lower counterfactual gain.  Promote online only
if it adds information beyond edge duplicate rate and TT hit rate.

---

## A15 — Evaluator service-curve scheduler

### Question

Which batch size and global inflight credit maximize useful evaluator throughput
under the current device, model, and search workload?

### Existing base

The repo already has `quartz/experiments/service_curve.py`,
`scripts/service_curve_lab.py`, tests, and hardware-aware manifests.  The
foundry skeleton is a decision adapter over those measured curves.

### Actions

`SET_BATCH`, `SET_INFLIGHT`, and later `SET_THREADS`.  Scheduler changes are
made at safe epochs, not during arbitrary selection steps.

### Skeletons

- Rust: `A15ServiceCurveScheduler`
- Python: `A15ServiceCurveScheduler`

### Measurements

- items/s and ms/batch;
- queue wait and batch occupancy;
- GPU seconds/game and optional power proxy;
- search decision quality per wall-clock;
- device/model/checkpoint identity.

The current service-curve lab is quality-free.  Its “lane alive” result only
authorizes an online quality experiment.

---

## A16 — Monte-Carlo graph/state sharing

### Question

Which information should be shared when transpositions make the search a DAG,
and which statistics must remain parent-edge specific?

### State split

```text
Shared state node:
  game state identity
  terminal result
  neural policy/value/uncertainty cache
  child-state identities

Parent edge:
  N, W, Q, prior
  pending/virtual counts
  root-relative search contribution
```

Graph-wide visit/value merging is a separate MCGS experimental mode.  The safe
baseline shares state identity/evaluation only and backs up each parent edge
independently.

### Skeletons

- Rust: `A16MonteCarloGraphSharing`
- Python: `A16MonteCarloGraphSharing`

### Tests

- two legal paths reaching the same state;
- exact state hash includes side to move, rules, repetition/ko where relevant;
- no parent-edge statistic contamination in cache-only mode;
- graph-sharing comparator reports memory, eval reuse, contention, and strength;
- cycle/repetition safeguards for games that are not naturally acyclic.

---

## Recommended backend sequence

```text
A13 pending accounting correctness
    → A15 device service curve
    → A14 semantic shadow telemetry
    → A16 cache-only graph sharing
    → A11 root particle backend
    → optional online A14 and graph-stat sharing
```

All backend experiments run under a common runtime contract and produce both a
mechanism report and a paired decision-quality report.
