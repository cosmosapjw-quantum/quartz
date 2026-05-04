# QUARTZ Theory and Architecture

## Overview

QUARTZ (Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable)
is a search controller for Monte Carlo Tree Search (MCTS) that adapts its
behavior based on observable search statistics. It sits on top of standard
AlphaZero-style PUCT search and modulates three aspects:

1. **Score shaping** — how root actions are ranked
2. **Adaptive stopping** — when to stop searching
3. **Parallel thread distribution** — how to manage virtual loss

The runtime signals are derived from search/runtime state, but the controller
also depends on explicit per-run search hyperparameters such as `sigma_0`,
`min_visits`, `check_interval`, `c_puct`, and `hbar_penalty_cap`. Those values
are not learned online, but they are legitimate sweep and tuning targets.

## Current empirical status

This document is an architecture note, not a claim that one frozen controller
formula has already won across every game.

As of the current repository-local Gomoku7 studies:

- short-budget training and frozen-checkpoint confirmatory arenas still favor
  no-refresh legacy-family variants over refresh-enabled anchors
- the wider Optuna sweep also stayed in the no-refresh basin
- the strongest current candidate is a tuned no-refresh legacy-family variant
  with `root_only_shaping=true` and retuned search constants

The honest reading is: QUARTZ is a family of state-driven controllers plus
explicit search hyperparameters, not a hyperparameter-free law.

## Legacy design lineage

The repository uses the word "legacy" in two slightly different senses. This
section spells them out because current ablations still depend on both.

### 1. Original low-level legacy mode: `PenaltyMode::Legacy`

This is the earliest QUARTZ-style root-shaping path preserved in the Rust
selector. In code terms it is the default `QuartzConfig` shape in
`src/mcts/quartz.rs`:

- `penalty_mode = Legacy`
- `enable_one_loop = true`
- `prior_refresh_rate = 0.0`
- `root_only_shaping = true`
- `enable_fisher_puct = false`

The resulting root score is conceptually:

    score(a) = standard_PUCT(a) + one_loop_bonus(stats) - min(hbar_eff, cap) / N_a

where the negative term is `one_loop_visit_penalty(...)` and the positive term
is the off-diagonal `B_1loop` bonus when available.

This original design tried to keep the intervention minimal:

- leave the interior tree close to standard PUCT
- shape only the root ranking
- use `σ_Q / σ_0` as the uncertainty scale
- disable prior refresh by default

### 2. Historical training heuristic: `GatedRefreshLegacy`

Later training experiments introduced a second "legacy" branch that is now
what most recent Gomoku7 controller ablations mean by "legacy family."

In `src/mcts/select.rs`, `GatedRefreshLegacy` does:

- penalty = `effective_penalty_v2(N_a, O_a, hbar_penalty_cap)`
- compute a refresh gate from `P_flip`
- set `rho_t = rho_max * min(P_flip / flip_thresh, 1)`
- mix the original prior with a Q-based signal using `tau`. Historically
  this was a hardcoded `tau = 0.5`; it now follows `config.prior_refresh_temp`
  and falls back to the legacy `0.5` only when that knob is explicitly
  zeroed (`< 0.01`). `PenaltyMode::PFlipMixture`'s Q-refresh branch honors
  the same field, so `prior_refresh_temp` sweeps against either of those
  two modes now report real effects instead of silent nulls.

Conceptually:

    score(a) = standard_PUCT(a; blended_prior) - nu / (1 + N_a + O_a)

with:

    blended_prior ∝ prior^(1-rho_t) · exp(rho_t * Q_a / tau)

This branch is "legacy" because it preserves the historical training-time
heuristic that existed before the current doc-aligned `GatedRefresh` branch
was added. It is not the same thing as the older `PenaltyMode::Legacy`.

### 3. Historical shallow-tree blend

There is one more historical behavior worth documenting. When
`root_only_shaping = false`, selection does not turn on full QUARTZ everywhere.
Instead, the code uses a shallow blend for `depth <= 3`:

- compute standard PUCT
- compute full controller-shaped score
- interpolate between them with weight `1 / (1 + depth)`

This path exists to preserve older experiments that wanted some non-root
controller influence without fully replacing the interior-tree policy.

### How this maps to current ablation names

- `A1_legacy_base` and `A2_legacy_krefresh` refer to the
  `GatedRefreshLegacy` family, not the older `PenaltyMode::Legacy`.
- The original `PenaltyMode::Legacy` path still exists in the codebase and is
  part of the design lineage, but it is no longer the main label used by the
  recent training-level controller shortlists.

## 1. Search Policy: Boltzmann on Discrete Exponential Family

### Standard PUCT

In AlphaZero, the selection rule at each node is:

    a* = argmax_a [ Q(a) + c_puct · π(a) · √N_parent / (1 + N_a) ]

where Q(a) is the mean value, π(a) is the prior from the neural network,
and N_a is the visit count.

### QUARTZ Modification (Root Only)

QUARTZ adds controller-side score shaping at the root node (depth = 0):

    score(a) = PUCT(a) − penalty(a) + optional refresh/shaping terms

The exact penalty term is mode-dependent in the current Rust implementation.
The active codebase contains multiple controller variants
(`GatedRefreshLegacy`, `GatedRefresh`, and related helper paths), so this
document treats the root penalty conceptually:

- it is visit-dependent
- it is modulated by Q-uncertainty statistics such as `σ_Q` and `σ_0`
- it tries to reduce premature lock-in on one root action while the root is
  still uncertain

**Scope**: Score shaping applies at root depth only. In current QUARTZ this
includes the penalty, prior-refresh, and related controller-side shaping
terms. Interior tree nodes use standard PUCT. This is by design — root action
ranking is where controller guidance has the highest impact-to-cost ratio.

## 2. Uncertainty Estimation: σ_Q and Related Statistics

### Q-value Standard Deviation (σ_Q)

σ_Q is the standard deviation of Q-values across root children:

    σ_Q = std({Q(a) : a ∈ children(root)})

This is the primary uncertainty signal. When σ_Q is high, the evaluator
disagrees strongly about different actions, suggesting more search is
needed. When σ_Q is low, the evaluator is confident.

### Normalized Uncertainty (hbar_eff)

    hbar_eff = σ_Q / σ_0

where σ_0 is a reference scale (configurable per run/game). This
normalizes the uncertainty to a dimensionless ratio. The name is a legacy
from an earlier QFT-motivated framing; it functions as a normalized
Q-uncertainty ratio.

### Prior-Q Divergence

    prior_q_divergence = KL(π_prior || π_visits)

Measures how much the visit distribution has diverged from the neural
network prior. High divergence suggests search has found information
the prior missed.

## 3. Risk Measure: P_flip

P_flip is the probability that the current best move would change if
search continued. It serves as a risk measure for adaptive stopping.

### Definition

Given the root node with children sorted by visit count, let a₁ be the
most-visited action and a₂ the second-most-visited. Define:

    P_flip = P(N_{a₂} > N_{a₁} after k more iterations)

### Computation

QUARTZ computes P_flip via two methods:

1. **Gaussian approximation**: Models visit counts as normally distributed
   with variance proportional to visits. Fast but inaccurate at low counts.

2. **Saddlepoint approximation**: More accurate tail probability estimate
   using the cumulant generating function. Preferred when available.

The final P_flip is the minimum of both estimates (conservative).

### Convergence Properties

- With a strong evaluator (NN loss < ~1.0), P_flip converges toward 0 as
  search progresses. This enables adaptive stopping.
- With a weak evaluator (loss > ~1.5), P_flip stays at 0.4–0.5 regardless
  of budget. Adaptive stopping correctly does not trigger.
- The stopping threshold is a configured threshold plus several code-level
  guards. It is not a learned quantity.

## 4. Adaptive Stopping: VOC and Halt Modes

### Value of Computation (VOC)

The VOC framework asks: "Is the expected gain from one more search
iteration worth the computational cost?"

    VOC = expected_improvement − cost

When VOC < 0, search should stop.

### Two distinct VOC computations in this repo

The word "VOC" appears in two unrelated places. They share statistics
but live in different decision layers:

1. **halt-VOC** (in `src/mcts/quartz.rs`) — used by the `HaltMode::VOC`
   stopping rule. Computes three accounting channels (`voc_focus`,
   `voc_expand`, `voc_merge`) and aggregates them as
   `voc_total = max(voc_focus, voc_expand, voc_merge)`. **Only
   `voc_total` is consumed by the halt decision.** The individual
   channels are recorded in telemetry but do not route the policy.
   The argmax channel at each halt check is now emitted as
   `voc_argmax_channel` so attribution work can falsify single-channel
   dominance.
2. **PW-VOC** (in `src/mcts/gvoc.rs`) — a progressive-widening width
   scheduler. `GvocState::update()` reads `voc_total` (not the channels)
   and expands or contracts `n_visible` against simple thresholds. This
   is a heuristic PW knob, not an optimal-stopping controller.

When this document or the README says "VOC", it almost always means
halt-VOC. PW-VOC is documented per-game where it is wired in
(`chess.rs`, `gomoku15.rs`, `go.rs`) and is independent of the
penalty-mode dispatch.

### Halt Modes

| Mode | Description |
|------|-------------|
| Fixed | Always use full budget |
| SimpleThreshold | Stop when P_flip < threshold |
| VOC | Full VOC computation with cost model |
| ConfAdaptive | Confidence-adaptive threshold |

For attribution-grade ablation studies, the `controller_axes` and
`controller_factorial` presets pin `HaltMode::Fixed` so penalty
changes cannot silently shift the effective compute budget. To
study `HaltMode` itself at fixed NN/eval/visit-cap, use the
`halt_attribution` preset (see `docs/ABLATION_GUIDE.md`).

### Production fixed-budget execution

`HaltMode::Fixed` and plain `FixedIterations` are implemented as exact
work-budget paths in production search loops. Parallel workers reserve visit
tickets from a shared counter instead of repeatedly polling root visits and
racing past the budget. This is an execution optimization only:

- root selection still uses the same PUCT / QUARTZ scoring law
- QUARTZ root statistics are still refreshed on `check_interval`
- halt telemetry is recorded at the final budget boundary
- adaptive halt modes still use the controller's normal `should_stop` path
- parallel fixed-budget searches reserve tickets in small chunks when
  `n_threads > 1`, but each worker executes only tickets strictly below the
  requested limit; internal counter overshoot is not exposed as extra visits

The practical effect is lower CPU coordination overhead and exact same-budget
accounting for fixed-budget ablations.

### Automatic Thread Selection

Explicit thread counts remain the ablation surface. A study that compares
controller modes should still pin `n_threads` so the controller cannot gain or
lose compute through the scheduler.

For production throughput, the engine also exposes opt-in automatic thread
selection:

- `MctsEngine::run_auto(controller, AutoThreadPolicy::throughput())`
- `MctsEngine::run_quartz_auto(controller, AutoThreadPolicy::throughput())`
- `AutoThreadPolicy::quality()` for lower duplicate-selection pressure

The policy is game-agnostic and device-agnostic. It reads:

- host `available_parallelism()`
- remaining fixed-visit budget
- root legal-move count
- whether progressive widening is enabled
- whether the game supports reusable select scratch

It does **not** change PUCT, QUARTZ penalty, refresh, halt, backup, or
evaluator semantics. It only selects the number of worker threads before
entering the existing serial or parallel search loop.

The single-position Rust NN server path exposes this as an opt-in execution
policy:

```json
{"cmd":"search_nn","game":"gomoku15","iters":800,"n_threads":"auto"}
{"cmd":"search_nn","game":"gomoku15","iters":800,"thread_policy":"quality","thread_cap":12}
```

Server responses preserve the old effective `n_threads` field and add
`requested_threads`, `effective_threads`, `thread_policy`, and
`auto_thread_reason` in both the top-level result and `search_manifest`.
Multi-position/session paths intentionally continue to require explicit
`n_threads` because their worker count, per-job inflight limit, and batched NN
queue interact; automatic scheduling there needs a separate fairness contract.

`throughput` mode may use the full host cap when the budget can absorb worker
overhead. `quality` mode caps low-branching and small-budget searches more
aggressively because high raw NPS can coincide with high virtual-loss duplicate
selection. Profile harness support is available via:

```bash
QUARTZ_PROFILE_THREADS=auto cargo test --release --locked \
  profile_mcts_parallel_all_supported_games -- --ignored --nocapture

QUARTZ_PROFILE_THREADS=quality cargo test --release --locked \
  profile_mcts_parallel_all_supported_games -- --ignored --nocapture
```

### Stop Reasons (recorded in telemetry)

| Reason | Meaning |
|--------|---------|
| Budget | Hit iteration limit |
| VOCNegative | VOC went negative |
| PFlipConverged | P_flip below threshold |
| ConfidenceHigh | Confidence threshold met |
| Unknown | Default / unclassified |

Each per-position halt check now also records `voc_argmax_channel ∈
{focus, expand, merge}` so that artifacts can show whether a "VOC
halt" was driven by FOCUS, EXPAND, or MERGE. Channel histograms are
aggregated per game in the replay search summary.

## 5. Prior Refresh

Standard MCTS uses the neural network prior once at node expansion. QUARTZ
can refresh the prior during search based on accumulated Q-value information.

### Modes

- **GatedRefreshLegacy**: The legacy-family path used in the recent controller
  shortlists and confirmatory arenas. Refresh gate keyed on `P_flip`; uses
  `prior_refresh_temp` (with a `1e-6` floor — no hidden 0.5 fallback).
- **GatedRefresh**: The theory-family path used in the same studies. Refresh
  gate keyed on `prior_q_divergence` exceeding the per-check `epsilon_t`
  threshold.
- **PFlipMixture**: Mixes Q-refresh and VF-refresh by `p_ratio`. The mixture
  weight is computed from `P_flip`; by default this mode does **not** consult
  `prior_q_divergence` (unlike `GatedRefresh`). Setting the opt-in flag
  `pflip_mixture_divergence_gate = true` (Q8) additionally masks off the
  refresh contribution when `prior_q_divergence ≤ epsilon_t`, making
  divergence a real sweep axis for this mode at the cost of changing the
  baseline math. The default-false preserves prior published numbers.
- **Other refresh modes**: Additional paths remain for lower-level experiments
  and historical comparisons.

Current evidence matters more than the menu of available modes. Recent
short-budget Gomoku7 studies did not support enabling prior refresh as the
default. Refresh remains a valid search axis, but it is currently best treated
as optional/experimental rather than as the recommended deployment profile.

## 6. Parallel Search: Adaptive Virtual Loss

### Problem

Tree-parallel MCTS sends multiple threads into the tree simultaneously.
Without coordination, threads pile up on the same path, wasting compute.
Virtual loss is the standard solution: temporarily penalize in-flight nodes.

### Standard Virtual Loss

Fixed VL adds a constant penalty (typically 1.0) to both the visit count
and Q-value of in-flight nodes. This is simple but over-pessimistic —
it treats all positions equally regardless of uncertainty.

### QUARTZ Split Virtual Loss

QUARTZ separates virtual loss into two components:

    VL = (vvisit, vvalue)

- **vvisit = 1.0** (always): Inflates the effective visit count, reducing
  PUCT exploration bonus. This is the reservation mechanism.
- **vvalue**: Pessimizes Q-value, making the node less attractive.
  Scaled by search state:

      vvalue = σ_Q × depth_decay × entropy_factor × contention_amplifier

### Control Law (2nd Generation)

    depth_decay = 1 / (1 + depth)
    entropy_factor = clamp(root_entropy, 0.5, 2.0) / 2.0
    contention = min(max_pending / n_threads, 2.0)
    amplifier = 1 + dup_rate × (1 + contention)

    vvalue = σ_Q × depth_decay × entropy_factor × amplifier

All five inputs are observable search/runtime state:
- **σ_Q**: Q-value uncertainty (from QUARTZ, one-way read)
- **root_entropy**: Policy entropy at root (high = uncertain)
- **dup_rate**: Fraction of selections hitting already-pending edges
- **max_pending**: Maximum in-flight threads at any node
- **n_threads**: Thread count

### Feedback Loop

The contention amplifier creates genuine runtime feedback:
- High dup_rate + high max_pending → amplified vvalue → threads spread out
  → dup_rate decreases
- Low dup_rate + low max_pending → reduced vvalue → threads can overlap
  when safe → less overhead

This makes adaptive VL a runtime-feedback controller, not a static heuristic.

### Empirical Evidence (ablation_vl.rs)

Component isolation (gomoku7, 500 iters, 4 threads):
- Fixed VL: AvgVV ≈ 1.0, DupRt ≈ 0.27 (over-pessimistic)
- Adaptive VL: AvgVV ≈ 0.17, DupRt ≈ 0.38 (controlled overlap)
- Both achieve comparable move agreement with serial reference

Budget scaling:
- Adaptive advantage grows with budget (+5–15pp at 300–1000 iters)

QUARTZ interaction:
- SelfAdaptive + Fixed = worst combination (double pessimism)
- SelfAdaptive + Adaptive = rescued by σ_Q auto-correction

## 7. Controller Architecture

### Separation of Concerns

QUARTZ separates two controllers:

1. **QuartzController** (quartz.rs): What to search
   - Penalty modes, halt modes, prior refresh
   - Reads: root visits, Q-values, prior distribution
   - Writes: stop decision, score adjustments

### Production Hot-Path Contract

The implementation should preserve the controller meaning above without
charging avoidable heap and snapshot costs to every search refresh:

- QUARTZ statistics read the published root edge slab directly. Passing
  `priors=None` means "use the original priors stored on root edges", not
  "pretend the prior is uniform".
- Short root-child working arrays use stack-backed `SmallVec` and spill only
  for large boards. This keeps Gomoku7/low-budget controller refreshes
  allocation-free while retaining the same σ_Q, P_flip, VOC, and
  prior-divergence formulas.
- The MCTS select path stores the root-to-leaf path in a stack-backed
  `SmallVec<[PathEdge; 32]>`; only unusually deep selections spill.
- Synchronous search can reuse one mutable root-state scratch per worker for
  games that explicitly advertise a measured win from compact `Undo` deltas.
  This removes the per-iteration root-state clone for opted-in games without
  applying the same path to games whose undo payload is a full state snapshot
  or whose clone-and-descend path is faster in practice.
- Expansion may skip re-sorting an already non-increasing evaluator policy
  only for games that explicitly opt in after measurement. This keeps
  move-order-sensitive equal-prior searches from silently changing behavior
  just because a low-level optimization was added.
- Search loops keep a local QUARTZ stats snapshot and reload it only when the
  engine's monotonic stats epoch changes. This preserves the same controller
  semantics while avoiding an `RwLock` read and `QuartzStats` clone on every
  selection in fixed-budget and parallel searches.
- The ignored Rust profile tests accept `QUARTZ_PROFILE_REPEATS=N` and emit
  median/min/max summary rows. Use repeated rows for CPU claims; single-run
  NPS is treated as a smoke signal only, especially for parallel searches.
- With `QUARTZ_MCTS_HOTPATH_METRICS=1`, profile rows include per-iteration
  TT probe counts, aggregate TT lock-wait nanoseconds, split TT read/write
  lock-wait nanoseconds, edge-materialization lock waits, best-effort
  materialization skips, and select/expand/backprop phase timings. These
  counters are diagnostic instrumentation, not a fairness axis; benchmark
  comparisons should either enable them for all conditions or disable them for
  all conditions.
- TT hit/miss/get-or-create/get counters are bucket-local padded atomics and
  are folded only when `hit_rate()` or `contention_snapshot()` is requested.
  This preserves exact reported statistics while avoiding a single global
  atomic cache-line hotspot during parallel search.
- The production TT geometry remains 256 buckets x 1024 slots. A 512 x 512
  geometry reduced measured TT write-lock wait on the Gomoku15 parallel
  hotpath, but did not produce a clear non-instrumented throughput win in
  repeated local profiles, so it is documented as an evaluated candidate
  rather than kept as a production change.
- A seqlock-style lock-free TT read probe was also evaluated and passed the
  same-hash concurrency tests, but it expanded each TT slot from 16 B to 24 B
  and regressed the miss-heavy Gomoku15 parallel `get_or_create` path. The
  production TT therefore keeps read-locked 16 B slots; any future lock-free
  attempt should avoid increasing slot footprint or should target a genuinely
  hit-heavy workload.
- Parallel selection uses best-effort progressive widening after at least one
  child edge is already visible. If another worker is materializing the same
  node, the selector may score the currently published prefix instead of
  blocking on the materialization mutex. Serial selection and first-edge
  publication remain blocking, so a node is never selected with zero visible
  children because of this optimization.
- Edge materialization prepares child hashes and TT nodes before taking the
  per-node materialization mutex; the mutex protects only raw slab writes and
  the final `edge_cursor` publication. The common <=64-edge batch stays
  stack-backed, so Gomoku7-style full-width expansion does not pay a pending
  edge heap allocation.
- The best-effort widening path also uses a per-node owner claim before the
  expensive child-hash/TT preparation step. This prevents several parallel
  selectors from preparing the same widening batch concurrently. A busy
  materialization mutex is still checked before preparation, preserving the
  older "skip instead of prepare-and-wait" behavior.
- When materialization prepares four or more child hashes, TT lookup/insert is
  coalesced by bucket. This keeps the same per-hash hit/miss accounting and
  duplicate-hash semantics, but avoids taking the same bucket read/write lock
  once per child. Smaller progressive-widening increments use the original
  single-lookup path to avoid grouping overhead.
- Uniform/simple rollout evaluators return root policy through
  `GameState::uniform_eval(value)` after computing value. Games with measured
  direct uniform-policy builders, currently Gomoku and Gomoku15, avoid a
  redundant root `legal_moves()` Vec without changing the value rollout.
- Production split-VL modes publish `vvalue` together with a `vvisit`
  reservation. Selection therefore reads `virtual_value` only when an edge has
  a live reservation; the explicit `VvalueOnly` ablation keeps the old
  standalone-vvalue semantics.
- Game implementations used by production search should implement compact
  make/unmake. `Gomoku` and `Gomoku15` both avoid clone-and-replace in
  `apply_move_in_place`; the pure `apply_move` API remains for callers that
  require value semantics.

2. **ParallelismController** (parallel.rs): How to parallelise
   - Split virtual loss, contention management
   - Reads: σ_Q, root_entropy, dup_rate, max_pending (one-way from QUARTZ)
   - Writes: vvisit/vvalue for select()

The coupling is one-way: ParallelismController reads QUARTZ state but
never modifies it. This prevents feedback loops between the two controllers.

### Refresh Cycle

Every `check_interval` iterations (default: 20):
1. Compute root statistics (σ_Q, entropy, visit distribution)
2. Update QuartzController stats
3. Check halt condition
4. Update ParallelismController (σ_Q, root_entropy)
5. Optionally refresh prior (if mode enables it)

## 8. Implementation Notes

### What the controller claim means

Because score shaping is root-only, "controller improved search" should be
read narrowly:

- improved root move selection
- improved adaptive stopping
- improved parallel thread distribution

It does not mean QUARTZ rewrites the full interior-tree selection policy.

### Virtual-loss feedback summary

The active virtual-loss control law is:

    vvalue = σ_Q × depth_decay × entropy_factor × contention_amplifier

with:

- `depth_decay = 1 / (1 + depth)`
- `entropy_factor = clamp(root_entropy, 0.5, 2.0) / 2.0`
- `contention_amplifier = 1 + dup_rate × (1 + max_pending / n_threads)`

All inputs are observable runtime/search state. The runtime law is
state-derived, but the surrounding scales and clamps are explicit
configuration.

### Legacy helper path

`TreeMCTS` remains in the repo as a simplified arena helper, but it is not the
training engine and not the benchmark-grade search stack. It lacks the full
TT/VL/progressive-widening/runtime-controller combination used by the Rust
training and evaluation path.

## 9. Limitations and Honest Scope

### What QUARTZ Does Well

- Adaptive stopping with strong evaluators (saves 20–60% compute)
- Root action ranking improvement via uncertainty-aware penalty
- Thread distribution in parallel search (6× less pessimism than fixed VL)
- Interaction between QUARTZ penalty and VL (σ_Q auto-correction)

### What QUARTZ Does Not Do

- Does not modify interior tree policy (root-only score shaping)
- Does not learn controller parameters online
- Does not replace the neural network evaluator
- Adaptive stopping requires evaluator loss < ~1.0 to trigger convergence
- Does not guarantee improvement with weak/untrained evaluators
- Does not currently justify enabling prior refresh by default on Gomoku7

### Search Hyperparameters and Fixed Thresholds

The controller mixes two classes of numbers:

1. Explicit search hyperparameters that are configured per run and should be
   treated as sweep targets.
2. Hard-coded thresholds/clamps that live in the current Rust implementation.

The most important explicit hyperparameters are:

| Hyperparameter | Role | Current status |
|----------|-------|---------|
| `sigma_0` | uncertainty reference scale | swept in controller studies |
| `hbar_penalty_cap` | root penalty cap | swept in controller studies |
| `min_visits` | minimum evidence before some controller actions | swept in controller studies |
| `check_interval` | refresh/telemetry cadence | swept in controller studies |
| `c_puct` | base exploration strength | swept in controller studies |
| `prior_refresh_rate` | refresh strength | currently not favored in Gomoku7 sweeps |
| `prior_refresh_temp` | refresh temperature | currently not favored in Gomoku7 sweeps |

Examples of fixed thresholds/clamps still hard-coded in the implementation:

| Fixed threshold | Purpose |
|----------|---------|
| `P_flip` threshold and related guards | adaptive stopping trigger logic |
| entropy clamp | VL entropy factor range |
| contention cap | maximum contention amplifier |
| `vvalue` floor | minimum pessimism floor |

The honest claim is therefore not "hyperparameter-free." The honest claim is:
the runtime signals are state-derived, while the controller family sits on top
of explicit search hyperparameters and a smaller set of fixed implementation
thresholds.
