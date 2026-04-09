# QUARTZ Theory Foundations

## Overview

QUARTZ (Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable)
is a search controller for Monte Carlo Tree Search (MCTS) that adapts its
behavior based on observable search statistics. It sits on top of standard
AlphaZero-style PUCT search and modulates three aspects:

1. **Score shaping** — how root actions are ranked
2. **Adaptive stopping** — when to stop searching
3. **Parallel thread distribution** — how to manage virtual loss

All controller inputs are derived from search state. The control law uses
fixed constants only (no learned or user-tuned hyperparameters).

## 1. Search Policy: Boltzmann on Discrete Exponential Family

### Standard PUCT

In AlphaZero, the selection rule at each node is:

    a* = argmax_a [ Q(a) + c_puct · π(a) · √N_parent / (1 + N_a) ]

where Q(a) is the mean value, π(a) is the prior from the neural network,
and N_a is the visit count.

### QUARTZ Modification (Root Only)

QUARTZ adds a score-shaping penalty at the root node (depth = 0):

    score(a) = PUCT(a) − penalty(a)

The penalty is computed via the one-loop visit penalty:

    penalty(a) = h_cap · (N_a / N_parent)

where h_cap = min(σ_Q / σ_0, penalty_cap). This penalizes over-visited
actions proportionally to the Q-value uncertainty, encouraging exploration
when the evaluator is uncertain.

**Scope**: Score shaping applies at root depth only. Interior tree nodes
use standard PUCT. This is by design — root action ranking is where
controller guidance has the highest impact-to-cost ratio.

## 2. Uncertainty Estimation: σ_Q and Related Statistics

### Q-value Standard Deviation (σ_Q)

σ_Q is the standard deviation of Q-values across root children:

    σ_Q = std({Q(a) : a ∈ children(root)})

This is the primary uncertainty signal. When σ_Q is high, the evaluator
disagrees strongly about different actions, suggesting more search is
needed. When σ_Q is low, the evaluator is confident.

### Normalized Uncertainty (hbar_eff)

    hbar_eff = σ_Q / σ_0

where σ_0 is a reference scale (configurable, default 0.5). This
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
- The threshold for stopping is P_flip < 0.159 (configurable).

## 4. Adaptive Stopping: VOC and Halt Modes

### Value of Computation (VOC)

The VOC framework asks: "Is the expected gain from one more search
iteration worth the computational cost?"

    VOC = expected_improvement − cost

When VOC < 0, search should stop.

### Halt Modes

| Mode | Description |
|------|-------------|
| Fixed | Always use full budget |
| SimpleThreshold | Stop when P_flip < threshold |
| VOC | Full VOC computation with cost model |
| ConfAdaptive | Confidence-adaptive threshold |

### Stop Reasons (recorded in telemetry)

| Reason | Meaning |
|--------|---------|
| Budget | Hit iteration limit |
| VOCNegative | VOC went negative |
| PFlipConverged | P_flip below threshold |
| ConfidenceHigh | Confidence threshold met |
| Unknown | Default / unclassified |

## 5. Prior Refresh

Standard MCTS uses the neural network prior once at node expansion. QUARTZ
can refresh the prior during search based on accumulated Q-value information.

### Modes

- **GatedRefresh**: Refresh prior only when prior_q_divergence exceeds a
  threshold. Mixes original prior with visit-proportional policy.
- **SelfAdaptive**: Continuous refresh with strength proportional to σ_Q.
  Per-action refresh weight α_a = N_a / (N_a + K), temperature
  τ = ln(1 + N_total / K). Most effective with strong NN evaluators.
- **PFlipMixture**: Weight refresh by P_flip (high P_flip = more refresh).

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

## 8. Limitations and Honest Scope

### What QUARTZ Does Well

- Adaptive stopping with strong evaluators (saves 20–60% compute)
- Root action ranking improvement via uncertainty-aware penalty
- Thread distribution in parallel search (6× less pessimism than fixed VL)
- Interaction between QUARTZ penalty and VL (σ_Q auto-correction)

### What QUARTZ Does Not Do

- Does not modify interior tree policy (root-only score shaping)
- Does not learn controller parameters (all fixed constants)
- Does not replace the neural network evaluator
- Adaptive stopping requires evaluator loss < ~1.0 to trigger convergence
- Does not guarantee improvement with weak/untrained evaluators

### Design Constants

The control law uses the following fixed constants (not hyperparameters
in the sense that they are not tuned per-game or per-run):

| Constant | Value | Purpose |
|----------|-------|---------|
| σ_0 | 0.5 | Reference scale for hbar_eff |
| penalty_cap | 0.3 | Maximum one-loop penalty |
| P_flip threshold | 0.159 | Adaptive stopping trigger |
| check_interval | 20 | Refresh cycle frequency |
| entropy clamp | [0.5, 2.0] | VL entropy factor range |
| contention cap | 2.0 | Maximum contention amplifier |
| vvalue floor | 0.01 | Minimum vvalue (prevents zero) |

These are design choices, not learned parameters. They could in principle
be tuned, but the current values work across all tested games without
modification. The claim is "state-derived with fixed constants," not
"hyperparameter-free."
