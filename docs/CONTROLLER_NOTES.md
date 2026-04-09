# QUARTZ Controller Architecture Notes

## Design: Separated Controllers

### Score Shaping Scope

QUARTZ score shaping (penalty, prior refresh, Fisher term) applies at
**root depth only** (depth==0 in select.rs). Interior tree nodes use
standard PUCT. This is by design: root action ranking is where controller
guidance has the highest impact-to-cost ratio. The parallelism controller
(virtual loss) operates at all depths.

This means "controller improved search" is more precisely "controller
improved root move selection + adaptive stopping + parallel thread distribution".

QUARTZ separates search semantics from parallelism control:

- **QUARTZ controller** (quartz.rs): what to search
  - Penalty modes: None, GatedRefresh, SelfAdaptive, PFlipMixture
  - Adaptive stopping: P_flip convergence, VOC halt
  - Prior refresh: Q-value feedback to policy

- **ParallelismController** (parallel.rs): how to parallelise
  - Split virtual loss: vvisit (reservation) + vvalue (pessimism)
  - 2nd-gen feedback: dup_rate x contention severity
  - One-way coupling: reads sigma_Q from QUARTZ, never writes

## Virtual Loss Control Law

vvalue = sigma_Q x depth_decay x entropy_factor x contention_amplifier

Where:
- sigma_Q: Q-value uncertainty from QUARTZ (search state)
- depth_decay: 1/(1+depth), root contention most expensive
- entropy_factor: clamp(root_entropy, 0.5, 2.0) / 2.0
- contention_amplifier: 1 + dup_rate x (1 + max_pending/n_threads)

All inputs are observable search/runtime state.
Fixed constants only (no learned or user-tuned parameters).

## Feedback Loop

High dup_rate + high max_pending -> amplified vvalue -> threads spread
Low dup_rate + low pending -> reduced overhead -> threads can overlap

This creates genuine runtime feedback, not a static heuristic.

## Legacy Components (not used in training)

TreeMCTS implements a simplified QUARTZ-lite controller for arena evaluation:
- GatedRefresh, SelfAdaptive, None modes
- Root-only penalty (no tree-level controller)
- No TT, no virtual loss, no progressive widening
