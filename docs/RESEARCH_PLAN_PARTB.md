# QUARTZ Part B — Refined Research Plan

> **Status:** living research-design document (Part B of
> `~/.claude/plans/parallel-popping-owl.md`). It refines the CPU-friendly /
> human-GM-signature / low-MCTS-budget research direction into measurable
> lanes with pre-registered kill-criteria. It is a *plan*, not evidence:
> nothing here is `VALIDATED`.
>
> Read [`THESIS.md`](THESIS.md) first — this document operationalizes its
> four propositions. Every lane below carries its adversarial-audit
> provenance (`~/.claude/plans/parallel-popping-owl.md` §0.5 adversarial
> self-ask, §0.6 CCoT discriminating experiment, §0.7 PDR revision).

Central narrative (THESIS.md P1/P3): **implement the single principle
`E[ΔR]/cost` as game-agnostic statistics; the human-GM signatures are
measured as pre-registered *emergent predictions*, never as optimization
targets.** The discriminating signature (P3, SB3) is *compute-VOC
allocation tightness*, not node count.

Section order = increasing verification cost. Where a lane can be
partially exercised without the (currently unavailable) GPU/engine loop,
the offline-computable substrate is implemented and unit-tested now; the
online efficacy claim stays `SPECIFIED` / `PROPOSED` until a paired
ablation runs.

---

## B1 — Signature battery (measurement infrastructure)

**Deliverable (this session):** `quartz/phase15_signatures.py` +
`tests/test_phase15_signatures.py`. Game-agnostic, trace-only signature
functions with hand-derived regression pins.

The battery is split by the P3 two-pattern requirement. **Macro** metrics
are predicted skill-*invariant* (matching them is a sanity check, never a
claim). The **discriminating** metric is predicted skill-*increasing*.

| ID | Signature | Source data | Predicted skill relation | Status |
|---|---|---|---|---|
| O1 | `K_eff = exp(H(policy))` candidate concentration + its `d K_eff / d log-budget` trajectory | `trace_policies`, `trace_budgets` (present) | **invariant** (de Groot) | implemented (`k_eff`, `concentration_vs_budget`) |
| O2 | per-move budget Gini / entropy | per-move realized budgets (present) | descriptive | implemented (`budget_gini`, `budget_entropy`) |
| O3 | commit-time vs **external** difficulty | oracle top-2 margin / cross-seed disagreement | discriminating | **needs oracle** (analyst offline) |
| O4 | depth-selectivity = top-2 subtree visit share × mean max depth | small Rust telemetry addition | **invariant** (de Groot) | **needs Rust telemetry** |
| O5 | revision dynamics: `first_revision_step`, `flip_flop_rate`, `final_sparsity` | `trace_policies` (present) | descriptive | implemented |
| O6 | surprise-conditional burst precision `P(hard\|burst)/P(hard)` | burst events (H3) + external oracle | discriminating | **needs H3 + oracle** |
| **VOC** | `voc_tightness = corr(per_move_budget, voc_proxy)` | budgets (present) + **offline oracle proxy** | **increasing** (Russek 2025) | implemented (`voc_tightness`) |

### Non-circularity guard (THESIS.md P3, mandatory)

`voc_proxy` = shallow-vs-deep argmax disagreement × Q-gap, computed
**offline by the analyst from a high-budget reference oracle**. The
function `voc_tightness(per_move_budgets, voc_proxy_values, ...)` takes the
proxy as an explicit argument on purpose: the engine never computes or
consumes VOC during search. If it did, tightness would be tautological.
This guard is what makes P3 falsifiable rather than "optimize the metric".

### What each signature can and cannot decide

- O1/O4 (macro) turning out skill-*discriminating* would falsify the "node
  counts are skill-invariant" premise — the whole story would be wrong,
  not merely incomplete. That is a *good* falsifier to have.
- VOC-tightness flat across checkpoint strength falsifies the motto (P3
  falsifier b).
- The reportable signature claim is the **conjunction** (O1/O4 invariant
  AND VOC-tightness increasing) — a single-metric match is explicitly
  insufficient (THESIS.md P3).

### Wiring plan (deferred, needs engine/oracle)

1. O4 telemetry: add top-2 subtree visit share + mean max depth to the
   Rust root telemetry that already feeds `trace_policies`; surface in the
   phase15 trace artifact (schema bump).
2. O3/O6/VOC proxy: an offline analyzer pass that replays each committed
   position at a high reference budget to produce the oracle margin and
   the `voc_proxy`, then joins on `(checkpoint, position)`.
3. Study-level rollup: `budget_gini` / `voc_tightness` are computed across
   many moves at the study level, not per trace; `trace_signature_summary`
   only bundles the per-position O1/O5 signatures.

**Kill/keep for B1 itself:** B1 is infrastructure, not a hypothesis — it
cannot be "killed". Its job is to make P3 measurable. It is done when the
double-pattern (§B3 dissociation) can be computed from real traces.
