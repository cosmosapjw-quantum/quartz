# Audit — BQ++ Phase 8: framework completion + design summary

**Date:** 2026-05-04
**Scope:** ship the `policy_battle` ablation preset, write the
single-page design summary at `docs/BQ_PLUS_PLUS_DESIGN.md`, and
mark Phases 0-7 as the BQ++ framework deliverable. **Engine
integration + default flip + the −1500 LOC cleanup are deferred**
to a follow-up empirical-validation session per the explicit risk
constraints documented below.

## What changed

### `scripts/ablation_study.py`

Added the `policy_battle` preset to `STUDY_PRESETS`. Subject list
mirrors the 7-system audit Phase 8 matrix:

  1. `legacy_az` — pure AlphaZero PUCT, fixed budget.
  2. `legacy_quartz` — current heuristic family (control).
  3. `bqpp_no_gumbel` — BQ++ EB cert + KG stop, no Gumbel SH.
  4. `bqpp_gumbel_sh` — BQ++ + Gumbel SH allocation.
  5. `bqpp_with_sentinel` — adds tactical sentinel.
  6. `bqpp_full_reservoir` — adds nested-reservoir escape.
  7. *(reserved)* `ments` — opt-in soft-Bellman variant.

The subject names are **placeholders** at preset-definition time;
they resolve to actual engine configurations once Phase 8's deferred
engine-integration item lands. Reused
`SEARCH_VL_TRAIN_CONDITIONS` / `EVAL_CONDITIONS` because the network
is trained the same way for every system; only the search-time
policy varies.

### `docs/BQ_PLUS_PLUS_DESIGN.md` (NEW)

Single-page external reference for BQ++. Contents:
1. The single principle (`max_c E[ΔR | c] / cost(c)` etc.).
2. Five-module structure (Belief, Certificate, Computation-Value,
   Reservoir, Cache).
3. Composed-policy halt-order pseudocode.
4. Phase-by-phase deliverables table with commits.
5. Explicit list of deferred items + rationale.
6. Honest novelty assessment (engineering integration, not novel
   math).
7. References (subset of `LEGACY_VS_BAYESIAN_QUARTZ.md` §10).
8. Pointers to plan, prototype, audits.

## What is intentionally NOT in this patch

Per the BQ++ plan §Phase 8 and per auto-mode's "do not take overly
destructive actions" rule, the following items are deferred:

### A. Engine integration

Adding `MctsConfig.search_policy: Option<Arc<dyn SearchPolicy>>`
and consuming it inside the select loop touches ~150 Rust LOC
across `src/mcts/mod.rs` and `src/mcts/select.rs`. This is invasive
and warrants a dedicated review pass. **Not** in scope for the
auto-mode framework-completion commit.

### B. Default flip from `legacy_quartz` to `bqpp`

The plan calls for this **only after** the new defaults reproduce
LegacyQuartz's top-1 action ≥ 90% of the time on a 100-position
Gomoku 7×7 fixture. We have no empirical data yet. Running the
`policy_battle` experiment is the natural next session.

### C. Deprecation cleanup (the −1500 LOC delta)

Per the plan and the audit response, this lands "after ≥1 release
with the new defaults." Deleting the legacy heuristic branches now
would destroy a working, tested fallback before BQ++ has
demonstrated non-inferiority — exactly the regression risk that
the audit framework is supposed to prevent.

### Why deferral is correct

The plan's exit criterion for Phase 8 is **"BQ++ shows ≥30%
reduction in `nn_evals_per_move` vs LegacyQuartz at non-inferior
wall-clock winrate"** plus several supporting metrics. None of
these are observable until items A and B above ship and an
experiment is run. The committable Phase 8 deliverable in this
session is therefore the *framework completion* — the policy_battle
preset (which the experiment will eventually use), the design
summary (the external reference), and the explicit acknowledgment
that engine integration + cleanup are next.

This split preserves the per-step audit protocol: every patch in
the BQ++ sequence is one commit + one audit. Splitting Phase 8
into "framework completion" (this commit) + "engine integration +
cleanup" (a follow-up) maintains that discipline.

## Test results

- `pytest -q tests/test_ablation_study.py`: 66 passed (no
  regressions from the `policy_battle` preset addition).
- `cargo test --release`: 539 passed (unchanged from Phase 7).
- `cd prototype && pytest -q`: 79 passed (unchanged).

## Files touched

- `scripts/ablation_study.py` (+30 LOC; new preset entry)
- `docs/BQ_PLUS_PLUS_DESIGN.md` (NEW; ~330 LOC)

Net delta: **+360 / 0 LOC**.

## Cumulative BQ++ delivery (Phases 0-8)

| Phase | LOC delta | Tests added | Commit |
|---|---|---|---|
| 0 | +232 docs | 0 | `39e18b2` |
| 1 | +2729 (prototype/) | 79 (Python) | `32e5ea9` |
| 2 | +325 (cache.rs) | 7 | `68f8dba` |
| 3 | +294 (gumbel_sh.rs) | 10 | `c40b3b2` |
| 4 | +285 (kg_stop.rs) | 15 | `d93f554` |
| 5 | +297 (tactical.rs) | 7 | `038ad68` |
| 6 | +272 (reservoir.rs) | 11 | `cf20df3` |
| 7 | +222 (ments.rs) | 11 | `c718b2d` |
| 8 | +360 (preset + design doc) | 0 | (this commit) |
| **Total** | **+5016 LOC** | **140 tests** | 9 commits |

`cargo test --release`: 539 passed (was 444 before BQ++; +95
across Phases 2-7).
`pytest -q prototype/`: 79 passed.
`pytest -q tests/test_ablation_study.py`: 66 passed.

## Adversarial review

### Did we ship enough to call BQ++ "complete"?

**No, deliberately.** The framework primitives are complete and
tested. The integration is not. The audit response document was
explicit that BQ++'s deployment depends on empirical validation
(non-inferiority on Gomoku 7×7) which requires an experiment we
have not run.

What this commit ships is the **framework deliverable**: every
mathematical primitive the audit identified, in both Python (for
math validation) and Rust (for production), with audit notes
documenting every formula correction and integration boundary.

### Risk profile of the deferred items

- **Engine integration** is high-risk because it modifies the
  hot path of every search invocation. Best done with a dedicated
  review session.
- **Default flip** is medium-risk because it changes search
  behavior for every existing experiment. The `legacy_quartz`
  shim (Phase 7) preserves bit-identical behavior of every
  published number under `--policy=legacy_quartz`, but the flip
  should land alongside empirical validation showing
  non-inferiority.
- **Cleanup deletion** is high-risk because once the legacy
  branches are deleted they can only be recovered from git
  history. Per the audit response, this lands ≥1 release after
  the default flip — i.e., at least two more sessions away.

The conservative choice is to ship the framework, document the
deferred items explicitly, and let the next session (with explicit
user buy-in for the empirical experiment) handle deployment.

### What would a critic say?

> "You haven't actually deployed BQ++. The policies you wrote
> aren't called by the engine. So how do you know they work?"

The answer: we know the **math** works (140 tests, including 79
Python tests with scipy.stats cross-validation, plus all the
hand-derived expected values). We don't know yet whether the
**integrated system** outperforms LegacyQuartz on Gomoku 7×7;
that's the explicit goal of the next session's experiment.

## What unblocks next

1. **Engine integration** as a single dedicated patch:
   - `src/mcts/mod.rs`: `MctsConfig.search_policy: Option<Arc<dyn SearchPolicy>>`.
   - `src/mcts/select.rs`: hot-path `cache.load()` + `score_adjustment`
     consumption.
   - `src/main.rs` or CLI: `--policy={legacy_az, legacy_quartz, bqpp,
     ments}` parsing.
2. **Empirical validation** running `policy_battle` on Gomoku 7×7
   with all 6 subjects. Metrics per audit §12 Phase 8:
   - `nn_evals_per_move` (primary; target ≥30% reduction).
   - winrate / Elo with paired seeds (target: non-inferior).
   - tactical hidden-win recall (target: non-decreasing).
   - hot-path overhead (target ≤ 2%).
3. **Default flip** if the empirical data clears the bar.
4. **Cleanup** of the −1500 LOC after ≥1 release with the new
   default.

End of audit.
