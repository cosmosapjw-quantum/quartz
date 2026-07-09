# BQ++ Design Summary — Bounded-rational Bayesian Free-Energy Search Controller

**Status (2026-05-04):** Phases 0-7 of the 8-phase plan ship as
primitives + audit notes + Python prototype. Phase 8 partial
completion adds the `policy_battle` ablation preset and this design
summary; engine integration + cleanup are deferred to a follow-up
empirical-validation session.

This document is the single-page external reference for BQ++. Read
it after the project README and `audit_external_review_response.md`.

---

## 1. The single principle

BQ++ is governed by one objective:

> **Choose the next computation `c` that maximizes
> `E[ΔR(B_t) | c] / cost(c)`** — expected decision-loss reduction
> per compute cost — **and stop when `max_c E[ΔR | c] ≤ cost(c)`
> or when a confidence-interval certificate `L_b > max_{a≠b} U_a`
> holds.**

`R(B_t) = E_θ[max_a θ_a − θ_b̂_t]` is the posterior simple regret of
the currently-identified best arm `b̂_t`. `B_t` is the posterior
belief at time `t`. `cost(c)` is the actual NN-eval / CPU-select
latency.

This single principle replaces the cancelled `BayesianQuartz` design's
229k-mode-combination heuristic surface. Every primitive in BQ++ is
an approximation of the principle, applied to a specific allocation
or stopping sub-problem.

---

## 2. The five modules

The codebase implements BQ++ as five lock-free, edge-local-indexed
primitives. Each is shipped as standalone tests; the production
policy (Phase 8 engine integration) composes them.

### Module 1 — Calibrated Belief

`src/mcts/policy/cache.rs::PolicyCache.lower / .upper` (Phase 2).
Per-arm posterior std-dev via Welford + empirical-Bayes shrinkage
on the [0, 1] scale. Math primitive: `EdgeView::sigma_a` from P06
+ Python prototype `belief.py`.

### Module 2 — Confidence Certificate

`src/mcts/policy/cache.rs::PolicyCache.cert_gap` (Phase 2). The
Empirical-Bernstein certificate `L_b > max_{a≠b} U_a`. Maurer-Pontil
2009 width formula; widths in `lower/upper` arrays. KL-LUCB
certificate (`src/mcts/policy/kl_lucb.rs`) is a secondary option for
calibrated Bernoulli backups (terminal win/loss).

### Module 3 — Computation-Value

`src/mcts/policy/kg_stop.rs` (Phase 4). Knowledge Gradient
approximation `KG_a = E[max_j μ_j⁺] − max_j μ̂_j` evaluated as the
full `s · φ(Δ/s) − Δ · Φ(−Δ/s)` (audit §1.2 correction). KG-stop
fires when `max_a KG_a < kg_threshold · cost_per_pull_ms`.

### Module 4 — Candidate Reservoir

`src/mcts/policy/gumbel_sh.rs` (Phase 3) + `tactical.rs` (Phase 5)
+ `reservoir.rs` (Phase 6). Three sources of root candidates:
- **Gumbel without-replacement sampling** (Danihelka 2022) for
  prior-driven candidates with policy-improvement guarantees at
  small budgets.
- **Tactical sentinel** for forced-move detection (Gomoku
  immediate-win and forced-block; Chess and Go deferred).
- **Nested-reservoir** with quantile pruning + cooldown hysteresis
  for live-set maintenance against local minima.

### Module 5 — CPU-friendly Cache

`src/mcts/policy/cache.rs::PolicyCachePublisher` (Phase 2).
`arc-swap::ArcSwap<Arc<PolicyCache>>` lock-free publish. Hot-path
`load()` is ~10ns (atomic pointer + counter increment). `EdgeRef`
enforces `edge_pos` (dense) vs `action_id` (sparse) separation
by construction.

---

## 3. Composed policy (Phase 8 production target)

The `BQPP` policy (engine integration deferred) composes the modules
into one named policy with this halt order:

```
loop:
    if cache.is_stale_for(snap):
        Continue              # audit §1.7: no halt from stale data
    if cache.forced_move_pos.is_some():
        Stop(TacticalForced)
    if snap.root_visits >= max_visits:
        Stop(MaxVisits)
    if snap.elapsed_ms >= time_cap_ms:
        Stop(MaxTime)
    if snap.root_visits < min_total:
        Continue              # not enough data
    if cache.cert_gap > 0:
        Stop(EmpBernsteinCertified)   # primary EB cert
    if cache.max_kg_per_ms < kg_threshold:
        Stop(PolicyConverged)         # KG-stop
    Continue
```

`observe()` populates the cache with: per-arm σ_a (Welford+shrink),
EB lower/upper CI, KG per arm, χ² statistic, forced_move_pos (from
sentinel), Gumbel SH bracket state.

`score_adjustment(edge)` is hot-path: `cache.load() → indexed
read of p_eff[edge.edge_pos]`. No allocation, no mutex.

---

## 4. Phase-by-phase deliverables

| Phase | Deliverable | Commit | LOC | Tests |
|---|---|---|---|---|
| 0 | claim freeze + math fixes (docs only) | `39e18b2` | +232 | 0 |
| 1 | Python prototype + 79 numerical tests | `32e5ea9` | +2729 | 79 |
| 2 | ArcSwap PolicyCache + EdgeRef | `68f8dba` | +325 | 7 |
| 3 | Gumbel SH + Sequential Halving | `c40b3b2` | +294 | 10 |
| 4 | KG-per-cost stop + corrected EI | `d93f554` | +285 | 15 |
| 5 | Tactical sentinel (Gomoku) | `038ad68` | +297 | 7 |
| 6 | Nested-reservoir live-set | `cf20df3` | +272 | 11 |
| 7 | MENTS soft-Bellman primitives | `c718b2d` | +222 | 11 |
| 8 | `policy_battle` preset + this doc | (this commit) | +60 | (re-uses 79 + 61) |
| **Total** | | | **+4716** | **140 (Rust + Python)** |

`cargo test --release`: 539 passed (was 444 before BQ++; +95 new tests
across Phases 2-7 inclusive of P08 KLLUCBStop = 9 from earlier P08).

`pytest -q` in `prototype/`: 79 passed.

`pytest -q tests/test_ablation_study.py`: 66 passed (preserved).

---

## 5. What is intentionally NOT in this commit

The full BQ++ deployment requires three items that are **deliberately
deferred** to a separate empirical-validation session:

1. **Engine integration.** `MctsConfig.search_policy:
   Option<Arc<dyn SearchPolicy>>` field; hot-path consumption inside
   the select loop; `--policy={legacy_az, legacy_quartz, bqpp,
   ments}` CLI translator. This is invasive (~150 Rust LOC across
   src/mcts/mod.rs and select.rs) and warrants a dedicated review
   pass.
2. **Default flip from `legacy_quartz` to `bqpp`.** The plan calls
   for this only after the new defaults reproduce LegacyQuartz's
   top-1 action ≥ 90% of the time on a 100-position Gomoku 7×7
   fixture. We have no empirical data yet — running the experiment
   is the natural next session.
3. **Deprecation cleanup (the −1500 LOC delta).** Per the plan and
   the audit response, this lands "after ≥1 release with the new
   defaults." Deleting the legacy heuristic branches now would
   destroy a working, tested fallback before BQ++ has demonstrated
   non-inferiority.

The `policy_battle` ablation preset added in this Phase 8 commit
contains placeholder subject names (`legacy_az`, `legacy_quartz`,
`bqpp_no_gumbel`, etc.) that will be resolved to actual engine
configurations once item (1) lands.

---

## 6. Honest novelty assessment

(Same as `LEGACY_VS_BAYESIAN_QUARTZ.md` §6.) BQ++ as a *stack of
techniques* is **not novel**. Every primitive — Welford,
empirical-Bernstein, Gumbel SH, Knowledge Gradient, MENTS, nested
sampling — has been in the literature for decades.

The **contribution** is the integration into AlphaZero MCTS root
control:
- A single named, code-reviewable policy with one set of
  hyperparameters (no 229k-combination surface).
- Falsifiable telemetry — every halt decision carries
  `gap_bits`, `chi2`, `max_kg_per_ms`, `prior_surprise` in the
  `controller_summary.extended` JSON block (P01, schema_version 6).
- CPU-friendly hot path for the *actually wired* policy
  (`KLLUCBStop`, dispatchable via `--policy=kl_lucb_stop`): edge-local
  indexing, no allocation on `observe`/`should_halt` beyond the
  per-call `EdgeView` `Vec` built in `policy_halt_check`
  (`src/mcts/mod.rs`); it uses a plain `parking_lot::Mutex<Cache>`
  for its own small cached-certificate state, not the ArcSwap design
  below. **Re-scoped (A4-b audit)**: "no mutex, ArcSwap publish"
  describes `PolicyCache`/`PolicyCachePublisher`
  (`src/mcts/policy/cache.rs`), which is real, tested code but is
  **not constructed or read by the engine anywhere** — grep-verified
  zero references outside `src/mcts/policy/`. Do not read this bullet
  as a property of the live search path until that module is wired;
  see `docs/CLAIM_LEDGER.md` for the per-module status table.
- Reproducibility of every existing experiment via
  `--policy=legacy_quartz` shim (P07).
- Bit-identical numerical behavior between Python prototype
  (`prototype/`) and Rust (`src/mcts/policy/`) — drift is caught
  at port time by hand-derived expected values.

This is engineering, not novel math. A reviewer for an external
audit should know this exactly.

---

## 7. References

(Subset of `LEGACY_VS_BAYESIAN_QUARTZ.md` §10 references most
directly applied in BQ++.)

- Danihelka et al. 2022, "Policy Improvement by Planning with
  Gumbel" (ICLR). → Phase 3.
- Frazier-Powell-Dayanik 2009, "The Knowledge-Gradient Policy."
  → Phase 4.
- Karnin-Koren-Somekh 2013, "Almost Optimal Exploration in Multi-
  Armed Bandits." → Phase 3.
- Kaufmann-Kalyanakrishnan 2013, "Information Complexity in Bandit
  Subset Selection." → P08 / KL-LUCB module.
- Maurer-Pontil 2009, "Empirical Bernstein Bounds and Sample
  Variance Penalization." → Phase 2 / Phase 4.
- Russo-Van Roy 2018, "Satisficing in Time-Sensitive Bandit
  Learning." → Phase 4.
- Skilling 2006, "Nested Sampling for General Bayesian
  Computation." → Phase 6 (live-set idea only; NOT evidence
  estimation).
- Welford 1962, "Note on a Method for Calculating Corrected Sums
  of Squares." → Phases 1, 2.
- Wu 2019, "Accelerating Self-Play Learning in Go" (KataGo).
  Plan-level inspiration for the playout-cap pattern.
- Xiao et al. 2019, "Maximum Entropy Monte-Carlo Planning"
  (MENTS, NeurIPS). → Phase 7.

---

## 8. Pointers

- Audit response: [`../audit_external_review_response.md`](../audit_external_review_response.md)
- Original v1.0 plan (preserved):
  `~/.claude/plans/iridescent-giggling-bachman.md`
- BQ++ implementation plan: `~/.claude/plans/bq_plus_plus_plan.md`
- Per-phase audit notes: `audit_phase{0..7}_*.md` in repo root.
- Python prototype: [`../prototype/`](../prototype/).
- Rust modules: [`../src/mcts/policy/`](../src/mcts/policy/).
- Comparison doc with superseded sections:
  [`LEGACY_VS_BAYESIAN_QUARTZ.md`](LEGACY_VS_BAYESIAN_QUARTZ.md).

End of document.
