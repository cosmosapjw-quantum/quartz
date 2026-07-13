# Audit — Stage 7: Live-Engine Conditional Work

**Date:** 2026-07-13
**Scope:** wire the four green-lit metacognitive-lab lanes into the live
engine / online phase15 stack, run their pre-registered experiments, and close
the Stage 7 conditional gate table. Every claim is anchored to the Stage 1-6
lab verdicts (CLAIM_LEDGER; `docs/RESEARCH_PLAN_PARTB.md` Metacognitive-lab
campaign section).

This file is the **pre-registration record**. Kill/success criteria for every
lane are fixed here *before* any Stage 7 experiment runs; CLAIM_LEDGER rows
reference this file. Per-commit audit sections are appended below as work lands.

## Gate table (resolved from Stage 1-6)

| Lane | Gate input | Status |
|---|---|---|
| KG-stop `SearchPolicy` wrapper (Rust) | Stage 1 `kg_rank_risk` CI-separated regret win | GREEN — build authorized; play claim must be re-earned on real MCTS |
| H1 online halt + flip-calibration vs P_flip (Python) | Stage 3 discrimination gate `gate_pass=True` | GREEN |
| H3 backflow burst + O6 precision via forked_voc labels | Stage 2 VOC labels non-degenerate | GREEN (impl was SPECIFIED) |
| B13 research-grade verdict | unconditional residual item | OPEN |
| Danihelka ranking restoration | Stage 3 narrowing net value | NOT triggered → closed (below) |
| Constraint | Stage 5 H5 dup lane KILLED | No Stage 7 claim may cite adaptive-VL duplication reduction |

## Danihelka closure (cancel reason)

Stage 3 `candidate_morphology_lab` produced **zero** CI-separated *total*-regret
improvements from widening/narrowing (omission relief always repaid in ranking
regret under a shared budget; run_contract_hash `c02c718b…`). The condition for
restoring the guarantee-preserving Gumbel-SH ranking (`g(a)+logits(a)+σ(q̂(a))`)
— "narrowing shows evidenced value" — is therefore not met. The Danihelka
guarantee-restoration row is closed **DEPRECATED (Stage 7 gate: NOT triggered)**.
Reopen only if a future lane shows a CI-separated net-total gain from
widening/narrowing. The bracket's honest budget accounting (a separate,
IMPLEMENTED row) is untouched.

## Pre-registered kill / success criteria

### KG-stop engine smoke (E3 / C4)

Paired per position: `QUARTZ_SEARCH_POLICY=kg_stop` vs env-unset fixed halt, on
`seed_101/gen_20.pt`, positions × budgets {64,128,256} × kg_threshold
{1e-4,1e-3,1e-2}, `check_interval = max(4, budget//8)`.

- **Success (SMOKE ceiling):** some grid point achieves **mean budget saved ≥ 20%
  with top-1 argmax agreement ≥ 0.95** vs the fixed-halt decision.
- **Kill:** zero halts anywhere in the grid ⇒ "KG scale does not transfer to
  real adaptive shared-tree backups" — the wrapper stays IMPLEMENTED, the lane
  closes (echoes the KL-LUCB A1-a low-budget-unreachability history).
- **Demote:** halts fire but top-1 agreement < 0.80 at every saving level ⇒
  anti-conservative; diagnostic only, no efficacy claim.
- **Tier ceiling:** SMOKE-VALIDATED (engine result). NOT a play-strength or P2
  nn_evals claim — that must be re-earned under the Ablation Start Conditions.

### H1 flip-calibration lane (E2 / C8)

Predictors at each chunk boundary of the same trace: `s_H1 =
argmax_stability(counts(π_b, b))`, `s_Pflip = 1 − p_flip_b` (engine's own
incumbent p_flip). Outcome `y = 1[argmax(π_b) == argmax(π_holdout)]`, holdout =
ladder max (64), secondary = oracle-256. Confirmatory statistic = paired
Δagreement (H1 − P_flip) at **matched realized budget** (P_flip threshold tuned
to match H1's mean realized budget within ±5%), `paired_bootstrap_ci` (2000
resamples, seed 0). θ\* = 0.9 is the pre-registered confirmatory operating
point; {0.85, 0.95} are descriptive calibration only.

- **Kill (H1 dies):** matched-budget Δagreement CI excludes zero **in P_flip's
  favor**.
- **Survive:** CI straddles zero (match) or excludes zero in H1's favor.
- Reliability diagram (10 bins), ECE, Brier are descriptive.
- Restart-per-chunk and root-continuation rows are stratified, never pooled;
  the headline uses the majority mode.

### H3 / O6 burst precision (E2 / C9)

Burst event = B15 row with `budget_burst_triggered == 1`. Difficulty label
`hard := forked_voc.final_overturns_shallow` on the shared A4 trace bundle for
the same `(checkpoint_id, position_id)` (label uses the full ladder incl.
budgets above the decision point ⇒ different source than the entropy trigger ⇒
non-circular). Statistic = lift `P(hard|burst)/P(hard)`, position-level
bootstrap CI (2000, seed 0).

- **Kill:** lift CI includes 1 ("burst fires at the base difficulty rate").
- **Degeneracy demotion (diagnostic only, no O6 claim):** burst rate > 0.9 or
  < 0.02, or fewer than 30 pooled burst events.

### B13 research-grade verdict (E1 / C11d)

6 checkpoints (3 seed families × weak/mid/strong) × {A4, B5, B13@1.0,
B13@0.25} × budgets {8,16,32,64}, 96 positions, oracle 256, under the ported
`--research-grade` gate. Bonferroni over 2 curvatures × 3 metrics.

- **VALIDATED (improvement):** for some curvature, `delta_kl_to_oracle` CI
  excludes 0 favorably AND neither `delta_accuracy_to_oracle` nor
  `delta_topk_recall` CI excludes 0 unfavorably.
- **Kill (no efficacy):** all delta CIs straddle 0 ⇒ verdict "NON-HARMFUL, no
  efficacy"; B13 stays SMOKE-VALIDATED, the H2 lane closes.
- **Harmful:** any accuracy/topk CI excludes 0 unfavorably ⇒ demote that
  curvature.
- **Interpretation gate (not a quality verdict):** median `one_loop_top1_delta`
  at budget 64 must be below budget 8's, else "finite-N curvature" is demoted to
  "diagnostic reweighting" (honors the corrected non-monotone claim).

### voc_tightness (P3 bonus — measurement, not a claim)

H1 continuation early-stop gives positions different realized budgets ⇒ first
real `forked_voc.measure_tightness(bundles, realized_budgets)`. Report Spearman
ρ; if realized budgets collapse to one value it returns None and P3 stays
unmeasured — recorded honestly, no claim either way.

## Constraint (Stage 5)

No Stage 7 claim may cite adaptive VL duplication reduction as a rationale
(Stage 5 killed it: neither the synthetic screen nor the real engine supports
"adaptive VL lowers dup_rate"; adaptive VL's engine-default basis is the
measured ~6× virtual-loss pessimism reduction at preserved agreement).

---

## Per-commit audit log

### C0 — pre-registration + Danihelka closure

- Wrote this file (pre-registration record).
- CLAIM_LEDGER: Danihelka guarantee-restoration row → DEPRECATED with the
  Stage-3 cancel reason above; added Stage 7 SPECIFIED rows referencing this
  file's kill/success criteria; added the adaptive-VL-dup constraint row.
- No code touched; no regression run required (docs only).

### C1 — `KgStop` `SearchPolicy` wrapper

- `src/mcts/policy/kg_stop.rs`: appended `KgStop` (struct + `impl SearchPolicy`)
  and `KgCostSource` below the existing tested primitives. Wraps
  `compute_kg_array` / `should_halt_by_kg` exactly as `KLLUCBStop` wraps its KL
  helpers: `parking_lot::Mutex<KgCache>`, heavy work in `observe`, O(1)
  `should_halt`, identity `score_adjustment`, `telemetry` maps `max_kg →
  bayes_voi`.
- Design points honoring the exploration facts: best arm derived from edge `q`
  (gated by `min_pulls`, fallback argmax `n`), **never** the stubbed
  `snap.best_idx`; per-arm variance = `EdgeView::sigma_a(lambda0)²` (shrinks an
  unvisited arm toward `sigma_q_root²`); `cost_per_pull_ms =
  elapsed_ms/iteration` (measured, no fitted constant — FORBIDDEN-safe);
  `!observed ⇒ Continue` guard (R3: default cache must not spuriously halt);
  halt reason = reserved `HaltReason::PolicyConverged`.
- `src/mcts/policy/mod.rs`: export `KgStop`, `KgCostSource`.
- Tests (9 `test_s7_kg_stop_*`): identity adjustment, no-halt-before-observe,
  below-min-total, max-visits, resolved-root-halts-PolicyConverged,
  underpulled-arm-blocks-halt, derives-best-ignores-stub, telemetry, clamp.
- Regression: `cargo test --bin mcts_demo kg_stop` 24/24 (15 primitive + 9 new);
  full `cargo test --bin mcts_demo` 562 passed / 89 ignored / 0 failed.

### C2 — `kg_stop` env-var registration

- `src/mcts_server.rs`: added a `"kg_stop"` arm to the `QUARTZ_SEARCH_POLICY`
  dispatch mirroring `kl_lucb_stop` — `KgStop::default_for_budget(max_visits)`
  with optional env overrides `QUARTZ_KG_THRESHOLD` (f32) and
  `QUARTZ_KG_MIN_TOTAL` (u32); prints the attach line with resolved params.
  Updated the unknown-name WARN's expected-name list to include `kg_stop`.
  Resolves the docs-vs-reality gap: the mechanism is the env var, not a
  `--policy=` argv flag (no JSON key threaded — out of scope).
- Regression: `cargo build --bin mcts_demo` clean (exit 0).

### C3 — KG-stop engine-integration tests

- `src/mcts/mod.rs` (test module, beside `EdgeSpyPolicy`): two tests.
  - `test_s7_kg_stop_engine_halts_before_budget_on_resolved_root`: attach
    `KgStop::new(1000.0, …, min_total=20, Fixed(1.0))` to
    `MctsConfig::evaluation(2.0)`, Gomoku7 + `UniformEval`,
    `FixedIterations(400)`; asserts `iterations < 400` AND
    `policy_halt_count_snapshot()[PolicyConverged] > 0` — proves the wrapper
    halts the *real* engine through the policy path (observe fires at iter 64
    in non-quartz, min_total met, permissive threshold ⇒ deterministic halt).
  - `test_s7_kg_stop_engine_respects_min_total`: `min_total=300 >
    FixedIterations(200)` ⇒ `iterations == 200`, PolicyConverged count 0.
- Regression: both pass; full `cargo test --bin mcts_demo` 564 passed / 89
  ignored / 0 failed.

### C4 — KG-stop engine smoke script

- `scripts/kg_stop_engine_smoke.py`: pure `summarize_kg_smoke(rows)` (per-cell
  halt_rate / mean_budget_saved / top1_agreement + the pre-registered
  kill/success/demote flags) and the live `run_kg_smoke(...)` grid. The grid
  reuses `FrozenCheckpointHarness` but calls `client.search_move` directly to
  capture `iterations` (the harness row drops it); forces
  `check_interval=max(4,budget//8)` via a minimal `Phase15System`; sets
  `QUARTZ_SEARCH_POLICY`/`QUARTZ_KG_THRESHOLD` before constructing the harness
  so the lazily-started Rust server inherits them; runs a fixed-halt baseline
  with the env cleared; pairs on `(position, budget)`. Positions via
  `load_or_generate_positions`, base cfg via `controller_sweep.build_base_cfg`.
- Tests (`tests/test_kg_stop_smoke_summary.py`, 4): success cell, kill (no
  halts), demote (anti-conservative), grouping + best cell.
- The live grid runs at E3 (needs a trained checkpoint + GPU). Regression:
  4/4 pytest, `py_compile` clean.

### C5 — H1 online halt wiring (B14)

- `quartz/phase15_ablation.py`: added `"argmax_stability_stop"` to
  `POSTHOC_OPERATORS`; `apply_argmax_stability_readout` (final-snapshot policy —
  H1 is a halt rule, never transforms the policy — plus Dirichlet
  argmax-stability metadata) + `argmax_stability_stop_params` helper; dispatch
  branch; **B14** `Phase15System` (`refresh_operator="argmax_stability_stop"`,
  `search_overrides=a4` ⇒ shares the A4/B13 trace, `execution_mode="online"`,
  params threshold 0.9 / min_visits 8 / alpha 0.5 / n_boot 4000);
  `PHASE15_PARTB_SYSTEMS = ("B13", "B14")`.
- `quartz/phase15_online.py`: 3rd decision point in `run_online_readout` — for
  `argmax_stability_stop`, at each sub-target chunk compute
  `should_stop_by_argmax_stability(counts_from_policy(π_b, b))`; on stop
  early-return with `decision_notes=["h1_stop@b"]` and `online_stop_budget=b`.
- `scripts/phase15_online_ablation.py`: `run_online_readout_continuation` gains
  `early_stop_fn` — when it fires the resident session is not stepped further
  (real compute saved, and positions realize genuinely different budgets, which
  feeds the voc_tightness P3 bonus); `_early_stop_predicate(system)` builds the
  H1 predicate; `build_online_trace_bundle` passes it.
- Tests: +5 (B14 registration + A4-signature share, B14 readout metadata /
  policy-unchanged, H1 stops-early on stable trace, H1 continues on unstable,
  continuation `early_stop_fn` prevents later steps). Count tests updated for
  B14 (PARTB/FULL). Regression: phase15 + argmax_stability suites 136 passed;
  compileall clean.
