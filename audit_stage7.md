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

### C6 — trace p_flip channel (single schema-touching commit)

- `quartz/phase15_trace.py`: `TRACE_CACHE_SCHEMA_VERSION` 4→5;
  `build_trace_artifact(..., trace_p_flips=None)` stores `trace_p_flips`
  (None-padded, one per budget — back-compat: pre-C6 bundles omit the field).
  The schema bump + `phase15_trace.py`'s presence in `TRACE_CACHE_RELEVANT_PATHS`
  auto-flips the cache salt exactly once (R1); Stage 7 uses a fresh
  `--trace-cache-dir`.
- `scripts/phase15_ablation_study.py`: `build_search_trace` captures
  `row["p_flip"]` per chunk and passes `trace_p_flips` to `build_trace_artifact`,
  so cached trace bundles carry the engine's own incumbent P_flip aligned with
  each policy — the substrate the flip-calibration lane (C8) reads directly
  (same approach as `forked_voc_lab` reading trace-cache bundles).
- `quartz/phase15_online.py`: `run_online_readout` collects per-chunk
  `trace_p_flips` and includes it in all four return metas.
- Tests: new `tests/test_phase15_trace.py` (3: records p_flips + schema bump,
  None-pad back-compat, cache roundtrip) + `test_run_online_readout_meta_
  includes_trace_p_flips`. Regression: phase15 + trace suites 147 passed;
  compileall clean.

### C7 — H3 two-signal entropy-burst operator (B15)

- `quartz/phase15_ablation.py`: factored `apply_budget_routing`'s
  burst-fetch-and-select body into `_apply_routing_with_signal` (shared by both
  routing operators — they now differ ONLY in the instability rule). New
  `h3_burst_signal` = the 2-signal gate `(ΔH_root > entropy_floor on a
  Dirichlet-smoothed posterior with a min-visit floor) AND (top-2 margin
  shrinking < −margin_slope_floor)`, root/search-stats only (game-agnostic).
  New `apply_entropy_burst_routing`. Registered operator
  `"entropy_burst_routing"` + dispatch; **B15** `Phase15System`
  (`search_overrides=a4`, online, smooth_alpha 0.5 / floors 0 / min_visit_floor
  8); `PHASE15_PARTB_SYSTEMS = ("B13","B14","B15")`.
- `quartz/phase15_online.py`: generalized the online burst branch to
  `_ROUTING_OPERATORS = (budget_routing, entropy_burst_routing)` via
  `_routing_burst_signal` dispatch; the supra-target chunk's p_flip is appended
  to `trace_p_flips` too.
- Tests +4 (B15 registration + A4 signature, h3 2-gate requires BOTH signals,
  B15 readout burst fields, online H3 burst fetches supra-target + logs
  `burst@16->32`); count tests updated for B15. Regression: phase15 suites 144
  passed; compileall clean. Kill (O6 lift CI vs 1) pre-registered for E2/C9.

### C8 — flip-calibration analyzer

- `scripts/phase15_flip_calibration.py`: reads schema-≥5 trace bundles (with
  `trace_p_flips` from C6); per sub-target budget computes `s_H1 =
  argmax_stability(counts(π_b,b))`, `s_Pflip = 1 − p_flip_b`, and `y =
  1[argmax(π_b)==argmax(π_holdout)]`. `reliability_diagram` (10-bin + ECE +
  Brier, descriptive); `virtual_stop_budget` (replay a stop rule over the
  recorded chunk boundaries); `matched_budget_calibration` — the CONFIRMATORY
  statistic: paired argmax-agreement delta (H1 − P_flip) at the P_flip threshold
  whose mean realized budget matches H1's within ±5%, `paired_bootstrap_ci`
  (2000, seed 0). `h1_dies` iff the CI excludes zero in P_flip's favor.
- Tests (`tests/test_phase15_flip_calibration.py`, 7): reliability/ECE + Brier
  hand-computed, virtual-stop budget accounting, holdout excluded from decision
  records, matched-budget h1_dies (P_flip agrees more at matched budget) and
  h1_survives (tie), analyze end-to-end. All pass; py_compile clean.
- Live run is E2 (needs Stage 7 online trace bundles).

### Note — training timeline (Lane T)

Measured throughput on the real 200-games/iter gomoku7 loop is ~8 min/iter
(iter 4 at ~33 min), so 3 seeds × 20 gens ≈ ~8 h — the ~2 h plan estimate came
from the B13 toy-capped smoke (~116 s/gen) and is wrong for the full loop. Code
lanes C1-C10 are independent of training and proceed regardless; the experiment
scope (single-seed mid-training vs full multi-seed) is a timeline decision
raised with the user before E1-E3. **User chose: reduce to 3 seeds × 8 gens
(~3.2 h); relaunched unbuffered.**

### C9 — O6 burst-precision analyzer

- `scripts/phase15_o6_burst_precision.py`: joins B15 burst events (online rows,
  `budget_burst_triggered`) with the external difficulty label `hard :=
  forked_voc.final_overturns_shallow` on the shared A4 trace bundle, keyed by
  `(checkpoint_id, position_id)`. `compute_o6_lift` = lift `P(hard|burst)/P(hard)`
  with a position-level bootstrap CI (seed 0); pre-registered kill = CI includes
  1; degeneracy demotion = burst rate >0.9/<0.02 or <30 events.
  `build_records` ORs per-budget rows for the burst flag and excludes (counts)
  rows whose bundle is missing — never a silent match. Non-circular: the label
  uses the full ladder above the decision point.
- Tests (`tests/test_phase15_o6_precision.py`, 5): lift alive when burst tracks
  difficulty, kill when burst fires at the base rate, degeneracy (too-few
  events / saturated burst rate), join with missing-bundle exclusion. All pass;
  py_compile clean. Live run at E2.

### C10 — research-grade gate ported to phase15

- `quartz/phase15_research_grade.py`: encodes the CLAIM_LEDGER Ablation Start
  Conditions as checkable functions — `check_seed_families` (>= N distinct
  `seed_<n>` families), `check_paired_coverage` (identical
  `(checkpoint,position,budget)` set per system), `check_single_salt`,
  `check_artifact_hashes` (manifest sha256 for every checkpoint + positions +
  config), `check_rows_preserved` (row count = ckpt×pos×budget×system) + the
  A2-b interpretation-flags presence. `check_research_grade` aggregates;
  `enforce_research_grade` raises SystemExit with the unmet list.
- `scripts/phase15_ablation_study.py`: `--research-grade` + `--min-seed-families`;
  a fail-fast seed-family precheck after checkpoint resolution (before the
  expensive run).
- `scripts/phase15_analyze_results.py`: `--research-grade` + `--manifest`; runs
  the full gate on the actual rows/manifest/report and enforces at analysis time.
- Tests (`tests/test_phase15_research_grade.py`, 5): seed-family parse/count,
  paired coverage equal/mismatch, compliant + failure (missing hash, too-few
  families) with enforce raising, rows-preserved drop detection. All pass;
  compileall clean across both scripts + the module.

### E-prep — harness honors checkpoint net architecture

- Trained gomoku7 checkpoints are 96f/6b (the training default), while the
  ablation profiling default (`build_base_cfg`/`GAME_CONFIGS`) is 64f/4b, so the
  checkpoints would not load into the harness net. Fix (`7dcad72`):
  `FrozenCheckpointHarness` reads each checkpoint's stored `cfg`
  (`_read_checkpoint_cfg`) and overrides `filters/blocks/vh/ch` before building
  `AlphaZeroNet`; each checkpoint has its own harness, so different-sized
  checkpoints can be mixed in one run. Verified loading + running the real
  96f/6b checkpoint; 97 phase15/kg-smoke tests still green.

### E3 / C11a — KG-stop engine smoke verdict: lane CLOSED at low budgets

- Run: `seed_101/latest.pt` (gen_8, 96f/6b), 32 positions × budgets {64,128,256}
  × kg_threshold {1e-4, 1e-3, 1e-2, 0.1, 1.0} (last two diagnostic, to locate the
  halt regime), `QUARTZ_SEARCH_POLICY=kg_stop` vs env-unset baseline.
- **Result: 2 halt events across 480 cells; max 0.6% budget saved.** The
  pre-registered Success (≥20% saved @ ≥0.95 agreement) is NOT met — the rule
  effectively never fires. `max_kg` stays above `kg_threshold·cost_per_pull_ms`
  even at threshold 1.0: a formally-correct KG-stop certificate near-unreachable
  at 8-256 visits — the SAME low-budget-unreachability as KL-LUCB (A1-a). The
  synthetic Stage-1 KG-**allocation** green did not transfer to the KG-**stop**
  rule on adaptive shared-tree backups (different mechanism).
- Honesty note: the top-1 agreement (~0.5) is confounded by cross-process MCTS
  nondeterminism (halt rate ~0 yet agreement <1), so it is not diagnostic; the
  no-savings verdict rests on the halt/iteration count, which is robust.
- Disposition: KG-stop wrapper stays IMPLEMENTED + SMOKE-VALIDATED (wired, halts
  on a resolved synthetic root per the engine tests, selectable via the env
  var); the low-budget efficiency claim is NOT earned. CLAIM_LEDGER row updated.

### E2-prep — bundles self-identify + short position id

- Trained checkpoints are 96f/6b; bundles now carry `checkpoint_id`/`position_id`
  (schema 5→6, `47885db`) so the O6 join keys line up. build_search_trace stores
  the SHORT position id (matches the analysis rows' `position_id`); the full
  `_position_key` still keys the cache. The 288 existing bundles were patched
  in place to the short id.

### E1 posthoc run (serves C11b + C11d)

- `phase15_ablation_study.py --systems A4,B5,B13,B13c025,B14,B15 --checkpoints
  <6: seed_{101,102,103}/{gen_5,latest}> --budgets 8,16,32,64 --oracle-budget 256
  --suite-size 48 --research-grade`. Produced 288 shared trace bundles (all
  systems share the A4 signature) + 6912 rows. The `--research-grade`
  seed-family precheck passed (3 families).

### C11b — H1 flip-calibration: calibration WIN, confirmatory insufficient

- `phase15_flip_calibration.py` on the 288 bundles (864 decision records).
- **Reliability: H1 stability ECE 0.080 / Brier 0.143 vs P_flip (1−p_flip) ECE
  0.504 / Brier 0.446** — H1 predicts held-out argmax agreement far better than
  the incumbent P_flip.
- **Confirmatory matched-budget: INSUFFICIENT** at every H1 threshold (0.5-0.9):
  H1 is conservative (mean realized budget ~29-32) while P_flip is
  degenerate-over-eager (p_flip = 0 at budget 8 ⇒ immediate stop), so no P_flip
  threshold matches H1's realized budget within ±5% and no paired comparison
  forms. The kill therefore CANNOT fire ⇒ **H1 is NOT demoted**.
- Disposition: H1 survives; its stability is the better-calibrated signal, but
  at 8-32 budgets neither stop saves budget (H1 too conservative, P_flip
  degenerate). Recalibrate the H1 stop threshold / go to higher budgets before
  wiring a real online halt.

### C11c — H3/O6 burst precision: degeneracy demotion (gate never fires)

- `phase15_o6_burst_precision.py` on the posthoc B15 burst rows + forked_voc
  labels from the 288 bundles.
- **0/288 bursts** (burst rate 0.0) at the default floors (0.0) ⇒ pre-registered
  degeneracy demotion (<0.02). O6 lift is unmeasurable. The forked_voc
  difficulty labels are HEALTHY (p_hard = 0.79) — only the burst TRIGGER never
  engages. H3 remains wired-but-unproven; recalibrate the floors before any O6
  claim.

### C11d — B13 research-grade: CI-separated KL improvement (positive)

- `phase15_analyze_results.py` on the 6912 posthoc rows (Bonferroni paired CIs).
- **B13 (curvature 1.0): delta_kl_to_oracle = −0.0297, CI [−0.033, −0.026]
  excludes 0 favorably; B13c025 (0.25): −0.0104, CI [−0.0115, −0.0093]** — both
  vs A4 AND B5, with delta_accuracy and delta_topk EXACTLY tied (1152/1152). B13
  reshapes the full-policy distribution toward the oracle (lower KL) WITHOUT
  changing any decision — larger at curvature 1.0. This FLIPS the random-init
  +0.47 KL harm and the gen-5 KL-neutral smoke. Meets the pre-registered
  efficacy bar; formal VALIDATED needs the full research-grade artifact-hash gate
  (only the seed-family precheck ran). Decision-neutral (KL-only) readout
  improvement, not a play/P2 claim.

## C12 — Stage 7 closure (conditional gate table)

Each Stage 7 conditional-table row closed with its executed disposition + earned
tier. Training manifest (6 checkpoint sha256s) at
`results/phase15_stage7/gomoku7/training/manifest.json`.

| Stage 7 task | Gate | Executed disposition | Tier |
|---|---|---|---|
| KG-stop `SearchPolicy` wrapper + `kg_stop` env registration | Stage 1 GREEN | BUILT + wired + engine-tested; E3 smoke shows it does NOT engage on real gomoku7 backups (2/480 halts, no savings) — low-budget-unreachable like KL-LUCB. Lane CLOSED at low budgets | IMPLEMENTED + SMOKE-VALIDATED (halts on resolved roots); efficiency NOT earned |
| H1 online halt + flip-calibration vs P_flip | Stage 3 GREEN | WIRED (B14, continuation early-stop); E2 shows H1 stability is far better CALIBRATED than P_flip (ECE 0.08 vs 0.50), confirmatory matched-budget INSUFFICIENT (both stops mis-tuned at 8-32) — H1 NOT demoted | SMOKE-VALIDATED (calibration win; stop threshold needs recalibration) |
| H3 backflow burst + O6 precision via forked_voc | Stage 2 GREEN | WIRED (B15, 2-signal gate); E2 shows 0/288 bursts at default floors → O6 unmeasurable (degeneracy demotion). Difficulty labels healthy (p_hard 0.79) | SMOKE-VALIDATED (wired); O6 unproven — recalibrate floors |
| B13 research-grade verdict | unconditional | RAN on trained 3-seed net: CI-separated KL-to-oracle improvement (Bonferroni), decisions unchanged. FLIPS the random-init harm | SMOKE-VALIDATED (efficacy direction; formal VALIDATED needs artifact-hash gate) |
| Danihelka ranking restoration | Stage 3 net-total = 0 | CANCELLED (C0), DEPRECATED — no evidenced payoff path | — |
| Constraint: no adaptive-VL dup claim (Stage 5) | — | In force through closure; not cited anywhere | — |

**Stage 7 theme.** Every wrapper/operator/analyzer works (code proven by unit +
engine tests). On real trained-net traces, the STOP/BURST mechanisms (KG-stop,
H1, H3) are calibrated too conservatively for the 8-64 budget regime and do not
engage — a consistent, honest negative that mirrors the KL-LUCB A1-a history.
The two POSITIVE Stage-7 results are offline/readout: **B13's one-loop readout
lowers KL-to-oracle (CI-separated) on a trained net**, and **H1's stability
signal is dramatically better-calibrated than the incumbent P_flip**. No result
was promoted above SMOKE-VALIDATED; the synthetic-lab tiers (Stages 1-6) remain
pinned at `synthetic_screening`.
