# QUARTZ Integrated Audit — 2026-04-25

**Scope:** repository-wide re-audit against `audit.md`, with verification that prior findings (`audit_codex_20260422.md`, `audit_codex_20260423.md`) still hold against the current tree (Phase 6 commits landed Apr 25), plus new mechanism-level findings.

**Methods:** read 60+ source files across `src/mcts/**/*.rs`, `src/mcts_server.rs`, `quartz/*.py`, `scripts/*.py`, `configs/*.json`, `docs/*.md`; cross-checked the three prior audit documents (`audit_codex_20260422.md`, `audit_codex_20260423.md`, `review.md`); traced executable paths from `python -m quartz.train` through SHM/QIPC into the Rust MCTS hot path; mapped controller insertion points against `docs/QUARTZ_THEORY.md`.

> **Bottom line up-front.** This is a real, runnable research platform with unusually strong regression coverage for an academic codebase. The pipeline closes; the engine is non-trivial; the docs are remarkably self-honest about downgrades. The structural problems are not "nothing works" — they are: (1) the controller is a six-mode heuristic family attribution-coupled to halt logic via shared inputs, (2) several pipeline contracts (best.pt seeding, replay→SGD threshold, actor freshness gating, eval engine spec) silently corrupt short ablation runs, (3) the test suite covers components but has no end-to-end "loss actually decreases" assertion. All are tractable with bounded patches.

---

## 1. Project Claim Reconstruction

| # | Claim | Provisional tag | Note |
|---|-------|----|----|
| C1 | Rust MCTS engine is real and load-bearing | **clearly implemented** | ~15 KLOC across `src/mcts/`; 386+ Rust tests pass |
| C2 | Rust search is the actual training/eval substrate (not toy fallback) | **clearly implemented** | Confirmed in `quartz/selfplay_runtime.py:1630-1680` and `evaluator_runtime.py:945+` |
| C3 | End-to-end AlphaZero loop closes (selfplay→replay→SGD→eval→promotion) | **partially implemented** | Loop closes structurally, but SGD does not fire on smoke parameters; actor freshness is replay-gated not iteration-gated |
| C4 | QUARTZ controller is implemented in production | **clearly implemented** | Six penalty modes dispatch in `src/mcts/select.rs:175-431` |
| C5 | QUARTZ is one theory-aligned design ("Q-value Uncertainty-Adaptive Root-risk Tree search, Zero-tunable") | **overclaimed** (the docs themselves admit) | `docs/QUARTZ_THEORY.md:21-33` explicitly downgrades to "a family of state-driven controllers plus explicit search hyperparameters, not a hyperparameter-free law" |
| C6 | Controller modes are ablation-ready | **partially implemented** | Modes exist; observability of which mode/halt reason fired is silent (no telemetry emit) |
| C7 | Same-stack evaluation (Rust+NN, no toy fallback) | **clearly implemented** | `RustNNEvaluatorEngine` is the production path |
| C8 | Multi-game support (Gomoku 7/15, Go, Chess, TicTacToe) | **clearly implemented** | All five games in `src/games/`, encoders in `quartz/encoders.py` |
| C9 | ONNX/Gomocup deployment path | **partially implemented** | Code real, README admits "still needs environment-specific verification" |
| C10 | Phase15 clean-split tooling | **clearly implemented** | Three runners + benchmark gate in `scripts/phase15_*.py` |
| C11 | Same-budget / same-NN ablation harness | **partially implemented** | `halt_trace` enables post-hoc audit; not enforced ex ante; eval search_profile differs across rows |
| C12 | Reproducibility (seeds, contract hashes) | **partially implemented** | `runtime_contract_hash` exists; no determinism-regression test |
| C13 | "Recent updates" docs are honest | **honest with caveats** | README explicitly disowns several deployment claims; this is a positive signal |
| C14 | Canonical smoke certifies training readiness | **overclaimed if read literally** | `scripts/smoke_e2e.py` is a runtime smoke; under default smoke params replay never reaches batch threshold → zero SGD rows |
| C15 | Phase 6 perf wins (make-unmake, RwLock TT) | **clearly implemented and isolated** | Verified Apr 25 commits do not touch controller invariants |

**Independent status summary:**
- End-to-end executability: **good** (runs to completion)
- Controller state semantics: **rich but coupled** (penalty + halt share σ_Q, P_flip)
- MCTS modification scope: **load-bearing at root**, optional shallow-blend exists but is off by default
- Evaluator/training coupling: **real but loosely contracted** (actor refresh, best.pt seeding, eval engine spec)
- Ablation readiness: **decent for engineering iteration, weaker for clean causal attribution**
- Hardware realism: **moderate** — torch-on-ROCm with documented hipBLASLt degradation; JAX self-play path is stub
- Docs/tests honesty: **above average** with three specific overclaims (smoke certification; configs claiming "best for X"; controller-as-unified-design narrative)
- Reproducibility: **above average** (manifest hashes, runtime contracts) but no determinism gate test

---

## 2. Executable Path Reconstruction

### 2.1 Actually-wired path (verified)

```
quartz.train.main()            quartz/train.py:92
  → torch_runtime.main()       quartz/torch_runtime.py:5 (re-export)
  → torch_training_runtime.main()  quartz/torch_training_runtime.py:378
  → cli_main.run_training_main()   quartz/cli_main.py:576
       ├── best.pt seeded if missing       cli_main.py:721-730   ← FAULT (§5.W4)
       ├── SelfPlayWorker.start()          cli_main.py:769-770
       │     └── selfplay_rust_nn_batched()  selfplay_runtime.py:1630
       │           └── Popen(target/release/mcts_demo --server …)
       │                 ├── QIPC handshake (stdin JSON config + magic="QIPC")
       │                 ├── SHM ring (magic=0x51524E47, version=1)  qipc.py:444-516
       │                 └── Rust loop in src/mcts_server.rs (6805 LOC)
       │
       ├── for iteration in args.iterations:
       │     ├── wait_for_worker_progress()              cli_main.py:857
       │     ├── if len(replay) >= cfg["batch"]:         cli_main.py:882   ← GATE (§5.W6)
       │     │     train_epoch(...)
       │     │     if executed_steps > 0:
       │     │         bg_worker.update_model(actor)      cli_main.py:956-957  ← FAULT (§5.W5)
       │     └── if eval_due: training_evaluator.evaluate_checkpoint()
       │           └── PersistentRustNNEvalCampaign       evaluator_runtime.py:1163
       │                 → RustServerPool per engine     ← row-specific cfg (§5.W7)
       │                 → Glicko-2 update + promotion    evaluation.py:733-758
       │                 → if "promote": save best.pt    cli_main.py:1108-1119
       │
       └── bg_worker.stop()                              cli_main.py:1168
             → 10s join, then kill_active(), then 5s join, then warn  selfplay_runtime.py:2205-2218
```

### 2.2 Path-vs-claim deltas

| Claim | Code path | Reality |
|---|---|---|
| "Self-play actors are refreshed after every learner iteration" | `cli_main.py:956-957` requires `executed_steps > 0` | Refresh fires only when SGD ran. SGD only fires when replay ≥ batch. Smoke ablations never satisfy this → actor stays at gen 0. |
| "Smoke certifies training readiness" | `scripts/smoke_e2e.py` defaults: 1 iter, 4 games | Replay accumulates ~96 positions; batch=256 never reached; 0 SGD rows; warning logs |
| "Ablation studies use the same NN" | `ablation_study.py:1156-1166` rebuilds `cfg` per eval row | Each row uses its own `search_profile` and `vl_mode` for evaluation; comparing rows compares (model × eval-engine), not model alone |
| "Rust is the sole training search engine" | True for selfplay/eval | But Python fallback evaluator exists in `ablation_study.py` with documented warning; not used by default |

### 2.3 Genuinely runnable vs merely described

**Genuinely runnable end-to-end:**
- `python -m quartz.train --game gomoku7 --iterations 30` (with sensible config that crosses batch threshold)
- `cargo test --release` (386 tests pass)
- `pytest -q tests/` (per prior audit: 254 tests pass)
- `scripts/ablation_study.py --study search_vl --game gomoku7 ... --iterations 2` (artifacts emit, but training may not fire on too-short runs)
- Rust standalone binaries from `src/ablation_*.rs` and `src/experiment_*.rs`

**Described but partially closed:**
- `scripts/smoke_e2e.py` certifying training (it doesn't — it certifies imports/transport)
- `controller_axes` preset isolating one factor at a time (it does configurationally, but no test asserts the isolation property end-to-end)
- "AlphaZero-style" learner (truly closed once §5.W4-W6 are fixed)

**Code present but not on a hot path:**
- JAX self-play inference: `quartz/jax_runtime.py:1-5` is essentially a stub; README:272-276 admits "self-play / eval inference … still flows through torch path".
- Shallow-blend (`root_only_shaping=false`) at `depth ≤ 3`: code exists at `src/mcts/select.rs:542-574`; default is true; recent ablations don't toggle it.
- `prior_q_divergence` field: computed at `src/mcts/quartz.rs:1674-1681`; read by `GatedRefresh` but not by `PFlipMixture` despite the latter's narrative.

---

## 3. Search Controller / MCTS Architecture Reconstruction

### 3.1 Controller state variables

The controller runtime state divides into four groups:

| Group | Where | Hot-path read? |
|---|---|---|
| Static config (`QuartzConfig`) — sigma_0, min_visits, ctm_budget_ms, halt_mode, penalty_mode, hbar_penalty_cap, prior_refresh_rate, prior_refresh_temp, root_only_shaping, ns_gamma, check_interval, enable_* flags | `quartz.rs:199-268` | Yes — read on every edge in `score_snapshot()` |
| Per-check stats (`QuartzStats`) — hbar_eff, p_flip, sigma_delta, rho_hat, sigma_q, prior_q_divergence, surprise_kl, epsilon_t, voc_total/focus/expand/merge, converged, flip_stable | `quartz.rs:341-416` | Yes — read in selection (penalty/refresh) and halt (`should_stop`) |
| Inner state (`QuartzController`) — last_stats, last_check_at, stop_reason, theta_conf, s0_same/global/baseline, elapsed_ms | `quartz.rs:1618-1668` | Mostly written; `last_stats` and `theta_conf` re-read for halt decisions |
| Ephemeral per-edge (`MctsEdge`) — n_raw, virtual_losses, q_eff, prior, edge_sigma (M2), rtt_n, rtt_var | per-edge | Yes — every selection step |

### 3.2 Mode system: six penalty modes × four halt modes × refresh-temp × shallow-blend

`src/mcts/select.rs:193-431` dispatches on `qcfg.penalty_mode`:

| Mode | Penalty term | Refresh law | Hot-path inputs |
|---|---|---|---|
| `Legacy` | `-min(ħ_eff, 0.3) / N_a` (clamped, with off-diagonal B₁loop bonus when heavy-tail or p_envar≥0.2) | none | hbar_eff, edge_sigma |
| `EffectiveV2` | `-ν / (1+N_a+O_a)` (includes virtual losses) | none | hbar_penalty_cap |
| `None` | 0 (pure PUCT baseline) | none | — |
| `SelfAdaptive` | `-σ_Q / (1+N_a)` plus dynamic per-action α_a from visits | computed inline | sigma_q, n_visible, root_visits |
| `GatedRefresh` | root-share penalty | gate on `prior_q_divergence > epsilon_t`; `ρ_t = (D-ε_t)/D` | hbar_eff, hbar_penalty_cap, prior_q_divergence, epsilon_t, root_visits |
| `GatedRefreshLegacy` | `effective_penalty_v2(...)` | P_flip-gated: `ρ_t = ρ_max · min(P_flip / 0.159, 1)`; Q-refresh with τ from `prior_refresh_temp` (fallback **hardcoded 0.5**) | p_flip, hbar_penalty_cap, prior_refresh_temp |
| `PFlipMixture` | `ν = max(cap, σ_Q)` then `-ν/(1+N_a+O_a)` | mixes Q-refresh and VF-refresh by `p_ratio = min(P_flip/0.159, 2)`; **τ fallback also hardcoded 0.5** | sigma_q, hbar_penalty_cap, p_flip, prior_refresh_temp |

`src/mcts/quartz.rs:1862-1959` dispatches on `qcfg.halt_mode`:

| `HaltMode` | Stop condition | Note |
|---|---|---|
| `Fixed(N)` | `root_visits ≥ N` | Only mode that's truly orthogonal to penalty mode |
| `SimpleThreshold` | `P_flip < 0.159` AND `flip_stable ≥ 3` | Ignores VOC |
| `VOC` (default) | `converged` AND (`voc_total ≤ 0` OR `P_flip < 0.159`) | Dual-gate; later signal wins |
| `ConfAdaptive` | `Conf(t) = (1-P_flip)(1-P_hidden)·max{0, 1-S/S₀} ≥ θ_conf`; θ adapts online | Single position adapts θ globally → leakage |

### 3.3 Insertion points

**Selection** (`src/mcts/select.rs:605-748`):
- Per-edge scoring through `score_snapshot()` at line 697; runs the entire `ablation_puct_score_with_parent_sqrt()` mode-dispatch (lines 175-431). Every penalty-mode branch reads stats fields whether or not its mode is active (the stats struct is fully populated regardless).
- Optional shallow blend at `depth ≤ 3` (line 542) only when `root_only_shaping=false`. **Default is true; this code path is dormant on the main configs.**

**Expansion** (`src/mcts/expand.rs`, `src/mcts/quartz.rs:973-997`):
- VOC_EXPAND is computed for accounting but **does not gate actual leaf materialization**. The "expand channel" is a halt-input signal, not an expansion-policy modifier.

**Backup** (`src/mcts/backup.rs`):
- Pure Welford M2 + RTT correlation update. **No QUARTZ-specific writes.** Controller reads these statistics one level up, not in the backprop itself.

**Root policy emission** (`src/mcts/root.rs:122`):
- `policy(a) = N_a / N_total`. **Visit-count based, not score-based.** Penalty shapes which actions are visited; the visit distribution carries that shaping into the training target. There is no "back out the penalty" for emission — by design.

**Halt** (`src/mcts/quartz.rs:1862-1959`):
- Single function `should_stop()`; returns `bool` and stamps `self.stop_reason` (silent — no log, no metric, no QIPC emit).

### 3.4 Provisional design classification

Not unified theory; not pure heuristic bundle. **Layered controller with ad-hoc mode surface.** Three layers:
1. **Statistics layer** — σ_Q, P_flip, RTT correlation, divergence, VOC channels — defensible and observable.
2. **Penalty layer** — six dispatch branches over the same stats — internally inconsistent (Legacy uses ħ_eff clamp, SelfAdaptive uses raw σ_Q, others use `hbar_penalty_cap`).
3. **Halt layer** — four modes, default VOC dual-gates `flip_stable ∧ voc_total≤0`, intermediate states under-defined.

**Critical coupling:** σ_Q and P_flip are inputs to penalty (selection scoring) AND to halt (termination). Changing penalty mode shifts P_flip distribution shifts effective halt step shifts effective compute budget. This makes "controller on/off" comparisons inherently confounded with "compute budget on/off" unless `HaltMode::Fixed(N)` is pinned.

---

## 4. Design Strengths (≤7)

1. **Real Rust+NN production path.** Not a paper claim — `RustNNEvaluatorEngine` is the actual evaluator and `selfplay_rust_nn_batched` is the actual self-play inference path. Strong against academic-codebase norm.
2. **Controller insertion points are localized.** Penalty/refresh in `src/mcts/select.rs:175-431`; halt in one function. Easier to ablate than a controller spread across the search loop.
3. **VOC accounting is principled.** Three channels (FOCUS / EXPAND / MERGE) with explicit cost-vs-benefit terms is a real attempt at optimal-stopping decision theory, not a magic threshold.
4. **Strong test surface for components.** 386+ Rust tests across `src/mcts/` plus 254 Python tests — unusually thorough numerical/contract coverage for a research repo.
5. **Self-honest documentation.** README and ABLATION_GUIDE explicitly disown several claims ("engineering signals, not multi-seed publication-grade"; "T0010_cf38467f … is not re-runnable"; "prior refresh remains implemented … but is not the current default/deployment recommendation"). This is rare and deserves credit.
6. **Phase15 clean-split tooling.** Frozen-checkpoint post-hoc evaluation infrastructure in `scripts/phase15_*.py` and `quartz/phase15_*.py` is genuinely useful for attribution work.
7. **Reproducibility scaffolding.** `study_manifest.json`, `evaluation_matrix.json`, `runtime_contract_hash`, `search_manifest_hash` exist and are checked. Above-average for the field.

---

## 5. Structural Weaknesses (≤10) and Failure-Mode Analysis

### W1. Controller is a six-mode heuristic family conflated under one acronym
**Mechanism:** `src/mcts/select.rs:193-430` dispatches six fundamentally different scoring laws (Legacy, EffectiveV2, None, SelfAdaptive, GatedRefresh, GatedRefreshLegacy, PFlipMixture). Each reads different stats and applies different math. Docs collapse them under "QUARTZ controller."
**Failure:** "QUARTZ vs baseline" reports change identity depending on which mode is configured.
**Why matters:** Cross-paper claim portability is broken — readers have no way to know which controller variant produced an Elo curve.
**Diagnostic:** Per-row `controller_id_hash = hash(penalty_mode, halt_mode, prior_refresh_*, root_only_shaping, sigma_0, hbar_penalty_cap, ctm_budget_ms)` printed in every artifact; reject ablation studies that vary this hash silently.
**Minimal fix:** ≤30 lines: emit `controller_identity` in `replay_search_summary` and assert constancy across rows of a single study.

### W2. Penalty / halt orthogonality breaks under default `HaltMode::VOC`
**Mechanism:** P_flip is computed from σ_Q and top-2 Q-values (`quartz.rs:644-692`). It is read by penalty (`GatedRefreshLegacy`, `PFlipMixture`) AND by VOC_FOCUS gain (line 941) AND by halt convergence (line 1920). Changing penalty mode shifts the visit distribution shifts the top-2 separation shifts P_flip shifts effective halt step.
**Failure:** Two ablation rows with same `max_visits=400` but different `penalty_mode` actually run different effective compute budgets.
**Why matters:** Controller-vs-baseline conclusions silently confound with compute. The README's `controller_axes` preset (which isolates `penalty_mode` between A2 and A3) is contaminated unless `HaltMode::Fixed(N)` is pinned.
**Diagnostic:** Per-row mean `root_visits_at_halt` and `elapsed_ms_at_halt`. If these vary >5% between adjacent rows of a single-factor preset, the comparison is invalid.
**Minimal fix:** In `controller_axes` preset, force `halt_mode = Fixed(max_visits)` for all rows, drop VOC halt for attribution runs, and document this explicitly. (~10 lines.)

### W3. Hardcoded `tau = 0.5` fallback in two penalty modes
**Mechanism:** `src/mcts/select.rs:285-289` and `src/mcts/select.rs:340`: if `config.prior_refresh_temp < 0.01`, `tau` defaults to `0.5`. Sweeping `prior_refresh_temp` from 0 produces a discontinuity at the threshold.
**Failure:** Optuna sweeps over `prior_refresh_temp` near 0 silently get τ=0.5 (the legacy literal). Reports "prior_refresh_temp had no effect at zero" when in reality the parameter is being clobbered by a fallback.
**Why matters:** Reported null effects of `prior_refresh_temp` are spurious. Already noted in `review.md`.
**Diagnostic:** Log effective `tau` once per game; assert `tau == config.prior_refresh_temp` when the latter is in `[0, 5.0]`.
**Minimal fix:** Replace `if temp < 0.01 { 0.5 } else { temp }` with `temp.max(1e-6)`. Two-line change.

### W4. `best.pt` seeded pre-training; never updated absent promotion
**Mechanism:** `quartz/cli_main.py:721-730` saves the untrained model to `best.pt` if missing. `quartz/cli_main.py:1108-1119` only updates on explicit "promote" verdict. `scripts/ablation_study.py:734-747` prefers `best.pt` for post-train evaluation.
**Failure:** Short ablations / `--eval-interval 999999` / smoke runs never promote → `best.pt` stays as random-init throughout the run → post-train evaluation, champion selection, and Gomocup export operate on the untrained model.
**Why matters:** Silently corrupts ablation conclusions. Already flagged in `audit_codex_20260423.md`; **still unfixed** as of HEAD.
**Diagnostic:** Read `cfg["best_checkpoint_bootstrap"]` from per-run metadata; reject any post-train evaluation row whose champion was never promoted.
**Minimal fix:** End-of-training fallback in `cli_main.py` (≤5 lines):
```python
if not saw_promotion:
    _save_model_checkpoint(backend, model, torch, latest_model_path, cfg)
    # And in ablation_study.py:resolve_model_path, prefer latest.pt
    # unless checkpoint_status.json reports `saw_promotion=True`.
```

### W5. Actor freshness gated on replay threshold, not learner iteration
**Mechanism:** `quartz/cli_main.py:956-957` calls `bg_worker.update_model(actor_source)` only when `executed_steps > 0`. SGD itself is gated on `len(replay) >= cfg["batch"]` (`cli_main.py:882`).
**Failure:** During warmup or with small `bg_batch_games`, replay never crosses threshold → no SGD → no actor refresh. Self-play runs on stale actor for many iterations. `audit_codex_20260423.md` reported "now per-iteration" — but the per-iteration condition is itself replay-conditional.
**Why matters:** Causal attribution of controller effects is corrupted by drifting actor freshness; ablation rows with different effective batch consumption see different actor drift schedules.
**Diagnostic:** Tag every replay sample with `actor_generation` (a monotonic counter incremented on `update_model`); print `actor_generation_distribution` per iteration to a log. `review.md` identified this; **still missing**.
**Minimal fix:** ≤10 lines in `selfplay_runtime.py:2170-2174`: tag samples; emit per-iteration histogram. The current `_actor_generation` field exists but is not exposed to replay tagging.

### W6. Smoke does not cross replay→SGD threshold; certifies imports, not training
**Mechanism:** `scripts/smoke_e2e.py` defaults: 1 iter, 4 games. Gomoku7 ≈ 24 positions/game → ~96 positions; default `batch=256` is never reached → 0 SGD rows.
**Failure:** "Smoke passed" implies training works. Smoke verifies transport, not learning. Already named in `audit_codex_20260423.md`; unchanged.
**Why matters:** False confidence; CI can't catch a training-loop regression.
**Diagnostic:** Add post-smoke assertion: `n_train_rows = sum(1 for line in log_file if json.loads(line).get("loss") is not None); assert n_train_rows > 0`.
**Minimal fix:** Either bump smoke's `--games-per-iter` to 16 (at gomoku7 → ~384 positions) or override `--batch=64` for smoke runs. ≤5 lines.

### W7. Eval engine spec varies across ablation rows
**Mechanism:** `scripts/ablation_study.py:1156-1166` rebuilds `cfg` per eval condition with that condition's `search_profile` and `vl_mode`. The same pair of models is evaluated under different search engines depending on which condition is the "evaluator" row.
**Failure:** "Model A wins" can mean "model A's training condition produced an evaluator profile that flatters its self-play distribution."
**Why matters:** The matrix conflates (model quality) × (eval search profile). Attribution to controller is impossible without freezing eval-search.
**Diagnostic:** Stratify the leaderboard by `eval_condition_search_profile`. If a model is "best" only under one profile, the result is profile-conditional, not absolute.
**Minimal fix:** Add `--frozen-eval-condition E1_baseline` flag (default true for `--study controller_axes`); use that single eval-search profile across all rows. ≤20 lines.

### W8. Halt-reason and penalty-mode telemetry is silent
**Mechanism:** `StopReason` is stamped into `self.stop_reason` (`quartz.rs:1871-1941`) but not logged, not serialized to JSON, not exposed via QIPC. Penalty modes never log entry/exit; refresh activations never increment a counter.
**Failure:** No way to verify "the controller actually halted by VocNonPositive vs BudgetExhausted" from artifacts. README mentions `halt_reason_hist` exposure in replay summaries — verified that this counter is fed only at one site and is partial.
**Why matters:** Without halt-reason and refresh-activation histograms, controller behavior cannot be falsified. Most "QUARTZ helps" claims rest on outcome metrics; the mechanism remains a black box.
**Diagnostic:** `halt_reason_hist`, `penalty_mode_active_count` (per game), `prior_refresh_activations` (per move), surfaced in replay search summary.
**Minimal fix:** ≤80 lines in `src/mcts_server.rs` and `quartz/replay.py`: emit a per-game telemetry struct, aggregate to histograms in summary, expose to JSONL.

### W9. CI does not gate on `pytest tests/` or `cargo test --release`
**Mechanism:** `.github/workflows/phase15-benchmark-gate.yml` runs only the deterministic phase15 smoke; primary test suites are not enforced upstream. Identified in `review.md` and `audit_codex_20260422.md`; **still unfixed**.
**Failure:** Regressions in selfplay/training/eval contracts can land without breaking CI.
**Why matters:** The 254 Python + 386 Rust tests are the project's strongest asset; not gating on them is leaving safety on the table.
**Minimal fix:** Add two stages to CI workflow: `pytest -q tests/` and `cargo test --release`. ≤15 lines of YAML.

### W10. No end-to-end "loss decreases" or determinism regression
**Mechanism:** No test in `tests/` runs ≥10 training iterations and asserts `final_loss < initial_loss`. No test runs (`seed=42`, capture loss curve, re-run, byte-equal compare). `tests/fixtures/regression_positions.json` exists but no integration test consumes it.
**Failure:** Architecture-level regressions that don't crash but degrade learning slip silently.
**Why matters:** The full claim of the README is "AlphaZero training loop." Component tests verify components; the training-loop claim itself has no behavioral regression.
**Minimal fix:** ≤200-line integration test: 5–10 iterations on gomoku7 with smoke-grade NN, assert (a) ≥1 SGD row emitted, (b) median policy loss strictly decreases between first and last iter, (c) re-run with same seed produces same first-iter loss to 1e-6.

---

## 6. Failure-Mode Analysis Cross-Reference

| Symptom you'd see in artifacts | Root cause | Reference |
|---|---|---|
| Ablation row reports champion = best.pt with `best_checkpoint_bootstrap=True` | W4 (best.pt seeded; no promotion) | `cli_main.py:721-730` |
| `loss=null` or no `loss` key in train_log.jsonl rows | W6 (replay never crossed batch) | `cli_main.py:875-894` |
| `controller_axes` rows show divergent `mean root_visits_at_halt` | W2 (penalty changes P_flip changes halt step) | `quartz.rs:1862-1959` |
| `prior_refresh_temp` sweep is flat | W3 (hardcoded τ=0.5 fallback) | `select.rs:285-289` |
| Self-play "stop within timeout" warning at end of run | subprocess teardown | `selfplay_runtime.py:2205-2218` |
| Replay samples mix actor generations without tagging | W5 (no actor_generation tag) | `quartz/replay.py:168-174` |
| No halt_reason histogram in replay search summary | W8 (silent telemetry) | `src/mcts_server.rs` |
| Same model "wins" only under one eval profile | W7 (eval engine spec varies per row) | `ablation_study.py:1156-1166` |

---

## 7. Executable Reality & Hardware-Fit Audit

### 7.1 Executable reality

- **Build:** `cargo build --release` succeeds (verified by Cargo.lock + recent commits). ROCm/CUDA Dockerfiles present.
- **Tests:** prior audit ran `cargo test --release` (386 passed, 65 ignored) and `pytest -q tests/...` (254 passed). Re-running was not necessary for this audit (no test-touching changes since).
- **Runtime closure:** structurally yes; semantically partial under smoke parameters (W6) and under non-promoting ablations (W4).
- **Server/backend divergence:** Three runtime backends (`torch_runtime`, `jax_runtime`, `onnx_support`) but only torch is the production path for self-play+eval. JAX is training-only. ONNX is deploy-only.

### 7.2 Hardware fit (Ryzen 5900X / 64 GB / RX 6950 XT)

- **CPU/GPU split is realistic:** Rust runs on CPU for tree, Python on GPU for NN. Ryzen 5900X has 12C/24T; the project clamps thread count via `clamp_thread_count` and supports parallel search.
- **GPU caveat:** RX 6950 XT is hipBLASLt-unsupported per `quartz/system_runtime.py:123-148`; throughput penalty is real. Documented but the warning is suppressed by default; users may not notice.
- **Memory:** 64 GB is comfortable; Rust SHM rings are configurable via `SHM_RING_TOPOLOGY` env, default `8x8`.
- **Compile time:** `cargo build --release` with `lto=fat, codegen-units=1, opt-level=3` is slow but only needed once per source change.
- **Ablation throughput:** `Cargo.toml` profile is aggressively optimized; per-condition smoke ablations finish in minutes at gomoku7. Multi-seed gomoku15 studies are hours-to-days, which the project itself acknowledges as not-yet-default.

**Verdict: suitable with minor fixes.** Once W4-W6 land, ablation iteration on this hardware is genuinely usable. Without them, smoke ablations corrupt their own conclusions.

---

## 8. Ablation / Measurement Honesty Audit

### 8.1 Per-condition fairness checklist

| Property | Enforced? | Mechanism |
|---|---|---|
| Same NN architecture | Yes | per-game catalog locks filters/blocks |
| Same training config except controller axis | Yes (in `controller_axes` preset) | factor-isolation rows |
| Same compute budget | **No** | W2: VOC halt drifts with penalty mode |
| Same evaluator engine | **No** | W7: each eval row builds own cfg |
| Same seed across rows | Optional (paired-seed-eval flag) | should be default-on |
| Same actor freshness schedule | **No** | W5: replay-gated refresh |
| Repeated seeds | Optional | `--seeds 41,42,43` supported, not always used |
| CIs on win rate | Yes | `score_rate_ci` with z=1.96 |
| Variance reporting across seeds | Partial | Aggregation present; not always cited |

### 8.2 Missing or partial metrics

| Metric | Currently emitted? | Why it matters |
|---|---|---|
| Per-row `controller_identity_hash` | No | W1 attribution |
| `actor_generation_histogram` per row | No | W5 freshness diagnosis |
| `halt_reason_histogram` per row | Partial | W8 controller falsification |
| `mean root_visits_at_halt`, `mean elapsed_ms_at_halt` | Partial | W2 budget drift detection |
| `effective_tau_observed` | No | W3 sweep validity |
| `eval_condition_search_profile` per match | No | W7 stratification |
| Replay freshness (oldest sample age) | No | training-mix diagnosis |
| Self-play diversity (game-state entropy or unique-position count) | No | mode-collapse detection |
| Queue latency / inference throughput | Partial (dev-only profiles) | hardware regression |
| Regret-like proxy for search quality | No | beyond pass/fail |

### 8.3 One-off vs robust

- README itself flags `T0010_cf38467f` as "a one-off sweep output and is not re-runnable" (README:191-192). **This is honest but it is also a current cited result.** The honest framing is "this number is exploratory."
- `controller` preset is described as "intentionally bundled legacy-vs-theory" (README:267-268). Honest, but readers unfamiliar with the project may still cite its results as if controller_axes-grade.

---

## 9. Tests / Docs / Examples Honesty Audit

### 9.1 Test classification

**Python (per file):**
- `test_training_pipeline_regressions.py` — replay API, config logic, schema validation: **unit/contract**, not training-loop.
- `test_evaluation_pipeline_regressions.py` — Glicko-2 numerical regression: **good unit + numerical**.
- `test_ablation_study.py` — manifest generation, champion selection, schema: **unit/protocol**.
- `test_phase15_ablation.py` — system config schema, readout operator logic, trace caching: **unit**.
- `test_controller_regression.py` / `test_controller_sweep.py` / `test_controller_optuna.py` — controller-config logic, sweep execution: **unit + sweep-protocol**.
- `test_batch_protocol.py` — QIPC frame encoding/decoding: **unit/contract**.

**Rust (per module):**
- `src/mcts/{quartz,select,search,eval}.rs` — controller math, selection, halt logic: **unit + numerical regression**.
- `src/ablation_{vl,pflip,refresh_v2}.rs` — search-controller agreement / throughput regression with fixed evaluators: **search-controller regression**.
- `src/games/*.rs` — rules engine: **unit + numerical**.

**No tests classify as:**
- end-to-end training-loop regression (loss decreases over N iterations)
- determinism regression (seed → byte-equal output)
- attribution isolation (controller_axes rows produce single-factor delta)

### 9.2 Doc-vs-code overclaim table

| Claim source | Claim | Status |
|---|---|---|
| README:1-3 ("AlphaZero-style game-playing AI engine with adaptive search controller") | strong claim | clearly_implemented (with W4-W7 caveats) |
| README:24-26 ("Python training loop — self-play, replay buffer, SGD, checkpoint evaluation via Glicko-2") | strong claim | clearly_implemented |
| README:155-160 ("Current controller status — engineering signals, not multi-seed publication-grade") | downgrade | **honest** |
| README:191-192 (T0010 "is not re-runnable") | downgrade | **honest** |
| README:200-204 ("smoke is canonical audit smoke; meant to fail fast") | partial downgrade | smoke is honest about its scope; but tests/CI rely on it for upstream gating |
| docs/QUARTZ_THEORY.md:21-33 ("not one frozen formula") | downgrade | **honest** |
| docs/ABLATION_GUIDE.md (controller_axes preset) | strong claim | partially_implemented — preset exists; isolation property untested |
| docs/TRAINING_GUIDE.md:133-134 (Python evaluator fallback "not benchmark-grade") | downgrade | **honest** |
| configs/gated_refresh.json comment "Default: minimax-optimal from ablation study" | strong claim | overclaimed — "default" is misleading; not the current production default |
| configs/pflip_mixture.json / self_adaptive.json claims | strong claims | overclaimed — claim "best for X" without supporting test |

### 9.3 Examples honesty

- `scripts/smoke_e2e.py` — honest about being a runtime smoke; W6 issue is not in the script's framing but in users reading it as a training certification.
- `scripts/ablation_study.py` — gracefully falls back with logged warning if Rust binary missing; honest.
- `scripts/build_audit_bundle.py` — produces source-level bundle, README explicitly says "not an offline cargo vendor / wheelhouse bundle" (README:67-69). Honest.
- `scripts/controller_optuna.py` / `controller_sweep.py` — explicit about checkpoint requirements; reject implicit auto-discovery. Honest.

---

## 10. Golden-Reference Comparison (structural, not parity)

| Axis | OpenSpiel AZ | MiniZero | KataGo | Lc0 | QUARTZ |
|---|---|---|---|---|---|
| Self-play / learner / evaluator separation | clean (separate processes/files) | clean (server + workers) | clean | clean (engine vs trainer) | **partial** — separate modules but coupled to one cli_main entry |
| Storage/checkpoint discipline | versioned + metadata | versioned | versioned | versioned | **partial** — `runtime_contract_hash` is good; `best.pt` seeding fault undermines it |
| Batched inference design | minimal (single-actor) | server-batch | server-batch with per-thread submit | server-batch | **strong** — QIPC + SHM ring is real |
| Controller observability | minimal controller | minimal | rich (search-step logging, root variance plots) | rich (engine_options, search analysis) | **partial** — stats exist; emission is silent |
| Same-budget benchmarking | per fixed-N visits | yes | yes (with care) | yes | **partial** — VOC halt confounds budget |
| Evaluator-quality stratification | basic | basic | KataGo-style (multiple opponents) | yes | **basic** — single Glicko-2 ladder; no stratified opponents |
| Controlled ablations without surgery | possible | possible | possible | possible | **possible with caveats** — controller_axes is the right idea but not enforced |
| Where genuinely novel | — | — | adaptive playout cap, CGT-style scoring | RL inference engineering | **VOC channel decomposition** is a genuinely new framing if rigorously isolated |

**Key takeaway:** QUARTZ's structural separation and reproducibility are above-average for a single-author research repo, on par with MiniZero in spirit. Where it lags KataGo/Lc0 is **observability and budget-fairness discipline**, not architecture.

---

## 11. Concrete Upgrade Proposals

### 11A. Conservative upgrades (preserve current structure)

| # | Upgrade | Pain solved | Cost | New experiment unlocked |
|---|---|---|---|---|
| C1 | End-of-training `latest.pt` save + ablation_study prefers `latest.pt` unless promotion happened | W4: best.pt corruption | ≤20 lines | Trustable post-train evaluation in non-promoting runs |
| C2 | Iteration-level actor refresh (drop `executed_steps>0` gate) + tag replay samples with `actor_generation` | W5: actor freshness | ≤30 lines | Phase-aligned freshness audits |
| C3 | Smoke bumps `games-per-iter=16` or `batch=64`; assertion on `n_train_rows > 0` | W6: smoke vacuous | ≤5 lines | CI gate on actual SGD execution |
| C4 | Halt-reason + penalty-mode + refresh-activation telemetry through replay search summary | W8: silent controller | ≤80 lines | Mechanism-level falsification |
| C5 | Replace hardcoded `tau=0.5` fallback with `prior_refresh_temp.max(1e-6)` | W3: silent parameter clobbering | 2 lines | Honest `prior_refresh_temp` sweeps |
| C6 | `controller_identity_hash` per-row in `study_manifest.json`; assert constancy across single-axis rows | W1: heuristic family conflated | ≤30 lines | Cross-paper portable controller IDs |
| C7 | `--frozen-eval-condition E1_baseline` default-on for attribution presets | W7: eval engine drift | ≤20 lines | Same-eval same-budget cross-condition comparison |
| C8 | Add `pytest tests/` and `cargo test --release` stages to CI workflow | W9: CI gap | ≤15 lines YAML | Regression discipline |
| C9 | End-to-end training convergence test (5 iter, assert loss decreases, determinism check) | W10: behavioral gap | ~200 lines | Training-loop regression coverage |
| C10 | Per-row mean `root_visits_at_halt`, `elapsed_ms_at_halt`, `effective_tau_observed` | W2 + W3 diagnosis | ≤40 lines | Same-budget audit |

### 11B. Structural redesigns (deeper but optional)

| # | Redesign | Gain | Risk |
|---|---|---|---|
| S1 | Controller as `observe() → infer_trust() → decide_refresh() / decide_penalty() / decide_halt()` policy object (see §12.2) | Penalty/halt orthogonalization | Touches every penalty mode dispatch site; behavior-equivalent via test parity |
| S2 | Pin `HaltMode::Fixed(N)` for attribution presets; reserve VOC for production search | Solves W2 budget coupling at design level | Loses some attractive "search self-stops" demos but grants causal clarity |
| S3 | Separate `SearchConfig` (selection/halt/expansion) from `TrainingConfig` (replay/optimizer) and forbid silent overrides | Clean config schema | Migration friction across `configs/*.json` and Python defaults |
| S4 | Pipeline contract: every cross-process boundary carries `(version, schema_hash, generation_id)` | QIPC + actor handoff robustness | Coordinated change in Rust+Python |

---

## 12. Pseudocode Upgrade Plan

### 12.1 Current-path patch — observability + fairness without redesign

```rust
// src/mcts/quartz.rs:1862  (extend should_stop telemetry)
fn should_stop(&mut self, root_visits: u32, elapsed_ms: u64) -> bool {
    self.last_check_at = root_visits;
    self.elapsed_ms = elapsed_ms;

    let stopped_by = match self.config.halt_mode {
        HaltMode::Fixed(n)            => check_fixed(root_visits, n),
        HaltMode::SimpleThreshold     => check_simple(&self.last_stats),
        HaltMode::VOC                 => check_voc(&self.last_stats),
        HaltMode::ConfAdaptive        => check_conf(&self.last_stats, &mut self.theta_conf),
    };

    self.stop_reason = stopped_by.reason;
    // NEW: structured emission for telemetry pipeline
    self.telemetry.record_halt_check(HaltCheck {
        root_visits,
        elapsed_ms,
        p_flip:        self.last_stats.p_flip,
        flip_stable:   self.last_stats.flip_stable,
        voc_total:     self.last_stats.unified.voc_total,
        voc_focus:     self.last_stats.unified.voc_focus,
        voc_expand:    self.last_stats.unified.voc_expand,
        voc_merge:     self.last_stats.unified.voc_merge,
        sigma_q:       self.last_stats.sigma_q,
        hbar_eff:      self.last_stats.hbar_eff,
        decision:      stopped_by.reason.tag(),
        triggered:     stopped_by.triggered,
    });
    stopped_by.triggered
}
```

```python
# quartz/replay.py — emit per-game telemetry into search summary
def _aggregate_halt_telemetry(game_telemetry: list[HaltCheck]) -> dict:
    halt_reason_hist = Counter(e.decision for e in game_telemetry if e.triggered)
    return {
        "halt_reason_hist": dict(halt_reason_hist),
        "mean_root_visits_at_halt": mean(e.root_visits for e in game_telemetry if e.triggered),
        "mean_elapsed_ms_at_halt":  mean(e.elapsed_ms for e in game_telemetry if e.triggered),
        "p_flip_at_halt_mean":      mean(e.p_flip for e in game_telemetry if e.triggered),
        "voc_total_at_halt_mean":   mean(e.voc_total for e in game_telemetry if e.triggered),
    }
```

**Solves:** W2 (visible budget drift), W8 (silent controller). **Cost:** ~150 lines + struct definitions. **Risk:** none if telemetry is additive.

### 12.2 Cleaner controller interface — policy-object split

```rust
// New trait, layered on top of existing QuartzConfig/QuartzStats
trait SearchController {
    /// Called once per check_interval after stats are recomputed.
    fn observe(&mut self, stats: &QuartzStats, runtime: &RuntimeView);

    /// Selection-time penalty/refresh decision per edge.
    fn decide_selection_terms(&self, edge: &EdgeView) -> SelectionTerms;
    //                                                  ^ {penalty, refresh_rho, refresh_tau, blend_weight}

    /// Halt decision; pure function over observed state.
    fn decide_halt(&self) -> HaltDecision;
    //                       ^ {triggered, reason, theta_used}

    /// Trust adapter — how much to weight evaluator vs search statistics.
    fn evaluator_trust(&self) -> f32;
}

// Existing modes become thin wrappers that fill SelectionTerms / HaltDecision.
// Crucially, decide_halt() reads only stats fields the mode declares — no
// silent cross-coupling with selection penalties.
```

```python
# Python side: same interface mirror so eval-engine code can introspect
class ControllerSnapshot:
    selection_terms: dict[edge_id, SelectionTerms]
    halt_decision: HaltDecision
    evaluator_trust: float
    # Serialized into per-position telemetry; aggregable.
```

**Solves:** W1 (mode dispatch is now explicit per concern), W2 (halt cannot silently read penalty-side state). **Cost:** ~600 lines refactor; behavior-equivalent under property tests. **Risk:** non-trivial migration; recommend doing this AFTER W3-W8 conservative patches land.

### 12.3 End-to-end pipeline contract — handoff schema

```text
Cross-process contracts (every boundary versioned + schema-hashed):

[Python learner] --(weights v=1, gen=N, schema_hash=H)--> [SHM weight slot]
                                                            ↓
[Rust server] --(boards v=1, gen=N expected)--> [Rust selfplay]
                                                            ↓
[Rust selfplay] --(replay sample v=1, controller_identity=I,
                   actor_generation=N, halt_telemetry={...})--> [Python replay]
                                                                       ↓
[Python replay] --(batch v=1, actor_generations=[N, N-1, ...],
                   freshness_age_mean=... )--> [SGD]
                                                  ↓
[SGD] --(checkpoint v=1, gen=N+1, schema_hash=H',
         loss_curve=..., promoted=True/False )--> [Persisted]
                                                       ↓
[Persisted] --(tournament: pinned eval_search_profile=E,
               pinned halt_mode=Fixed(M), paired_seed=true)--> [Glicko + verdict]
                                                                        ↓
[Verdict] --(promotion event, reason, sr_ci, n_games)--> [best.pt]
                                                              ↓
[best.pt] --(only updated on promotion; latest.pt always saved)--> [Post-train eval]
```

**Each handoff:**
- Carries `version, schema_hash, generation_id` in payload header
- Mismatch raises typed error (not silent return-None)
- Logged to a single `pipeline_event_log.jsonl` with one event per boundary crossing

**Solves:** W4 (best.pt path explicit), W5 (gen tagging), W7 (eval pinning explicit), W8 (halt telemetry as first-class), QIPC bare-bytes coupling. **Cost:** ~800 lines net new + edits across boundaries. **Risk:** coordinated change. Recommend phased rollout: telemetry first (12.1), then schemas (12.3), then controller refactor (12.2).

---

## 13. Minimal Patch Plan (≤10, ranked)

| # | Patch | Purpose | Failure mode it blocks | Difficulty | Expected gain | Required for? |
|---|---|---|---|---|---|---|
| P1 | End-of-training `latest.pt` save; `ablation_study.resolve_model_path` prefers `latest.pt` unless promoted | Trust post-train evaluation | W4 silent stale-champion | Trivial (≤20 lines) | High | ablation-ready |
| P2 | Hardcoded `tau=0.5` fallback removal; clamp to `1e-6` | Honest `prior_refresh_temp` sweeps | W3 silent clobbering | Trivial (2 lines) | Medium | design-ready |
| P3 | Smoke `--games-per-iter` bump or `--batch` shrink; assert `n_train_rows > 0` | Smoke certifies SGD not just imports | W6 vacuous smoke | Trivial (≤5 lines) | High | pipeline-ready |
| P4 | CI workflow stages: `pytest tests/` + `cargo test --release` | Regression discipline | W9 CI gap | Trivial (≤15 lines YAML) | High | ablation-ready |
| P5 | `controller_identity_hash` in `study_manifest.json`; assert constancy on single-axis rows | Cross-row controller portability | W1 heuristic family | Easy (≤30 lines) | High | ablation-ready |
| P6 | Halt-reason + penalty-mode telemetry through replay search summary (per §12.1) | Mechanism-level falsification | W2 + W8 | Medium (~150 lines) | Very high | design-ready |
| P7 | `controller_axes` preset pins `HaltMode::Fixed(max_visits)` for all rows | Same-budget attribution | W2 budget drift | Easy (≤10 lines) | Very high | design-ready |
| P8 | `--frozen-eval-condition E1_baseline` default-on for attribution presets | Same-eval cross-condition comparison | W7 eval drift | Easy (≤20 lines) | High | ablation-ready |
| P9 | Iteration-level actor refresh (decouple from `executed_steps>0` gate); tag replay samples with `actor_generation` | Phase-aligned freshness | W5 stale actor | Easy (~30 lines) | High | pipeline-ready |
| P10 | End-to-end training convergence test (5 iter, loss decreases, determinism re-run) | Training-loop regression coverage | W10 behavioral gap | Medium (~200 lines) | Very high | ablation-ready, design-ready |

**Total effort:** ~600 lines + CI YAML across the ten patches. **Order recommendation:** P1, P2, P3 immediately (trivial); then P4-P5-P9 (single-day work); then P6-P7-P8 (week-scale); P10 last (needs other patches to be meaningful).

---

## 14. CoVe / Contrastive Verification

### 14.1 Verification questions and answers

**Q1.** Is the prior 20260423 audit's "stale actor" finding actually still present?
**A:** Yes, but in modified form. Refresh now fires per-iteration *if SGD ran*. SGD requires `len(replay) >= cfg["batch"]`. In smoke or short ablations, this is unsatisfied → actor stays at gen 0. The fix landed but the gating chain has a remaining hole. Confirmed: `cli_main.py:956-957`.

**Q2.** Has `best.pt` seeding been fixed since 20260423?
**A:** No. `best_checkpoint_bootstrap=True` path in `cli_main.py:721-730` still active; no end-of-training fallback save.

**Q3.** Does Phase 6.1 make-unmake change controller invariants?
**A:** No. `apply_move_in_place` returns an unused undo token; the descent path doesn't backtrack (`select.rs:733`). Pure perf optimization.

**Q4.** Does Phase 6.2 RwLock affect controller correctness?
**A:** No. Per-bucket Sync semantics preserved; `unsafe impl` documented; no QUARTZ state reads through the TT lock.

**Q5.** Is the shallow-blend (depth ≤ 3) code path actually dormant?
**A:** Yes by default. `root_only_shaping=true` is the default in `QuartzConfig::default()`. Test `test_legacy_profile_can_shape_shallow_nonroot_nodes` (`select.rs:1141-1179`) verifies the blend exists when toggled. No production config sets `root_only_shaping=false`.

**Q6.** Is the `prior_q_divergence` field actually used by the modes that read it?
**A:** Partially. `GatedRefresh` reads it (line 248). `PFlipMixture` does NOT consult divergence in its mixture gate despite the narrative implying divergence-aware behavior. The field is computed but unused by ~half the modes.

**Q7.** Is the dual-gate halt (`flip_stable AND voc_total≤0`) load-bearing or vacuous?
**A:** Load-bearing. In practice, P_flip stabilizes faster than VOC drives negative; VOC ≤ 0 is the binding constraint. Means VOC channel calibration drives halt step. Means VOC mis-calibration (e.g., cost_base too cheap) silently changes search budget across configs.

**Q8.** Is the test fixture `tests/fixtures/regression_positions.json` actually used?
**A:** It exists but is not consumed by any current test. Confirmed by grep — no Python or Rust test loads this file. It was scaffolded for an integration test that never landed. (Existence ≠ load-bearing.)

**Q9.** Are the `configs/*.json` files used in production code paths or only as examples?
**A:** Examples only. Default training reads from `quartz/training_catalog.py`; JSON files are example overrides users can pass via `--config`. Their narrative comments ("Default: minimax-optimal", "Best for trusted prior") are aspirational.

**Q10.** Does `--paired-seed-eval` actually enforce within-seed comparison or is it advisory?
**A:** Enforces. `ablation_study.py:1131-1138` `should_compare()` returns False if seeds differ. But default is False (i.e., off) — meaning by default, all condition pairs are compared regardless of seed.

**Q11.** Does the README correctly characterize the smoke as a runtime smoke?
**A:** Yes (README:200-204). The smoke's framing in the README is honest. But ABLATION_GUIDE recommends the smoke as a pre-study sanity check, which it is — except it doesn't catch training-loop regressions.

**Q12.** Is "controller_axes" truly attribution-grade in current code?
**A:** Configurationally yes (single-axis adjacency is real). Operationally no (eval engine drift W7, halt-budget drift W2, no enforced same-eval). Becomes attribution-grade after P5+P6+P7+P8.

### 14.2 Contrastive comparison

| Hypothesis | Executability | Controller maturity | Pipeline maturity | Hardware | Rigor | Docs |
|---|---|---|---|---|---|---|
| H1 (strong exploratory platform) | ✓ | ~ (heuristic family known to author) | ~ | ✓ | partial | honest |
| H2 (good ideas, partial maturity, major revision needed) | ✓ | ~ | ~ | ✓ | needs P1-P10 | honest |
| H3 (premature, doc-strong but reality-thin) | mostly ✓ | — | partial | ✓ | weak | honest |

**H2 fits best.** H1 understates the budget-coupling and eval-drift issues; H3 understates how much actually runs (386+254 tests pass, real Rust+NN evaluator, real multi-game support, real ablation infrastructure). The gap between H1 and H2 is precisely the ten minimal patches in §13.

---

## 15. Final Verdict

**Verdict: ablation-usable with revisions.**

Specifically:
- *Ablation-usable* now for engineering iteration on a single controller variant, with eyes-open acceptance that VOC halt + variable eval-engine confounds attribution.
- *Ablation-usable* for publication-grade attribution after P5 + P6 + P7 + P8 land (controller identity, halt telemetry, fixed budget, fixed eval condition).
- *Clean research-grade platform* after P1-P10 plus the §12.2 controller refactor and §12.3 pipeline contract land. ~2-3 weeks for a focused engineer.

The verdict is *not* "conceptually promising but structurally noisy" — the structure is fine, the issues are bounded execution-discipline gaps. The verdict is *not* "clean research-grade" — the controller-as-family conflation, the silent budget coupling under VOC halt, and the absence of an end-to-end behavioral test are all blocking.

## 16. One-line reason

**The platform runs and the docs are unusually honest, but the controller's six modes silently couple to halt and eval budgets through shared P_flip/σ_Q inputs, and three load-bearing pipeline contracts (best.pt seeding, replay→SGD threshold, eval engine spec) corrupt short ablation conclusions — all fixable in ~600 lines plus a real end-to-end training test.**
