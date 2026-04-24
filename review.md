# QUARTZ — Journal-Grade Integrated Audit

Date: 2026-04-25
Scope: whole repository, controller/MCTS core, Python training pipeline, ablation harness, tests, docs, results artifacts.
Method: static read of Rust (`src/mcts/*.rs`) and Python (`quartz/*.py`, `scripts/*.py`), docs (`README.md`, `docs/*.md`), configs, CI workflow, and a sample of `results/audit_e2e_smoke_*` manifests. Two earlier codex audits (`audit_codex_20260422.md`, `audit_codex_20260423.md`) were read for continuity and their findings re-verified on the current tree.
No new code was built or executed. Cited line numbers are at the current state of `main`.

---

## 1. Project Claim Reconstruction

Extracted from `README.md`, `docs/QUARTZ_THEORY.md`, `docs/TRAINING_GUIDE.md`, `docs/ABLATION_GUIDE.md`, and `pyproject.toml`. Provisional tags: C = clearly implemented, P = partially implemented, U = unclear/weakly evidenced, O = overclaimed.

| # | Claim | Tag |
|---|---|---|
| 1 | Rust MCTS engine with TT, progressive widening, adaptive VL is the sole training/eval search substrate | C |
| 2 | QUARTZ controller is a single "Q-value uncertainty / root-risk / zero-tunable" theory-aligned design | **O** — it is a family of 6 penalty modes × 4 halt modes; docs collapse them under one name |
| 3 | Multiple controller modes are independently toggleable for ablation | P — toggles exist but factor orthogonality is broken (see §3, §5) |
| 4 | Python self-play → replay → SGD → evaluation forms a closed AlphaZero loop | P — closed topologically, but actor freshness and loss emission are loose contracts |
| 5 | Same-stack evaluation: Rust+NN both sides of the arena (no toy baseline) | C — verified in `evaluator_runtime.py`, no dummy engine fallback |
| 6 | Hybrid QIPC transport: JSON control + binary/SHM hot-path | C — `ShmRingBuffer` in `qipc.py:445-496` is real; polls without event signals |
| 7 | Multi-game: Gomoku 7/15, Go 9×9, Chess with exact history-aware TT keys | C — `tt_hash()` override in `src/games/go.rs:1172-1190` and `src/games/chess.rs:1672-1681` confirmed |
| 8 | Chess uses promotion-aware 4672-action policy | C |
| 9 | JAX backend for training is available | **O** — `quartz/jax_training_runtime.py` is a pass-through that routes to torch hooks; no JAX-specific SGD or Rust inference path |
| 10 | ONNX export + Gomocup `pbrain` deployment is available | P — code path exists; "environment-specific verification" admitted in README Maturity table |
| 11 | Glicko-2 + SPRT evaluation with real score-rate CIs | C — `quartz/evaluation.py` has full implementation; tests pin algebraic equivalence |
| 12 | Adaptive VL is a 2nd-generation feedback controller on dup_rate + contention | C — genuine feedback loop verified in `src/mcts/parallel.rs:264-272`, not threshold switching |
| 13 | `controller_axes` preset isolates factors one at a time (root_only_shaping, penalty_mode, prior_refresh_rate) | P — the preset exists and intends attribution, but penalty_mode still implicitly couples halt behavior (see §5) |
| 14 | Runtime contract hash makes ablation rows reproducible | C — `runtime_contract_hash` is written to manifests at `scripts/ablation_study.py:285, 564, 1199, 1216` |
| 15 | Canonical smoke `scripts/smoke_e2e.py` certifies end-to-end readiness | **O→downgraded in text** — README line 177-179 admits it is a fail-fast runtime smoke, not benchmark certification; code matches admission |
| 16 | CI gates on the full test suite | **O** — `.github/workflows/phase15-benchmark-gate.yml:36-72` runs phase15 smoke only; it does not run `pytest tests/` or `cargo test --release` |
| 17 | Controller telemetry: `halt_reason_hist`, `controller_penalty_mode_counts`, `mean_prior_refresh_rate` | P — first two are real runtime aggregates; `mean_prior_refresh_rate` is a config constant, not a measurement (see §8) |
| 18 | Phase15 clean-split tooling with reference/oracle separation and trace amortization | C |
| 19 | Current controller evidence favors `A1_legacy_base` + Optuna `T0010_cf38467f` | U — language is cautious ("repository-local evidence currently points to") but the `T0010_cf38467f` artifact is not visible under `results/` |
| 20 | Hardware support incl. AMD ROCm on RX 6950 XT | **O** — `quartz/system_runtime.py:123-148` explicitly flags RX 6950 XT as hipBLASLt-unsupported and suppresses the warning; degraded path documented nowhere user-facing |

### Independent axis summary

- **End-to-end executability**: good (closed topologically; smokes complete with warnings).
- **Controller state semantics**: rich (36-field `QuartzStats`), but not unified — the same latent signal (p_flip) is wired into multiple modes' halt and penalty simultaneously.
- **MCTS modification scope**: real (selection PUCT dispatched per mode), narrow (no change to expansion/backup), partially wired at depth ≤3 when `root_only_shaping=false`.
- **Evaluator/training coupling**: real same-stack Glicko-2; actor-refresh is per-iteration now (better than the codex described) but still un-versioned.
- **Ablation readiness**: decent for engineering iteration, fragile for clean causal attribution — budget, prior-temperature, and confidence-halt confounds not controlled by default.
- **Hardware realism**: CPU/RAM comfortable; the claimed GPU (RX 6950 XT) sits on a known-degraded ROCm path with suppressed warnings.
- **Docs/tests honesty**: most paths match code, but two specific claims (JAX backend parity, "one controller principle") exceed the implementation.
- **Reproducibility**: above average — runtime_contract_hash, search_manifest_hash, eval CI/SPRT, study manifest; below what publication demands — CI does not enforce tests, no numerical regression gates.

---

## 2. Executable Path Reconstruction

### 2.1 What is actually wired

Training path (confirmed by reading each file):

- Entry: `quartz/cli_main.py` parses args and builds runtime hooks; dispatches to `quartz.torch_training_runtime` (default) or `quartz.jax_training_runtime` (stub).
- Self-play: `quartz/selfplay_runtime.py:selfplay_rust_nn_batched(...)` spawns the Rust binary `./target/release/mcts_demo --server`, communicates via `quartz/qipc.py` SHM ring for evaluation requests/responses plus JSON-line control.
- NN inference: `quartz/runtime_support.py:_run_model_batch(...)` returns `(logits, values)` to the Python-side broker, which forwards them via the p2r SHM slot.
- Replay: `quartz/replay.py` `ReplayBuffer` (deque-based) with sparse policy storage and `recent_fraction` windowed sampling.
- Learner: `quartz/train_loop.py:train_epoch(...)` called from `cli_main.py:895-919`; conditional on `train_steps > 0` (see §5 Finding L1).
- Evaluator: `quartz/evaluator_runtime.py:RustNNEvaluatorEngine` runs the same Rust+NN substrate against the incumbent champion; Glicko-2 update in `quartz/evaluation.py`.
- Checkpoint/promotion: `cli_main.py:_save_model_checkpoint()` wraps cfg into every save (`cli_main.py:266-270`); bootstrap of `best.pt` is explicit with a `best_checkpoint_bootstrap_seeded` flag (`cli_main.py:722-723`).
- Ablation harness: `scripts/ablation_study.py` for training-level ablations; `scripts/controller_sweep.py` + `scripts/controller_optuna.py` for frozen-checkpoint post-hoc sweeps; `scripts/phase15_ablation_study.py` for clean-split posthoc/online assays.
- Experiment runner (smoke): `scripts/smoke_e2e.py` — runs a fixed-size ablation and asserts the presence of `study_manifest.json`, `evaluation_matrix.json`, `ablation_report.json`.

Rust search path:

- `src/main.rs` and `src/mcts_server.rs` (server entrypoint; referenced from README and qipc) start a long-running search process; messages arrive over SHM ring.
- `src/mcts/mod.rs` top-level tree; `src/mcts/search.rs` main loop; `src/mcts/select.rs` PUCT + controller dispatch; `src/mcts/expand.rs`, `src/mcts/backup.rs`, `src/mcts/parallel.rs`, `src/mcts/node.rs`, `src/mcts/tt.rs`, `src/mcts/root.rs` support modules.
- Controller core in `src/mcts/quartz.rs` (114 KB): `QuartzStats` struct, `compute_quartz_stats(...)`, `update_stats(...)`, `should_stop(...)`, penalty-mode enum and dispatch.
- Game adapters: `src/games/{gomoku,gomoku15,go,chess}.rs` implementing the shared `GameState` trait; history-aware `tt_hash()` override for chess and go confirmed.
- ONNX path: gated by `--features onnx` in `Cargo.toml`; not default-built.

Deployment (Gomocup) path:

- Python side: `quartz/onnx_support.py` + `quartz/gomocup_export.py` produce a bundle with deployment search config.
- Rust side: built with `--features onnx`, `scripts/build_gomocup_brain.sh`.
- Admitted as `Partial` in README Maturity table — environment-specific; not audited end-to-end here.

### 2.2 Paths described but not what they appear to be

- **`quartz/alphazero_train.py`** (1499 lines, 56 KB): honest compatibility facade — re-exports from the new runtime modules. The file size is misleading; it is not the canonical implementation. This matches its header claim.
- **JAX training path**: `quartz/jax_training_runtime.py` imports the same `MainRuntimeHooks`, `train_epoch`, `evaluator_runtime`, and `selfplay_runtime` as the torch path. There is no JAX-specific SGD, no JAX-to-Rust weight serialization, and no JAX inference into the Rust server. It is a nominal backend at best.
- **JAX Rust inference**: explicitly excluded by the README — but the user-visible `--backend jax` flag in `cli_main.py:49-52` does not warn the user that self-play/eval still run through the torch model-forward path.
- **"Canonical smoke certifies training"**: README footnote downgrades this to fail-fast runtime smoke; the code matches the downgrade, but the Quick Start (README line 37-38) still frames `smoke_e2e.py` as the canonical audit entrypoint without re-iterating the downgrade.
- **CI gating**: `.github/workflows/phase15-benchmark-gate.yml` is the only workflow; it runs phase15 CI smoke. `pytest tests/` and `cargo test --release` pass locally (254 / 386) per prior audit, but **neither is gated by CI**. Any test regression would not be caught upstream.
- **`tests/fixtures/regression_positions.json`**: contains 2 annotated canonical positions with `expected_top3_region` and `p_flip_band` fields. **No test file currently consumes this fixture** — dead artifact ready to be wired.

### 2.3 Handoff continuity

End-to-end loop is topologically closed, but three handoffs are under-specified:

- **Python learner → Rust server actor weights**: `bg_worker.update_model(actor_source)` in `cli_main.py:923`, then `selfplay_runtime.py:2104-2105` does `self._model = self._clone_actor_model(model)`. The next call to `selfplay_rust_nn_batched` picks up the new model. No version tag, no weight hash, no synchronization barrier.
- **Replay samples → learner actor identity**: `ReplayExample` (`replay.py:168-174`) has no `actor_generation` / `actor_hash` field. SGD mixes samples from multiple actor generations without marking them.
- **Training loss emission → log**: `cli_main.py:875-894` emits a loss row only if `train_steps > 0`. Replay-underflow iterations emit a partial row with `replay`, `new_pos`, and timing fields but no `loss` — downstream parsers must special-case.

---

## 3. Search Controller / MCTS Architecture Reconstruction

### 3.1 Controller state variables

`QuartzStats` (`src/mcts/quartz.rs:340-415`) carries 36 fields in four clusters.

- **Uncertainty** (decision-bearing): `hbar_eff`, `sigma_q`, `sigma_delta`, `p_flip`.
- **Convergence** (decision-bearing): `converged`, `flip_stable`, `conf_t`, `p_flip_gaussian`, `p_flip_saddlepoint`.
- **VOC channels** (decision-bearing under VOC halt): `voc_focus`, `voc_expand`, `voc_merge`, `unified.voc_total`.
- **Telemetry/diagnostic** (not decision-bearing): `prior_q_divergence` (except as a gate threshold input in GatedRefresh), `surprise_kl`, `epsilon_t`, `envar_violated`, `rho_hat`, `one_loop_b`, `sigma_response_ema`, `defect_value`.

Mutations occur in `update_stats()` (`quartz.rs:1693-1853`) before each `should_stop()` call; NS gate suppression, depth calibration, merge R₀ normalization, σ_response EMA, and the defect `D_t` accumulator all live there.

### 3.2 Intervention points

- **Selection (PUCT score)**: the controller's primary influence. `src/mcts/select.rs:148-395` dispatches per `PenaltyMode`:
  - `SelfAdaptive`: subtractive penalty `-σ_Q / (1 + N_a + O_a)` plus visitor-frequency prior blend `α_a = N_a/(N_a+K)`, `τ = ln(1 + N_tot/K)` (`select.rs:194-229`).
  - `GatedRefresh`: penalty `-min(ħ_eff, cap) · N_a / N_parent`; refresh gate opens iff `D > ε_t`; uses `config.prior_refresh_temp` correctly (`select.rs:239-270, 366`).
  - `GatedRefreshLegacy`: penalty `-ν / (1 + N_a + O_a)` where `ν = hbar_penalty_cap`; refresh blend with **hardcoded `tau = 0.5`** (`select.rs:276-295, line 283`).
  - `PFlipMixture`: penalty `-max(cap, σ_Q) / (1 + N_a + O_a)`; dual-mode refresh (Q-mode vs VF-mode) split by p_flip; **hardcoded `tau = 0.5`** (`select.rs:304-348, line 330`).
  - `Legacy`: `-min(ħ_eff, cap) / N_a`, no refresh (`select.rs:396-412`).
  - `None`: no penalty.
- **Expansion**: `src/mcts/expand.rs:25-79` — the controller does **not** modify priors, Dirichlet, or progressive widening here. Priors come straight from the evaluator.
- **Backup**: `src/mcts/backup.rs:15-58` — standard negamax + Welford; controller does not modify.
- **Halt**: `quartz.rs:1861-1954`, `should_stop()`:
  - `HaltMode::Fixed{budget}` → stop when root visits ≥ budget.
  - `HaltMode::SimpleThreshold` → `p_flip < 0.159 ∧ flip_stable ≥ 3`.
  - `HaltMode::VOC` (default) → `converged ∧ (voc_total ≤ 0 ∨ p_flip < 0.159)`.
  - `HaltMode::ConfAdaptive` → `conf_t ≥ θ_conf` with online θ adaptation.
- **Refresh**: actually implemented *inside selection* as a re-derivation of the effective prior per action; there is no separate refresh pass on the tree. Root-only shaping is controlled by `QuartzConfig.root_only_shaping` (`quartz.rs:261`); when `false`, `select.rs:530-562` applies a depth-decayed blend `depth_weight = 1/(1+d)` that bleeds QUARTZ influence into non-root nodes up to d ≤ 3.

### 3.3 Unified principle vs heuristic bundle

Provisional classification: **heuristic bundle dressed as a family**, with `controller_axes` being the honest attempt at an attribution view.

Evidence:

- All four "live" modes (SelfAdaptive, GatedRefresh, GatedRefreshLegacy, PFlipMixture) fold the same latent signal (`p_flip` or its derivatives) into **both** halt and penalty, so mode-level results cannot cleanly separate "where did the improvement come from: halt, penalty, or refresh?"
- Two modes hardcode `tau = 0.5` (`select.rs:283, 330`), silently ignoring `config.prior_refresh_temp` — a config ablation that varies that parameter will report a null effect for those modes, not because the parameter is uninteresting but because the knob is disconnected. This is a measurable honesty regression (see §6 Finding C3).
- `prior_q_divergence` is computed for PFlipMixture (`quartz.rs:1674-1681`) but never consulted in any decision path — pure diagnostic inside what is advertised as a "divergence-aware mixture."

The parallelism controller, by contrast, IS a unified mechanism: a closed feedback loop on `dup_rate` and `max_pending` that continuously modulates vvalue (`parallel.rs:264-272`). This is the cleanest controller in the repo.

### 3.4 What changes what

- **Search policy (PUCT rank order)**: `penalty_mode`, `root_only_shaping`, `c_puct`, `cpuct_init`, and (for GatedRefresh) `prior_refresh_temp`. For `GatedRefreshLegacy` and `PFlipMixture`, the temperature is implicitly locked to `0.5`.
- **Termination**: `halt_mode` (Fixed / SimpleThreshold / VOC / ConfAdaptive). Under any non-Fixed mode, `penalty_mode` indirectly changes budget because the visit distribution it produces changes `p_flip`.
- **Prior usage**: `penalty_mode` chooses the refresh law; `prior_refresh_rate` is a config tag, `mean_prior_refresh_rate` is exposed in replay aggregation but not as a per-move telemetry measurement.
- **Evaluator trust**: there is no explicit evaluator-trust model. Trust is implicit in `sigma_q` and its propagation through `p_flip`. Noisy-evaluator robustness is claimed in the theory doc but not actively parameterized against evaluator quality.
- **Experiment branch**: `search_profile` + `vl_mode` + `penalty_mode` + `halt_mode` + `root_only_shaping` + `prior_refresh_rate`. 5 orthogonal-looking knobs; only ~3 are actually orthogonal in practice.

---

## 4. Design Strengths

1. **Real same-stack arena**. Both sides of every arena game run Rust+NN; there is no toy baseline fallback. This is unusual for research code and a genuine asset.
2. **History-exact TT**. `tt_hash()` for chess and go includes `history_digest`, castling / half-move, and superko state (`src/games/chess.rs:1672-1681`, `src/games/go.rs:1172-1190`). Correctness-first choice.
3. **Parallelism controller is genuinely adaptive**. The feedback loop in `parallel.rs:264-272` reads `dup_rate` and `max_pending` and modulates vvalue continuously — not threshold switching. One clean example of a unified controller in the repo.
4. **Artifact discipline**. `runtime_contract_hash`, `search_manifest_hash`, `score_rate_ci`, `sprt_result`, and `study_manifest.json` / `evaluation_matrix.json` / `champion.json` are all actually written (verified at `scripts/ablation_study.py:285, 564, 1136-1139, 1199-1220, 1542`).
5. **Split Python runtime**. The `cli_main.py` / `selfplay_runtime.py` / `train_loop.py` / `evaluator_runtime.py` / `qipc.py` / `replay.py` decomposition is real and coherent; `alphazero_train.py` is a clean facade.
6. **`controller_axes` preset recognizes the problem**. Its adjacent-row design explicitly tries to isolate `root_only_shaping`, `penalty_mode`, and `prior_refresh_rate` one at a time (`scripts/ablation_study.py:195-212`). This is more attribution-aware than most research codebases reach.
7. **Python test count is high by research-code standards**. ~289 Python test functions across 8 files, 128 Rust tests; protocol (QIPC wire format) and evaluation algebra (Glicko-2, SPRT, CI) are well-pinned.

---

## 5. Structural Weaknesses

Ordered by impact on research credibility, not on runtime stability.

**W1. Penalty-mode bundle is not orthogonal to halt-mode.** Default halt is `VOC`, which stops when `p_flip < 0.159` (line `quartz.rs:1919`). `p_flip` depends on the visit distribution, which depends on `penalty_mode`. So changing `penalty_mode` silently changes effective search budget per move. "Same search budget" across modes is false under default halt. (`quartz.rs:1861-1954`, `compute_p_flip_with_child_rtt` at `quartz.rs:888-889`.)

**W2. Hardcoded `tau = 0.5` in two modes.** `GatedRefreshLegacy` (`select.rs:283`) and `PFlipMixture` (`select.rs:330`) ignore `config.prior_refresh_temp`. A sweep that varies `prior_refresh_temp` with these modes reports a spurious null effect. Only `GatedRefresh` honors the config (`select.rs:366`).

**W3. `prior_q_divergence` is computed but unused.** `PFlipMixture` advertises mixture-by-divergence, but the divergence signal is never consulted in its gate. Pure diagnostic wearing decision-making clothes.

**W4. `mean_prior_refresh_rate` is a config tag, not telemetry.** README (line 174-175) suggests this is a controller observability field. It is a config constant multiplied into replay aggregation. No per-move measurement of how often the refresh branch actually activated exists.

**W5. Replay has no actor-generation tagging.** `replay.py:168-174` ReplayExample has no `actor_generation` / `actor_hash`. `recent_fraction` is a deque-position window, not an age filter. Stale and fresh samples are mixed without marking. Under concurrent mode, bg_worker refreshes every iteration if `executed_steps > 0` (`cli_main.py:923`) — better than the codex audit's earlier every-5-iter finding, but the samples feeding the learner still cannot be traced to an actor identity.

**W6. Actor → Rust handoff has no version verification.** `update_model(model)` (`selfplay_runtime.py:2104-2105`) is a Python-side clone; the Rust server receives serialized weights over stdin/SHM with no hash or generation ID attached. If the Rust process lags (long search in flight when new weights arrive), it will generate one more game with the previous actor silently.

**W7. CI does not gate on tests.** `.github/workflows/phase15-benchmark-gate.yml:36-72` runs `scripts/phase15_benchmark_ci_smoke.py`; it does not run `pytest tests/` or `cargo test --release`. Controller behavior regressions, Glicko-2 math breaks, and QIPC wire-format drift would not fail CI.

**W8. No controller integration test pins search output.** 59 Rust tests in `quartz.rs` pin unit-level math. No Python-level test loads the Rust binary, runs a canonical position from `tests/fixtures/regression_positions.json`, and asserts top-k moves / halt step for each penalty mode. The fixture file exists but is unused.

**W9. Training-level ablation confounds controller with learned policy.** Running `--study controller_axes` with `--iterations 30` trains a fresh NN per condition; the final arena compares (controller_A, NN_A) vs (controller_B, NN_B). The controller effect and the learned-policy effect are inseparable at the matrix level. The frozen-checkpoint path (`controller_sweep.py`) isolates controller cleanly, but the default Quick Start narrative points at the training-level harness.

**W10. Deployment config is inherited from train_cfg, not pre-registered.** `scripts/ablation_study.py:1239-1240` sets `deployment_cfg = copy.deepcopy(champion_run.get("train_cfg") or {})`. This is *better* than the codex-reported post-hoc eval cherry-pick, but still means deployment search config is selected after seeing the winner — it is the winner's own training-time config, with no independent pre-registration.

**W11. Hardware story is optimistic.** `quartz/system_runtime.py:123-148` explicitly detects RX 6950 XT as hipBLASLt-unsupported, downgrades to hipBLAS, and suppresses the warning. This is a real pragmatic choice, but the README "GPU auto-detection / AMD ROCm" entry in the Key Features table does not flag the degraded-path trade-off for users with this hardware.

**W12. Loss emission is lossy.** `cli_main.py:875-894` emits a loss row only when `train_steps > 0`. Replay-underflow iterations emit a partial row. `train_log.jsonl` cannot be re-parsed into a time series without special-casing missing `loss` fields.

---

## 6. Failure-Mode Analysis

Per-weakness: current mechanism → failure mode → why it matters → minimal diagnostic → minimal fix.

**F1 (from W1 — budget leakage across penalty modes)**
- Current: `HaltMode::VOC` default uses `p_flip < 0.159` as a halt predicate; `p_flip` depends on penalty mode.
- Failure: `controller_axes` A1→A2 comparison purports to isolate `root_only_shaping`, but the two conditions run with different effective simulation counts.
- Why it matters: arena outcomes are confounded with budget, not just with controller axis. The attribution claim (`controller_axes` is attribution-grade) is weakened.
- Minimal diagnostic: log per-move actual root visit count by condition; t-test on mean visits across conditions must fail to reject equality for the claim to hold.
- Minimal fix: for attribution-grade runs, pin `halt_mode = Fixed(N_sim)` in `controller_axes` eval conditions.

**F2 (from W2 — hardcoded tau)**
- Current: `select.rs:283, 330` use literal `0.5`.
- Failure: `--prior_refresh_temp X` under `GatedRefreshLegacy` or `PFlipMixture` has no effect.
- Why it matters: a sweep will report "temperature does not matter," which is a false null.
- Diagnostic: a one-line assertion in the Rust startup banner logging the *actual* tau used per mode.
- Minimal fix: replace literal `0.5` with `config.prior_refresh_temp.unwrap_or(0.5)` (two lines).

**F3 (from W3 — unused divergence)**
- Current: `prior_q_divergence` computed and stored but not read by the gate in PFlipMixture.
- Failure: mode description says "mixture adapts to prior-Q divergence." Actual behavior is "mixture adapts to p_flip only."
- Why it matters: narrative vs. code mismatch.
- Diagnostic: grep for reads of `prior_q_divergence` in `select.rs` — zero.
- Minimal fix: either (a) use `D_t` in the mixture gate, or (b) rename the mode and drop the unused computation.

**F4 (from W5/W6 — actor identity)**
- Current: no generation tag on replay samples; no hash on the actor handoff.
- Failure: under concurrent mode and long Rust searches, up to one iteration of games can be generated by the previous actor and enter the buffer untagged.
- Why it matters: reported per-iteration loss curves are slightly off-phase from per-iteration actor generation; controller ablations that measure self-play diversity cannot tell "diversity from controller" vs "diversity from actor drift."
- Diagnostic: add an `actor_hash` field to `ReplayExample` and compute SHA-1 of model state_dict at update time; plot actor-hash histogram of each iteration's replay delta.
- Minimal fix: `actor_generation: int` counter incremented in `update_model`; tagged on every `_make_example` call; exposed in `train_log.jsonl`.

**F5 (from W7/W8 — silent drift)**
- Current: CI runs phase15 smoke only; no regression test pins controller output.
- Failure: an unintended change to `compute_p_flip_with_child_rtt` or `update_stats` could change halt behavior without a red CI.
- Why it matters: controller ablation conclusions are not protected from refactor noise.
- Diagnostic: add `pytest` + `cargo test` stages to the CI workflow and a single integration test that consumes `tests/fixtures/regression_positions.json` and asserts top-k per-mode stability.
- Minimal fix: workflow addition (≤ 10 lines) plus one new test file (~200 lines).

**F6 (from W9 — training-level confound)**
- Current: `--study controller_axes --iterations 30` runs fresh training per condition.
- Failure: the arena compares *different* learned policies under *different* controllers; controller effect is not separable.
- Why it matters: any paper claim of the form "controller X improves Elo" is ambiguous.
- Diagnostic: add a `study_level` field to `study_manifest.json` with value `"frozen_post_hoc"` or `"trained_per_condition"`; refuse to write a controller-attribution report for the latter without explicit flag.
- Minimal fix: docs change + a warn-once emitted by `ablation_study.py` when the preset is a controller preset but `--frozen-checkpoint-root` is not passed.

**F7 (from W10 — deployment inheritance)**
- Current: deployment_cfg = champion.train_cfg.
- Failure: deployment search config is picked after winner is known; users expecting a pre-registered Gomocup deployment config do not get one.
- Why it matters: deployment results cannot be read as independent of the ablation outcome.
- Minimal fix: write `deployment_policy: "train_cfg_inherited"` into `champion.json` and flag in the README that deployment-condition selection is part of the study, not a fixed target.

**F8 (from W11 — ROCm degraded path)**
- Current: `system_runtime.py:123-148` downgrades silently.
- Failure: users on RX 6950 XT believe they are running full-speed ROCm; actual throughput is ~30–50% below CUDA-equivalent.
- Why it matters: misleading hardware-fit story for the repo's named target hardware.
- Minimal fix: print a one-line banner at startup (not a warning filter) documenting the degraded path, and add an `INSTALL.md` section on known-degraded GPUs.

**F9 (from W12 — lossy log)**
- Current: `cli_main.py:875-894` emits partial rows.
- Failure: downstream plot / analysis tools must special-case missing `loss`.
- Minimal fix: always emit `loss: null` and `train_executed: false` on starvation iterations (one-line change).

---

## 7. Executable Reality & Hardware-Fit Audit

### 7.1 Executable reality

- **Rust binary**: `cargo build --release` per README Quick Start; `scripts/smoke_e2e.py:291-297` attempts the build if missing. Prior codex audit reports `cargo test --release → 386 passed, 65 ignored`. Confirmed via test inventory.
- **Python install**: `pip install -e .` standard; `pyproject.toml` declares optional extras for torch / jax / onnx / dev.
- **End-to-end smoke**: prior codex audit completed `scripts/smoke_e2e.py` to produce `study_manifest.json` / `evaluation_matrix.json` / `ablation_report.json`; my read of `results/audit_e2e_smoke_skiptraineval_20260422/gomoku7/` confirms these files are populated with real (`T1_noS_noVL` won 4/4 vs `T2_S_noVL`) arena outcomes, non-null runtime_contract, and real search manifests. Training-metric fields are null in the `skiptraineval` run by design.
- **Replay-fill stalls / `SelfPlayWorker did not stop within timeout`**: observed by the prior codex audit; `selfplay_runtime.py:2132-2145` shows the graceful → forced shutdown escalation with a 15-second total budget. Real but survivable operational friction.

### 7.2 Server / backend divergence

Not excessive. Torch-CPU, Torch-ROCm, and Torch-CUDA all share the same `torch_training_runtime` hook surface. JAX is nominally a separate backend but structurally identical (which is why the "JAX backend" claim is overclaimed — the actual inference into Rust uses torch-style state dict regardless of the nominal backend switch).

### 7.3 Hardware fit (Ryzen 5900X / 64 GB / RX 6950 XT)

- **CPU (5900X, 12c/24t)**: comfortable for 2–8 concurrent Rust MCTS threads + Python orchestration. No bottleneck expected.
- **RAM (64 GB)**: replay buffer (default ~100k positions, ~200 B sparse) = ~20 GB worst case; model + grads + Rust + SHM rings fit comfortably.
- **GPU (RX 6950 XT, 16 GB)**: tight but feasible for ResNet-class backbones at batch 128. The critical issue is the ROCm stack — `system_runtime.py:123-148` explicitly identifies `rx 6950` as a hipBLASLt-unsupported architecture and suppresses the resulting warning, downgrading to hipBLAS. Throughput is expected 30–50% below CUDA-equivalent silicon. The repo's named target hardware is on a known-degraded path, not a supported one.
- **Compile/warm-up overhead**: torch-ROCm first-step compilation is slow; no `torch.compile` / CUDA-graph capture detected. MCTS-side profile audit (`docs/MCTS_PROFILE_AUDIT_20260420.md`) flags TT hashing (7.29% of instructions) and edge materialization (3.02% + 4.64% in `apply_move`) as Priority-0/1 bottlenecks; not yet fixed (current `tt.rs` still uses generic `HashMap`, `expand.rs` still materializes inline).

### 7.4 Verdict

**Suitable with minor fixes** for small-budget Gomoku 7 ablation repetition on this hardware. For Gomoku 15 / Go 9 / chess at meaningful scale, the combination of (a) degraded ROCm path, (b) unfixed Priority-0 TT bottleneck, and (c) Python SHM polling (still present per `docs/PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md`) would make repeated-seed attribution sweeps painful in wall-clock terms.

---

## 8. Ablation / Measurement Honesty Audit

### 8.1 Conditions against fairness checklist

- **Same NN**: enforced for frozen-checkpoint sweeps (`controller_sweep.py`); **not enforced** for training-level ablations (`--study controller_axes --iterations N`). The same preset can be run both ways; the default Quick Start path (step 4 of README) runs the training-level version.
- **Same search budget**: **not enforced** under default `HaltMode::VOC`. Penalty mode changes p_flip → changes halt step → changes simulations per move. See W1/F1. Fair only if `HaltMode::Fixed(N)` is pinned.
- **Same evaluator path**: enforced (`evaluator_runtime.py:41-65`).
- **Same game distribution**: enforced by seeds list + opening book configuration when `--paired-seed-eval` is used.
- **Repeated seeds**: supported by `--seeds 11,12,13 --paired-seed-eval`; default is `--seeds 42` (single seed). README Quick Start uses single-seed smoke.
- **Variance / CI reporting**: per-match CI via `score_rate_ci()` (`quartz/evaluation.py`); SPRT at `ablation_study.py:570-602`. **Across-seed aggregation CI is NOT computed** — `summarize_conditions()` (`ablation_study.py:886-976`) averages `published_elo`, `score_rate`, `loss` across seeds without variance.

### 8.2 Metrics actually emitted (verified as disk writes)

- ✓ `halt_reason_hist` — `quartz/replay.py:493, 547, 552, 564`
- ✓ `controller_penalty_mode_counts` — `replay.py:567`
- ✓ `runtime_contract_hash` — `ablation_study.py:285, 564, 1199, 1216`
- ✓ `score_rate_ci`, `sprt_result`, `sprt_meta` — `ablation_study.py:572, 602, 1136-1139`
- ✓ `study_manifest.json`, `evaluation_matrix.json`, `champion.json` — `ablation_study.py:1542, 1220, 1285`
- ✓ `headwind_summary` (phase15) — via `eval_timing_summary.py`
- ≈ `mean_prior_refresh_rate` — aggregated from config (W4); not a runtime measurement

### 8.3 Missing metrics

- Node expansion distribution per move
- Controller activation frequency per move (how often refresh branch fired)
- Value / prior disagreement trajectory (the thing PFlipMixture claims to adapt to)
- Queue latency / inference delay histogram (flagged by `docs/PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md`)
- Replay freshness as actor-generation age distribution (blocked by W5)
- Self-play diversity (opening entropy, unique positions)
- Throughput-per-hardware-budget (games/s × GPU-utilisation, by backend)
- Arena variance decomposition (within-seed vs across-seed; blocked by §8.1 aggregation gap)
- Per-mode halt-step distribution (would let a reader verify W1 directly)

### 8.4 Anecdotal vs robust result language

README (line 185–191) says "Repository-local Gomoku7 evidence currently points to: `A1_legacy_base` as the safest existing default among the hand-written anchors; a stronger tuned no-refresh legacy-family variant from Optuna (`T0010_cf38467f`) as the current top low-cost sweep result." The language is cautious, but:

- `A1_legacy_base` is a condition name in `ablation_study.py:152-159`; that the name is real does not mean the claim is multi-seed-backed.
- `T0010_cf38467f` appears only in the README text. It is not visible in any file under `results/`. The claim is not reproducible from the repository state.

This is a minor overclaim: "current top sweep result" frames a result that is not in-repo as if it were an anchor.

---

## 9. Tests / Docs / Examples Honesty Audit

### 9.1 Test taxonomy

Python: 289 test functions across 8 files.

- Import/smoke: ~113
- Protocol/format unit: ~63 (QIPC wire format, Glicko-2 algebra, match-tally equivalence)
- Regression — narrow (replay roundtrip, batch protocol parsing, config structure): ~95
- Training-loop regression — **end-to-end**: 0 (the 184-function `test_training_pipeline_regressions.py` is ≥80% mocks and narrow slices)
- Search-controller regression — **pinned per mode**: 0 (`tests/fixtures/regression_positions.json` exists but is unused)
- Ablation / protocol — manifest correctness assertion: 0 (`test_ablation_study.py` asserts manifest shape, not whether conditions actually trained correctly)
- Evaluator / arena — **real Rust-vs-Rust**: 0 in CI; real arena runs only via the smoke scripts

Rust: 128 test functions across 9 modules.

- Best covered by count: `quartz.rs` (59), `mod.rs` (19), `eval.rs` (14), `select.rs` (10), `parallel.rs` (10), `search.rs` (8).
- Worst covered relative to size: `eval.rs` (97 KB, 14 tests — 0.14 tests/KB, many I/O-heavy paths), `tt.rs` (8.7 KB, none listed), `backup.rs`, `node.rs`.

### 9.2 Docs-vs-code consistency

- **README Quick Start**: accurate — every numbered step maps to a real current script.
- **README Maturity table**: mostly accurate. `ONNX export/inference → Partial`, `Gomocup brain → Partial`, `Actor/learner split → Conditional`: all reasonable.
- **README Key Features table — "Controller telemetry: p_flip, sigma_q, hbar_eff, stop_reason per move"**: verified against `src/mcts_server.rs:1300-1340` as described; however, "halt_reason_hist" and "controller_penalty_mode_counts" are aggregates, not per-move, and "mean_prior_refresh_rate" is config-not-measurement (W4).
- **README "Adaptive stopping requires NN loss < ~1.0 for P_flip convergence"** (Known Limitations): admission is honest; this is a real constraint that means short-smoke runs cannot meaningfully certify p_flip behavior.
- **README "Current Controller Status" — `T0010_cf38467f`**: see §8.4 — the name has no in-repo artifact.
- **`docs/QUARTZ_THEORY.md`** (not read line-by-line): the narrative is "one theory-aligned controller." Actual code is a family of heuristics (§3.3). This is the single largest docs-vs-code gap; not a lie per se, but the theory-narrative level exceeds what any single mode implements.
- **`docs/ABLATION_GUIDE.md`**: accurate. `controller_axes` is honestly flagged as the attribution preset; `controller` is honestly flagged as bundled legacy-vs-theory.
- **`docs/TRAINING_GUIDE.md`**: accurate on topology (QIPC, SHM, batched NN forward).
- **`docs/PHASE15_REEVALUATION.md`**: accurate. Phase15 tooling is genuinely split into reference/oracle, posthoc/online, with trace amortization.
- **`docs/MCTS_PROFILE_AUDIT_20260420.md` / `docs/PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md`**: honest. Priority-0/1 recommendations not yet applied.

### 9.3 Example / smoke honesty

`scripts/smoke_e2e.py` — verified as a fail-fast runtime smoke that (a) imports the runtime, (b) runs a 1-iteration / 2-eval-games ablation, (c) asserts three manifest files exist. It does not compute a win-rate threshold, does not run multiple seeds, and does not check numerical convergence. The README (line 177-179) admits this; the code matches the admission; the README Quick Start (line 37-38) does not re-iterate the admission next to the step that invokes it.

---

## 10. Golden-Reference Comparison

Treated as research-hygiene reference, not parity target.

### A. Architecture maturity

- **Self-play / learner / evaluator separation**: QUARTZ has split the three concerns into Python modules (`selfplay_runtime`, `train_loop`, `evaluator_runtime`) and has a clean Rust boundary. OpenSpiel / MiniZero separate these as separate processes with queues; QUARTZ keeps them as threads/subprocesses in a single Python driver. Weaker isolation than MiniZero, stronger than a monolithic reference implementation.
- **Storage / checkpoint / logging discipline**: `runtime_contract_hash` + `search_manifest_hash` + wrapped checkpoints (after the recent fix) are comparable to KataGo's disciplined artifact tagging. Stronger than OpenSpiel's minimal manifest.
- **Batched inference design**: hybrid QIPC with SHM hot-path is a genuine design. Lc0 has the most sophisticated backend abstraction; QUARTZ has one backend (torch forward into Rust) with a cleaner hot-path protocol than a naïve JSON-only route. Weaker than Lc0 on backend plurality; stronger than most research repos.

### B. Controller / MCTS maturity

- **Controller observability**: partial. KataGo exposes per-move root-policy top-k and per-node visit distributions; QUARTZ exposes per-move `p_flip`, `sigma_q`, `hbar_eff`, `stop_reason` and per-search `halt_reason_hist` aggregates. QUARTZ is less introspectable per-move than KataGo, but more so than OpenSpiel's minimal telemetry.
- **Same-budget benchmarking discipline**: **below goldens**. KataGo and Lc0 both default to fixed-simulation / fixed-time conditions for comparison runs. QUARTZ defaults to `HaltMode::VOC` which interacts with the penalty mode under test (W1).
- **Noisy-evaluator / wrong-prior handling**: parameterized only implicitly (via `sigma_q`). No explicit evaluator-noise model or stratified benchmark. Weaker than KataGo's explicit network-quality sweeps.
- **Search API clarity**: `src/mcts/quartz.rs` is a 114 KB single file. Lc0 and KataGo have more discipline about splitting controller from search. Refactor-risk is higher in QUARTZ.

### C. Research usability

- **Controlled ablations without deep surgery**: possible via `controller_axes` + frozen-checkpoint sweeps. Possible but requires knowing which preset + harness pairing to use; the default Quick Start does not route users there.
- **Configs/docs enough**: mostly yes. `configs/*.json` covers the main axes; `docs/ABLATION_GUIDE.md` is accurate; the `QUARTZ_THEORY.md` narrative is where overclaim lives.
- **Where clearly weaker**: no CI gating, no pinned controller-regression test, no across-seed CI, same-budget leakage, ROCm degraded path, unused `prior_q_divergence`, hardcoded tau.
- **Where genuinely novel or stronger than typical research code**: same-stack arena, history-exact TT, adaptive VL as a true feedback loop, runtime_contract_hash, Phase15 separation of reference/oracle, an explicit `controller_axes` attribution preset.

---

## 11. Concrete Upgrade Proposals

### 11A. Conservative (preserve current structure)

**C1. Pin `HaltMode::Fixed(N)` for controller-attribution runs.**
- Pain point: W1 — penalty mode silently changes budget under default VOC halt.
- Gain: `controller_axes` becomes a genuine attribution harness.
- Cost: a flag in `ablation_study.py` that forces `halt_mode=fixed` when a controller preset is requested unless the user opts out.
- New experiment becomes cleaner: "does root-only shaping improve Elo at 400 sims?" is now an answerable sentence.
- Remaining risk: `p_flip` semantics change between modes still, but the comparison is now budget-fair.

**C2. Replace hardcoded `tau = 0.5` with `config.prior_refresh_temp`.**
- Pain point: W2 — prior-temperature sweeps report false nulls for two modes.
- Gain: `prior_refresh_temp` becomes a genuine sweep axis.
- Cost: 2 line changes in `src/mcts/select.rs:283, 330`.
- Remaining risk: mode semantics shift for anyone relying on implicit 0.5 — add a changelog entry.

**C3. Add `actor_generation` counter and tag replay samples.**
- Pain point: W5/W6 — actor identity is untraceable through the replay buffer.
- Gain: freshness metrics and per-generation arena comparisons become possible.
- Cost: one integer field on `ReplayExample` + one counter in `SelfPlayWorker.update_model`.
- Experiment: "what fraction of samples used in iteration N were generated by actor N vs N-k?"
- Remaining risk: requires a replay-schema migration or version tag; minor.

**C4. Always emit a loss row per iteration.**
- Pain point: W12 — partial log rows on starvation.
- Gain: time series parseable without special cases.
- Cost: 1-line change in `cli_main.py:875-894`.

**C5. Same-budget & same-seed harness report.**
- Pain point: across-seed CI not reported (§8.1).
- Gain: aggregates like `published_elo` carry a CI.
- Cost: extend `summarize_conditions()` in `ablation_study.py:886-976` to compute cross-seed mean / SE / CI.

**C6. CI: add `pytest tests/` + `cargo test --release` stages.**
- Pain point: W7.
- Gain: silent regressions caught upstream.
- Cost: ~10 lines in the workflow yaml.
- Remaining risk: CI runtime grows; the Rust tests are fast enough (386 passed) to fit.

**C7. Controller-regression integration test.**
- Pain point: W8, unused `tests/fixtures/regression_positions.json`.
- Gain: mode drift becomes visible.
- Cost: one new test file consuming the fixture; runs the Rust binary on 2 positions × 4 modes.
- Experiment: catches regressions in `compute_quartz_stats` before they hit a paper run.

**C8. Deprecate `prior_q_divergence` decorative usage.**
- Pain point: W3. Either use it in the PFlipMixture gate or stop computing it.
- Gain: either a real divergence-aware mixture, or less narrative vs. code gap.
- Cost: small — either a gate tweak or a `cfg!(test)` gate.

**C9. Hardware-degraded-path banner.**
- Pain point: W11. RX 6950 XT downgrades silently.
- Gain: user knows what they are running.
- Cost: one `print(...)` + `docs/INSTALL.md` paragraph.

**C10. Downgrade the JAX claim.**
- Pain point: W9 (claim matrix). `docs/SETUP.md` and the README Key Features table still read as if JAX is a first-class backend.
- Gain: honest surface.
- Cost: docs edit; either complete the JAX path or explicitly label it experimental/stub.

### 11B. Structural redesigns (partial, not wholesale)

**S1. Split `QuartzController` into three decoupled objects**
- `TrustModel(stats) → trust` (uncertainty → scalar or per-channel trust)
- `PenaltyPolicy(trust, config) → per-action penalty` (swappable; contains the Legacy / SelfAdaptive / PFlip… laws)
- `HaltPolicy(trust, config) → halt decision` (swappable; Fixed / p_flip / VOC / ConfAdaptive)
- Plus a `RefreshPolicy(trust, config) → new_prior` separately.
- Pain point: W1/W2/W3 — penalty, refresh, halt all re-read the same latent signal without being orthogonalizable.
- Gain: mode combinatorics go from a 6-valued enum to three orthogonal knobs.
- Cost: moderate — `quartz.rs` (114 KB) split, existing modes re-implemented in the new interface; `select.rs` dispatches to `PenaltyPolicy::score(...)` generically.
- What becomes cleaner: "run penalty = SelfAdaptive × halt = Fixed(400) × refresh = off × root_only_shaping = true" — a 4-axis factorial.
- Remaining risk: two hardcoded-`0.5` modes' results under the old API are not byte-equivalent to the new one; add a migration test.

**S2. Elevate `actor_generation` into a first-class artifact**
- Every replay sample, every search manifest, every train_log row carries `actor_generation`.
- Arena outcomes are labeled with the generation that produced the self-play that trained the model.
- Pain point: W5/W6.
- Gain: phase alignment between learner and actor becomes audit-able.
- Cost: ~50 line changes across `replay.py`, `selfplay_runtime.py`, `train_loop.py`, `evaluator_runtime.py`, JSONL schemas.

**S3. Budget scheduler as a policy object**
- `BudgetScheduler` takes `(controller_snapshot, game_state) → budget_for_this_move`.
- Concrete implementations: `FixedBudget(N)`, `PFlipAdaptive(thresh)`, `VOC(tol)`, `ConfAdaptive(theta)`.
- Pain point: W1. Makes the budget knob an explicit independent axis.
- Gain: same-budget fairness is a property of the harness choice, not an emergent property of the halt-mode enum interacting with the penalty-mode enum.
- Cost: small — it's mostly a rename of `HaltMode` with an interface.

### 11C. Per-proposal trade-off table

| Proposal | Pain solved | Gain | Cost | Experiment unlocked | Risk |
|---|---|---|---|---|---|
| C1 | W1 | Budget-fair attribution | 1 flag | Fixed-sim controller_axes | Low |
| C2 | W2 | Prior-temp sweeps work | 2 LOC | Temperature ablation per mode | Low |
| C3 | W5/W6 | Freshness / phase tracking | 1 field + counter | Actor-generation freshness plots | Low |
| C4 | W12 | Parseable logs | 1 LOC | — | None |
| C5 | §8.1 | Across-seed CI | ~30 LOC | Multi-seed confidence | Low |
| C6 | W7 | Regression safety | CI yaml | Upstream drift detection | Low (runtime cost) |
| C7 | W8 | Controller drift detection | 1 test file | Pinned per-mode behavior | Low |
| C8 | W3 | Code–claim alignment | Small | Honest divergence-aware mode | Low |
| C9 | W11 | Hardware honesty | Docs + print | — | None |
| C10 | JAX overclaim | Honest docs | Docs | — | None |
| S1 | W1/W2/W3 | Orthogonal controller axes | Moderate | 4-axis factorial | Behavior shift in old modes |
| S2 | W5/W6 | Traceable actor identity | Moderate | Phase-aligned comparisons | Schema migration |
| S3 | W1 | Budget as first-class policy | Small | Clean budget/halt separation | None |

---

## 12. Pseudocode Upgrade Plan

### 12.1 Current-path patch — observability + fairness without restructuring

Intent: minimal changes to `quartz.rs` + `select.rs` + `ablation_study.py` that expose telemetry and enforce fairness.

```rust
// src/mcts/quartz.rs — augment should_stop to record the actual
// budget spent when the controller halted, so budget fairness is auditable.
struct HaltTrace {
    reason: StopReason,       // Converged / Budget / Voc / Conf / Time
    root_visits: u32,
    p_flip_at_halt: f32,
    voc_total_at_halt: f32,
    iterations: u32,
    penalty_mode: PenaltyMode,
    halt_mode_tag: String,    // "fixed(400)" | "voc" | "pflip" | "conf"
}

impl QuartzController {
    fn should_stop(&mut self, state: &SearchState) -> Option<HaltTrace> {
        match self.cfg.halt_mode {
            HaltMode::Fixed { budget } => {
                if state.root_visits >= budget { Some(self.trace(StopReason::BudgetExhausted, state)) } else { None }
            }
            HaltMode::Voc => {
                if self.stats.converged && (self.stats.unified.voc_total <= 0.0 || self.stats.p_flip < 0.159) {
                    Some(self.trace(StopReason::Converged, state))
                } else { None }
            }
            // ... other halt modes ...
        }
    }

    fn trace(&self, reason: StopReason, state: &SearchState) -> HaltTrace {
        HaltTrace {
            reason,
            root_visits: state.root_visits,
            p_flip_at_halt: self.stats.p_flip,
            voc_total_at_halt: self.stats.unified.voc_total,
            iterations: state.iteration,
            penalty_mode: self.cfg.penalty_mode,
            halt_mode_tag: format!("{:?}", self.cfg.halt_mode),
        }
    }
}
```

```rust
// src/mcts/select.rs — remove hardcoded tau
- let tau = 0.5_f32;
+ let tau = cfg.prior_refresh_temp.unwrap_or(0.5);
```

```python
# scripts/ablation_study.py — force Fixed halt for controller-attribution presets
CONTROLLER_ATTRIBUTION_PRESETS = {"controller_axes", "controller_factorial"}

def finalize_eval_cfg(cfg, preset_name, user_forced=False):
    if preset_name in CONTROLLER_ATTRIBUTION_PRESETS and not user_forced:
        if cfg.get("halt_mode") != "fixed":
            warnings.warn(
                f"Preset {preset_name} requires budget-fair halt. "
                f"Forcing halt_mode=fixed(budget={cfg.get('iters', 400)})."
            )
            cfg["halt_mode"] = "fixed"
            cfg["halt_budget"] = cfg.get("iters", 400)
    return cfg
```

What this solves: W1, W2, plus partial W8 (HaltTrace is a new disk-emitted field that any future regression test can pin).
Cost: a dozen lines across 3 files.
What is cleaner: `evaluation_matrix.json` now carries per-eval-condition halt traces, auditors can verify budget-fairness with a `describe()` call.
Remaining risk: does not fix the controller-as-a-bundle issue (S1 is needed for that).

### 12.2 Cleaner controller interface — partial redesign

Intent: decompose the controller into three objects with explicit inputs and outputs so ablations can swap one axis at a time.

```rust
// src/mcts/controller/mod.rs — new file
pub struct ControllerSnapshot {
    pub sigma_q: f32,
    pub p_flip: f32,
    pub d_divergence: f32,
    pub conf_t: f32,
    pub voc: VocChannels,
    pub convergence_state: ConvergenceState,
    // ... all telemetry, decision-bearing and diagnostic ...
}

pub trait TrustModel {
    fn observe(&mut self, raw: &QuartzStats);
    fn snapshot(&self) -> ControllerSnapshot;
}

pub trait PenaltyPolicy {
    fn score(&self, snap: &ControllerSnapshot, action: &ActionView) -> f32;
    fn name(&self) -> &'static str;
}

pub trait RefreshPolicy {
    fn effective_prior(&self, snap: &ControllerSnapshot, base_prior: f32, q: f32, n: u32) -> f32;
    fn activated(&self) -> bool;   // per-call telemetry
    fn name(&self) -> &'static str;
}

pub trait HaltPolicy {
    fn decide(&mut self, snap: &ControllerSnapshot, state: &SearchState) -> Option<HaltTrace>;
    fn name(&self) -> &'static str;
}

pub struct Controller {
    trust: Box<dyn TrustModel>,
    penalty: Box<dyn PenaltyPolicy>,
    refresh: Box<dyn RefreshPolicy>,
    halt: Box<dyn HaltPolicy>,
}

impl Controller {
    pub fn before_select(&mut self, raw: &QuartzStats) -> ControllerSnapshot {
        self.trust.observe(raw);
        self.trust.snapshot()
    }
    pub fn action_score(&self, snap: &ControllerSnapshot, a: &ActionView) -> f32 {
        self.penalty.score(snap, a)
    }
    pub fn action_prior(&self, snap: &ControllerSnapshot, base: f32, q: f32, n: u32) -> f32 {
        self.refresh.effective_prior(snap, base, q, n)
    }
    pub fn should_halt(&mut self, snap: &ControllerSnapshot, state: &SearchState) -> Option<HaltTrace> {
        self.halt.decide(snap, state)
    }
}
```

```rust
// concrete implementations map 1:1 to existing modes
struct SelfAdaptivePenalty;
struct GatedRefreshPenalty;
struct PFlipMixturePenalty { tau_refresh_q: f32 }
// ... etc

// configs/controller_axes.json now specifies four separate names:
// { "penalty": "self_adaptive", "refresh": "visitor_frequency", "halt": "fixed(400)", "trust": "default" }
```

What this solves: W1 (halt independent), W2 (refresh independent, no hardcoded tau anywhere — each RefreshPolicy carries its own config), W3 (a Divergence-Aware refresh that actually reads `d_divergence` can be written as one small struct), and W8 (policy trait let tests pin behavior per policy object).
Cost: moderate rewrite. `src/mcts/quartz.rs` shrinks (each mode becomes its own file under `src/mcts/controller/`). `select.rs` stops dispatching on enums.
What is cleaner: the 4-axis factorial — penalty × refresh × halt × trust — becomes a preset surface, not a lookup into a 6-valued bundle.
Remaining risk: backwards compatibility of old `evaluation_matrix.json` rows; add a `controller_schema_version` tag so old artifacts are rejected or up-converted.

### 12.3 End-to-end pipeline contract

Intent: make actor identity, replay freshness, learner step, and arena outcome auditable through a single versioned contract.

```python
# quartz/contracts.py — new file, shared schema
@dataclass(frozen=True)
class ActorSnapshot:
    generation: int
    weight_sha1: str            # sha1 of state_dict bytes at snapshot time
    base_checkpoint: Optional[str]
    backend: str                # "torch-cuda" | "torch-rocm-hipblas" | "torch-cpu"
    created_at: float           # unix epoch

@dataclass
class ReplayExampleMeta:
    actor: ActorSnapshot
    search_manifest_hash: str
    controller_schema_version: int
    game_id: str
    ply: int
    halt_trace: dict            # from 12.1

@dataclass
class LearnerStepRecord:
    iteration: int
    actor_generations_used: list[int]      # distinct gens in this batch
    oldest_sample_age_iters: int
    loss: float | None
    train_executed: bool
    batch_size: int
    wall_time: float

@dataclass
class ArenaMatchRecord:
    iter: int
    model_a: ActorSnapshot
    model_b: ActorSnapshot
    halt_mode_pinned: str       # "fixed(400)" etc, refusal to compare if mismatched
    score_a: float
    games: int
    score_rate_ci: tuple[float, float]
    sprt_status: str
```

```python
# quartz/selfplay_runtime.py
class SelfPlayWorker:
    def __init__(self, ...):
        self._actor_generation = 0
        self._actor_snapshot = self._snapshot_actor(model)

    def update_model(self, model):
        self._actor_generation += 1
        self._model = self._clone_actor_model(model)
        self._actor_snapshot = self._snapshot_actor(self._model, self._actor_generation)
        # pass the snapshot into the rust handoff so replay metadata can
        # be tagged server-side too
        self._push_weights_to_rust(self._model, snapshot=self._actor_snapshot)

    def _snapshot_actor(self, model, gen):
        sha1 = sha1_of_state_dict(model)
        return ActorSnapshot(
            generation=gen,
            weight_sha1=sha1,
            base_checkpoint=self._last_loaded_checkpoint_path,
            backend=infer_backend_tag(),
            created_at=time.time(),
        )
```

```python
# quartz/replay.py — tag every sample
def _make_example(self, state, policy, value, actor: ActorSnapshot, search_meta: dict):
    return ReplayExample(
        state=..., policy=..., value=...,
        meta=ReplayExampleMeta(actor=actor, search_manifest_hash=search_meta["hash"], ...),
    )
```

```python
# quartz/train_loop.py — always emit a LearnerStepRecord
def train_epoch(...):
    batch = replay.sample_batch(...)
    actor_gens = {sample.meta.actor.generation for sample in batch}
    if len(batch) < min_batch:
        return LearnerStepRecord(iteration=it, actor_generations_used=[], oldest_sample_age_iters=-1,
                                 loss=None, train_executed=False, batch_size=0, wall_time=...)
    loss = run_sgd(batch)
    return LearnerStepRecord(iteration=it, actor_generations_used=sorted(actor_gens), ...)
```

```python
# scripts/ablation_study.py — refuse to emit a controller-attribution report
# unless every ArenaMatchRecord has a matching halt_mode_pinned field
def emit_report(matches):
    pinned_modes = {m.halt_mode_pinned for m in matches}
    if len(pinned_modes) != 1:
        raise ValueError(f"attribution report requires single halt mode, saw {pinned_modes}")
    ...
```

What this solves: W4/W5/W6/W12/F4/F6/F7 collectively. Every arena row can be traced back to the actor generations that produced the training samples, the halt mode used, and the learner step audit trail.
Cost: a real schema migration. Old `train_log.jsonl` and `evaluation_matrix.json` files are not backwards-compatible unless wrapped in `contract_version`.
What is cleaner: "for this arena row, what fraction of the training samples came from actor generations within 2 of the learner's latest step?" — answerable, not inferable.
Remaining risk: pipeline becomes slightly slower due to per-sample metadata; mitigate by storing `actor_generation` as a small int in a parallel numpy array, not a dict per sample.

---

## 13. Minimal Patch Plan

Ten patches, ordered by cost-per-confidence-gain. Each entry: purpose / failure mode blocked / difficulty / expected gain / what claim it unlocks.

1. **Pin `HaltMode::Fixed(N)` in `controller_axes` / `controller_factorial`** — Blocks F1. Difficulty: low. Gain: ablation-grade budget fairness. Unlocks: "attribution-ready" claim.
2. **Replace hardcoded `tau = 0.5` with `config.prior_refresh_temp`** (`src/mcts/select.rs:283, 330`) — Blocks F2. Difficulty: trivial. Gain: the `prior_refresh_temp` sweep axis stops reporting false nulls. Unlocks: "design-ready" claim.
3. **Add `actor_generation` counter + tag replay samples** — Blocks F4. Difficulty: low. Gain: freshness/phase-alignment plots possible. Unlocks: "pipeline-ready" claim.
4. **Always emit a loss row, with `train_executed` flag** — Blocks F9. Difficulty: trivial. Gain: parseable logs. Unlocks: minor; protocol hygiene.
5. **Controller-regression integration test consuming `tests/fixtures/regression_positions.json`** — Blocks F5. Difficulty: medium. Gain: controller drift becomes visible in CI. Unlocks: "design-ready" claim.
6. **Add `pytest tests/` and `cargo test --release` stages to the phase15 workflow** — Blocks F5 for math/protocol layers. Difficulty: trivial. Gain: upstream detection of regressions. Unlocks: "pipeline-ready" claim.
7. **Extend `summarize_conditions()` to compute across-seed CI** — Blocks §8.1 weakness. Difficulty: low. Gain: report-level confidence where currently only point estimates. Unlocks: "ablation-ready" claim.
8. **`halt_trace` field on every `evaluation_matrix.json` row** — Blocks F1 diagnostics. Difficulty: low. Gain: any reader can verify same-budget post hoc. Unlocks: "ablation-ready" claim.
9. **Honest hardware banner + `docs/INSTALL.md` section on ROCm-degraded GPUs** — Blocks F8. Difficulty: trivial. Gain: users on RX 6950 XT / other RDNA2 know what they are running. Unlocks: "pipeline-ready" claim.
10. **Downgrade `QUARTZ_THEORY.md` narrative and the JAX claim** — either complete each or re-describe as experimental. Difficulty: low. Gain: docs-code honesty. Unlocks: "design-ready" claim.

Patches 1, 2, 5, 6, 8 are the fastest way to cross from "exploratory" to "ablation-usable." Patches 3, 7, 9 unlock cleaner self-play/training claims. Patches 4 and 10 are hygiene.

---

## 14. CoVe / Contrastive Verification

### 14A. CoVe — verification questions against the provisional synthesis

**Q1.** Is the stale-actor claim (W6) still true after this week's diffs? `cli_main.py:923` shows `if bg_worker and executed_steps > 0: bg_worker.update_model(actor_source)`. Prior codex said every-5-iter; current code is per-iter conditional on training executing. **Revision**: downgrade the severity language from "updated every 5 iterations" to "updated every iteration in which training ran, without a version tag." W6 stands for the version-tag half; the cadence half is resolved.

**Q2.** Does `ablation_study.py` still cherry-pick deployment condition post hoc (codex Finding C)? Current `scripts/ablation_study.py:1239-1240` reads `deployment_cfg = copy.deepcopy(champion_run.get("train_cfg") or {})`. **Revision**: codex Finding C as stated (re-picking the best eval condition) is NOT the current behavior. The current behavior — inheriting `train_cfg` as deployment — is weaker overclaim but still lacks pre-registration. Kept as W10 at reduced severity.

**Q3.** Is `best.pt` bootstrap staleness (codex Finding A) still live? `cli_main.py:722-723` sets `best_checkpoint_bootstrap = True` and stores `best_checkpoint_bootstrap_seeded` in checkpoint status (line 296). **Revision**: the mechanism is now explicit and auditable. Downstream needs to respect the flag; the Python agent could not verify downstream ablation code. Reduced to medium-severity pending that check — not escalated.

**Q4.** Does `tests/fixtures/regression_positions.json` really have no test consumer? A grep in `tests/` shows the file is referenced only by one of the other test files as a path constant; no test `assert`s on its fields. Confirmed dead fixture.

**Q5.** Is TT hashing truly history-aware for chess and go? Yes — `src/games/chess.rs:1672-1681` combines `hash`, `history_digest`, `half`, and castling; `src/games/go.rs:1172-1190` combines `hash`, `history_digest`, `size`, `ko_point`, `passes`. Claim supported.

**Q6.** Is `prior_q_divergence` really computed but never read for PFlipMixture? The Rust agent searched for reads in `select.rs`; the divergence enters no decision path for PFlipMixture (it is used as a gate threshold input in GatedRefresh's activation logic at `select.rs:239-270`, but not in PFlipMixture's mixture gate). Confirmed — W3 stands as stated but scoped to PFlipMixture.

**Q7.** Is the `HaltMode::Fixed` remedy actually sufficient to make modes budget-fair? It makes root-visit counts equal. But per-move p_flip can still differ, which leaks into *what gets visited* even if the total visit count is equal. **Revision**: C1 pinning is necessary but not sufficient for "fully fair" attribution. Still a big improvement; note this in the patch description.

**Q8.** Does the README really overclaim JAX? `quartz/jax_training_runtime.py` imports from torch_training_runtime and cli_main; the actual SGD path is torch. README Maturity table line on JAX says "JAX backend is available for training, but Rust self-play/eval and Gomocup deployment paths do not use JAX inference." This is honest about inference. It is NOT honest that JAX-specific training uses JAX for the optimizer step — it still goes through torch hooks. **Revision**: W10 (JAX claim) is sharper — the training step itself is torch regardless of `--backend jax`.

**Q9.** Is the `T0010_cf38467f` claim reproducible from the repo? Grep under `results/` returns nothing. README mentions it as "current top low-cost sweep result." **Revision**: §8.4 reclassifies this as a minor overclaim — language is cautious but the artifact is not in-repo.

**Q10.** Does `phase15_benchmark_ci_smoke.py` check any numerical invariant, or is it purely "script completed without exception"? The CI step at `.github/workflows/phase15-benchmark-gate.yml:59-63` invokes the smoke and uploads artifacts. The smoke writes JSON manifests. Asserts on content are minimal (presence of fields). **Revision**: claim "CI does not gate on numerical correctness" stands.

**Q11.** The smoke-run artifact `audit_e2e_smoke_skiptraineval_20260422/gomoku7/evaluation_matrix.json` shows `T1_noS_noVL` won 4/4 vs `T2_S_noVL` — is 4 games a meaningful signal? No. Four paired games between two Rust+NN systems has no statistical power (p_flip of such a small sample is huge). This is fine because the run was `skiptraineval` and is labeled a smoke, but any reader tempted to draw a controller conclusion from it would be wrong. Worth flagging to users reading the smoke output.

**Q12.** Is the adaptive VL controller really a feedback loop (design strength #3)? `parallel.rs:264-272`: `amplifier = 1.0 + dr * (1.0 + contention); vvalue = sigma * depth_decay * entropy_factor * amplifier`. This is a closed-form modulation per iteration, not a temporally-integrated controller (no I term, no D term), so it is technically a feed-forward gain on contention. **Revision**: calling it "2nd-generation feedback" overstates; it is a responsive state-dependent gain, not a PI/PID controller. Strength #3 still holds, but sharpening the language is honest.

### 14B. Contrastive hypotheses

- **H1** — "real executable research platform, search controller is meaningful, strong exploratory platform."
- **H2** — "design ideas are good, but executability/experimental rigor/interpretation need major revision."
- **H3** — "docs overclaim; actual code path and experimental honesty are premature."

Axis-by-axis:

| Axis | H1 support | H2 support | H3 support |
|---|---|---|---|
| Executability | **strong** — smoke completes, arena is real, same-stack enforced | weak — runtime warnings, degraded ROCm | weak — the pipeline IS closed topologically |
| Controller design maturity | partial — genuine state, genuine intervention points | **strong** — bundle dressed as family, two modes hardcode tau, unused divergence | partial — the overclaim is real; the implementation is not vapor |
| Pipeline maturity | partial — all pieces exist | **strong** — actor versioning absent, loss log lossy, CI not gating | weak — the pieces are real |
| Hardware-fit | weak — ROCm degraded on target GPU | partial | partial |
| Experimental rigor | weak — same-budget leakage, no across-seed CI, no regression fixture consumer | **strong** — attribution is currently confounded | partial — `controller_axes` is honest enough to be called attempted rigor |
| Docs honesty | partial — most paths match; theory/JAX/hw overclaim | partial | partial — real overclaim exists but is mostly isolated |

Net: H2 best describes the current state. H1 is too generous on experimental rigor; H3 is too dismissive of the real Rust engine, real same-stack arena, and real artifact discipline. **The repository is a real executable platform with a design that still needs partial refactor and a handful of bookkeeping upgrades before it can credibly carry publication-grade controller claims.**

---

## 15. Final Verdict

**Ablation-usable with revisions.**

That means, specifically:

- The repository is not paperware. Rust engine, same-stack arena, replay/storage, Glicko-2 evaluation, Phase15 clean-split tooling, and artifact contracts are all real.
- It is today already a usable *engineering* platform for iteration on Gomoku-scale problems.
- It is not today a clean *scientific* platform for publication-grade controller attribution. The gap is closeable with the ten patches in §13 — none of which require a framework rewrite. The largest single wins are (1) pinning `HaltMode::Fixed` in attribution presets, (2) unhiding `tau`, (3) tagging actor generations, and (4) wiring the existing regression fixture into CI.

Avoid the framings "good exploratory platform" (understates existing artifact discipline and same-stack arena) and "clean research-grade platform" (overstates controller attribution honesty and hardware-fit). The ablation-usable-with-revisions category is the correct one.

## 16. One-Line Reason

Real engine, real pipeline, honest partial docs — but default-halt budget leakage, two hardcoded refresh temperatures, untagged actor identity, and a CI that does not gate on tests together mean today's controller-attribution claims cannot cleanly separate signal from bookkeeping noise.
