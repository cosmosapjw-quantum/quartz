# QUARTZ Integrated Audit (Current Workspace, 2026-04-22)

기준:
- 작업 디렉터리: current workspace tree
- 평가 우선순위: code-path reality > docs claim > test volume
- 실제 확인:
  - `venv/bin/python -m quartz.train --help` 성공
  - `venv/bin/python -c "from quartz.evaluation import _run_all; _run_all()"` 성공
  - `pytest -q tests/test_training_pipeline_regressions.py tests/test_evaluation_pipeline_regressions.py tests/test_phase15_ablation.py` 결과: `223 passed, 2 failed`
  - `venv/bin/python scripts/smoke_e2e.py --output results/audit_e2e_smoke_codex` 실행 시 기본 self-play 경로에서 실패 후 replay-fill 상태로 정체
  - `cargo test -q test_gated_refresh_opens_on_prior_divergence` 성공
  - `cargo test -q test_1a_halt_mode_default_is_voc` 성공

## 1. Project Claim Reconstruction

### Claimed capabilities

| Claim | Provisional tag |
|---|---|
| Rust MCTS core | clearly implemented |
| QUARTZ controller with adaptive halt | clearly implemented |
| Rust+NN self-play training path | partially implemented |
| Rust+NN evaluation path | clearly implemented |
| AlphaZero-style replay/learner/checkpoint loop | clearly implemented |
| Glicko-2 promotion pipeline | clearly implemented |
| Training-level ablation runner | clearly implemented |
| Frozen-checkpoint controller search | clearly implemented |
| Phase 1.5 clean-split assay layer | clearly implemented |
| Gomocup export/build flow | partially implemented |
| ONNX deployment flow | partially implemented |
| Multi-game support (gomoku/go/chess) | partially implemented |
| One-command e2e smoke | overclaimed on current tree |
| “same Rust+NN stack for training/eval” | partially implemented |
| Controller attribution cleanliness | weakly evidenced |

### Independent judgments

- end-to-end executability: partial
- controller state semantics: rich but noisy
- MCTS modification scope: substantial
- evaluator/training coupling: real
- ablation readiness: medium
- hardware realism: plausible
- docs/tests honesty: mixed
- reproducibility: medium, not fully closed

## 2. Executable Path Reconstruction

### Actually load-bearing path

1. `python -m quartz.train`
2. `quartz/train.py` chooses torch/jax runtime
3. `quartz/cli_main.py` builds config, selects runtime modes, opens replay/checkpoint paths
4. self-play path goes through `quartz/torch_training_runtime.py:selfplay_rust_nn_batched()`
5. that delegates into `quartz/selfplay_runtime.py:_selfplay_rust_nn_batched_impl()`
6. Rust server launched by `quartz/qipc.py:launch_rust_server()`
7. Rust server path lives in `src/mcts_server.rs` behind `mcts_demo --server`
8. replay stored in `quartz/replay.py`
9. learner update in `quartz/train_loop.py:train_epoch()`
10. checkpoint/promotion via `quartz/evaluation.py:TrainingEvaluator`
11. Rust evaluation engine in `quartz/evaluator_runtime.py:RustNNEvaluatorEngine`
12. experiment harness in `scripts/ablation_study.py`

### What is truly runnable now

- CLI/help path: runnable
- evaluation math + promotion self-tests: runnable
- selected Rust controller/unit tests: runnable
- Python evaluation pipeline tests: runnable
- phase15 Python tests: runnable

### What is described but not presently closed

- canonical concurrent training smoke
- default Rust self-play state-machine training path
- one-command end-to-end smoke as advertised in README/docs

### Broken handoffs

- Default self-play mode selection in [quartz/cli_main.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/quartz/cli_main.py:333) selects `rust_selfplay_state_machine` whenever the Rust binary exists and the game is supported.
- Support detection in [quartz/selfplay_runtime.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/quartz/selfplay_runtime.py:2427) returns true for essentially all supported games.
- The chosen path immediately crashes because [quartz/torch_training_runtime.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/quartz/torch_training_runtime.py:190) constructs `BatchedSelfPlayRuntimeHooks` with a field that the dataclass no longer accepts ([quartz/selfplay_runtime.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/quartz/selfplay_runtime.py:304)).

## 3. Search Controller / MCTS Architecture Reconstruction

### Core controller state

`QuartzStats` is not cosmetic. It carries:
- `hbar_eff`
- `p_flip`
- `sigma_q`
- `rho_hat`
- `p_envar`
- `prior_q_divergence`
- `flip_stable`
- `conf_t`
- unified VOC channel scores

### What changes search policy

- `search_profile` can disable QUARTZ and VL entirely in `src/mcts_server.rs:1350`
- `penalty_mode` changes selection semantics in `src/mcts/select.rs`
- `root_only_shaping` changes whether QUARTZ shapes only root or shallow tree
- `prior_refresh_rate/temp` changes effective priors
- `enable_fisher_puct` changes prior weighting

### What changes termination

- `HaltMode` in `src/mcts/quartz.rs`
- `QuartzController::should_stop()` in [src/mcts/quartz.rs](/home/cosmosapjw/Dropbox/personal_projects/quartz/src/mcts/quartz.rs:1860)
- hard budget/time caps in `src/mcts/quartz.rs:1868-1876`

### What changes prior usage

- manual refresh path
- divergence-gated visit-share refresh
- PFlipMixture refresh
- SelfAdaptive visit-frequency refresh

### What changes evaluator trust

There is no clean explicit uncertainty adapter object. Trust is inferred indirectly through:
- `p_flip`
- `sigma_q`
- `hbar_eff`
- `prior_q_divergence`
- heavy-tail logic

### Provisional architecture verdict

This is not a unified controller in the strict sense.

Best description:
- unified vocabulary on top
- heuristic bundle in implementation
- experiment-driven patchwork in branching surface

## 4. Design Strengths

1. Controller insertion points are real, not decorative.
2. Stop semantics are explicit and typed (`StopReason`).
3. Search manifest hashing is unusually serious for a repo of this size.
4. Shared Rust evaluation client enforces manifest matching instead of silently mixing configs.
5. Replay traces carry search/controller metadata, not only outcome targets.
6. There is a genuine frozen-checkpoint experimentation layer, not just train-and-pray loops.
7. Phase 1.5 contracts and cache-salt discipline are stronger than typical hobby RL code.

## 5. Structural Weaknesses

1. The default end-to-end path is broken today, so docs overstate executability.
2. Controller semantics are spread across `search_profile`, `vl_mode`, `penalty_mode`, `halt_mode`, `root_only_shaping`, refresh params, and experimental gates.
3. Penalty, refresh, halt, and “trust” all reuse overlapping latent signals (`p_flip`, `sigma_q`, `hbar_eff`, divergence).
4. Root-only versus shallow-tree shaping is bundled into user-facing controller comparisons, which muddies attribution.
5. The main Rust binary mixes server entry, research harness, and legacy acceptance logic in a 6k+ line file.
6. Python runtime is split into modules, but a 1.5k-line compatibility facade still leaks assumptions everywhere.
7. Replay telemetry helpers depend on concrete buffer internals rather than a replay contract.
8. Experimental resident-session/selfplay modes are close to default behavior, not isolated side paths.
9. There is strong contract metadata, but no CI gate on the canonical smoke path that README advertises.
10. The controller theory doc talks like a clean reduction, but `QuartzConfig` still exposes a wide heuristic surface.

## 6. Failure-Mode Analysis

### A. Broken default self-play path

- current mechanism: CLI defaults to Rust self-play state machine
- failure mode: TypeError on runtime hook construction
- why it matters: training smoke and ablation runner do not close end-to-end
- minimal diagnostic: current smoke + failing test
- minimal fix: align `BatchedSelfPlayRuntimeHooks` constructor with all callers, then gate with one real smoke test

### B. Replay abstraction leak

- current mechanism: `ReplayMetrics.search_summary()` reads `replay.buf[idx]` directly
- failure mode: any alternate replay implementation that satisfies `__len__` but not `.buf` breaks logging
- why it matters: contracts are weaker than module boundaries imply
- minimal diagnostic: current failing regression test
- minimal fix: add replay iterator/sampling accessor and stop peeking at `.buf`

### C. Slow failure detection during replay fill

- current mechanism: background worker failure is surfaced only through periodic polling and stall age
- failure mode: canonical smoke sits in replay fill printing `err=n` without forward progress
- why it matters: “one-command smoke” can hang instead of fail fast
- minimal diagnostic: current smoke run
- minimal fix: abort replay fill immediately once the worker records repeated bootstrap errors

### D. Attribution drift in controller studies

- current mechanism: controller families change multiple factors at once
- failure mode: win-rate deltas are not clean controller deltas
- why it matters: search-controller claims become hard to publish honestly
- minimal diagnostic: manifest diff by adjacent condition
- minimal fix: make `controller_axes` the primary preset and demote bundled presets

### E. Theory/implementation gap

- current mechanism: docs frame QUARTZ as unified and reduced
- failure mode: users infer theoretical cleanliness not present in config/control surface
- why it matters: experimental honesty degrades
- minimal diagnostic: compare doc language to actual `QuartzConfig`
- minimal fix: explicitly label QUARTZ as a controller family with bundled experimental switches

## 7. Executable Reality & Hardware-Fit Audit

### Runtime reality

- The workspace has a real Rust binary and real GPU-aware training stack.
- Evaluation-only self-test passes.
- The concurrent self-play bootstrap path fails on current tree.
- Therefore install/build/import success does not imply end-to-end training readiness.

### Hardware fit on this machine

Observed from smoke startup:
- CPU: 24 logical / 12 physical
- RAM: ~64 GB
- GPU: AMD Radeon RX 6950 XT, ~16 GB VRAM

This hardware is enough for the intended Gomoku-scale studies.
The problem is not raw hardware fit. The problem is path closure and orchestration stability.

### 4.3 verdict

`exploratory only`

Reason:
- hardware is suitable
- controller/MCTS core is real
- current default training path is not reliably executable end-to-end

## 8. Ablation / Measurement Honesty Audit

### What is good

- paired-seed evaluation exists
- evaluation matrix stores CI and SPRT metadata
- search manifests are hashed and stale eval rows are discarded
- strict reference profile exists
- frozen-checkpoint controller search is a better isolation layer than training-only ablations

### What is still weak

- training-level controller comparisons are not same-NN comparisons
- default study presets still expose bundled comparisons prominently
- controller telemetry remains partial: `refresh_count`, `penalty_sum`, and full halt histograms are not first-class everywhere
- queue latency / inference delay / replay freshness are not unified into the main fairness report

### Missing or underpowered metrics

- controller activation frequency
- refresh activation frequency
- penalty accumulation
- queue latency
- inference delay distribution
- self-play diversity beyond coarse replay freshness
- throughput per hardware budget
- regret-like search quality proxy

### Overclaim risk

The repo is no longer blatantly anecdotal, but README/documentation still read closer to “strong exploratory platform already closed” than current executability justifies.

## 9. Tests / Docs / Examples Honesty Audit

### Test classification

- import/smoke: `test_play_gui.py`, parts of `test_batch_protocol.py`
- unit/regression: large parts of `test_training_pipeline_regressions.py`
- evaluation/arena: `test_evaluation_pipeline_regressions.py`
- ablation/protocol: `test_ablation_study.py`, `test_controller_sweep.py`, `test_controller_optuna.py`
- phase15/protocol: `test_phase15_ablation.py`

### What the current test evidence says

- There is substantial regression coverage.
- The suite is not green on the selected core paths: 2 failures remain, and one is exactly the default self-play path that breaks the smoke.
- So the tests are useful, but they currently refute the strongest end-to-end claim.

### Docs vs code

- README quick-start smoke currently overclaims.
- Maturity table overstates “implemented” for Rust self-play at current HEAD.
- The docs are much more honest than earlier styles about bundled comparisons, but still too optimistic on closure.

## 10. Golden-Reference Comparison

### OpenSpiel / MiniZero

Compared with these, QUARTZ is weaker on clean actor/learner/server separation.
It is stronger than many small repos on experiment metadata and manifest discipline.

### KataGo

Compared with KataGo, controller sophistication is rhetorically closer than experimentally.
KataGo-level strength here would require much better observability and tighter same-budget benchmarking.

### Lc0

Backend/search separation is directionally similar, but QUARTZ still has much more orchestration coupling and compatibility-layer debt.

### Net judgment

This is below the golden references in maturity, but above a toy project in experimental scaffolding.

## 11. Concrete Upgrade Proposals

### Conservative upgrades

1. Fix the broken self-play runtime-hook contract and add a smoke test that exercises it.
2. Add fast-fail replay-fill bootstrap logic in training CLI.
3. Make replay telemetry consume an interface instead of `.buf`.
4. Emit `refresh_count`, `penalty_sum`, and halt histograms as first-class controller telemetry.
5. Make `controller_axes` the default attribution preset in docs/examples.
6. Add a single CI job that runs `scripts/smoke_e2e.py` on a tiny CPU config.

### Partial redesigns

1. Split controller into `observe -> infer_trust -> decide_penalty -> decide_refresh -> decide_halt`.
2. Introduce a search-policy object separate from stop-policy object.
3. Contractize replay/search telemetry as typed records rather than free-form dicts.
4. Move Rust server entry out of `src/main.rs` monolith into narrower binaries/modules.

## 12. Pseudocode Upgrade Plan

### 12.1 Current-path patch

```text
every check_interval:
  stats = observe_root(root, priors)
  telemetry = summarize_controller(stats, applied_refresh, applied_penalty)
  if halt_policy.should_stop(stats, elapsed_ms):
    return stop(telemetry)
  selection_policy = selection_policy.with_stats(stats)
```

Fixes:
- observability gap
- halt vs selection coupling ambiguity

### 12.2 Cleaner controller interface

```text
obs = controller.observe(root_state, root_snapshot, elapsed_ms)
trust = controller.infer_trust(obs)
refresh = controller.decide_refresh(obs, trust)
penalty = controller.decide_penalty(obs, trust)
halt = controller.decide_halt(obs, trust)
score = base_puct(edge) + penalty(edge, obs, trust, refresh)
```

Fixes:
- heuristic bundle feel
- theory/implementation mismatch

### 12.3 End-to-end pipeline contract

```text
selfplay_worker:
  request_search(model_id, search_manifest, position_batch)
  receive(search_result_batch, controller_telemetry_batch)
  append_replay(examples, telemetry)

learner:
  sample_replay(contract)
  update_model()
  checkpoint(model_id, replay_state, runtime_contract)

evaluator:
  require_same(search_manifest_hash)
  run_match_matrix()
  publish(promotion, ci, sprt, runtime_contract)
```

Fixes:
- path ambiguity
- eval/train drift
- weak replay/search contract boundaries

## 13. Minimal Patch Plan

1. Repair `BatchedSelfPlayRuntimeHooks` callsite/dataclass mismatch.
2. Add a real smoke test for the default self-play path.
3. Fail replay bootstrap immediately after repeated worker bootstrap errors.
4. Add replay telemetry accessors so metrics stop reaching into `.buf`.
5. Promote `controller_axes` in docs and demote bundled presets.
6. Emit full controller activation telemetry.
7. Split `src/main.rs` server path from acceptance/demo harness.
8. Downgrade README maturity wording until the smoke path is green.

## 14. CoVe / Contrastive Verification

### Questions that could break the provisional conclusion

1. Is the Rust controller core itself broken?  
Answer: selected Rust tests passed, so not obviously.

2. Is evaluation path fake?  
Answer: evaluation self-test and evaluator runtime are real.

3. Is the self-play failure only a test double artifact?  
Answer: no, the same TypeError surfaced in the real smoke.

4. Could the smoke have eventually succeeded?  
Answer: not with the observed TypeError in the worker bootstrap path.

5. Are docs fully dishonest?  
Answer: no. They are partly honest about bundled comparisons and fallback paths.

6. Is the controller purely decorative?  
Answer: no. It directly changes selection and stopping.

7. Is the project unresearchable?  
Answer: no. Frozen-checkpoint and phase15 layers are genuinely useful.

8. Is it already a clean research-grade platform?  
Answer: no. Current path closure and attribution cleanliness are not at that bar.

### H1 / H2 / H3

- H1 “strong exploratory platform already executable”: too optimistic
- H2 “good ideas, but major revision needed on executability/interpretability”: best fit
- H3 “mostly premature marketing”: too harsh, because substantial real machinery exists

## 15. Final Verdict

`ablation-usable with revisions`

## 16. One-line Reason

Rust MCTS, shared Rust evaluation, ablation manifests, and phase15 tooling are real, but the current default self-play/training path is broken and the controller remains too bundled for clean attribution claims.
