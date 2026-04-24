# QUARTZ Integrated Audit

Date: 2026-04-23

Scope: repository-wide audit against `audit.md`, with code-path reconstruction, controller/MCTS review, pipeline maturity review, experimental-honesty review, and minimal upgrade roadmap.

Verification performed:
- `pytest -q tests/test_training_pipeline_regressions.py tests/test_evaluation_pipeline_regressions.py tests/test_ablation_study.py tests/test_batch_protocol.py tests/test_phase15_ablation.py` -> `254 passed`
- `cargo test --release` -> `386 passed`, `65 ignored`
- `python scripts/smoke_e2e.py --output /tmp/quartz_audit_smoke` -> completed and produced `/tmp/quartz_audit_smoke/gomoku7/{study_manifest.json,evaluation_matrix.json,ablation_report.json}`, but runtime showed replay-fill stalls and `SelfPlayWorker did not stop within timeout`

## Executive Verdict

This project is not paperware. There is a real Rust search engine, a real Python training/evaluation orchestrator, a real post-train ablation harness, and a real mixed Rust+NN execution path. The repository has unusually strong regression coverage for a research codebase, and the current tree is executable in the narrow sense.

The main problem is not "nothing works." The main problem is that several load-bearing research claims are less clean than the docs suggest:
- the controller is a family of materially different heuristics, not one experimentally isolated principle,
- concurrent training uses a stale actor snapshot for self-play,
- short/no-promotion ablations can evaluate the wrong checkpoint (`best.pt` seeded before training),
- deployment search config is selected post hoc from the same evaluation matrix used to declare the winner.

So the honest status is:
- runnable research platform: yes
- closed AlphaZero-style learner/search/eval loop: mostly yes, but with a loose actor-refresh contract
- controller attribution platform: partially yes
- publication-grade controller evidence without qualification: not yet

## Phase 0: Claim Reconstruction

Repository-level claims and provisional tags:

| Claim | Provisional tag |
| --- | --- |
| Rust MCTS engine is real and load-bearing | clearly implemented |
| Rust search is the actual training/eval substrate | clearly implemented |
| Python training loop closes self-play -> replay -> SGD -> evaluation | partially implemented |
| QUARTZ controller is implemented in production search paths | clearly implemented |
| Controller is a single theory-aligned design | overclaimed |
| Multiple controller modes are ablation-ready | partially implemented |
| Same-stack evaluation uses Rust+NN, not a toy fallback | clearly implemented |
| Multi-game support (Gomoku/Go/Chess) is real | clearly implemented |
| Exact TT handling for chess/go rule state is real | clearly implemented |
| ONNX/Gomocup deployment exists | partially implemented |
| Phase15 clean-split tooling exists | clearly implemented |
| Controller benchmarking is fair by default | partially implemented |
| Docs/test surface is generally honest | mostly implemented |
| Canonical smoke demonstrates benchmark readiness | overclaimed if read literally; the code treats it as a runtime smoke |

Independent status summary:
- End-to-end executability: good
- Controller state semantics: rich but not cleanly unified
- MCTS modification scope: real and load-bearing at root, optional shallow-tree bleed-through in legacy mode
- Evaluator/training coupling: real, but actor freshness is coarse
- Ablation readiness: decent for engineering iteration, weaker for clean causal attribution
- Hardware realism: moderate; CPU and mixed local paths are real, GPU/JAX claims stay conditional
- Docs/tests honesty: mostly good, with a few important mismatches
- Reproducibility: above average for a research repo, thanks to manifests/hashes/contracts

## Phase 1: Executable Path Reconstruction

### What is actually wired

Training path:
- `quartz/train.py` dispatches to `quartz.torch_runtime` or `quartz.jax_runtime`.
- `quartz/cli_main.py` prepares config/runtime topology and runs the actual loop.
- Self-play goes through Rust via `quartz.selfplay_runtime.selfplay_rust_nn_batched(...)`.
- NN inference is served back to Rust via QIPC/SHM in `quartz/qipc.py` and `quartz/runtime_support.py`.
- Replay storage is real (`quartz/replay.py`) and learner updates are real (`quartz/train_loop.py`).
- Evaluation is real and Rust-backed via `quartz.evaluator_runtime.RustNNEvaluatorEngine`.
- Ablations run real training subprocesses and real post-train Rust-vs-Rust evaluation (`scripts/ablation_study.py`).

Search path:
- Rust engine core: `src/mcts/mod.rs`, `src/mcts/select.rs`, `src/mcts/expand.rs`, `src/mcts/backup.rs`, `src/mcts/parallel.rs`
- QUARTZ stats/stop logic: `src/mcts/quartz.rs`
- Server/runtime glue: `src/mcts_server.rs`

Deployment path:
- ONNX export is real on Python side (`quartz/onnx_support.py`, `quartz/gomocup_export.py`)
- Gomocup binary path is real on Rust side when built with `--features onnx`

### What is not as closed as the naming suggests

- `quartz/alphazero_train.py` is now a compatibility facade, not the primary architecture. This is fine, but external readers should not treat it as the canonical implementation center.
- The canonical smoke is a runtime smoke, not a learner-readiness certification. In the executed smoke, each condition produced replay and artifacts, but no actual SGD loss row was emitted because replay never reached the train batch threshold.
- The self-play worker and teardown path are operational but still rough: the smoke completed only after long replay-fill stalls, and both training subprocesses warned `SelfPlayWorker did not stop within timeout`.

## Findings

### 1. High: short or no-promotion ablations can evaluate the wrong checkpoint

Current mechanism:
- `quartz/cli_main.py:657-661` seeds `best.pt` before training starts if it does not already exist.
- `scripts/ablation_study.py:371-376` resolves model paths by preferring `best.pt` over `latest.pt`.
- `scripts/ablation_study.py:729-747` builds evaluation engines from that resolved path.

Failure mode:
- If a run does not hit a promotion event, `best.pt` remains the pre-training snapshot.
- Post-train evaluation, champion selection, and Gomocup export can therefore use the initial model instead of the trained latest model.
- This is especially damaging in the exact path used by the canonical smoke, where `--eval-interval 999999` suppresses internal promotion, so `best.pt` stays stale by construction.

Why it matters:
- It directly corrupts ablation conclusions.
- It can make short studies compare random-init or stale models while reporting them as trained outputs.
- It also contaminates downstream champion/export selection.

Minimal fix:
- Either update `best.pt` at the end of training when no promotion occurred, or make post-train evaluation prefer `latest.pt` unless `champion.json` or the training log proves that `best.pt` was promoted after training began.

Minimal pseudocode:

```python
# in cli_main.py at end of training
if not os.path.exists(best_model_path) or no_promotion_happened:
    save_checkpoint(best_model_path, model, cfg)

# in ablation_study.py
def resolve_model_path(run_dir):
    if promoted_best_exists(run_dir):
        return run_dir / "best.pt"
    if (run_dir / "latest.pt").exists():
        return run_dir / "latest.pt"
    if (run_dir / "best.pt").exists():
        return run_dir / "best.pt"
    return None
```

### 2. High: concurrent self-play is fed by a stale actor snapshot

Current mechanism:
- `quartz/cli_main.py:700-701` starts a background self-play worker with a cloned actor snapshot.
- `quartz/selfplay_runtime.py:1988-1989` replaces that snapshot only when `update_model(...)` is called.
- `quartz/cli_main.py:880-888` calls `bg_worker.update_model(actor_source)` only every 5 iterations, at checkpoint cadence.

Failure mode:
- The learner updates every iteration, but self-play continues using an older actor for up to 5 iterations.
- In short runs, the background actor may never be refreshed at all.
- The effective loop is actor-lagged asynchronous training, not the tighter iteration-level AlphaZero loop implied by the high-level narrative.

Why it matters:
- It weakens the meaning of per-iteration training curves.
- It makes controller effects harder to interpret, because search-policy changes and network-quality changes are phase-shifted.
- It also means the replay buffer is not clearly attributable to the current learner checkpoint.

Minimal fix:
- Refresh the actor snapshot after every successful learner update, or at least after every iteration that executed nonzero train steps.
- If copy cost is too high, use a versioned checkpoint file or shared immutable actor snapshots with explicit generation IDs.

Minimal pseudocode:

```python
if executed_steps > 0 and bg_worker:
    bg_worker.update_model(actor_source)
    bg_worker.actor_generation = iteration + 1
```

### 3. Medium: the "controller" is a heuristic family, not one cleanly isolated mechanism

Current mechanism:
- `src/mcts/quartz.rs:100-122` defines materially different penalty modes: `Legacy`, `EffectiveV2`, `SelfAdaptive`, `GatedRefresh`, `GatedRefreshLegacy`, `PFlipMixture`.
- `src/mcts/select.rs:194-289` dispatches to different score formulas depending on `penalty_mode`.
- `src/mcts/select.rs:530-553` also allows legacy-style shallow non-root shaping when `root_only_shaping` is false.

Failure mode:
- "QUARTZ controller on/off" does not name one intervention. It names a bundle of possible interventions: penalty law, refresh law, root-vs-shallow scope, and sometimes Fisher/oneloop toggles.
- Some diagnostics are explicitly computed but not decision-bearing, e.g. `prior_q_divergence` for `PFlipMixture` (`src/mcts/quartz.rs:120-121`).

Why it matters:
- This is the main reason controller ablations are still only partially interpretable.
- A result attributed to "the controller" may actually be driven by root-only shaping, a refresh rule, a penalty form, or shallow-tree bleed-through.

Minimal fix:
- Treat each decision-bearing mechanism as its own named subsystem in both docs and manifests.
- Freeze one canonical publication controller and rebrand the rest as alternate search profiles.
- Keep `controller_axes` as the main attribution preset and demote bundled presets to exploratory sweeps.

### 4. Medium: deployment search config is chosen post hoc from the same evaluation matrix used to crown the winner

Current mechanism:
- `scripts/ablation_study.py:1175-1190` selects the overall winning model from the aggregated evaluation matrix.
- It then picks `deployment_condition` as whichever evaluation condition gave that same model the highest score rate.

Failure mode:
- Model selection and deployment search-config selection are both made from the same comparison surface.
- This is effectively search-config cherry-picking after observing the leaderboard.

Why it matters:
- It inflates deployment claims.
- The exported Gomocup bundle can inherit the most favorable eval search settings rather than a preregistered deployment target.
- This is acceptable for engineering optimization, but not for clean research reporting unless explicitly labeled as post hoc deployment tuning.

Minimal fix:
- Pre-register one deployment eval condition and always export with that.
- Alternatively, report deployment-condition selection as a second-stage tuning pass, not as part of the main ablation result.

### 5. Low/Medium: checkpoint metadata discipline regresses at final save

Current mechanism:
- Intermediate checkpoints are saved with config metadata (`quartz/cli_main.py:881-884`).
- Final save drops that metadata and writes a bare state dict (`quartz/cli_main.py:1056-1059`).
- `scripts/ablation_study.py:692-707` explicitly tries to recover architecture settings from checkpoint metadata and only partially falls back when metadata is missing.

Failure mode:
- `latest.pt` at end of training can be less self-describing than earlier checkpoints.
- Downstream tools may need to infer architecture from state_dict keys and can only partially recover config.

Why it matters:
- It weakens reproducibility and artifact portability for exactly the checkpoint most users will grab.

Minimal fix:
- Make the final save use the same wrapped checkpoint format as the periodic save path.

## Search-Controller / MCTS Design Assessment

Steelman:
- The controller is not fake. It has real state variables (`sigma_q`, `p_flip`, `prior_q_divergence`, `conf_t`, VOC channels), real routing into selection and halt, and real telemetry exposure into runtime artifacts.
- Root-only shaping is a defensible attempt to localize controller action where attribution is easiest.
- Parallelism control is cleanly split out from search-policy control; that is a real design strength.

Strongest objection:
- The codebase still mixes theory-aligned and legacy heuristic behaviors under one controller umbrella.
- Experimental language can therefore outrun causal identifiability.

Net judgment:
- Conceptually interesting and technically real.
- Not yet a single clean controller principle in the strict scientific sense.
- Best framed as a controller family platform with one increasingly clean canonical branch.

## AlphaZero-Style Pipeline Maturity

### Strong parts

- Self-play generation is real and Rust-backed.
- Replay/storage is real.
- Learner updates are real.
- Evaluation/promotion is real and uses the same Rust+NN substrate rather than a toy arena by default.
- The code enforces Rust as required for real training.

### Weak parts

- Actor freshness is coarse in concurrent mode.
- Promotion/best-checkpoint semantics bleed into post-hoc evaluation incorrectly.
- Runtime liveness is good enough to complete, but still operationally rough.

Net judgment:
- The pipeline is closed enough to be useful as a research platform.
- It is not yet clean enough to claim "iteration-level AlphaZero closure" without qualification.

## Experimental / Ablation Honesty

Strengths:
- Search manifests and hashes are a real reproducibility asset.
- `controller_axes` is substantially more honest than bundled controller toggles.
- Eval rows record CI/SPRT and runtime-contract data instead of placeholder fields.
- Phase15 tooling explicitly distinguishes posthoc vs online execution and records continuation mode.

Weaknesses:
- The stale-`best.pt` issue can silently invalidate post-train comparisons.
- Champion export currently compounds winner selection and deployment search tuning.
- Controller-family language still exceeds what the ablations can cleanly attribute.

## Software Design / Maintainability

Strengths:
- Split runtime modules are a real improvement over the old monolith.
- Test coverage is excellent by research-code standards.
- Mixed Rust/Python boundary is explicit and mostly well-factored.

Liabilities:
- There is still some compatibility-surface debt around `alphazero_train.py`.
- Runtime topology is sophisticated but operationally fragile enough to show worker-stop warnings and long replay-fill stalls in the canonical smoke.
- Controller/config growth remains a real complexity-debt vector.

## Minimal High-Leverage Upgrade Roadmap

1. Fix artifact truthfulness first.
   - Repair `best.pt` / `latest.pt` selection and final checkpoint metadata.

2. Tighten the learner/self-play contract.
   - Refresh actor snapshots after each iteration with learner progress.
   - Record actor-generation IDs into replay metadata.

3. Make controller claims narrower and cleaner.
   - Designate one canonical publication controller.
   - Rebrand the rest as alternative search profiles.

4. Pre-register deployment policy.
   - Export Gomocup bundles from a fixed deployment eval condition, not the best post-hoc row.

5. Harden runtime liveness.
   - Add explicit worker shutdown/health assertions to the smoke path.
   - Make `MPLCONFIGDIR` and similar environment prerequisites explicit in the smoke launcher.

## Bottom Line

Compared with the audit rubric, the repository is already a real executable research platform, not a speculative scaffold. Its strongest assets are executable reality, unusually strong regression coverage, and good artifact-contract discipline.

Its weakest point is not implementation absence. It is interpretation discipline. The fastest way to raise research trust is to fix checkpoint-selection truthfulness, tighten actor freshness, and stop treating the whole controller family as if it were one causal object.
