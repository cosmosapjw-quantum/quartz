# Ablation Guide

## Levels

### Level 1: Search-only Rust experiments

Use these for controller-only questions with fixed evaluators and fixed budgets.

- `ablation_vl` — adaptive virtual loss and duplicate suppression
- `ablation_pflip` — stopping behavior and convergence curves
- `ablation_refresh_v2` — prior-refresh behavior

These runs are useful for:

- agreement vs serial reference
- duplicate-leaf behavior
- root entropy / visit spread
- throughput under fixed semantics

### Level 2: Training-level ablation

Use [scripts/ablation_study.py](../scripts/ablation_study.py) for full-pipeline ablations.

Default train-condition matrix:

- `T1_noS_noVL` — baseline search, VL disabled
- `T2_S_noVL` — QUARTZ search, VL disabled
- `T3_noS_VL` — baseline search, adaptive VL
- `T4_S_VL` — QUARTZ search, adaptive VL

Default eval-condition matrix:

- `E1_noS_noVL`
- `E2_S_noVL`
- `E3_noS_VL`
- `E4_S_VL`

Optional strict reference:

- `E0_baseline_strict`

Example:

```bash
venv/bin/python scripts/ablation_study.py \
  --study search_vl \
  --game gomoku15 \
  --iterations 30 \
  --eval-games 80 \
  --seeds 41,42
```

Recommended smoke run before any larger study:

```bash
venv/bin/python scripts/smoke_e2e.py
```

If `./target/release/mcts_demo` is missing, the smoke script will attempt
`cargo build --release --bin mcts_demo` first and will write
`smoke_contract.json` so binary provenance and required output artifacts are
explicit. Treat this as a fail-fast integration smoke, not as a benchmark
certification step.

Or run just the training/eval ablation smoke directly:

```bash
venv/bin/python scripts/ablation_study.py \
  --study search_vl \
  --game gomoku7 \
  --iterations 2 \
  --eval-games 4 \
  --eval-interval 1 \
  --seeds 11 \
  --paired-seed-eval \
  --include-strict-reference \
  --resident-session \
  --timeout-hours 1 \
  --output results/ablation_smoke_search_vl
```

This is the preferred pipeline sanity check because it exercises:

- the real Rust server path
- SHM ring transport and sparse search payload decode
- the compatibility-facade `NNSearchClient` path used by arena/eval wrappers
- contract hashing and cached-eval invalidation

Only after this passes should you scale to `gomoku15`, `gomoku15_omok`, or
`gomoku15_renju`.

Study intent:

- `search_vl` compares search profile and VL mode.
- `controller` is a bundled legacy-vs-theory comparison and is not a
  single-factor isolation study.
- `controller_factorial` preserves the historical 4-cell matrix.
- `controller_axes` is the attribution-first preset:
  - `A1 -> A2` isolates `root_only_shaping`
  - `A2 -> A3` isolates `penalty_mode`
  - `A3 -> A4` isolates `prior_refresh_rate`

Evaluation rows now store:

- `score_rate_a` — canonical point rate for side A
- `ci` — score-rate confidence interval
- `ci_kind` — interval model tag
- `sprt` — sequential-test status
- `sprt_meta` — decisive-game count and LLR metadata
- `errors` / `voids` / `scored_games` — arena validity counters
- `timing_s` — startup and per-match timing metadata

### Level 2.5: Frozen-checkpoint controller search

Use this level when controller family and search hyperparameters are confounded
enough that short training alone is too noisy or too expensive.

- [scripts/controller_sweep.py](../scripts/controller_sweep.py) runs a fixed
  candidate pool through:
  - stage1 surrogate probing on frozen checkpoints + fixed position suite
  - stage2 same-checkpoint arena on the shortlist
- [scripts/controller_optuna.py](../scripts/controller_optuna.py) replaces the
  fixed pool with an Optuna search over controller family plus search
  hyperparameters.

Typical usage:

```bash
# Confirmatory shortlist arena on known candidates
venv/bin/python scripts/controller_sweep.py \
  --resume-report results/controller_sweep_confirmatory_v1/gomoku7 \
  --candidate-ids A1_legacy_base,R03_7362f3bd,A2_legacy_krefresh \
  --arena-iters 96 \
  --stage2-games 12

# Optuna search over family + refresh + search hyperparameters
venv/bin/python scripts/controller_optuna.py \
  --game gomoku7 \
  --checkpoints results/ablation_controller_factorial_short/gomoku7/models/F1_legacy_base/seed_42/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F2_legacy_krefresh/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F3_theory_base/seed_42/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_42/best.pt \
  --positions-file results/controller_sweep_shortlist_v1/gomoku7/stage1_positions.json \
  --trials 48 \
  --enqueue-anchors \
  --probe-iters 64 \
  --arena-topk 4 \
  --arena-iters 96 \
  --stage2-games 6 \
  --output results/controller_optuna_v1
```

`--checkpoints` expects comma-separated checkpoint files. If you want recursive
discovery under a model directory, use `--checkpoint-dir` instead.

Anchor naming in controller sweeps:

- `A1_legacy_base` / `A2_legacy_krefresh` mean the `GatedRefreshLegacy`
  family used in recent training-time controller ablations
- they do not mean the older low-level `PenaltyMode::Legacy` path
- the full design lineage is described in [QUARTZ_THEORY.md](./QUARTZ_THEORY.md)

### Level 3: Deployment selection

The ablation runner can also produce deployment-ready outputs:

- `champion.json` — selected model, selection metrics, and deployment search config
- `gomocup_bundle/` — ONNX + manifest + checkpoint copy

```bash
venv/bin/python scripts/ablation_study.py \
  --report results/ablation/gomoku15 \
  --prepare-gomocup
```

## Artifact contract

Each ablation directory now carries:

- `study_manifest.json` — experiment definition, seeds, runtime settings, git head
- `models/<condition>[/seed_<n>]` — per-run training artifacts
- `evaluation_matrix.json` — post-train round-robin matches and leaderboards
- `champion.json` — final model selection, selection metrics, and deployment config
- `ablation_report.json` — summary report for humans/tools
- `study_manifest.json` now also carries:
  - `train_contract_summary`
  - `contract_summary` for evaluation conditions
- `study_manifest.json`, `evaluation_matrix.json`, and `ablation_report.json`
  also carry:
  - `runtime_contract`
  - `runtime_contract_hash`
  so backend/device/Rust-binary provenance is visible at the study, eval, and
  report layers
- `ablation_report.json` mirrors those summaries so training/evaluation contract
  drift can be read without reopening per-condition artifacts

Frozen-checkpoint controller sweeps carry:

- `optuna_manifest.json` / `sweep_manifest.json` — search-space and runtime definition
- `stage1_positions.json` — fixed position suite used for surrogate probing
- `optuna_report.json` / `sweep_report.json` — canonical summary report
- `stage2_round_robin.json` — same-checkpoint arena verification for shortlisted candidates
- `trials/trial_*.json` — per-trial telemetry snapshots

These artifacts are the intended basis for:

- paper figures
- internal comparisons
- Gomocup bundle export

Champion selection uses the post-train evaluation matrix when available. The
stored deployment search config is the best-scoring evaluation condition for
the chosen model, which is the profile exported into Gomocup bundles.

## Current ablation hygiene

For controlled comparisons, keep these fixed unless the systems stack itself is
the explicit ablation target:

- same `study_manifest.json` shape except for the intended factor
- same Rust binary
- same seeds
- same eval isolation policy
- same runtime-autotune policy
- same hardware class

The current system stack is no longer “pure JSON IPC”. Training and evaluation
share:

- binary sparse search-result payloads
- SHM ring transport on the hot path when available
- wider default SHM ring topology (`8x8`, env-overridable)
- stdout JSON fallback for compatibility

That transport is part of the common substrate, not an ablation axis by itself.

Replay/search summaries now also surface controller observability fields when
the runtime provides them:

- `halt_reason_hist`
- `mean_refresh_count`
- `mean_penalty_sum`
- `controller_penalty_mode_counts`
- `mean_prior_refresh_rate`
- `root_only_shaping_frac`
- `controller_telemetry_partial_frac`
- `halt_metric_coverage_frac`
- `refresh_metric_coverage_frac`
- `penalty_metric_coverage_frac`

`halt_reason_hist` is populated from replay metadata today. The refresh/penalty
aggregates remain intentionally partial until every Rust search path emits
those counters directly.

For external audit packaging, regenerate the review bundle with:

```bash
venv/bin/python scripts/build_audit_bundle.py
```

For controller sweeps, keep these fixed as well:

- the frozen checkpoint set
- the position suite
- `probe_iters` and `reference_multiplier`
- arena iteration budget and game count

## Exactness invariants

Recent corrections matter for fair comparisons:

- Chess policy targets are now promotion-aware (`4672` actions).
- Chess and Go TT keys are history-sensitive via `tt_hash()`.
- `QUARTZ_DISABLE_NN_CACHE` no longer changes model semantics.
- Checkpoint/eval cadence is not skipped when learner work for an iteration is zero.

If an ablation predates these fixes, do not compare it directly to current runs
without restating the older semantics.

The same warning now applies to cached evaluation matrices: current
`ablation_study.py` discards stale rows when `search_manifest_hash` changes
instead of quietly reusing them.

## Current controller findings

These are current repository-local findings, not universal claims for every
game or budget.

### Gomoku7 short-budget training and confirmatory arena

- The short factorial training runs did not support promoting prior refresh to
  the default.
- In the confirmatory frozen-checkpoint arena
  (`results/controller_sweep_confirmatory_v1/gomoku7/stage2_round_robin.json`),
  `A1_legacy_base` scored `55/96 = 0.5729`.
- In the same run, `A2_legacy_krefresh` scored `45/96 = 0.4688` and the tuned
  refresh challenger `R03_7362f3bd` scored `44/96 = 0.4583`.

### Gomoku7 Optuna controller search

- The first wider Optuna run
  (`results/controller_optuna_v1/gomoku7/optuna_report.json`) completed
  `18/48` trials and pruned `30/48`.
- Completed no-refresh trials outnumbered refresh trials `14` to `4`.
- The top quartile of completed trials was entirely no-refresh.
- The best surrogate trial was `T0010_cf38467f`:
  `GatedRefreshLegacy/root=1/pr=0.00/tau=1.00/h=0.50/s=0.15/mv=9/ci=25/cp=1.00`.
- Stage2 arena confirmation
  (`results/controller_optuna_v1/gomoku7/stage2_round_robin.json`) ranked:
  - `T0010_cf38467f` `45/72 = 0.625`
  - `T0013_c6df8981` `38/72 = 0.5278`
  - `T0005_0910eb73` `31/72 = 0.4306`
  - `T0002_f66a5653` `30/72 = 0.4167`

### Practical interpretation

- `prior refresh` should remain in the search space, but not as the current
  default or deployment profile.
- The stronger signal is that controller family is entangled with search
  hyperparameters such as `sigma_0`, `min_visits`, `check_interval`,
  `c_puct`, and `hbar_penalty_cap`.
- For current Gomoku7 work, the best basin is a no-refresh legacy-family
  controller with `root_only_shaping=true` and retuned search constants.

### Level 2.6: Phase 1.5 clean-split structure assays

Use [scripts/phase15_ablation_study.py](../scripts/phase15_ablation_study.py)
for the phase-1.5 redesign in
[phase15_strategy_revision_v2.md](../phase15_strategy_revision_v2.md).

Current implementation scope:

- frozen-checkpoint post-hoc assays for Group `A/B/C`
- explicit online chunked runner for Group `B/C`
- amortized full-trace reuse for post-hoc, online, and benchmark budget rows
- bucketized position-suite preparation
- bucket-balanced suite mining
- trace disk cache for repeated assay recomposition
- batched prior inference warmup for suite and checkpoint passes
- `position_suite.json` now embeds shared `prior_policy`, `low_budget_policy`,
  `reference_policy`, and `oracle_policy` artifacts via a compressed NPZ
  sidecar file
  so post-hoc and online runs can reuse suite-level evidence without bloating
  the main suite manifest
- explicit `A0-A4`, `B0-B3`, `C0-C2` system definitions
- root-level telemetry for commit/challenger/budget-routing analysis

Important current contract:

- Group A is the substrate/controller sanity matrix
- Group B in `phase15_ablation_study.py` is explicitly `posthoc`
- Group B in `phase15_online_ablation.py` is online chunked control with
  resident `root_continuation` preferred and `restart_per_chunk` fallback only
  on protocol failure
- Group C is legacy-anchor comparison only
- `reference_policy` and `oracle_policy` are stored separately
- `B0` is a report alias for `A4`, not a distinct search substrate
- checkpoint selection must be curated explicitly; do not rely on lexical
  `--checkpoint-dir` truncation for weak/mid/strong coverage

Post-hoc assay:

```bash
venv/bin/python scripts/phase15_ablation_study.py \
  --game gomoku7 \
  --checkpoints results/ablation_controller_factorial_short/gomoku7/models/F1_legacy_base/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F2_legacy_krefresh/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --systems-config configs/phase15_systems.default.json \
  --reference-checkpoint results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --budgets 8,16,32,64 \
  --oracle-budget 256 \
  --suite-size 96 \
  --suite-source mined \
  --bucket-min-count 4 \
  --groups A,B,C \
  --search-stall-timeout-s 45 \
  --output results/phase15_ablation_v1
```

Online chunked assay:

```bash
venv/bin/python scripts/phase15_online_ablation.py \
  --game gomoku7 \
  --checkpoints results/ablation_controller_factorial_short/gomoku7/models/F1_legacy_base/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F2_legacy_krefresh/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --systems-config configs/phase15_systems.default.json \
  --reference-checkpoint results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --systems B1,B2,B3,C0,C1,C2 \
  --groups B,C \
  --budgets 8,16,32,64 \
  --oracle-budget 256 \
  --suite-size 96 \
  --output results/phase15_online_v1
```

Continuation benchmark and CI gate smoke:

```bash
venv/bin/python scripts/phase15_benchmark.py \
  --game gomoku7 \
  --checkpoints results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --positions-file results/controller_sweep_shortlist_v1/gomoku7/stage1_positions.json \
  --max-positions 2 \
  --systems A4,B1,B2,B3 \
  --budgets 8,16,32,64 \
  --repeats 1 \
  --warmup-rounds 0 \
  --rust-binary ./target/release/mcts_demo \
  --enforce-gate
```

Default phase15 benchmark gate:

- `bundle_summary.wallclock_speedup_mean >= 1.80`
- `summary.tie_aware_match_rate >= 0.65`
- `summary.policy_kl_restart_vs_continuation.mean <= 0.25`

This is a smoke/CI gate, not a publishable research threshold. The tie-aware
bound is intentionally looser than a strict parity requirement because flat-root
positions still create benign top-1 ambiguity. If you want a tighter local
check, override the thresholds explicitly on the command line.

Mine a reusable suite artifact:

```bash
venv/bin/python scripts/phase15_mine_suite.py \
  --game gomoku7 \
  --checkpoints results/ablation_controller_factorial_short/gomoku7/models/F1_legacy_base/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F2_legacy_krefresh/seed_41/best.pt,results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --systems-config configs/phase15_systems.default.json \
  --reference-checkpoint results/ablation_controller_factorial_short/gomoku7/models/F4_theory_krefresh/seed_41/best.pt \
  --suite-size 96 \
  --candidate-count 384 \
  --bucket-min-count 4 \
  --positions-out results/phase15_suite_mining/gomoku7/mined_suite.json
```

To change phase-1.5 system definitions between reruns, copy and edit
`configs/phase15_systems.default.json` and pass the edited file via
`--systems-config`.

Set `--reference-checkpoint` explicitly so `reference_policy` is built from the
intended frozen model. Set `--oracle-checkpoint` or `--oracle-system` when you
need a stronger oracle contract than “same checkpoint, stricter profile”. The
runner rejects implicit `--checkpoint-dir` discovery when it would silently
truncate a larger directory into lexical first-N checkpoints.

Post-hoc artifacts:

- `phase15_manifest.json`
- `position_suite.json`
- `position_suite_artifacts.npz`
- `assays/phase15_rows.jsonl`
- `phase15_summary.json`

`phase15_manifest.json` and `phase15_online_manifest.json` now also record
`trace_cache_salt`. This is a code-signature hash over the phase15
search/readout stack, the suite/config schema, and the main phase15 runners.
It is part of the trace-cache key, so old cached traces are invalidated when
semantics-critical code or config contracts change.
  - `raw_summary`
  - `semantic_summary`
  - `headwind_summary`
  - `trace_cache_stats`
  - `trace_cache_unit = "trace_bundle"`

Online artifacts:

- `phase15_online_manifest.json`
- `assays/phase15_online_rows.jsonl`
- `phase15_online_summary.json`
  - `raw_summary`
  - `semantic_summary`
  - `headwind_summary`
  - `trace_cache_stats`
  - `trace_cache_unit = "trace_bundle"`
- online runner now prefers true root-continuation resident sessions and falls
  back to `restart_per_chunk` only when the Rust server path is unavailable

Benchmark artifacts:

- `phase15_continuation_benchmark_rows.jsonl`
- `phase15_continuation_benchmark_summary.json`
  - `summary`
  - `bundle_summary`
  - `gate`

`bundle_summary` is the primary speed artifact. It measures one amortized trace
run per `checkpoint x position x system x repeat`, then replays budget prefixes
through readout. Use it when you care about actual continuation-vs-restart
wallclock. `summary` remains useful for budget-level semantic drift metrics.

The benchmark summary also includes a coarse headwind decomposition for each
`checkpoint x system`:

- `continuation_trace_acquire_ms` / `restart_trace_acquire_ms`
- `continuation_overhead_ms` / `restart_overhead_ms`
- `readout_sensitivity`
- `speedup_headwind`

The post-hoc and online summary files also carry `headwind_summary`. That
payload does not compare continuation against restart; instead it summarizes the
same systems in assay space using:

- `trace_acquire_ms`
- `readout_ms`
- `effective_runtime_ms`
- `readout_ratio_mean`
- `accuracy_to_reference`
- `kl_to_reference`
- `speedup_headwind`

Interpret those labels as follows:

- `trace_acquire_cost` means the assay is dominated by search-trace acquisition.
- `readout_cost` means the final operator/readout is a meaningful share of
  runtime.
- `semantic_drift` means reference divergence is the larger concern.
- `mixed_readout_cost_and_semantic_drift` means both the readout and the
  semantics need attention.

Continuous integration:

- GitHub Actions workflow:
  [`.github/workflows/phase15-benchmark-gate.yml`](../.github/workflows/phase15-benchmark-gate.yml)
- self-contained smoke entrypoint:
  [scripts/phase15_benchmark_ci_smoke.py](../scripts/phase15_benchmark_ci_smoke.py)

That CI path generates a deterministic random checkpoint plus a tiny fixed
position suite at runtime, runs the real Rust-backed phase15 benchmark gate, and
uploads the resulting benchmark artifacts.

The smoke gate intentionally uses the stable subset `A4,B1,B2`. `B3` remains in
the full benchmark runner, but it is more path-sensitive on tiny deterministic
suites and can add noise to CI without improving regression coverage.

Interpretation:

- `session_overhead` means continuation speed is being limited mostly by
  resident-session orchestration cost.
- `readout_sensitivity` means policy drift indicators are the larger concern.
- `mixed_session_overhead_and_readout_sensitivity` means both effects matter.
- `search_cost` means continuation is still dominated by the actual search work
  rather than orchestration or readout instability.

## Recommended interpretation

- Level 1 tells you whether a controller change helps search behavior.
- Level 2 tells you whether that change survives the full Rust+NN pipeline.
- Level 2.5 tells you whether the apparent controller win survives once
  controller family and fixed search constants are allowed to move together.
- `evaluation_matrix.json` should decide the final champion, not loss alone.
- The deployment search profile in `champion.json` is the one to carry into Gomocup export.
