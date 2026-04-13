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
  --game gomoku15 \
  --iterations 30 \
  --eval-games 80 \
  --seeds 41,42
```

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
- stdout JSON fallback for compatibility

That transport is part of the common substrate, not an ablation axis by itself.

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

## Recommended interpretation

- Level 1 tells you whether a controller change helps search behavior.
- Level 2 tells you whether that change survives the full Rust+NN pipeline.
- Level 2.5 tells you whether the apparent controller win survives once
  controller family and fixed search constants are allowed to move together.
- `evaluation_matrix.json` should decide the final champion, not loss alone.
- The deployment search profile in `champion.json` is the one to carry into Gomocup export.
