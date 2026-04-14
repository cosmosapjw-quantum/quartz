# Quickstart: Training, Ablation, and Gomocup Export

## 1. Train a model

### Gomoku 7×7

```bash
venv/bin/python -m quartz.train \
  --game gomoku7 \
  --iterations 30 \
  --eval-interval 5 \
  --eval-games 200
```

Artifacts land in `models/alphazero_<game>/`:

- `latest.pt` — latest checkpoint
- `best.pt` — promoted champion checkpoint
- `replay.npz` — replay snapshot
- `train_log.jsonl` — per-iteration metrics
- `training_loss.png` / `training_elo.png` — regenerated from the log

### Useful flags

| Flag | Meaning |
|---|---|
| `--search-profile quartz|baseline|baseline_strict` | Search-controller ablation axis |
| `--vl-mode disabled|fixed|adaptive|vvisit_only|vvalue_only` | Virtual-loss ablation axis |
| `--resident-session` | Keep Rust search sessions resident for self-play |
| `--runtime-autotune` | Enable online runtime retuning |
| `--no-eval-selfplay-isolation` | Let background self-play continue during eval |
| `--seed <int>` | Reproducible training seed |

## 2. Understand current training semantics

Current training/eval uses the same Rust+NN stack:

1. Python launches the Rust server.
2. Rust drives self-play or evaluation game loops.
3. Batched NN eval requests cross QIPC.
4. Control traffic stays on JSON-line IPC.
5. Hot-path eval/search payloads use binary sparse payloads and SHM ring when available.

Correctness-sensitive updates already reflected in the codebase:

- Chess policy uses full `4672`-action promotion-aware encoding.
- Chess and Go TT keys use history-sensitive `tt_hash()` values.
- Disabling the NN cache no longer changes search semantics.
- Checkpoint/eval cadence is no longer skipped when a learner iteration has `0` train steps.

## 3. Run ablations

### Level 1: Search-only Rust experiments

```bash
# Adaptive virtual loss
cargo test --release -- ablation_vl::tests::vl_ablation_gomoku7 --ignored --nocapture

# P_flip convergence
cargo test --release -- ablation_pflip::tests::pflip_convergence_curves --ignored --nocapture

# Prior-refresh experiments
cargo test --release -- ablation_refresh_v2 --ignored --nocapture
```

### Level 2: Training-level ablation

Use the ablation runner instead of ad-hoc shell loops:

```bash
venv/bin/python scripts/ablation_study.py \
  --game gomoku15 \
  --iterations 30 \
  --eval-games 80
```

The runner writes:

- `study_manifest.json` — full experiment definition, seeds, and CLI/runtime choices
- `models/<condition>[/seed_<n>]` — per-condition training outputs
- `evaluation_matrix.json` — round-robin post-train arena results across eval conditions
- `champion.json` — selected final model, selection metrics, and recommended deployment search config
- `ablation_report.json` — summarized training/eval report

`champion.json` is chosen from the post-train evaluation matrix when it exists.
The deployment search profile stored there is the best-scoring evaluation
condition for that selected model, not just its training condition.

For multi-seed studies:

```bash
venv/bin/python scripts/ablation_study.py \
  --game gomoku15 \
  --iterations 20 \
  --eval-games 40 \
  --seeds 41,42,43
```

### Level 2.5: Low-cost controller search

Use these when you want to separate "controller family" from the fixed search
constants that turned out to matter just as much.

Shortlist + confirmatory arena:

```bash
venv/bin/python scripts/controller_sweep.py \
  --resume-report results/controller_sweep_confirmatory_v1/gomoku7 \
  --candidate-ids A1_legacy_base,R03_7362f3bd,A2_legacy_krefresh \
  --arena-iters 96 \
  --stage2-games 12
```

Optuna search over the same frozen checkpoints:

```bash
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

Artifacts to read first:

- `optuna_report.json` or `sweep_report.json` — canonical summary
- `stage2_round_robin.json` — shortlist/top-trial arena confirmation

`--checkpoints` expects comma-separated checkpoint files. Use
`--checkpoint-dir` when you want the script to scan a directory tree.

Current repository-local Gomoku7 evidence favors no-refresh legacy-family
variants. Keep prior refresh as an experimental axis, not the starting default.

### Level 3: Direct arena

```bash
venv/bin/python -m quartz.train \
  --arena models/alphazero_gomoku15/best.pt models/alphazero_gomoku15/latest.pt \
  --arena-games 100
```

## 4. Export the champion for Gomocup

From an existing ablation directory:

```bash
venv/bin/python scripts/ablation_study.py \
  --report results/ablation/gomoku15 \
  --prepare-gomocup
```

This creates a `gomocup_bundle/` directory containing:

- `gomocup_manifest.json`
- `gomocup_model.onnx`
- `champion.pt`

Then build the tournament binary:

```bash
scripts/build_gomocup_brain.sh \
  --bundle-dir results/ablation/gomoku15/gomocup_bundle \
  --target-name pbrain-quartz
```

For ONNX-backed Gomocup deployment the Rust binary is built with `--features onnx`.
The helper copies `pbrain-quartz` into the bundle directory so the directory can
be used directly as the runtime folder or passed via `INFO folder`.

## 5. Supported games

| Game | Board | Actions | Encoder | Notes |
|---|---|---:|---:|---|
| `gomoku7` | 7×7 | 49 | 17ch | full training + evaluation |
| `gomoku15` and variants | 15×15 | 225 | 17ch | full training + evaluation |
| `go9` | 9×9 | 82 | 17ch | ruleset/scoring presets supported |
| `chess` | 8×8 | 4672 | 36ch | promotion-aware policy + history-aware TT |

## 6. Project structure

```text
quartz/
├── train.py
├── alphazero_train.py
├── cli_main.py
├── train_loop.py
├── selfplay_runtime.py
├── arena_runtime.py
├── evaluator_runtime.py
├── eval_runtime.py
├── replay.py
├── qipc.py
├── evaluation.py
├── encoders.py
├── onnx_support.py
├── gomocup_export.py
├── models_torch.py
└── backend.py

scripts/
├── ablation_study.py
├── build_gomocup_brain.sh
├── controller_optuna.py
├── controller_sweep.py
└── profile_training_monitor.py

src/
├── gomocup_bundle.rs
├── gomocup_brain.rs
├── mcts_server.rs
├── mcts/
└── games/
```

`alphazero_train.py` is now a compatibility facade. New runtime logic lives in
the split modules above, and new code should import from those modules directly.
