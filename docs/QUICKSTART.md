# Quickstart: Training and Experiments

## 1. Training a Model from Scratch

### Gomoku 7×7 (recommended first experiment)

```bash
# Basic training run
venv/bin/python -m quartz.train \
    --game gomoku7 \
    --iterations 30 \
    --eval-interval 10 \
    --eval-games 200

# Output: models/alphazero_gomoku7/latest.pt, best.pt, train_log.jsonl
```

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--game` | gomoku7 | Game: gomoku7, gomoku15, go9, chess |
| `--iterations` | 30 | Total training iterations |
| `--config` | none | JSON overrides for runtime/game config |
| `--concurrent` | true | Background self-play worker |
| `--eval-interval` | 10 | Evaluate checkpoint every N iterations |
| `--eval-games` | 200 | Games per promotion evaluation |
| `--search-profile` | quartz | `quartz`, `baseline`, `baseline_strict` |
| `--runtime-autotune` | off | Enable online runtime retuning (default off) |
| `--no-eval-selfplay-isolation` | off | Disable eval-time self-play pause (default isolated) |

### Concurrent training (faster, requires more RAM)

```bash
venv/bin/python -m quartz.train \
    --game gomoku7 \
    --concurrent
```

This runs self-play in a background thread while the main thread trains.
The worker uses a frozen model snapshot, updated at each checkpoint.

## 2. Understanding Training Output

Training produces:
- `models/<game>/latest.pt` — latest training checkpoint
- `models/<game>/best.pt` — best model (promoted via Glicko-2)
- `train_log.jsonl` — per-iteration metrics

### Training log fields

```json
{
    "iter": 10,
    "loss": 2.15,
    "p_loss": 1.82,
    "v_loss": 0.33,
    "new_pos": 2400,
    "games_done": 200,
    "replay_freshness": 0.85,
    "policy_entropy": 2.1,
    "value_std": 0.42,
    "avg_pflip": 0.38,
    "time_s": 45.2
}
```

Key indicators:
- **loss decreasing**: model is learning
- **replay_freshness > 0.5**: replay buffer has recent data
- **avg_pflip decreasing**: search is converging (requires loss < ~1.0)

## 3. Running Search Ablations

### Level 1: Search-only (no training, fixed evaluator)

These run inside the Rust test suite:

```bash
# VL ablation: component isolation + budget scaling + QUARTZ interaction
cargo test --release -- ablation_vl::tests::vl_ablation_gomoku7 --ignored --nocapture

# P_flip convergence experiment
cargo test --release -- ablation_pflip::tests::pflip_convergence_curves --ignored --nocapture

# Prior refresh ablation
cargo test --release -- ablation_refresh_v2 --ignored --nocapture
```

### Level 2: Training-level ablation

Compare QUARTZ modes across full training runs:

```bash
# Baseline: no QUARTZ penalty
venv/bin/python -m quartz.train --game gomoku7 --search-profile baseline --iterations 30

# GatedRefresh (default)
venv/bin/python -m quartz.train --game gomoku7 --search-profile quartz --iterations 30

# Strict semantic baseline (same systems substrate, controller stack removed)
venv/bin/python -m quartz.train --game gomoku7 --search-profile baseline_strict --iterations 30
```

Compare by examining `train_log.jsonl` loss curves and arena win rates.

### Level 3: Arena evaluation

```bash
# Compare two checkpoints (strict Rust+NN engine)
venv/bin/python -m quartz.train \
    --arena models/alphazero_gomoku7/best.pt models/alphazero_gomoku7/latest.pt \
    --arena-games 100
```

## 7. Runtime Monitoring (recommended)

```bash
venv/bin/python scripts/profile_training_monitor.py \
    --model-dir models/alphazero_gomoku7 \
    --output-dir artifacts/runtime_monitor/gomoku7_run \
    --interval-s 0.5 \
    --run "venv/bin/python -m quartz.train --game gomoku7 --iterations 20 --retune --search-profile quartz" \
    --print-summary
```

## 4. Interpreting Ablation Results

### VL ablation table columns

| Column | Meaning |
|--------|---------|
| Agree | % moves matching serial (1-thread) reference |
| Entrop | Root policy entropy (higher = more exploration) |
| Q_Sprd | Q-value spread across root actions |
| NPS | Nodes per second |
| AvgVV | Average virtual value applied (lower = less pessimism) |
| DupRt | Duplicate leaf rate (how often threads collide) |
| MaxP | Maximum pending threads at any node |

Key relationships:
- **Fixed VL**: AvgVV≈1.0, DupRt≈0.27 (over-pessimistic, avoids overlap)
- **Adaptive VL**: AvgVV≈0.17, DupRt≈0.38 (controlled overlap, less waste)
- Fixed + SelfAdaptive = worst (double pessimism)
- Adaptive + SelfAdaptive = rescued (σ_Q auto-correction)

### P_flip interpretation

- P_flip ≈ 0.5: evaluator provides no useful signal (random)
- P_flip < 0.25: search is converging, safe to stop early
- P_flip threshold (0.159): adaptive stopping trigger

P_flip convergence requires NN loss < ~1.0. With weak evaluators
(ShortRollout, Uniform), P_flip stays at 0.4-0.5 regardless of budget.

## 5. Supported Games

| Game | Board | Actions | Encoder | Status |
|------|-------|---------|---------|--------|
| gomoku7 | 7×7 | 49 | 3ch | Full training + evaluation |
| gomoku15 | 15×15 | 225 | 3ch | Full training + evaluation |
| go9 | 9×9 | 82 | 17ch | Search + self-play |
| chess | 8×8 | 4096 | 16ch | Search + self-play (FEN, simplified from-to policy) |

## 6. Project Structure

```
quartz/
├── train.py              CLI entrypoint
├── alphazero_train.py    Core training loop
├── evaluation.py         Glicko-2 evaluation system
├── encoders.py           Game-specific board encoders
├── onnx_support.py       ONNX export/inference
├── gpu_detect.py         GPU auto-detection
├── backend.py            JAX/PyTorch backend abstraction

src/
├── mcts/
│   ├── quartz.rs         QUARTZ search controller
│   ├── parallel.rs       ParallelismController (adaptive VL)
│   ├── select.rs         PUCT selection + score shaping
│   ├── search.rs         Search loop + stop reasons
│   └── ...               backup, expand, eval, node, tt, rng
├── games/                chess, go, gomoku, gomoku15, tictactoe
├── mcts_server.rs        JSON-line IPC server
├── ablation_vl.rs        VL ablation suite (3 experiments)
├── ablation_pflip.rs     P_flip ablation suite (3 experiments)
└── main.rs               Entry point + legacy experiments

docs/                     This documentation
configs/                  Preset configurations
models/                   Trained checkpoints
```
