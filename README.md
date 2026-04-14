# QUARTZ AlphaZero

An AlphaZero-style game-playing AI engine with an adaptive search controller.

**QUARTZ** = Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable

## What Is This?

A research platform combining:
- **Rust MCTS engine** — tree-parallel search with transposition table,
  progressive widening, and adaptive virtual loss
- **QUARTZ controller** — uncertainty-aware search policy, adaptive stopping
  (P_flip), and experimental prior refresh
- **Python training loop** — self-play, replay buffer, SGD, checkpoint
  evaluation via Glicko-2

The Python side is now split into focused runtime modules. `quartz/alphazero_train.py`
remains as a compatibility facade for older imports, the GUI, and tests.

The Rust engine is the sole training search engine. Self-play and evaluation
run through the same Rust+NN stack, with a hybrid QIPC transport:
JSON-line control messages plus binary/SHM hot-path payloads for batched NN
evaluation and search responses.

## Quick Start

```bash
# 1. Build Rust engine
cargo build --release
cargo test --release

# 2. Install Python package
pip install -e .

# 3. Train on Gomoku 7×7
venv/bin/python -m quartz.train --game gomoku7 --iterations 30

# 4. Run training-level ablation on Gomoku 15×15
venv/bin/python scripts/ablation_study.py --game gomoku15 --iterations 30 --eval-games 80

# 5. Export the selected champion as a Gomocup bundle
venv/bin/python scripts/ablation_study.py \
  --report results/ablation/gomoku15 \
  --prepare-gomocup

# 6. Build the Gomocup tournament binary
scripts/build_gomocup_brain.sh \
  --bundle-dir results/ablation/gomoku15/gomocup_bundle \
  --target-name pbrain-quartz
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed setup and
[docs/QUICKSTART.md](docs/QUICKSTART.md) for training and experiment guides.

## Architecture

```
Python training loop
  │
  ├─ cli_main / train_loop
  ├─ selfplay_runtime / arena_runtime / evaluator_runtime
  ├─ replay / eval_runtime / qipc
  │
  ├─ selfplay/eval runners
  │    └─ Launch Rust server (--server)
  │         └─ QIPC: JSON control + binary/SHM eval/search payloads
  │              └─ Batched NN forward pass (PyTorch, GPU)
  │
  ├─ ReplayBuffer → train_epoch() → checkpoint
  │
  └─ RustNNEvaluatorEngine → Glicko-2 promotion
       └─ Same Rust+NN stack as training (no semantic mismatch)
```

## Recent Updates

- Training checkpoints, tournament evaluation, and Elo promotion now run on the
  intended cadence even when an iteration produces `0` learner steps.
- Loss and Elo plots were corrected so sparse loss series render and best Elo
  only advances on actual promotion.
- Chess now uses promotion-aware `4672` policy targets and chess/go TT entries
  use exact `tt_hash()` keys instead of board-only hashes.
- Training-level ablations now produce `study_manifest.json`,
  `evaluation_matrix.json`, `champion.json`, and optional Gomocup bundles with
  deployment search metadata.
- Low-cost controller search now has frozen-checkpoint confirmatory runs
  (`controller_sweep.py`) and Optuna-driven surrogate search
  (`controller_optuna.py`).
- Phase-1 prior revision assays now have a dedicated runner
  (`prior_revision_experiment.py`) for bucketized frozen-checkpoint `B0/B1/N1/N2`
  comparisons before training-contract promotion. Use explicit curated
  checkpoint paths; lexical `--checkpoint-dir` truncation is intentionally
  rejected for weak/mid/strong experiments.
- The old Python monolith was split into focused runtime modules; the public
  `alphazero_train.py` surface is now a thin compatibility facade.
- Current Gomoku7 controller evidence favors no-refresh legacy-family variants.
  Prior refresh remains implemented and searchable, but is not the current
  default/deployment recommendation.

## Current Controller Status

Repository-local Gomoku7 evidence currently points to:

- `A1_legacy_base` as the safest existing default among the hand-written anchors
- a stronger tuned no-refresh legacy-family variant from Optuna
  (`T0010_cf38467f`) as the current top low-cost sweep result
- `prior refresh` as an experimental axis worth preserving, not the default
  search profile to ship

## Key Features

| Feature | Description |
|---------|-------------|
| Adaptive VL | 2nd-gen feedback controller: dup_rate + contention |
| P_flip stopping | Adaptive budget based on move-flip probability |
| Prior refresh | Experimental search axis; not current default winner |
| Split virtual loss | Separate vvisit (reservation) + vvalue (pessimism) |
| Exact TT keys | History-sensitive TT hashing for chess/go rule state |
| Strict arena | Default strict mode; fallback explicitly non-benchmark |
| Controller telemetry | p_flip, sigma_q, hbar_eff, stop_reason per move |
| Multi-game | Gomoku 7/15, Go 9x9, Chess (all via same Rust engine) |
| Gomocup deployment | Champion export → ONNX bundle → `pbrain` binary |

## Maturity

| Component | Status | Notes |
|---|---|---|
| Rust MCTS engine | ✅ | Extensive test suite; `cargo test --release` |
| QUARTZ controller | ✅ | 6 penalty modes, adaptive stopping |
| ParallelismController | ✅ | 2nd-gen feedback: dup_rate + contention |
| Game encoders | ✅ | Gomoku, Go, Chess |
| Glicko-2 evaluation | ✅ | Comprehensive test suite |
| ONNX export/inference | ✅ | CPU, CUDA, ROCm |
| GPU auto-detection | ✅ | NVIDIA, AMD ROCm, Apple Metal |
| Rust+NN self-play | ✅ | All games; chess uses FEN + history tokens |
| Chess policy encoding | ✅ | 4672-action promotion-aware encoding |
| History-aware TT exactness | ✅ | Chess repetition / Go superko state included in TT key |
| Gomocup brain | ✅ | Bundle-driven ONNX path with CPU fallback |
| Actor/learner split | ✅ | `--concurrent` (Rust required) |

## Documentation

- [INSTALL.md](docs/INSTALL.md) — Prerequisites, build, verify
- [QUICKSTART.md](docs/QUICKSTART.md) — Training, ablation, interpretation
- [QUARTZ_THEORY.md](docs/QUARTZ_THEORY.md) — Controller theory and architecture
- [ABLATION_GUIDE.md](docs/ABLATION_GUIDE.md) — Ablation levels and protocol
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) — Training pipeline details
- [GOMOCUP_BRAIN.md](docs/GOMOCUP_BRAIN.md) — Gomocup bundle/export/build flow
- [TT_NOTES.md](docs/TT_NOTES.md) — TT design and exactness notes

## Design Principles

1. **Rust-native search, Python training**: Search performance in Rust,
   ML flexibility in Python. Connected via hybrid QIPC
   (JSON control + binary/SHM hot path).
2. **State-derived signals, explicit search hyperparameters**: Controller
   inputs come from observable search/runtime state, but constants such as
   `sigma_0`, `min_visits`, `check_interval`, and `c_puct` are explicit
   per-run hyperparameters and are valid sweep targets.
3. **Evaluation-training consistency**: Checkpoint evaluation uses the same
   Rust+NN stack as training self-play (RustNNEvaluatorEngine).
4. **Ablation-first design**: Search controller modes are independently
   toggleable for controlled experiments.
5. **Exactness before speed**: History-dependent rules use exact TT hashes
   instead of board-only keys, and chess policy targets preserve promotion choice.

## Known Limitations

- Score shaping applies at root depth only (not tree-wide)
- Adaptive stopping requires NN loss < ~1.0 for P_flip convergence
- Controller telemetry is partial (core stats; not all internal state exposed)
- Raw external chess FEN alone does not reconstruct prior repetition history;
  exactness for repeated search requires the returned history token path
- JAX backend is available for training, but Rust self-play/eval and Gomocup
  deployment paths do not use JAX inference
- Gomocup ONNX deployment requires building the Rust binary with `--features onnx`
- Prior refresh is implemented, but current short-budget Gomoku7 controller
  sweeps do not support enabling it by default
