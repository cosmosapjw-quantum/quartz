# QUARTZ AlphaZero

An AlphaZero-style game-playing AI engine with an adaptive search controller.

**QUARTZ** = Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable

## What Is This?

A research platform combining:
- **Rust MCTS engine** — tree-parallel search with transposition table,
  progressive widening, and adaptive virtual loss
- **QUARTZ controller** — uncertainty-aware search policy, adaptive stopping
  (P_flip), and prior refresh
- **Python training loop** — self-play, replay buffer, SGD, checkpoint
  evaluation via Glicko-2

The Rust engine is the sole training search engine. Self-play generates
game trajectories via JSON-line IPC with batched NN evaluation.

## Quick Start

```bash
# 1. Build Rust engine
cargo build --release
cargo test --release          # verify: ~270+ tests pass

# 2. Install Python package
pip install -e .

# 3. Train on Gomoku 7×7
python3 -m quartz.train --game gomoku7 --iterations 30

# 4. Play against a trained model in the browser
python3 -m quartz.play_gui --port 8080

# 5. Run search ablation
cargo test --release -- ablation_vl --ignored --nocapture
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed setup and
[docs/QUICKSTART.md](docs/QUICKSTART.md) for training and experiment guides.

## Architecture

```
Python training loop
  │
  ├─ selfplay_rust_nn_batched()
  │    └─ Launches Rust server processes (--server)
  │         └─ JSON-line IPC: search_nn request ↔ eval_req/eval_resp
  │              └─ Batched NN forward pass (PyTorch, GPU)
  │
  ├─ ReplayBuffer → train_epoch() → checkpoint
  │
  └─ RustNNEvaluatorEngine → Glicko-2 promotion
       └─ Same Rust+NN stack as training (no semantic mismatch)
```

## Key Features

| Feature | Description |
|---------|-------------|
| Adaptive VL | 2nd-gen feedback controller: dup_rate + contention |
| P_flip stopping | Adaptive budget based on move-flip probability |
| Prior refresh | GatedRefresh / SelfAdaptive / PFlipMixture modes |
| Split virtual loss | Separate vvisit (reservation) + vvalue (pessimism) |
| Strict arena | Default strict mode; fallback explicitly non-benchmark |
| Controller telemetry | p_flip, sigma_q, hbar_eff, stop_reason per move |
| Multi-game | Gomoku 7/15, Go 9x9, Chess (all via same Rust engine) |

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
| Rust+NN self-play | ✅ | All games incl. chess (FEN) |
| Actor/learner split | ✅ | `--concurrent` (Rust required) |

## Documentation

- [INSTALL.md](docs/INSTALL.md) — Prerequisites, build, verify
- [QUICKSTART.md](docs/QUICKSTART.md) — Training, ablation, interpretation
- [QUARTZ_THEORY.md](docs/QUARTZ_THEORY.md) — Theory foundations (rigorous)
- [CONTROLLER_NOTES.md](docs/CONTROLLER_NOTES.md) — Controller architecture
- [ABLATION_GUIDE.md](docs/ABLATION_GUIDE.md) — Ablation levels and protocol
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) — Training pipeline details

## Design Principles

1. **Rust-native search, Python training**: Search performance in Rust,
   ML flexibility in Python. Connected via JSON-line IPC.
2. **State-derived control law**: All controller inputs come from observable
   search state. Fixed constants only (no learned or user-tuned parameters).
3. **Evaluation-training consistency**: Checkpoint evaluation uses the same
   Rust+NN stack as training self-play (RustNNEvaluatorEngine).
4. **Ablation-first design**: Search controller modes are independently
   toggleable for controlled experiments.

## Known Limitations

- Score shaping applies at root depth only (not tree-wide)
- Adaptive stopping requires NN loss < ~1.0 for P_flip convergence
- Batched self-play spawns processes per batch (not persistent pool)
- Controller telemetry is partial (core stats; not all internal state exposed)
- JAX backend available for training but self-play eval uses PyTorch
