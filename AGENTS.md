# QUARTZ Repository Agent Guidelines

## 1. Project Overview
QUARTZ is a high-throughput AlphaZero Monte Carlo Tree Search (MCTS) engine and reinforcement learning pipeline written in Rust (core game logic & search engine) and Python (training loop, batch neural network evaluators, and experiment runners).

## 2. Environment & Execution Rules
- **Python Environment**: Always use the virtual environment binaries: `./venv/bin/python` and `./venv/bin/pytest`.
- **Rust Toolchain**: Use `cargo build` and `cargo test`.
- **Idea Foundry Feature**: Rust Idea Foundry modules (`src/mcts/foundry/`) are feature-gated. Use `cargo test --features idea-foundry` or `cargo build --features idea-foundry`.
- **Provenance Files**: Never delete, regenerate, or overwrite `quartz_idea_foundry_skeleton.zip` or `quartz_idea_foundry_skeleton.patch`.

## 3. Hardware & GPU Configuration
- **Hardware**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, Driver 595.84, CUDA 13.2 / PyTorch cu128), AMD Ryzen 9 5900X (12c/24t), 64GB RAM.
- **Software**: PyTorch 2.11.0+cu128, Python 3.12 (venv).
- **GPU Telemetry**: Use `nvidia-smi` for monitoring VRAM and GPU utilization.
- **Backend & Acceleration**:
  - PyTorch CUDA backend with `torch.compile(backend="inductor")` / CUDA graphs.
  - Checkpoint metadata serialization with full `{model_state_dict, cfg}` support.

## 4. Skills & Harnesses
- **QUARTZ Skills**: Available in `.agents/skills/` (`quartz-foundry`, `quartz-performance-harness`, `quartz-scientific-validation`).
- **Research & Coding Harnesses**: Registered via `.agents/skills.json` (includes `physmath-research-harness`, `physmath-coding-harness`, `superpowers`, and ML/GPU skills).
