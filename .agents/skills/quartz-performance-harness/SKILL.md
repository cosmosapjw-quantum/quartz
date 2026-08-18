---
name: quartz-performance-harness
description: Runs MCTS throughput profiling, PySpy/LineProfiler profiling harnesses, NVIDIA CUDA GPU runtime monitoring, and performance benchmarking for QUARTZ. Use when measuring or optimizing MCTS search speed, memory footprint, or GPU batching.
---

# QUARTZ Performance & Profiling Harness Skill

This skill provides procedures for profiling, benchmarking, and monitoring the high-throughput MCTS engine, NVIDIA CUDA GPU batch evaluator, and Python training pipeline.

## Hardware & Environment Reference

- **GPU**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, Driver 595.84, CUDA 13.2)
- **CPU**: AMD Ryzen 9 5900X (12c/24t), 64GB RAM
- **PyTorch**: 2.11.0+cu128
- **CUDA Monitoring**: `nvidia-smi`

## Profiling Workflows

### 1. Training & Runtime Monitor Telemetry

Profile self-play and training runtime metrics:
```bash
./venv/bin/python scripts/profile_training_monitor.py \
    --model-dir models/alphazero_gomoku7 \
    --print-summary
```

### 2. High-Throughput MCTS Benchmark

Measure raw node-per-second (NPS) throughput across games (Go, Gomoku, Renju, Chess):
```bash
# Rust hotpath benchmarks
GAME_BENCH_LOOPS=20000 cargo test --release bench_gomoku_hotpaths bench_go_hotpaths -- --ignored --nocapture

# End-to-end throughput profiling
./venv/bin/python scripts/throughput_profile.py
```

### 3. PySpy / LineProfiler Harness

Run sampling or line-by-line profilers:
```bash
# PySpy sampling harness
./venv/bin/python tmp/_pyspy_harness.py

# Line profiler harness
./venv/bin/python tmp/_line_profiler_harness.py
```

### 4. GPU Telemetry

Inspect NVIDIA GPU utilization, power, and memory:
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
```
