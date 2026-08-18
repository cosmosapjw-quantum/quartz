---
name: quartz-scientific-validation
description: Validates MCTS mathematical/scientific correctness, game invariant enforcement, Elo rating significance, and regression suites for QUARTZ. Use before promoting algorithmic modifications or claiming MCTS convergence improvements.
---

# QUARTZ Scientific & Numerical Validation Skill

This skill enforces rigorous validation standards for all AlphaZero MCTS algorithmic changes, game logic extensions, and policy evaluation improvements.

## Invariants & Guardrails

1. **Game Rule Symmetry & Invariants**:
   - Board symmetry orbits (D4 rotations and reflections) must produce invariant value predictions and equivariant policy distributions.
   - Zobrist hash consistency across move application and undo.
   - Legality masking must strictly forbid suicide moves (in Go) and invalid placements.

2. **Evaluation Protocol**:
   - Always run evaluations using `--backend torch` to avoid subprocess deadlocks (JAX backend eval stall).
   - Ensure Elo rating calculations report standard errors or 95% confidence intervals (e.g. SPRT / Bayesian rating).

## Validation Procedures

### 1. Regression & Unit Tests
```bash
# Core pipeline regressions
./venv/bin/python -m pytest tests/test_training_pipeline_regressions.py tests/test_evaluation_pipeline_regressions.py -v

# Full test suite
./venv/bin/pytest tests/ -q
```

### 2. Game Symmetry & Invariant Checks
```bash
# Symmetry orbit checks
./venv/bin/python scripts/symmetry_orbit_lab.py

# Rust game rules and unit tests
cargo test
```

### 3. Flip Calibration & Burst Precision
```bash
./venv/bin/python scripts/phase15_flip_calibration.py
./venv/bin/python scripts/phase15_o6_burst_precision.py
```

### 4. End-to-End Smoke Verification
```bash
./venv/bin/python scripts/smoke_e2e.py
```
