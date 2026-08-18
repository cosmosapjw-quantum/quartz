---
name: quartz-foundry
description: Runs, analyzes, and gates QUARTZ Idea Foundry 24/26-axis MCTS experiments, study campaigns, factorial sweeps, and axis contracts. Use when working on Idea Foundry experiments, preflight checks, and ablation campaigns.
---

# QUARTZ Idea Foundry Experimentation Skill

This skill guides the execution, validation, and gating of QUARTZ Idea Foundry experiments across the 24/26 axes catalog.

## Core Reference Files

- [00_INDEX_KO.md](file:///home/cosmosapjw/Dropbox/personal_projects/quartz/docs/idea_foundry/00_INDEX_KO.md) — Index and overview of Idea Foundry axes
- [README.md](file:///home/cosmosapjw/Dropbox/personal_projects/quartz/docs/idea_foundry/README.md) — Experimentation instructions
- [LOCAL_EXPERIMENT_LAB.md](file:///home/cosmosapjw/Dropbox/personal_projects/quartz/docs/LOCAL_EXPERIMENT_LAB.md) — Local experiment lab runner guide
- [idea_foundry.axes.v1.json](file:///home/cosmosapjw/Dropbox/personal_projects/quartz/configs/idea_foundry.axes.v1.json) — Formal axis definition schema
- [APPLY_TO_REPO.md](file:///home/cosmosapjw/Dropbox/personal_projects/quartz/APPLY_TO_REPO.md) — Provenance notice (never overwrite skeleton archives)

## Important Invariants

1. **Rust Feature Flag**: The Rust `src/mcts/foundry/` implementation is gated behind Cargo `--features idea-foundry`. It must remain absent from the default production build.
2. **Never overwrite archive files**: `quartz_idea_foundry_skeleton.zip` and `quartz_idea_foundry_skeleton.patch` are immutable provenance files.
3. **Python Environment**: Always run scripts using the project virtual environment: `./venv/bin/python`.

## Common Workflows

### 1. Preflight Verification

Run preflight checks before launching any axis campaign:
```bash
./venv/bin/python scripts/idea_foundry_preflight.py
```

### 2. Running an Idea Foundry Study Campaign

To execute a study campaign across specific axes:
```bash
# Run a specific axis study (e.g. A15)
./venv/bin/python scripts/idea_foundry_study.py --axis A15

# Run all registered axes
./venv/bin/python scripts/idea_foundry_study_all.py
```

### 3. Factorial Metacontroller Sweep

To execute factorial interaction studies:
```bash
./venv/bin/python scripts/metacontroller_factorial_study.py
```

### 4. Running Contract & Regression Tests

```bash
# Python contracts
./venv/bin/pytest tests/test_idea_foundry_skeletons.py -v

# Rust foundry tests
cargo test --features idea-foundry
```

### 5. Analyzing Results and Gating

```bash
./venv/bin/python scripts/idea_foundry_study_analyze.py
./venv/bin/python scripts/idea_foundry_axis_gate.py
```
