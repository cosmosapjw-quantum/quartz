# QUARTZ AlphaZero

An AlphaZero-style game-playing AI engine with an adaptive search controller.

**QUARTZ** = Q-value Uncertainty–Adaptive Root-risk Tree search, Zero-tunable

## What Is This?

A research platform combining:
- **Rust MCTS engine** — tree-parallel search with transposition table,
  progressive widening, and adaptive virtual loss
- **QUARTZ controller** — uncertainty-aware search policy, adaptive stopping
  (P_flip), experimental prior refresh, and a controller family surface across
  `search_profile`, halt, penalty, and refresh switches
- **Python training loop** — self-play, replay buffer, SGD, checkpoint
  evaluation via Glicko-2

The Python side is now split into focused runtime modules. `quartz/alphazero_train.py`
remains as a compatibility facade for older imports, the GUI, and tests.

The Rust engine is the sole training search engine. The main Rust-backed
self-play and evaluation paths run through the same Rust+NN substrate, with a
hybrid QIPC transport:
JSON-line control messages plus binary/SHM hot-path payloads for batched NN
evaluation and search responses.

## Quick Start

```bash
# 1. Build Rust engine
cargo build --release
cargo test --release

# 2. Install Python package
pip install -e .

# 3. Run the canonical end-to-end audit smoke
venv/bin/python scripts/smoke_e2e.py

# 4. Run a smoke ablation first
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

# 5. Train on Gomoku 7×7
venv/bin/python -m quartz.train --game gomoku7 --iterations 30

# 6. Generate a report from an existing ablation directory
venv/bin/python scripts/ablation_study.py \
  --report results/ablation_smoke_search_vl/gomoku7

# 7. Export a selected champion as a Gomocup bundle
venv/bin/python scripts/ablation_study.py \
  --report results/ablation/gomoku15 \
  --prepare-gomocup

# 8. Build the Gomocup tournament binary
scripts/build_gomocup_brain.sh \
  --bundle-dir results/ablation/gomoku15/gomocup_bundle \
  --target-name pbrain-quartz
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed setup and
[docs/QUICKSTART.md](docs/QUICKSTART.md) for training and experiment guides.

For external review packaging, use:

```bash
venv/bin/python scripts/build_audit_bundle.py
```

The generated audit zip is a source-level review bundle. It contains the code,
configs, docs, smoke script, and targeted regression tests needed for build +
ablation review, but it is not an offline `cargo vendor` / wheelhouse bundle.
`scripts/smoke_e2e.py` can now attempt `cargo build --release --bin mcts_demo`
when the Rust binary is missing, and it writes `smoke_contract.json` so the
binary provenance and expected artifacts are explicit.

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
- Phase-1.5 clean-split assays now have a dedicated runner
  (`phase15_ablation_study.py`) for bucketized frozen-checkpoint `A/B/C`
  post-hoc comparisons, a suite miner (`phase15_mine_suite.py`) for balanced
  bucket artifacts, and an explicit online chunked scaffold
  (`phase15_online_ablation.py`) for `B/C` follow-up. The phase15 manifest now
  separates `reference_policy` from `oracle_policy`, splits trace-acquisition
  time from readout time, and treats `B0` as an alias of `A4` instead of a
  distinct substrate. Post-hoc runs also warm the prior cache in batches, store
  shared suite policies in compressed NPZ sidecars, and report trace-cache hit
  rates plus semantic summaries over alias-equivalent systems. Trace cache keys
  are now salted against the phase15 codepath, runner contracts, and the
  systems config schema so stale artifacts are invalidated when semantics move.
  Post-hoc, online, and continuation-benchmark runs now amortize one full trace per
  `checkpoint x position x system` and reuse budget prefixes instead of
  reacquiring independent traces per budget row. The phase15 benchmark emits a
  `bundle_summary`, a built-in gate, and per-system headwind diagnostics
  (`session_overhead`, `readout_sensitivity`, or mixed). The post-hoc and
  online phase15 summaries now also emit `headwind_summary` so assay-space
  runtime and semantic bottlenecks can be read in the same family of terms.
  A self-contained GitHub Actions workflow now runs a deterministic smoke
  version of the phase15 benchmark gate and uploads the resulting artifacts.
  The online phase15 runner now prefers true root-continuation resident
  sessions and only falls back to restart-per-chunk when that server path is
  unavailable. Use explicit
  curated checkpoint paths; lexical
  `--checkpoint-dir` truncation is intentionally rejected for weak/mid/strong
  experiments.
- The old Python monolith was split into focused runtime modules; the public
  `alphazero_train.py` surface is now a thin compatibility facade.
- Python orchestrator profiling is now closed for the current micro-optimization
  loop. The kept changes are:
  - CPU / low-concurrency async-pipeline gating
  - replay collation fast path + import cache
  - direct memoryview/`struct` SHM ring access
  - wider default SHM ring topology (`8x8`, env-overridable)
  The larger Rust-side ring wakeup experiment regressed and was reverted.
- Current Gomoku7 controller evidence favors no-refresh legacy-family variants.
  Prior refresh remains implemented and searchable, but is not the current
  default/deployment recommendation.
- `ablation_study.py` now records both training and evaluation contract
  summaries, rejects stale cached eval rows when `search_manifest_hash` moves,
  and should be driven with explicit `--study ...` in user-facing commands.
- Evaluation rows now record real score-rate confidence intervals and SPRT
  status instead of placeholder `ci=[0,0]` / `sprt=None`.
- Study manifests, evaluation matrices, and ablation reports now all carry a
  `runtime_contract` / `runtime_contract_hash` so backend/device/Rust-binary
  provenance is visible without reopening per-run metadata.
- `controller_axes` is now the attribution-oriented study preset: adjacent
  comparisons isolate `root_only_shaping`, `penalty_mode`, and
  `prior_refresh_rate` one factor at a time. The older `controller` preset
  remains a bundled legacy-vs-theory comparison.
- Replay search summaries now expose controller observability and coverage
  fields such as `halt_reason_hist`, `controller_penalty_mode_counts`,
  `mean_prior_refresh_rate`, and telemetry coverage fractions so partial
  instrumentation is explicit in artifacts.
- `scripts/smoke_e2e.py` is the canonical audit smoke for the current tree. It
  is meant to fail fast on replay-bootstrap/runtime breakages rather than
  certify benchmark readiness, and
  `scripts/build_audit_bundle.py` regenerates the external audit zip with the
  files needed for install, ablation, and protocol review.

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
| Rust MCTS engine | Implemented | Extensive Rust test surface; validate with local `cargo test --release` |
| QUARTZ controller | Implemented | Multiple penalty families, adaptive stopping, still best treated as an experimental controller bundle |
| ParallelismController | Implemented | 2nd-gen feedback: dup_rate + contention |
| Game encoders | Implemented | Gomoku, Go, Chess |
| Glicko-2 evaluation | Implemented | Comprehensive math / gate self-tests |
| ONNX export/inference | Partial | Real code path exists; external deployment still needs environment-specific verification |
| GPU auto-detection | Implemented | NVIDIA, AMD ROCm, Apple Metal |
| Rust+NN self-play | Implemented | Rust binary required; chess uses FEN + history tokens |
| Chess policy encoding | Implemented | 4672-action promotion-aware encoding |
| History-aware TT exactness | Implemented | Chess repetition / Go superko state included in TT key |
| Gomocup brain | Partial | Bundle/export/build flow exists; tournament packaging remains environment-dependent |
| Actor/learner split | Conditional | `--concurrent` path is real, but Rust binary and runtime environment are mandatory |

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
   toggleable for controlled experiments. For controller attribution, prefer
   `controller_axes` over the bundled `controller` preset.
5. **Exactness before speed**: History-dependent rules use exact TT hashes
   instead of board-only keys, and chess policy targets preserve promotion choice.

## Known Limitations

- Score shaping applies at root depth only (not tree-wide)
- Adaptive stopping requires NN loss < ~1.0 for P_flip convergence
- Controller telemetry is partial (core stats; not all internal state exposed)
- The `controller` study preset is intentionally bundled. Use
  `controller_axes` or `controller_factorial` when you need cleaner factor
  isolation.
- Raw external chess FEN alone does not reconstruct prior repetition history;
  exactness for repeated search requires the returned history token path
- JAX backend is available for training, but Rust self-play/eval and Gomocup
  deployment paths do not use JAX inference
- Gomocup ONNX deployment requires building the Rust binary with `--features onnx`
- Prior refresh is implemented, but current short-budget Gomoku7 controller
  sweeps do not support enabling it by default
