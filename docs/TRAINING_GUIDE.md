# Training Guide

## Current training architecture

All training self-play uses the Rust MCTS engine with Python-owned NN forward
and learner logic. The active path is:

1. Python launches the Rust server.
2. Rust drives self-play/eval loops and search sessions.
3. Rust batches NN requests.
4. Python executes model forward passes.
5. Rust consumes the returned priors/values and continues search.

Control traffic uses JSON-line IPC. Hot-path payloads use binary sparse framing
and SHM ring transport when available, with stdout JSON fallback retained.

## Search features active during training

- transposition table
- progressive widening
- adaptive split virtual loss (`vvisit` + `vvalue`)
- QUARTZ controller
- batched Rust+NN evaluation
- history-sensitive TT exactness for chess/go

Important correctness details:

- Chess policy uses full `4672`-action promotion-aware encoding.
- Chess exact search state includes FEN plus returned history-token metadata.
- Go and chess TT keys use `tt_hash()` rather than board-only `hash()`.

## Runtime behavior

### Background actor (`--concurrent`)

Main thread:

- replay sampling
- SGD / backend update
- checkpointing
- checkpoint evaluation

Background worker:

- Rust+NN self-play with frozen model snapshot
- backpressure when replay is near capacity
- model snapshot refresh after checkpoint

By default, checkpoint evaluation pauses background self-play and resumes it
after the eval completes.

### Tuning and isolation defaults

- online runtime autotune is off by default
- eval/self-play isolation is on by default
- `search-profile` is explicit and ablation-safe:
  - `quartz`
  - `baseline`
  - `baseline_strict`

## Checkpoint and evaluation cadence

Current loop behavior:

- `latest.pt` is checkpointed every 5 iterations
- replay state is checkpointed alongside it
- training-time promotion evaluation runs on `--eval-interval`
- `best.pt` updates only on explicit promotion verdict
- plots regenerate from `train_log.jsonl`

The loop no longer drops checkpoint/eval work when an iteration has `0`
learner steps because self-play was still filling replay.

## Training artifacts

The model directory contains:

- `latest.pt`
- `best.pt`
- `replay.npz`
- `train_log.jsonl`
- `autotune_profile.json`
- `glicko2_ladder.json`
- `training_loss.png`
- `training_elo.png`

Training log rows include:

- losses
- replay size and freshness
- self-play throughput
- controller telemetry
- eval verdict / Elo / score rate

## Evaluation semantics

Checkpoint evaluation uses `RustNNEvaluatorEngine` when the Rust binary is
available. That keeps evaluation aligned with the same search stack used in
training self-play.

If the Rust binary is missing, the code falls back to lighter evaluation paths
with an explicit warning. Those fallback paths are not benchmark-grade and
should not be mixed into ablation claims.

## Cache and transport correctness notes

The following regressions were fixed and are now part of the expected behavior:

- disabling the NN cache does not replace model output with uniform priors
- sparse loss series are plotted correctly
- best-Elo progression only advances on actual promotion
- binary sparse search-result transport preserves the same search meaning as the older JSON payload path

## Remaining limitations

- QUARTZ score shaping is still root-only
- P_flip behavior still depends on evaluator quality and usually needs NN loss below roughly `1.0`
- raw external chess FEN alone cannot recreate prior repetition history; exact repeated search requires the returned history token path
- JAX remains a training backend, not the inference backend used by Rust self-play/eval or Gomocup deployment
