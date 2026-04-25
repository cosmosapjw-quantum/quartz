# Phase 3.2 — TSAN Audit Note

Date: 2026-04-25
HEAD at audit: 49f70f4 (Phase 3.1 arena landing)
Toolchain: nightly 1.97.0-nightly (7c61a357e 2026-04-24)

## Why this audit exists

Phase 3.1 introduced an `unsafe` boundary inside the MCTS hot path:
`ArenaRef<T>` wraps a `NonNull<T>` and hand-implements `Send + Sync`
under the bound `T: Sync`. The synchronization story for the bumpalo
arena is:

- Allocations into a `bumpalo::Bump` are serialized through the
  per-bucket `Mutex<TtBucket>` that already protects the TT map. `Bump`
  is `!Sync` — the lock is the only thing keeping concurrent `alloc`
  calls from corrupting bump state.
- Reads of `MctsNode<M>` through an `ArenaRef` are safe because
  `MctsNode<M>` is `Sync` (atomics + `parking_lot::RwLock` internals).
- `TtBucket::Drop` runs `drop_in_place` on every node body before the
  Bump frees its chunks. By construction, no live `ArenaRef` to one of
  those bodies exists at engine-drop time.

This audit verifies the synchronization story by running ThreadSanitizer
across the engine's full parallel test surface.

## How to reproduce

```sh
rustup toolchain install nightly --component rust-src

RUSTFLAGS='-Z sanitizer=thread' cargo +nightly test \
  --release --target x86_64-unknown-linux-gnu --tests \
  -Z build-std --no-run

TSAN_TB=$(ls -t target/x86_64-unknown-linux-gnu/release/deps/mcts_demo-*[0-9a-f] \
  | grep -v '\.d$' | grep -v 'long-type' | head -1)

# Single-test repro on the canonical stress case
"$TSAN_TB" --exact zobrist_tt_parallel_verify::v5_stress_parallel

# Full bin test suite
"$TSAN_TB"
```

## Result

| Surface | Outcome |
| --- | --- |
| `v5_stress_parallel` × 5 consecutive runs | 5 / 5 pass, zero data race reports |
| `v3_parallel_vs_sequential` | pass, zero races |
| `v2_virtual_loss_balance` | pass, zero races |
| `v4_tt_nn_integrity` (ignored canonical) | pass, zero races |
| Full bin test suite (390 tests, all configs) | 390 passed, zero races |

No suppression file was needed. `parking_lot` did not produce any
internal-frame false positives in this run (in past audits it has on
older versions; if a future TSAN run regresses, suppress only at the
`parking_lot::*` frame, never at application frames).

## Why no code change in this commit

Phase 3.2 in the original session prompt anticipated needing a switch
from `rayon::scope` to `crossbeam::scope` plus possible suppression
file work. With the `ArenaRef`-based design instead of the
`<'arena>`-everywhere design, `rayon::scope` continues to work
unchanged because `ArenaRef<T>: Send + Sync` whenever `T: Sync`. The
Phase 3.1 commit therefore landed both the single-thread and parallel
correctness in one go; this Phase 3.2 commit is the audit artifact
that documents the TSAN clearance.
