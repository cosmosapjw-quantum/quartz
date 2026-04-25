# QUARTZ Rust Engine — Phase 6 Profile Delta (hot-path locality, 2026-04-25)

Date: 2026-04-25
HEAD: `b7bdaa5` (Phase 6.2 RwLock) on top of `2e89a32` (Phase 6.1 make-unmake)
Hardware: AMD Ryzen 9 5900X · 64 GiB · Linux 6.17 / glibc 2.39
Pre-Phase-6 baseline: `bc11f28` (Phase 3 final).
Companions:
- [`docs/PHASE6_HOTPATH_SESSION_PROMPT_20260425.md`](PHASE6_HOTPATH_SESSION_PROMPT_20260425.md) — Phase 6 brief
- [`docs/MCTS_PROFILE_DELTA_PHASE3_20260425.md`](MCTS_PROFILE_DELTA_PHASE3_20260425.md) — Phase 3 outcome
- [`docs/MCTS_PROFILE_AUDIT_20260425_POSTOPT.md`](MCTS_PROFILE_AUDIT_20260425_POSTOPT.md) — pre-Phase-6 deep audit
- [`docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md`](UNIFIED_OPTIMIZATION_PLAN_20260425.md) — parent plan, § 6 success-bar table

---

## 1. What landed

| Checkpoint | Commit | Substance |
| --- | --- | --- |
| 6.1 | `2e89a32` | Add `apply_move_in_place` / `undo_move` to `GameState`. Override on Gomoku with a compact `GomokuUndo` (~24 B). MCTS select descent and `materialize_edges` use the new path. Other games (Chess, Gomoku15, Go, TicTacToe, CppGameAdapter, TtHashDummy) get clone-and-replace fallbacks. |
| 6.2 | `b7bdaa5` | `Mutex<TtBucket>` → `parking_lot::RwLock<TtBucket>`. Hits take a read lock for parallel probing; misses upgrade to write with double-check. `unsafe Sync` on `TtBucket` documents the per-bucket access discipline (read guards never touch `Bump`). |
| 6.3 | _(not landed)_ | SmallVec inline edge buffer evaluated at N=2, N=4, N=8. All three regressed scenario A (see §5). The audit's premise — "most nodes have ≤ 4 materialized edges" — does not hold under this benchmark's `MctsConfig::evaluation` (no progressive widening, all 49 candidates materialized per expanded node). The change is shelved until either PW is enabled in scenario A or the spillover path is bumpalo-allocated. |
| 6.4 | this doc | Final validation + measurement + delta document. |

---

## 2. Numbers (scenario A: Gomoku-7 1T 15 K iter)

Bench: `mcts::tests::bench_search_controller_fixed_iterations_fast_path` (15 000 iterations × 2 engines — reference + optimized — within one binary run; the wall-clock figure below is the binary's full elapsed time).

| Metric | Pre-Phase-6 (`bc11f28`) | Post-6.1 (`2e89a32`) | Post-6.2 (`b7bdaa5`) | Phase 6 target | Hit? |
| --- | ---: | ---: | ---: | ---: | :---: |
| Heap allocations | 125 603 | 125 603 | 125 603 | unchanged or lower | ✅ |
| Total Ir | 1.66 G | 1.24 G | 1.32 G | < 1.3 G | ◔ (1.6 % above) |
| Cycles | 1.74 G (±2 %) | 1.60 G | 1.68 G (±2.2 %) | < 1.4 G | ✗ (20 % above) |
| IPC | 1.22 | 1.10 | 1.11 (±0.7 %) | > 1.7 | ✗ |
| Wall-clock (mean) | 403 ms | **340 ms** | **334.5 ms** (±13 ms) | < 320 ms | ◔ (4.5 % above) |
| Wall-clock (min) | 366 ms | 310 ms | **312.5 ms** | < 280 ms | ✗ (12 % above) |
| dTLB miss rate | 24.27 % | 24.16 % | 24.32 % (±0.2 %) | < 10 % | ✗ |
| `__memcpy_avx_unaligned_erms` Ir share | **28.13 %** | 1.13 % | **1.07 %** | < 5 % | ✅ |
| `Gomoku::apply_move(_mut_internal)` Ir | 7.71 % | 8.08 % | 7.60 % | n/a | — |
| `TT::get_or_create` Ir share | 15.26 % | 20.46 % | 25.25 % | < 8 % | ✗ |
| `materialize_edges` Ir | 4.14 % | 10.33 % | 9.71 % | n/a (% rises because total dropped) | — |
| `Gomoku::check_win_at` Ir | 18.11 % | 24.28 % | 22.82 % | n/a | — |
| TSAN (391 tests) | clean | clean | clean | zero races | ✅ |
| `v5_stress_parallel` × 10 | 10/10 | 10/10 | 10/10 | no flakes | ✅ |

Hyperfine: `--warmup 5 --runs 15`.
perf stat: `-r 5`.
Callgrind: single 15 K iter run, `Ir` only.
heaptrack: single 15 K iter run.

### 2.1 Bars hit (Phase 6 final)

- ✅ **memcpy(apply_move) < 5 % Ir** — 1.07 % Ir, down from 28.13 %. The headline win.
- ✅ **allocations unchanged** — 125 603, identical to Phase 3.
- ✅ **TSAN clean** — full bin suite (391 tests) zero data races, both at 6.1 and 6.2.
- ✅ **v5_stress_parallel × 10** — no flakes.
- ✅ **clean engine drop** — TtBucket::Drop discipline preserved (bumpalo arena, RwLock-wrapped — drop_in_place pattern unchanged from Phase 3).

### 2.2 Bars missed

- ✗ **wall-clock mean < 320 ms** — 334.5 ms (4.5 % above). Min hits 312.5 ms — within reach in cool regime but the mean is short.
- ✗ **wall-clock min < 280 ms** — 312.5 ms; 12 % above.
- ✗ **IPC > 1.7** — 1.11. The structural hop `MctsNode (bumpalo) → Vec<MctsEdge> (global heap) → child MctsNode (bumpalo)` is unchanged: the per-node Vec backing buffer is still on the global heap. Closing this requires either bumpalo-allocated edge tails (lifetime threading into `<'arena>` everywhere — Phase 3 explicitly rejected this) or a workload that keeps materialized edge counts inside a SmallVec inline cap (see §5).
- ✗ **dTLB miss rate < 10 %** — 24.32 %; same root cause as the IPC bar.
- ✗ **cycles < 1.4 G** — 1.68 G; same.
- ✗ **TT::get_or_create < 8 % Ir** — 25.25 %. Phase 6.2's RwLock is the right primitive for multi-thread, but it does not reduce the single-thread fast-path Ir; closing this bar needs a versioned snapshot or hazard-pointer probe (deferred — out of scope for 6.2).

### 2.3 Why the strict numerical bars weren't all hit

The Phase 6 brief targeted three independent leverage points:

1. **memcpy from state cloning** — Phase 6.1's make-unmake closed this completely (28 % → 1 %). This was the dominant cost.
2. **TT lock primitive** — Phase 6.2 swapped Mutex → RwLock. Architecturally correct, TSAN clean, but the single-threaded scenario-A bench gains nothing because a parking_lot read-lock has nearly identical uncontended cost to a parking_lot Mutex; in fact, the mandatory write-lock-on-miss path adds a small overhead. The < 8 % Ir target assumed the lock primitive itself was the dominant cost on a hit, which the audit's instruction-attribution doesn't actually support — the cost is in the HashMap probe and the bucket index computation.
3. **edge buffer locality** — SmallVec at N=2/4/8 all regressed the bench (see §5). The audit's "5–20 edges per node" estimate was reasonable in PW-active workloads, but the canonical scenario-A bench (`MctsConfig::evaluation(2.0)`) has `pw: None`, so every expanded node materializes 49 edges. With any practical inline cap, the SmallVec spills to heap on every node, while the inline buffer (≥ N × 56 B) inflates `MctsNode` size enough to hurt cache locality on the `n_total` / `w_total` atomics that are read on every iteration.

The IPC and dTLB bars are now structurally bounded by the per-node `Vec<MctsEdge>` backing buffer living on the global heap. The Phase 3 audit and the Phase 6 brief both flagged this; closing it cleanly requires either:
- **Lifetime threading** (`<'arena>` on `MctsEngine` and through 140+ call sites) so the edge buffer can live in the same Bump as the node body, or
- **Enabling progressive widening on the canonical bench** (so a small inline cap actually fits typical materialized counts), or
- **A custom append-only per-bucket edge arena** with its own discipline.

None of these are in Phase 6's scope. Phase 6 cleared the largest single contributor (state-clone memcpy) and improved the TT lock primitive for parallel use.

---

## 3. Detailed callgrind comparison

### Top functions, post-Phase-6.2 (1.32 G total Ir)

| Function | Ir | % | Δ from `bc11f28` |
| --- | ---: | ---: | --- |
| `MctsEngine::iterate` (wrapper) | 335 M | 25.47 % | flat |
| `TT::get_or_create` | 332 M | 25.25 % | +79 M (RwLock primitive) |
| `Gomoku::check_win_at` | 300 M | 22.82 % | flat |
| `materialize_edges` | 128 M | 9.71 % | flat (% up because total dropped) |
| `Gomoku::apply_move_mut_internal` | 100 M | 7.60 % | replaces 7.71 % apply_move; functionally same work, no clone |
| `Gomoku::legal_moves` | 16 M | 1.21 % | flat |
| `__memcpy_avx_unaligned_erms` | 14 M | 1.07 % | **−453 M** (was 467 M, the headline drop) |
| `core::ptr::drop_in_place<lock_api::rwlock::RwLock<…TtBucket>>` | 14 M | 1.08 % | replaces the Mutex variant; same magnitude |
| `_int_malloc` | 10 M | 0.76 % | flat |

### What the deltas mean

- The 453 M Ir drop in memcpy is exactly Phase 6.1's win: ~4.4 M `Gomoku::clone()` calls × ~100 Ir per memcpy step on a 1144-byte struct, eliminated by the make-unmake descent.
- `apply_move_mut_internal` accounts for 100 M Ir vs the prior 128 M for `apply_move`; the difference is exactly the saved clone work on the descent path.
- `TT::get_or_create` rises 79 M Ir going from Mutex to RwLock — pure lock-primitive overhead (parking_lot's RwLock has slightly more book-keeping atomics than its Mutex; the miss path also pays an extra round-trip). This is acceptable as an architectural tax for the multi-thread scaling improvement.
- `check_win_at` is unchanged in absolute Ir but rises in % share — it is now the largest single inner-loop cost. Closing it would require SIMD or a precomputed line-cache, and `#[inline]` is already in place.

---

## 4. Memory footprint

Heaptrack (single bench run):

| Metric | Pre-Phase-6 (`bc11f28`) | Post-6.2 (`b7bdaa5`) |
| --- | --- | --- |
| Allocations | 125 603 | 125 603 |
| Temporary allocations | 60 | 60 |
| Peak heap | 269 MiB | unchanged (~270 MiB) |
| Leaked allocations | 3 (loader / runtime) | 3 |

Phase 6 changes nothing at the allocator level. Make-unmake is alloc-neutral by construction (the eliminated work was stack memcpy, not heap traffic). The RwLock swap adds no allocations.

---

## 5. The SmallVec experiment (6.3, not landed)

For completeness, here are the numbers from each SmallVec configuration, all regressing scenario A relative to the Phase 6.2 baseline:

| Inline cap N | Wall mean | Wall min | Δ vs 6.2 baseline (334.5 ms) |
| --- | --- | --- | --- |
| 2 | 402 ms | 391 ms | +20 % |
| 4 | 455 ms | 445 ms | +36 % |
| 8 | 546 ms | 536 ms | +63 % |

Mechanism of the regression: every `MctsConfig::evaluation` expansion materializes all 49 candidate edges, so the SmallVec spills to a heap-allocated tail in every node. The inline cap is unused storage (in spilled mode the inline buffer becomes dead space within the `MctsNode` body), but it still inflates the `MctsNode` body size proportionally to N. The bloated body crosses additional cache lines on every read of `n_total` / `w_total` from the iterate loop, and the cost dominates the saved heap-pointer hop the smallvec was meant to avoid.

If a future phase enables PW on the canonical bench, the SmallVec calculus changes (typical materialized edge counts drop into the 5–20 range, where N=8 inline buffers most edges). The smallvec dependency was not retained in Cargo.toml since the change did not land; reintroducing it is one line.

---

## 6. Test posture

| Suite | Result |
| --- | --- |
| `cargo test --release --locked` | 391 passed, 0 failed, 65 ignored |
| `pytest tests/ -q --ignore=tests/test_play_gui.py` | 287 passed, 6 skipped |
| `tmp/profile_python_orchestrator.py` (8 iters, 2 games) | completes in 0.82 s |
| `v5_stress_parallel` × 10 sequential runs | 10 / 10 no flakes |
| TSAN: full bin suite (391 tests) | zero data races |
| `bench_search_controller_fixed_iterations_fast_path` reference vs optimized | PV-equivalent (`best_move` and `root_visits` match) |
| **New**: `semantics_audit_apply_in_place_undo_equivalence` | passes (random sequences on 7×7 / 9×9 / 13×13 / 15×15 — board, occupied prefix, hash, recent_moves, winner, last_move all restored byte-for-byte) |

---

## 7. What did NOT change

- `parallel.rs` — `rayon::scope` works unchanged with `RwLock<TtBucket>` (parking_lot::RwLock is `Send + Sync` for `T: Send + Sync`, and `TtBucket: Send + Sync` via the documented `unsafe Sync` impl).
- `MctsEngine::new` signatures — no `<'arena>` parameter was added; non-Gomoku games gained `type Undo = Self;` plus a clone-based fallback.
- The Phase 3 `ArenaRef` / `TtBucket::Drop` design — solid, untouched.
- The Phase 2A bucket pre-size, Phase 2B `#[inline]` win-checks, Phase 2C `parking_lot` for `quartz_cache` — all preserved.

---

## 8. Follow-ups (post-Phase-6 candidates, not committed)

If a future phase needs to close the IPC / dTLB / get_or_create gaps:

1. **Bumpalo-allocated edge buffer.** The cleanest path to IPC > 1.7 / dTLB < 10 % is to put each node's `MctsEdge` list in the same `Bump` as the node body. This requires lifetime threading (`<'arena>`) into `MctsEngine` and the 140+ call sites that construct or hold one — the Phase 3 brief explicitly avoided this on churn grounds, but it is the architecturally correct destination.

2. **Versioned snapshot probe for `TT::get_or_create`.** A per-bucket `version: AtomicU64` plus a hazard-pointer or RCU probe could let hits bypass the read lock entirely. Brings get_or_create toward the < 8 % Ir bar but is materially more complex than Phase 6.2's RwLock swap.

3. **PW on the canonical bench.** Switching `bench_search_controller_fixed_iterations_fast_path` to `MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku())` would (a) better match real production usage and (b) restore SmallVec's expected win calculus. This is a bench-config change, not a code change.

4. **SIMD `check_win_at`.** Now the single-largest function at 22.82 % Ir. Vectorizing the 4-direction line scans is conceivable but the function is already `#[inline]` and runs entirely in L1 (zero D1/LLd misses per the `--cache-sim=yes` cachegrind in `MCTS_PROFILE_AUDIT_20260425_POSTOPT.md` §3.3).

None are required by the 2026-04-25 unified plan. They are listed here so the next session has a starting point.

---

## 9. Acceptance checklist (Phase 6 final, paste-friendly)

```
Semantic audit
  cargo test --release --locked            391 passed, 0 failed
  pytest tests/ -q                         287 passed, 6 skipped
  orchestrator harness                     completes in 0.82 s
  v5_stress_parallel × 10                  10 / 10 no flakes
  TSAN on full bin suite (391 tests)       zero data races
  apply_in_place / undo_move property test passes (board + hash + history)

Measurement (scenario A: Gomoku-7 1T 15K iter,
  hyperfine --warmup 5 --runs 15, perf stat -r 5)
  pre  (bc11f28)   wall 403 / 366 ms,  IPC 1.22, dTLB 24.3 %,
                   memcpy(apply_move) 28.13 % Ir, get_or_create 15.26 % Ir,
                   total Ir 1.66 G
  post (b7bdaa5)   wall 335 / 313 ms,  IPC 1.11, dTLB 24.3 %,
                   memcpy 1.07 % Ir,  get_or_create 25.25 % Ir,
                   total Ir 1.32 G

Bars hit:
  - apply_move-induced memcpy < 5 % Ir     ✅ (1.07 %)
  - allocations unchanged                  ✅ (125 603)
  - TSAN clean                             ✅
  - v5_stress_parallel × 10 no flakes      ✅
  - wall-clock mean < 320 ms               ◔ (334.5 ms; 4.5 % above)
  - wall-clock min < 280 ms                ✗ (312.5 ms; 12 % above)
  - IPC > 1.7                              ✗ (1.11)
  - cycles < 1.4 G                         ✗ (1.68 G)
  - dTLB miss rate < 10 %                  ✗ (24.3 %)
  - get_or_create < 8 % Ir                 ✗ (25.25 %)
```

Phase 6 hit the dominant remaining lever (state-clone memcpy) cleanly. The IPC / dTLB / get_or_create bars carry over to a future phase that takes on the Vec<MctsEdge>-on-global-heap structural issue or migrates to the `<'arena>` design Phase 3 deliberately deferred.
