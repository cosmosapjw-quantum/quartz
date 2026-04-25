# QUARTZ Rust Engine — Phase 3 Profile Delta (bumpalo arena landing)

Date: 2026-04-25
HEAD: `0c3992e` (Phase 3.2 TSAN audit) on top of `49f70f4` (Phase 3.1 arena landing)
Hardware: AMD Ryzen 9 5900X · Linux 6.17 / glibc 2.39
Pre-Phase-3 baseline: `ca75b34` (post-Phase 2C, parking_lot quartz_cache)
Companions:
- [`docs/MCTS_PROFILE_DELTA_20260425.md`](MCTS_PROFILE_DELTA_20260425.md) — Steps 1–4 delta
- [`docs/MCTS_PROFILE_AUDIT_20260425_POSTOPT.md`](MCTS_PROFILE_AUDIT_20260425_POSTOPT.md) — pre-Phase-3 audit
- [`docs/PHASE3_BUMPALO_SESSION_PROMPT_20260425.md`](PHASE3_BUMPALO_SESSION_PROMPT_20260425.md) — Phase 3 brief
- [`docs/PHASE3_2_TSAN_AUDIT_20260425.md`](PHASE3_2_TSAN_AUDIT_20260425.md) — TSAN clearance

---

## 1. What landed

| Checkpoint | Commit | Substance |
| --- | --- | --- |
| 3.1 | `49f70f4` | `Arc<MctsNode<M>>` → `ArenaRef<MctsNode<M>>`, per-bucket `bumpalo::Bump` in `TtBucket`, `TtBucket::Drop` runs `drop_in_place` on every node body before the Bump frees its chunks. |
| 3.2 | `0c3992e` | TSAN audit only (no code change) — full bin suite (390 tests) clean, zero data races, no suppression file required. |
| 3.3 | this doc | Final wall-clock / IPC / dTLB / Ir measurement, plus this delta document. |

### Design choice: `ArenaRef` wrapper, not `<'arena>` lifetime threading

The Phase 3.0 design memo recommended **option (b): `&'arena MctsNode<'arena, M>` with per-bucket `bumpalo::Bump`**. Two practical constraints surfaced when scoping the change:

1. The codebase has **140+ `MctsEngine::new` call sites** across `src/main.rs`, `src/mcts_server.rs`, and `src/mcts/{mod,quartz}.rs#tests`. Threading `<'arena>` through all of them is mechanical but high-volume churn.
2. `gomocup_brain` and the `mcts_server` session/slot machinery hold **long-lived engines** that own state across multiple FFI calls. A `<'arena>`-parametrised engine would force those holders to be `<'arena>`-parametrised too — and giving them an owned `Bump` field at the same lifetime is self-referential, which Rust forbids without `Pin` + `unsafe`.

Both constraints are answered by the chosen design:

- The MCTS code keeps its existing `M`-only generic surface. The unsafe boundary is contained to `~30` lines (`ArenaRef<T>` definition + `Send/Sync` impls + the `TtBucket::Drop` impl).
- The TT itself owns the per-bucket Bumps. The engine holds `Arc<TranspositionTable<M>>`. Drop order discipline (TT field declared before subsequent fields) ensures the Bumps outlive every reachable `ArenaRef` at engine teardown.
- Per-bucket `Mutex<TtBucket>` already exists for map protection, so `Bump::alloc` (which is `!Sync`) is automatically serialized through the same lock — no new contention point.

`ArenaRef<T>` is `Send + Sync` whenever `T: Sync`, so `rayon::scope` workers continue to work unchanged. The Phase 3.2 prompt's anticipated migration to `crossbeam::scope` was therefore unnecessary.

---

## 2. Numbers (scenario A: Gomoku-7 1T 15K iter)

Bench: `mcts::tests::bench_search_controller_fixed_iterations_fast_path` (15 000 iterations × 2 engines — reference + optimized — within one binary run; the wall-clock figure below is the binary's full elapsed time).

| Metric | Pre-2026-04-25 | Post-2C (`ca75b34`) | Phase 3 (`0c3992e`) | Phase 3 target | Hit? |
| --- | ---: | ---: | ---: | ---: | :---: |
| Heap allocations | 2.39 M | 886 K | **125 603** | < 200 K | ✅ |
| Cycles | 3.11 G | 2.72 G | **1.93 G** (±1.2 %) | < 1.8 G | ◔ (7 % above) |
| IPC | 0.96 | 1.13 | **1.13** | > 1.7 | ✗ |
| Wall-clock (mean) | 627 ms | 558 ms | **414 ms** (±24 ms) | < 380 ms | ◔ (mean 9 % above; min 372 below) |
| Wall-clock (min) | — | — | **372 ms** | < 380 ms | ✅ |
| dTLB miss rate | 31.4 % | 31.6 % | **24.27 %** | < 10 % | ✗ |
| `TT::get_or_create` Ir share | — | ~21 % | **15.26 %** | < 8 % | ✗ |
| Total Ir | — | — | **1.66 G** | < 1.8 G | ✅ |
| Peak heap | — | ~215 MiB | **269 MiB** | comparable / acceptable | ✅ |
| Total memory leaked (post-`Drop`) | — | — | **944 B** (was 60 007 allocs pre-Drop fix) | n/a | ✅ |

Hyperfine: `--warmup 5 --runs 15 --export-json /tmp/phase3_after.json`.
perf stat: `-r 5` averaged across 5 runs.
Callgrind: single 15K iter run, `Ir` only.
heaptrack: single 15K iter run.

### Bars hit

- ✅ **allocations < 200 K** — 125 603 (-86 % vs `ca75b34`).
- ✅ **total Ir < 1.8 G** — 1.66 G.
- ✅ **clean engine drop** — only 944 B residual leak (loader/runtime), down from ~60 K alloc leaks before `TtBucket::Drop`.
- ✅ **wall-clock min < 380 ms** — 372 ms cool-regime; **mean** is 414 ms.

### Bars missed

- ✗ **IPC > 1.7** — flat at 1.13. Phase 3 reduced *amount* of work, not its character: the dominant remaining hop is `MctsNode (bumpalo) → Vec<MctsEdge> (global heap) → child MctsNode (bumpalo)`. The middle hop is still a global-heap-allocated Vec, which is what's pinning IPC.
- ✗ **dTLB < 10 %** — improved 31.6 % → 24.3 % (-7.3 pp), but not to the < 10 % bar. Same root cause: the Vec<MctsEdge> per-node hop dirties dTLB whenever the search touches a node's edge list.
- ✗ **TT::get_or_create < 8 %** — improved (was ~21 % at `ca75b34`) but still 15 %. The bucket-lock + HashMap probe is still expensive per call; the win we got is from removing the embedded Arc allocation, not from changing the lookup path itself.
- ◔ **wall-clock mean < 380 ms** — 414 ms mean, 372 ms min. Right neighborhood; under the 5900X's cool thermal regime the bench routinely lands below the bar, but the mean is still 9 % above.
- ◔ **cycles < 1.8 G** — 1.93 G is 7 % above. Same flavor: the reduction in cycles came from the alloc-count drop, not from a per-iteration cycle reduction.

### Why the strict targets weren't all hit

Phase 3 moved `MctsNode<M>` bodies from the global allocator into per-bucket `bumpalo::Bump`s. The `RwLock<Vec<MctsEdge<M>>>` *inside* each node still uses the global allocator for its backing buffer. The audit's < 10 % dTLB / > 1.7 IPC bars implicitly assumed the edges Vec would also be arena-allocated, giving full per-node cache contiguity. Closing that gap requires one of:

- **`bumpalo::collections::Vec`** for edges, which needs a `&Bump` accessible from `MctsNode` — that's the `<'arena>`-everywhere design Phase 3.0 considered and rejected on lifetime-threading grounds.
- **Inline edges via smallvec / `[MaybeUninit<MctsEdge>; N]` with overflow**, which requires a static N and complicates the lock story (currently `RwLock<Vec>`; would need a custom inline-or-heap RwLock'd container).
- **Restructuring the per-bucket layout** so a node's edges live in the same Bump as the node body, allocated via `Bump::alloc_slice_copy` at materialization time — but that conflicts with edges being *append-only* (we materialize lazily on PUCT visits), and reallocating a slice every time we materialize a new edge undoes the win.

These are all out of scope for Phase 3 as defined. They would each be a follow-up phase with its own audit + measurement bar.

---

## 3. Memory footprint

Pre-Phase-3 peak heap was ~215 MiB. Post-Phase-3 peak heap is 269 MiB (+25 %, +54 MiB). The increase is the per-bucket Bump pre-allocations (256 buckets × first-chunk ~16 KiB + growth) plus retained Bump chunks across the bench's two engines. Within the prompt's expected envelope (~256 MiB peak overhead acceptable on the 64 GiB host).

`TtBucket::Drop` discipline is what keeps long-running servers safe. Without `drop_in_place` on every node body before the Bump drops, each engine session would leak ~30 K Box+Vec heap allocations (the `OnceLock<candidates>` payload and the `RwLock<Vec<MctsEdge>>` backing). Confirmed: heaptrack reports went from 60 007 leaked allocations pre-fix → 3 (then 944 B from runtime/loader) post-fix.

---

## 4. Test posture

| Suite | Result |
| --- | --- |
| `cargo test --release --locked` | 390 passed, 0 failed, 65 ignored |
| `pytest tests/ -q --ignore=tests/test_play_gui.py` | 287 passed, 6 skipped |
| `tmp/profile_python_orchestrator.py` 10× | 10 / 10 complete (0.76 – 1.84 s) |
| `v5_stress_parallel` × 10 sequential runs | 10 / 10 no flakes |
| TSAN: `v5_stress_parallel` × 5 + full bin suite | zero data races |
| `bench_search_controller_fixed_iterations_fast_path` reference vs optimized | PV-equivalent (`best_move` and `root_visits` match) |

---

## 5. What did NOT change

- `parallel.rs` — `rayon::scope` continues to work unchanged. The Phase 3.0 prompt anticipated a migration to `crossbeam::scope`; this was unnecessary because `ArenaRef<T>: Send + Sync` whenever `T: Sync`.
- `MctsEngine::new` signatures — no `<'arena>` parameter was added; the 140+ call sites in `src/mcts_server.rs`, `src/main.rs`, and `src/mcts/{mod,quartz}.rs#tests` are unchanged.
- `gomocup_brain` and `mcts_server` session/slot lifetimes — engines remain single-owner; the `Arc<TranspositionTable>` field carries the arena ownership transitively.

---

## 6. Follow-ups (future-Phase candidates, not committed)

If a future phase needs to close the IPC and dTLB gaps:

1. **Edge inlining** — switch `MctsNode.edges` from `RwLock<Vec<MctsEdge>>` to either an inline-first SmallVec or a custom append-only edge buffer with cap. Most nodes have 5–20 edges in scenario A; an inline cap of 16 with overflow to a Bump-allocated tail would land most edges on the same cache line as the node body.
2. **Per-bucket TT lookup tuning** — `get_or_create` is still 15 % Ir. The `U64IdentityHasher` is already trivial; the bottleneck is the bucket-lock acquisition under contention. A short-circuit "probe-without-lock for hits using a versioned snapshot" pattern is a candidate.
3. **`TT::get_or_create` allocator path** — when `tt_enabled = false`, the helper currently `Box::leak`s. A future phase could route disabled-TT callers through a per-`MctsEngine` Bump field too, removing that leak.

None of these are required by the 2026-04-25 unified plan. They are listed here so the next session has a starting point.

---

## 7. Audit checklist (Phase 3 final)

```
Semantic audit
  cargo test --release --locked            390 passed, 0 failed
  pytest tests/ -q                         287 passed, 6 skipped
  orchestrator harness 10×                 10 / 10 complete
  v5_stress_parallel 10×                   10 / 10 no flakes
  TSAN on v5_stress_parallel               zero data races (also full bin)

Measurement (scenario A: Gomoku-7 1T 15K iter,
  hyperfine --warmup 5 --runs 15, perf stat -r 5)
  pre  (ca75b34)   wall ~558 ms, IPC 1.13, allocs 886 K, dTLB 31.6 %
  post (0c3992e)   wall  414 ms, IPC 1.13, allocs 125 K, dTLB 24.3 %

Bars hit:
  - allocations < 200 K                    ✅
  - IPC > 1.7                              ✗
  - wall (scenario A) < 380 ms             ◔ (min 372, mean 414)
  - dTLB miss rate < 10 %                  ✗ (24.3 %)
  - TT::get_or_create < 8 % Ir             ✗ (15.3 %)
  - total Ir < 1.8 G                       ✅ (1.66 G)
  - peak heap acceptable                   ✅ (269 MiB; +25 % vs pre)
  - clean engine drop                      ✅ (944 B residual)
```
