# QUARTZ Rust Engine — Detailed Profiling Audit

Date: 2026-04-25
Commit: `29c452c` (post-audit-patch-set)
Hardware: AMD Ryzen 9 5900X (12C/24T) · 64 GB RAM · Linux 6.17 / glibc 2.39
Tools: `perf 6.17`, `valgrind/callgrind/cachegrind 3.22`, `heaptrack`, `hyperfine`, `cargo-flamegraph`, `cargo-bloat`, `cargo-llvm-lines`, plus a new static read of the engine
Scope: **Rust game-engine path only** — no Python orchestrator, no QIPC, no NN evaluator. ShortRollout / UniformEval evaluators only.

This audit complements the prior `MCTS_PROFILE_AUDIT_20260420.md` (now 5 days old). Some of its Priority-0 recommendations have landed; the picture has shifted, and new bottlenecks are now dominant.

---

## 0. TL;DR

- **Total instruction count is down 29%** (3.48 G Ir → 2.45 G Ir on the canonical Gomoku-7 single-thread benchmark) since the Apr-20 audit. The identity-hasher + bucket-striping work for the TT moved the needle.
- The engine is now **memory-management bound, not compute bound** on Gomoku-style games. Allocator + memcpy are ~38 % of cycles; per-move heap allocation (`Vec<i8>` board clone in `Gomoku::apply_move`) is the single largest contributor at **1.47 M of 2.39 M total allocations (61.6 %)** for a 15 K-iteration single-thread run.
- IPC contrast across games confirms it: Chess (Copy bitboards) reaches **2.87 IPC** at 0.71 % branch-miss; Gomoku (heap `Vec`) sits at **0.96 IPC** with **31 % dTLB miss**. Same engine, different game state representation, **3× IPC gap.**
- Two patches would, by static reasoning combined with the heaptrack call-site decomposition, eliminate ≥ 80 % of allocations in the Gomoku path:
  1. Convert `Gomoku::board: Vec<i8>` to `[i8; MAX_SQ]` so `apply_move` becomes a stack-Copy.
  2. Pool / arena `Arc<MctsNode>` so `TranspositionTable::get_or_create` stops doing `Arc::new` + `HashMap::insert` 760 K times.
- Several smaller wins remain: switch `node.edges` `RwLock` to `parking_lot`, hoist `lto = "fat"` and PGO, instrument `materialize_edges` for Vec-resize churn, and clip the always-on `Instant::now()` paths still visible in callgrind tail.
- Estimated cumulative wall-clock ROI on Gomoku-style searches: **~1.7×–2.5×** for the two P0 patches; **~3×** if all P0+P1 land.

---

## 1. Methodology and Workloads

### 1.1 Benchmark scenarios

All scenarios use the release test binary (`target/release/deps/mcts_demo-1a57d70367fc97b3`, 3.3 MB, built with `lto=thin`, `codegen-units=1`, `opt-level=3`).

| Tag | Test name | Workload | Purpose |
|---|---|---|---|
| **A** | `mcts::tests::bench_search_controller_fixed_iterations_fast_path` | 15 000 iters, Gomoku 7×7, single-thread, `UniformEval`, two engines (reference + optimized) | Pure-MCTS hot path; tiny game state; dominated by tree machinery |
| **B** | `experiment_chess::tests::bench_chess_nps` | Chess: 3 K iters single-thread, 10 K iters 4-thread, 100 K movegen calls | Stack-Copy game state (bitboards) |
| **C** | `experiment_go::tests::bench_go_nps` | Go 9/13/19, UniformEval + GoFastRollout, single + 4-thread | Stack-Copy mid-size state, real rollouts |
| **D** | `experiment_gomoku15::tests::test_nps_benchmark` | Gomoku-15 multi-variant, single-thread + 4-thread × 3 variants | Same heap-Vec allocator pattern at 15×15 |

### 1.2 Tools and outputs (all under `tmp/profiles_20260425/`)

| Tool | File | What it gives |
|---|---|---|
| `perf stat` | `{A,B,C,D}_perfstat.txt` | Cycles, IPC, branch miss, cache miss, dTLB miss |
| `perf record -F 997 -g --call-graph dwarf` | `A_perf.data`, `A_perf_report.txt` | Sampled CPU flame attribution |
| `valgrind --tool=callgrind` | `A_callgrind.out`, `A_callgrind.annotate.txt` | Exact instruction count per function |
| `valgrind --tool=cachegrind` | `A_cachegrind.out`, `A_cachegrind.annotate.txt` | Hierarchical Ir attribution |
| `heaptrack` | `A_heaptrack.zst`, `A_heaptrack.allocators.txt` | Per-call-site heap allocation counts + bytes |
| `hyperfine --warmup 1 --runs 5` | `hyperfine.txt`, `hyperfine.json` | Steady-state wall-clock comparison |

The `tmp/profiles_20260425/` directory is the persistent artifact set for this audit. Earlier `tmp/{game,mcts}_profiles/` directories remain for chronological comparison.

### 1.3 Caveats
- `perf record` produced 800 samples in 0.76 s — sampling is sparse, attribution is directional, not quantitative. Callgrind is the source of truth for instruction-level percentages.
- All scenarios run under `UniformEval` (no NN). NN-batched search behaviour is not in scope; that workload is dominated by Python/QIPC and was audited in `PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md`.
- Cachegrind ran without `--cache-sim=yes` (default `Ir` only) so cache-miss attribution per function comes from `perf stat` aggregates and inferred from heap-pattern, not from `cg_annotate`.

---

## 2. Hardware-Counter Cross-Section

`perf stat -d -d -d` on each scenario:

| Scenario | Wall (s) | Cycles | Insn | **IPC** | BrMiss | Cache miss | L1D miss | **dTLB miss** | NPS |
|---|---|---|---|---|---|---|---|---|---|
| **A** Gomoku 7 1T | 0.70 | 3.11 G | 3.00 G | **0.96** | 3.47 % | 19.6 % | 5.60 % | **31.4 %** | ~54 K (opt) |
| **B** Chess 1T+4T | 0.06 | 0.32 G | 0.93 G | **2.87** | 0.71 % | 17.1 % | 1.28 % | 7.79 % | 500 K (1T) / 770 K (4T) |
| **C** Go 1T+4T | 2.67 | 25.6 G | 29.0 G | **1.13** | 3.40 % | 9.64 % | 2.23 % | 7.59 % | 230 K (9 UE) / 7.5 K (4T FR) |
| **D** Gomoku-15 1T+4T | 0.12 | 1.06 G | 1.70 G | **1.60** | 2.04 % | 14.9 % | 2.03 % | 13.3 % | 600 K (1T) / 526 K (4T) |

### 2.1 Read-out

- **A vs B is the headline contrast.** Same engine, same selector, same TT, different game. Chess hits 3× the IPC and ⅕ the dTLB-miss rate of Gomoku. Branch prediction is also drastically better (0.71 % vs 3.47 %). Both deltas are textbook signatures of a workload that touches scattered heap pages vs one that lives in registers and L1D.
- **A's 31.4 % dTLB-miss rate is alarming.** With 4-KB pages and 64-entry L1 dTLB, a ~31 % miss rate means the working set spans hundreds of distinct pages — i.e., `Arc<MctsNode>` and `Vec<i8>` boards live on many different malloc arenas. Transparent huge pages don't help here because each allocation is small (a few hundred bytes) and `glibc` pools small objects by size class but not by access locality.
- **A's IPC = 0.96** means the CPU stalls roughly half its cycles waiting on memory, despite a healthy 96 % of the in-flight Ir budget being useful work. Cache miss + dTLB miss are the proximate cause.
- **Chess parallel scaling (B): 1.54×** for 4 threads is unimpressive but the workload is too short (10 K iters / 4 threads finished in 13 ms vs single-thread 6 ms). The 13 ms includes thread spawn cost. Re-run with 100 K+ iters to measure honest scaling; not done here because we already know what it is from prior audits and the present focus is allocation patterns, not parallel scaling.
- **Gomoku-15 4T (D) is *slower than 1T***: 526 K NPS at 4 T vs 600 K NPS at 1 T. Either thread spawn dominates a 22-ms run or genuine `node.edges`-`RwLock` contention. Static audit identified this `RwLock` (currently `std::sync::RwLock`, not `parking_lot::RwLock`) as a contention candidate; the dynamic data here is consistent.

### 2.2 What the IPC delta is *not*

- It is not branch-miss recovery: 3.47 % is bad but not catastrophic — branch-miss alone wouldn't drop IPC by 3×.
- It is not L1I miss: 0.44 % L1I-miss in scenario A. Code fits in icache.
- It is **memory access latency** — the L1D-miss rate is ~4× higher in A than B (5.60 % vs 1.28 %), and the TLB miss rate is **4× higher**. Each dTLB miss is a few-hundred-cycle stall; at 31 % miss rate over 1.1 G L1D loads, the TLB walks alone account for hundreds of millions of cycles.

---

## 3. Instruction-Level Hotspots (Callgrind)

### 3.1 Top-of-file program totals

```
2,449,047,192 (100.0%) PROGRAM TOTALS  (was 3,481,456,191 on Apr-20 → −29.7 %)
```

### 3.2 Top 20 functions

```
468,007,782  (19.11%)  TranspositionTable<M>::get_or_create
348,856,908  (14.24%)  __memcpy_avx_unaligned_erms                   (libc)
337,034,842  (13.76%)  MctsEngine<G>::iterate
308,469,378  (12.60%)  Gomoku::check_win_at
161,710,791  ( 6.60%)  Gomoku::apply_move (impl GameState)
136,004,279  ( 5.55%)  _int_free                                     (libc)
118,824,868  ( 4.85%)  malloc                                        (libc)
116,281,048  ( 4.75%)  _int_malloc                                   (libc)
 84,520,882  ( 3.45%)  expand::materialize_edges
 66,812,388  ( 2.73%)  free                                          (libc)
 57,857,119  ( 2.36%)  _int_free_merge_chunk                         (libc)
 54,877,430  ( 2.24%)  hashbrown::raw::RawTable::reserve_rehash
 26,244,228  ( 1.07%)  free → arena.c                                (libc)
 24,028,180  ( 0.98%)  drop_in_place<Vec<MctsEdge<usize>>>            (Vec drop)
 19,888,907  ( 0.81%)  _int_free_maybe_consolidate                   (libc)
 18,488,418  ( 0.75%)  unlink_chunk.isra.0                            (libc)
 17,421,196  ( 0.71%)  Gomoku::legal_moves
 14,629,132  ( 0.60%)  Arc::drop_slow
 11,155,862  ( 0.46%)  ipnsort                                        (sort_unstable in expand)
  9,991,528  ( 0.41%)  drop_in_place<TranspositionTable<usize>>       (TT teardown)
```

### 3.3 Bucketed view

| Bucket | Functions | % of cycles |
|---|---|---|
| **Allocator (libc)** | malloc / free / consolidate / merge_chunk / unlink / arena | **22.3 %** |
| **Memcpy** | `__memcpy_avx_unaligned_erms` (mostly Vec / Arc / state copies) | **14.2 %** |
| **TT** | get_or_create, reserve_rehash | **21.4 %** |
| **Game (Gomoku)** | check_win_at, apply_move, legal_moves | **19.9 %** |
| **MCTS core** | iterate, materialize_edges, expand_and_evaluate, ipnsort, Vec/Arc drop | **20.2 %** |

So **memory management ≈ 38 % of all cycles**, **game state machinery ≈ 20 %**, and the remaining ~42 % is split between TT, the search loop body, and engine bookkeeping.

### 3.4 Apr-20 → Apr-25 deltas (post-identity-hasher)

| Function | Apr-20 | Apr-25 | Δ |
|---|---|---|---|
| `BuildHasher::hash_one` | 7.29 % | < 0.5 % | **fixed** (identity hasher) |
| `DefaultHasher::write` | 4.92 % | < 0.5 % | **fixed** (identity hasher) |
| `TT::get_or_create` | 6.15 % | 19.11 % | **+12.96 pp** (was hidden under hash_one; now flat-attributed) |
| `_int_malloc` | 8.86 % | 4.75 % | **−4.11 pp** (raw allocator pressure dropped — smaller working set) |
| `MctsNode::set_parent` | 2.10 % | not visible | **fixed** (per the static audit, parent field appears to have been removed or refactored) |
| `Instant::elapsed` | 1.13 % | not visible | reduced |

The Apr-20 audit's Priority-0 fixes have landed. The 29.7 % total-Ir reduction is real and measurable. What's left is structural: **the Gomoku state representation forces malloc churn**, and **the TT path still does Arc::new + HashMap::insert per child**.

---

## 4. Allocation Pattern (Heaptrack)

### 4.1 Aggregate

```
total runtime:            1.24 s
calls to allocation:      2,386,239   (1.92 M / sec)
temporary allocations:    679,681     (28 % of total — alloc'd and freed within the same call window)
peak heap consumption:    220.96 MB
peak RSS:                 235.43 MB   (incl. heaptrack overhead)
total leaked:             944 B       (negligible — 3 leaks, all libc-side test framework)
```

### 4.2 Allocation by call site

```
1. Gomoku::apply_move                          1,470,098  (61.6 %)   peak 0 B
2. TranspositionTable::get_or_create             760,480  (31.9 %)   peak 103.43 MB
3. tcache / pthread setup / test runner            ~155,000 (6.5 %)   noise
```

`apply_move` calls 1.47 M times (≈ 49 alloc per iteration on a 15 K-iter run with two engines = 30 K total iter × ~49 each). Every call spawns a fresh `Vec<i8>` heap allocation for the cloned board. The static audit (`Gomoku::board: Vec<i8>` at `src/games/gomoku.rs:109`) is the structural cause; the 14.24 % memcpy attribution in callgrind is the runtime consequence (each Vec clone is a small memcpy that's rolled into the AVX path).

`get_or_create` calls 760 K times, peak 103 MB. Each call that misses creates one `Arc::new(MctsNode { … })` plus one `HashMap::insert`. The 256-bucket striping (`tt.rs:26`) keeps lock contention low for single-thread workloads, but the per-call heap cost is unavoidable as long as nodes are individually allocated `Arc`s.

### 4.3 Two call sites = 93.5 % of allocations

This is the cleanest finding of this audit. Eliminating allocation in those two sites is what changes the engine's allocation profile from "heavily heap-bound" to "near-Copy-game-throughput." The next two sections enumerate exactly how.

---

## 5. Static Audit Cross-References

The parallel static audit (separate sub-agent) corroborates the dynamic findings and adds these specific structural observations:

### 5.1 Gomoku state has 130–260 B per clone, ALL heap-resident

`src/games/gomoku.rs:106-121`:
```rust
pub struct Gomoku {
    pub size: usize,
    pub win_len: usize,
    board: Vec<i8>,            // ← 81-225 B HEAP per clone
    occupied: [u16; MAX_SQ],   // 722 B stack
    ...
}
```

The `occupied` field is already a fixed array sized to the maximum 19×19. The `board` field is `Vec<i8>` for no good reason — `MAX_SQ` is already declared in the file (line 29: `const MAX_SQ: usize = 19 * 19;`), so the same size bound applies. Switching `board` to `[i8; MAX_SQ]` makes `Gomoku` a `Copy`-or-cheap-Clone type:

- Clone cost: 0 heap alloc, 1 stack memcpy (~360 B)
- All `apply_move` allocations disappear.
- `Vec` capacity field, length, and the heap-pointer indirection go away — the resulting struct also gets cleaner `data layout` (everything contiguous), which improves cache behaviour for the inner `select` loop that touches `state.last_move`, `state.current_player`, etc.

This is a one-file, ~30-line patch.

### 5.2 TT bucket = `HashMap<u64, Arc<MctsNode<M>>>`

`src/mcts/tt.rs:62`:
```rust
type TtMap<M> = HashMap<u64, Arc<MctsNode<M>>, BuildHasherDefault<U64IdentityHasher>>;
```

Per insert: `MctsNode::new(...)` (one heap alloc for the node body, one for the inner Arc strong/weak counts pair → typical glibc small-bin allocation), plus `HashMap::insert` (which may grow → `reserve_rehash` at 2.24 %).

Two structurally cleaner alternatives, in increasing radicality:

a) **Pool** the `Arc<MctsNode<M>>` allocation. Either with `bumpalo` (single-threaded arena, then drop the whole arena at search end) or with a custom slab. Eliminates ~760 K allocations.

b) **Inline node body into the bucket**. `HashMap<u64, MctsNode<M>>` (no `Arc`) plus a separate parent-edge mechanism. Removes the Arc strong/weak counts entirely and gives sequential cache-friendly traversal at the cost of a substantial refactor (anywhere that holds an `Arc<MctsNode>` long-term needs to borrow instead).

(a) is the right size for "weekend patch with measurable wins." (b) is a refactor that would also let us replace the `HashMap` with an open-addressing dense table — but has too much surface to justify here.

### 5.3 `MctsEngine::iterate` at 13.76 % is loop overhead, not work

The `iterate` function in `mod.rs:1736` (the bench's harness) is the per-iteration outer-loop body. Of the 13.76 %, callgrind attributes substantial time to inlined Vec push/grow patterns from the `path` accumulator (`src/mcts/select.rs:621` — `let mut path = Vec::new();`), Arc clones in path-recording, and accessor calls. The static audit recommended `Vec::with_capacity(typical_depth)` here.

### 5.4 `node.edges: std::sync::RwLock<Vec<MctsEdge<M>>>`

`src/mcts/node.rs:242`. Used in select (read), expand (write), backup (read). The TT buckets use `parking_lot::Mutex` (tt.rs:15) but `node.edges` does not migrate. `parking_lot::RwLock` is ~2× faster on uncontended take and substantially better under contention. The Gomoku-15 4T regression (D, 0.88× scaling) is consistent with this being the binding constraint for parallel runs.

### 5.5 Always-on instrumentation

The static audit found that `profiling::maybe_start_timer()` is correctly gated at runtime by the `QUARTZ_MCTS_HOTPATH_METRICS` env var. The atomic counter `record_*` paths are similarly gated. This was the prior audit's biggest concern; it has been addressed. Callgrind doesn't show `Instant::now` in the top-25 in this run.

### 5.6 `check_win_at` at 12.6 % is *not* a runaway

Per the static audit, `check_win_at` is already O(win_len) per call — it scans 4 directions from the last move, not the whole board. At 30 K total iter × ~10 path levels per iter × win_len-per-direction = a lot of calls. It's hot because it's called a lot, not because it's badly written. The right way to reduce its cost is to reduce the *number of calls*, which is downstream of the iterate-loop and select-loop refactors.

---

## 6. Optimization Targets — Prioritized

Each item: rough estimated gain (cycles saved), patch difficulty (S/M/L), correctness risk, and validation handle.

### Priority 0 — high ROI, low risk

**P0-1. Gomoku: `board: Vec<i8>` → `[i8; MAX_SQ]`**
- **Gain (estimated):** kills ~62 % of all allocations in the Gomoku path. Combined with the `apply_move`-implicated portion of memcpy (likely ~6-8 pp of the 14.24 %), expect ~12-15 pp of total cycles back. **NPS gain: 1.5×–2× on Gomoku 7/15.**
- **Difficulty:** S. One file (`src/games/gomoku.rs`), ~30 lines. `MAX_SQ = 361` is already declared; the constraint is satisfied for all sizes.
- **Risk:** Memory-footprint per `Gomoku` struct grows from ~150-260 B (Vec heap + struct stack) to ~720+ B (full fixed array). Negligible at search scale.
- **Validation:** `cargo test --release`, then re-run scenario A. Heaptrack should show < 200 K allocations total. `apply_move` should drop out of the heaptrack top-3.

**P0-2. Pool `Arc<MctsNode>` allocations in the TT path**
- **Mechanism:** `bumpalo` per-search arena. `TranspositionTable::get_or_create` allocates `MctsNode` from the arena instead of `Arc::new`. Replace `Arc<MctsNode>` with `&'arena MctsNode` (or `Rc` on the single-thread path) where possible.
- **Gain:** kills the ~760 K TT allocations and trims `_int_malloc + _int_free + drop` cycles. Expected another 5-8 pp of total cycles.
- **Difficulty:** M. The `Arc<MctsNode>` is held in many places (edges, parent links, the Engine's root). A clean version would parameterize over a node-handle type; the brutal version replaces `Arc` with `Rc` for the test path and keeps `Arc` for parallel search via a small adapter trait.
- **Risk:** lifetime complexity, test-harness reach.
- **Validation:** allocations/s in heaptrack should fall below 200 K/s (from 1.92 M/s).

**P0-3. `cargo` profile: `lto = "fat"` and try PGO**
- **Mechanism:** `Cargo.toml`:
  ```toml
  [profile.release]
  opt-level = 3
  lto = "fat"            # was "thin"
  codegen-units = 1
  panic = "abort"
  ```
- **Gain:** typically 5-10 % across the board. PGO on top of that another 5-10 % if the training data includes a representative search profile.
- **Difficulty:** S (`lto = "fat"` is one line; build time ~3-5× slower for the release artifact). PGO is a multi-step `cargo pgo build` / instrumentation run.
- **Risk:** None for `lto = "fat"`. PGO requires a representative profile or the optimizer can pessimize edge cases.
- **Validation:** rerun scenario A under perf stat; expect Cycles to drop ~5-15 %.

### Priority 1 — moderate ROI

**P1-1. `node.edges`: migrate from `std::sync::RwLock` to `parking_lot::RwLock`**
- **Gain:** in single-thread runs, ~2× cheaper read-acquire (~15 ns → ~8 ns). In multi-thread runs, the Gomoku-15 4T regression should resolve. Estimate: 1-3 % cycles in single-thread, larger in multi-thread.
- **Difficulty:** S. `tt.rs` already uses `parking_lot`; just promote. Few callers.
- **Risk:** None. Same API.

**P1-2. Pre-size the `path` Vec in select**
- `src/mcts/select.rs:621` (`let mut path = Vec::new();`). Replace with `Vec::with_capacity(MAX_DEPTH_HINT)` (typically 16-20 for Gomoku).
- **Gain:** trivial, ~0.5 %, but eliminates one of the residual `Vec::push`/grow paths visible in `MctsEngine::iterate`.
- **Difficulty:** trivial.

**P1-3. Reuse a per-engine scratch buffer for `expand`'s candidate sort**
- `src/mcts/expand.rs:65` does `candidates.sort_unstable_by(...)`. This is unavoidable, but `ipnsort` showing 0.46 % at 11 M Ir suggests we're sorting the same shape repeatedly. The candidates Vec lifecycle could reuse capacity across expansions if the engine kept a thread-local scratch.
- **Gain:** small, < 1 %.
- **Difficulty:** S.

**P1-4. Investigate `hashbrown::reserve_rehash` (2.24 %)**
- Either pre-size each TT bucket at ~`MAX_ENTRIES_PER_BUCKET / 2` capacity, or accept the cost. The current 2.24 % is high for what should be a once-per-bucket-grow event. Likely an artifact of the bench starting with empty buckets and filling them rapidly during the warm-up.
- **Gain:** ~1-2 %.

### Priority 2 — defer

- **`MctsNode` field reordering** to separate `n_total + w_total` (cache-line-shared atomics) from cold fields. False-sharing risk is real but small; revisit only after P0-1/P0-2.
- **`MctsEdge` `n + w + m2` cache-line behaviour.** Same logic: trivial cost relative to the big wins above.
- **SIMD `check_win_at`.** It's already O(win_len); SIMD-ifying scans of length 4-5 buys nothing.
- **TT eviction policy.** Currently min-by-`n_total` linear scan up to 4096 entries; not in the hot path because eviction is rare in 15 K-iter runs.
- **Removing residual `Instant::now()` calls** in `iterate`. The static audit found these are gated; runtime cost is < 0.5 %.

### Priority 3 — speculative / breaks API

- Replace `std::collections::HashMap` in TT buckets with a fixed-capacity open-addressing table. Saves the `reserve_rehash` cost and improves cache locality, but breaks the eviction-by-min-visit policy and re-opens the collision-resolution choice.
- Pack `MctsEdge` to fit in one 32-byte cache line (current ~56 B). Requires moving `m2` and `virtual_value` to a side table accessed only on backup. Non-trivial.
- Lift the `RwLock<Vec<MctsEdge>>` to an `arc-swap`-style shared snapshot pattern. Read paths become wait-free, but writers reallocate the whole Vec — only profitable if reads vastly outnumber writes (probably true for late-stage searches but not for early expansion-heavy phases).

---

## 7. Concrete Patches (Pseudo-Code)

### 7.1 Gomoku board → fixed array

```rust
// src/games/gomoku.rs

pub struct Gomoku {
    pub size: usize,
    pub win_len: usize,
    board: [i8; MAX_SQ],          // was: Vec<i8>
    occupied: [u16; MAX_SQ],
    current_player: i8,
    hash: u64,
    move_count: u32,
    winner: i8,
    last_move: Option<usize>,
    recent_moves: [u16; GOMOKU_HISTORY_MOVES],
    recent_move_len: u8,
}

impl Gomoku {
    pub fn new_with_win(size: usize, win_len: usize) -> Self {
        debug_assert!(size * size <= MAX_SQ);
        Gomoku {
            size, win_len,
            board: [0; MAX_SQ],   // was: vec![0; size * size],
            occupied: [0; MAX_SQ],
            current_player: 1,
            hash: 0,
            move_count: 0,
            winner: 0,
            last_move: None,
            recent_moves: [0; GOMOKU_HISTORY_MOVES],
            recent_move_len: 0,
        }
    }
}

// All `&self.board[i]` indexing remains valid; only allocations of `Self::clone()`
// change from heap to stack.
```

The `Clone` derive remains. The change makes `Gomoku::clone()` a pure stack memcpy of ~728 B (board + occupied) + ~24 B of scalars = ~752 B. AMD's AVX2 memcpy handles this in one or two iterations; vs the current `Vec::clone()` which calls into the allocator.

### 7.2 Bumpalo arena for TT

```rust
// src/mcts/tt.rs (sketch)

use bumpalo::Bump;
use std::cell::UnsafeCell;

pub struct TranspositionTable<M: Copy + Send + Sync + 'static> {
    enabled: bool,
    arena: UnsafeCell<Bump>,        // single-threaded path; for parallel use a sharded set of bumps
    buckets: Vec<Mutex<TtBucket<'static, M>>>,
    // ...
}

pub fn get_or_create(&self, hash: u64, terminal_value: Option<f32>) -> &MctsNode<M> {
    if !self.enabled { return MctsNode::leaked(...); }   // fallback
    let idx = Self::bucket_idx(hash);
    let mut bucket = self.buckets[idx].lock();
    if let Some(node_ref) = bucket.map.get(&hash) { return *node_ref; }
    let node_ref: &MctsNode<M> = unsafe {
        let arena = &mut *self.arena.get();
        arena.alloc(MctsNode::new(hash, terminal_value))
    };
    bucket.map.insert(hash, node_ref);
    node_ref
}
```

The signature change from `Arc<MctsNode>` to `&'arena MctsNode` is the painful part. A milder version keeps the `Arc` API but routes the underlying allocation through `bumpalo::Bump::alloc` via `Arc::from_raw` after manual placement — saves the heap call per node but keeps Arc's reference counting. (This is the version I'd actually land first.)

### 7.3 `Cargo.toml` profile bump

```toml
[profile.release]
opt-level = 3
lto = "fat"          # was "thin"
codegen-units = 1
panic = "abort"      # new — saves ~1-2 % code size, no runtime impact

[profile.release-pgo]   # new optional profile for PGO builds
inherits = "release"
lto = "fat"

# Build with: cargo pgo instrument && ./target/release-pgo/mcts_demo --self-play …
#             cargo pgo optimize
```

### 7.4 `parking_lot::RwLock` for `node.edges`

```rust
// src/mcts/node.rs

use parking_lot::RwLock;   // was: use std::sync::RwLock;

pub struct MctsNode<M: Copy + Send + Sync + 'static> {
    pub hash: u64,
    pub terminal_value: Option<f32>,
    pub candidates: OnceLock<Box<[(M, f32)]>>,
    pub edges: RwLock<Vec<MctsEdge<M>>>,    // same type alias name, different impl
    // ...
}
```

`parking_lot::RwLock` is API-compatible at our usage level (`.read()`, `.write()` return guards). One subtlety: the std-lib `RwLock::read()` returns `LockResult<...>` (poison-aware); `parking_lot::RwLock` doesn't poison. Search for `.unwrap()` / `?` on the lock guards and remove them.

---

## 8. Validation Plan

After each patch, re-run:

1. **Sanity** — `cargo test --release --locked` (full suite, currently 386 passed).
2. **Per-scenario perf stat** — record IPC, branch miss, cache miss, dTLB miss for scenarios A/B/D.
3. **Heaptrack** — record total allocations, allocations/s, peak heap. Target: **< 250 K allocations** for scenario A (currently 2.39 M).
4. **Hyperfine** — 5-run wall-clock for scenarios A/B/D.
5. **Callgrind** — per-function attribution. Target: `TT::get_or_create` < 8 %, `apply_move` allocator path gone, total Ir < 1.5 G.

The artifact set (`tmp/profiles_20260425/`) is preserved as the **before** snapshot; the **after** runs land under `tmp/profiles_20260425_postP0/` etc.

A reasonable success bar:

| Metric | Before (Apr-25) | Target after P0 | Stretch (P0+P1) |
|---|---|---|---|
| Total Ir (scenario A) | 2.45 G | < 1.6 G | < 1.4 G |
| Allocations | 2.39 M | < 250 K | < 100 K |
| IPC (scenario A) | 0.96 | > 1.6 | > 2.0 |
| dTLB miss (scenario A) | 31 % | < 10 % | < 5 % |
| Wall-clock (scenario A) | 627 ms | < 380 ms | < 280 ms |
| 4T scaling (scenario D) | 0.88× | ≥ 2.5× | ≥ 3.5× |

---

## 9. Bottom Line

The Apr-20 audit's identity-hasher fix worked as advertised: the pure-hashing cost of TT lookup is gone, and total instruction count dropped 30 %.

What remains is **structural, not algorithmic**: the engine's tree allocates `Arc<MctsNode>` per child, and the Gomoku game state allocates `Vec<i8>` per move. Together those two facts account for **93.5 % of heap allocations** in the canonical benchmark and at least **30-35 % of total instruction cycles** (allocator + memcpy attributable to those two paths).

Two patches close that gap:

1. Make Gomoku's `board` a fixed-size array.
2. Pool MctsNode allocations in a per-search bumpalo arena.

Combined estimated wall-clock impact: **1.7×–2.5× faster** on Gomoku scenarios; smaller but non-zero (5–15 %) on Chess and Go where state is already Copy. With LTO=fat and `parking_lot::RwLock` on top, **3× faster** is plausible on Gomoku-15 4-thread runs.

The remaining hotspots after that — `MctsEngine::iterate`, `check_win_at`, `materialize_edges` — are intrinsic to the algorithm. Further wins from those require search-policy redesign or SIMD work that's outside the scope of "engine optimization with no semantic change."
