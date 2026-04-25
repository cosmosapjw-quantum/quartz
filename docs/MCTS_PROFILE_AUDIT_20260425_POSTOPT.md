# QUARTZ Rust Engine — Detailed Profiling Audit (post-optimization)

Date: 2026-04-25
Commit: `9232342` (post docs/MCTS_PROFILE_DELTA_20260425.md)
Hardware: AMD Ryzen 9 5900X (12C / 24T) · 64 GB DDR4 · Linux 6.17.0-20 / glibc 2.39
Compiler: rustc stable, `lto=fat`, `codegen-units=1`, `opt-level=3`, `-C target-cpu=native`
Binary: `target/release/deps/mcts_demo-569aa74e7f6c1833` (built with `RUSTFLAGS=-g`, 4.7 MiB)
Tools: `perf 6.17`, `valgrind/callgrind/cachegrind 3.22` (both `--cache-sim=yes` and default), `heaptrack 1.5`, `hyperfine 1.19`, `cargo-bloat`, `cargo-llvm-lines`, plus a full static code audit
Scope: **Rust MCTS engine only** — no Python orchestrator, no QIPC, no NN evaluator. `UniformEval` and short rollouts only.

This audit is the post-landing counterpart to [`docs/MCTS_PROFILE_AUDIT_20260425.md`](MCTS_PROFILE_AUDIT_20260425.md) (pre-patch) and [`docs/MCTS_PROFILE_DELTA_20260425.md`](MCTS_PROFILE_DELTA_20260425.md) (4-step delta). It adds `--cache-sim=yes` cachegrind, a fresh perf-record flame, a fresh heaptrack attribution at the post-patch steady state, and a focused static audit of the remaining hot path.

All raw artifacts are preserved under [`tmp/profiles_20260425_postopt/`](../tmp/profiles_20260425_postopt/).

---

## 0. TL;DR

- **Scenario A (Gomoku-7, 15 K iter, 1T) wall-clock: 627 ms → 558 ms (−11.0 %).** Instructions flat at ~2.40 G; cycles down to 2.72 G; IPC climbed 0.96 → 1.08 → **1.13** (up 18 % end-to-end). dTLB miss rate stayed at **31.6 %** — the structural TLB ceiling has not moved because the residual 760 K `Arc<MctsNode>` allocations still scatter the working set across hundreds of glibc pages.
- **Allocations collapsed from 2.39 M → 886 K (−62.9 %).** The `Gomoku::apply_move` Vec clones (1.47 M per run) are gone. Of the 886 K remaining, **760 K (85.8 %) are `TranspositionTable::get_or_create → Arc::new(MctsNode)`** — the single P0 lever that has *not* yet been pulled.
- **Callgrind top-1 is now `TT::get_or_create` at 21.67 %** (was 19.11 % pre-patch, up because the Vec-board allocator pressure that used to absorb % has evaporated). The #2 is `__memcpy_avx_unaligned_erms` at **19.65 %** — driven by `Gomoku::clone` (now a fat ~1 152-byte stack memcpy per `apply_move`) and by `Vec::push`/`grow` churn inside `materialize_edges`.
- **The Copy-state/heap-state gap is smaller but still decisive.** Chess (Copy bitboards) still hits 2.62 IPC and 9.28 % dTLB-miss; Gomoku-7 hits 1.13 IPC and 31.6 % dTLB-miss. Bumpalo + a thinner `Gomoku` clone are the remaining wins.
- **Three new static findings, all P1/quick-win:** (1) `check_win_at` (12.65 % cycles) lacks `#[inline]`; (2) `TtBucket::new()` uses `HashMap::default()`, forcing `reserve_rehash` (2.36 % cycles) during warmup; (3) `quartz_cache.read().unwrap()` at the top of every `iterate()` adds a guaranteed `std::sync::RwLock` acquisition pair to every iteration — this lock is still `std::sync`, not `parking_lot`, even though the Apr-25 delta migrated the *edges* lock.
- Estimated cumulative wall-clock ROI of the remaining known work: **another 1.6×–2.0× on Gomoku scenarios** once (a) bumpalo lands and (b) the three P1 items above are fixed. That would bring scenario A from 558 ms toward ~300 ms, closing the Copy-game gap from 9.9× to ~5×.

---

## 1. Methodology and Workloads

### 1.1 Scenarios

Release test binary `target/release/deps/mcts_demo-569aa74e7f6c1833`, compiled with `lto=fat`, `codegen-units=1`, `opt-level=3`, `target-cpu=native`, `RUSTFLAGS=-g` (debuginfo kept for DWARF call graphs).

| Tag | Test | Workload | Purpose |
|---|---|---|---|
| **A** | `mcts::tests::bench_search_controller_fixed_iterations_fast_path` | 15 000 iters × 2 engines (reference + optimized), Gomoku 7×7, single-thread, `UniformEval` | Canonical pure-MCTS hot path |
| **B** | `experiment_chess::tests::bench_chess_nps` | Chess: 3 K 1T + 10 K 4T + 100 K movegen | Copy (344-byte) game state baseline |
| **C** | `experiment_go::tests::bench_go_nps` | Go 9×9/13×13/19×19: UniformEval + GoFastRollout, 1T + 4T | Mid-size Copy state with real rollouts |
| **D** | `experiment_gomoku15::tests::test_nps_benchmark` | Gomoku-15 multi-variant, 1T + 4T × 3 variants | Post-patch 15×15 fixed-array state |

All scenarios run the optimized code path only; the two engines in Scenario A are the already-published "reference" and "optimized" variants of the MCTS inner loop, both using the same Gomoku state implementation.

### 1.2 Tools and outputs (all under [`tmp/profiles_20260425_postopt/`](../tmp/profiles_20260425_postopt/))

| Tool | File | What it gives |
|---|---|---|
| `perf stat -d -d -d` | `{A,B,C,D}_perfstat.txt` | Cycles, IPC, branch miss, L1/LL/TLB miss |
| `perf record -F 997 -g --call-graph dwarf` | `A_perf.data`, `A_perf_report.txt` | Sampled CPU flame (698 samples) |
| `valgrind --tool=callgrind` | `A_callgrind.out`, `A_callgrind.annotate.txt` | Exact per-function Ir |
| `valgrind --tool=cachegrind` (default) | `A_cachegrind.out`, `A_cachegrind.annotate.txt` | Hierarchical Ir |
| `valgrind --tool=cachegrind --cache-sim=yes` | `A_cachegrind_sim.out`, `A_cachegrind_sim.annotate.txt` | Simulated I1/LLi/D1/LLd load-miss + store-miss per function |
| `heaptrack` | `A_heaptrack.zst` | Per-call-site allocation counts + bytes |
| `hyperfine --warmup 3 --runs 10` | `hyperfine.txt`, `hyperfine.json` | Steady-state wall-clock |
| `cargo bloat --release --bin --crates -n 30` | `bloat_crates.txt`, `bloat_fns.txt` | `.text` section attribution |
| `cargo llvm-lines --release --bin` | `llvm_lines.txt` | LLVM-IR line counts (monomorphization surface) |

### 1.3 Caveats

- **`perf record` produced 698 samples in 0.58 s** — directional only. Callgrind and cachegrind's simulated counters are the source of truth for per-function attribution.
- All scenarios use `UniformEval`. NN-batched search is out of scope (audited separately in `PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md`).
- `cachegrind --cache-sim=yes` takes ~7.7 s to complete scenario A (vs. 0.58 s uninstrumented). The simulated cache geometry is the AMD 5900X's real geometry: I1/D1 = 32 KiB × 8-way, LL = 64 MiB direct-mapped (the tool auto-detects L3).
- Scenario A's bench `run_reference_fixed` → `run_optimized` pair allocates **two** full search trees back-to-back. The reported 886 K allocations / 2.40 G Ir / 558 ms all reflect both engines together. Per-engine numbers are half of these.

---

## 2. Hardware-Counter Cross-Section

`perf stat -d -d -d` per scenario, single run (variance quoted from `hyperfine` below):

| Scenario | Wall (s) | Cycles | Insn | **IPC** | BrMiss | L1D miss | **dTLB miss** | L1I miss | NPS |
|---|---|---|---|---|---|---|---|---|---|
| **A** Gomoku-7 1T, 15 K iter × 2 | 0.576 | 2.72 G | 3.06 G | **1.13** | 3.77 % | 5.65 % | **31.59 %** | 0.54 % | ~55 K (opt) |
| **B** Chess 1T+4T | 0.062 | 0.33 G | 0.87 G | **2.62** | 0.79 % | 1.26 % | 9.28 % | 0.51 % | 500 K (1T) / 834 K (4T) |
| **C** Go 1T+4T UE+FR | 2.17 | 21.5 G | 30.7 G | **1.43** | 3.22 % | 1.39 % | 5.42 % | 0.39 % | 230 K (9 UE) / 11 K (4T FR) |
| **D** Gomoku-15 1T+4T × 3 variants | 0.088 | 0.80 G | 1.52 G | **1.90** | 1.87 % | 1.78 % | 8.08 % | 0.61 % | 600 K (1T) / 1 111 K (4T) |

### 2.1 Delta vs. the pre-patch audit (same scenarios, `MCTS_PROFILE_AUDIT_20260425.md §2`)

| Scenario | IPC pre-patch | IPC now | Δ | dTLB miss pre-patch | dTLB miss now | Δ |
|---|---|---|---|---|---|---|
| A | 0.96 | 1.13 | **+18 %** | 31.4 % | 31.6 % | +0.2 pp |
| B | 2.87 | 2.62 | −9 % † | 7.79 % | 9.28 % | +1.5 pp |
| C | 1.13 | 1.43 | **+27 %** | 7.59 % | 5.42 % | −2.2 pp |
| D | 1.60 | 1.90 | **+19 %** | 13.3 % | 8.08 % | −5.2 pp |

† Scenario B is so short (62 ms end-to-end) that single-run perf-stat variance dominates; the hyperfine wall-clock shows B unchanged ±2 ms.

### 2.2 Read-out

1. **The Gomoku-7/Chess IPC gap closed from 3.0× to 2.3×** (0.96 → 1.13 vs. 2.87 → 2.62). Most of the A-side gain came from `lto=fat` and the board array migration feeding better decode throughput. What remains is a structural TLB+LLC gap from the scattered `Arc<MctsNode>` heap layout.
2. **A's 31.6 % dTLB-miss rate did not move.** This was flagged as a risk in the delta report — the Vec-to-array migration moved the board *into* the heap-allocated parent struct, so the page-count of the search's hot set did not shrink. The fix is bumpalo (P0-2, below).
3. **Gomoku-15 (D) 4T scaling is now healthy.** 1T 600 K NPS → 4T 1 111 K NPS on the Standard variant ≈ 1.85× with 4 threads on a 22-ms run (i.e., bounded by thread-spawn + epoch overhead rather than contention). The 4T regression from the pre-patch profile is gone — consistent with the `node.edges` parking_lot migration (Step 2) being the binding fix.
4. **Go (C) gained 27 % IPC and 2.2 pp dTLB improvement** for free, purely from the LTO-fat flip. Go's state is already fixed-size and its rollouts are bounded.
5. **Branch miss on A is still 3.77 %.** That's 21 M mispredicts over 559 M branches. The `select` tree descent has inherent data-dependent branching (UCB comparisons); this floor is hard to beat without redesigning the selection kernel.

---

## 3. Instruction-Level Hotspots (Callgrind, Scenario A)

```
2,373,659,187 (100.0%) PROGRAM TOTALS  (was 2,449,047,192 pre-patch → −3.1 %)
```

Ir flat vs. the pre-patch audit because the patches removed work per-iteration but didn't change the *count* of iterations; the IPC rise converts the same Ir into fewer cycles.

### 3.1 Top 20 functions (callgrind `--auto=no`)

| % Ir | Ir | Function |
|---|---|---|
| 21.67 % | 514.3 M | `TranspositionTable<M>::get_or_create` |
| 19.65 % | 466.5 M | `__memcpy_avx_unaligned_erms` (libc) |
| 14.50 % | 344.3 M | `MctsEngine<G>::iterate` |
| 12.65 % | 300.3 M | `Gomoku::check_win_at` |
| 5.39 % | 127.9 M | `Gomoku::apply_move` (impl GameState) |
| 4.90 % | 116.3 M | `_int_malloc` (libc) |
| 3.48 % | 82.5 M | `expand::materialize_edges` |
| 2.44 % | 57.9 M | `_int_free_merge_chunk` (libc) |
| 2.39 % | 56.7 M | `hashbrown::raw::RawTable::reserve_rehash` |
| 2.32 % | 55.0 M | `_int_free` (libc) |
| 2.16 % | 51.3 M | `malloc` (libc) |
| 1.05 % | 24.8 M | `free` (libc) |
| 0.97 % | 23.0 M | `drop_in_place<Vec<MctsEdge<usize>>>` |
| 0.84 % | 19.9 M | `_int_free_maybe_consolidate` (libc) |
| 0.78 % | 18.5 M | `unlink_chunk.isra.0` (libc) |
| 0.67 % | 15.9 M | `Gomoku::legal_moves` |
| 0.62 % | 14.6 M | `Arc::drop_slow` |
| 0.49 % | 11.5 M | `drop_in_place<TranspositionTable<usize>>` |
| 0.46 % | 11.0 M | `core::slice::sort::unstable::ipnsort` |
| 0.41 % | 9.7 M | `expand::expand_and_evaluate` |

### 3.2 Bucketed view

| Bucket | Functions | % Ir |
|---|---|---|
| **Allocator (libc)** | malloc / _int_malloc / free / _int_free / merge / consolidate / unlink / arena | **14.8 %** (was 22.3 %, −7.5 pp) |
| **Memcpy** | `__memcpy_avx_unaligned_erms` | **19.7 %** (was 14.2 %, +5.5 pp) |
| **TT** | get_or_create + reserve_rehash | **24.1 %** (was 21.4 %, +2.7 pp) |
| **Game (Gomoku)** | check_win_at + apply_move + legal_moves | **18.7 %** (was 19.9 %) |
| **MCTS core** | iterate + materialize_edges + expand_and_evaluate + drops + sort | **20.4 %** (was 20.2 %) |

**Interpretation.** Allocator cost dropped ~8 pp (the Vec-board allocations vanished), but memcpy *rose* ~5.5 pp — the pre-patch Vec::clone was a heap-bound memcpy rolled into the allocator budget, and the post-patch struct-clone is a fat 1 152-byte stack memcpy attributed cleanly to libc memcpy. Net memory-management share is slightly down (34.5 % vs. 38.0 %), and the *composition* has shifted from allocator to bulk copy — which is a qualitatively better place to be because bulk memcpy runs at DRAM bandwidth (≥ 30 GB/s on AVX2) while `_int_malloc` is latency-bound.

### 3.3 Cachegrind `--cache-sim=yes` per-function miss attribution (Scenario A)

| Function | Ir | D1 read-miss | LLd read-miss | D1 write-miss | LLd write-miss |
|---|---|---|---|---|---|
| `TT::get_or_create` | 21.4 % | **51.1 %** | 4.4 % | **31.8 %** | **29.8 %** |
| `__memcpy_avx_unaligned_erms` | 19.4 % | 1.9 % | 0.0 % | 9.9 % | 0.1 % |
| `MctsEngine::iterate` | 14.3 % | 17.4 % | 9.1 % | 3.1 % | 0.0 % |
| `check_win_at` | 13.8 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| `apply_move` | 5.3 % | 0.9 % | 0.0 % | 0.0 % | 0.0 % |
| `materialize_edges` | 3.4 % | 0.5 % | 0.3 % | **23.4 %** | **31.6 %** |
| `reserve_rehash` | 2.4 % | 0.9 % | 0.0 % | 5.0 % | 7.2 % |
| `drop_in_place<Vec<MctsEdge>>` | 1.2 % | **9.4 %** | **42.1 %** | 0.0 % | 0.0 % |
| `_int_malloc` | 4.9 % | 0.7 % | 0.0 % | **16.2 %** | **21.6 %** |
| `Arc::drop_slow` | 0.6 % | 2.1 % | **13.7 %** | 0.0 % | 0.0 % |

This is the cleanest signal in the whole audit. **`TT::get_or_create` alone accounts for 51 % of *all* D1 read-misses** in the benchmark and 30 % of LLd write-misses. That matches the heaptrack finding exactly — the Arc+HashMap-insert pattern is scattering pointer-chased loads across the D1 cache. Likewise, `drop_in_place<Vec<MctsEdge>>` contributes 42 % of LLd read-misses at only 1.2 % Ir — scanning across 760 K deallocated Arc'd edges at program exit.

`check_win_at`, despite being 13.8 % Ir, misses *nothing* (0.0 % D1, 0.0 % LLd). It is pure compute over the 361-byte board array that lives entirely in L1D. It is hot because it's called ~30 K × ~10 path levels × ~2 directions × win_len = lots of calls, not because it's badly written.

### 3.4 perf record (698 samples, exclusive time)

| % cycles | Function |
|---|---|
| 18.43 % | `Gomoku::apply_move` |
| 7.70 % | `TranspositionTable::get_or_create` |
| 5.68 % | `_int_malloc` |
| 5.06 % | `__memmove_avx_unaligned_erms` |
| 3.62 % | `MctsEngine::iterate` |
| 3.42 % | kernel `clear_page_erms` (mmap zero-fill for arena growth) |
| 3.27 % | `Gomoku::check_win_at` |
| 3.14 % | `_int_free_merge_chunk` |
| 2.49 % | `malloc` |
| 2.43 % | `_int_free` |
| 2.06 % | `drop_in_place<Vec<MctsEdge<usize>>>` |
| 1.35 % | kernel `do_mprotect_pkey` |
| 1.08 % | kernel `__alloc_frozen_pages_noprof` |

The interesting divergence between perf-record (sample-based) and callgrind (Ir-based) is `apply_move`: **perf says 18 %, callgrind says 5 %**. The 5 % callgrind number is the hot instructions inside `apply_move`; the 18 % perf number is the *cycles*, and the bulk of those cycles are spent waiting on memcpy (the `self.clone()` on line 385) which is inlined into `apply_move` at -O3 and attributed to `apply_move` in the flat sample profile. Together they triangulate the same ~1 152-byte stack-copy cost.

The kernel-side 3.42 % `clear_page_erms` + 1.35 % `do_mprotect_pkey` + 1.08 % `__alloc_frozen_pages_noprof` = **5.85 % of cycles spent in the kernel satisfying glibc's heap growth requests**. Another symptom of the Arc-churn pattern.

---

## 4. Allocation Pattern (Heaptrack)

### 4.1 Aggregate

```
total runtime:            0.83 s
calls to allocation:      886,083    (1.08 M / sec — was 2.39 M)
temporary allocations:    60         (was 679,681 — one-time at startup only)
peak heap consumption:    214.88 MB  (was 220.96 MB)
peak RSS:                 235.29 MB
total leaked:             944 B      (3 leaks, all libc test framework)
```

### 4.2 Allocations by call site

```
1. TranspositionTable::get_or_create                 760,480  (85.83 %)   peak 97.34 MB
2. pthread / tcache / test runner                   ~125,603  (14.17 %)   noise
```

The Vec-board path is *gone* from this table — zero temporary allocations, zero bytes. Every remaining allocation with any mass originates from the TT's Arc creation.

The 760 K TT allocations decompose roughly as:
- `Arc::new(MctsNode { ... })` per miss (760 K)
- Internal `HashMap::insert` that occasionally triggers `reserve_rehash` (~2 516 growth events capturing 8.92 MB peak)
- Dependent: `MctsEdge::new(mv, child, prior)` pushes into the per-node edges Vec (~380 K edges materialized)

### 4.3 Temporary allocations collapsed to 60

Pre-patch there were 680 K temporaries (alloc + free within the same ms). Post-patch there are 60 — a 99.99 % reduction, all attributable to the Vec board. This confirms the P0-1 patch worked as intended.

---

## 5. Static Code Audit (current tree, HEAD=9232342)

A parallel static read of the engine found three new quick-win items that were not called out in the Apr-25 pre-patch audit because the signals that expose them only surface *after* the Vec-board path was fixed.

### 5.1 `check_win_at` has no `#[inline]` pragma

`src/games/gomoku.rs:288` — `fn check_win_at(&self, pos: usize, player: i8) -> bool`. At **12.65 % of Ir** and called from the `apply_move` hot path, this function should carry an inline hint. The function body is ~40 LOC and well below LLVM's default inlining threshold, but with `lto=fat` the compiler *can* inline it across crate boundaries if invited. Currently it is called at `src/games/gomoku.rs:399` inside `apply_move`; static disassembly suggests it is inlined there today (it's visible as a separate symbol in perf only because it's also called from `random_legal_move` and `check_win_at_hypothetical`). An explicit `#[inline]` would pin the inline decision across future refactors at zero cost. Expected gain: 0 – 1 %, mostly insurance against regressions.

### 5.2 `TtBucket::new()` creates an empty `HashMap` without capacity

`src/mcts/tt.rs:70` —

```rust
impl<M: ...> TtBucket<M> {
    fn new() -> Self {
        TtBucket { map: TtMap::default() }
    }
}
```

The 256 buckets are created empty. As scenario A fills them with ~760 K / 256 ≈ 2 970 entries each, every bucket's underlying `hashbrown::RawTable` goes through ~12 doublings (1 → 2 → 4 → … → 4096) and each doubling calls `reserve_rehash`. That is the 2.39 % (~56.7 M Ir, 260 K D1 write-miss) signal in callgrind.

Fix (one line):

```rust
fn new() -> Self {
    TtBucket { map: TtMap::with_capacity_and_hasher(
        MAX_ENTRIES_PER_BUCKET / 2,
        Default::default(),
    ) }
}
```

`MAX_ENTRIES_PER_BUCKET` is 4 096; preallocating half eliminates all but the last doubling. Expected gain: 1 – 2 % Ir, measurable as `reserve_rehash` disappearing from the callgrind top-20.

### 5.3 `quartz_cache` is still `std::sync::RwLock`, and `.read().unwrap()` runs every iteration

`src/mcts/mod.rs:348–356`:

```rust
let qstate: Option<(QuartzStats, QuartzConfig)> = if let Some(qcfg) = &self.config.quartz {
    self.quartz_cache
        .read()
        .unwrap()                   // std::sync::RwLock — poison-aware
        .as_ref()
        .map(|s| (s.clone(), qcfg.clone()))
} else {
    None
};
```

Three concerns:

1. **`std::sync::RwLock::read()` with poison unwrap** is ~15–20 ns uncontended on glibc. The `node.edges` lock was migrated to `parking_lot` (Step 2) but `quartz_cache` was explicitly kept on `std` per the delta doc's §3 audit note. That was defensible at the time; post-patch it's the only std-sync lock left in the hot loop, and scenario A calls `iterate` 30 K times. 30 K × ~20 ns = ~0.6 ms of lock-acquire work per scenario, or ~0.1 % wall-clock. Low priority, but the asymmetry with `node.edges` and `tt::Mutex` (both parking_lot) is now an inconsistency the codebase could retire.
2. **The `if let Some(qcfg) = &self.config.quartz` branch is taken every iteration**. In scenario A the MCTS config does *not* enable quartz, so the `None` branch runs 30 K × 2 engines = 60 K times. The short-circuit is correct, but the compiler has no way to hoist this check out of the loop because `iterate` is called from a virtual-ish `run` loop. Not a bug, but a structural pessimism — moving this into the `MctsEngine` construction (a boolean on the struct) would save a single load-and-compare per iteration.
3. **`qcfg.clone()`** appears inside the `Some` branch. Clone of a `QuartzConfig` (small struct of mostly POD fields) is cheap, but the presence of `.clone()` in a per-iteration read path is a minor code-smell.

Collectively, low priority (P2). Listed here because the codebase has now optimized everything more expensive; this is the remaining idiomatic improvement.

### 5.4 Observations carrying over from the delta report

- `node.edges` is already `parking_lot::RwLock` (confirmed at `src/mcts/node.rs:22` and `:248`). No action.
- `select` path Vec is already pre-sized (confirmed at `src/mcts/select.rs:626-627`: `let path_capacity = if max_depth > 0 { max_depth } else { 16 }; let mut path = Vec::with_capacity(path_capacity);`). No action.
- `Gomoku::board` is `[i8; MAX_SQ]` (confirmed at `src/games/gomoku.rs:117`). The semantics-audit tests are in place. No action.
- `Cargo.toml` has `lto = "fat"`, `codegen-units = 1`, `opt-level = 3` (confirmed). `panic = "abort"` is **still intentionally not set** (comment in `Cargo.toml:27-30` — tests rely on `catch_unwind`). No action.

### 5.5 Residual hot-path observations

- **`iterate` uses Instant-based timers guarded by `profiling::maybe_start_timer()`**. Each call path involves an atomic load (to check the env-var cache) before deciding to call `Instant::now`. At 2 × 30 K iterations and 5 timer sites per iteration, that's 300 K atomic loads, ~0.1 % of Ir. The load itself is relaxed and predicted; callgrind confirms `Instant::now` is not in the top-50. No action.
- **`expand::materialize_edges` does `guard.reserve(additional)` inside the write lock.** The `reserve` call runs ~380 K times; each is a constant-time check (the Vec already has capacity if a previous materialization sized it). Defensible. No action.
- **`Gomoku::apply_move`'s 1 152-byte struct clone** is unavoidable in the current value-semantics design. The `occupied: [u16; MAX_SQ]` field at 722 B is the dominant cost; `board: [i8; MAX_SQ]` is 361 B. The `occupied` field is an optimization — it caches piece positions to avoid O(n²) scans in some code paths. A size-aware or bit-packed alternative could halve the clone size at the cost of non-trivial refactor. Deferred (P2).

### 5.6 Codegen and binary geometry

- **`.text` section: 2.1 MiB, 63.4 % of which is `mcts_demo` itself, 28.1 % `std`.** No individual crate beyond std and rayon-core (48 KiB) cracks 50 KiB. Nothing to trim at the linker level.
- **LLVM IR lines: 863 K total across 15 942 functions.** The top consumers are server/FFI paths (`handle_selfplay_nn_run_generic` at 25 K lines × 5 copies, `handle_eval_nn_run_generic` at 17 K × 5). Nothing in the hot MCTS path exceeds 5 K lines; `mcts::select::select` is at 4 300 × 6 copies (one per concrete `GameState` — Gomoku, Gomoku15, Chess, Go, TicTacToe, and one for the test harness). Monomorphization footprint is reasonable.
- **No dead code in the hot path.** A handful of unused helpers (`all_occ`, `group_liberties_on_board`, `fisher_puct_score_with_parent_sqrt`, `eft_puct_score`) warn under `cargo build`. These don't ship but clutter warnings — a small cleanup PR.

---

## 6. Optimization Targets — Prioritized (post-patch)

Each item: rough gain, patch size, correctness risk, validation handle.

### Priority 0 — big remaining structural lever

**P0-A. Pool `Arc<MctsNode>` in a per-search `bumpalo::Bump` arena**

- **Gain:** Eliminates 760 K of the 886 K allocations (86 %). Expected to move `TT::get_or_create` from 21.67 % → ~6 % Ir and drop the D1 read-miss share it owns (51 %) by roughly the same factor. Because the arena allocates contiguously, the **dTLB-miss rate should finally move**; target < 10 % (vs. stuck at 31.6 %). Expected wall-clock: **scenario A from 558 ms → ~320–380 ms (−30 – 45 %)**.
- **Difficulty:** M. Two viable designs:
  - **(a) Preserve the `Arc` API.** Keep `Arc<MctsNode>` signatures everywhere; route only the *body* allocation through `bumpalo::Bump::alloc` + `Arc::from_raw`. Saves the body heap call per node but retains Arc strong/weak counts. This is ~1 day of work, single-threaded first, then a sharded-bump + atomic-count for parallel.
  - **(b) Replace `Arc<MctsNode>` with `&'arena MctsNode`.** Higher ceiling (also removes the strong/weak counts and makes `Arc::drop_slow` disappear), but requires lifetime surgery in `mcts/tt.rs`, `node.rs` (`MctsEdge::child`), `select.rs`, `expand.rs`, `backup.rs`, `parallel.rs`, `mod.rs`, the search harness, and any Python-FFI consumer holding a handle.
- **Risk:** Lifetime surgery (b); `Arc::from_raw` correctness + drop-timing (a); parallel semantics in both.
- **Validation:** `cargo test --release --locked` (390 tests); scenario A heaptrack shows < 200 K allocations; perf stat dTLB miss < 10 %; TSAN on `zobrist_tt_parallel_verify::v5_stress_parallel`.

This is P0-2 from the Apr-25 audit, still open, and it is the single biggest measurable win remaining in the engine.

### Priority 1 — quick wins identified here

**P1-A. Pre-size TT bucket HashMap**

- **Patch:** `src/mcts/tt.rs:70` — change `TtMap::default()` to `TtMap::with_capacity_and_hasher(MAX_ENTRIES_PER_BUCKET / 2, Default::default())`. One line.
- **Gain:** `reserve_rehash` (currently 2.39 % Ir, 56.7 M Ir) disappears. Expect callgrind top-20 to lose it entirely.
- **Risk:** Zero — pre-sizing a HashMap is a semantically invisible capacity hint.

**P1-B. `#[inline]` on `check_win_at` and `check_win_at_hypothetical`**

- **Patch:** `src/games/gomoku.rs:288` and `:245`. Add `#[inline]`.
- **Gain:** 0 – 1 %; insurance.
- **Risk:** Zero.

**P1-C. Hoist `quartz_cache` lock acquisition out of `iterate` when `self.config.quartz.is_none()`**

- **Patch:** At `src/mcts/mod.rs:347-356`, structure as:
  ```rust
  let qstate = self.config.quartz.as_ref().and_then(|qcfg| {
      self.quartz_cache.read().ok()
          .and_then(|guard| guard.as_ref().map(|s| (s.clone(), qcfg.clone())))
  });
  ```
  Or (cleaner): migrate `quartz_cache` to `parking_lot::RwLock` for consistency with the rest of the codebase.
- **Gain:** Minor, < 0.5 %.
- **Risk:** Lock-API difference (`parking_lot::RwLock::read()` doesn't return `LockResult`). Verified in the delta-report Step 2 audit that this is safe for the other migrated lock; same applies here.

**P1-D. Investigate Vec-drop LLd read-miss tail**

- **Observation:** `drop_in_place<Vec<MctsEdge<usize>>>` is **42.1 % of LLd read-misses** at only 1.2 % Ir — a pure pointer-chasing pattern that occurs at search teardown when each edge Vec drops its per-child `Arc<MctsNode>`. Per-node edges Vec is created fresh each materialization; at teardown we traverse 760 K Arcs to drop them.
- **Mitigation:** Subsumed by P0-A — an arena that drops whole in O(1) makes both the Arc drop and the Vec drop cheap.
- **Gain alone:** Not worth pursuing without the arena.

### Priority 2 — defer

- **`MctsNode` / `MctsEdge` field reordering.** `MctsNode` currently packs `n_total: AtomicU32` (4 B) and `w_total: AtomicU64` (8 B) next to three `rtt_*` atomics. No false-sharing risk detected in the current profile (scenario A is single-thread; scenario D shows healthy 4T scaling). Revisit only if Gomoku-15 4T regresses again.
- **`Gomoku::occupied: [u16; MAX_SQ]` size.** 722 B of stack memcpy per move. Not touched in the Apr-25 patchset because `occupied` is load-bearing (see `legal_moves` and `random_legal_move`). A 15×15 game only uses the first 225 entries; an alternative representation (bitboard or RLE) would cut the clone cost. Estimated 2–4 % wall-clock. Deferred pending P0-A (after arena-induced cache locality gains, the struct-clone profile will shift).
- **PGO.** Independent of everything above. After P0-A lands, a PGO build over scenario A+B+D training data could capture another 5–10 %. Ordering: land P0-A first so the profile has the right shape.
- **`panic = "abort"` in release.** 1–2 % binary size, no runtime impact, but explicitly off per `Cargo.toml:27-30` comment.

### Priority 3 — speculative / redesign

- **Open-addressing TT with fixed-capacity buckets.** Saves `reserve_rehash` entirely and improves cache locality over `hashbrown`, but changes eviction semantics. Deferred.
- **`MctsEdge` cache-line packing.** Edge is currently ~56 B; padding to 64 B or compressing aux fields to a side table is ~4 % bandwidth but breaks API. Deferred.
- **SIMD `check_win_at`.** Already O(win_len) — not worth vectorizing 4-element scans.

---

## 7. Concrete Patches (code sketches)

### 7.1 Pre-size TT buckets (P1-A)

```rust
// src/mcts/tt.rs
const MAX_ENTRIES_PER_BUCKET: usize = 4096;

impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        TtBucket {
            map: TtMap::with_capacity_and_hasher(
                MAX_ENTRIES_PER_BUCKET / 2,
                Default::default(),
            ),
        }
    }
}
```

Cost: 256 × ~16 KiB HashMap backing array = ~4 MiB fixed overhead per `TranspositionTable`. Already dwarfed by the current 215 MiB peak.

### 7.2 Inline `check_win_at` (P1-B)

```rust
// src/games/gomoku.rs:288
#[inline]
fn check_win_at(&self, pos: usize, player: i8) -> bool {
    // existing body
}

// src/games/gomoku.rs:245
#[inline]
fn check_win_at_hypothetical(&self, pos: usize, player: i8) -> bool {
    // existing body
}
```

### 7.3 Bumpalo arena sketch (P0-A, single-thread first)

```rust
// src/mcts/tt.rs — version (a): keep Arc API, route body through arena

use bumpalo::Bump;
use std::cell::UnsafeCell;
use std::sync::Arc;

pub struct TranspositionTable<M: Copy + Send + Sync + 'static> {
    enabled: bool,
    arena: UnsafeCell<Bump>,                          // single-thread path
    buckets: Vec<Mutex<TtBucket<M>>>,
    // ...
}

// SAFETY: arena is only touched while holding a bucket Mutex.
unsafe impl<M: Copy + Send + Sync + 'static> Sync for TranspositionTable<M> {}

impl<M: Copy + Send + Sync + 'static> TranspositionTable<M> {
    pub fn get_or_create(&self, hash: u64, terminal_value: Option<f32>) -> Arc<MctsNode<M>> {
        if !self.enabled {
            return MctsNode::new(hash, terminal_value);
        }
        let idx = Self::bucket_idx(hash);
        let mut bucket = self.buckets[idx].lock();
        if let Some(node) = bucket.map.get(&hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(node);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        // ... eviction ...

        // Allocate body in arena, wrap in Arc via from_raw
        let arena = unsafe { &mut *self.arena.get() };
        let body_ptr: &mut MctsNodeBody<M> = arena.alloc(MctsNodeBody::new(hash, terminal_value));
        let node: Arc<MctsNode<M>> = unsafe {
            // Construct Arc over arena-owned storage by manually managing refcount.
            // This requires moving the Arc strong/weak counts next to the body
            // or keeping them in a parallel small-object pool.
            Arc::from_raw(body_ptr as *const _ as *const MctsNode<M>)
        };
        bucket.map.insert(hash, Arc::clone(&node));
        node
    }
}
```

The sketch is illustrative — `Arc::from_raw` + arena ownership has sharp edges (the arena must outlive every Arc; the `MctsNode` must contain a matching `ArcInner` layout). A more conservative landing is a slab/pool allocator (`typed-arena` or a hand-rolled `Vec<MctsNode>`), which keeps Rust's ownership model intact at the cost of needing a `typed-arena::Arena::alloc` returning a `&'arena mut MctsNode` and replacing `Arc<MctsNode>` with `&'arena MctsNode` everywhere. That's the version I'd actually land.

Scope estimate: **1–2 working days** for single-thread + all 390 tests green; **+1 day** for parallel sharded arena + TSAN pass.

### 7.4 Consistency migration: `quartz_cache` to `parking_lot::RwLock` (P1-C)

```rust
// src/mcts/mod.rs
use parking_lot::RwLock;  // was: std::sync::RwLock

pub struct MctsEngine<G: GameState> {
    // ...
    quartz_cache: RwLock<Option<QuartzStats>>,
    // ...
}

// And update the read sites from `.read().unwrap().as_ref()` to `.read().as_ref()`.
```

---

## 8. Validation Plan

The pre-patch audit's validation framework carries over unchanged:

1. `cargo test --release --locked` — **390 tests must pass** (currently 390 pass, 65 ignored as `#[ignore]` benches).
2. Per-scenario `perf stat -d -d -d` — record IPC, branch miss, cache miss, dTLB miss.
3. Heaptrack — target **< 200 K allocations** for scenario A (currently 886 K) after P0-A.
4. `hyperfine --warmup 3 --runs 10` — 10-run wall-clock for A / B / D.
5. Callgrind — per-function attribution. After P0-A target: `TT::get_or_create` < 8 %, Vec/Arc drop tail < 2 %, total Ir < 1.8 G.
6. `zobrist_tt_parallel_verify::v1_parallel_self` through `v5_stress_parallel` — parallel correctness. Run under TSAN for P0-A.

### Success bar — incremental

| Metric | **Pre-patch** (Apr-25) | **Post-patch now** (measured) | **Target after P1-A/B/C** (est.) | **Target after P0-A** (est.) |
|---|---|---|---|---|
| Total Ir (A) | 2.45 G | 2.40 G | < 2.3 G | < 1.8 G |
| Allocations (A) | 2.39 M | 886 K | 886 K | < 200 K |
| IPC (A) | 0.96 | 1.13 | 1.15 | > 1.7 |
| dTLB miss (A) | 31.4 % | 31.6 % | 31 % | < 10 % |
| Wall-clock (A) | 627 ms | 558 ms | ~540 ms | < 380 ms |
| 4T scaling (D) | 0.88× | 1.85× | 1.85× | ≥ 2.5× |

---

## 9. Cross-scenario hyperfine (steady-state, `--warmup 3 --runs 10`)

| Scenario | Wall-clock (mean ± σ) | Range | vs. Chess |
|---|---|---|---|
| A: Gomoku-7 1T (15 K iter × 2) | **558.1 ms ± 10.8 ms** | 541.5 – 578.1 | 9.91× |
| B: Chess 1T+4T | **56.3 ms ± 1.4 ms** | 54.6 – 58.0 | 1.00× |
| D: Gomoku-15 multi | **85.6 ms ± 4.0 ms** | 80.1 – 94.8 | 1.52× |

Chess is the reference — the engine's upper bound on per-iteration throughput, achieved by a fully-Copy game state. Gomoku-7 at 9.91× Chess and Gomoku-15 at 1.52× Chess show that the structural gap is real, is state-representation-driven (not engine-core-driven), and is amenable to further optimization.

---

## 10. Bottom Line

The Apr-25 patch set (lto=fat, parking_lot edges, path capacity, Gomoku fixed-array board) landed cleanly and delivered on the audit's predictions:

- **Wall-clock scenario A: 627 ms → 558 ms (−11 %).**
- **IPC: 0.96 → 1.13 (+18 %).**
- **Allocations: 2.39 M → 886 K (−63 %).**
- **Cycles: 3.11 G → 2.72 G (−13 %).**
- Zero test regressions (386 → 390 tests, +4 semantics-audit invariants).

What remains is **one big structural lever** (bumpalo arena for the 760 K `Arc<MctsNode>` allocations, P0-A) and **three low-risk quick wins** (pre-size TT buckets, `#[inline]` on `check_win_at`, `parking_lot` for `quartz_cache`). Landing all four is expected to take scenario A from 558 ms to ~320–380 ms (−30 – 45 % further), closing the Gomoku-7/Chess gap from 9.9× to ~5×.

The dTLB miss rate (31.6 %) is the load-bearing number to watch: it has not moved in this patch cycle and will not move until the `Arc<MctsNode>` heap scatter is pooled. When that number drops below 10 %, IPC on scenario A will cross 1.7 for the first time.

Further algorithmic wins beyond bumpalo+P1 (e.g., search-policy redesign, SIMD `check_win_at`, open-addressing TT) are no longer the right targets — they require semantic changes and are dominated by the structural gains still on the table.

---

### Appendix A. Raw artifact index (`tmp/profiles_20260425_postopt/`)

| File | Content |
|---|---|
| `A_perfstat.txt` … `D_perfstat.txt` | `perf stat -d -d -d` per scenario |
| `A_perf.data` | `perf record -F 997 -g --call-graph dwarf`, scenario A |
| `A_perf_report.txt` | `perf report --stdio --no-children` top-200 |
| `A_callgrind.out`, `A_callgrind.annotate.txt` | Callgrind Ir per function, threshold 100 |
| `A_cachegrind.out`, `A_cachegrind.annotate.txt` | Cachegrind Ir-only |
| `A_cachegrind_sim.out`, `A_cachegrind_sim.annotate.txt` | Cachegrind with `--cache-sim=yes` (I1/LLi/D1/LLd miss) |
| `A_heaptrack.zst`, `A_heaptrack.stdout` | Heaptrack raw + runtime summary |
| `hyperfine.txt`, `hyperfine.json` | 10-run wall-clock |
| `bloat_crates.txt`, `bloat_fns.txt` | `cargo bloat` by crate and function |
| `llvm_lines.txt` | `cargo llvm-lines` top-40 |
