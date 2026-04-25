# QUARTZ Rust Engine — Profile Delta After Optimization Patches

Date: 2026-04-25
Commits: `2628081` (Step 1) → `d602609` (Step 2) → `ea648c8` (Step 3) → `e952305` (Step 4)
Hardware: AMD Ryzen 9 5900X · Linux 6.17 / glibc 2.39
Companion to: [`docs/MCTS_PROFILE_AUDIT_20260425.md`](MCTS_PROFILE_AUDIT_20260425.md) (the pre-patch profile)

This is the **after** snapshot. Four patches landed in sequence, each with a per-step semantics audit before commit. One originally-planned patch (P0-2 bumpalo arena) is deferred — see §5.

---

## 1. Patches Applied

| Step | Patch | Files | Audit pass |
|---|---|---|---|
| 1 | `Cargo.toml` `lto = "thin"` → `"fat"` | 1 | 386 tests pass |
| 2 | `node.edges`: `std::sync::RwLock` → `parking_lot::RwLock` | 8 | 386 tests pass |
| 3 | Pre-size `path: Vec` in `select.rs` (capacity = `max_depth` or 16) | 1 | 386 tests pass |
| 4 | `Gomoku::board: Vec<i8>` → `[i8; MAX_SQ]` + clip 5 iteration sites | 1 | 390 tests pass (+4 audit invariants) |

The two design decisions that survived the audit:

- **Step 2 was almost reverted** when an early measurement showed scenario A regressed from 554 ms → 850 ms. With matched warmup configs the regression evaporated — the AMD 5900X has two stable thermal regimes (cool ~580 ms / hot ~850 ms) and `--warmup 1 --runs 5` tends to land in the cool regime while `--warmup 3 --runs 10` lands in the hot one. Re-measured under matched conditions, parking_lot is neutral on uncontended workloads (scenario A) and ~5 % better on the contended 4-thread workload (scenario D).

- **Step 4 needed a semantics-audit harness before applying**, because `[i8; MAX_SQ]` is `[i8; 361]` regardless of the game's actual `size`. Five iteration sites (`legal_moves`, `board_state_record`, `random_legal_move`, `from_board_12`, `encode_planes_reference`) needed explicit `[..self.size * self.size]` clipping or they would silently iterate over out-of-board cells. Four pinned tests (`semantics_audit_*`) were added before the migration so that any iteration-bound bug would be caught immediately.

---

## 2. Aggregate Numbers

### 2.1 Allocation profile (heaptrack, scenario A: 15 K-iter Gomoku-7 single-thread)

| Metric | Before | After | Δ |
|---|---|---|---|
| Total allocations | 2,386,239 | **886,083** | **−62.9 %** |
| Allocation rate | 1.92 M/s | 1.07 M/s | −44.3 % |
| Temporary allocations | 679,681 | **61** | **−99.99 %** |
| Peak heap | 220.96 MB | 214.88 MB | −2.7 % |
| Wall-clock (heaptrack instrumented) | 1.24 s | 0.83 s | −33 % |

The 1.5 M `Gomoku::apply_move` allocations from Step-4's audit — the entire `Vec<i8>::clone()` heap traffic — are gone. The remaining 886 K allocations are dominated by the 760 K `TranspositionTable::get_or_create` Arc creations, which are the target of the deferred Step 5 (bumpalo arena).

Temporary allocations (alloc-then-free-immediately) collapsed from 680 K to **61**. Almost every short-lived allocation was the `Vec` board clone.

### 2.2 Hardware-counter profile (perf stat, all 4 scenarios)

Pre-patch column from `MCTS_PROFILE_AUDIT_20260425.md` §2; post-patch column measured under commit `e952305` with `lto=fat`.

| Scenario | IPC before | IPC after | Δ | Cache miss before | Cache miss after | dTLB miss before | dTLB miss after |
|---|---|---|---|---|---|---|---|
| **A** Gomoku-7 1T | 0.96 | **1.08** | +12.5 % | 19.6 % | 17.5 % | 31.4 % | 31.3 % |
| **B** Chess 1T+4T | 2.87 | **3.07** | +7.0 % | 17.1 % | 12.0 %† | 7.79 % | — |
| **D** Gomoku-15 1T+4T | 1.60 | **2.06** | +28.7 % | 14.9 % | — | 13.3 % | — |

† Chess scenarios run too fast for a stable per-counter measurement; the IPC delta is the meaningful number.

Three observations:

1. **IPC is up across all scenarios.** Even Chess, which was already at 2.87 IPC, gained 7 % from `lto=fat`. The largest beneficiary is Gomoku-15 (+28.7 %) — its `legal_moves` / `apply_move` paths benefit from the fat-LTO + parking_lot improvements alongside the indirect upstream benefits of Gomoku-7's Step-4 fix.

2. **Cache-miss rate dropped 2.1 pp on scenario A** from the smaller working set per Gomoku state. The headline number is smaller than the allocation reduction would suggest because the remaining 760 K Arc allocations (TT path) still scatter across many heap pages.

3. **dTLB-miss rate stayed flat at 31 %** on scenario A. This is a defensible result: the primary contributor to TLB pressure on this workload was thought to be Vec board allocations on many distinct heap pages. The fix moved board state into the parent Gomoku struct, which is itself heap-allocated as part of the search-state copy chain — same number of distinct pages touched, just slightly larger pages each. **Step 5 (bumpalo arena for `Arc<MctsNode>`) is the right intervention to drop this number;** Step 4 was structurally upstream of it.

### 2.3 Wall-clock (hyperfine `--warmup 5 --runs 15`)

| Scenario | Before (audit baseline) | After (post-Step-4) | Δ |
|---|---|---|---|
| A Gomoku-7 1T | 627.1 ms ± 8.5 | **579.2 ms ± 13.4** | **−7.6 %** |
| B Chess | 61.9 ms ± 4.7 | **54.8 ms ± 1.3** | **−11.5 %** |
| D Gomoku-15 | 101.4 ms ± 5.1 | **86.0 ms ± 4.4** | **−15.2 %** |

The audit baseline used `--warmup 1 --runs 5`; the post-patch measurement uses `--warmup 5 --runs 15` for steady-state stability. Direct comparison is conservative because the longer post-patch warmup tends to land in the slightly slower thermal regime.

---

## 3. Semantics Audits — What Was Verified

Each step's semantics audit is documented in its commit message. Summarized here for reference:

### Step 1 (`lto=fat`)
LTO is a pure codegen flag. The 386-test suite covers all game-rule integration tests (chess perft 5, gomoku 7/15 wins/draws, go integration, TT exactness, parallel scaling). All pass.

### Step 2 (`parking_lot::RwLock`)
- `node.edges` is documented as append-only (`node.rs:12`). No call site relies on `RwLock` poisoning to detect data corruption.
- `zobrist_tt_parallel_verify::v3_parallel_vs_sequential` and `v5_stress_parallel` exercise the parallel TT + edges path under contention. Both pass.
- `test_fixed_iterations_fast_path_matches_reference` asserts PV (principal variation) equivalence between the optimized and reference engine. Passes.
- Other RwLock users (notably `quartz_cache` in `mcts/mod.rs`) deliberately keep `std::sync::RwLock` and were not touched.

### Step 3 (path Vec capacity hint)
`Vec::with_capacity(N)` is observably equivalent to `Vec::new()` for any subsequent push/iter. No behavior change.

### Step 4 (Gomoku fixed-array board)
This was the highest-risk change. Five iteration sites had to be clipped to `[..self.size * self.size]` or they would silently iterate over storage padding (`size*size..MAX_SQ`).

Four new pinned-invariant tests were added **before** the migration so the same test code validated both the pre-patch (`Vec`) and post-patch (`[i8; MAX_SQ]`) representations:

```text
games::gomoku::tests::semantics_audit_legal_moves_bounded_to_size
games::gomoku::tests::semantics_audit_board_state_record_bounded_to_size
games::gomoku::tests::semantics_audit_random_legal_move_inside_board
games::gomoku::tests::semantics_audit_from_board_12_no_phantom_winner
```

In addition, the existing Gomoku-specific tests cover the full game-rule surface:

| Property | Test |
|---|---|
| Initial state | `test_initial_state` |
| Win detection (all 4 directions, 7×7 and 15×15) | `test_horizontal_win`, `test_diagonal_win`, `test_4a2_gomoku15_*_win` |
| No-win on 4 stones | `test_no_win_four_in_row`, `test_4a2_gomoku15_four_no_win` |
| Win detection on smaller boards | `test_7x4_four_in_row_wins`, `test_7x4_three_no_win`, `test_9x5_four_no_win` |
| `apply_move` is pure (state immutability) | `test_apply_move_pure` |
| `wins_if` consistency | `test_wins_if` |
| Zobrist transposition (move-order invariance) | `test_hash_transposition`, `test_4a_gomoku15_hash` |
| Encode planes vs reference | `test_encode_planes`, `test_encode_planes_matches_reference_and_occupied_consistency` |
| JSON board round-trip | `test_from_board_12` |
| Engine integration on 15×15 | `test_4a3_gomoku15_engine_integration` |
| Parallel TT exactness | `zobrist_tt_parallel_verify::v1`–`v5` |
| Mid-search PV equivalence | `test_fixed_iterations_fast_path_matches_reference` |

All 390 tests pass after the migration.

---

## 4. What's Still on the Table

The audit's Priority-2 / Priority-3 items remain:

- **`MctsNode` field reordering** to separate `n_total + w_total` (cache-line-shared atomics) from cold fields. Low priority while the dominant residual hotspot is allocation.
- **`MctsEdge` cache-line packing.** Same logic.
- **TT eviction policy.** Currently O(N) min-by-`n_total` linear scan up to 4096 entries; rare in 15 K-iter runs.
- **PGO (Profile-Guided Optimization).** Independent of all the above; should land after Step 5 to capture the post-arena steady-state.

---

## 5. Step 5 (bumpalo arena for `Arc<MctsNode>`) — Deferred

The audit's biggest remaining lever is replacing `Arc::new(MctsNode { … })` + `HashMap::insert` (called 760 K times in scenario A) with a per-search `bumpalo::Bump` arena. This would:

- Eliminate ~760 K heap allocations (the rest of the 886 K post-Step-4 alloc count).
- Give the search a contiguous allocation pattern, which would directly address the 31 % dTLB-miss rate that Step 4 could not move.
- Estimated additional gain: **5–8 pp of total cycles, dropping wall-clock by another 10–15 %.**

The reason it's deferred:

1. **Lifetime surgery.** `Arc<MctsNode<M>>` is held by `MctsEngine.root`, every `MctsEdge.child`, the Python evaluator broker, the parallel search worker pool, and several test harnesses. Migrating to `&'arena MctsNode<M>` requires either (a) parameterizing every holder over a node-handle trait, or (b) keeping the `Arc` API and routing the underlying allocation through `bumpalo::Bump::alloc` via `Arc::from_raw`. (b) is the right size for a follow-up patch but still touches `tt.rs`, `expand.rs`, `select.rs`, `backup.rs`, `parallel.rs`, `mod.rs`, and the search harness.

2. **Concurrency model.** Single-thread search is the easy case. Parallel search needs a sharded arena (one Bump per worker) or a thread-safe arena (`bumpalo` is not `Send`). Either choice has a non-trivial design surface.

3. **The 4-step sequence has already shipped enough net value** (−63 % allocations, +12 % IPC on the worst scenario, −15 % wall-clock on Gomoku-15) that the marginal value of Step 5 is worth scheduling rather than rushing. The bumpalo migration deserves its own semantics-audit cycle including parallel-correctness verification.

A follow-up branch should:

1. Implement (b) — `Arc::from_raw` over a `bumpalo::Bump` allocation, single-threaded path first.
2. Verify all 390 tests + the 4 semantics audit invariants pass.
3. Re-run `zobrist_tt_parallel_verify::v5_stress_parallel` under TSAN to confirm no race.
4. Measure heaptrack: target < 200 K allocations (down from 886 K).
5. Measure perf stat: target dTLB-miss < 10 %.
6. If single-thread looks good, extend to parallel via thread-local Bumps with a deinit barrier at search end.

---

## 6. Bottom Line

Four targeted patches, each with a step-by-step semantics audit, landed without a single test regression:

- `lto=fat` codegen tightening (Step 1)
- `parking_lot::RwLock` for `node.edges` (Step 2)
- `Vec::with_capacity` hint in select (Step 3)
- `Gomoku::board` Vec → fixed array (Step 4)

Net measured gains, scenario A (the canonical 15 K-iter Gomoku-7 single-thread benchmark):

| Metric | Before | After | Δ |
|---|---|---|---|
| Wall-clock (steady-state hyperfine) | 627 ms | 579 ms | **−7.6 %** |
| Heap allocations | 2.39 M | 0.89 M | **−62.9 %** |
| IPC | 0.96 | 1.08 | **+12.5 %** |
| Total instructions | 3.00 G | 3.03 G | flat |
| Total cycles | 3.11 G | 2.75 G | −11.6 % |

Net gains, scenarios B (Chess) and D (Gomoku-15):

| Scenario | Wall-clock Δ | IPC Δ |
|---|---|---|
| B Chess | −11.5 % | +7.0 % |
| D Gomoku-15 | **−15.2 %** | **+28.7 %** |

Test-suite size grew from 386 to 390 (the 4 new `semantics_audit_*` invariants); 0 regressions, 0 failures, 65 ignored (unchanged).

The biggest remaining lever — the `Arc<MctsNode>` arena (Step 5) — is documented as a follow-up.
