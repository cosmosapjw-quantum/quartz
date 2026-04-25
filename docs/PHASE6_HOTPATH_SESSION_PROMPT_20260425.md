# Phase 6 — Hot-path locality + state-clone elimination

**Session-start prompt for the next session.**
This file is dual-purpose:
1. The **deep brief** for the next session that will tackle the IPC / dTLB / wall-clock targets that Phase 3 missed.
2. A **paste-into-fresh-session prompt** at the bottom (§9).

---

## 0. Where the codebase is right now

**HEAD:** `bc11f28  docs: Phase 3.3 — final perf delta + measurement audit`

The 2026-04-25 unified plan is fully landed:

| Phase | Status | Key commit(s) |
|---|---|---|
| 1. Python landings + audit docs | ✅ | `34e2dee`, `2617df7` |
| 2A. TT bucket capacity hint | ✅ | `e4d009b` |
| 2B. `#[inline]` win-checks | ✅ | `498a6fb` |
| 2C. `quartz_cache → parking_lot` | ✅ | `3608579` |
| 3.1. bumpalo arena landing | ✅ | `49f70f4` |
| 3.2. TSAN audit | ✅ | `0c3992e` |
| 3.3. final delta + measurement | ✅ | `bc11f28` |
| 4. Orchestrator hang | ✅ | `0425b42`, `ca75b34` |
| 5. Python lint cleanup | ✅ | `06e42ec` |

**Test baseline (must hold throughout this phase):**
- `cargo test --release --locked` → **390 passed**, 0 failed, 65 ignored
- `pytest tests/ -q --ignore=tests/test_play_gui.py` → **287 passed**, 6 skipped
- Orchestrator harness 10/10 in 0.76–1.70 s
- TSAN: full bin suite (390 tests), zero data races

**Headline scenario-A numbers (Gomoku-7 1T 15K iter, post-`bc11f28`, cool regime):**

| Metric | Pre-2026-04-25 | After Phase 3 | Phase 6 target |
|---|---|---|---|
| Heap allocations | 2.39 M | 125 K | unchanged or lower |
| Cycles | 3.11 G | 1.74 G | **< 1.4 G** |
| IPC | 0.96 | 1.22 | **> 1.7** |
| Wall-clock (mean / min) | 627 / — ms | 403 / 366 ms | **< 320 / < 280 ms** |
| dTLB miss rate | 31.4 % | 24.35 % | **< 10 %** |
| `TT::get_or_create` Ir share | — | 15.3 % | **< 8 %** |
| Total Ir | — | 1.66 G | < 1.3 G |

---

## 1. The dominant remaining cost (callgrind, 2026-04-25 post-Phase-3)

| Function | Ir % | Source of the cost |
|---|---:|---|
| `__memcpy_avx_unaligned_erms` | **28.13 %** | Almost entirely (~95 %) from `Gomoku::apply_move`'s board clone — `[i8; 361]` byte-copy per call |
| `MctsEngine::iterate` | 20.23 % | wrapper attribution; the work is in expand_and_evaluate / select |
| `Gomoku::check_win_at` | 18.11 % | Already `#[inline]` (Phase 2B); may be near-optimal |
| `TT::get_or_create` | 15.26 % | bucket-lock + HashMap probe per call (every PUCT visit hits this) |
| `Gomoku::apply_move` | 7.71 % (self) | clone + `check_win_at` recheck |
| `materialize_edges` | 4.14 % | Vec<MctsEdge> push / RwLock write |

The **headline insight from this audit** is:

> Phase 3 (bumpalo arena for `Arc<MctsNode>`) addressed an alloc-count
> problem that turned out to be a *secondary* cost. The dominant
> remaining cost is **state cloning in MCTS select**:
> `cur_state = cur_state.apply_move(best_mv)` clones the 361-byte
> Gomoku board on every PUCT step, ~10 steps per iteration ×
> 30 K iterations = ~4.4 M clones × 361 bytes = ~1.6 GB byte-copy.
> That's where the 28 % memcpy Ir and most of the dTLB pressure live.

This entirely re-orders the optimization priority for the next phase.

---

## 2. Pre-flight (6.0) — design candidates

The IPC / dTLB / TT bars Phase 3 missed correspond to three independent
leverage points. They should be tackled in **callgrind-priority order**:

### 6A. Make-unmake instead of clone (highest leverage — kills the 28 %)

Replace the cloning select pattern with apply-on-descend / unapply-on-ascend:

```rust
// Current select.rs:
cur_state = cur_state.apply_move(best_mv);   // clones 361-byte board

// Proposed:
let undo = cur_state.apply_move_in_place(best_mv);   // mutates, returns undo info
// ... descend ...
cur_state.undo_move(undo);                            // restores in place
```

This requires every `GameState` impl to grow a `Move::Undo` associated type
plus `apply_move_in_place` / `undo_move`. The MCTS hot path then carries
*one* mutable game state down each select traversal and rewinds it on
the way back up.

**Files affected:**
- `src/game.rs` (trait surface)
- `src/games/gomoku.rs`, `src/games/gomoku15.rs`, `src/games/go.rs`,
  `src/games/ttt.rs`, `src/games/chess.rs` (impls)
- `src/mcts/select.rs` (descent loop)
- `src/mcts/expand.rs` (`materialize_edges` still clones for child hash —
  decide whether that path can also avoid clone)
- Many test sites that construct `apply_move(..)` results

**Risk:** the `MctsEdge` materialization uses `state.apply_move(mv)` to
compute the child's TT hash. If that path also moves to in-place /
undo, every node's hash computation needs careful audit (the hash MUST
match what the child sees when it reaches that state via select).

**Alternative (cheaper but less win):** convert Gomoku's board from
`[i8; 361]` → `[u8; 49]` (Gomoku-7) or bitboard. With 49 bytes the
clone is 7× cheaper; with bitboard, ~16 bytes total. Still does the
clone, but it's much smaller.

### 6B. Lock-free hit path on `TT::get_or_create` (closes the 15 % → < 8 %)

Currently every `get_or_create` call acquires the bucket Mutex even when
the entry already exists (the hit case, ~80 % of calls in scenario A
once tree is built). Candidates:

- **Per-bucket `RwLock<TtBucket>` instead of `Mutex<TtBucket>`.** Hits take
  read lock (parallel with other readers). Misses upgrade to write. The
  bumpalo `Bump::alloc` requires `&mut Bump`, but only on the miss path,
  so the upgrade is rare and acceptable.
- **Versioned snapshot probe.** Add an atomic `version: AtomicU64` per
  bucket. Read version, probe map without lock (UB unsafe — needs
  hazard pointers / RCU). Probably too complex for the win.
- **`dashmap` / `scc` / `flurry`.** Replace the per-bucket Mutex+HashMap
  with an off-the-shelf concurrent hashmap. Harder to keep the bumpalo
  storage tied per-bucket; would need to redesign the arena layout.

**Recommended:** start with the RwLock swap. Smallest blast radius, audit
shows hits dominate, the upgrade path is only on miss.

### 6C. Edge buffer locality (chips into dTLB and IPC)

Per the Phase 3 follow-up §6, `MctsNode.edges: RwLock<Vec<MctsEdge<M>>>`
allocates the Vec backing buffer on the global heap — separate from the
bumpalo-allocated MctsNode body. Every PUCT visit dereferences:
`MctsNode (bumpalo) → Vec<MctsEdge> backing buffer (global heap) → child`.

Candidates:
- **smallvec inline-or-spill** with N=4 inline. ~80 % of nodes in scenario A
  have ≤ 4 materialized edges; those land in the node body, contiguous.
  Overflow uses heap (acceptable; rare).
- **Pre-size at peak.** At first materialization, allocate
  `Box<[MctsEdge; n_candidates]>` (or in a per-engine bumpalo edge-arena
  with a per-engine Mutex<Bump>). Tracks growth via `edge_cursor`.
  Wastes memory on unvisited subtrees, but PW caps materialization count
  so peak is rarely reached.
- **Restructure storage so edges live in the same Bump as their parent
  node.** Requires either lifetime threading (the option the Phase 3.0
  brief rejected as too invasive) or a separate per-bucket "edge Bump"
  with its own discipline.

**Recommended:** smallvec is the lowest-risk win. Profile first with
N=4 and N=8 to see which configuration is most cache-friendly.

---

## 3. Per-checkpoint plan

Each checkpoint = its own commit + its own audit + measurement block.
Per the standing per-step protocol (memory:
`feedback_per_step_audits.md`), do not batch.

### 6.1 — Make-unmake for Gomoku (the highest-leverage move)

Scoped to **Gomoku only** for the first commit, since Gomoku-7 is
scenario A's game and where the measurement bar lives. Other games can
follow as separate commits.

1. Add `apply_move_in_place(&mut self, mv) -> Self::Undo` and
   `undo_move(&mut self, undo: Self::Undo)` to `GameState`. Default-impl
   them on top of `apply_move` + `clone()` so other games don't break
   immediately.
2. Implement on-Gomoku in-place + undo. Undo type: `(prev_to_move, last_move_pos)`.
   `apply_move_in_place` writes the cell, flips `to_move`, runs
   `check_win_at` to record terminal status. `undo_move` restores.
3. Convert `select.rs` descent to use the in-place pattern.
4. Convert `expand.rs::materialize_edges`. The child-hash path needs
   apply_in_place + undo too.
5. `cargo test --release --locked` → all 390 must still pass; PV
   equivalence (`bench_search_controller_fixed_iterations_fast_path`)
   must remain.
6. Measure on scenario A: callgrind `__memcpy_avx_unaligned_erms` Ir share
   should drop from 28 % to < 5 %.

### 6.2 — Lock-free hit path on TT (`Mutex` → `RwLock<TtBucket>`)

1. `src/mcts/tt.rs`: swap `Mutex<TtBucket>` → `parking_lot::RwLock<TtBucket>`.
2. `get_or_create`: read-lock; if hit, return. If miss, drop read lock
   and re-acquire write lock; re-check then alloc.
3. `get`: read-lock, return.
4. Bench scenario A; expect `TT::get_or_create` Ir share to drop from
   ~15 % to < 8 %.
5. TSAN re-run (RwLock has different annotations than Mutex).

### 6.3 — Edge buffer inlining (smallvec)

1. Add `smallvec = "1"` (or `tinyvec`) to `Cargo.toml`.
2. `src/mcts/node.rs`: `edges: RwLock<SmallVec<[MctsEdge<M>; 4]>>`.
3. Profile two configurations: N=4 and N=8.
4. Bench scenario A; expect dTLB miss rate to drop from 24 % to < 12 %
   (all edges of typical nodes land in same cache line as node body).

### 6.4 — Final validation + commit

1. Full suites green: cargo 390, pytest 287, orchestrator harness 10/10,
   TSAN clean.
2. **Numerical bars (scenario A, hyperfine --warmup 5 --runs 15):**
   - wall-clock mean **< 320 ms**
   - wall-clock min **< 280 ms**
   - IPC **> 1.7**
   - cycles **< 1.4 G**
   - dTLB miss rate **< 10 %**
   - `TT::get_or_create` < 8 % Ir
   - `Gomoku::apply_move`-induced memcpy < 5 % Ir
3. Write `docs/MCTS_PROFILE_DELTA_PHASE6_<date>.md` in the same style
   as the existing delta docs.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| `apply_move_in_place` correctness — Gomoku has Zobrist hashing that incrementally XORs. The undo path must un-XOR cleanly. | Add a property test: random sequences of (apply_in_place, undo) leave board state and hash identical to the original. Run before changing select.rs. |
| Other game impls regressing if the trait gains a required method | Default-impl `apply_move_in_place` on top of `apply_move` + `clone` so non-migrated games work via fallback. Migrate Gomoku first; other games are post-bar. |
| MctsEdge materialization needs the child's TT hash, computed via `state.apply_move(mv)` | Switch that path too: `state.apply_move_in_place(mv); let h = state.tt_hash(); state.undo_move(undo);`. The `tt_hash` is read-only, so this is a clean apply/undo bracket. |
| RwLock<TtBucket> false sharing with Bump | Bumpalo's Bump internal state is in the bucket struct; under write lock, the writer has exclusive access. RwLock's reader path doesn't touch the Bump (read-only map probe). |
| smallvec inline storage may bloat MctsNode size enough to cause cache-line crossings | Profile sizeof::<MctsNode<u8>> before and after. Target is to keep MctsNode body ≤ 1 cache line (64 B) for the *fixed* part, with the inline edge buffer being the second cache line at most. If size > 128 B, drop inline N to keep it tight. |
| Phase 3 invariant: `TtBucket::Drop` runs `drop_in_place` per node | smallvec inline storage drops cleanly via standard Drop, so this still works. RwLock<TtBucket> has the same field topology, so Drop semantics are unchanged. |
| TSAN false positives on parking_lot::RwLock | Same suppression strategy as Phase 3.2 — at worst, suppress only `parking_lot::*` frames, never application frames. Phase 3.2 had zero false positives, so this is a low risk. |
| Thermal regime variance (5900X) | `--warmup 5 --runs 15` minimum. perf-stat counters are thermal-immune; trust them more than wall-clock for small claims. Reference: `memory/reference_thermal_regime.md`. |

---

## 5. Profiling commands (canonical)

```sh
# Build the test binary
cargo test --release --locked --no-run

# Pick the newest mcts_demo test binary
TB=$(ls -t target/release/deps/mcts_demo-*[0-9a-f] | grep -v '\.d$' | grep -v 'long-type' | head -1)
echo $TB

# Hyperfine (use --warmup 5 --runs 15 minimum on this host)
hyperfine --warmup 5 --runs 15 --export-json /tmp/phase6_after.json \
  "$TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path"

# perf stat (thermal-immune intrinsics)
perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses -r 5 \
  $TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path

# heaptrack (allocation count is the headline number for Phase 3)
heaptrack -o /tmp/phase6_after.heaptrack \
  $TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path
heaptrack_print /tmp/phase6_after.heaptrack.zst | grep -E "^total|^peak|^calls" | head -5

# Callgrind with cache simulation (absolute truth)
valgrind --tool=callgrind --callgrind-out-file=/tmp/phase6_after.callgrind \
  $TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path
callgrind_annotate /tmp/phase6_after.callgrind | head -50
callgrind_annotate --tree=calling /tmp/phase6_after.callgrind | head -100  # caller→callee tree

# TSAN (nightly)
RUSTFLAGS='-Z sanitizer=thread' cargo +nightly test --release \
  --target x86_64-unknown-linux-gnu --tests -Z build-std --no-run
TSAN_TB=$(ls -t target/x86_64-unknown-linux-gnu/release/deps/mcts_demo-*[0-9a-f] | grep -v '\.d$' | grep -v 'long-type' | head -1)
"$TSAN_TB"   # full bin suite under TSAN
```

Pre-Phase-6 artifacts: `tmp/profiles_20260425_postopt/` (pre-Phase 3),
`/tmp/phase3_after.{callgrind,heaptrack.zst}` (post-Phase 3).

---

## 6. Hardware / tooling

- AMD Ryzen 9 5900X · 64 GiB · Linux 6.17 / glibc 2.39
- Profilers: perf, samply, heaptrack, valgrind (callgrind/cachegrind/massif),
  hyperfine, flamegraph, cargo-flamegraph, cargo-pgo, cargo-bloat,
  cargo-llvm-lines
- Thermal regimes: ~370 ms cool / ~470 ms hot for scenario A post-Phase-3.
  See `memory/reference_thermal_regime.md`.
- ptrace_scope = 1 (or 0 if previously relaxed). When debugging
  multi-process hangs, ask the user to run `sudo gdb -p $PID`
  themselves — see `memory/feedback_gdb_via_user.md`.

---

## 7. Files this plan does NOT touch

- `quartz/jax_training_runtime.py` — backend facade, not in MCTS hot loop.
- `quartz/onnx_support.py`, `quartz/gomocup_export.py` — deployment paths.
- `src/calibration.rs`, `src/ablation_*.rs` — test/experiment paths only.
- The Phase 3 `ArenaRef` / `TtBucket::Drop` design (Phase 3.1) is solid;
  do not revisit unless a TSAN / safety regression points back to it.
- Phase 6's edge-inlining work (6.3) interacts with `TtBucket::Drop` —
  smallvec drops cleanly, so the existing Drop impl needs no changes.

---

## 8. Acceptance checklist (paste into the final commit message)

```
Semantic audit
  cargo test --release --locked            390 passed, 0 failed
  pytest tests/ -q                         287 passed, 6 skipped
  orchestrator harness 10×                 10 / 10 complete
  v5_stress_parallel 10×                   10 / 10 no flakes
  TSAN on full bin suite (390 tests)       zero data races
  apply_in_place / undo_move property test passes (random sequences,
                                            board + tt_hash equivalence)

Measurement (scenario A: Gomoku-7 1T 15K iter,
  hyperfine --warmup 5 --runs 15, perf stat -r 5)
  pre  (bc11f28)   wall ~403 ms, IPC 1.22, allocs 125 K, dTLB 24.3 %,
                   memcpy(apply_move) 26.9 % Ir, get_or_create 15.3 % Ir
  post             wall <X> ms,  IPC <Y>,  allocs <Z>, dTLB <W> %,
                   memcpy <M> %, get_or_create <T> %

Bars hit:
  - wall mean < 320 ms                     ☐
  - wall min < 280 ms                      ☐
  - IPC > 1.7                              ☐
  - cycles < 1.4 G                         ☐
  - dTLB miss rate < 10 %                  ☐
  - get_or_create < 8 % Ir                 ☐
  - apply_move-induced memcpy < 5 % Ir     ☐
```

---

## 9. Session-start prompt — paste into a fresh session

```
You are continuing optimization work on the QUARTZ AlphaZero codebase
at /home/cosmosapjw/Dropbox/personal_projects/quartz. HEAD is bc11f28.

The 2026-04-25 unified plan (Phases 1-5 plus Phase 3.1/3.2/3.3) is
fully landed and green:
  cargo 390 / pytest 287 / orchestrator 10/10 / TSAN clean
  scenario A wall-clock 403 ms mean / 366 ms min, IPC 1.22, dTLB 24.3 %.

Phase 3 hit the headline alloc-count bar (886 K → 125 K) cleanly, but
missed three strict targets: IPC > 1.7, dTLB < 10 %, get_or_create
< 8 % Ir. The Phase 3 delta doc identified these as architectural
gaps in the ArenaRef-wrapper design (per-node Vec<MctsEdge> still on
the global heap).

A post-Phase-3 callgrind audit then surfaced a much larger leverage
point: ~28 % of total Ir is in __memcpy_avx_unaligned_erms, ~95 %
of which traces back to Gomoku::apply_move's [i8; 361] board clone.
The MCTS select loop clones the board ~10 times per iteration; over
30 K iterations that's ~4.4 M clones × 361 bytes ≈ 1.6 GB of byte-copy
work. Phase 3's alloc-elimination was secondary cost; THIS is the
dominant remaining bottleneck.

Read in order before starting:
  1. docs/PHASE6_HOTPATH_SESSION_PROMPT_20260425.md   ← THIS prompt
                                                       expanded form
  2. docs/MCTS_PROFILE_DELTA_PHASE3_20260425.md       ← Phase 3 outcome,
                                                       § 6 follow-ups
  3. docs/MCTS_PROFILE_AUDIT_20260425_POSTOPT.md      ← latest deep audit
  4. docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md       ← parent plan, § 6
                                                       success-bar table

Three checkpoints in callgrind-priority order — each its own commit:

  6.1  Make-unmake for Gomoku (kills the 28 % memcpy)
       Add `apply_move_in_place` / `undo_move` to GameState; implement
       on Gomoku first; convert select.rs descent and
       materialize_edges to use it. Target: memcpy(apply_move) < 5 %
       Ir, wall-clock mean < 350 ms.

  6.2  TT::get_or_create lock-free hit path
       Mutex<TtBucket> → parking_lot::RwLock<TtBucket>. Hits take read
       lock; misses upgrade to write. Target: get_or_create < 8 % Ir,
       TSAN clean.

  6.3  Edge buffer inlining (smallvec N=4 or N=8)
       MctsNode.edges from RwLock<Vec> to RwLock<SmallVec<[_; N]>>.
       Target: dTLB < 10 %, IPC > 1.7.

  6.4  Final validation + measurement + delta doc

Per-step protocol (mandatory; see memory/feedback_per_step_audits.md):
  1. Read all affected files fully before editing.
  2. Apply patch.
  3. Semantic audit:
     - cargo test --release --locked        target 390 pass
     - pytest tests/ -q                     target 287 pass
     - parallel: v5_stress_parallel 10×, no flakes
     - TSAN (6.2 in particular): zero data races
     - 6.1: property test on apply_in_place / undo_move equivalence
  4. Measure:
     - hyperfine --warmup 5 --runs 15 (5900X needs this for stable wall)
     - perf stat -r 5 for cycles/IPC/dTLB
     - callgrind --tree=calling for per-function Ir attribution
  5. Commit with HEREDOC. Include semantic-audit lines and
     measurement block. Do NOT batch checkpoints.

The 5900X has two stable thermal regimes (~370 ms cool / ~470 ms hot
for scenario A post-Phase-3). Always use --warmup 5 --runs 15.
perf-stat counters are thermal-immune; trust them more than wall-clock
for small claims. Reference: memory/reference_thermal_regime.md.

If you hit a multi-process hang during testing and gdb refuses to
attach (ptrace_scope = 1), ask the user to run sudo gdb -p $PID
themselves — see memory/feedback_gdb_via_user.md for the exact
recipe. Don't burn time on attach workarounds.

Authorized to execute autonomously. Phase 6's checkpoint discipline
is non-negotiable; the user has explicitly asked for it. Auto mode
does NOT mean skip checkpoints or batch commits.

Strict numerical bars for the final commit (scenario A):
  - wall-clock mean < 320 ms, min < 280 ms
  - IPC > 1.7
  - cycles < 1.4 G
  - dTLB miss rate < 10 %
  - TT::get_or_create < 8 % Ir
  - apply_move-induced memcpy < 5 % Ir

Phase 3's design (ArenaRef wrapper, TtBucket::Drop discipline,
per-bucket bumpalo Bump) is solid and should not be revisited unless
TSAN or test failures point back to it. Phase 6 builds on top.
```
