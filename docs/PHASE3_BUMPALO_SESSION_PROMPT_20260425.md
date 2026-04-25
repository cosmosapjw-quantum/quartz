# Phase 3 — bumpalo Arena Session-Start Prompt
## (continuation of `docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md`)

This file is dual-purpose:
1. The **deep brief** for the next session that will tackle Phase 3.
2. A **paste-into-fresh-session prompt** at the bottom (§9).

---

## 0. Where the codebase is right now

**HEAD:** `ca75b34  mcts_server: route eval through broker whenever SHM ring is live`

Everything from the 2026-04-25 unified plan is landed and green
EXCEPT Phase 3:

| Phase | Status | Key commit(s) |
|---|---|---|
| 1. Python landings + audit docs | ✅ | `34e2dee`, `2617df7` |
| 2A. TT bucket capacity hint | ✅ | `e4d009b` |
| 2B. `#[inline]` win-checks | ✅ | `498a6fb` |
| 2C. `quartz_cache → parking_lot` | ✅ | `3608579` |
| **3. bumpalo arena** | **⏸ this session** | — |
| 4. Orchestrator hang | ✅ both deadlocks fixed | `0425b42`, `ca75b34` |
| 5. Python lint cleanup | ✅ | `06e42ec` |

**Test baseline (must hold throughout this phase):**
- `cargo test --release --locked` → **390 passed**, 0 failed, 65 ignored
- `pytest tests/ -q --ignore=tests/test_play_gui.py` → **287 passed**, 6 skipped
- Orchestrator harness 10/10 in 0.95–1.62 s
  (`tmp/profile_python_orchestrator.py --iters 8 --n-games 2 --parallel 2 --batch-size 2`)

**Headline scenario-A numbers (Gomoku-7 1T 15K iter, post-`ca75b34`):**

| Metric | Pre-2026-04-25 | Now | Phase 3 target |
|---|---|---|---|
| Heap allocations | 2.39 M | 886 K | **< 200 K** |
| Cycles | 3.11 G | 2.72 G | **< 1.8 G** |
| IPC | 0.96 | 1.13 | **> 1.7** |
| Wall-clock | 627 ms | 558 ms | **< 380 ms** |
| dTLB miss rate | 31.4 % | 31.6 % | **< 10 %** |

The 31.6 % dTLB miss rate is the dominant remaining bottleneck and
the entire reason Phase 3 exists. 760 K `Arc<MctsNode>` allocations
account for 86 % of the remaining heap and 51 % of all D1 read-misses.

---

## 1. Pre-flight (3.0) — design decision is already made

The 2026-04-25 audit recommended starting with **option (a)** —
"preserve `Arc` API, allocate `MctsNode` body in a per-search
`bumpalo::Bump`, wrap as `Arc<MctsNode>` via `Arc::from_raw`." After
careful analysis during the 2026-04-25 session, **option (a) is not
implementable on stable Rust:**

- `Arc::from_raw` requires a pointer that came from `Arc::into_raw`
  (i.e. a previous `Arc::new` allocation or `Arc::new_in` with
  `allocator_api`). You cannot fabricate an `ArcInner<T>` in arena
  memory and hand it to `Arc::from_raw`.
- When the strong count drops to 0, `Arc::drop_slow` unconditionally
  frees the `ArcInner` via the **global allocator**. If that pointer
  is in arena memory, it corrupts the arena.
- `Arc::new_in` (which would actually let you do this) is gated
  behind `#![feature(allocator_api)]` on nightly. We're on stable.

**The path forward is option (b):** replace `Arc<MctsNode>` with
`&'arena MctsNode<'arena, M>`. The arena owns nodes for the engine's
lifetime; refcounting is unnecessary because the TT holds master
ownership and everything else borrows.

**Recommended primitive:** `bumpalo::Bump` per TT bucket. Rationale:
- `Bump` is `!Sync`. We need allocations to be serialized somehow.
- The TT already has 256 per-bucket `Mutex<TtBucket>`. Putting a
  `Bump` in each bucket means allocations are serialized through
  the **existing** lock — no new contention point.
- 256 bumpalo arenas × ~1 MiB each = ~256 MiB peak overhead. That
  is comparable to the current 215 MiB peak; not a memory blocker
  on 64 GiB host.

`typed-arena` is a viable second-choice (one global arena, simpler
API) but its `alloc(&self, T) -> &mut T` is also `!Sync`. Putting it
behind `Mutex<Arena>` would add a new contention point that the
per-bucket `Bump` design avoids. Stick with per-bucket `bumpalo`.

**Lifetime topology decision:**
- The arena CANNOT live inside `MctsEngine` — that would be a
  self-referential struct, which Rust forbids without `Pin` + unsafe.
- The arena lives **outside** the engine. Each caller of
  `MctsEngine::new` allocates a `Bump` (or a `[Bump; 256]` for the
  per-bucket layout) on its own stack/scope and passes
  `&'arena ...` to the engine.

---

## 2. File inventory — every site that needs surgery

`grep -rn "Arc<MctsNode\|Arc::clone\|Arc::ptr_eq" src/` enumerates 21
sites across 11 production files plus the test harnesses in
`src/mcts/mod.rs:1175+` and `src/main.rs`.

| File | Lines | Type of change |
|---|---|---|
| `src/mcts/node.rs` | 98, 111, 191, 230, 248, 268-281, 343-388 | `MctsNode<'arena, M>`, `MctsEdge<'arena, M>`, `MctsEdgeSnapshot<'arena, M>`, `PathEdge<'arena, M>`. The `Arc::clone(&edge.child)` at :355 becomes a copy of the `&'arena` reference. |
| `src/mcts/tt.rs` | 14-19, 36-72, 98-198, 233-263 | Per-bucket `Bump`. `TranspositionTable<'arena, M>`. `get_or_create` returns `&'arena MctsNode<'arena, M>` (via the bucket's Bump). `tests` mod assertions: `Arc::ptr_eq` becomes `std::ptr::eq`. |
| `src/mcts/select.rs` | 441, 607, 628, 728, 733 | `SelectResult<'arena, G>.leaf: &'arena MctsNode<...>`. `select` takes `&'arena Bump` parameter (or threaded through TT). |
| `src/mcts/expand.rs` | 26, 41, 88 | `expand_and_evaluate(node: &'arena MctsNode<...>, ...)`. |
| `src/mcts/backup.rs` | (entire file) | `backprop` walks `&[PathEdge<'arena, M>]`. |
| `src/mcts/parallel.rs` | (worker entry points) | The thread-pool's `spawn` closures need to capture `&'arena` borrows. `crossbeam::scope` is the canonical fix. |
| `src/mcts/mod.rs` | 23-24, 188-252, 296-407, 580-695, 815-841, 866-891, 1175+ | `MctsEngine<'arena, G>`. `new(arena: &'arena Bump, ...)`. `advance_root`/`replace_root_state` need the same arena (or a fresh one). All test setups under `#[cfg(test)] mod tests` need `let arena = Bump::new();`. |
| `src/mcts/root.rs` | 59, 93 | Simple lifetime threading on the helpers. |
| `src/mcts/gvoc.rs` | 96 | Single signature change. |
| `src/mcts/search.rs` | 194 | Simple lifetime threading. |
| `src/mcts/quartz.rs` | 798, 1695 | Two signature changes. |
| `src/mcts_server.rs` | many | Every handler that constructs `MctsEngine` (`handle_search_nn`, `handle_search_nn_multi`, `handle_selfplay_nn_run_generic`, `handle_eval_nn_run`, session-step handlers) needs `let arena = Bump::new();` before the engine. |
| `src/main.rs` | acceptance tests | Same pattern: arena per `MctsEngine::new` call. |
| `src/gomocup_brain.rs` | engine ownership | Has a long-lived `MctsEngine`; needs an owned `Bump` field at the same lifetime. |

`MctsEdge.child: Arc<MctsNode<M>>` becoming `child: &'arena MctsNode<'arena, M>` is the single most invasive type change — it ripples to every snapshot/path/edge type.

**Cargo.toml:** add `bumpalo = "3"`. (Already in the workspace? check; if not, add to `[dependencies]`.)

---

## 3. Per-checkpoint plan

Each checkpoint = its own commit + its own audit + measurement
block. Per the standing per-step protocol (memory:
`feedback_per_step_audits.md`), do not batch.

### 3.1 — single-thread (target: ~370 of 390 tests pass)

1. Add `bumpalo = "3"` to `Cargo.toml`.
2. Lift the `'arena` lifetime through `node.rs`, `tt.rs`, `select.rs`,
   `expand.rs`, `backup.rs`, `mod.rs` only. Leave `parallel.rs`
   broken at this checkpoint (it's the next checkpoint's job).
3. Provide `MctsEngine::new(arena: &'arena Bump, ...)` and update
   every caller in `src/main.rs`, `src/mcts/mod.rs#tests`,
   `src/mcts_server.rs#non-parallel-handlers` to construct an arena.
4. `cargo test --release --locked --tests` filtering OUT
   parallel-marked tests. Target: all single-thread tests pass.
5. Validate the single-thread reference equivalence:
   `cargo test --release --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path`
   — speedup ratio reference vs. optimized must stay ≈ 1.000 (PV
   equivalence under controller-disabled config).
6. heaptrack on scenario A: `< 200 K allocations` is the bar.

### 3.2 — parallel correctness + TSAN (target: 390 / 390 pass)

1. Update `parallel.rs` worker spawn to use `crossbeam::scope` (or
   pass an `Arc<TT>` whose internal arena is ready to outlive the
   threads — but `Bump` is `!Sync`, so `crossbeam::scope` with
   per-bucket alloc-under-Mutex is the cleaner shape).
2. `cargo test --release --locked` — must hit **390 passed**.
3. `v5_stress_parallel` ×10 consecutive runs, no flakes:
   ```sh
   for i in $(seq 1 10); do
     target/release/deps/mcts_demo-* --exact zobrist_tt_parallel_verify::v5_stress_parallel
   done
   ```
4. **TSAN** on `v5_stress_parallel`. Requires nightly:
   ```sh
   RUSTFLAGS='-Z sanitizer=thread' \
     cargo +nightly test --release --target x86_64-unknown-linux-gnu \
       zobrist_tt_parallel_verify::v5_stress_parallel
   ```
   Zero data races required to advance to 3.3.

### 3.3 — final validation + commit

1. Full suites green: cargo 390, pytest 287, orchestrator harness
   10/10.
2. **Numerical bars (scenario A):**
   - heaptrack: `< 200 K` allocations
   - perf stat: dTLB-miss `< 10 %`, IPC `> 1.7`
   - hyperfine `--warmup 5 --runs 15`: wall `< 380 ms`
   - callgrind: `TT::get_or_create < 8 %` of total Ir,
     total Ir `< 1.8 G`
3. Update `docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md` §6 success
   bars with achieved numbers; or write a `MCTS_PROFILE_DELTA_PHASE3_<date>.md`
   in the same style as the existing delta docs.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Lifetime self-reference in `MctsEngine` if you try to put `Bump` inside | The arena lives in the caller's frame. EVERY engine call site (handlers, tests, FFI) gets `let arena = Bump::new();` immediately before `MctsEngine::new(&arena, …)`. |
| `Bump` is `!Sync` → parallel arena alloc is UB if mishandled | Per-bucket `Bump` serialized by the existing per-bucket `Mutex`. Don't add a global `Mutex<Bump>`. Don't add `unsafe impl Sync` to `Bump` directly. |
| `advance_root` / `replace_root_state` keep nodes alive across "logical search resets" | The arena is keyed to the engine's lifetime, not to a single search. `advance_root` keeps nodes; `replace_root_state` rebuilds the engine and the caller is responsible for the arena lifetime. Document this. |
| FFI surface (Python ctypes calls into the engine) | The arena lives for the duration of one FFI call. Each call into `MctsEngine::new` from FFI gets its own scope-bounded `Bump`. |
| `MctsNode<'arena>` Drop semantics | `Bump` doesn't run Drop on its allocations — bumpalo just resets the underlying arena. Anything in `MctsNode` that needs `Drop` (Vec, RwLock guards, atomics) must be Drop-safe in this pattern. Atomics are fine; `RwLock<Vec<...>>` will leak the inner Vec heap if not explicitly run. Use `bumpalo::collections::Vec` for the edges Vec, OR keep `parking_lot::RwLock<Vec<MctsEdge<'arena, M>>>` with the understanding that the Vec's heap (containing `&'arena` references — inline, no further heap) is freed when the lock drops. |
| Thermal regime variance (5900X) | Use `--warmup 5 --runs 15` minimum. perf-stat counters are thermal-immune for instruction/cycle/cache-miss; use those for definitive claims. cachegrind `--cache-sim=yes` for absolute geometric truth. |
| TSAN false positives on `parking_lot` | `parking_lot` has historically had TSAN annotations issues. If TSAN flags an internal `parking_lot` race, suppress with `TSAN_OPTIONS=suppressions=/path/to/suppressions.txt` listing only the `parking_lot::*` frames. Don't suppress at the application-frame level. |

---

## 5. Profiling commands (canonical)

```sh
# Build the test binary (matches how baseline artifacts were generated)
cargo test --release --locked --no-run

# Pick the newest mcts_demo test binary
TB=$(ls -t target/release/deps/mcts_demo-*[0-9a-f] | grep -v '\.d$' | grep -v 'long-type' | head -1)
echo $TB

# Hyperfine (use --warmup 5 --runs 15 minimum on this host)
hyperfine --warmup 5 --runs 15 --export-json /tmp/phase3_after.json \
  "$TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path"

# perf stat (thermal-immune intrinsics)
perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses -r 5 \
  $TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path

# heaptrack (allocation count is the headline number for Phase 3)
heaptrack -o /tmp/phase3_after.heaptrack \
  $TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path
heaptrack_print /tmp/phase3_after.heaptrack.zst | head -60

# Callgrind with cache simulation (absolute truth)
valgrind --tool=callgrind --callgrind-out-file=/tmp/phase3_after.callgrind \
  $TB --exact --nocapture --ignored mcts::tests::bench_search_controller_fixed_iterations_fast_path
callgrind_annotate /tmp/phase3_after.callgrind | head -60
```

Pre-Phase-3 artifacts (compare against): `tmp/profiles_20260425_postopt/`.

---

## 6. Hardware / tooling

- AMD Ryzen 9 5900X · 64 GiB · Linux 6.17 / glibc 2.39
- Profilers: perf, samply, heaptrack, valgrind (callgrind/cachegrind/massif),
  hyperfine, flamegraph, cargo-flamegraph, cargo-pgo, cargo-bloat,
  cargo-llvm-lines
- Thermal regimes: ~580 ms cool / ~850 ms hot for scenario A.
  See `memory/reference_thermal_regime.md`.
- ptrace_scope = 1 (or 0 if previously relaxed). When debugging
  multi-process hangs, ask the user to run `sudo gdb -p $PID`
  rather than burning time on attach workarounds — see
  `memory/feedback_gdb_via_user.md`.

---

## 7. Files this plan does NOT touch

- `quartz/jax_training_runtime.py` — backend facade, not in MCTS hot loop.
- `quartz/onnx_support.py`, `quartz/gomocup_export.py` — deployment paths.
- `src/calibration.rs`, `src/ablation_*.rs` — test/experiment paths only.
- The Python orchestrator hang fix (Phase 4) is fully landed; do not
  revisit unless TSAN or test failures point back to that surface.

---

## 8. Acceptance checklist (paste into the final commit message)

```
Semantic audit
  cargo test --release --locked            390 passed, 0 failed
  pytest tests/ -q                         287 passed, 6 skipped
  orchestrator harness 10×                 10 / 10 complete
  v5_stress_parallel 10×                   10 / 10 no flakes
  TSAN on v5_stress_parallel               zero data races

Measurement (scenario A: Gomoku-7 1T 15K iter,
  hyperfine --warmup 5 --runs 15, perf stat -r 5)
  pre  (ca75b34)   wall ~558 ms, IPC 1.13, allocs 886 K, dTLB 31.6 %
  post             wall <X> ms,  IPC <Y>,  allocs <Z>, dTLB <W> %

Bars hit:
  - allocations < 200 K                    ☐
  - IPC > 1.7                              ☐
  - wall (scenario A) < 380 ms             ☐
  - dTLB miss rate < 10 %                  ☐
  - TT::get_or_create < 8 % Ir             ☐
```

---

## 9. Session-start prompt — paste into a fresh session

```
You are continuing optimization work on the QUARTZ AlphaZero codebase
at /home/cosmosapjw/Dropbox/personal_projects/quartz. HEAD is ca75b34.

Phases 1, 2A/B/C, 4, and 5 of the 2026-04-25 unified plan are all
landed and green (cargo 390, pytest 287, orchestrator harness 10/10).
The only remaining item from that plan is Phase 3 — bumpalo arena
for Arc<MctsNode>.

Read these in order before starting:
  1. docs/PHASE3_BUMPALO_SESSION_PROMPT_20260425.md   ← THIS PROMPT
                                                       expanded form
  2. docs/MCTS_PROFILE_AUDIT_20260425_POSTOPT.md      ← latest Rust
                                                       state with
                                                       cachegrind data
  3. docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md       ← parent plan
  4. docs/MCTS_PROFILE_DELTA_20260425.md              ← Step-by-step
                                                       Rust delta

The Phase-3 design decision is already made (option (b),
&'arena MctsNode<'arena, M>, per-bucket bumpalo::Bump). Audit
option (a) Arc::from_raw is unsound on stable Rust without
allocator_api — see §1 of the prompt doc for the analysis. Don't
re-litigate.

Execute the four checkpoints in order — each its own commit:
  3.1  single-thread arena landing (~370/390 tests pass)
  3.2  parallel correctness + TSAN (390/390 pass, no flakes,
       zero data races on v5_stress_parallel)
  3.3  final validation + measurement commit

Per-step protocol (mandatory; see memory/feedback_per_step_audits.md):
  1. Read all affected files fully before editing.
  2. Apply patch.
  3. Semantic audit:
     - cargo test --release --locked        target 390 pass
     - pytest tests/ -q                     target 287 pass
     - parallel: v5_stress_parallel 10×, no flakes
     - TSAN (3.2 only): zero data races
  4. Measure (per checkpoint):
     - 3.1 heaptrack: < 200 K allocations
     - 3.3 hyperfine --warmup 5 --runs 15 (5900X needs this for
       stable wall-clock)
     - 3.3 perf stat -r 5 for cycles/IPC/dTLB
     - 3.3 callgrind for per-function Ir attribution
  5. Commit with HEREDOC. Include semantic-audit lines and
     measurement block. Do NOT batch checkpoints.

The 5900X has two stable thermal regimes (~580 ms cool / ~850 ms
hot for scenario A). Always use --warmup 5 --runs 15. perf-stat
counters are thermal-immune; trust them more than wall-clock for
small claims. Reference: memory/reference_thermal_regime.md.

If you hit a multi-process hang during testing and gdb refuses to
attach (ptrace_scope = 1), ask the user to run sudo gdb -p $PID
themselves — see memory/feedback_gdb_via_user.md for the exact
recipe. Don't burn time on attach workarounds.

Authorized to execute autonomously. Phase 3's checkpoint discipline
is non-negotiable; the user has explicitly asked for it. Auto mode
does NOT mean skip checkpoints or batch commits.
```
