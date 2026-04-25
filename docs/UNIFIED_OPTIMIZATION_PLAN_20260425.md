# QUARTZ — Unified Optimization Plan & Session-Start Prompt
## (synthesized from 4 audit/delta documents on 2026-04-25)

This file is dual-purpose:
1. **A synthesis** of the four 2026-04-25 audit/delta documents into one
   prioritized plan.
2. **A self-contained session-start prompt** — paste sections §0–§4 verbatim
   into a fresh Claude Code session to resume work without re-discovery.

---

## 0. Source Documents (read in this order)

All under `docs/` on commit `9232342`:

| # | File | What it gives |
|---|---|---|
| 1 | [`MCTS_PROFILE_AUDIT_20260425.md`](MCTS_PROFILE_AUDIT_20260425.md) | Pre-patch Rust profile audit (28 KB). Established the baseline: 2.4 M alloc, 0.96 IPC, 31 % dTLB-miss on Gomoku-7. Identified P0/P1/P2/P3 targets. |
| 2 | [`MCTS_PROFILE_DELTA_20260425.md`](MCTS_PROFILE_DELTA_20260425.md) | Step-by-step delta after 4 Rust patches landed (12 KB). Wall-clock −7.6 %, allocations −63 %, IPC +12.5 %. |
| 3 | [`MCTS_PROFILE_AUDIT_20260425_POSTOPT.md`](MCTS_PROFILE_AUDIT_20260425_POSTOPT.md) | Deeper post-patch Rust audit (36 KB) with cachegrind `--cache-sim=yes`. Surfaces 3 new P1 quick wins; pins **P0-A bumpalo arena** as the dominant remaining lever. |
| 4 | [`PYTHON_PROFILE_AUDIT_20260425.md`](PYTHON_PROFILE_AUDIT_20260425.md) | Multi-tool Python audit (30 KB). 5 patched, several deferred. Flags an **orchestrator hang regression** introduced by `7274442..29c452c`. |
| 5 | [`PYTHON_PROFILE_DELTA_20260425.md`](PYTHON_PROFILE_DELTA_20260425.md) | Records what Python patches landed (13 KB). All 292 tests + parity goldens pass. Patches are **in working tree, uncommitted**. |

---

## 1. Repository State (snapshot at synthesis time)

**HEAD:** `9232342  docs: post-optimization profile delta report`

**Recent commits** (most recent first):
```
9232342 docs: post-optimization profile delta report
e952305 games/gomoku: replace Vec<i8> board with fixed [i8; MAX_SQ]
ea648c8 mcts/select: pre-size the path Vec to avoid first-grow churn
d602609 mcts: migrate node.edges RwLock from std::sync to parking_lot
2628081 Profile: switch release LTO from thin to fat
29c452c Apply audit minimal-patch plan (10 patches)
```

**Working tree (uncommitted, must be reviewed and committed):**
```
M quartz/alphazero_train.py        (F401 sweep)
M quartz/encoders.py                (vectorize encode/decode + heuristic_prior)
M quartz/qipc.py                    (struct.Struct cache, F401)
M quartz/runtime_support.py         (F401)
M quartz/selfplay_runtime.py        (plan_selfplay_runner_chunk + F401)
M quartz/torch_training_runtime.py  (F401)
?? docs/MCTS_PROFILE_AUDIT_20260425.md
?? docs/MCTS_PROFILE_AUDIT_20260425_POSTOPT.md
?? docs/PYTHON_PROFILE_AUDIT_20260425.md
?? docs/PYTHON_PROFILE_DELTA_20260425.md
?? 0    (accidental empty file — delete)
```

**Test baseline:** `cargo test --release --locked` → 390 passed, 0 failed,
65 ignored. `pytest tests/` → 287 passed, 6 skipped (controller-regression
strict tier — gated behind `QUARTZ_RUN_CONTROLLER_REGRESSION_STRICT=1`).

**CI:** `.github/workflows/tests-gate.yml` runs cargo + pytest on push/PR.

---

## 2. Headline Numbers (where we are now)

Scenario A is the canonical 15 K-iter Gomoku-7 single-thread benchmark.

| Metric | Pre-patch (Apr-25 baseline) | Now (`9232342`) | Δ |
|---|---|---|---|
| Heap allocations | 2,386,239 | 886,083 | **−62.9 %** |
| Temporary allocs | 679,681 | 60 | −99.99 % |
| Cycles | 3.11 G | 2.72 G | −12.5 % |
| IPC | 0.96 | **1.13** | +18 % |
| Wall-clock (steady-state) | 627 ms | 558 ms | −11.0 % |
| dTLB miss rate | 31.4 % | 31.6 % | **flat** ← key remaining issue |

The dTLB-miss rate has not moved because the dominant remaining heap
pattern is **760 K `Arc<MctsNode>` allocations** in the TT path. That is
P0-A below.

**Cross-scenario** (post-patch hyperfine `--warmup 3 --runs 10`):

| Scenario | Wall | Comment |
|---|---|---|
| A Gomoku-7 1T (15 K iter × 2) | 558 ± 11 ms | 9.91× Chess |
| B Chess (1T+4T) | 56 ± 1.4 ms | reference (Copy state) |
| D Gomoku-15 multi (1T+4T) | 86 ± 4 ms | 4T scaling now 1.85× (was 0.88× pre-patch) |

**Python (post-uncommitted-patches, from `PYTHON_PROFILE_DELTA_20260425.md`):**

| Function | Pre | Post | Speedup |
|---|---|---|---|
| `heuristic_prior` (4 K calls) | 5.47 s | 0.53 s | **10.4×** |
| `encode` (4 K calls) | 0.38 s | 0.024 s | **15.9×** |
| `pack_qipc_arena_eval_req` (4 K) | 0.59 s | 0.075 s | **7.8×** |
| `plan_selfplay_runner_chunk` (20 K) | 0.28 s | 0.043 s | **6.4×** |

---

## 3. Unified Plan — 5 Phases

Ordered by `(dependency × ROI × risk)`. Each phase declares its goal,
files, semantic audit, validation, expected gain. Every phase ends with
its own commit; no phase batches multiple unrelated changes.

### Phase 1 — Land the Python patches that are in working tree

**State:** 5 patches already implemented in working tree by a parallel
agent. Parity verifier confirms behavior preserved
(`tmp/_parity_verify.py` exists). Tests pass (292/292 from delta doc).
Just need to: verify tests still pass on this machine, commit, push.

**Steps:**

1. **Inspect the working tree changes** — `git diff quartz/` and confirm
   they match the delta doc. (Spot-check `encoders.py:106-198`,
   `qipc.py:758-783, 1006-1131`, `selfplay_runtime.py:58, 96-175`.)
2. **Run parity verifier** if `tmp/_parity_check.py` and `tmp/_parity_verify.py`
   exist:
   ```sh
   venv/bin/python tmp/_parity_verify.py
   ```
   Expected output: `=== ALL PARITY TESTS PASSED ===`
3. **Run full Python test suite:**
   ```sh
   venv/bin/python -m pytest tests/ -q --ignore=tests/test_play_gui.py
   ```
   Expected: 287+ passed, 0 failed.
4. **Run Rust test suite** to confirm no cross-impact:
   ```sh
   cargo test --release --locked
   ```
   Expected: 390 passed, 0 failed.
5. **Commit** the 6 modified Python files + the 4 new docs in one or
   two commits:
   - Commit 1 (docs): add the four `*_20260425*.md` docs.
   - Commit 2 (code): the 5 Python patches.
   - Delete the stray `0` file.

**Semantic audit:** parity verifier covers wire format (qipc), encoder
output bit-equality, plan dict shape. Tests cover everything else. The
patches are intentionally narrow (vectorization / struct caching /
unused-import sweep) and pre-existing parity goldens guard against
behavior drift.

**Expected gain:** none from this commit (already in tree). Locks in
the 6–16× per-call wins the delta doc records.

---

### Phase 2 — Rust P1 quick wins (3 small patches)

From `MCTS_PROFILE_AUDIT_20260425_POSTOPT.md` §6 / §7. Three independent
one-liner changes; commit each separately with a per-step semantics
audit.

#### 2A. Pre-size TT bucket HashMap (P1-A)

**File:** `src/mcts/tt.rs:70-72` (the `TtBucket::new()` impl).

**Patch:**
```rust
impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        TtBucket {
            map: TtMap::with_capacity_and_hasher(
                MAX_ENTRIES_PER_BUCKET / 2,    // 2048
                Default::default(),
            ),
        }
    }
}
```

**Semantic audit:** `with_capacity_and_hasher` is observably equivalent
to `default()` for any subsequent insert/get. Capacity hint only.

**Validation:**
- `cargo test --release --locked` → 390 passed.
- callgrind on scenario A → `hashbrown::raw::RawTable::reserve_rehash`
  drops out of the top-20.
- perf stat → expect ~1–2 % wall-clock improvement on A.

**Expected gain:** 1–2 % Ir.

**Cost:** 256 buckets × ~16 KiB = ~4 MiB fixed overhead per
`TranspositionTable`. Already dwarfed by the current 215 MiB peak.

---

#### 2B. `#[inline]` on `check_win_at` and `check_win_at_hypothetical` (P1-B)

**File:** `src/games/gomoku.rs:288` and `:245`.

**Patch:** add `#[inline]` above each function signature.

**Semantic audit:** `#[inline]` is a hint, not a binding directive. Zero
behavior change.

**Validation:** `cargo test --release --locked` → 390 passed.

**Expected gain:** 0–1 %; insurance against future refactors that might
move these functions across compilation units.

---

#### 2C. `quartz_cache` → `parking_lot::RwLock` (P1-C, consistency)

**File:** `src/mcts/mod.rs:24, 195, 247, 297-298, 304, 585, 608, 661,
695, 823, 840, 881` (every site that touches `quartz_cache`).

**Patch:**
```rust
// In src/mcts/mod.rs
- use std::sync::{Arc, RwLock};
+ use std::sync::Arc;
+ // quartz_cache uses parking_lot::RwLock, consistent with node.edges.
+ // tt buckets continue to use parking_lot::Mutex.
+ use parking_lot::RwLock;

// And at each .read().unwrap() / .write().unwrap() site:
- self.quartz_cache.read().unwrap()
+ self.quartz_cache.read()
- self.quartz_cache.write().unwrap()
+ self.quartz_cache.write()
```

**Semantic audit (full audit per Step 2 protocol):**
- `quartz_cache` is `RwLock<Option<QuartzStats>>`. Used only by
  QuartzController code paths to read the latest computed stats and
  to invalidate / write fresh ones. No code relies on poison detection.
- Cross-check: `zobrist_tt_parallel_verify::v3_parallel_vs_sequential`
  and `v5_stress_parallel` exercise multi-threaded controller usage.
  Must pass.
- Cross-check: `test_fixed_iterations_fast_path_matches_reference`
  asserts PV equivalence under controller-disabled config — should
  be unaffected, but must pass.

**Validation:**
- `cargo test --release --locked` → 390 passed.
- perf stat → expect < 0.5 % wall-clock impact (this is a hygiene
  patch, not a perf patch).

**Expected gain:** < 0.5 %. Done for codebase consistency.

---

### Phase 3 — Rust P0-A: bumpalo arena for `Arc<MctsNode>` (THE big lever)

From `MCTS_PROFILE_AUDIT_20260425_POSTOPT.md` §6 / §7.3. This is the
single biggest measurable remaining win. Audited cost: 760 K Arc
allocations = 86 % of remaining heap allocations and 51 % of all D1
read-misses. Expected gain: scenario A from 558 ms → 320–380 ms
(−30–45 %), dTLB-miss from 31.6 % → < 10 %.

**This phase must be done in its own branch with its own audit cycle.**
It is the highest-risk patch in this plan (lifetime surgery, parallel
correctness). Below is the recommended sequence, not a single commit.

#### 3.0 Pre-flight: pick the design

Two options from the audit:

- **(a) Preserve Arc API.** Allocate `MctsNode` body in a per-search
  `bumpalo::Bump` arena, wrap as `Arc<MctsNode>` via `Arc::from_raw`.
  Saves the body heap call per node (760 K → 1) but retains Arc
  strong/weak counts (32 B per node still in arena). Compatible with
  current parallel design.
- **(b) Replace `Arc<MctsNode>` with `&'arena MctsNode`.** Higher ceiling
  (also kills `Arc::drop_slow`'s 13.7 % of LLd read-misses). Requires
  lifetime surgery in 6+ files (`tt.rs`, `node.rs`, `select.rs`,
  `expand.rs`, `backup.rs`, `parallel.rs`, `mod.rs`, `mcts_server.rs`,
  test harnesses).

**Recommendation:** start with (a). Land it, measure, then decide
whether (b)'s additional 5–8 pp is worth the surgery.

A still safer landing: **`typed-arena::Arena<MctsNode>`** with
`&'arena MctsNode`. No `Arc::from_raw` unsafety; clearer ownership.
The cost is the lifetime surgery of (b), but with simpler semantics
than raw bumpalo.

#### 3.1 Single-thread first

Steps:
1. Add `bumpalo = "3"` (or `typed-arena = "2"`) to `Cargo.toml`.
2. Add an `arena: UnsafeCell<Bump>` field to `TranspositionTable`. Mark
   the impl `Sync` only by careful argument: the arena is mutated only
   while holding a bucket `Mutex`.
3. Modify `TT::get_or_create` to allocate the node body via the arena.
4. Run `cargo test --release --locked` against the **single-thread
   tests only** first (gomoku/go/chess unit tests, the
   `bench_search_controller_fixed_iterations_fast_path` reference vs
   optimized comparison).

**Semantic audit at this checkpoint:**
- All non-parallel tests pass (target: ~370 of 390).
- `test_fixed_iterations_fast_path_matches_reference` PV-equivalence holds.
- heaptrack scenario A → < 200 K allocations (target).

#### 3.2 Parallel correctness

Now confirm `zobrist_tt_parallel_verify::v1_*..v5_stress_parallel`
pass. The arena's `!Send` constraint may force one of:
- Per-thread shard of arenas (each worker owns one Bump; coordinator
  drops them at search end).
- A wrapper type that uses `parking_lot::Mutex<Bump>` for cross-thread
  allocs.

The first is faster but more code. The second is simpler but introduces
a Mutex contention point.

**Semantic audit at this checkpoint:**
- All 390 tests pass.
- `v5_stress_parallel` passes 10 consecutive runs without flakes.
- Run under TSAN (`RUSTFLAGS=-Z sanitizer=thread cargo test --release`
  on a nightly toolchain) on `v5_stress_parallel`. Zero data races
  reported.

#### 3.3 Validation and commit

- `cargo test --release --locked` → 390 passed.
- heaptrack scenario A → target < 200 K allocations.
- perf stat → target dTLB-miss < 10 %.
- callgrind → target `TT::get_or_create` < 8 %, total Ir < 1.8 G.
- hyperfine → target wall-clock A < 380 ms.

**Estimated effort:** 1–2 days for single-thread + tests; +1 day for
parallel sharded arena + TSAN.

**Risk register (must verify):**
- `Arc::from_raw` correctness (option a). Drop timing must align with
  arena lifetime.
- `MctsNode` may be referenced after `TranspositionTable` is dropped if
  any external holder retains an Arc/reference. Check `MctsEngine`'s
  drop order in `mcts/mod.rs`.
- Parallel search uses `node.edges` writes; if `MctsNode` body lives in
  a `&mut Bump` allocation, the `RwLock<Vec<MctsEdge>>` inside it must
  remain `Send`/`Sync`-correct. Confirm.

---

### Phase 4 — Cross-cutting: orchestrator hang triage

From `PYTHON_PROFILE_AUDIT_20260425.md` §7.

**Symptom:** `tmp/profile_python_orchestrator.py` (the harness used
for the 2026-04-20 SHM/broker baseline) hangs indefinitely. Verified:
- Python is parked in `time.sleep()` (`hrtimer_nanosleep`).
- `mcts_demo --server` child is parked on the futex
  (`futex_do_wait`).
- Both sides waiting on each other → missing wakeup.

**Bisect range:** `9232342..bc1a7aa` on the `qipc.py + selfplay_runtime.py
+ runtime_support.py` triple. The audit pins
[`7274442` "backup"] as the most likely culprit (added 280 lines to
`qipc.py`, 117 to `runtime_support.py`, 216 to `selfplay_runtime.py`,
including `pack_qipc_arena_eval_req`).

**Steps:**
1. `git bisect start; git bisect bad HEAD; git bisect good ...some_pre-7274442_commit...`
2. Use the orchestrator harness as the reproducer:
   ```sh
   timeout 60 venv/bin/python tmp/profile_python_orchestrator.py \
       --iters 8 --n-games 2 --parallel 2 --batch-size 2
   ```
   Hangs → `bad`. Completes → `good`.
3. The bisect should converge on the wakeup-deadlock-introducing commit.
4. Fix is likely Rust-side: the new arena-eval frame path must signal
   the Python broker (or vice versa) on slot transitions. Look for
   matching `notify_one()` / cond-var / eventfd writes that are missing.
5. **The fix may need to land Rust-side** — see
   `PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md` §10 for the
   "wakeup-via-Rust" recommendation that was already on the table.

**Semantic audit:** the orchestrator harness must reach
`selfplay_elapsed_s ≈ 0.14 s` (the 2026-04-20 baseline) without hangs.
End-to-end `cargo test --release --locked` + `pytest tests/` must
remain green.

**Expected gain:** unblocks all future end-to-end profiling and the
deferred `unpack_qipc_arena_eval_resp` / `unpack_shm_search_response`
optimization (audit §3.2).

---

### Phase 5 — Python lint cleanup (B023, F821)

From `PYTHON_PROFILE_AUDIT_20260425.md` §10.3 / §11. Deliberately last
because each hit needs human review (auto-fix risks introducing bugs).

**Targets:**
- 30 × `B023` loop-variable-capture in closures (bug-shaped).
- 9 × `F821` undefined-name.
- 2 × `F822` undefined-export.
- 13 × `PERF203` try/except-in-hot-loop (modest perf win).

**Steps:**
1. `venv/bin/python -m ruff check quartz/ --select B023 --statistics`
   → list all 30 sites.
2. For each `B023`: read context, decide whether intent was bind-by-
   value (`lambda x=x: ...`) or accept-by-reference. Fix accordingly.
3. Same for `F821` / `F822` (9+2 sites). These are runtime hazards;
   may indicate dead code paths or refactor leftovers.
4. `PERF203`: hoist `try` outside the hot loop where the `except`
   branch is unreachable from the steady-state body.

**Semantic audit:** test suite must pass after each fix. Each `B023`
fix changes lambda capture semantics — verify the lambda is called in
contexts where the new semantics match the original intent.

**Expected gain:** 0 % wall-clock; correctness improvement.

---

## 4. Quick-Start Commands (for the new session)

```sh
cd /home/cosmosapjw/Dropbox/personal_projects/quartz

# Confirm state
git log --oneline -6
git status --short

# Re-establish baselines
cargo test --release --locked 2>&1 | tail -3        # 390 passed
venv/bin/python -m pytest tests/ -q                 # 287 passed

# Read the docs in order (can use Read tool):
#   1. docs/MCTS_PROFILE_AUDIT_20260425_POSTOPT.md   (latest Rust state)
#   2. docs/PYTHON_PROFILE_DELTA_20260425.md          (what's in working tree)
#   3. docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md     (this file)

# Profile artifacts (for callgrind/heaptrack/perf comparisons):
ls tmp/profiles_20260425/         # pre-patch baseline
ls tmp/profiles_20260425_postopt/ # post-Step-4 baseline

# Test binary (pinned in audit doc):
ls -la target/release/deps/mcts_demo-* | grep -v '\.d$'

# Profiler tool inventory:
which perf samply heaptrack valgrind hyperfine flamegraph cargo-flamegraph cargo-pgo
```

---

## 5. Session-Start Prompt (paste this into a fresh session)

```
You are continuing optimization work on the QUARTZ AlphaZero codebase at
/home/cosmosapjw/Dropbox/personal_projects/quartz on commit 9232342.

CONTEXT
The codebase has a Rust MCTS engine (src/mcts/, src/games/) and a
Python training/orchestration layer (quartz/, scripts/). Two parallel
profile-audit cycles ran on 2026-04-25 and produced four documents
under docs/:

  - MCTS_PROFILE_AUDIT_20260425.md          (Rust pre-patch baseline)
  - MCTS_PROFILE_DELTA_20260425.md          (4-step Rust delta committed)
  - MCTS_PROFILE_AUDIT_20260425_POSTOPT.md  (deeper Rust post-patch audit)
  - PYTHON_PROFILE_AUDIT_20260425.md        (Python multi-tool audit)
  - PYTHON_PROFILE_DELTA_20260425.md        (5 Python patches landed,
                                              currently uncommitted in
                                              the working tree)

The synthesis document docs/UNIFIED_OPTIMIZATION_PLAN_20260425.md
contains the merged plan in 5 phases. Read it first.

CURRENT WORKING-TREE STATE
git status will show:
  - 6 modified Python files (encoders.py, qipc.py, selfplay_runtime.py,
    alphazero_train.py, runtime_support.py, torch_training_runtime.py)
    — these are the landed-but-uncommitted Python patches.
  - 4 untracked .md docs in docs/ — the audit documents.
  - 1 stray empty file `0` — delete it.
  - Test baseline: cargo test → 390 passed; pytest → 287 passed.

YOUR TASK
Execute the unified plan in order. Each phase ends with its own
commit; never batch unrelated phases into one commit.

Per-step protocol (mandatory for every patch):
  1. Identify the patch + files from the plan.
  2. Read all affected files fully before editing.
  3. Apply the patch.
  4. Run the semantic audit specific to the patch:
     - For Rust: cargo test --release --locked (must hit 390 pass)
     - For Python: pytest tests/ -q (must hit 287+ pass)
     - For game-rule changes: also check the semantics_audit_*
       invariant tests in src/games/gomoku.rs:954+
  5. Measure with the appropriate profiler (perf stat / hyperfine /
     heaptrack — see plan §3 success bar).
  6. Commit with a HEREDOC message including the measurement results
     and the semantic audit findings.
  7. Update todos.

CRITICAL
- Phase 3 (bumpalo arena) is the highest-risk change. Do NOT skip
  the per-checkpoint audits inside it (single-thread → parallel → TSAN).
- Phase 4 (orchestrator hang) may require Rust-side changes; do not
  attempt Python-only fixes if the bisect points at Rust.
- Always preserve the 4 semantics_audit_* tests in src/games/gomoku.rs.
- The user has explicitly asked for a per-step semantics audit on
  every change — never skip or batch them.

START with Phase 1: confirm Python parity, run both test suites,
commit the 6 Python files + 4 docs in two separate commits, delete
the stray `0` file. Then proceed to Phase 2A.

Authorized to execute autonomously. The user expects: every commit
green, per-step audits documented in commit messages, the unified
plan followed in order.
```

---

## 6. Reference: success bars per phase

| Phase | Allocations | IPC | Wall (A) | dTLB miss | Tests |
|---|---|---|---|---|---|
| Phase 0 (now, `9232342`) | 886 K | 1.13 | 558 ms | 31.6 % | 390 + 287 |
| After Phase 1 (Python committed) | 886 K | 1.13 | 558 ms | 31.6 % | 390 + 292 |
| After Phase 2 (3 Rust quick wins) | ~880 K | ~1.15 | ~540 ms | 31 % | 390 + 292 |
| After Phase 3 (bumpalo) | < 200 K | > 1.7 | < 380 ms | < 10 % | 390 + 292 |
| After Phase 4 (hang fixed) | unchanged | unchanged | unchanged | unchanged | 390 + 292 + orchestrator-harness completes |
| After Phase 5 (lint) | unchanged | unchanged | unchanged | unchanged | 390 + 292 + ruff-clean |

---

## 7. Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| Bumpalo Arc::from_raw drop ordering bug | 3 | TSAN run on v5_stress_parallel; consider `typed-arena` first |
| Python parity verifier files (`tmp/_parity_check.py`) missing | 1 | Inspect git status for `tmp/`, fall back to test suite as audit |
| Orchestrator hang fix may need Rust changes | 4 | Bisect first, scope second |
| `parking_lot::RwLock` for quartz_cache changes lock semantics under panic | 2C | Cross-check zobrist_tt_parallel_verify tests; review every `.read().unwrap()` site |
| ruff B023 auto-fix introduces wrong capture semantics | 5 | Per-site human review; never auto-fix B023 |

---

## 8. Files this plan does NOT touch (out of scope)

- `quartz/jax_training_runtime.py` — JAX backend is documented as a
  thin facade over torch; not in this round.
- `quartz/onnx_support.py`, `quartz/gomocup_export.py` — deployment
  paths; not on the hot loop.
- `src/calibration.rs`, `src/ablation_*.rs`, `src/experiment_*.rs` —
  ignored test paths only; no production wiring.
- The `controller_axes` / `controller_factorial` attribution presets —
  already audited in `review.md` (the prior journal-grade integrated
  audit); semantics are tracked separately.

---

## 9. Acknowledgements

This synthesis pulls from 4 audit cycles done by separate agents on
2026-04-25 plus the per-step optimization landing for the Rust side.
Re-running the plan from scratch on a different machine should
reproduce the numbers within ±5 % thermal-state noise on the same
AMD 5900X / Linux / glibc class of system.
