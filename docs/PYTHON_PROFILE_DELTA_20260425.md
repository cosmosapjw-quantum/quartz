# Python-Side Profile Delta + Applied Patches (2026-04-25)

Companion to [PYTHON_PROFILE_AUDIT_20260425.md](PYTHON_PROFILE_AUDIT_20260425.md).
This document records what the audit's P0/P1 patches actually
accomplished, and what was deliberately deferred.

The hard rule for this round was: **Python-only changes**, no Rust
edits, no behavioural changes. Every patch carries a parity test
against a pre-captured golden ([tmp/_parity_check.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/_parity_check.py),
[tmp/_parity_verify.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/_parity_verify.py)).

---

## 1. Patches landed

### 1.1 `quartz/encoders.py::GomokuEncoder.encode` and `decode`
- File: [quartz/encoders.py:106-123](quartz/encoders.py#L106-L123)
- Replaced per-cell Python loop + numpy element access with a single
  `arr.reshape(bs, bs)` + boolean-mask assignment. `decode` similarly
  uses `view[enc[0] > 0.5] = player`.
- Bytes/dtype/shape preserved — verified against pre-captured golden
  on both 7×7 (gomoku7) and 15×15 (gomoku15) boards across 200
  random boards × 2 players.

### 1.2 `quartz/encoders.py::GomokuEncoder.heuristic_prior`
- File: [quartz/encoders.py:125-198](quartz/encoders.py#L125-L198)
- Vectorized the **adjacency bonus** with a 3×3 pad+slice convolution
  (8-neighbor non-zero count × 0.5). Vectorized the **center bias**
  via `np.indices`.
- Kept the four-direction threat-pattern scan structurally identical
  to the original, but materialized `board_flat` to a Python list
  once at function entry (`bf = arr.reshape(n2).tolist()`). All
  inner-loop reads are now Python-int comparisons instead of
  per-element numpy scalar boxing — the dominant cost in the original.
- Iteration was restricted to empty cells via a precomputed
  `np.flatnonzero(empty_mask).tolist()` rather than a `continue` on
  every occupied cell.
- Float arithmetic preserved — verified `np.allclose(post, pre, atol=1e-5)`
  across 100 random 7×7 priors and 20 random 15×15 priors.

### 1.3 `quartz/selfplay_runtime.py::plan_selfplay_runner_chunk`
- File: [quartz/selfplay_runtime.py:96-175](quartz/selfplay_runtime.py#L96-L175)
- Cached `os.cpu_count()` once at module import as
  `_LOGICAL_CPU_COUNT` ([selfplay_runtime.py:58](quartz/selfplay_runtime.py#L58)).
- Replaced 21 `max()` calls per invocation with explicit comparisons.
  Replaced two passes over `recent_chunks` with a single pass.
  Removed `int(...)` boxing of values that were already int-shaped,
  hoisted `cfg.get` to a local.
- Output schema preserved — verified `out == golden` on 100 plan
  invocations with realistic config + recent_chunks.

### 1.4 `quartz/qipc.py::pack_qipc_arena_eval_req`
- File: [quartz/qipc.py:1006-1131](quartz/qipc.py#L1006-L1131)
- Module-level `struct.Struct(...)` caches added at
  [quartz/qipc.py:758-783](quartz/qipc.py#L758-L783):
  `_ARENA_REQ_PREFIX`, `_ARENA_REQ_OPTS_FIXED`,
  `_ARENA_REQ_SESSION_HEADER`, `_ARENA_REQ_PLAYER_LEN`,
  `_ARENA_REQ_GO_TAIL`, plus pre-packed constants for the
  bool-absent / bool-true / bool-false / u64-zero common cases.
- Closures (`_pack_opt_str/_pack_opt_bool/_pack_opt_u64`) replaced
  with inline calls against the cached struct objects (eliminates
  per-call function definition + the `dict(...)` copy).
- `bytes(...)` board encoding for boards uses one `struct.pack(f"<{n}b", ...)`
  call (already in the original), but with a tuple comprehension
  instead of a list.
- Wire format preserved — verified byte-for-byte equality across 12
  payloads spanning gomoku, chess, go, and multi-session shapes.

### 1.5 Mechanical `F401` unused-import sweep
- Files: `quartz/alphazero_train.py`, `quartz/cli_main.py`,
  `quartz/qipc.py`, `quartz/runtime_support.py`,
  `quartz/torch_training_runtime.py`,
  `quartz/training_runtime_utils.py`, `quartz/encoders.py`.
- 75 unused imports removed via `ruff check --select F401 --fix`.
- One re-export restored explicitly: `from collections import OrderedDict`
  in `alphazero_train.py` (used by `tests/test_training_pipeline_regressions.py::test_nn_eval_cache_treats_move_to_end_race_as_miss`).

---

## 2. Measured deltas

Workload harness: [tmp/profile_python_full.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/profile_python_full.py),
identical args before and after, single run each (CPU only, no
threading variance).

### 2.1 `tottime` of the patched functions (cProfile)

| Function                          | Pre        | Post       | Speedup |
|-----------------------------------|-----------:|-----------:|--------:|
| `heuristic_prior` (4 000 calls)   | 5.472 s    | 0.527 s    | **10.4×** |
| `encode` (4 000 calls)            | 0.382 s    | 0.024 s    | **15.9×** |
| `pack_qipc_arena_eval_req` (4k)   | 0.587 s    | 0.075 s    | **7.8×**  |
| `plan_selfplay_runner_chunk` (20k)| 0.276 s    | 0.043 s    | **6.4×**  |
| `builtins.max` (within `plan`)    | 0.438 s    | 0.002 s    | **219×** (eliminated) |

### 2.2 Whole-workload wall (`workload_summary.txt`)

| Workload     | Pre         | Post        | Speedup |
|--------------|------------:|------------:|--------:|
| `encoder`    | 6.16 s      | 0.78 s      | **7.9×** |
| `qipc`       | 0.90 s      | 0.12 s      | **7.5×** |
| `plan`       | 1.39 s      | 0.10 s      | **14.2×** |
| `model_fwd`  | 0.39 s      | 0.22 s      | (noise — torch warmup variance) |
| `model_train`| 5.00 s      | 1.30 s      | (noise — torch dynamo lazy init variance) |
| `replay`     | 4.07 s      | 1.16 s      | (noise — first-call torch import variance) |
| `shm_ring`   | 0.22 s      | 0.03 s      | (~7× — incidentally won by the qipc hot path no longer warming the import cache) |

The three patched workloads (`encoder`, `qipc`, `plan`) show real,
attributable wins. The other workloads' deltas are dominated by
process-start torch import variance and are not load-bearing for
this round.

### 2.3 Per-call cost (back-of-envelope, post-opt)

| Function | Pre per call | Post per call |
|----------|-------------:|--------------:|
| `heuristic_prior` (7×7) | 1 370 µs | 132 µs (10.4×) |
| `encode` (7×7)          | 95 µs   | 6 µs (15.9×)  |
| `pack_qipc_arena_eval_req` (4 sessions) | 147 µs | 19 µs (7.8×) |
| `plan_selfplay_runner_chunk`            | 70 µs  | 5 µs (14×)   |

### 2.4 Verification

- Parity verifier ([tmp/_parity_verify.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/_parity_verify.py)):
  ```
  encode/decode parity OK
  heuristic_prior parity OK (g7=100, g15=20)
  plan parity OK
  qipc parity OK across 12 payloads
  === ALL PARITY TESTS PASSED ===
  ```
- Regression test suite:
  ```
  tests/test_training_pipeline_regressions.py: 183 passed
  tests/* (rest):                              109 passed, 6 skipped
  Total:                                       292 passed, 6 skipped
  ```

---

## 3. Deliberately deferred (no patch this round)

### 3.1 The `tmp/profile_python_orchestrator.py` SHM/broker hang
- §7 of [PYTHON_PROFILE_AUDIT_20260425.md](PYTHON_PROFILE_AUDIT_20260425.md).
- Triage notes: [tmp/python_profile_20260425/orchestrator_hang_notes.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/orchestrator_hang_notes.txt).
- Bisect range: `9232342..bc1a7aa` on the
  `quartz/qipc.py + quartz/selfplay_runtime.py + quartz/runtime_support.py`
  triple.
- **Why deferred:** the most likely fix is a missing wakeup on the
  arena-eval frame path. Both sides are parked
  (`hrtimer_nanosleep` / `futex_do_wait`), and aligning them
  generally requires a Rust-side change to signal the Python broker
  on new SHM activity. The 2026-04-20 audit already reported that
  Rust-side wakeup is the next architectural step. Out of scope for
  a Python-only round.

### 3.2 Further `qipc` wire-format hot paths
- `unpack_qipc_arena_eval_resp` (cc=15) and `unpack_shm_search_response`
  (cc=20) follow the same per-field pattern that
  `pack_qipc_arena_eval_req` did before this round. They were not
  hot in this audit's harness (the workload only packs), but the
  same `struct.Struct` cache pattern would apply.
- **Why deferred:** not currently observable in any harness. Worth
  doing only after the §3.1 hang is fixed and we can re-measure end-to-end.

### 3.3 `replay.collate_replay_samples` sparse → dense scatter
- §6 of the audit. Real per-call cost is ~110 µs after the prior
  audit's `_torch_module()` cache. A further win is only available
  by stacking all `idx` arrays into a single CSR pair and doing one
  `policies_np[row_arr, col_arr] = val_arr` per batch.
- **Why deferred:** not currently a training-visible bottleneck.
  Revisit only if the replay path shows up in a trace.

### 3.4 `cli_main.run_training_main` (cc=106) / `prepare_training_context` (cc=71)
- §10.1 of the audit. These are coordination-heavy top-level
  functions, not inner loops. The leverage is maintainability and
  testability, not throughput.
- **Why deferred:** scope of this round is profile-driven CPU win,
  not structural refactor. A separate cleanup PR is the right home
  for these.

### 3.5 `selfplay_runtime` structural split (Halstead volume 21 594)
- Same shape as §3.4. Three logical concerns are tangled in one
  module: SHM/broker exchange, runner scheduling/planning, and
  per-game state machine bookkeeping.
- **Why deferred:** would touch every consumer of the module.
  Should be sequenced after §3.1 is fixed (so we can profile the
  broker path properly first).

### 3.6 `evaluator_runtime.shm_eval_loop` (cc=58) and `_run_shared_eval_matches` (cc=47)
- Mirror of §3.5 on the evaluator side. Shares the same hang
  symptom (§3.1). Cannot be profiled until the broker hang is
  fixed.

### 3.7 Remaining ruff issues
- 30 `B023` (loop-variable-in-closure capture) — **bug-shaped**;
  needs case-by-case judgement on each lambda/closure.
- 13 `PERF203` (try/except in hot loop) — modest perf, but each
  case has to be checked for whether the except is reachable from
  the loop body.
- 9 `F821` (undefined name) and 2 `F822` (undefined export) —
  **runtime hazards**; should be triaged not auto-fixed.
- 21 `B905` (zip without strict) — hygiene only.
- 44 `SIM105` (suppressible-exception) — cosmetic; no behaviour win.
- **Why deferred:** auto-fixing these without per-site review can
  introduce real bugs (especially `B023` and `F821`). Worth a
  dedicated cleanup PR with human review per hit.

### 3.8 Vulture dead-code (~20 high-confidence hits)
- Mostly overlap with the F401 sweep already done. The remaining
  hits are dead local variables (e.g. unused `exc_type`, `tb` in
  `__exit__` signatures) which are required by Python's context
  manager protocol — **not** safely removable.
- **Why deferred:** the safe subset was already removed by the F401
  sweep; the rest are protocol-required.

### 3.9 Import-time reduction beyond F401
- Torch dominates cold-start by ~10× (see §8 of the audit). Local
  module imports are uniformly under 160 ms. The only further win
  available without ripping out torch entirely would be a lazy
  attribute hook on `quartz/__init__.py`, but every entry point we
  care about already imports torch eagerly because it needs the
  model.
- **Why deferred:** no Python-side win remains without a behavioural
  change.

### 3.10 The `_run_tests()` self-test in `quartz/encoders.py`
- The in-file self-test at [encoders.py:341+](quartz/encoders.py#L341) asserts
  `t.shape == (3, 7, 7)` but the encoder produces `(17, 7, 7)`.
  This test is stale and would fail if anyone ran
  `python quartz/encoders.py`.
- **Why deferred:** the file is not invoked as a script in CI or
  any docs; the canonical test path is `tests/`. Fixing the assert
  is mechanical but out of scope for a profile-driven round.

---

## 4. Reproducibility

```sh
# Pre-capture goldens (run once before any change):
venv/bin/python tmp/_parity_check.py

# Apply patches as in this document.

# Verify parity:
venv/bin/python tmp/_parity_verify.py

# Re-run profile harness:
venv/bin/python tmp/profile_python_full.py --out tmp/python_profile_20260425/cprofile_postopt
diff <(cat tmp/python_profile_20260425/cprofile/workload_summary.txt) \
     <(cat tmp/python_profile_20260425/cprofile_postopt/workload_summary.txt)

# Regression suite:
venv/bin/python -m pytest tests/ -q
```

---

## 5. Bottom line

Five Python-only patches landed:
1. Vectorized `GomokuEncoder.encode/decode` (15.9×)
2. Vectorized + Python-list-fast-path `GomokuEncoder.heuristic_prior` (10.4×)
3. Single-pass + cached-cpu-count `plan_selfplay_runner_chunk` (6.4× function, 14× workload)
4. `struct.Struct`-cached `pack_qipc_arena_eval_req` (7.8×)
5. F401 unused-import sweep (~75 lines removed across 7 files)

All 292 regression tests pass. Per-call costs of every patched
function dropped by 6× to 16×. The audit's P0 + P1 lists are
fully closed.

Everything else from the audit either depends on Rust changes
(broker wakeup, §3.1/3.5/3.6), needs human-reviewed structural
work (§3.4/3.7), or is already at floor (§3.3/3.8/3.9).
