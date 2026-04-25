# Python-Side Profile Audit (2026-04-25)

Comprehensive multi-tool Python profile + static audit, scoped to the
`quartz/` package and `scripts/`. Complements
[MCTS_PROFILE_AUDIT_20260425_POSTOPT.md](MCTS_PROFILE_AUDIT_20260425_POSTOPT.md)
and [PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md](PYTHON_ORCHESTRATOR_PROFILE_AUDIT_20260420.md).

The previous orchestrator audit closed itself out with the
recommendation that further wins required Rust-side changes. Five days
later the hot Python surface looks the same, but a richer multi-tool
sweep surfaces a small set of high-leverage CPU-bound functions that
are still pure-Python loops, and a SHM/orchestrator path regression
that is currently blocking the same baseline harness.

---

## 0. Tool Inventory

Profile tools used (versions from `venv/bin/pip list`):

| Tool | Version | Role |
|------|---------|------|
| `cProfile + pstats` (stdlib) | 3.11.15 | Function-level cumulative + total time |
| `tracemalloc` (stdlib) | 3.11.15 | Python allocation site attribution |
| `pyinstrument` | 5.1.2 | Wall-clock statistical, call-tree HTML/text |
| `py-spy` | 0.4.2 | Attach/sub-process sampler, flamegraph |
| `scalene` | 2.2.1 | CPU + memory + GPU line-level sampler |
| `viztracer` | 1.1.1 | Function-level event trace (`.json` for Perfetto) |
| `memray` | 1.19.3 | Process-level heap profiler with stats/summary |
| `line_profiler` | 5.0.2 | Line-level timing of explicit functions |
| `yappi` | 1.7.6 | Wall-clock + thread-aware profiler (available, not run because cProfile already covered the same surface) |
| `python -X importtime` | 3.11.15 | Cold import startup cost |
| `ruff` | 0.15.9 | Lint + perf rules (PERF/B/C90/SIM/UP/PIE) |
| `vulture` | 2.16 | Dead code + unused imports/vars |
| `radon` | 6.0.1 | Cyclomatic complexity, maintainability index, raw LOC, Halstead |

Artifact root: [tmp/python_profile_20260425/](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/)

---

## 1. Workload Harness

Built a Python-only harness at
[tmp/profile_python_full.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/profile_python_full.py)
that exercises the load-bearing Python paths without depending on the
SHM/Rust broker handshake (which currently hangs — see §7). All numbers
below come from this harness.

| Workload | Description | Wall (s) | Description of cost |
|----------|-------------|----------|---------------------|
| `replay`        | `ReplayBuffer.add_sparse` × 240 + `build_dataloader` + iterate 256 batches | 4.07 | Dominated by first-call `import torch` inside `collate_replay_samples` |
| `qipc`          | `pack_qipc_arena_eval_req` × 4 000                                          | 0.90 | Small per-field `struct.pack` + `bytearray.extend` |
| `encoder`       | `GomokuEncoder.encode` + `heuristic_prior` × 4 000                          | 6.16 | Pure-Python O(n²×4) loop with numpy element access |
| `model_fwd`     | AlphaZeroNet inference, batch 32 × 50                                       | 0.39 | Torch fwd                                                |
| `model_train`   | AlphaZeroNet train step, batch 32 × 30                                      | 5.07 | Torch backward + dynamo lazy-init dominate first run |
| `shm_ring`      | `ShmRingBuffer.p2r_try_write` + `r2p_try_read_meta` × 4 000                 | 0.22 | Slot-state byte ops + per-call `struct.pack_into` |
| `plan`          | `plan_selfplay_runner_chunk` × 20 000                                       | 1.39 | 420k `builtins.max` calls + `os.cpu_count()` per call |

Per-tool reports per workload sit under
[tmp/python_profile_20260425/](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/):
`cprofile/`, `pyinstrument/`, `scalene/`, `memray/`, `viztracer/`,
`line_profiler/`, `pyspy/`, `importtime/`.

Cross-tool agreement on the hot leaves was strong: cProfile, scalene,
pyinstrument and line_profiler all attribute >90 % of representative
encoder time to `heuristic_prior`, and >85 % of `qipc` time to
`pack_qipc_arena_eval_req`. That convergence is the basis for the
prioritized list in §6.

---

## 2. Hotspot 1 — `encoders.GomokuEncoder.heuristic_prior`

This is the single largest Python-side hotspot found in this round.

### 2.1 Numbers
- cProfile: `5.472 s tottime / 5.588 s cumtime / 4 000 calls / 1.37 ms per call` ([encoder_tot.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/cprofile/encoder_tot.txt))
- scalene: 91.3 % of process CPU on the `heuristic_prior` line ([encoder.json](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/scalene/encoder.json))
- line_profiler: hot inner lines at [encoders.py:163-168](quartz/encoders.py#L163-L168) (`while 0 <= nr < bs and 0 <= nc < bs:` and `if board_flat[nr * bs + nc] == side:`) account for 33 % of function time on their own ([encoder_qipc.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/line_profiler/encoder_qipc.txt))

### 2.2 Why it is slow
Pure-Python double `for`-loop walking a numpy array element-by-element,
with `board_flat[i]` element access bouncing through the numpy scalar
boxing path. For a 7×7 board that is 49 outer × ~16 inner-direction
iterations each calling 1–2 numpy element loads.

### 2.3 What to do
- Keep the existing semantics, but rewrite `heuristic_prior` against a
  reshaped `board_2d = board_flat.reshape(bs, bs)` and use
  `np.pad(board_2d, 1, constant_values=0)` plus four directional
  slices (`pad[:-2, 1:-1]`, `pad[1:-1, 2:]`, …) to compute neighbor
  counts in 4 vectorized passes. That alone should land 10–30× even on
  the small 7×7 board.
- For the threat-pattern scan, the cheapest correctness-preserving
  rewrite is per-direction `np.lib.stride_tricks.sliding_window_view`
  over the padded board, yielding `(bs, bs, 2*wl-1)` windows where a
  single boolean comparison + `np.cumsum` gives the run-length tally.
- The function is already exposed only from `GomokuEncoder` and is
  called once per root, so a numba/cython rewrite is also viable but
  unnecessary if the numpy version lands the win.

### 2.4 Evidence files
- [tmp/python_profile_20260425/cprofile/encoder_tot.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/cprofile/encoder_tot.txt)
- [tmp/python_profile_20260425/line_profiler/encoder_qipc.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/line_profiler/encoder_qipc.txt)
- [tmp/python_profile_20260425/pyinstrument/encoder.html](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/pyinstrument/encoder.html)

---

## 3. Hotspot 2 — `encoders.GomokuEncoder.encode`

Less dramatic, same disease.

### 3.1 Numbers
- cProfile: `0.382 s / 4 000 calls / 95 µs per call`
- line_profiler: 28 % of function time on `if board_flat[i] == player:` and 17 % on `elif board_flat[i] != 0:` — both element-wise numpy comparisons inside a pure-Python loop ([encoders.py:111-116](quartz/encoders.py#L111-L116))

### 3.2 What to do
Two-line vectorized rewrite:

```python
def encode(self, board_flat, player):
    bs = self._board_size
    enc = np.zeros((17, bs, bs), dtype=np.float32)
    arr = np.asarray(board_flat).reshape(bs, bs)
    enc[0] = (arr == player).astype(np.float32)
    enc[1] = ((arr != 0) & (arr != player)).astype(np.float32)
    if player == 1:
        enc[16] = 1.0
    return enc
```

Expected speedup on 7×7 boards: 10–20×; on 15×15 boards much larger.

`encode` is on the per-position hot path, so this matters more in
aggregate than `heuristic_prior`'s per-root usage even though
`heuristic_prior` looks more dramatic on the per-call axis.

---

## 4. Hotspot 3 — `qipc.pack_qipc_arena_eval_req`

The arena-eval request packer is the single largest qipc-side Python
hot path now that the SHM ring read/write is direct memoryview/struct.

### 4.1 Numbers
- cProfile: `0.587 s tottime / 0.865 s cumtime / 4 000 calls / 147 µs per call` ([qipc_tot.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/cprofile/qipc_tot.txt))
- 144 000 `struct.pack` calls (= 36 per request) and 192 000
  `bytearray.extend` calls (= 48 per request) for a 4-session payload
- line_profiler: per-session block at [qipc.py:1009-1064](quartz/qipc.py#L1009-L1064) is ~45 % of function time, and the per-field `_pack_opt_str / _pack_opt_bool / _pack_opt_u64` stanza at lines 986-1006 is another ~25 %
- radon `cc = 40 (E)` for this single function

### 4.2 What to do
- Replace the chain of small `struct.pack(...)` + `bytearray.extend(...)`
  calls with one `struct.Struct(...)`-cached compound packer per
  session shape. The inner per-session block already uses a fixed
  `<IIQIdB` layout — bind that struct once at module import:

  ```python
  _SESSION_HEADER = struct.Struct("<IIQIdB")
  ```

  and emit `payload += _SESSION_HEADER.pack(...)` instead of the
  inline `struct.pack(...)`.
- Hoist the closures `_pack_opt_str`, `_pack_opt_bool`, `_pack_opt_u64`
  out of the hot function — they are re-defined on every call (visible
  as the 153 ns per-line cost on lines 967, 972, 978 in line_profiler).
- The repeated `int(sess.get(...) or 0)` pattern boxes/unboxes ints
  twice; a single `sess.get("...", 0)` with explicit `int()` is
  cheaper.
- Defaults like `int(options.get("min_visits", 50) or 50)` cost 220 ns
  per field in line_profiler — most of that is the `dict.get`. A
  `try/except KeyError`-style or pre-resolved local is materially
  cheaper in the hot loop.

A conservative refactor along these lines should cut per-request cost
by roughly 30–50 % without changing the on-the-wire layout. That is
particularly worth it because this packer runs once per arena/eval
request, which on the SHM path is per slot per match move.

### 4.3 Same-shape sibling: `unpack_qipc_arena_eval_resp`
- cc = 15 (C), structurally identical pattern (per-field
  `struct.unpack_from`). Same fix applies symmetrically. Not
  dominantly hot in this audit because the harness only packs, but
  worth bundling into the same patch.

---

## 5. Hotspot 4 — `selfplay_runtime.plan_selfplay_runner_chunk`

Not the single biggest cost, but the most micro-fixable.

### 5.1 Numbers
- cProfile: `0.276 s tottime / 1.346 s cumtime / 20 000 calls / 70 µs per call` ([plan_tot.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/cprofile/plan_tot.txt))
- 420 000 `builtins.max` calls (= 21 per call), 80 000 `builtins.min`
  calls (= 4 per call), 20 000 `os.cpu_count()` calls (107 ms total — 5.4 µs per call), 100 000 generator expressions

### 5.2 What to do
- Cache `os.cpu_count()` once at module load:
  `_LOGICAL_CPU_COUNT = max(1, os.cpu_count() or 1)`. The 5.4 µs per
  call is 7.7 % of plan-fn cost on its own, and the value never
  changes within a process.
- Collapse the nested `max(min(max(...), ...), ...)` chains into local
  variables — a careful rewrite at [selfplay_runtime.py:108-119](quartz/selfplay_runtime.py#L108-L119) drops most of the redundant
  comparisons. The function is ~12 lines of business logic but
  currently calls `max` 21 times per invocation.
- `estimate_selfplay_positions_per_game` at [selfplay_runtime.py:58](quartz/selfplay_runtime.py#L58) computes
  `sum(int(c.get("games", 0) or 0) for c in recent_chunks)` twice in
  practice (once for `games`, once for `positions`). One pass is
  enough.
- This function is called once per self-play scheduling tick — the
  absolute amount of CPU saved is small in production, but the
  rewrite is local, mechanical, and removes a lot of avoidable work
  per tick.

---

## 6. Hotspot 5 — `replay.collate_replay_samples`

Same shape as the 2026-04-20 audit found. Numbers reconfirmed.

### 6.1 Numbers
- line_profiler: 200-call total wall is 1.33 s, of which **98.4 % is
  the very first `_torch_module()` call inside the function** (it
  triggers `import torch` for line_profiler's process). After that
  first call, real per-call cost is ~110 µs dominated by
  `policies_np[row, ex.policy.idx] = ex.policy.val` (sparse → dense
  scatter, 0.6 % of total) and per-row `np.asarray` copies (0.4 %)
- This means: there is no real per-call torch overhead; the previous
  audit's caching of `_TORCH_MODULE` is doing its job.

### 6.2 What to do
- Leave the per-call structure alone — it is not hot.
- The dense scatter `policies_np[row, ex.policy.idx] = ex.policy.val`
  is correct and uses numpy's CSR-equivalent fast path. A further
  win is only available by stacking all `idx` arrays into a single
  CSR pair and calling
  `policies_np[row_arr, col_arr] = val_arr` once per batch. That is
  worth doing only if the replay path becomes training-visible — it
  did not in this audit.

### 6.3 Other replay observations
- `ReplayBuffer.build_dataloader` at [replay.py:278](quartz/replay.py#L278) snapshots `tuple(self.buf)` under a lock,
  then materializes `[snapshot[i] for i in all_indices]` in pure
  Python. For very large `total_needed` this is wasteful; a
  numpy-indexed gather over an object array is an option, but with
  current batch sizes this is not a hot path.
- `ReplayMetrics.search_summary` (radon `cc = 47, F`) is dead-cold in
  the harness but worth a structural rewrite from a complexity-debt
  perspective.

---

## 7. Regression — Existing Orchestrator Harness Hangs

The existing harness
[tmp/profile_python_orchestrator.py](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/profile_python_orchestrator.py)
that produced the 2026-04-20 baseline (`selfplay_elapsed_s ≈ 0.14 s`)
now hangs indefinitely. Verified twice:

- `--iters 8 --n-games 2 --parallel 2 --batch-size 2`: still alive at
  6 min, killed
- Default args: same

Process state at hang:
- Python: `S` (sleeping), `wchan = hrtimer_nanosleep` — i.e. inside a
  `time.sleep()` backoff loop
- `mcts_demo --server` child: `S` (sleeping), `wchan = futex_do_wait`
  — waiting on the SHM coordination pair

py-spy attach failed (`ptrace_scope = 1`, py-spy must be elevated or
parent the process). The harness was the parent of the python
process, not of py-spy, so py-spy could not attach.

This is consistent with a regression in the SHM/broker path. Recent
relevant commits:

| Commit | Touched | Notes |
|--------|---------|-------|
| `7274442` "backup" | `qipc.py +280`, `runtime_support.py +117`, `selfplay_runtime.py +216` | Adds `pack_qipc_arena_eval_req`, arena/eval QIPC frame kinds, ring-side hooks |
| `bc1a7aa` "Fix self-play bootstrap diagnostics" | `selfplay_runtime.py +168 -36` | Reworks `wait_for_worker_progress` stall handling |
| `29c452c` "Apply audit minimal-patch plan (10 patches)" | `selfplay_runtime.py +28` | Replay-fill ceiling + parallel/batch caps |

The 2026-04-20 harness predates `7274442`. The most likely failure
mode is a missing wakeup on the new arena-eval frame path: the Rust
server is parked on the futex, the Python side is parked in
`time.sleep()`, neither is signalling the other.

This audit explicitly does **not** attempt to fix that — it scoped
itself to Python profiling with a workload harness that bypasses the
broker. But it does mean the existing 2026-04-20 baseline number
(`selfplay_elapsed_s ≈ 0.14 s`) is currently unreproducible, and the
regression should be triaged before any next round of
broker-attribution work.

Tracker: [tmp/python_profile_20260425/orchestrator_hang_notes.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/orchestrator_hang_notes.txt) (to be filed).

---

## 8. Import Time Audit

Per-module cold-import wall (one shot per module) from `python -X importtime`:

| Module | Total (ms) | Dominant cost |
|--------|------------|---------------|
| `quartz.alphazero_train`        | **1 137.9** | `torch._C` 404 ms + `torch._meta_registrations` 217 ms (cum 1 008 ms in `torch` alone) |
| `quartz.torch_training_runtime` | **1 187.5** | Same — torch import dominates |
| `quartz.evaluator_runtime`      |   150.3 | numpy + `quartz.evaluation` + `quartz.selfplay_runtime` (no torch unless torch backend is loaded) |
| `quartz.runtime_support`        |   144.1 | numpy + `quartz.selfplay_runtime` + `quartz.evaluation` |
| `quartz.replay`                 |   113.9 | numpy only |
| `quartz.qipc`                   |   111.2 | numpy only |
| `quartz.selfplay_runtime`       |   110.0 | numpy only |
| `quartz.cli_main`               |    97.5 | numpy only |
| `quartz` (`__init__`)           |    86.6 | numpy only |

Conclusions:
- Local Python imports are uniformly cheap (<160 ms each). No tree
  reshape work is justified.
- The `>1 s` ceiling on training-entry imports is **entirely** torch.
  The only way to win here is to defer or skip torch import for paths
  that don't need it — already done for `qipc`, `cli_main`, `replay`,
  `selfplay_runtime`, `evaluator_runtime`, `runtime_support`. This is
  in good shape; do not regress it.
- Reports: [tmp/python_profile_20260425/importtime/](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/importtime/), summarized in [_top_self.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/importtime/_top_self.txt).

---

## 9. Memory Profile (memray + tracemalloc)

### 9.1 Workload heap totals

| Workload | Total allocated | Peak | Top allocation site |
|----------|-----------------|------|---------------------|
| `model_train` | 7.21 GB        | 4.52 GB | `_engine_run_backward` (5.78 GB) — autograd graph |
| `replay`      | 113 MB+        | —      | torch+sympy import (`_create_fn` dataclasses 105 MB) |
| `encoder`     | 91.6 MB total | 44 MB  | numpy `blas_fpe_check` 33.6 MB (one-shot startup) |
| `qipc`        | 69.8 MB total | 43.7 MB | numpy `blas_fpe_check` 33.6 MB (one-shot startup) |
| `plan`        | 69.8 MB total | 43.7 MB | numpy `blas_fpe_check` 33.6 MB (one-shot startup) |
| `shm_ring`    | small         | small  | qipc + numpy startup |

Findings:
- The training-step path allocates **7.2 GB across 1.8 M allocations**
  per 30-step run. The histogram is dominated by 84 B–7 kB
  allocations: characteristic of fine-grained autograd tensor + node
  churn, not Python-object churn. This is expected; the leverage is
  in `torch.compile` / AOT autograd / smaller batch shapes, not
  Python.
- The non-training workloads' "large allocator sites" are all numpy
  startup overhead (`blas_fpe_check` 33.6 MB) and dataclass
  construction during torch's lazy init (`_create_fn`). This is
  one-shot per process and not a steady-state cost.
- tracemalloc per-workload top sites ([cprofile/*_tracemalloc.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/cprofile/)):
  - `qipc` workload: top live alloc site is `qipc.py:965`
    (`options = dict(search_options or {})`) at 9.3 KB across 79
    allocations — this is the "build a fresh `dict()` per call" cost
    flagged in §4. Fixable.
  - `encoder` workload: 1.3 KB total live across all 4 000 calls — the
    function does not leak; it is CPU-bound, not memory-bound.
  - `replay` workload: 28 MB live, 99 % of which is module-load
    bytecode caching from importing torch. The replay buffer itself
    contributes a few hundred KB.

---

## 10. Static Audit

### 10.1 Cyclomatic complexity (radon)

Average over `quartz/` is **D (21.85)** across 86 functions/methods.
Top offenders by cc:

| File | Function | cc | rank |
|------|----------|----|------|
| `cli_main.py`         | `run_training_main`              | 106 | F |
| `selfplay_runtime.py` | `selfplay_rust_nn_batched`       | 74  | F |
| `cli_main.py`         | `prepare_training_context`       | 71  | F |
| `selfplay_runtime.py` | `arena_rust_nn_impl`             | 64  | F |
| `evaluator_runtime.py`| `shm_eval_loop`                  | 58  | F |
| `game_adapters.py`    | `GoGameAdapter._score`           | 49  | F |
| `evaluator_runtime.py`| `_run_shared_eval_matches`       | 47  | F |
| `replay.py`           | `ReplayMetrics.search_summary`   | 47  | F |
| `autotune_runtime.py` | `plan_online_runtime_overrides`  | 44  | F |
| `qipc.py`             | `pack_qipc_arena_eval_req`       | 40  | E |
| `selfplay_runtime.py` | `_exchange_search_request`       | 36  | E |
| `evaluation.py`       | `play_match_tally_batched`       | 35  | E |
| `eval_runtime.py`     | `run_batched_eval_groups`        | 34  | E |
| `arena_runtime.py`    | `arena_compare`                  | 35  | E |
| `train_loop.py`       | `generate_training_plots`        | 34  | E |

Maintainability index of these files lands at C (0.00) — i.e. they
are already saturated on radon's scale. None of these is on the
inner-most CPU hot path **except** `pack_qipc_arena_eval_req` (§4)
and `shm_eval_loop` (which we cannot directly profile in this audit
because of the orchestrator hang). The rest are coordination-heavy
top-level functions and the leverage on them is correctness +
maintainability, not throughput.

### 10.2 Dead code (vulture, ≥70 % confidence)

20 confident hits ([reports/vulture.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/static/vulture.txt) — to be filed). Highlights:

- `quartz/qipc.py:6` unused `import ctypes` (90 %)
- `quartz/alphazero_train.py:19` unused `import signal` (90 %)
- `quartz/alphazero_train.py:119,236,319,431,1374` repeated unused
  imports of `_eval_request_cache_key_impl`,
  `_run_autotune_benchmark_impl`, `EvalEngine`, `EvalRandomEngine`,
  `GameAdapter` (90 %)
- `quartz/torch_training_runtime.py:13` unused
  `_run_autotune_benchmark_impl` (90 %)
- `quartz/alphazero_train.py:720,728` unused loop variables `kw`,
  `leave` (100 %)
- 100 %-confidence dead local variables in `evaluator_runtime.py`,
  `jax_training_runtime.py`, `evaluation.py`, `play_gui.py`

These are mostly cosmetic, but unused module-level imports cost real
import-time work because Python still loads them. The
`alphazero_train.py` cluster is the only one large enough to matter
on cold-start.

### 10.3 ruff (PERF + B + C90 + SIM + UP + PIE)

434 issues in `quartz/`, 256 in `scripts/`. Top categories:

| Rule | Count | Class | Notes |
|------|-------|-------|-------|
| `E702` semicolon-stmts | 178 | style | low value |
| `F401` unused-import   | 120 | hygiene + import-time | overlaps vulture |
| `E701` colon-stmts     | 64  | style | low value |
| `B023` loop-variable-binding | **30** | **bug-shaped** | closure capture in loops; needs review |
| `B905` zip-without-strict | 21 | hygiene | safe to add `strict=False` |
| `PERF203` try-except-in-loop | **13** | **perf** | move except outside loop |
| `F821` undefined-name | **9** | **runtime hazard** | needs review |
| `PERF401` manual-listcomp | 8 | perf | mechanical |
| `F841` unused-variable | 7 | hygiene |
| `B904` raise-without-from | 5 | error chain |
| `B007` unused-loop-var | 3 | hygiene |
| `PERF403` manual-dictcomp | 2 | perf |
| `F822` undefined-export | 2 | bug-shaped |
| `SIM105` suppressible-exception | 44 | cosmetic |

The four meaningful categories are:

1. **`B023` (30 hits)** — function captures loop variable. These are
   shaped like real bugs in lambdas/closures defined inside `for`
   loops. Worth a focused PR to walk every hit and decide whether the
   intent was to bind-by-value (`lambda x=x: ...`) or accept the
   reference. **Owner: a single engineer-day.**
2. **`F821` (9 hits)** + **`F822` (2 hits)** — undefined names /
   exports. These are runtime crash hazards and should be triaged.
3. **`PERF203` (13 hits)** — `try / except` inside hot loops.
   Python's `try` is cheap, `except` is not. Hoist the handler outside
   when the body is hot.
4. **`F401` / vulture overlap (120+20)** — module-level dead imports.
   Worth a single mechanical PR.

Reports:
- [tmp/python_profile_20260425/static/ruff_quartz.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/static/ruff_quartz.txt) (to be filed)
- [tmp/python_profile_20260425/static/vulture.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/static/vulture.txt) (to be filed)
- [tmp/python_profile_20260425/static/radon_cc.txt](/home/cosmosapjw/Dropbox/personal_projects/quartz/tmp/python_profile_20260425/static/radon_cc.txt) (to be filed)

### 10.4 Halstead volume (radon hal)

Top by computed volume / "estimated bug" indicator:
- `selfplay_runtime.py`: volume 21 594, effort 314 236, "bugs" 7.20
- `evaluator_runtime.py`: volume 5 363, effort 78 528, "bugs" 1.79
- `runtime_support.py`: volume 1 488, effort 14 735, "bugs" 0.50

These align with radon cc — `selfplay_runtime.py` is by far the
biggest, most error-prone module. Worth a structural split (broker
loop vs runner planning vs game adapters) at some future point, but
nothing in this audit forces that today.

---

## 11. Prioritized Action List

Ordered by `(measured win) × (mechanical-ness)`:

### P0 — measurable wall wins, low-risk patches
1. **Vectorize `GomokuEncoder.encode`** ([encoders.py:107](quartz/encoders.py#L107)).
   Two-line numpy rewrite. Expected 10–20× per call; this is on the
   per-position hot path and matters in aggregate. (§3)
2. **Vectorize `GomokuEncoder.heuristic_prior`** ([encoders.py:132](quartz/encoders.py#L132)).
   Padding + 4-direction slice rewrite, then per-direction
   `sliding_window_view` for threat counting. Expected 10–30× per
   call; called once per root, but at 1.37 ms/call this is the single
   most expensive Python function in the workload set. (§2)

### P1 — modest wins, mechanical patches
3. **Cache `os.cpu_count()` and collapse `max/min` chains in
   `plan_selfplay_runner_chunk`** ([selfplay_runtime.py:93](quartz/selfplay_runtime.py#L93)).
   Single pass through `recent_chunks` instead of two. Expected
   ~30 % drop on this function; small absolute, but mechanical. (§5)
4. **Tighten `pack_qipc_arena_eval_req`** ([qipc.py:959](quartz/qipc.py#L959)):
   - hoist closure helpers out of the function
   - bind `_SESSION_HEADER = struct.Struct("<IIQIdB")` once at module
     scope, similarly for the per-options compound layout
   - avoid `dict(search_options or {})` per call when caller passes a
     dict literally
   Expected 30–50 % drop on per-request packing cost. Same shape
   patch applies to `unpack_qipc_arena_eval_resp`. (§4)
5. **Mechanical lint cleanup**:
   - remove the 120 `F401` unused imports + 20 vulture confirms
   - fix the 30 `B023` closure captures (real bug class)
   - move the 13 `PERF203` try/except blocks outside their hot loops
   - investigate the 9 `F821` undefined names

### P2 — investigate, do not auto-patch
6. **Triage the orchestrator harness regression** (§7). The 2026-04-20
   baseline is currently unreproducible because the SHM/broker path
   hangs after the `7274442 → 29c452c` series. This blocks future
   broker attribution audits. Bisect range:
   `9232342..bc1a7aa` on the `quartz/qipc.py +
   quartz/selfplay_runtime.py + quartz/runtime_support.py` triple.
7. **Consider splitting `selfplay_runtime.py`** (Halstead volume
   21 594, cc-rank C aggregate, two F-rank functions). Mechanical
   refactor; not a perf win. Defer.

### Not worth chasing
- Further `_torch_module()` micro-tuning in
  `collate_replay_samples` — cached, real per-call cost is ~110 µs and
  not a bottleneck. (§6)
- Local module import cleanup beyond the F401 sweep — torch dominates
  the cold-start cost by 10× over everything local. (§8)
- Per-allocation reduction in `model_train` — the 7.2 GB churn is
  inside autograd, not Python. (§9)
- Further SHM polling/sleep tuning in Python — the 2026-04-20 audit
  already concluded this needs Rust-side wakeup, not Python edits,
  and the §7 regression confirms naive changes are easy to break.

---

## 12. Reproducibility

```sh
# all profiles
venv/bin/python tmp/profile_python_full.py --with-tracemalloc \
    --out tmp/python_profile_20260425/cprofile

# single tool, single workload examples:
venv/bin/python -m pyinstrument -o report.html -r html \
    tmp/profile_python_full.py --workloads encoder

venv/bin/python -m scalene run -o scalene.json \
    tmp/profile_python_full.py --- --workloads qipc

venv/bin/python -m memray run -q -o trace.bin \
    tmp/profile_python_full.py --workloads model_train
venv/bin/python -m memray stats trace.bin

venv/bin/python -X importtime -c 'import quartz.alphazero_train' 2> imp.txt

venv/bin/python -m ruff check quartz/ --select PERF,B,C90 --statistics
venv/bin/python -m vulture quartz/ --min-confidence 70
venv/bin/python -m radon cc quartz/ -s -a -nc
venv/bin/python -m radon hal quartz/
```

---

## 13. Bottom Line

The Python orchestrator surface is in roughly the same shape the
2026-04-20 audit left it. The previous "no more obvious Python wins"
verdict is **partially superseded** by this round, because:

- A multi-tool sweep (cProfile + scalene + line_profiler + pyinstrument)
  unanimously agrees that `GomokuEncoder.encode` and
  `GomokuEncoder.heuristic_prior` are pure-Python loops over numpy
  arrays. These two functions are easy 10–30× wins and are still
  measurable in the workload harness.
- `qipc.pack_qipc_arena_eval_req` is the one remaining Python-side
  request-path packer that is still doing per-field `struct.pack`.
  A struct-cached rewrite is mechanical and worth ~30–50 %.
- `plan_selfplay_runner_chunk` does 21 `max()` calls and one
  `os.cpu_count()` per invocation. Trivial cleanup.

Beyond those, the previous audit's conclusion still stands: the
remaining big costs are torch import startup (unavoidable on training
paths), torch autograd allocations (inside torch), and the SHM
polling/sleep architecture (needs Rust-side wakeup, and is currently
broken — see §7).
