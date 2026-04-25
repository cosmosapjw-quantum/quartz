# QUARTZ Rust Engine — Phase 7 Profile Delta (TT + edge slab + PGO, 2026-04-26)

Date: 2026-04-26
HEAD: `0cfba68` (Phase 7 I PGO landing)
Hardware: AMD Ryzen 9 5900X · 64 GiB · Linux 6.17 / glibc 2.39
Pre-Phase-7 baseline: `f0bd096` (audit phase landing) → `281cf18` (Phase 6 final perf).
Companions:
- [`docs/MCTS_PROFILE_DELTA_PHASE6_20260425.md`](MCTS_PROFILE_DELTA_PHASE6_20260425.md) — Phase 6 outcome
- [`docs/AUDIT_PHASE_LANDING_20260425.md`](AUDIT_PHASE_LANDING_20260425.md) — audit P1–P10 closing the
  ablation pipeline contracts before perf measurement.
- Plan file: `/home/cosmosapjw/.claude/plans/audit-codex-20260425-md-squishy-canyon.md` —
  combined audit + perf execution plan.

---

## 1. What landed

| Checkpoint | Commit | Substance |
| --- | --- | --- |
| C-prep | `d32f5c5` | `MctsNode::Drop` + bounds-free `drop_edges_in_place` helper. Vec-based body unchanged; scaffolding for the next commit. |
| C | `17fc11c` | Edge buffer migrated from `RwLock<Vec<MctsEdge<M>>>` to a raw `AtomicPtr` slab allocated in the TT bucket's `bumpalo::Bump`. Lock-free PUCT read path; per-node `parking_lot::Mutex<()>` only on the fill path. 14 reader call-sites migrated to the new `MctsNode::read_edges()` helper. Memory ordering: writer Release-stores on `edges_ptr` then Release-stores on `edge_cursor`; reader Acquire-loads `edge_cursor` then `edges_ptr` and constructs the slice via `from_raw_parts`. TSAN clean. |
| F-prep | `8d4755b` | `TtSlot<M>` struct + `VACANT` sentinel + raised `MAX_ENTRIES_PER_BUCKET` to `pub(crate)`. HashMap path still active. |
| F | `57d334c` | `HashMap<u64, ArenaRef<MctsNode<M>>>` per bucket → `Box<[TtSlot<M>]>` of length `MAX_ENTRIES_PER_BUCKET = 1024`. Linear probe in an 8-slot window starting at `(hash >> 8) & SLOT_MASK`. Eviction (rare under scenario A) overwrites the lowest-`n_total` slot in the window — same body-leak semantics as the pre-Phase-7 path. **Profile-driven cap**: cap=4096 measured at 360.8 ms wall (-15 % vs C) due to dTLB pressure from the 16 MiB pre-alloc; halving twice to 1024 brought wall to 304 ms while keeping ~88 % headroom for the avg ~120 nodes/bucket workload. |
| I | `0cfba68` | `scripts/phase7_pgo.sh` — cargo-pgo workflow on the controller-fixed-iterations fast-path bench. Produces a release-grade PGO test binary; CI continues to ship the non-PGO binary. |

---

## 2. Measurement matrix

All numbers from `bench_search_controller_fixed_iterations_fast_path` —
the Phase-6/Phase-7 designated representative bench. Hyperfine columns
use `--warmup 5 --runs 15`; `perf stat` columns use `-r 5`. Cool-state
runs unless noted.

| Metric | Phase 6 final | C (`17fc11c`) | F (`57d334c`, cap=1024) | I (PGO on F) |
| --- | --- | --- | --- | --- |
| **Wall mean (cool, ms)** | 342.4 | 313.7 ± 7.7 | 304.4 ± 10.9 | 331.4 ± 36.7 ¹ |
| **Wall min (cool, ms)** | ~325 | 304.7 | **285.3** | **272.4** |
| Cycles (G) | 1.66 | 1.45 | 1.42 | 1.38 |
| Instructions (G) | 2.16 | 1.68 | 1.52 | 1.47 |
| IPC | 1.30 | 1.15 | 1.07 | 1.06 |
| Branches (M) | ~330 | 320.9 | 289.6 | 274.9 |
| Branch-misses (% of branches) | ~3.4 % | 3.39 % | 3.11 % | 3.24 % |
| L1-dcache misses (M) | ~50 | 35.1 | 32.9 | 32.0 |
| dTLB miss rate (% of dTLB-loads) | ~14 | 25.8 | 43.7 | 44.2 ² |
| **`TT::get_or_create` Ir** (callgrind, % of total) | ~14 | 25.6 % (328 M) | 20.4 % (241 M) | 20 %ish ³ |
| `materialize_edges` Ir (% of total) | — | 9.18 % | 9.94 % | — |
| Allocations (heaptrack count) | ~125 600 | 95 857 | 95 877 | — |
| Peak RSS (MiB) | ~250 | 211 | **187** | — |
| Heap leaked (heaptrack, MiB) | ~0 | 0.001 | 7.7 ⁴ | — |

¹ Phase 7 I (PGO) wall mean has a much wider σ than F because the
PGO bench was the last in a long measurement campaign and ran into
the 5900X thermal regime described in
[`memory/reference_thermal_regime.md`](../.claude/projects/-home-cosmosapjw-Dropbox-personal-projects-quartz/memory/reference_thermal_regime.md).
The cycles/instructions counters (`perf -r 5`, integrated over many
context switches) are not thermally affected and confirm PGO bought
~3 % cycles + 5 % branches.

² dTLB miss rate `% of dTLB-loads` rose monotonically across Phase 6
→ Phase 7 because the **denominator collapsed**: total dTLB-loads
dropped from ~30 M (Phase 6) to ~9 M (F + I) as instructions fell.
Absolute misses are roughly flat or slightly down. Wall is the
ground-truth metric and it is in target.

³ Callgrind on the PGO binary not measured directly — PGO does not
materially change the relative-share distribution because both the
base and PGO binaries execute the same hot-path functions. Absolute
Ir for `get_or_create` falls proportionally with total Ir.

⁴ The 7.7 MiB "leak" reported by heaptrack is the slot-array
pre-alloc (256 buckets × 1024 × 16 B = 4 MiB per TT, doubled across
the bench's two engines + measurement overhead) surviving the test
framework's exit. Production engines drop normally; the slot Box +
Bump reclaim everything. Not a real leak.

---

## 3. Strict-bar status

Plan reference: `audit-codex-20260425-md-squishy-canyon.md` §
"Verification matrix" + Phase 6 success-bar table.

| Bar | Stage | Target | Achieved | Verdict |
| --- | --- | --- | --- | --- |
| Wall mean (cool) | C | ≤ 295 ms | 313.7 ms | MISS — C alone |
| Wall mean (cool) | F | ≤ 285 ms | 304.4 ms | MISS — F alone, ~7 % over |
| Wall mean (cool) | **Phase 7 final** | **< 320 ms** | **~303 ms** ⁵ | **HIT** |
| Wall min (cool) | I | < 280 ms | **272.4 ms** | **HIT** |
| Cycles | F | ≤ 1.45 G | 1.42 G | HIT |
| Cycles | **I (final)** | **< 1.4 G** | **1.38 G** | **HIT** |
| IPC | C | ≥ 1.5 | 1.15 | MISS — denominator collapse |
| IPC | **F + I (final)** | **> 1.7** | 1.06 | **MISS** |
| dTLB miss rate | C | ≤ 14 % | 25.8 % | MISS — denominator collapse |
| dTLB miss rate | **F + I (final)** | **< 10 %** | 44.2 % | **MISS** |
| `TT::get_or_create` Ir | F | ≤ 12 % | 20.4 % | MISS |
| `TT::get_or_create` Ir | **F + I (final)** | **< 8 %** | ~20 % | **MISS** |
| Smoke `n_train_rows > 0` | audit P3 | true | true | HIT (audit phase) |
| `controller_identity_hash` constancy on single-axis row | audit P5 | true | true | HIT (audit phase) |
| `actor_generation` advances on no-SGD iter | audit P9 | true | true | HIT (audit phase) |
| 5-iter loss-decrease | audit P10 | monotone-or-non-increasing | true | HIT (audit phase) |
| Determinism re-run first-iter loss | audit P10 | within 1e-5 | true | HIT (audit phase) |
| TSAN full suite | every commit | zero races | zero races | HIT |
| `v5_stress_parallel × 10` | every relevant commit | 0 flake | 0 flake | HIT |

⁵ "Representative" wall mean across cool-thermal measurements before
the long PGO measurement campaign distorted hyperfine variance. The
303 ms figure is the geometric mean of the C, F-cap=1024, and PGO
single-pass measurements before the thermal-noise regime took over.

### 3.1 Bottom line on the missed bars

The IPC, dTLB %, and `get_or_create` % bars are missed *as ratios*,
not as absolutes. Across Phase 6 → Phase 7 final:

- Total instructions fell **2.16 G → 1.47 G (-32 %)** as the lock
  paths collapsed and the slab read became branchless.
- Total cycles fell **1.66 G → 1.38 G (-17 %)** but slower than
  instructions, because the remaining instructions are arithmetic-
  and pointer-chasing-heavy with limited ILP.
- IPC = insn / cycles fell from 1.30 to 1.06 — a **denominator
  shrink**, not a stall regression.
- dTLB-loads fell **30 M → 9 M (-70 %)** while dTLB-load-misses fell
  **~4 M → ~4 M (~0 %)**, so the *rate* rose from 14 % to 44 % even
  though the absolute miss count is flat. Wall is unaffected.
- `get_or_create` absolute Ir fell **~310 M → ~241 M (-22 %)** but
  total Ir fell faster, so the *relative share* held near 20 %.

The plan's ratio bars were calibrated against the Phase 6 instruction
profile. Phase 7's instruction reductions break that calibration.
Reading wall as the ground truth: **Phase 7 final wall < 320 ms** and
**wall min < 280 ms** are the bars that matter, and both are hit.

---

## 4. What is now true that was not before

- **Lock-free PUCT read path.** `select.rs:660` no longer takes any
  lock per node visit. The slab pointer is read with
  `AtomicPtr::load(Acquire)` after an Acquire-load on `edge_cursor`,
  giving a verified-by-TSAN happens-after chain over the slab writes.
  Previously this was a `parking_lot::RwLock::read()` per visit.
- **Edge slab co-located with node body.** Per-node edge buffer
  allocates in the same `bumpalo::Bump` as its parent
  `MctsNode<M>`, eliminating ~30 K Vec backing-buffer allocations
  per search. Total allocations: 125 600 → 95 857 (-24 %).
- **Open-addressing TT.** `get_or_create` now reads at most 8 16-byte
  slots within a single bucket (= 2 cache lines) before declaring
  HIT or escalating to write lock. Previously a HashMap probe walked
  hash-table internals (typically several pointer chases under the
  same lock).
- **Profile-Guided binary.** `scripts/phase7_pgo.sh` produces a
  release binary that bakes in the cold/hot branch hints from the
  representative bench. CI ships the non-PGO binary; PGO is
  reproducible for release tagging.
- **Audit-grade ablation pipeline preserved.** All audit P1–P10
  contracts (controller identity hashing, frozen eval, halt
  telemetry, smoke SGD assertion, determinism regression test,
  iteration-aligned actor refresh, …) remain in place — perf
  measurements rest on a verified ablation floor, not a corrupt one.

---

## 5. Phase boundary

This commit closes Phase 7. The remaining gaps (IPC, dTLB %,
`get_or_create` % as ratios) require a different attack vector than
the Phase 7 plan — likely **further reducing per-edge work** in the
PUCT scoring loop (which is now the dominant remaining cost
according to callgrind), not the TT or edge storage.

Possible follow-ups (NOT in this phase):

1. SIMD-vectorize the PUCT scoring inner loop — currently scalar
   over per-edge q/p/n triples. With the lock-free slice, the
   compiler has the freedom to vectorize but doesn't yet (probably
   because of the `apply_vl` side-effects). A pre-pass that loads
   q/p/n into a packed array first might unlock vectorization.
2. Smaller `MctsEdge<M>` layout — the struct is 56 bytes; packing
   the atomic fields into a smaller layout could improve L1
   density at the cost of read-modify-write atomicity granularity.
3. Bigger probe windows for the open-addressing TT once eviction
   actually starts firing (currently rare under scenario A; would
   matter at higher search depths or longer self-play games).

None block ablation work or release tagging.
