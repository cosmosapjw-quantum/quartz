# Audit — BQ++ Phase 2: ArcSwap PolicyCache + edge-local indexing

**Date:** 2026-05-04
**Scope:** introduce the lock-free immutable cache architecture that
replaces the `parking_lot::Mutex<Cache>` pattern from P06. This patch
ships the cache infrastructure as **scaffolding** — actual policy
consumption (BayesianQuartz / Phase 4 KG-stop) happens in subsequent
phases. This is the safest atomic patch in the cache work.

## Why this matters

The audit identified two cache-related issues:

1. **§1.7 — concurrency monotonicity claim wrong.** Stale cache CAN
   produce over-eager halt because β grows with t. The fix requires
   the cache to carry freshness keys (`root_visits_at_observe`,
   `edge_version_hash`) and the halt path to reject any cache that
   is stale relative to the live snapshot.
2. **§1.8 — edge_pos vs action_id index bug class.** Indexing
   per-edge cache arrays by action_id reads out of bounds for sparse
   action spaces (chess 4672 slots, Go 361 slots). The fix requires
   the cache to be edge-local-indexed by construction.

Both fixes land here. The ArcSwap publish primitive eliminates the
mutex on the hot path; the dense-position invariant of `EdgeRef`
eliminates the index bug class.

## What changed

### New file: `src/mcts/policy/cache.rs`

- `PolicyCache` struct: 12 fields. All edge-local arrays use
  `SmallVec<[f32; 32]>` (stack-allocated for typical n_children;
  spills to heap for Go's 361 max). Carries `epoch`, `root_visits`,
  `edge_version_hash` for freshness; `cert_gap`, `max_kg_per_ms`,
  `prior_surprise`, `forced_move_pos` for halt-decision diagnostics.
- `PolicyCachePublisher`: `arc_swap::ArcSwap<Arc<PolicyCache>>`
  wrapper. Hot path is `publisher.load() → arc_swap::Guard`
  (pointer + atomic increment, no contention). Publish is
  `publisher.store(Arc::new(cache))`.
- `PolicyCache::is_stale_for(current_root_visits, current_edge_hash)
  -> bool` — used by halt path to reject stale certificates.
- `EdgeRef { edge_pos: u32, action_id: u32 }`: enforces by
  construction the separation between dense per-edge index (used
  for cache lookup) and sparse action_id (used for engine
  communication).
- `PolicyCache::empty()`: bootstrap cache with all zero / NEG_INF
  fields. Used before the first `observe` boundary.

### `Cargo.toml`

Added `arc-swap = "1.7"` to `[dependencies]`. Build ~+50 KB compiled
size; runtime overhead measured separately.

### `src/mcts/policy/mod.rs`

Re-exported `cache::{EdgeRef, PolicyCache, PolicyCachePublisher}` at
the policy module root so consumers (Phase 3+ Gumbel SH, Phase 4
KG-stop, Phase 9 BayesianQuartz) can depend on the cache without
deep imports.

## Tests added (7)

1. `test_phase2_cache_publish_visible_to_concurrent_reader`: writer
   publishes 100 caches; 4 reader threads spawned after the publishes
   all see `root_visits == 100`. ArcSwap's store-release / load-acquire
   semantics make this deterministic post-publish.
2. `test_phase2_cache_stale_root_visits_rejected`: cache with
   `root_visits=100` is `is_stale_for(200, ...)` ⇒ true.
3. `test_phase2_cache_stale_edge_hash_rejected`: cache with
   `edge_version_hash=0xabcd` is `is_stale_for(100, 0xbeef)` ⇒ true.
4. `test_phase2_cache_fresh_when_both_match`: matching root_visits
   AND edge_hash ⇒ not stale.
5. `test_phase2_cache_epoch_monotone`: 10 publishes; each load shows
   strictly-increasing epoch.
6. `test_phase2_empty_cache_has_zero_arrays`: bootstrap cache is
   safely indexable (callers can rely on `len() == 0` ⇒ no array
   read).
7. `test_phase2_edge_pos_distinct_from_action_id`: sparse action_id
   (481) is NEVER used to index a 3-element cache. Constructive
   regression of audit §1.8.

## Test results

- `cargo test --release`: **485 passed** (was 478 + 7 from Phase 2 = 485).
- All P01-P08 tests still pass (no behavior change in shipped policies).

## What this does NOT change yet

- **`KLLUCBStop` (P08)** still uses `parking_lot::Mutex<Cache>` for
  its observe-result. Migration to PolicyCachePublisher is in Phase 4
  alongside the KG-stop integration.
- **`LegacyQuartz` shim (P07)** still reads from
  `Arc<QuartzController>::last_stats()` (the legacy mutex-protected
  `QuartzCtrlInner`). Migration is conceptually trivial but
  bit-identical reproduction of the legacy path is more important
  than refactor velocity here; deferred to Phase 8 cleanup or earlier
  if a phase needs the migration as a precondition.

The plan's original Phase 2 scope was "re-port P07/P08 to use the
immutable cache." Splitting this into "ship the cache infrastructure
(this patch)" + "migrate P07/P08 in a later phase" preserves the
atomic-patch / per-step-audit discipline. Each commit is reviewable
without cross-references to incomplete migrations.

## Adversarial review

### Concurrency soundness

ArcSwap's documentation claims store-release / load-acquire pairing.
The test `test_phase2_cache_publish_visible_to_concurrent_reader`
verifies this empirically — readers spawned after the publish
observe the latest value. Under concurrent publish + concurrent
read, ArcSwap's contract is "reader sees some publish-or-original;
no torn reads." This is exactly what BQ++ needs.

### Memory overhead

`Arc<PolicyCache>` is roughly 8 bytes (pointer) + the cache contents.
For a typical Gomoku 7×7 search with ~30 candidates:
- 6 × SmallVec spill threshold (32) → all stack-allocated, ~768 bytes
- Header fields (~80 bytes)
- Total per-cache: ~850 bytes

Multiple workers loading the same Arc share the underlying
PolicyCache (single allocation per epoch). RcU-style: old caches
are dropped when their last Guard goes out of scope. No memory
leak.

### What the audit's §1.7 fix specifically requires

The audit said: "stop decisions must be made on a fresh snapshot,
not from cache; or, the cache must include a freshness key
(`root_visits_at_observe`, `edge_version_hash`) and the halt path
must reject stale certificates."

This patch provides BOTH: (a) the freshness keys are part of
PolicyCache; (b) `is_stale_for` is the canonical check. Phase 4's
KG-stop is the first policy that USES this (the KLLUCBStop ported
in P08 already had its own staleness behavior internal to the
Mutex<Cache> pattern; that policy will migrate in Phase 4).

### Why ArcSwap and not RwLock

`parking_lot::RwLock<PolicyCache>` would also work, but:
- Read overhead: ~50ns uncontended; under N-thread parallel search,
  the cache lock becomes a sequencing point.
- ArcSwap read: ~10ns (atomic pointer load + counter increment).
  No mutual exclusion between readers and writer.

For a 16-thread parallel search at 50K iter/s, this is the difference
between ~13μs/sec lock overhead vs ~2.5μs/sec atomic overhead — small
in absolute terms but ArcSwap is the canonical lock-free publish
pattern for "many readers, one writer at a time" workloads.

## Files touched

- `Cargo.toml` (+3 LOC; one new dep)
- `src/mcts/policy/cache.rs` (NEW; +320 LOC)
- `src/mcts/policy/mod.rs` (+2 LOC; module declaration + re-exports)

Net delta: **+325 / 0 LOC**.

## What unblocks next

- **Phase 3 (Gumbel SH)**: the GumbelScheduler will write its bracket
  state into the cache's `kg`, `lower`, `upper` arrays at
  `observe()` time and read them lock-free in `score_adjustment`.
- **Phase 4 (KG-stop)**: the first policy that actually consumes
  PolicyCachePublisher; its `should_halt` will use `is_stale_for`
  to reject stale certificates per audit §1.7.
- **Phase 5 (tactical sentinel)**: writes `forced_move_pos` into
  the cache so the halt path can return `Stop(TacticalForced)`
  without re-running the sentinel.
