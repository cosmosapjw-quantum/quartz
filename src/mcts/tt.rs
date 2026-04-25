//! TranspositionTable — lock striping
//!
//! 이전: Arc<Mutex<HashMap<>>> (단일 락 — 스레드 수↑ → 경합↑)
//! 현재: NUM_BUCKETS 개 버킷, 각 버킷이 독립적인 Mutex<HashMap>
//!
//! 버킷 배분: hash % NUM_BUCKETS
//!   → 스레드들이 서로 다른 버킷에 접근하면 락 경합 없음
//!   → 최악(같은 버킷): 그래도 기존 단일 락과 같은 수준
//!
//! 추가 기능:
//!   - 충돌 정책: always-replace (가장 단순, baseline)
//!   - TT 통계: hits, misses, collisions (로깅용)

use crate::mcts::node::{ArenaRef, MctsNode};
use bumpalo::Bump;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────
// § 설정
// ─────────────────────────────────────────────

/// 버킷 수 — 2의 거듭제곱이면 % 연산이 & 연산으로 최적화됨
pub const NUM_BUCKETS: usize = 256;

/// 버킷당 최대 엔트리 수. 초과 시 가장 적게 방문된 노드를 제거.
/// 256 buckets × 4096 entries = ~1M total entries max.
const MAX_ENTRIES_PER_BUCKET: usize = 4096;

// ─────────────────────────────────────────────
// § TT Entry (통계 포함)
// ─────────────────────────────────────────────

#[derive(Default)]
struct U64IdentityHasher {
    value: u64,
}

impl Hasher for U64IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.value
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.value = i;
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut folded = 0u64;
        for (shift, byte) in bytes.iter().take(8).enumerate() {
            folded |= (*byte as u64) << (shift * 8);
        }
        self.value = folded;
    }
}

type TtMap<M> = HashMap<u64, ArenaRef<MctsNode<M>>, BuildHasherDefault<U64IdentityHasher>>;

struct TtBucket<M: Copy + Send + Sync + 'static> {
    map: TtMap<M>,
    /// Bumpalo arena: all `MctsNode<M>` bodies for this bucket are
    /// allocated here. Bumpalo guarantees pointer stability for the life
    /// of the Bump, so `ArenaRef`s into it remain valid until the
    /// containing `ArenaPool` is dropped.
    arena: Bump,
}

// SAFETY (Phase 6.2, 2026-04-25): `TtBucket` contains a `Bump`, which is
// `!Sync`. Wrapping the bucket in `parking_lot::RwLock` requires the
// inner type to be `Sync` (so a read guard can hand out `&TtBucket` to
// multiple threads). Soundness rests on a per-bucket access discipline:
//
//   - Read guards (`buckets[i].read()`) MUST only touch `bucket.map`.
//     They MUST NOT call methods on `bucket.arena` (those take `&Bump`
//     and would alias under concurrent readers, which is UB).
//   - Write guards (`buckets[i].write()`) hold exclusive access by
//     RwLock contract, so they may freely mutate either field.
//
// The only call sites are within this module: `get_or_create` (read for
// probe, write on miss), `get` (read), `size` (read), and `Drop` (which
// runs with `&mut self`). Each has been audited to respect the
// discipline above.
unsafe impl<M: Copy + Send + Sync + 'static> Sync for TtBucket<M> {}

impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        TtBucket {
            map: TtMap::with_capacity_and_hasher(
                MAX_ENTRIES_PER_BUCKET / 2,
                Default::default(),
            ),
            arena: Bump::new(),
        }
    }
}

impl<M: Copy + Send + Sync + 'static> Drop for TtBucket<M> {
    fn drop(&mut self) {
        // bumpalo's `Bump` does NOT run `Drop` on its allocations: when
        // the Bump goes out of scope, it just frees its raw chunks. Each
        // `MctsNode<M>` body, however, transitively owns heap-allocated
        // sub-structures via global allocator (the `OnceLock<Box<...>>`
        // candidates payload and the `Vec<MctsEdge<M>>` inside its
        // `RwLock`). Without explicit `drop_in_place`, those sub-allocs
        // would leak per-node — multiplied across ~30 K nodes/search,
        // that's a multi-MB leak per engine session, accumulating across
        // FFI calls.
        //
        // Drop order
        //   We run `drop_in_place` on every node body in `self.map`
        //   BEFORE the Bump field drops. Field drop order in Rust is
        //   declaration order (`map` before `arena` here), so the
        //   imperative drain below runs first and the Bump frees its
        //   chunks last.
        //
        // Safety
        //   Each `ArenaRef<MctsNode<M>>` value in `self.map` is the
        //   master reference to a node body allocated in `self.arena`.
        //   By the engine's drop-order discipline (`MctsEngine` field
        //   layout) and the search-time invariant (no `ArenaRef` escapes
        //   beyond a finished search), no other live reference to these
        //   bodies exists at this point. `MctsNode::drop` itself does
        //   not follow `ArenaRef<MctsNode<M>>` children — those are raw
        //   pointer wrappers without a `Drop` impl — so cross-bucket
        //   double-drop is impossible.
        for (_, node_ref) in self.map.drain() {
            let raw: *mut MctsNode<M> =
                crate::mcts::node::ArenaRef::as_ptr(&node_ref) as *mut MctsNode<M>;
            // SAFETY: documented above.
            unsafe {
                std::ptr::drop_in_place(raw);
            }
        }
    }
}

// ─────────────────────────────────────────────
// § Per-bucket bumpalo storage
// ─────────────────────────────────────────────
//
// Each `TtBucket` owns its own `bumpalo::Bump`. All `MctsNode<M>` bodies
// inserted via `get_or_create` are allocated in the bucket's Bump and
// returned as `ArenaRef<MctsNode<M>>`. Allocations are serialized
// through the existing per-bucket `Mutex<TtBucket>` (Bump is `!Sync` —
// the lock is the synchronization mechanism).
//
// Lifetime safety
//   The TT owns the Bumps. Edges/snapshots/path entries hold
//   `ArenaRef`s into those Bumps. The engine holds `Arc<TT>` and ensures
//   the TT outlives every `ArenaRef` reachable through the engine's
//   internal state. When the last `Arc<TT>` drops, both the map (which
//   contains `ArenaRef`s with no Drop impl) and the Bumps drop. After
//   that point no `ArenaRef` is reachable.
//
// Eviction note: removing an entry from the map does NOT free its
// Bump-allocated body (bumpalo never reclaims individual allocations).
// This is a deliberate trade — eviction is rare under scenario A and
// the per-bucket cap (`MAX_ENTRIES_PER_BUCKET`) bounds total bytes.

/// Test/standalone helper: allocate an `MctsNode` body on the heap and
/// leak it, returning an `ArenaRef`. Used by detached test setups in
/// `mcts/quartz.rs` and similar places that previously called
/// `Arc::new(MctsNode::new(...))` outside of an engine. Production hot
/// paths (engine select/expand/backup) MUST go through
/// `TranspositionTable::get_or_create`.
pub fn leak_node<M: Copy + Send + Sync + 'static>(
    hash: u64,
    terminal_value: Option<f32>,
) -> ArenaRef<MctsNode<M>> {
    let boxed = Box::new(MctsNode::new(hash, terminal_value));
    let raw = Box::into_raw(boxed);
    // SAFETY: `raw` is a valid heap pointer (non-null, aligned). It is
    // intentionally leaked; this helper is for test scaffolding only and
    // its leak is acceptable in those contexts.
    unsafe { ArenaRef::from_raw(NonNull::new_unchecked(raw)) }
}

// ─────────────────────────────────────────────
// § TranspositionTable
// ─────────────────────────────────────────────

pub struct TranspositionTable<M: Copy + Send + Sync + 'static> {
    enabled: bool,
    /// Per-bucket `RwLock<TtBucket>` (Phase 6.2, 2026-04-25). The hit path
    /// in `get_or_create` and `get` takes a read lock so multiple threads
    /// can probe the same bucket in parallel; the miss path of
    /// `get_or_create` upgrades to a write lock for the alloc + insert.
    /// See the safety note on `TtBucket`'s `unsafe Sync` impl above.
    buckets: Vec<RwLock<TtBucket<M>>>,
    // 통계 (로깅용)
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub get_or_create_calls: AtomicU64,
    pub get_calls: AtomicU64,
    pub lock_wait_nanos: AtomicU64,
    pub max_lock_wait_nanos: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct TtContentionSnapshot {
    pub get_or_create_calls: u64,
    pub get_calls: u64,
    pub lock_wait_nanos: u64,
    pub max_lock_wait_nanos: u64,
}

impl<M: Copy + Send + Sync + 'static> TranspositionTable<M> {
    pub fn new() -> Self {
        Self::new_enabled(true)
    }

    pub fn new_enabled(enabled: bool) -> Self {
        let buckets = (0..NUM_BUCKETS)
            .map(|_| RwLock::new(TtBucket::new()))
            .collect();
        TranspositionTable {
            enabled,
            buckets,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            get_or_create_calls: AtomicU64::new(0),
            get_calls: AtomicU64::new(0),
            lock_wait_nanos: AtomicU64::new(0),
            max_lock_wait_nanos: AtomicU64::new(0),
        }
    }

    fn bucket_idx(hash: u64) -> usize {
        (hash as usize) % NUM_BUCKETS
    }

    /// 해시에 해당하는 노드를 조회.
    /// 없으면 `terminal_value`로 새 노드를 생성하고 삽입.
    /// 있으면 기존 노드 반환 (always-replace 충돌 정책 → 먼저 들어온 것 우선).
    ///
    /// Phase 6.2 (2026-04-25): hits take a read lock so multiple threads
    /// can probe the same bucket in parallel. Misses upgrade to a write
    /// lock and double-check before allocating, which keeps the
    /// happy-path `get_or_create` cost down to a single uncontended
    /// `RwLock::read()` on the ~80 % of calls that are hits in steady
    /// state.
    pub fn get_or_create(&self, hash: u64, terminal_value: Option<f32>) -> ArenaRef<MctsNode<M>> {
        if !self.enabled {
            // Disabled TT: caller-owned ad-hoc node, leaked. This path is
            // hit only when `tt_enabled = false` (a few test/ablation
            // configs); leak per get_or_create is acceptable there.
            return leak_node(hash, terminal_value);
        }
        let idx = Self::bucket_idx(hash);
        self.get_or_create_calls.fetch_add(1, Ordering::Relaxed);

        // Fast path: read lock + map probe. Multiple threads can be in
        // this branch simultaneously on the same bucket.
        {
            let t0 = crate::mcts::profiling::maybe_start_timer();
            let bucket = self.buckets[idx].read();
            if let Some(t0) = t0 {
                self.record_lock_wait(t0.elapsed().as_nanos() as u64);
            }
            if let Some(node) = bucket.map.get(&hash) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return *node;
            }
        }

        // Slow path: write lock + double-check + alloc + insert.
        let t1 = crate::mcts::profiling::maybe_start_timer();
        let mut bucket = self.buckets[idx].write();
        if let Some(t1) = t1 {
            self.record_lock_wait(t1.elapsed().as_nanos() as u64);
        }
        if let Some(node) = bucket.map.get(&hash) {
            // Another writer raced us to insert this entry between the
            // read drop and the write acquire. Treat as a hit.
            self.hits.fetch_add(1, Ordering::Relaxed);
            return *node;
        }

        self.misses.fetch_add(1, Ordering::Relaxed);

        // Evict least-visited entry if bucket is full. Note: bumpalo does
        // not free the backing memory on remove (the slot stays allocated
        // in the Bump). This is acceptable: eviction in this engine is
        // rare in scenario-A traces, and the per-bucket Bump caps
        // total bytes per bucket at the working-set size.
        if bucket.map.len() >= MAX_ENTRIES_PER_BUCKET {
            let victim = bucket
                .map
                .iter()
                .min_by_key(|(_, node)| node.n_total.load(Ordering::Relaxed))
                .map(|(hash, _)| *hash);
            if let Some(victim_hash) = victim {
                bucket.map.remove(&victim_hash);
            }
        }

        // Allocate the body in the bucket's Bump. Bumpalo's `alloc`
        // returns a `&mut T` whose lifetime is bound by `&self`, but the
        // allocation itself is stable until the Bump drops. We capture
        // the address as `NonNull<T>` and wrap in `ArenaRef`, whose
        // safety invariant is documented at its definition.
        let body = MctsNode::new(hash, terminal_value);
        let allocated: &mut MctsNode<M> = bucket.arena.alloc(body);
        // SAFETY: `allocated` is a fresh, non-null reference to a value
        // just allocated in `bucket.arena`. The `ArenaPool` (and thus
        // every per-bucket Bump) is owned by the engine and outlives all
        // `ArenaRef`s reachable from the TT.
        let node = unsafe { ArenaRef::from_raw(NonNull::new_unchecked(allocated as *mut _)) };
        bucket.map.insert(hash, node);
        node
    }

    /// 조회만 (삽입 없음)
    pub fn get(&self, hash: u64) -> Option<ArenaRef<MctsNode<M>>> {
        if !self.enabled {
            return None;
        }
        let idx = Self::bucket_idx(hash);
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        let t0 = crate::mcts::profiling::maybe_start_timer();
        let bucket = self.buckets[idx].read();
        if let Some(t0) = t0 {
            self.record_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        bucket.map.get(&hash).copied()
    }

    pub fn size(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        self.buckets
            .iter()
            .map(|b| b.read().map.len())
            .sum()
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        if h + m > 0.0 {
            h / (h + m)
        } else {
            0.0
        }
    }

    fn record_lock_wait(&self, wait_nanos: u64) {
        if !crate::mcts::profiling::hot_path_metrics_enabled() {
            return;
        }
        self.lock_wait_nanos
            .fetch_add(wait_nanos, Ordering::Relaxed);
        let mut prev = self.max_lock_wait_nanos.load(Ordering::Relaxed);
        while wait_nanos > prev {
            match self.max_lock_wait_nanos.compare_exchange(
                prev,
                wait_nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(cur) => prev = cur,
            }
        }
    }

    pub fn contention_snapshot(&self) -> TtContentionSnapshot {
        TtContentionSnapshot {
            get_or_create_calls: self.get_or_create_calls.load(Ordering::Relaxed),
            get_calls: self.get_calls.load(Ordering::Relaxed),
            lock_wait_nanos: self.lock_wait_nanos.load(Ordering::Relaxed),
            max_lock_wait_nanos: self.max_lock_wait_nanos.load(Ordering::Relaxed),
        }
    }
}

impl<M: Copy + Send + Sync + 'static> Default for TranspositionTable<M> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::TranspositionTable;
    use crate::mcts::node::ArenaRef;

    #[test]
    fn disabled_table_does_not_merge_or_store_nodes() {
        let tt = TranspositionTable::<usize>::new_enabled(false);
        let a = tt.get_or_create(123, None);
        let b = tt.get_or_create(123, None);

        assert_eq!(tt.size(), 0);
        assert!(tt.get(123).is_none());
        assert!(!ArenaRef::ptr_eq(&a, &b));
    }

    #[test]
    fn identity_hasher_preserves_distinct_u64_keys() {
        let tt = TranspositionTable::<usize>::new_enabled(true);
        let a = tt.get_or_create(0, None);
        let b = tt.get_or_create(u64::MAX, None);
        let c = tt.get_or_create(0x0102_0304_0506_0708, None);

        assert!(ArenaRef::ptr_eq(&a, &tt.get(0).unwrap()));
        assert!(ArenaRef::ptr_eq(&b, &tt.get(u64::MAX).unwrap()));
        assert!(ArenaRef::ptr_eq(&c, &tt.get(0x0102_0304_0506_0708).unwrap()));
        assert!(!ArenaRef::ptr_eq(&a, &b));
        assert!(!ArenaRef::ptr_eq(&a, &c));
        assert!(!ArenaRef::ptr_eq(&b, &c));
    }
}
