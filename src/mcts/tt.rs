//! TranspositionTable — lock-striped open-addressing
//!
//! Phase 7 F (2026-04-26): per-bucket storage migrated from
//! `HashMap<u64, ArenaRef<MctsNode<M>>>` to a fixed-size open-addressing
//! slot array (`Box<[TtSlot<M>]>`, 1024 slots × 16 B = 16 KiB / bucket;
//! initial cap=4096 measured at -16 % wall vs. C due to dTLB pressure
//! from a 16 MiB pre-alloc — reduced to 1024 after profile-driven
//! re-tuning, see commit-message numbers).
//! Lookups linearly probe an 8-slot window starting from `(hash >> 8) &
//! SLOT_MASK`. Misses on a full window evict the lowest-`n_total` slot
//! in the window (matches the pre-Phase-7 HashMap eviction policy
//! semantically — no body reclaim, the Bump retains the bytes until
//! `TtBucket::Drop`).
//!
//! Why: callgrind on the Phase 7 C bench shows
//! `TranspositionTable::get_or_create` at ~25 % Ir share. The HashMap
//! path traverses dozens of cycles per lookup (hash compute, bucket
//! redirection, entry walk). The open-addressing window touches at
//! most 8 16-byte slots = 2 cache lines, with a single comparison per
//! slot. Phase 7 plan target: get_or_create < 8 % Ir.
//!
//! 버킷 배분: hash & 0xFF (low byte). Slot start within bucket: bits
//! 8..20 of hash. Different bits ensure low-bucket-collision hashes
//! still spread within a bucket.

use crate::mcts::node::{ArenaRef, MctsEdge, MctsNode};
use bumpalo::Bump;
use parking_lot::RwLock;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────
// § 설정
// ─────────────────────────────────────────────

/// 버킷 수 — 2의 거듭제곱이면 % 연산이 & 연산으로 최적화됨
pub const NUM_BUCKETS: usize = 256;

/// 버킷당 최대 엔트리 수. 초과 시 가장 적게 방문된 노드를 제거.
/// Phase 7 F (2026-04-26): cap = 1024. Profile-driven: scenario A
/// creates ~30 K unique nodes/search → ~120 nodes/bucket avg, so
/// cap=1024 keeps ~88 % headroom while shrinking the per-TT slot
/// pre-allocation to 256 buckets × 1024 × 16 B = 4 MiB (vs 16 MiB at
/// cap=4096). Smaller working set reduces dTLB pressure on the
/// linear-probe path. Cap is a power of two so `& SLOT_MASK` stays
/// branch-free.
pub(crate) const MAX_ENTRIES_PER_BUCKET: usize = 1024;

/// Phase 7 F-prep (2026-04-26): per-slot record for the open-addressing
/// TT landing in the next commit. **Currently unused.** The active TT
/// implementation in this file is the `HashMap<u64, ArenaRef<...>>`
/// path; this struct is declared now so the F commit can swap the
/// storage layer without simultaneously introducing a new data shape.
///
/// Vacancy invariant (Phase 7 F semantics, sealed in next commit):
///   - `hash == 0 && node.is_none()` ↔ vacant slot.
///   - Insertion always populates `node = Some(...)` and any non-zero
///     hash. The TT bucket index uses the full 64-bit `hash`, so a
///     genuine `hash = 0` value would be a Zobrist collision with the
///     sentinel — the engine's Zobrist setup guarantees the empty
///     position hash is non-zero on every game type currently shipped
///     (verified in `zobrist_tt_parallel_verify::v1_zobrist_collision_rate`).
///
/// Layout (M = Copy + Send + Sync + 'static):
///   - `hash: u64` — 8 bytes
///   - `node: Option<ArenaRef<MctsNode<M>>>` — 8 bytes (niche-opt over
///     `NonNull<MctsNode<M>>`, so `None` == null pointer)
/// Total: 16 bytes. With default alignment (8), four slots fit per
/// 64-byte L1 line; the F commit's 8-slot probe window covers two
/// adjacent cache lines.
#[derive(Clone, Copy)]
pub(crate) struct TtSlot<M: Copy + Send + Sync + 'static> {
    pub hash: u64,
    pub node: Option<ArenaRef<MctsNode<M>>>,
}

impl<M: Copy + Send + Sync + 'static> TtSlot<M> {
    /// Sentinel for an empty slot. Used to initialize the slot array
    /// when the F commit replaces `HashMap` with `Box<[TtSlot<M>]>` of
    /// length `MAX_ENTRIES_PER_BUCKET`.
    pub const VACANT: Self = Self {
        hash: 0,
        node: None,
    };

    /// True iff this slot has no published node. Equivalent to
    /// `self.node.is_none()`; the helper exists for symmetry with the
    /// open-addressing probe-window scan in F.
    #[inline]
    pub fn is_vacant(&self) -> bool {
        self.node.is_none()
    }
}

// ─────────────────────────────────────────────
// § TT Bucket (Phase 7 F open-addressing)
// ─────────────────────────────────────────────

/// Bitmask for slot index reduction. `MAX_ENTRIES_PER_BUCKET` is a
/// power of two so `& SLOT_MASK` replaces `%` (cheaper, branch-free).
const SLOT_MASK: usize = MAX_ENTRIES_PER_BUCKET - 1;

/// Linear-probe window. A probe walks at most `PROBE_WINDOW` slots
/// before declaring "no match" (read path) or evicting the
/// lowest-`n_total` slot (write path). At 16 B / slot the window
/// covers two 64-byte cache lines, which is a tight working set even
/// under heavy concurrent probing.
const PROBE_WINDOW: usize = 8;

struct TtBucket<M: Copy + Send + Sync + 'static> {
    /// Phase 7 F: fixed-size open-addressing slot array.
    /// `len() == MAX_ENTRIES_PER_BUCKET`. Each slot is `TtSlot::VACANT`
    /// at construction; insertion sets `(hash, Some(node))` and
    /// eviction overwrites in place (the displaced node body is left
    /// in the Bump until `TtBucket::Drop`).
    slots: Box<[TtSlot<M>]>,
    /// Bumpalo arena: all `MctsNode<M>` bodies AND per-node edge slabs
    /// (Phase 7 C, 2026-04-26) for this bucket are allocated here.
    /// Bumpalo guarantees pointer stability for the life of the Bump,
    /// so `ArenaRef`s and `edges_ptr`s into it remain valid until the
    /// containing TT is dropped.
    arena: Bump,
}

// SAFETY (Phase 6.2 → Phase 7 F): `TtBucket` contains a `Bump` (!Sync)
// AND a `Box<[TtSlot<M>]>` of plain (Copy, no interior mutability) records.
// Wrapping in `parking_lot::RwLock` requires `Sync`; soundness rests on
// the per-bucket access discipline:
//
//   - Read guards (`buckets[i].read()`) may concurrently load fields
//     from `bucket.slots[..]` (plain reads of Copy data — no race) and
//     MUST NOT touch `bucket.arena` (`&Bump` aliasing across readers
//     would be UB).
//   - Write guards (`buckets[i].write()`) hold exclusive access by
//     RwLock contract and may mutate either field freely.
//
// Phase 7 F adds one new caller: `allocate_edge_slab` (write guard,
// touches arena). The original `get_or_create` paths remain (read for
// probe, write on miss).
unsafe impl<M: Copy + Send + Sync + 'static> Sync for TtBucket<M> {}

impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        // Pre-allocate the slot array filled with `VACANT`. Allocation
        // pattern: vec! macro → Box<[TtSlot<M>]> via into_boxed_slice.
        // The vec is stack-built then heap-shrunk; no separate raw
        // alloc helper needed.
        let slots = vec![TtSlot::<M>::VACANT; MAX_ENTRIES_PER_BUCKET].into_boxed_slice();
        TtBucket {
            slots,
            arena: Bump::new(),
        }
    }
}

impl<M: Copy + Send + Sync + 'static> Drop for TtBucket<M> {
    fn drop(&mut self) {
        // Phase 7 F (2026-04-26): walk the slot array and drop_in_place
        // every populated MctsNode<M> body before the Bump frees its
        // chunks. Same discipline as the pre-Phase-7 HashMap drain: the
        // arena does not run Drop on its allocations, so the explicit
        // walk is required to release each node's heap-owned sub-allocs
        // (OnceLock<Box<...>> candidates) and — Phase 7 C — invoke
        // `MctsNode::Drop` which drains the edge slab.
        //
        // Field drop order: `slots` (declaration-order first) drains
        // surviving bodies, then `arena` (next field) frees the Bump
        // chunks. Bodies that were *evicted* during the bucket's life
        // are not reachable from `slots` and thus are NOT
        // drop_in_place'd (matches the HashMap path's pre-Phase-7
        // eviction behavior — see the eviction note in §
        // "Per-bucket bumpalo storage" below).
        //
        // Safety
        //   Each `Some(node)` slot is the master reference to a node
        //   body in `self.arena`. By the engine's drop-order discipline
        //   no concurrent reader exists; `&mut self` excludes anyone
        //   else. `drop_in_place` runs `MctsNode::Drop` exactly once
        //   per surviving body.
        for slot in self.slots.iter() {
            if let Some(node_ref) = slot.node {
                let raw: *mut MctsNode<M> =
                    crate::mcts::node::ArenaRef::as_ptr(&node_ref) as *mut MctsNode<M>;
                // SAFETY: documented above.
                unsafe {
                    std::ptr::drop_in_place(raw);
                }
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

    #[inline]
    fn bucket_idx(hash: u64) -> usize {
        (hash as usize) & (NUM_BUCKETS - 1)
    }

    /// Phase 7 F (2026-04-26): slot start within a bucket. Uses bits
    /// 8..20 of the hash so the bucket index (low 8 bits) and slot
    /// index draw from disjoint regions of the Zobrist key.
    #[inline]
    fn slot_start(hash: u64) -> usize {
        ((hash >> 8) as usize) & SLOT_MASK
    }

    /// 해시에 해당하는 노드를 조회.
    /// 없으면 `terminal_value`로 새 노드를 생성하고 삽입.
    /// 있으면 기존 노드 반환 (always-replace 충돌 정책 → 먼저 들어온 것 우선).
    ///
    /// Phase 7 F (2026-04-26): open-addressing replacement for the
    /// pre-Phase-7 `HashMap` lookup. The fast path reads at most
    /// `PROBE_WINDOW` (= 8) 16-byte slots within a single bucket,
    /// terminating on either a hash match (HIT, return the slot's
    /// `ArenaRef`) or a vacant slot (MISS — insertion path).
    ///
    /// Lock ladder unchanged from Phase 6.2: read lock for probe,
    /// upgrade to write lock on miss. A second probe under write lock
    /// double-checks for racing writers before alloc + insert.
    pub fn get_or_create(&self, hash: u64, terminal_value: Option<f32>) -> ArenaRef<MctsNode<M>> {
        if !self.enabled {
            // Disabled TT: caller-owned ad-hoc node, leaked. This path is
            // hit only when `tt_enabled = false` (a few test/ablation
            // configs); leak per get_or_create is acceptable there.
            return leak_node(hash, terminal_value);
        }
        let bucket_idx = Self::bucket_idx(hash);
        self.get_or_create_calls.fetch_add(1, Ordering::Relaxed);
        let slot_start = Self::slot_start(hash);

        // Fast path: read lock + linear probe. Multiple threads can be
        // in this branch simultaneously on the same bucket.
        {
            let t0 = crate::mcts::profiling::maybe_start_timer();
            let bucket = self.buckets[bucket_idx].read();
            if let Some(t0) = t0 {
                self.record_lock_wait(t0.elapsed().as_nanos() as u64);
            }
            for offset in 0..PROBE_WINDOW {
                let i = (slot_start + offset) & SLOT_MASK;
                // SAFETY: `i & SLOT_MASK` is in [0, MAX_ENTRIES_PER_BUCKET);
                // `bucket.slots.len() == MAX_ENTRIES_PER_BUCKET` (set in
                // `TtBucket::new`).
                let slot = unsafe { bucket.slots.get_unchecked(i) };
                if slot.hash == hash {
                    if let Some(node) = slot.node {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        return node;
                    }
                }
                if slot.is_vacant() {
                    // Linear probe terminates at the first vacant slot:
                    // any subsequent slot with this hash would have been
                    // placed here on insert (the `find_insertion_slot`
                    // logic below picks the first vacant slot in the
                    // window before considering eviction).
                    break;
                }
            }
        }

        // Slow path: write lock + re-probe + alloc + insert.
        let t1 = crate::mcts::profiling::maybe_start_timer();
        let mut bucket = self.buckets[bucket_idx].write();
        if let Some(t1) = t1 {
            self.record_lock_wait(t1.elapsed().as_nanos() as u64);
        }

        // Re-probe under write lock to catch racing inserts.
        for offset in 0..PROBE_WINDOW {
            let i = (slot_start + offset) & SLOT_MASK;
            let slot = unsafe { bucket.slots.get_unchecked(i) };
            if slot.hash == hash {
                if let Some(node) = slot.node {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return node;
                }
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);

        // Find insertion slot: first vacant in window, else evict the
        // lowest-`n_total` slot. Eviction does NOT drop_in_place the
        // displaced body — outstanding `ArenaRef`s elsewhere in the
        // tree may still reference it. The body is reclaimed at
        // bucket-Drop time only if it remains in a slot; otherwise it
        // lives unreferenced in the Bump (acceptable per the Phase 3
        // arena leak contract). This matches the pre-Phase-7 HashMap
        // eviction policy (which `bucket.map.remove(victim_hash)`
        // dropped the ArenaRef without touching the body).
        let mut insert_idx = slot_start;
        let mut min_n: u32 = u32::MAX;
        let mut min_idx = slot_start;
        let mut found_vacant = false;
        for offset in 0..PROBE_WINDOW {
            let i = (slot_start + offset) & SLOT_MASK;
            let slot = unsafe { bucket.slots.get_unchecked(i) };
            if slot.is_vacant() {
                insert_idx = i;
                found_vacant = true;
                break;
            }
            // Occupied — track lowest n_total candidate for eviction.
            // SAFETY: `slot.is_vacant()` would have returned true for
            // `node = None`; we are in the else branch, so node is Some.
            let n = unsafe { slot.node.unwrap_unchecked() }
                .n_total
                .load(Ordering::Relaxed);
            if n < min_n {
                min_n = n;
                min_idx = i;
            }
        }
        if !found_vacant {
            insert_idx = min_idx;
        }

        // Allocate the body in the bucket's Bump. Bumpalo's `alloc`
        // returns a `&mut T` whose lifetime is bound by `&self`, but the
        // allocation itself is stable until the Bump drops. We capture
        // the address as `NonNull<T>` and wrap in `ArenaRef`, whose
        // safety invariant is documented at its definition.
        let body = MctsNode::new(hash, terminal_value);
        let allocated: &mut MctsNode<M> = bucket.arena.alloc(body);
        let node = unsafe { ArenaRef::from_raw(NonNull::new_unchecked(allocated as *mut _)) };

        // Publish the new entry. SAFETY: insert_idx ≤ SLOT_MASK <
        // bucket.slots.len(); we hold the write guard so no other
        // thread observes this slot mid-update.
        unsafe {
            *bucket.slots.get_unchecked_mut(insert_idx) = TtSlot {
                hash,
                node: Some(node),
            };
        }
        node
    }

    /// Phase 7 C (2026-04-26): allocate (or look up) the per-node edge
    /// slab in the bucket's bumpalo Bump.
    ///
    /// Idempotent: if `node.edges_ptr` is already non-null, returns the
    /// existing (ptr, cap) under an Acquire load — no lock taken. On
    /// first call, takes the bucket write lock, allocates a slab of
    /// `n_candidates` `MctsEdge<M>` slots in `bucket.arena`, and
    /// publishes the pointer via Release-store.
    ///
    /// # Memory ordering
    ///   - Writer: bucket-write-locked alloc, then `edges_ptr.store(Release)`,
    ///     then `edges_cap` is implicit (capacity = `n_candidates`,
    ///     known to all callers via `node.candidates`).
    ///   - Reader (in `materialize_edges` fill loop): observes the
    ///     non-null ptr via Acquire load, then writes to slots under
    ///     `materialize_lock`.
    ///   - Lock-free PUCT reader (`MctsNode::read_edges`): sees ptr
    ///     publication transitively via the Release-store of
    ///     `edge_cursor` at the end of the materialize fill loop.
    ///
    /// # Safety
    ///   The returned pointer is valid for `n_candidates` aligned
    ///   `MctsEdge<M>` slots and remains valid until the bucket's Bump
    ///   is dropped (i.e., until the engine's TT drops). All writes to
    ///   the slab MUST be serialized via `node.materialize_lock`.
    pub fn allocate_edge_slab<MM: Copy + Send + Sync + 'static>(
        &self,
        node: &MctsNode<MM>,
        n_candidates: usize,
    ) -> (*mut MctsEdge<MM>, u32) {
        debug_assert!(n_candidates > 0);
        // Fast path: slab already published.
        let existing = node.edges_ptr.load(Ordering::Acquire);
        if !existing.is_null() {
            // The cap was set once and is not stored on the node — it
            // is implicit in `node.candidates.get().unwrap().len()`.
            // Callers that need it should consult that. We return the
            // published pointer plus the *requested* `n_candidates`
            // verbatim; callers always pass `candidates.len()`, so the
            // cap is identical across calls.
            return (existing as *mut MctsEdge<MM>, n_candidates as u32);
        }

        let idx = Self::bucket_idx(node.hash);
        // Slab alloc requires bucket write lock (per the unsafe Sync
        // discipline on TtBucket: only write guards may touch arena).
        let bucket = self.buckets[idx].write();

        // Double-check after acquiring the write lock.
        let existing = node.edges_ptr.load(Ordering::Relaxed);
        if !existing.is_null() {
            return (existing as *mut MctsEdge<MM>, n_candidates as u32);
        }

        let layout = std::alloc::Layout::array::<MctsEdge<MM>>(n_candidates)
            .expect("MctsEdge slab layout");
        // SAFETY: `Bump::alloc_layout` returns a `NonNull<u8>` with the
        // requested layout. The pointer is valid until the Bump drops
        // (the TT owns the bucket; the bucket owns the Bump).
        let raw = bucket.arena.alloc_layout(layout);
        let slab_ptr = raw.as_ptr() as *mut MctsEdge<MM>;

        // Release-store publishes the pointer to lock-free readers.
        node.edges_ptr.store(slab_ptr, Ordering::Release);

        (slab_ptr, n_candidates as u32)
    }

    /// 조회만 (삽입 없음)
    /// Phase 7 F (2026-04-26): open-addressing probe — same probe
    /// window as `get_or_create` fast path.
    pub fn get(&self, hash: u64) -> Option<ArenaRef<MctsNode<M>>> {
        if !self.enabled {
            return None;
        }
        let bucket_idx = Self::bucket_idx(hash);
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        let slot_start = Self::slot_start(hash);
        let t0 = crate::mcts::profiling::maybe_start_timer();
        let bucket = self.buckets[bucket_idx].read();
        if let Some(t0) = t0 {
            self.record_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        for offset in 0..PROBE_WINDOW {
            let i = (slot_start + offset) & SLOT_MASK;
            let slot = unsafe { bucket.slots.get_unchecked(i) };
            if slot.hash == hash {
                return slot.node;
            }
            if slot.is_vacant() {
                return None;
            }
        }
        None
    }

    pub fn size(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        // Phase 7 F (2026-04-26): linear count of occupied slots per
        // bucket. With 256 buckets × 1024 slots = 256 K slots, a full
        // walk is ~1 M reads — only used for diagnostics, not hot path.
        self.buckets
            .iter()
            .map(|b| {
                let bucket = b.read();
                bucket.slots.iter().filter(|s| !s.is_vacant()).count()
            })
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
