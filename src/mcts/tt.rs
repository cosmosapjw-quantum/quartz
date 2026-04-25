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

use crate::mcts::node::MctsNode;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────
// § 설정
// ─────────────────────────────────────────────

/// 버킷 수 — 2의 거듭제곱이면 % 연산이 & 연산으로 최적화됨
const NUM_BUCKETS: usize = 256;

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

type TtMap<M> = HashMap<u64, Arc<MctsNode<M>>, BuildHasherDefault<U64IdentityHasher>>;

struct TtBucket<M: Copy + Send + Sync + 'static> {
    map: TtMap<M>,
}

impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        TtBucket {
            map: TtMap::with_capacity_and_hasher(
                MAX_ENTRIES_PER_BUCKET / 2,
                Default::default(),
            ),
        }
    }
}

// ─────────────────────────────────────────────
// § TranspositionTable
// ─────────────────────────────────────────────

pub struct TranspositionTable<M: Copy + Send + Sync + 'static> {
    enabled: bool,
    buckets: Vec<Mutex<TtBucket<M>>>,
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
            .map(|_| Mutex::new(TtBucket::new()))
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
    pub fn get_or_create(&self, hash: u64, terminal_value: Option<f32>) -> Arc<MctsNode<M>> {
        if !self.enabled {
            return MctsNode::new(hash, terminal_value);
        }
        let idx = Self::bucket_idx(hash);
        self.get_or_create_calls.fetch_add(1, Ordering::Relaxed);
        let t0 = crate::mcts::profiling::maybe_start_timer();
        let mut bucket = self.buckets[idx].lock();
        if let Some(t0) = t0 {
            self.record_lock_wait(t0.elapsed().as_nanos() as u64);
        }

        if let Some(node) = bucket.map.get(&hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(node);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);

        // Evict least-visited entry if bucket is full
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

        let node = MctsNode::new(hash, terminal_value); // already Arc<MctsNode<M>>
        bucket.map.insert(hash, Arc::clone(&node));
        node
    }

    /// 조회만 (삽입 없음)
    pub fn get(&self, hash: u64) -> Option<Arc<MctsNode<M>>> {
        if !self.enabled {
            return None;
        }
        let idx = Self::bucket_idx(hash);
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        let t0 = crate::mcts::profiling::maybe_start_timer();
        let bucket = self.buckets[idx].lock();
        if let Some(t0) = t0 {
            self.record_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        bucket.map.get(&hash).map(Arc::clone)
    }

    pub fn size(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        self.buckets
            .iter()
            .map(|b| b.lock().map.len())
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
    use std::sync::Arc;

    #[test]
    fn disabled_table_does_not_merge_or_store_nodes() {
        let tt = TranspositionTable::<usize>::new_enabled(false);
        let a = tt.get_or_create(123, None);
        let b = tt.get_or_create(123, None);

        assert_eq!(tt.size(), 0);
        assert!(tt.get(123).is_none());
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn identity_hasher_preserves_distinct_u64_keys() {
        let tt = TranspositionTable::<usize>::new_enabled(true);
        let a = tt.get_or_create(0, None);
        let b = tt.get_or_create(u64::MAX, None);
        let c = tt.get_or_create(0x0102_0304_0506_0708, None);

        assert!(Arc::ptr_eq(&a, &tt.get(0).unwrap()));
        assert!(Arc::ptr_eq(&b, &tt.get(u64::MAX).unwrap()));
        assert!(Arc::ptr_eq(&c, &tt.get(0x0102_0304_0506_0708).unwrap()));
        assert!(!Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &c));
        assert!(!Arc::ptr_eq(&b, &c));
    }
}
