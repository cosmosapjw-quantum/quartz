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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::mcts::node::MctsNode;

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

struct TtBucket<M: Copy + Send + Sync + 'static> {
    map: HashMap<u64, Arc<MctsNode<M>>>,
}

impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        TtBucket {
            map: HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────
// § TranspositionTable
// ─────────────────────────────────────────────

pub struct TranspositionTable<M: Copy + Send + Sync + 'static> {
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
    pub hits: u64,
    pub misses: u64,
    pub get_or_create_calls: u64,
    pub get_calls: u64,
    pub lock_wait_nanos: u64,
    pub max_lock_wait_nanos: u64,
}

impl<M: Copy + Send + Sync + 'static> TranspositionTable<M> {
    pub fn new() -> Self {
        let buckets = (0..NUM_BUCKETS)
            .map(|_| Mutex::new(TtBucket::new()))
            .collect();
        TranspositionTable {
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
        let idx = Self::bucket_idx(hash);
        self.get_or_create_calls.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let mut bucket = self.buckets[idx].lock().unwrap();
        self.record_lock_wait(t0.elapsed().as_nanos() as u64);

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
                .min_by_key(|(_, n)| n.n_total.load(Ordering::Relaxed))
                .map(|(k, _)| *k);
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
        let idx = Self::bucket_idx(hash);
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let bucket = self.buckets[idx].lock().unwrap();
        self.record_lock_wait(t0.elapsed().as_nanos() as u64);
        bucket.map.get(&hash).map(Arc::clone)
    }

    pub fn size(&self) -> usize {
        self.buckets
            .iter()
            .map(|b| b.lock().unwrap().map.len())
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

    pub fn clear(&self) {
        for b in &self.buckets {
            b.lock().unwrap().map.clear();
        }
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.get_or_create_calls.store(0, Ordering::Relaxed);
        self.get_calls.store(0, Ordering::Relaxed);
        self.lock_wait_nanos.store(0, Ordering::Relaxed);
        self.max_lock_wait_nanos.store(0, Ordering::Relaxed);
    }

    fn record_lock_wait(&self, wait_nanos: u64) {
        self.lock_wait_nanos.fetch_add(wait_nanos, Ordering::Relaxed);
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
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
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
