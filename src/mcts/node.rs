//! MCTS 노드 구조체 (v0.3.1 — Compact Candidates)
//!
//! 메모리 최적화 변경:
//!   candidates: Box<[(M, f32, u64, Option<f32>)]>  →  Box<[(M, f32)]>
//!   (28 bytes/entry → 12 bytes/entry, 57% 절약)
//!
//!   child_hash, terminal_value는 candidates에 저장하지 않고,
//!   materialize_edges(node, state, ...) 시점에 state.apply_move(mv)로 lazy 계산.
//!
//! 불변식:
//!   1. candidates: OnceLock — CAS exactly-once
//!   2. edges: RwLock<Vec<Arc<MctsEdge>>> — append-only, 인덱스 영구 안정
//!   3. edge_cursor ≤ edges.len() (항상)
//!   4. edge_cursor는 edges.len()의 단조 증가 근사치 (RwLock 외부 빠른 확인용)

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

// ─────────────────────────────────────────────
// § Atomic f64 헬퍼
// ─────────────────────────────────────────────

pub fn atomic_f64_add(a: &AtomicU64, delta: f64) {
    let mut cur = a.load(Ordering::Relaxed);
    let mut spins = 0u32;
    loop {
        let new = f64::from_bits(cur) + delta;
        match a.compare_exchange_weak(cur, new.to_bits(), Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return,
            Err(v) => {
                cur = v;
                spins += 1;
                if spins <= 3 {
                    core::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

pub fn atomic_f64_load(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Acquire))
}

static EDGES_LOCK_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
static EDGES_LOCK_WAIT_MAX_NANOS: AtomicU64 = AtomicU64::new(0);
static EDGES_LOCK_CALLS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn record_edges_lock_wait(wait_nanos: u64) {
    if !crate::mcts::profiling::hot_path_metrics_enabled() {
        return;
    }
    EDGES_LOCK_CALLS.fetch_add(1, Ordering::Relaxed);
    EDGES_LOCK_WAIT_NANOS.fetch_add(wait_nanos, Ordering::Relaxed);
    let mut cur = EDGES_LOCK_WAIT_MAX_NANOS.load(Ordering::Relaxed);
    while wait_nanos > cur {
        match EDGES_LOCK_WAIT_MAX_NANOS.compare_exchange_weak(
            cur,
            wait_nanos,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => cur = next,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeLockContentionSnapshot {
    pub calls: u64,
    pub wait_nanos: u64,
    pub max_wait_nanos: u64,
}

pub fn edge_lock_contention_snapshot() -> EdgeLockContentionSnapshot {
    EdgeLockContentionSnapshot {
        calls: EDGES_LOCK_CALLS.load(Ordering::Relaxed),
        wait_nanos: EDGES_LOCK_WAIT_NANOS.load(Ordering::Relaxed),
        max_wait_nanos: EDGES_LOCK_WAIT_MAX_NANOS.load(Ordering::Relaxed),
    }
}

// ─────────────────────────────────────────────
// § MctsEdge
// ─────────────────────────────────────────────

pub struct MctsEdge<M> {
    pub mv: M,
    pub child: Arc<MctsNode<M>>,
    pub p: f32,
    pub n: AtomicU32,
    pub w: AtomicU64, // f64 bits
    /// Welford M2 accumulator (f64 bits) — σᵢ² = M2/(N-1)
    pub m2: AtomicU64,
    /// Virtual visit count (integer, for PUCT N inflation)
    pub virtual_losses: AtomicI32,
    /// Virtual value penalty (f64 bits, for Q pessimism)
    pub virtual_value: AtomicU64,
}

impl<M: Copy + Send + Sync + 'static> MctsEdge<M> {
    pub fn new(mv: M, child: Arc<MctsNode<M>>, prior: f32) -> Self {
        MctsEdge {
            mv,
            child,
            p: prior.max(0.0),
            n: AtomicU32::new(0),
            w: AtomicU64::new(0),
            m2: AtomicU64::new(0),
            virtual_losses: AtomicI32::new(0),
            virtual_value: AtomicU64::new(0),
        }
    }

    /// 논문 §6.1.1: σᵢ = √(M2/(N-1)), N≥2 시 유효
    /// N<2이면 fallback (σ_Q proxy 불필요 — 호출자가 처리)
    pub fn edge_sigma(&self) -> Option<f32> {
        let n = self.n.load(Ordering::Acquire);
        if n < 2 {
            return None;
        }
        let m2 = f64::from_bits(self.m2.load(Ordering::Acquire));
        let var = (m2 / (n - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    }

    /// Q with split virtual loss: Q_eff = (W - Σvvalue) / (N + Σvvisit)
    #[inline]
    pub fn q_eff(&self) -> f32 {
        let n = self.n.load(Ordering::Acquire);
        let vl = self.virtual_losses.load(Ordering::Acquire).max(0) as u32;
        let n_eff = (n + vl).max(1) as f32;
        let w = atomic_f64_load(&self.w) as f32;
        let vv = atomic_f64_load(&self.virtual_value) as f32;
        (w - vv) / n_eff
    }

    /// Q — 실제 누적 (출력용, no VL)
    #[inline]
    pub fn q(&self) -> f32 {
        let n = self.n.load(Ordering::Acquire);
        if n == 0 {
            return 0.0;
        }
        atomic_f64_load(&self.w) as f32 / n as f32
    }

    /// Apply split virtual loss (called during select)
    #[inline]
    pub fn apply_vl(&self, vvisit: f32, vvalue: f32) {
        // vvisit: round to nearest integer for atomic increment
        let vi = vvisit.round() as i32;
        if vi > 0 {
            self.virtual_losses.fetch_add(vi, Ordering::AcqRel);
        }
        // vvalue: add to f64 accumulator
        if vvalue.abs() > 1e-9 {
            atomic_f64_add(&self.virtual_value, vvalue as f64);
        }
    }

    /// Remove split virtual loss (called during backup)
    #[inline]
    pub fn remove_vl(&self, vvisit: f32, vvalue: f32) {
        let vi = vvisit.round() as i32;
        if vi > 0 {
            self.virtual_losses.fetch_sub(vi, Ordering::AcqRel);
        }
        if vvalue.abs() > 1e-9 {
            atomic_f64_add(&self.virtual_value, -(vvalue as f64));
        }
    }

    pub fn add_w(&self, delta: f32) {
        atomic_f64_add(&self.w, delta as f64);
    }
}

#[derive(Clone)]
pub struct MctsEdgeSnapshot<M> {
    pub mv: M,
    pub child: Arc<MctsNode<M>>,
    pub p: f32,
    pub n: u32,
    pub w: f64,
    pub m2: f64,
    pub virtual_losses: i32,
    pub virtual_value: f64,
}

impl<M: Copy + Send + Sync + 'static> MctsEdgeSnapshot<M> {
    #[inline]
    pub fn q(&self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        (self.w as f32) / self.n as f32
    }

    #[inline]
    pub fn q_eff(&self) -> f32 {
        let vl = self.virtual_losses.max(0) as u32;
        let n_eff = (self.n + vl).max(1) as f32;
        ((self.w - self.virtual_value) as f32) / n_eff
    }

    pub fn edge_sigma(&self) -> Option<f32> {
        if self.n < 2 {
            return None;
        }
        let var = (self.m2 / (self.n - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    }
}

// ─────────────────────────────────────────────
// § PathEdge — backprop용 경로 기록 (with applied VL)
// ─────────────────────────────────────────────

pub struct PathEdge<M> {
    pub parent: Arc<MctsNode<M>>,
    pub edge_idx: usize,
    /// VL that was applied during select — must be removed during backup
    pub applied_vl: (f32, f32), // (vvisit, vvalue)
}

// ─────────────────────────────────────────────
// § MctsNode
// ─────────────────────────────────────────────

pub struct MctsNode<M> {
    pub hash: u64,
    pub terminal_value: Option<f32>,

    /// CAS exactly-once: (move, prior) 쌍, prior 내림차순 정렬
    pub candidates: OnceLock<Box<[(M, f32)]>>,

    /// Lazy materialized edges — append-only
    pub edges: RwLock<Vec<MctsEdge<M>>>,

    /// edges.len()의 단조 증가 근사
    pub edge_cursor: AtomicU32,

    pub n_total: AtomicU32,
    pub w_total: AtomicU64,

    // ── §6.3 MERGE channel: RTT holonomy residual curvature ──────
    // RTT(s) = Var[Q_γ(s)], γᵢ = 서로 다른 경로
    // TT hit 때마다 backed-up Q를 Welford online variance로 누적
    /// TT 히트 횟수 (서로 다른 경로 수)
    pub rtt_n: AtomicU32,
    /// Welford W: 누적 Q 합 (f64 bits)
    pub rtt_w: AtomicU64,
    /// Welford M2: 누적 분산 (f64 bits)
    pub rtt_m2: AtomicU64,
}

impl<M: Copy + Send + Sync + 'static> MctsNode<M> {
    pub fn new(hash: u64, terminal_value: Option<f32>) -> Arc<Self> {
        Arc::new(MctsNode {
            hash,
            terminal_value,
            candidates: OnceLock::new(),
            edges: RwLock::new(Vec::new()),
            edge_cursor: AtomicU32::new(0),
            n_total: AtomicU32::new(0),
            w_total: AtomicU64::new(0),
            rtt_n: AtomicU32::new(0),
            rtt_w: AtomicU64::new(0),
            rtt_m2: AtomicU64::new(0),
        })
    }

    pub fn is_expanded(&self) -> bool {
        self.candidates.get().is_some()
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.get().map(|c| c.len()).unwrap_or(0)
    }

    pub fn materialized_count(&self) -> usize {
        self.edge_cursor.load(Ordering::Acquire) as usize
    }

    #[inline]
    pub fn mean_q(&self) -> f32 {
        let n = self.n_total.load(Ordering::Acquire);
        if n == 0 {
            return 0.0;
        }
        atomic_f64_load(&self.w_total) as f32 / n as f32
    }

    /// §5.3 GVOC: impact weight wimp(s) = N(s)/N(root)
    /// root가 알려진 경우 루트 N(root) 전달
    pub fn wimp(&self, root_n: u32) -> f32 {
        if root_n == 0 {
            return 1.0;
        }
        let my_n = self.n_total.load(Ordering::Acquire);
        (my_n as f32 / root_n as f32).min(1.0)
    }

    /// backup.rs에서 TT hit path에 대해 호출
    pub fn record_rtt_hit(&self, q_value: f32) {
        let q = q_value as f64;
        let n_old = self.rtt_n.fetch_add(1, Ordering::AcqRel) as f64;
        let n_new = n_old + 1.0;
        // Welford update
        let w_old = f64::from_bits(self.rtt_w.load(Ordering::Acquire));
        let w_new = w_old + q;
        atomic_f64_add(&self.rtt_w, q);
        // M2 += (q - mean_old)(q - mean_new)
        if n_old >= 1.0 {
            let mean_old = w_old / n_old;
            let mean_new = w_new / n_new;
            let delta_m2 = (q - mean_old) * (q - mean_new);
            atomic_f64_add(&self.rtt_m2, delta_m2);
        }
    }

    /// RTT variance = M2 / (n-1)  →  σ_RTT = √Var
    /// None if fewer than 2 TT hits
    pub fn rtt_variance(&self) -> Option<f32> {
        let n = self.rtt_n.load(Ordering::Acquire) as f64;
        if n < 2.0 {
            return None;
        }
        let m2 = f64::from_bits(self.rtt_m2.load(Ordering::Acquire));
        Some(((m2 / (n - 1.0)).max(0.0)) as f32)
    }

    /// RwLock 내 Arc clone 스냅샷 — read lock으로 병렬 접근 허용
    pub fn edge_snapshot(&self, n_snap: usize) -> Vec<MctsEdgeSnapshot<M>> {
        let lock_started = crate::mcts::profiling::maybe_start_timer();
        let guard = self.edges.read().unwrap();
        if let Some(t0) = lock_started {
            record_edges_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        let n = n_snap.min(guard.len());
        guard[..n]
            .iter()
            .map(|edge| MctsEdgeSnapshot {
                mv: edge.mv,
                child: Arc::clone(&edge.child),
                p: edge.p,
                n: edge.n.load(Ordering::Acquire),
                w: atomic_f64_load(&edge.w),
                m2: atomic_f64_load(&edge.m2),
                virtual_losses: edge.virtual_losses.load(Ordering::Acquire),
                virtual_value: atomic_f64_load(&edge.virtual_value),
            })
            .collect()
    }

    /// Snapshot only the priors for the first `n_snap` edges, avoiding Arc clones
    /// when callers only need root prior values.
    pub fn edge_priors_snapshot(&self, n_snap: usize) -> Vec<f32> {
        let lock_started = crate::mcts::profiling::maybe_start_timer();
        let guard = self.edges.read().unwrap();
        if let Some(t0) = lock_started {
            record_edges_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        let n = n_snap.min(guard.len());
        guard[..n].iter().map(|edge| edge.p).collect()
    }

    /// Read-only access to the first `n_snap` edges without cloning the Arc list.
    pub fn with_edge_slice<R>(&self, n_snap: usize, f: impl FnOnce(&[MctsEdge<M>]) -> R) -> R {
        let lock_started = crate::mcts::profiling::maybe_start_timer();
        let guard = self.edges.read().unwrap();
        if let Some(t0) = lock_started {
            record_edges_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        let n = n_snap.min(guard.len());
        f(&guard[..n])
    }
}
