//! Negamax Backpropagation + Welford online variance
//!
//! 매 backup step:
//!   1. Virtual Loss 환원
//!   2. N += 1, W += value  (negamax 부호 반전)
//!   3. M2 Welford update: M2 += (value - Q_prev)(value - Q_new)
//!      → edge_sigma() = √(M2/(N-1)) = 정확한 per-edge std
//!   4. parent.n_total += 1

use std::sync::atomic::Ordering;

use crate::game::GameState;
use crate::mcts::node::{atomic_f64_add, atomic_f64_load, PathEdge};

pub fn backprop<G: GameState>(path: &[PathEdge<G::Move>], leaf_value: f32) {
    let mut value = leaf_value;

    for pe in path.iter().rev() {
        value = -value; // negamax: 부모 관점 부호 반전

        // Phase 7 C (2026-04-26): lock-free slab read.
        let edges = pe.parent.read_edges();
        let e = &edges[pe.edge_idx];

        // Virtual Loss: remove the exact (vvisit, vvalue) applied during select
        let (vv, vq) = pe.applied_vl;
        e.remove_vl(vv, vq);

        // N 증가
        let n_old = e.n.fetch_add(1, Ordering::AcqRel) as f64;
        let n_new = n_old + 1.0;

        // Snapshot W before our update to compute correct Welford delta locally
        let w_before = atomic_f64_load(&e.w) as f64;

        // W 갱신
        e.add_w(value);

        // Welford M2 (per-edge σᵢ for §6.1.1)
        // Use locally computed w_old/w_new to avoid reading e.w again
        // (another thread may have updated it between add_w and the read)
        if n_old >= 1.0 {
            let mu_old = w_before / n_old;
            let mu_new = (w_before + value as f64) / n_new;
            let delta_m2 = (value as f64 - mu_old) * (value as f64 - mu_new);
            atomic_f64_add(&e.m2, delta_m2);
        }

        // §6.3 MERGE: 같은 엣지가 2번 이상 백업 = 여러 경로가 이 child에 수렴
        // n_old >= 1 (이번 backup 전 N이 1 이상) = 이전에도 방문됐음 = transposition
        if n_old >= 1.0 {
            e.child.record_rtt_hit(value);
        }
        // Phase 7 C: no guard to drop — slab read is lock-free.

        atomic_f64_add(&pe.parent.w_total, value as f64);
        pe.parent.n_total.fetch_add(1, Ordering::AcqRel);
    }
}
