//! Negamax Backpropagation + Welford online variance
//!
//! 매 backup step:
//!   1. Virtual Loss 환원
//!   2. N += 1, W += value  (negamax 부호 반전)
//!   3. M2 Welford update: M2 += (value - Q_prev)(value - Q_new)
//!      → edge_sigma() = √(M2/(N-1)) = per-edge std 근사 (아래 A3-b 참고)
//!   4. parent.n_total += 1
//!
//! Writers are serialized per edge by stable-address stripes in node.rs.
//! Concurrent selection readers still see weak snapshots; joined N/W/M2
//! are coherent. RTT is actual incoming-edge sharing, not repeat visits.

use std::sync::atomic::Ordering;

use crate::game::GameState;
use crate::mcts::node::{atomic_f64_add, PathEdge};

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

        e.record_sample(value);
        // Parent perspective above, child perspective for incoming telemetry.
        e.child.record_incoming_sample(e as *const _ as usize,-value);

        atomic_f64_add(&pe.parent.w_total, value as f64);
        pe.parent.n_total.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod preflight07_tests {
    use super::*;
    use crate::games::gomoku15::{Gomoku15,GomokuVariant};
    use crate::mcts::{MctsEngine,MctsConfig};
    use crate::mcts::eval::UniformEval;
    use crate::mcts::node::MctsEdge;
    use crate::mcts::tt::leak_node;
    #[test]
    fn preflight_concurrent_moments_and_context() {
        use crate::mcts::node::atomic_f64_load;
        let child=leak_node::<u16>(123,None);
        let e1=MctsEdge::new(1,child,0.5);let e2=MctsEdge::new(2,child,0.5);
        std::thread::scope(|sc|{
            for k in 0..8{let e=&e1;sc.spawn(move||{
                for i in 0..2000{e.record_sample(if (i+k)%2==0{1.}else{-1.});}
            });}
        });
        assert_eq!(e1.n.load(Ordering::Acquire),16000);
        assert_eq!(atomic_f64_load(&e1.w),0.);
        assert!((atomic_f64_load(&e1.m2)-16000.).abs()<1e-7);
        let a=&e1 as *const _ as usize;let b=&e2 as *const _ as usize;
        for _ in 0..5{child.record_incoming_sample(a,1.);}
        assert_eq!(child.rtt_n.load(Ordering::Acquire),0);
        child.record_incoming_sample(b,-1.);
        child.record_incoming_sample(a,1.);
        assert_eq!(child.rtt_n.load(Ordering::Acquire),2);
        assert!((child.rtt_variance().unwrap()-2.).abs()<1e-6);
    }
    #[test]
    fn preflight_same_incoming_is_not_rtt() {
        let mut cfg=MctsConfig::evaluation(2.0);
        cfg.pw=Some(crate::mcts::PwConfig::new(1.0,0.0));cfg.max_depth=1;cfg.root_forced_win=false;
        let e=MctsEngine::new(Gomoku15::new(GomokuVariant::Freestyle),std::sync::Arc::new(UniformEval),cfg);
        for _ in 0..5{e.iterate();}
        let edge=&e.root.read_edges()[0];
        assert_eq!(edge.n.load(Ordering::Acquire),5);
        assert_eq!(edge.child.rtt_n.load(Ordering::Acquire),0,"one incoming edge is not a transposition");
    }
}
