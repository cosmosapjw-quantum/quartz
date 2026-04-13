//! Expansion — Compact candidates + Lazy materialization
//!
//! Phase 1: expand_and_evaluate(node, state, evaluator, tt, pw)
//!   - evaluator.evaluate(state) → policy + value
//!   - candidates 저장: (mv, prior) 쌍, prior 내림차순, OnceLock CAS
//!   - k_initial개만 즉시 materialize (apply_move + TT 등록)
//!
//! Phase 2: materialize_edges<G>(node, state, target, tt)
//!   - select.rs에서 호출: state AT this node 전달
//!   - candidates[current..target] 에 대해 apply_move → TT 등록
//!   - Mutex 내 append-only push, double-check pattern

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::game::{EvalResult, Evaluator, GameState};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::node::{MctsEdge, MctsNode};
use crate::mcts::tt::TranspositionTable;

// ─────────────────────────────────────────────
// § Phase 1: expand_and_evaluate
// ─────────────────────────────────────────────

pub fn expand_and_evaluate<G: GameState>(
    node: &Arc<MctsNode<G::Move>>,
    state: &G,
    evaluator: &dyn Evaluator<G>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let eval = evaluator.evaluate(state);
    expand_with_result(node, state, eval, tt, pw)
}

pub fn expand_with_result<G: GameState>(
    node: &Arc<MctsNode<G::Move>>,
    state: &G,
    eval: EvalResult<G::Move>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let value = eval.value;

    let legal = state.legal_moves();
    if legal.is_empty() {
        return value;
    }

    // [OPT] Linear search instead of HashMap — avoids heap allocation for small move sets
    let policy = eval.policy;

    let mut candidates: Vec<(G::Move, f32)> = legal
        .iter()
        .map(|&mv| {
            let prior = policy
                .iter()
                .find(|(m, _)| *m == mv)
                .map(|(_, p)| *p)
                .unwrap_or(0.0)
                .max(0.0);
            (mv, prior)
        })
        .collect();
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let n_candidates = candidates.len();

    // ── 2. CAS exactly-once ─────────────────────────────────────
    let _ = node.candidates.set(candidates.into_boxed_slice());

    // ── 3. k_initial 개 즉시 materialization ────────────────────
    let k_initial = match pw {
        Some(cfg) => cfg.k(0).max(1).min(n_candidates),
        None => n_candidates,
    };
    materialize_edges(node, state, k_initial, tt);

    value
}

// ─────────────────────────────────────────────
// § Phase 2: materialize_edges
// ─────────────────────────────────────────────

/// `target`개까지 edges materialization
/// state: 이 노드에 해당하는 게임 상태 (apply_move용)
pub fn materialize_edges<G: GameState>(
    node: &Arc<MctsNode<G::Move>>,
    state: &G,
    target: usize,
    tt: &TranspositionTable<G::Move>,
) {
    let candidates = match node.candidates.get() {
        Some(c) => c,
        None => return,
    };
    let actual = target.min(candidates.len());

    // fast-path: 이미 충분히 materialized
    let current = node.edge_cursor.load(Ordering::Acquire) as usize;
    if current >= actual {
        return;
    }

    let lock_started = std::time::Instant::now();
    let mut guard = node.edges.write().unwrap();
    crate::mcts::node::record_edges_lock_wait(lock_started.elapsed().as_nanos() as u64);
    // double-check inside lock
    let current = node.edge_cursor.load(Ordering::Relaxed) as usize;
    if current < actual {
        let additional = actual.saturating_sub(guard.len());
        guard.reserve(additional);
        for i in current..actual {
            let (mv, prior) = candidates[i];
            let child_state = state.apply_move(mv);
            let child_hash = child_state.tt_hash();
            let tv = if child_state.is_terminal() {
                Some(child_state.outcome())
            } else {
                None
            };
            let child = tt.get_or_create(child_hash, tv);
            // §5.3 GVOC: child.parent = node (weak ptr)
            child.set_parent(node);
            guard.push(MctsEdge::new(mv, child, prior));
        }
        node.edge_cursor.store(actual as u32, Ordering::Release);
    }
}
