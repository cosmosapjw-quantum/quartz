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

use crate::game::{EvalResult, Evaluator, GameState};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::node::{ArenaRef, MctsEdge, MctsNode};
use crate::mcts::tt::TranspositionTable;

// ─────────────────────────────────────────────
// § Phase 1: expand_and_evaluate
// ─────────────────────────────────────────────

pub fn expand_and_evaluate<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
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
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &G,
    eval: EvalResult<G::Move>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let value = eval.value;

    // [OPT] Use policy directly as candidates — avoids redundant legal_moves() call
    // and eliminates O(n²) policy-to-legal lookup.
    // Evaluator.evaluate() returns policy containing only legal moves with priors.
    let mut candidates = eval.policy;
    if candidates.is_empty() {
        return value;
    }

    // Clamp negative priors to 0
    for entry in candidates.iter_mut() {
        entry.1 = entry.1.max(0.0);
    }
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
    node: &ArenaRef<MctsNode<G::Move>>,
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

    let lock_started = crate::mcts::profiling::maybe_start_timer();
    let mut guard = node.edges.write();
    if let Some(t0) = lock_started {
        crate::mcts::node::record_edges_lock_wait(t0.elapsed().as_nanos() as u64);
    }
    // double-check inside lock
    let current = node.edge_cursor.load(Ordering::Relaxed) as usize;
    if current < actual {
        let additional = actual.saturating_sub(guard.len());
        guard.reserve(additional);
        // Phase 6.1: clone the parent state once and use apply_in_place +
        // undo_move per candidate to avoid an N-way per-candidate clone of
        // the (~1144 B for Gomoku-7) game state. Net effect on Gomoku-7 is
        // (N - 1) clones eliminated per materialize call.
        let mut probe = state.clone();
        for i in current..actual {
            let (mv, prior) = candidates[i];
            let undo = probe.apply_move_in_place(mv);
            let child_hash = probe.tt_hash();
            let tv = if probe.is_terminal() {
                Some(probe.outcome())
            } else {
                None
            };
            probe.undo_move(undo);
            let child = tt.get_or_create(child_hash, tv);
            guard.push(MctsEdge::new(mv, child, prior));
        }
        node.edge_cursor.store(actual as u32, Ordering::Release);
    }
}
