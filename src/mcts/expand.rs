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
///
/// Phase 7 C (2026-04-26): edge buffer is a raw slab in the TT bucket's
/// bumpalo Bump (not a `Vec`). Allocation is done via
/// `tt.allocate_edge_slab(...)` (takes the bucket write lock for the
/// one-shot Bump alloc, then releases it). The fill loop runs under the
/// per-node `materialize_lock` (parking_lot::Mutex) and writes each
/// slot via `std::ptr::write`. The Release-store of `edge_cursor` at
/// the end of the loop publishes all written slots to lock-free
/// readers.
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

    // Allocate / look up the slab. Idempotent: only the first caller
    // takes the bucket write lock; subsequent calls observe the
    // published `edges_ptr` and return its (ptr, cap).
    let (slab_ptr, slab_cap) = tt.allocate_edge_slab::<G::Move>(&**node, candidates.len());
    debug_assert!(actual <= slab_cap as usize);

    // Acquire the per-node materialize lock. Wait time is fed into the
    // same telemetry channel that pre-Phase-7 recorded for the
    // `RwLock<Vec<...>>` write lock, so the diagnostic API
    // (`edge_lock_contention_snapshot`) keeps its semantic meaning.
    let lock_started = crate::mcts::profiling::maybe_start_timer();
    let _mat_guard = node.materialize_lock.lock();
    if let Some(t0) = lock_started {
        crate::mcts::node::record_edges_lock_wait(t0.elapsed().as_nanos() as u64);
    }

    // Double-check inside lock (another writer may have caught up).
    let current = node.edge_cursor.load(Ordering::Relaxed) as usize;
    if current >= actual {
        return;
    }

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
        // SAFETY: we hold `materialize_lock`, so no other writer is
        // touching slot `i`. `i < slab_cap` was established by the
        // `actual <= slab_cap` debug_assert above. The slot has not
        // been initialized before (cursor < i+1 at this point), so
        // `std::ptr::write` does NOT drop any prior value — it just
        // writes the new edge.
        unsafe {
            std::ptr::write(slab_ptr.add(i), MctsEdge::new(mv, child, prior));
        }
    }
    // Single Release-store publishes every slot we just wrote.
    // Lock-free readers see either the old cursor (and old slots) or
    // the new cursor (and all slots through `actual`).
    node.edge_cursor.store(actual as u32, Ordering::Release);
}
