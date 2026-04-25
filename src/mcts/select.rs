//! PUCT Selection (v0.7)
//! Standard PUCT + Fisher Metric PUCT (§4 Information Geometry)
//! + EFT-PUCT (§6.1.2 one-loop bonus + visit penalty)

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::game::GameState;
use crate::mcts::expand::materialize_edges;
use crate::mcts::mod_types::PwConfig;
use crate::mcts::node::{atomic_f64_load, MctsEdge, MctsNode, PathEdge};
use crate::mcts::quartz::{
    eft_action_bonus, fisher_prior_weight, one_loop_visit_penalty, QuartzConfig, QuartzStats,
};
use crate::mcts::tt::TranspositionTable;

// ─────────────────────────────────────────────
// § PUCT Variants
// ─────────────────────────────────────────────

/// Standard AlphaZero PUCT (baseline)
#[inline]
pub fn puct_score(
    n_eff: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    n_parent_eff: u32,
    c_puct: f32,
) -> f32 {
    puct_score_with_parent_sqrt(
        n_eff,
        q_eff,
        prior,
        noise_adj,
        (n_parent_eff as f32).sqrt(),
        c_puct,
    )
}

#[inline]
fn puct_score_with_parent_sqrt(
    n_eff: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
) -> f32 {
    let p = (prior + noise_adj).max(0.0);
    let explore = c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32);
    q_eff + explore
}

/// (4) Fisher Metric PUCT — α = FISHER_ALPHA = 0.5 (fixed by natural gradient)
#[inline]
pub fn fisher_puct_score(
    n_eff: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    n_parent_eff: u32,
    c_puct: f32,
    _alpha: f32, // ignored — always 0.5 (Fisher metric)
) -> f32 {
    fisher_puct_score_with_parent_sqrt(
        n_eff,
        q_eff,
        prior,
        noise_adj,
        (n_parent_eff as f32).sqrt(),
        c_puct,
    )
}

#[inline]
fn fisher_puct_score_with_parent_sqrt(
    n_eff: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
) -> f32 {
    let p = (prior + noise_adj).max(0.0);
    let p_fish = fisher_prior_weight(p); // √π
    let explore = c_puct * p_fish * sqrt_n_parent_eff / (1.0 + n_eff as f32);
    q_eff + explore
}

/// EFT-PUCT: Fisher PUCT + one-loop bonus + one-loop visit penalty
/// = (4) Fisher natural gradient + (1) QFT diagonal one-loop
#[inline]
pub fn eft_puct_score(
    n_eff: u32,
    n_raw: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    n_parent_eff: u32,
    c_puct: f32,
    stats: &QuartzStats,
    _qcfg: &QuartzConfig, // reserved
) -> f32 {
    let base = fisher_puct_score(
        n_eff,
        q_eff,
        prior,
        noise_adj,
        n_parent_eff,
        c_puct,
        crate::mcts::quartz::FISHER_ALPHA,
    );
    let bonus = eft_action_bonus(stats);
    // One-loop diagonal: -ħ_eff/N_a (EFT-PUCT penalty)
    let penalty = one_loop_visit_penalty(
        n_raw,
        stats.hbar_eff,
        1.0,
        crate::mcts::quartz::HBAR_PENALTY_CAP,
    );
    base + bonus + penalty
}

#[inline]
fn blended_prior(base_prior: f32, alt_prior: f32, rho: f32) -> f32 {
    let rho = rho.clamp(0.0, 1.0);
    if rho <= 1e-6 {
        return base_prior.max(1e-8);
    }
    let log_p0 = base_prior.max(1e-8).ln();
    let log_p1 = alt_prior.max(1e-8).ln();
    ((1.0 - rho) * log_p0 + rho * log_p1).exp().max(1e-8)
}

#[inline]
fn root_share_penalty(n_raw: u32, sqrt_n_parent_eff: f32, hbar_eff: f32, cap: f32) -> f32 {
    if n_raw == 0 {
        return 0.0;
    }
    let n_parent_eff = (sqrt_n_parent_eff * sqrt_n_parent_eff).max(1.0);
    let h_cap = hbar_eff.min(cap);
    -(h_cap * n_raw as f32 / n_parent_eff)
}

/// PR-1B: Ablation-aware PUCT — respects penalty_mode and prior refresh settings.
/// SelfAdaptive mode: state-derived with fixed constants. All inputs from search observables.
#[inline]
pub fn ablation_puct_score(
    n_eff: u32,
    n_raw: u32,
    o_a: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    n_parent_eff: u32,
    c_puct: f32,
    stats: &QuartzStats,
    qcfg: &QuartzConfig,
) -> f32 {
    ablation_puct_score_with_parent_sqrt(
        n_eff,
        n_raw,
        o_a,
        q_eff,
        prior,
        noise_adj,
        (n_parent_eff as f32).sqrt(),
        c_puct,
        stats,
        qcfg,
    )
}

#[inline]
fn ablation_puct_score_with_parent_sqrt(
    n_eff: u32,
    n_raw: u32,
    o_a: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
    stats: &QuartzStats,
    qcfg: &QuartzConfig,
) -> f32 {
    use crate::mcts::quartz::PenaltyMode;

    // ═══════════════════════════════════════════
    // SelfAdaptive: zero config reads for penalty + refresh
    // All derived from: sigma_q, surprise_kl, n_visible, root_visits
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::SelfAdaptive {
        let k = (stats.n_visible as f32).max(2.0);
        let n_total = (stats.root_visits as f32).max(1.0);
        let sigma_q = stats.sigma_q.max(0.001);

        // ── Penalty: ν = σ_Q  →  -σ_Q / M_a ──
        let penalty = if n_raw > 0 || o_a > 0 {
            let m_a = 1.0 + n_raw as f32 + o_a as f32;
            -sigma_q / m_a
        } else {
            0.0
        };

        // ── Dynamic prior refresh: per-action Bayesian weight ──
        // α_a = N_a / (N_a + K)  — no global ρ, no hardcoded constants
        //   N_a=0 → α=0 (pure original prior)
        //   N_a=K → α=0.5 (balanced blend)
        //   N_a>>K → α→1 (fully visit-based)
        // τ = ln(1 + N_total/K)  — self-normalizing temperature
        let effective_prior = if n_raw > 0 {
            let n_a = n_raw as f32;
            let alpha_a = n_a / (n_a + k);
            let n_avg = n_total / k;
            let tau = (1.0 + n_avg).ln().max(0.1);
            let log_p0 = prior.max(1e-8).ln();
            let log_visit = (1.0 + n_a).ln();
            let log_pt = (1.0 - alpha_a) * log_p0 + alpha_a * log_visit / tau;
            log_pt.exp().max(1e-8)
        } else {
            prior
        };

        let p = (effective_prior + noise_adj).max(0.0);
        let base = q_eff + c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32);
        return base + penalty;
    }

    // ═══════════════════════════════════════════
    // Canonical GatedRefresh
    // Root-only visit-share penalty + divergence-gated visit refresh.
    //
    // - penalty(a) = -min(ħ_eff, cap) * N_a / N_parent
    // - gate opens when prior_q_divergence exceeds ε_t
    // - refresh blends original prior with visit share at the root
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::GatedRefresh {
        let penalty = root_share_penalty(
            n_raw,
            sqrt_n_parent_eff,
            stats.hbar_eff,
            qcfg.hbar_penalty_cap,
        );

        let effective_prior = if n_raw > 0 && stats.root_visits > 0 {
            let gate_threshold = stats.epsilon_t.max(1e-6);
            let divergence = stats.prior_q_divergence.max(0.0);
            if divergence > gate_threshold {
                let rho_t = ((divergence - gate_threshold) / divergence.max(gate_threshold))
                    .clamp(0.0, 1.0);
                let visit_share = (n_raw as f32 / stats.root_visits as f32).max(1e-8);
                blended_prior(prior, visit_share, rho_t)
            } else {
                prior
            }
        } else {
            prior
        };

        let p = (effective_prior + noise_adj).max(0.0);
        let base = if qcfg.enable_fisher_puct {
            let p_fish = fisher_prior_weight(p);
            q_eff + c_puct * p_fish * sqrt_n_parent_eff / (1.0 + n_eff as f32)
        } else {
            q_eff + c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32)
        };
        return base + penalty;
    }

    // ═══════════════════════════════════════════
    // Historical GatedRefresh heuristic kept for direct ablation
    // against the canonical doc-aligned controller.
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::GatedRefreshLegacy {
        let penalty = crate::mcts::quartz::effective_penalty_v2(n_raw, o_a, qcfg.hbar_penalty_cap);

        let flip_thresh = 0.159_f32;
        let rho_max = 0.3_f32;
        let rho_t = rho_max * (stats.p_flip / flip_thresh).min(1.0);

        // Honor config.prior_refresh_temp; fall back to the legacy 0.5 only when
        // the knob is explicitly zeroed (< 0.01), matching the GatedRefresh
        // manual_mode convention at the bottom of this file.
        let tau = if qcfg.prior_refresh_temp > 0.01 {
            qcfg.prior_refresh_temp
        } else {
            0.5_f32
        };
        let effective_prior = if n_raw > 0 && rho_t > 1e-4 {
            let log_p0 = prior.max(1e-8).ln();
            let log_pt = (1.0 - rho_t) * log_p0 + rho_t * q_eff / tau;
            log_pt.exp().max(1e-8)
        } else {
            prior
        };

        let p = (effective_prior + noise_adj).max(0.0);
        let base = q_eff + c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32);
        return base + penalty;
    }

    // ═══════════════════════════════════════════
    // Adaptive (H6): P_flip-mediated VF↔Q mixture + σ_Q-adaptive penalty
    //
    // P_flip high (uncertain) → Q-refresh corrects prior
    // P_flip low (converged) → VF-refresh concentrates search
    // penalty ν = max(cap, σ_Q) → captures SA's σ_Q amplification
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::PFlipMixture {
        let k = (stats.n_visible as f32).max(2.0);
        let n_total = (stats.root_visits as f32).max(1.0);
        let sigma_q = stats.sigma_q.max(0.001);

        // ── Penalty: ν = max(cap, σ_Q) ──
        let nu = qcfg.hbar_penalty_cap.max(sigma_q);
        let penalty = if n_raw > 0 || o_a > 0 {
            let m_a = 1.0 + n_raw as f32 + o_a as f32;
            -nu / m_a
        } else {
            0.0
        };

        // ── P_flip-mediated mixture ──
        let flip_thresh = 0.159_f32;
        let rho_max = 0.3_f32;
        let p_ratio = (stats.p_flip / flip_thresh).min(2.0); // cap at 2.0

        // Q-refresh: strong when P_flip high (uncertain)
        let rho_q = rho_max * p_ratio.min(1.0);
        // VF-refresh: strong when P_flip low (converged)
        let rho_vf = rho_max * (1.0 - p_ratio).max(0.0);

        let effective_prior = if n_raw > 0 && (rho_q + rho_vf) > 1e-4 {
            let log_p0 = prior.max(1e-8).ln();
            // Q-refresh temperature: follow config.prior_refresh_temp; fall back
            // to the legacy 0.5 only when the knob is explicitly zeroed (< 0.01).
            let tau_q = if qcfg.prior_refresh_temp > 0.01 {
                qcfg.prior_refresh_temp
            } else {
                0.5_f32
            };
            let q_signal = q_eff / tau_q;

            let n_a = n_raw as f32;
            let n_avg = n_total / k;
            let tau_vf = (1.0 + n_avg).ln().max(0.1);
            let vf_signal = (1.0 + n_a).ln() / tau_vf;

            let rho_total = rho_q + rho_vf;
            let log_pt = (1.0 - rho_total) * log_p0 + rho_q * q_signal + rho_vf * vf_signal;
            log_pt.exp().max(1e-8)
        } else {
            prior
        };

        let p = (effective_prior + noise_adj).max(0.0);
        let base = q_eff + c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32);
        return base + penalty;
    }

    // ═══════════════════════════════════════════
    // Non-SelfAdaptive: config-driven modes (backward compatible)
    // ═══════════════════════════════════════════

    // Dynamic prior refresh (manual/K-adaptive modes)
    let effective_prior = if qcfg.prior_refresh_rate > 1e-6 && n_raw > 0 {
        let manual_mode = qcfg.prior_refresh_temp > 0.01;
        let rho_effective = if manual_mode {
            qcfg.prior_refresh_rate
        } else {
            let k_ref = 10.0_f32;
            let k = (stats.n_visible as f32).max(2.0);
            qcfg.prior_refresh_rate * (k_ref / k).min(1.0)
        };
        if rho_effective > 1e-4 {
            if manual_mode {
                let tau = qcfg.prior_refresh_temp;
                let log_p0 = prior.max(1e-8).ln();
                let log_pt = (1.0 - rho_effective) * log_p0 + rho_effective * q_eff / tau.max(0.01);
                log_pt.exp().max(1e-8)
            } else {
                let log_p0 = prior.max(1e-8).ln();
                let log_visit = (1.0 + n_raw as f32).ln();
                let tau_visit = 2.0_f32;
                let log_pt = (1.0 - rho_effective) * log_p0 + rho_effective * log_visit / tau_visit;
                log_pt.exp().max(1e-8)
            }
        } else {
            prior
        }
    } else {
        prior
    };

    // Base PUCT
    let p = (effective_prior + noise_adj).max(0.0);
    let base = if qcfg.enable_fisher_puct {
        let p_fish = fisher_prior_weight(p);
        q_eff + c_puct * p_fish * sqrt_n_parent_eff / (1.0 + n_eff as f32)
    } else {
        q_eff + c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32)
    };

    // Off-diagonal bonus
    let bonus = if qcfg.enable_one_loop {
        eft_action_bonus(stats)
    } else {
        0.0
    };

    // Penalty dispatch
    let penalty = match qcfg.penalty_mode {
        PenaltyMode::Legacy => {
            if qcfg.enable_one_loop {
                one_loop_visit_penalty(n_raw, stats.hbar_eff, 1.0, qcfg.hbar_penalty_cap)
            } else {
                0.0
            }
        }
        PenaltyMode::EffectiveV2 => {
            crate::mcts::quartz::effective_penalty_v2(n_raw, o_a, qcfg.hbar_penalty_cap)
        }
        PenaltyMode::None => 0.0,
        PenaltyMode::SelfAdaptive => unreachable!(), // handled above
        PenaltyMode::GatedRefresh => unreachable!(), // handled above
        PenaltyMode::GatedRefreshLegacy => unreachable!(), // handled above
        PenaltyMode::PFlipMixture => unreachable!(), // handled above
    };

    base + bonus + penalty
}

// ─────────────────────────────────────────────
// § SelectResult
// ─────────────────────────────────────────────

pub struct SelectResult<G: GameState> {
    pub path: Vec<PathEdge<G::Move>>,
    pub leaf: Arc<MctsNode<G::Move>>,
    pub leaf_state: G,
}

#[derive(Clone, Copy, Debug)]
struct EdgeScoreSnapshot {
    n_raw: u32,
    o_a: u32,
    n_eff: u32,
    q_eff: f32,
    terminal_q: Option<f32>,
    prior: f32,
    noise_adj: f32,
    edge_sigma: Option<f32>,
}

#[inline]
fn snapshot_edge<M>(
    edge: &MctsEdge<M>,
    noise_adj: f32,
    need_sigma: bool,
    exact_terminal_value: bool,
) -> EdgeScoreSnapshot {
    let n_raw = edge.n.load(Ordering::Relaxed);
    let o_a = edge.virtual_losses.load(Ordering::Relaxed).max(0) as u32;
    let n_eff = n_raw + o_a;
    let q_eff = {
        let denom = n_eff.max(1) as f32;
        let w = atomic_f64_load(&edge.w) as f32;
        let vv = atomic_f64_load(&edge.virtual_value) as f32;
        (w - vv) / denom
    };
    let edge_sigma = if need_sigma && n_raw >= 2 {
        let m2 = f64::from_bits(edge.m2.load(Ordering::Acquire));
        let var = (m2 / (n_raw - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    } else {
        None
    };
    let terminal_q = if exact_terminal_value {
        edge.child.terminal_value.map(|v| -v)
    } else {
        None
    };

    EdgeScoreSnapshot {
        n_raw,
        o_a,
        n_eff,
        q_eff,
        terminal_q,
        prior: edge.p,
        noise_adj,
        edge_sigma,
    }
}

#[inline]
fn score_snapshot(
    snapshot: EdgeScoreSnapshot,
    depth: usize,
    parent_q: f32,
    parent_visits: u32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
    fpu_reduction: f32,
    quartz: Option<(&QuartzStats, &QuartzConfig)>,
) -> f32 {
    let q_eff = if let Some(terminal_q) = snapshot.terminal_q {
        terminal_q
    } else if snapshot.n_raw == 0 && snapshot.o_a == 0 && parent_visits >= 4 && fpu_reduction > 1e-6
    {
        parent_q - fpu_reduction * (1.0 - snapshot.prior)
    } else {
        snapshot.q_eff
    };

    match quartz {
        Some((stats, qcfg)) if depth == 0 => {
            // Full QUARTZ scoring at root
            let base = ablation_puct_score_with_parent_sqrt(
                snapshot.n_eff,
                snapshot.n_raw,
                snapshot.o_a,
                q_eff,
                snapshot.prior,
                snapshot.noise_adj,
                sqrt_n_parent_eff,
                c_puct,
                stats,
                qcfg,
            );
            let b1 = if stats.lambda_1loop > 1e-6 {
                crate::mcts::quartz::paper_b1loop_bonus(
                    snapshot.edge_sigma.unwrap_or(0.0),
                    stats.lambda_1loop,
                )
            } else {
                0.0
            };
            base + b1
        }
        Some((stats, qcfg)) if !qcfg.root_only_shaping && depth <= 3 => {
            // Historical shallow-tree blend preserved only for controller ablations.
            let depth_weight = 1.0 / (1.0 + depth as f32);
            let base_puct = puct_score_with_parent_sqrt(
                snapshot.n_eff,
                q_eff,
                snapshot.prior,
                snapshot.noise_adj,
                sqrt_n_parent_eff,
                c_puct,
            );
            let full_quartz = ablation_puct_score_with_parent_sqrt(
                snapshot.n_eff,
                snapshot.n_raw,
                snapshot.o_a,
                q_eff,
                snapshot.prior,
                snapshot.noise_adj,
                sqrt_n_parent_eff,
                c_puct,
                stats,
                qcfg,
            );
            let blended = depth_weight * full_quartz + (1.0 - depth_weight) * base_puct;
            let b1 = if stats.lambda_1loop > 1e-6 {
                crate::mcts::quartz::paper_b1loop_bonus(
                    snapshot.edge_sigma.unwrap_or(0.0),
                    stats.lambda_1loop,
                ) * depth_weight
            } else {
                0.0
            };
            blended + b1
        }
        _ => puct_score_with_parent_sqrt(
            snapshot.n_eff,
            q_eff,
            snapshot.prior,
            snapshot.noise_adj,
            sqrt_n_parent_eff,
            c_puct,
        ),
    }
}

#[inline]
fn should_replace_best(
    best_score: f32,
    best_idx: usize,
    candidate_score: f32,
    candidate_idx: usize,
) -> bool {
    best_score
        .partial_cmp(&candidate_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(candidate_idx.cmp(&best_idx))
        == std::cmp::Ordering::Less
}

// ─────────────────────────────────────────────
// § select
// ─────────────────────────────────────────────

pub fn select<G: GameState>(
    root: &Arc<MctsNode<G::Move>>,
    root_state: &G,
    c_puct: f32,
    root_prior_noise: Option<&[f32]>,
    pw: Option<&PwConfig>,
    max_depth: usize,
    tt: &TranspositionTable<G::Move>,
    // QUARTZ EFT-PUCT (None = standard PUCT)
    quartz: Option<(&QuartzStats, &QuartzConfig)>,
    exact_terminal_value: bool,
    fpu_reduction: f32,
    // Parallelism controller (1st-class runtime object)
    par_ctrl: &super::parallel::ParallelismController,
) -> SelectResult<G> {
    // Pre-size the path accumulator to avoid the first 1-3 Vec re-grows during
    // typical 10-20 ply selection traversals. `max_depth` is the engine's
    // configured ceiling when set; otherwise 16 is a reasonable default that
    // covers the vast majority of search depths in profiled workloads.
    // (Apr-25 profile audit Step 3 / P1-2.)
    let path_capacity = if max_depth > 0 { max_depth } else { 16 };
    let mut path = Vec::with_capacity(path_capacity);
    let mut cur_node = Arc::clone(root);
    let mut cur_state = root_state.clone();
    let mut depth = 0usize;

    loop {
        if cur_node.terminal_value.is_some()
            || !cur_node.is_expanded()
            || (max_depth > 0 && depth >= max_depth)
        {
            break;
        }

        let n_candidates = cur_node.candidate_count();
        if n_candidates == 0 {
            break;
        }

        let n_total = cur_node.n_total.load(Ordering::Acquire);
        let n_visible = match pw {
            Some(cfg) => {
                let base_k = cfg.k(n_total).max(1).min(n_candidates);
                // Depth-aware PW: reduce materialized edges at deeper nodes
                // Root (depth=0) gets full k; deeper nodes get progressively fewer
                if depth > 0 {
                    let depth_factor = 1.0 / (1.0 + 0.3 * depth as f32);
                    (base_k as f32 * depth_factor).ceil() as usize
                } else {
                    base_k
                }
            }
            None => n_candidates,
        };

        let n_mat = cur_node.materialized_count();
        if n_mat < n_visible {
            materialize_edges(&cur_node, &cur_state, n_visible, tt);
        }

        // [OPT] Hold RwLock read guard directly instead of edge_snapshot (avoids N Arc clones)
        let guard = cur_node.edges.read();
        let n_edges = n_visible.min(guard.len());
        if n_edges == 0 {
            drop(guard);
            break;
        }
        let edges = &guard[..n_edges];

        let vl_sum: u32 = edges
            .iter()
            .map(|e| e.virtual_losses.load(Ordering::Relaxed).max(0) as u32)
            .sum();
        let n_parent_eff = n_total + vl_sum;

        let parent_q = cur_node.mean_q();
        let parent_visits = n_total;
        let sqrt_n_parent_eff = (n_parent_eff as f32).sqrt();
        let need_sigma = matches!(quartz, Some((stats, qcfg)) if (depth == 0 || (!qcfg.root_only_shaping && depth <= 3)) && stats.lambda_1loop > 1e-6);
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_has_pending = false;

        for (idx, edge) in edges.iter().enumerate() {
            let noise_adj = if depth == 0 {
                root_prior_noise
                    .and_then(|noise| noise.get(idx).copied())
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let snapshot = snapshot_edge(edge, noise_adj, need_sigma, exact_terminal_value);
            let score = score_snapshot(
                snapshot,
                depth,
                parent_q,
                parent_visits,
                sqrt_n_parent_eff,
                c_puct,
                fpu_reduction,
                quartz,
            );
            if idx == 0 || should_replace_best(best_score, best_idx, score, idx) {
                best_idx = idx;
                best_score = score;
                best_has_pending = snapshot.o_a > 0;
            }
        }

        // Parallelism telemetry: record pending count at this node
        par_ctrl.telemetry.record_select(vl_sum);

        // Detect duplicate path: if selected edge already has pending VL
        if best_has_pending {
            par_ctrl.telemetry.record_dup_leaf();
        }

        let vl_split = par_ctrl.vl_at_depth(depth as u32);
        edges[best_idx].apply_vl(vl_split.vvisit, vl_split.vvalue);
        par_ctrl.telemetry.record_vvalue(vl_split.vvalue);

        let best_mv = edges[best_idx].mv;
        let next_node = Arc::clone(&edges[best_idx].child);
        drop(guard); // release Mutex before apply_move

        cur_state = cur_state.apply_move(best_mv);
        path.push(PathEdge {
            parent: Arc::clone(&cur_node),
            edge_idx: best_idx,
            applied_vl: (vl_split.vvisit, vl_split.vvalue),
        });
        cur_node = next_node;
        depth += 1;
    }

    SelectResult {
        path,
        leaf: cur_node,
        leaf_state: cur_state,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::quartz::{PenaltyMode, QuartzConfig, QuartzStats};
    use std::hint::black_box;
    use std::time::Instant;

    fn linear_best_index(
        snapshots: &[EdgeScoreSnapshot],
        depth: usize,
        parent_q: f32,
        parent_visits: u32,
        sqrt_n_parent_eff: f32,
        c_puct: f32,
        fpu_reduction: f32,
        quartz: Option<(&QuartzStats, &QuartzConfig)>,
    ) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (idx, snapshot) in snapshots.iter().copied().enumerate() {
            let score = score_snapshot(
                snapshot,
                depth,
                parent_q,
                parent_visits,
                sqrt_n_parent_eff,
                c_puct,
                fpu_reduction,
                quartz,
            );
            if idx == 0 || should_replace_best(best_score, best_idx, score, idx) {
                best_idx = idx;
                best_score = score;
            }
        }
        best_idx
    }

    fn reference_best_index(
        snapshots: &[EdgeScoreSnapshot],
        depth: usize,
        parent_q: f32,
        parent_visits: u32,
        sqrt_n_parent_eff: f32,
        c_puct: f32,
        fpu_reduction: f32,
        quartz: Option<(&QuartzStats, &QuartzConfig)>,
    ) -> usize {
        snapshots
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                let sa = score_snapshot(
                    **a,
                    depth,
                    parent_q,
                    parent_visits,
                    sqrt_n_parent_eff,
                    c_puct,
                    fpu_reduction,
                    quartz,
                );
                let sb = score_snapshot(
                    **b,
                    depth,
                    parent_q,
                    parent_visits,
                    sqrt_n_parent_eff,
                    c_puct,
                    fpu_reduction,
                    quartz,
                );
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(j.cmp(i))
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn test_linear_scan_matches_reference_puct() {
        let snapshots = vec![
            EdgeScoreSnapshot {
                n_raw: 8,
                o_a: 0,
                n_eff: 8,
                q_eff: 0.42,
                terminal_q: None,
                prior: 0.30,
                noise_adj: 0.00,
                edge_sigma: None,
            },
            EdgeScoreSnapshot {
                n_raw: 2,
                o_a: 1,
                n_eff: 3,
                q_eff: 0.10,
                terminal_q: None,
                prior: 0.45,
                noise_adj: 0.05,
                edge_sigma: None,
            },
            EdgeScoreSnapshot {
                n_raw: 0,
                o_a: 0,
                n_eff: 0,
                q_eff: 0.00,
                terminal_q: None,
                prior: 0.25,
                noise_adj: 0.00,
                edge_sigma: None,
            },
        ];
        let sqrt_n_parent_eff = 4.0;
        let c_puct = 2.0;
        let parent_q = 0.35;

        let expected = reference_best_index(
            &snapshots,
            1,
            parent_q,
            8,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            None,
        );
        let best_idx = linear_best_index(
            &snapshots,
            1,
            parent_q,
            8,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            None,
        );
        assert_eq!(best_idx, expected);
    }

    #[test]
    fn test_linear_scan_matches_reference_quartz_root() {
        let snapshots = vec![
            EdgeScoreSnapshot {
                n_raw: 10,
                o_a: 1,
                n_eff: 11,
                q_eff: 0.30,
                terminal_q: None,
                prior: 0.34,
                noise_adj: 0.02,
                edge_sigma: Some(0.08),
            },
            EdgeScoreSnapshot {
                n_raw: 5,
                o_a: 0,
                n_eff: 5,
                q_eff: 0.28,
                terminal_q: None,
                prior: 0.33,
                noise_adj: 0.01,
                edge_sigma: Some(0.11),
            },
            EdgeScoreSnapshot {
                n_raw: 1,
                o_a: 0,
                n_eff: 1,
                q_eff: 0.15,
                terminal_q: None,
                prior: 0.33,
                noise_adj: 0.00,
                edge_sigma: None,
            },
        ];
        let sqrt_n_parent_eff = 6.0;
        let c_puct = 2.25;
        let stats = QuartzStats {
            lambda_1loop: 0.09,
            sigma_q: 0.12,
            p_flip: 0.04,
            root_visits: 36,
            n_visible: 3,
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            ..Default::default()
        };

        let expected = reference_best_index(
            &snapshots,
            0,
            0.22,
            12,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            Some((&stats, &qcfg)),
        );
        let best_idx = linear_best_index(
            &snapshots,
            0,
            0.22,
            12,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            Some((&stats, &qcfg)),
        );
        assert_eq!(best_idx, expected);
    }

    #[test]
    fn test_quartz_scoring_is_root_only() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 6,
            o_a: 0,
            n_eff: 6,
            q_eff: 0.22,
            terminal_q: None,
            prior: 0.45,
            noise_adj: 0.0,
            edge_sigma: Some(0.08),
        };
        let stats = QuartzStats {
            lambda_1loop: 0.11,
            sigma_q: 0.18,
            hbar_eff: 0.6,
            p_flip: 0.03,
            prior_q_divergence: 0.8,
            epsilon_t: 0.1,
            root_visits: 24,
            n_visible: 4,
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            ..Default::default()
        };

        let quartz_nonroot =
            score_snapshot(snapshot, 1, 0.20, 24, 5.0, 2.0, 0.0, Some((&stats, &qcfg)));
        let plain_nonroot = puct_score_with_parent_sqrt(
            snapshot.n_eff,
            snapshot.q_eff,
            snapshot.prior,
            snapshot.noise_adj,
            5.0,
            2.0,
        );
        assert!((quartz_nonroot - plain_nonroot).abs() < 1e-6);
    }

    #[test]
    fn test_gated_refresh_ignores_pflip_when_other_state_matches() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.70,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            hbar_penalty_cap: 0.3,
            ..Default::default()
        };
        let stats_a = QuartzStats {
            hbar_eff: 0.8,
            p_flip: 0.01,
            prior_q_divergence: 0.6,
            epsilon_t: 0.1,
            root_visits: 20,
            n_visible: 5,
            ..Default::default()
        };
        let stats_b = QuartzStats {
            p_flip: 0.95,
            ..stats_a.clone()
        };

        let score_a = score_snapshot(snapshot, 0, 0.0, 20, 5.0, 2.0, 0.0, Some((&stats_a, &qcfg)));
        let score_b = score_snapshot(snapshot, 0, 0.0, 20, 5.0, 2.0, 0.0, Some((&stats_b, &qcfg)));
        assert!((score_a - score_b).abs() < 1e-6);
    }

    #[test]
    fn test_gated_refresh_opens_on_prior_divergence() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.70,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            hbar_penalty_cap: 0.3,
            ..Default::default()
        };
        let gate_off = QuartzStats {
            hbar_eff: 0.8,
            prior_q_divergence: 0.05,
            epsilon_t: 0.10,
            root_visits: 20,
            n_visible: 5,
            ..Default::default()
        };
        let gate_on = QuartzStats {
            prior_q_divergence: 0.60,
            ..gate_off.clone()
        };

        let score_off = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&gate_off, &qcfg)),
        );
        let score_on = score_snapshot(snapshot, 0, 0.0, 20, 5.0, 2.0, 0.0, Some((&gate_on, &qcfg)));
        assert!(
            score_on < score_off,
            "visit-share refresh should pull a 0.70 prior toward the 0.20 visit share"
        );
    }

    #[test]
    fn test_legacy_gated_refresh_depends_on_pflip() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.20,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefreshLegacy,
            hbar_penalty_cap: 0.3,
            ..Default::default()
        };
        let stats_low = QuartzStats {
            p_flip: 0.01,
            prior_q_divergence: 0.9,
            epsilon_t: 0.1,
            root_visits: 20,
            n_visible: 5,
            ..Default::default()
        };
        let stats_high = QuartzStats {
            p_flip: 0.15,
            ..stats_low.clone()
        };

        let score_low = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_low, &qcfg)),
        );
        let score_high = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_high, &qcfg)),
        );
        assert!(score_high > score_low);
    }

    #[test]
    fn test_legacy_profile_can_shape_shallow_nonroot_nodes() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 6,
            o_a: 0,
            n_eff: 6,
            q_eff: 0.22,
            terminal_q: None,
            prior: 0.45,
            noise_adj: 0.0,
            edge_sigma: Some(0.08),
        };
        let stats = QuartzStats {
            lambda_1loop: 0.11,
            sigma_q: 0.18,
            hbar_eff: 0.6,
            p_flip: 0.10,
            prior_q_divergence: 0.8,
            epsilon_t: 0.1,
            root_visits: 24,
            n_visible: 4,
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefreshLegacy,
            root_only_shaping: false,
            ..Default::default()
        };

        let shallow_score =
            score_snapshot(snapshot, 1, 0.20, 24, 5.0, 2.0, 0.0, Some((&stats, &qcfg)));
        let plain_nonroot = puct_score_with_parent_sqrt(
            snapshot.n_eff,
            snapshot.q_eff,
            snapshot.prior,
            snapshot.noise_adj,
            5.0,
            2.0,
        );
        assert!((shallow_score - plain_nonroot).abs() > 1e-5);
    }

    #[test]
    fn test_fpu_improves_unvisited_child_score_without_affecting_visited_edge() {
        let snapshots = vec![
            EdgeScoreSnapshot {
                n_raw: 4,
                o_a: 0,
                n_eff: 4,
                q_eff: 0.05,
                terminal_q: None,
                prior: 0.30,
                noise_adj: 0.0,
                edge_sigma: None,
            },
            EdgeScoreSnapshot {
                n_raw: 0,
                o_a: 0,
                n_eff: 0,
                q_eff: 0.0,
                terminal_q: None,
                prior: 0.55,
                noise_adj: 0.0,
                edge_sigma: None,
            },
        ];

        let no_fpu = linear_best_index(&snapshots, 1, 0.40, 8, 4.0, 2.0, 0.0, None);
        let with_fpu = linear_best_index(&snapshots, 1, 0.40, 8, 4.0, 2.0, 0.20, None);
        assert_eq!(no_fpu, 1);
        assert_eq!(with_fpu, 1);

        let unvisited_no_fpu = score_snapshot(snapshots[1], 1, 0.40, 8, 4.0, 2.0, 0.0, None);
        let unvisited_with_fpu = score_snapshot(snapshots[1], 1, 0.40, 8, 4.0, 2.0, 0.20, None);
        let visited_no_fpu = score_snapshot(snapshots[0], 1, 0.40, 8, 4.0, 2.0, 0.0, None);
        let visited_with_fpu = score_snapshot(snapshots[0], 1, 0.40, 8, 4.0, 2.0, 0.20, None);
        assert!(unvisited_with_fpu > unvisited_no_fpu);
        assert!((visited_with_fpu - visited_no_fpu).abs() < 1e-6);
    }

    #[test]
    fn test_should_replace_best_keeps_lower_index_on_tie_and_nan() {
        assert!(!should_replace_best(1.0, 0, 1.0, 1));
        assert!(!should_replace_best(f32::NAN, 0, 2.0, 1));
        assert!(should_replace_best(0.5, 1, 0.6, 2));
    }

    #[test]
    #[ignore]
    fn bench_linear_scan_vs_reference() {
        let snapshots: Vec<_> = (0..128)
            .map(|idx| EdgeScoreSnapshot {
                n_raw: (idx * 7 % 19) as u32,
                o_a: (idx % 3) as u32,
                n_eff: (idx * 7 % 19) as u32 + (idx % 3) as u32,
                q_eff: ((idx as f32 * 0.137).sin() * 0.5).clamp(-1.0, 1.0),
                terminal_q: None,
                prior: 1.0 / 128.0 + (idx as f32 * 0.017).cos().abs() * 0.01,
                noise_adj: if idx < 32 {
                    (idx as f32 * 0.031).sin().abs() * 0.02
                } else {
                    0.0
                },
                edge_sigma: Some(0.05 + idx as f32 * 0.001),
            })
            .collect();
        let stats = QuartzStats {
            lambda_1loop: 0.09,
            sigma_q: 0.18,
            p_flip: 0.07,
            root_visits: 512,
            n_visible: snapshots.len(),
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            ..Default::default()
        };
        let sqrt_n_parent_eff = (640.0f32).sqrt();
        let c_puct = 2.25;
        let reps = 200_000usize;

        let t_ref = Instant::now();
        let mut acc_ref = 0usize;
        for _ in 0..reps {
            acc_ref ^= black_box(reference_best_index(
                black_box(&snapshots),
                0,
                0.18,
                32,
                sqrt_n_parent_eff,
                c_puct,
                0.20,
                Some((&stats, &qcfg)),
            ));
        }
        let ref_ns = t_ref.elapsed().as_nanos();

        let t_new = Instant::now();
        let mut acc_new = 0usize;
        for _ in 0..reps {
            acc_new ^= black_box(linear_best_index(
                black_box(&snapshots),
                0,
                0.18,
                32,
                sqrt_n_parent_eff,
                c_puct,
                0.20,
                Some((&stats, &qcfg)),
            ));
        }
        let new_ns = t_new.elapsed().as_nanos();

        assert_eq!(acc_new, acc_ref);
        eprintln!(
            "[BENCH] select reference={}ns linear={}ns speedup={:.2}x",
            ref_ns,
            new_ns,
            ref_ns as f64 / new_ns.max(1) as f64
        );
    }
}
