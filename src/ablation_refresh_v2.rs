//! Prior Refresh 비판적 분석 + 추가 Ablation Study
//!
//! ## 핵심 발견: SelfAdaptive는 이미 refresh를 포함한다
//!
//! mcts__select.rs 97-130행:
//!   SelfAdaptive = penalty(σ_Q/M_a) + built-in_refresh(visit-frequency, α_a=N/(N+K))
//!
//! 따라서 Phase 1A-ext의 "SA vs NoPen+Refresh" 비교는:
//!   (penalty + refresh) vs (no penalty + refresh) = penalty 효과 검증
//!
//! 진정한 refresh 분리 검증은:
//!   (penalty-only) vs (penalty + refresh) = **EffV2 vs SelfAdaptive**
//!
//! Phase 1A-ext 재해석:
//!   EffV2_0.3 (penalty only) = 0.200
//!   SelfAdaptive (penalty + refresh) = 0.125
//!   Δ = 0.075 → refresh가 penalty 위에 추가적 이득을 줄 수 있음 (CI 겹침으로 미확정)
//!
//! ## 추가 실험 설계
//!
//! Exp-A: BiasedEval — 잘못된 prior에서 refresh의 교정 능력
//! Exp-B: Low-budget regime — prior 영향이 큰 조건
//! Exp-C: Gomoku 7×7 — larger action space에서의 domain dependence
//! Exp-D: EffV2 matched-ν — SA의 penalty를 분리 재현

use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use std::sync::Arc;

use crate::ablation_refresh::{
    bootstrap_ci, cis_overlap, gen_gomoku_positions, gen_ttt_positions, measure_flips,
    print_result, FlipResult,
};
use crate::game::{EvalResult, Evaluator, GameState};
use crate::games::tictactoe::TicTacToe;
use crate::games::Gomoku;
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController};
use crate::mcts::{MctsConfig, MctsEngine};

// ═══════════════════════════════════════════════════════
// § BiasedEval: 체계적으로 잘못된 prior
// ═══════════════════════════════════════════════════════

/// Evaluator that assigns biased priors — first K/3 actions get 80% mass,
/// the rest share 20%. This simulates a "confidently wrong" NN.
pub struct BiasedEval {
    pub bias_fraction: f32, // fraction of actions that get most mass
    pub bias_mass: f32,     // total mass assigned to biased actions
}

impl BiasedEval {
    pub fn strongly_biased() -> Self {
        BiasedEval {
            bias_fraction: 0.33,
            bias_mass: 0.80,
        }
    }
    pub fn mildly_biased() -> Self {
        BiasedEval {
            bias_fraction: 0.33,
            bias_mass: 0.60,
        }
    }
}

impl Evaluator<TicTacToe> for BiasedEval {
    fn evaluate(&self, state: &TicTacToe) -> EvalResult<usize> {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult::uniform(
                &[],
                if state.is_terminal() {
                    state.outcome()
                } else {
                    0.0
                },
            );
        }
        let n = legal.len();
        let n_biased = ((n as f32 * self.bias_fraction) as usize).max(1).min(n);
        let p_high = self.bias_mass / n_biased as f32;
        let p_low = (1.0 - self.bias_mass) / (n - n_biased).max(1) as f32;

        // Assign high prior to first n_biased moves (which may NOT be the best)
        let policy: Vec<(usize, f32)> = legal
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let p = if i < n_biased { p_high } else { p_low };
                (m, p)
            })
            .collect();

        // Random rollout for value (same as ShortRollout)
        let value = random_playout(state, 20);
        EvalResult { policy, value }
    }
}

impl Evaluator<Gomoku> for BiasedEval {
    fn evaluate(&self, state: &Gomoku) -> EvalResult<usize> {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult::uniform(
                &[],
                if state.is_terminal() {
                    state.outcome()
                } else {
                    0.0
                },
            );
        }
        let n = legal.len();
        let n_biased = ((n as f32 * self.bias_fraction) as usize).max(1).min(n);
        let p_high = self.bias_mass / n_biased as f32;
        let p_low = (1.0 - self.bias_mass) / (n - n_biased).max(1) as f32;
        let policy: Vec<(usize, f32)> = legal
            .iter()
            .enumerate()
            .map(|(i, &m)| (m, if i < n_biased { p_high } else { p_low }))
            .collect();
        let value = random_playout(state, 20);
        EvalResult { policy, value }
    }
}

fn random_playout<G: GameState>(state: &G, max_depth: usize) -> f32 {
    let mut s = state.clone();
    let root = state.current_player();
    let mut rng_state = state
        .hash()
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    for _ in 0..max_depth {
        if s.is_terminal() {
            let flip = if s.current_player() == root {
                1.0
            } else {
                -1.0
            };
            return s.outcome() * flip;
        }
        let legal = s.legal_moves();
        if legal.is_empty() {
            break;
        }
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s = s.apply_move(legal[(rng_state >> 33) as usize % legal.len()]);
    }
    0.0
}

// ═══════════════════════════════════════════════════════
// § Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────
    // Exp-A: BiasedEval — refresh의 prior 교정 능력
    // ────────────────────────────────────────

    #[test]
    #[ignore]
    fn exp_a_biased_prior_ttt() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Exp-A: BiasedEval TTT (N=200) — Refresh의 잘못된 prior 교정 능력");
        eprintln!("  Hypothesis: 잘못된 prior 하에서 refresh가 교정적 역할을 할 수 있다.");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_ttt_positions(200, 42);
        let eval: Arc<BiasedEval> = Arc::new(BiasedEval::strongly_biased());
        let base = MctsConfig::evaluation(2.0);
        let (b1, b2) = (2000u32, 10000u32);

        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "NoPenalty",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.3 (pen only)",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "SelfAdaptive (pen+ref)",
                QuartzConfig {
                    penalty_mode: PenaltyMode::SelfAdaptive,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "NoPen+ExtRefresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
            (
                "EffV2+ExtRefresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
        ];

        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }

        eprintln!("\n  ── Analysis ──");
        let pen_only = &results[1]; // EffV2
        let pen_ref = &results[2]; // SA (pen+refresh)
        let overlap = cis_overlap(pen_only.ci, pen_ref.ci);
        eprintln!(
            "  EffV2(pen) vs SA(pen+ref): Δ={:.3}, overlap={}",
            pen_only.rate - pen_ref.rate,
            overlap
        );
        if pen_ref.rate < pen_only.rate && !overlap {
            eprintln!("  → Refresh HELPS with biased prior (SIGNIFICANT)");
        } else if pen_ref.rate < pen_only.rate {
            eprintln!("  → Refresh might help (not significant)");
        } else {
            eprintln!("  → Refresh does NOT help even with biased prior");
        }
        eprintln!();
    }

    // ────────────────────────────────────────
    // Exp-B: Low-budget regime
    // ────────────────────────────────────────

    #[test]
    #[ignore]
    fn exp_b_low_budget_ttt() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Exp-B: Low Budget TTT (200/2000 vs 2000/10000) — N=200");
        eprintln!("  Hypothesis: 낮은 budget에서 prior 영향이 커져 refresh가 더 유효할 수 있다.");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_ttt_positions(200, 42);
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let base = MctsConfig::evaluation(2.0);

        for (b1, b2, label) in [(200, 2000, "LOW(200/2K)"), (2000, 10000, "HIGH(2K/10K)")] {
            eprintln!("  ── {} ──", label);
            let configs: Vec<(&str, QuartzConfig)> = vec![
                (
                    "NoPenalty",
                    QuartzConfig {
                        penalty_mode: PenaltyMode::None,
                        min_visits: 15,
                        check_interval: 30,
                        ..Default::default()
                    },
                ),
                (
                    "EffV2_0.3",
                    QuartzConfig {
                        penalty_mode: PenaltyMode::EffectiveV2,
                        hbar_penalty_cap: 0.3,
                        min_visits: 15,
                        check_interval: 30,
                        ..Default::default()
                    },
                ),
                (
                    "SelfAdaptive",
                    QuartzConfig {
                        penalty_mode: PenaltyMode::SelfAdaptive,
                        min_visits: 15,
                        check_interval: 30,
                        ..Default::default()
                    },
                ),
                (
                    "NoPen+Refresh",
                    QuartzConfig {
                        penalty_mode: PenaltyMode::None,
                        min_visits: 15,
                        check_interval: 30,
                        prior_refresh_rate: 0.3,
                        prior_refresh_temp: 0.5,
                        ..Default::default()
                    },
                ),
            ];
            for (clabel, qcfg) in &configs {
                let r = measure_flips(
                    clabel,
                    &positions,
                    &eval,
                    b1 as u32,
                    b2 as u32,
                    qcfg.clone(),
                    base.clone(),
                );
                print_result(&r);
            }
            eprintln!();
        }
    }

    // ────────────────────────────────────────
    // Exp-C: Gomoku 7×7 larger action space
    // ────────────────────────────────────────

    #[test]
    #[ignore]
    fn exp_c_gomoku_refresh() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Exp-C: Gomoku 7×7 + ShortRollout (N=50) — Budget 300/1200");
        eprintln!("  Hypothesis: 큰 action space(K=49)에서 refresh 효과가 다를 수 있다.");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_gomoku_positions(50, 42);
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
        let base = MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku());
        let (b1, b2) = (300u32, 1200u32);

        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "NoPenalty",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                },
            ),
            (
                "Legacy_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.3,
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                },
            ),
            (
                "SelfAdaptive",
                QuartzConfig {
                    penalty_mode: PenaltyMode::SelfAdaptive,
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                },
            ),
            (
                "NoPen+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    min_visits: 15,
                    check_interval: 20,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
            (
                "Leg+Refresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::Legacy,
                    hbar_penalty_cap: 0.3,
                    min_visits: 15,
                    check_interval: 20,
                    prior_refresh_rate: 0.3,
                    prior_refresh_temp: 0.5,
                    ..Default::default()
                },
            ),
        ];

        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }

        eprintln!("\n  ── Key Comparisons ──");
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                if (results[i].rate - results[j].rate).abs() > 0.05 {
                    let ov = cis_overlap(results[i].ci, results[j].ci);
                    eprintln!(
                        "  {} vs {}: Δ={:.3} {}",
                        results[i].label,
                        results[j].label,
                        results[i].rate - results[j].rate,
                        if ov { "(overlap)" } else { "*** SIG ***" }
                    );
                }
            }
        }
        eprintln!();
    }

    // ────────────────────────────────────────
    // Exp-D: SA 내부 refresh의 독립 기여도 측정
    // ────────────────────────────────────────

    #[test]
    #[ignore]
    fn exp_d_sa_refresh_isolation() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  Exp-D: SelfAdaptive 내부 refresh 기여도 분리 (N=200)");
        eprintln!("  SA = penalty(σ_Q/M) + refresh(visit-freq, α=N/(N+K))");
        eprintln!("  EffV2 ≈ penalty만 (σ_Q와 cap 매칭 시 SA penalty 근사)");
        eprintln!("  Δ(SA - EffV2) ≈ refresh의 독립 기여도");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_ttt_positions(200, 42);
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let base = MctsConfig::evaluation(2.0);
        let (b1, b2) = (2000u32, 10000u32);

        // σ_Q ≈ 0.30 (from Phase 1A SA results)
        // Test EffV2 with various caps to find the closest match
        let configs: Vec<(&str, QuartzConfig)> = vec![
            (
                "EffV2_0.20",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.20,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.25",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.25,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.30",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.30,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "EffV2_0.35",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.35,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
            (
                "SelfAdaptive",
                QuartzConfig {
                    penalty_mode: PenaltyMode::SelfAdaptive,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
        ];

        let mut results = Vec::new();
        for (label, qcfg) in &configs {
            let r = measure_flips(label, &positions, &eval, b1, b2, qcfg.clone(), base.clone());
            print_result(&r);
            results.push(r);
        }

        let sa = results.last().unwrap();
        let best_effv2 = results[..4]
            .iter()
            .min_by(|a, b| a.rate.partial_cmp(&b.rate).unwrap())
            .unwrap();
        eprintln!("\n  ── Refresh Isolation ──");
        eprintln!(
            "  Best EffV2 (penalty only): {} = {:.3} [{:.3},{:.3}]",
            best_effv2.label, best_effv2.rate, best_effv2.ci.0, best_effv2.ci.1
        );
        eprintln!(
            "  SelfAdaptive (pen+refresh): {} = {:.3} [{:.3},{:.3}]",
            sa.label, sa.rate, sa.ci.0, sa.ci.1
        );
        let delta = best_effv2.rate - sa.rate;
        let overlap = cis_overlap(best_effv2.ci, sa.ci);
        eprintln!(
            "  Δ(refresh contribution) = {:.3} {}",
            delta,
            if overlap {
                "(not significant)"
            } else {
                "*** SIGNIFICANT ***"
            }
        );

        if delta > 0.03 && !overlap {
            eprintln!("  → SA의 내장 refresh는 penalty 위에 유의미한 추가 이득을 제공");
            eprintln!("  → Refresh 제거 시 성능 저하 예상. Phase 1B(GatedRefresh) 진행 권장.");
        } else if delta > 0.02 {
            eprintln!(
                "  → 소규모 이득 ({:.1}%), CI 겹침. 확증 불가.",
                delta * 100.0
            );
            eprintln!("  → N=500+ 추가 실험 또는 다른 domain에서 교차 검증 필요.");
        } else {
            eprintln!("  → Refresh 기여도 미미 (<2%). Penalty만으로 충분.");
        }
        eprintln!();
    }
}
