//! Phase 1B: GatedRefresh 종합 ablation
//!
//! GatedRefresh = EffV2 penalty + P_flip 역전 게이팅 + Q-based refresh
//! 검증 조건:
//!   (a) UniformEval + High budget — SA 최선 조건에서 harm 없음?
//!   (b) BiasedEval — prior 교정 능력
//!   (c) Low budget — Q-refresh 효과
//!   (d) Gomoku 7×7 — 큰 action space + noisy evaluator ("do no harm")

use crate::ablation_refresh::{
    bootstrap_ci, cis_overlap, gen_gomoku_positions, gen_ttt_positions, measure_flips, print_result,
};
use crate::ablation_refresh_v2::BiasedEval;
use crate::game::{Evaluator, GameState};
use crate::games::tictactoe::TicTacToe;
use crate::games::Gomoku;
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController};
use crate::mcts::{MctsConfig, MctsEngine};
use std::sync::Arc;

fn qcfg_base(mode: PenaltyMode, cap: f32) -> QuartzConfig {
    QuartzConfig {
        penalty_mode: mode,
        hbar_penalty_cap: cap,
        min_visits: 30,
        check_interval: 50,
        prior_refresh_rate: 0.0,
        ..Default::default()
    }
}

fn qcfg_ext_refresh(mode: PenaltyMode, cap: f32) -> QuartzConfig {
    QuartzConfig {
        penalty_mode: mode,
        hbar_penalty_cap: cap,
        min_visits: 30,
        check_interval: 50,
        prior_refresh_rate: 0.3,
        prior_refresh_temp: 0.5,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══════════════════════════════════════════════════
    // Phase 1B 종합: GatedRefresh vs 모든 대안
    // ═══════════════════════════════════════════════════

    #[test]
    #[ignore]
    fn phase_1b_comprehensive() {
        eprintln!("\n{}", "═".repeat(75));
        eprintln!("  Phase 1B: GatedRefresh 종합 검증 (4 conditions × 5 configs)");
        eprintln!("  GatedRefresh = EffV2 penalty + P_flip-역전 Q-refresh");
        eprintln!("{}\n", "═".repeat(75));

        let configs: Vec<(&str, QuartzConfig)> = vec![
            ("NoPenalty", qcfg_base(PenaltyMode::None, 0.0)),
            ("EffV2_0.3", qcfg_base(PenaltyMode::EffectiveV2, 0.3)),
            ("SelfAdaptive", qcfg_base(PenaltyMode::SelfAdaptive, 0.0)),
            (
                "EffV2+ExtQRef",
                qcfg_ext_refresh(PenaltyMode::EffectiveV2, 0.3),
            ),
            (
                "GatedRefresh",
                QuartzConfig {
                    penalty_mode: PenaltyMode::GatedRefresh,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
        ];

        // ── (a) TTT + UniformEval + High budget ──
        {
            eprintln!("  ─── (a) TTT + UniformEval, budget 2K/10K, N=200 ───");
            let positions = gen_ttt_positions(200, 42);
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let base = MctsConfig::evaluation(2.0);
            for (label, qcfg) in &configs {
                let r = measure_flips(
                    label,
                    &positions,
                    &eval,
                    2000,
                    10000,
                    qcfg.clone(),
                    base.clone(),
                );
                print_result(&r);
            }
            eprintln!();
        }

        // ── (b) TTT + BiasedEval + High budget ──
        {
            eprintln!("  ─── (b) TTT + BiasedEval, budget 2K/10K, N=200 ───");
            let positions = gen_ttt_positions(200, 42);
            let eval: Arc<BiasedEval> = Arc::new(BiasedEval::strongly_biased());
            let base = MctsConfig::evaluation(2.0);
            for (label, qcfg) in &configs {
                let r = measure_flips(
                    label,
                    &positions,
                    &eval,
                    2000,
                    10000,
                    qcfg.clone(),
                    base.clone(),
                );
                print_result(&r);
            }
            eprintln!();
        }

        // ── (c) TTT + UniformEval + Low budget ──
        {
            eprintln!("  ─── (c) TTT + UniformEval, budget 200/2K, N=200 ───");
            let positions = gen_ttt_positions(200, 42);
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let base = MctsConfig::evaluation(2.0);
            let low_configs: Vec<(&str, QuartzConfig)> = vec![
                (
                    "NoPenalty",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_base(PenaltyMode::None, 0.0)
                    },
                ),
                (
                    "EffV2_0.3",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_base(PenaltyMode::EffectiveV2, 0.3)
                    },
                ),
                (
                    "SelfAdaptive",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_base(PenaltyMode::SelfAdaptive, 0.0)
                    },
                ),
                (
                    "EffV2+ExtQRef",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_ext_refresh(PenaltyMode::EffectiveV2, 0.3)
                    },
                ),
                (
                    "GatedRefresh",
                    QuartzConfig {
                        penalty_mode: PenaltyMode::GatedRefresh,
                        hbar_penalty_cap: 0.3,
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    },
                ),
            ];
            for (label, qcfg) in &low_configs {
                let r = measure_flips(
                    label,
                    &positions,
                    &eval,
                    200,
                    2000,
                    qcfg.clone(),
                    base.clone(),
                );
                print_result(&r);
            }
            eprintln!();
        }

        // ── (d) Gomoku 7×7 + ShortRollout ("do no harm") ──
        {
            eprintln!("  ─── (d) Gomoku 7×7 + ShortRollout, budget 300/1200, N=50 ───");
            let positions = gen_gomoku_positions(50, 42);
            let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
            let base = MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku());
            let gom_configs: Vec<(&str, QuartzConfig)> = vec![
                (
                    "NoPenalty",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_base(PenaltyMode::None, 0.0)
                    },
                ),
                (
                    "Legacy_0.3",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_base(PenaltyMode::Legacy, 0.3)
                    },
                ),
                (
                    "SelfAdaptive",
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..qcfg_base(PenaltyMode::SelfAdaptive, 0.0)
                    },
                ),
                (
                    "GatedRefresh",
                    QuartzConfig {
                        penalty_mode: PenaltyMode::GatedRefresh,
                        hbar_penalty_cap: 0.3,
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    },
                ),
                (
                    "Leg+ExtQRef",
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
            for (label, qcfg) in &gom_configs {
                let r = measure_flips(
                    label,
                    &positions,
                    &eval,
                    300,
                    1200,
                    qcfg.clone(),
                    base.clone(),
                );
                print_result(&r);
            }
            eprintln!();
        }

        eprintln!("  ═══════════════════════════════════════════════════════════════");
        eprintln!("  DECISION FRAMEWORK:");
        eprintln!("  (a) UniformEval+High: GatedRefresh ≤ SA? → 'do no harm' on easy case");
        eprintln!("  (b) BiasedEval+High:  GatedRefresh < SA? → prior correction value");
        eprintln!("  (c) UniformEval+Low:  GatedRefresh < NoPen? → low-budget help");
        eprintln!("  (d) Gomoku+SR:        GatedRefresh ≤ Legacy? → 'do no harm' on hard case");
        eprintln!("  ═══════════════════════════════════════════════════════════════\n");
    }
}
