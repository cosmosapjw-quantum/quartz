//! H3 Adaptive Ablation: D-based gating benchmark
//! D = KL(π_0 ‖ softmax(Q/τ)), D_thresh sweep + 4-condition comparison

use crate::ablation_refresh::{
    bootstrap_ci, cis_overlap, gen_gomoku_positions, gen_ttt_positions, measure_flips, print_result,
};
use crate::ablation_refresh_v2::BiasedEval;
use crate::game::GameState;
use crate::games::tictactoe::TicTacToe;
use crate::games::Gomoku;
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController};
use crate::mcts::{MctsConfig, MctsEngine};
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    fn adaptive_cfg(d_thresh: f32) -> QuartzConfig {
        // D_thresh is encoded in hbar_penalty_cap's decimal part for now
        // (Adaptive mode internally uses d_thresh=0.5, but we'll test variants)
        QuartzConfig {
            penalty_mode: PenaltyMode::PFlipMixture,
            hbar_penalty_cap: 0.3,
            min_visits: 30,
            check_interval: 50,
            ..Default::default()
        }
    }

    // ─── D 관측 실험: D값 분포 확인 ───

    #[test]
    #[ignore]
    fn observe_d_values() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  D = KL(π_0 ‖ softmax(Q/τ)) 관측");
        eprintln!("  τ = max(σ_Q·√K, 0.1)");
        eprintln!("{}\n", "═".repeat(70));

        let positions_ttt = gen_ttt_positions(30, 42);
        let positions_gom = gen_gomoku_positions(20, 42);

        // TTT + UniformEval
        {
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let base = MctsConfig::evaluation(2.0);
            let qcfg = QuartzConfig {
                penalty_mode: PenaltyMode::PFlipMixture,
                hbar_penalty_cap: 0.3,
                min_visits: 30,
                check_interval: 50,
                ..Default::default()
            };
            let mut ds = Vec::new();
            for state in &positions_ttt {
                let cfg = base.clone().with_quartz(QuartzConfig {
                    halt_mode: HaltMode::Fixed { budget: 2000 },
                    ..qcfg.clone()
                });
                let eng = MctsEngine::new(state.clone(), eval.clone(), cfg);
                let mut ctrl = QuartzController::new(
                    2000,
                    QuartzConfig {
                        halt_mode: HaltMode::Fixed { budget: 2000 },
                        ..qcfg.clone()
                    },
                );
                eng.run_quartz(&mut ctrl);
                ds.push(ctrl.last_stats().prior_q_divergence);
            }
            ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mean_d: f32 = ds.iter().sum::<f32>() / ds.len() as f32;
            eprintln!(
                "  TTT+Uniform (N={}): mean_D={:.4} median={:.4} min={:.4} max={:.4}",
                ds.len(),
                mean_d,
                ds[ds.len() / 2],
                ds[0],
                ds.last().unwrap()
            );
        }

        // TTT + BiasedEval
        {
            let eval: Arc<BiasedEval> = Arc::new(BiasedEval::strongly_biased());
            let base = MctsConfig::evaluation(2.0);
            let qcfg = QuartzConfig {
                penalty_mode: PenaltyMode::PFlipMixture,
                hbar_penalty_cap: 0.3,
                min_visits: 30,
                check_interval: 50,
                ..Default::default()
            };
            let mut ds = Vec::new();
            for state in &positions_ttt {
                let cfg = base.clone().with_quartz(QuartzConfig {
                    halt_mode: HaltMode::Fixed { budget: 2000 },
                    ..qcfg.clone()
                });
                let eng = MctsEngine::new(state.clone(), eval.clone(), cfg);
                let mut ctrl = QuartzController::new(
                    2000,
                    QuartzConfig {
                        halt_mode: HaltMode::Fixed { budget: 2000 },
                        ..qcfg.clone()
                    },
                );
                eng.run_quartz(&mut ctrl);
                ds.push(ctrl.last_stats().prior_q_divergence);
            }
            ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mean_d: f32 = ds.iter().sum::<f32>() / ds.len() as f32;
            eprintln!(
                "  TTT+Biased  (N={}): mean_D={:.4} median={:.4} min={:.4} max={:.4}",
                ds.len(),
                mean_d,
                ds[ds.len() / 2],
                ds[0],
                ds.last().unwrap()
            );
        }

        // Gomoku + ShortRollout
        {
            let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
            let base = MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku());
            let qcfg = QuartzConfig {
                penalty_mode: PenaltyMode::PFlipMixture,
                hbar_penalty_cap: 0.3,
                min_visits: 15,
                check_interval: 20,
                ..Default::default()
            };
            let mut ds = Vec::new();
            for state in &positions_gom {
                let cfg = base.clone().with_quartz(QuartzConfig {
                    halt_mode: HaltMode::Fixed { budget: 300 },
                    ..qcfg.clone()
                });
                let eng = MctsEngine::new(state.clone(), eval.clone(), cfg);
                let mut ctrl = QuartzController::new(
                    300,
                    QuartzConfig {
                        halt_mode: HaltMode::Fixed { budget: 300 },
                        ..qcfg.clone()
                    },
                );
                eng.run_quartz(&mut ctrl);
                ds.push(ctrl.last_stats().prior_q_divergence);
            }
            ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mean_d: f32 = ds.iter().sum::<f32>() / ds.len() as f32;
            eprintln!(
                "  Gomoku+SR   (N={}): mean_D={:.4} median={:.4} min={:.4} max={:.4}",
                ds.len(),
                mean_d,
                ds[ds.len() / 2],
                ds[0],
                ds.last().unwrap()
            );
        }
        eprintln!();
    }

    // ─── 4-Condition Comprehensive Benchmark ───

    #[test]
    #[ignore]
    fn h3_adaptive_benchmark() {
        eprintln!("\n{}", "═".repeat(75));
        eprintln!("  H3 Adaptive Benchmark: D-gated Q-refresh vs all alternatives");
        eprintln!("  6 configs × 4 conditions × N=200");
        eprintln!("{}\n", "═".repeat(75));

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
                "EffV2_0.3",
                QuartzConfig {
                    penalty_mode: PenaltyMode::EffectiveV2,
                    hbar_penalty_cap: 0.3,
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
            (
                "EffV2+ExtQRef",
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
            (
                "PFlipMixture",
                QuartzConfig {
                    penalty_mode: PenaltyMode::PFlipMixture,
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                },
            ),
        ];

        let mut all_results: Vec<Vec<crate::ablation_refresh::FlipResult>> = Vec::new();

        // (a) TTT + UniformEval + High budget
        {
            eprintln!("  ─── (a) TTT + UniformEval, 2K/10K, N=200 ───");
            let positions = gen_ttt_positions(200, 42);
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let base = MctsConfig::evaluation(2.0);
            let mut cond_results = Vec::new();
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
                cond_results.push(r);
            }
            eprintln!();
            all_results.push(cond_results);
        }

        // (b) TTT + BiasedEval + High budget
        {
            eprintln!("  ─── (b) TTT + BiasedEval, 2K/10K, N=200 ───");
            let positions = gen_ttt_positions(200, 42);
            let eval: Arc<BiasedEval> = Arc::new(BiasedEval::strongly_biased());
            let base = MctsConfig::evaluation(2.0);
            let mut cond_results = Vec::new();
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
                cond_results.push(r);
            }
            eprintln!();
            all_results.push(cond_results);
        }

        // (c) TTT + UniformEval + Low budget
        {
            eprintln!("  ─── (c) TTT + UniformEval, 200/2K, N=200 ───");
            let positions = gen_ttt_positions(200, 42);
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let base = MctsConfig::evaluation(2.0);
            let low_configs: Vec<(&str, QuartzConfig)> = configs
                .iter()
                .map(|(l, q)| {
                    (
                        *l,
                        QuartzConfig {
                            min_visits: 15,
                            check_interval: 20,
                            ..q.clone()
                        },
                    )
                })
                .collect();
            let mut cond_results = Vec::new();
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
                cond_results.push(r);
            }
            eprintln!();
            all_results.push(cond_results);
        }

        // (d) Gomoku + ShortRollout
        {
            eprintln!("  ─── (d) Gomoku 7×7 + ShortRollout, 300/1200, N=50 ───");
            let positions = gen_gomoku_positions(50, 42);
            let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
            let base = MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku());
            let gom_configs: Vec<(&str, QuartzConfig)> = configs
                .iter()
                .map(|(l, q)| {
                    (
                        *l,
                        QuartzConfig {
                            min_visits: 15,
                            check_interval: 20,
                            ..q.clone()
                        },
                    )
                })
                .collect();
            let mut cond_results = Vec::new();
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
                cond_results.push(r);
            }
            eprintln!();
            all_results.push(cond_results);
        }

        // ─── Summary table ───
        eprintln!("  ═══ SUMMARY ═══");
        eprintln!(
            "  {:28} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "", "(a)", "(b)", "(c)", "(d)", "worst"
        );
        let labels = [
            "NoPenalty",
            "EffV2_0.3",
            "SelfAdaptive",
            "EffV2+ExtQRef",
            "GatedRefresh",
            "PFlipMixture",
        ];
        for i in 0..6 {
            let rates: Vec<f32> = (0..4).map(|c| all_results[c][i].rate).collect();
            let worst = rates.iter().cloned().fold(0.0f32, f32::max);
            eprintln!(
                "  {:28} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                labels[i], rates[0], rates[1], rates[2], rates[3], worst
            );
        }

        // ─── Decision ───
        eprintln!("\n  ═══ DECISION ═══");
        // Find config with lowest worst-case
        let mut best_minimax = (0usize, 1.0f32);
        for i in 0..6 {
            let worst = (0..4)
                .map(|c| all_results[c][i].rate)
                .fold(0.0f32, f32::max);
            if worst < best_minimax.1 {
                best_minimax = (i, worst);
            }
        }
        eprintln!(
            "  Minimax-optimal: {} (worst-case={:.3})",
            labels[best_minimax.0], best_minimax.1
        );

        // Check if Adaptive is within 0.03 of each condition's best
        let adaptive_idx = 5;
        let mut within_threshold = true;
        for c in 0..4 {
            let cond_best = (0..6)
                .map(|i| all_results[c][i].rate)
                .fold(1.0f32, f32::min);
            let adaptive_rate = all_results[c][adaptive_idx].rate;
            let gap = adaptive_rate - cond_best;
            if gap > 0.05 {
                within_threshold = false;
            }
            eprintln!(
                "  Condition {}: best={:.3} Adaptive={:.3} gap={:.3} {}",
                ["(a)", "(b)", "(c)", "(d)"][c],
                cond_best,
                adaptive_rate,
                gap,
                if gap <= 0.03 {
                    "✓"
                } else if gap <= 0.05 {
                    "~"
                } else {
                    "✗"
                }
            );
        }
        if within_threshold {
            eprintln!("  → PFlipMixture은 모든 조건에서 최선의 5% 이내. 범용 default 후보.");
        }
        eprintln!();
    }

    // ─── Patch 3: c_puct confound control ───

    #[test]
    #[ignore]
    fn cpuct_sweep() {
        eprintln!("\n{}", "═".repeat(70));
        eprintln!("  c_puct Sweep: disentangle controller from exploration strength");
        eprintln!("  3 modes × 3 c_puct values × condition (a) TTT+Uniform N=100");
        eprintln!("{}\n", "═".repeat(70));

        let positions = gen_ttt_positions(100, 42);
        let eval: Arc<UniformEval> = Arc::new(UniformEval);

        let modes = [
            ("NoPenalty", PenaltyMode::None),
            ("GatedRefresh", PenaltyMode::GatedRefresh),
            ("SelfAdaptive", PenaltyMode::SelfAdaptive),
        ];
        let cpucts = [1.5f32, 2.0, 2.5];

        eprintln!("  {:20} {:>8} {:>8} {:>8}", "", "c=1.5", "c=2.0", "c=2.5");
        for (label, pm) in &modes {
            let mut rates = Vec::new();
            for &cp in &cpucts {
                let base = MctsConfig::evaluation(cp);
                let qcfg = QuartzConfig {
                    penalty_mode: pm.clone(),
                    hbar_penalty_cap: 0.3,
                    min_visits: 30,
                    check_interval: 50,
                    ..Default::default()
                };
                let r = measure_flips(label, &positions, &eval, 2000, 10000, qcfg, base);
                rates.push(r.rate);
            }
            eprintln!(
                "  {:20} {:>8.3} {:>8.3} {:>8.3}  Δ={:.3}",
                label,
                rates[0],
                rates[1],
                rates[2],
                rates.iter().cloned().fold(0.0f32, f32::max)
                    - rates.iter().cloned().fold(1.0f32, f32::min)
            );
        }
        eprintln!("\n  If Δ > 0.05 for any mode, c_puct is a significant confound.");
        eprintln!(
            "  If controller modes reverse rank at different c_puct, the comparison is invalid.\n"
        );
    }
}
