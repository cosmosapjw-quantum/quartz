//! § Publication Ablation: P_flip Convergence & Adaptive Stopping
//!
//! Core hypothesis: P_flip converges to 0 as search progresses when the
//! evaluator is strong enough (loss < ~1.0). QUARTZ should:
//! - Stop early with good evaluators (P_flip < threshold)
//! - Use full budget with weak evaluators (P_flip stays high)
//!
//! Experiments:
//! 1. P_flip convergence curves (P_flip vs iteration) × evaluator quality
//! 2. Adaptive stopping validation (VOC halt vs Fixed budget)
//! 3. QUARTZ mode comparison (None/GatedRefresh/SelfAdaptive × evaluator)
//! 4. VL × QUARTZ interaction under adaptive stopping

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::ablation_refresh_v2::BiasedEval;
use crate::game::GameState;
use crate::games::gomoku::Gomoku;
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::parallel::VlMode;
use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController, FLIP_THRESH};
use crate::mcts::search::SearchController;
use crate::mcts::{MctsConfig, MctsEngine};

fn gen_gomoku7_positions(n: usize, seed: u64) -> Vec<Gomoku> {
    let mut s = seed;
    let mut next = || -> u64 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 33
    };
    let mut out = Vec::new();
    for _ in 0..n * 2 {
        if out.len() >= n {
            break;
        }
        let nm = 4 + (next() % 12) as usize;
        let mut state = Gomoku::new_with_win(7, 4);
        for _ in 0..nm {
            let moves = state.legal_moves();
            if moves.is_empty() {
                break;
            }
            state = state.apply_move(moves[(next() as usize) % moves.len()]);
            if state.is_terminal() {
                break;
            }
        }
        if !state.is_terminal() {
            out.push(state);
        }
    }
    out
}

/// Track P_flip at regular intervals during search.
fn pflip_curve(
    state: &Gomoku,
    eval: &Arc<dyn crate::game::Evaluator<Gomoku> + Send + Sync>,
    penalty: PenaltyMode,
    budget: u32,
    checkpoints: &[u32],
) -> Vec<(u32, f32, f32)> {
    // Returns: Vec<(iteration, p_flip, sigma_q)>
    let qcfg = QuartzConfig {
        penalty_mode: penalty,
        hbar_penalty_cap: 0.3,
        min_visits: 10,
        check_interval: 10,
        halt_mode: HaltMode::Fixed { budget },
        ..Default::default()
    };
    let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::default()).with_quartz(qcfg.clone());
    let eng = MctsEngine::new(state.clone(), eval.clone(), cfg);
    let mut ctrl = QuartzController::new(budget, qcfg);

    let mut curve = Vec::new();
    let mut cp_idx = 0;

    // Manual iteration loop to capture snapshots
    ctrl.reset();
    let mut it = 0u32;
    loop {
        let rv = eng.root.n_total.load(Ordering::Relaxed);
        if it > 0 && it % 10 == 0 {
            let _rp = eng.root_priors();
            ctrl.update_stats(&eng.root, Some(&_rp));
        }
        if ctrl.should_stop(rv, 0) {
            break;
        }
        eng.iterate();
        it += 1;

        // Capture at checkpoints
        if cp_idx < checkpoints.len() && it >= checkpoints[cp_idx] {
            let stats = ctrl.last_stats();
            curve.push((it, stats.p_flip, stats.sigma_q));
            cp_idx += 1;
        }
    }

    // Final snapshot
    let _rp = eng.root_priors();
    ctrl.update_stats(&eng.root, Some(&_rp));
    let final_stats = ctrl.last_stats();
    curve.push((it, final_stats.p_flip, final_stats.sigma_q));

    curve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn pflip_convergence_curves() {
        eprintln!("\n{}", "═".repeat(80));
        eprintln!("  PUBLICATION ABLATION 1: P_flip Convergence Curves");
        eprintln!("  Does P_flip converge to 0 with good evaluators?");
        eprintln!("  Threshold: P_flip < {:.3} → QUARTZ halt", FLIP_THRESH);
        eprintln!("{}", "═".repeat(80));

        let positions = gen_gomoku7_positions(10, 42);
        let budget = 2000u32;
        let checkpoints: Vec<u32> = vec![50, 100, 200, 400, 800, 1200, 1600, 2000];

        let evaluators: Vec<(&str, Arc<dyn crate::game::Evaluator<Gomoku> + Send + Sync>)> = vec![
            ("Uniform", Arc::new(UniformEval)),
            ("BiasedMild", Arc::new(BiasedEval::mildly_biased())),
            ("BiasedStrong", Arc::new(BiasedEval::strongly_biased())),
            ("ShortRollout", Arc::new(ShortRollout::new(7))),
        ];

        eprintln!(
            "\n  {:>14} {:>6}  P_flip at checkpoints:",
            "Evaluator", "Pos"
        );
        eprintln!(
            "  {:>14} {:>6}  {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
            "", "", "50", "100", "200", "400", "800", "1200", "1600", "final"
        );
        eprintln!("  {}", "─".repeat(78));

        for (eval_name, eval) in &evaluators {
            let mut all_curves: Vec<Vec<f32>> = vec![Vec::new(); checkpoints.len() + 1];

            for (pi, state) in positions.iter().enumerate() {
                let curve =
                    pflip_curve(state, eval, PenaltyMode::GatedRefresh, budget, &checkpoints);
                for (ci, &(_, pf, _)) in curve.iter().enumerate() {
                    if ci < all_curves.len() {
                        all_curves[ci].push(pf);
                    }
                }
            }

            // Compute means
            let means: Vec<f32> = all_curves
                .iter()
                .map(|v| {
                    if v.is_empty() {
                        f32::NAN
                    } else {
                        v.iter().sum::<f32>() / v.len() as f32
                    }
                })
                .collect();

            eprint!("  {:>14} {:>6}", eval_name, positions.len());
            for m in &means {
                if m.is_nan() {
                    eprint!("  {:>6}", "—");
                } else {
                    eprint!("  {:>6.3}", m);
                }
            }
            eprintln!();
        }

        eprintln!("\n  Interpretation:");
        eprintln!("  - Uniform: P_flip should stay ~0.5 (no information → random best move)");
        eprintln!("  - BiasedStrong: P_flip should drop below threshold with enough budget");
        eprintln!(
            "  - ShortRollout: noisy — P_flip may or may not converge (eval quality dependent)"
        );
        eprintln!(
            "  - If P_flip < {:.3} at any checkpoint, QUARTZ VOC would trigger early stop",
            FLIP_THRESH
        );
    }

    #[test]
    #[ignore]
    fn adaptive_stopping_validation() {
        eprintln!("\n{}", "═".repeat(80));
        eprintln!("  PUBLICATION ABLATION 2: Adaptive Stopping Validation");
        eprintln!("  Does QUARTZ save budget when evaluator is strong?");
        eprintln!("{}", "═".repeat(80));

        let positions = gen_gomoku7_positions(15, 42);
        let max_budget = 2000u32;

        let evaluators: Vec<(&str, Arc<dyn crate::game::Evaluator<Gomoku> + Send + Sync>)> = vec![
            ("Uniform", Arc::new(UniformEval)),
            ("BiasedMild", Arc::new(BiasedEval::mildly_biased())),
            ("BiasedStrong", Arc::new(BiasedEval::strongly_biased())),
            ("ShortRollout", Arc::new(ShortRollout::new(7))),
        ];

        let halt_modes = [
            ("Fixed", HaltMode::Fixed { budget: max_budget }),
            ("VOC", HaltMode::VOC),
            ("Threshold", HaltMode::SimpleThreshold),
        ];

        eprintln!(
            "\n  {:>14} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "Evaluator", "HaltMode", "MeanIts", "StdIts", "P_flip", "σ_Q"
        );
        eprintln!("  {}", "─".repeat(62));

        for (eval_name, eval) in &evaluators {
            for &(halt_name, ref halt) in &halt_modes {
                let mut iterations = Vec::new();
                let mut pflips = Vec::new();
                let mut sigma_qs = Vec::new();

                for state in &positions {
                    let qcfg = QuartzConfig {
                        penalty_mode: PenaltyMode::GatedRefresh,
                        hbar_penalty_cap: 0.3,
                        min_visits: 30,
                        check_interval: 20,
                        halt_mode: halt.clone(),
                        ..Default::default()
                    };
                    let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::default())
                        .with_quartz(qcfg.clone());
                    let eng = MctsEngine::new(state.clone(), eval.clone(), cfg);
                    let mut ctrl = QuartzController::new(max_budget, qcfg);
                    let stats = eng.run_quartz(&mut ctrl);

                    iterations.push(stats.root_visits as f32);
                    let qs = ctrl.last_stats();
                    pflips.push(qs.p_flip);
                    sigma_qs.push(qs.sigma_q);
                }

                let n = positions.len() as f32;
                let mean_it = iterations.iter().sum::<f32>() / n;
                let std_it = (iterations
                    .iter()
                    .map(|x| (x - mean_it).powi(2))
                    .sum::<f32>()
                    / n)
                    .sqrt();
                let mean_pf = pflips.iter().sum::<f32>() / n;
                let mean_sq = sigma_qs.iter().sum::<f32>() / n;

                eprintln!(
                    "  {:>14} {:>10} {:>8.0} {:>8.0} {:>8.3} {:>8.3}",
                    eval_name, halt_name, mean_it, std_it, mean_pf, mean_sq
                );
            }
        }

        eprintln!("\n  Key result:");
        eprintln!("  - VOC/Threshold with BiasedStrong should use FEWER iterations than Fixed");
        eprintln!(
            "  - VOC/Threshold with Uniform should use ~SAME iterations as Fixed (no convergence)"
        );
        eprintln!("  - Budget savings = (Fixed_iters - VOC_iters) / Fixed_iters");
    }

    #[test]
    #[ignore]
    fn quartz_mode_comparison() {
        eprintln!("\n{}", "═".repeat(80));
        eprintln!("  PUBLICATION ABLATION 3: QUARTZ Mode Comparison");
        eprintln!("  Which penalty mode gives best search quality per budget?");
        eprintln!("{}", "═".repeat(80));

        let positions = gen_gomoku7_positions(20, 42);
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(7));
        let budget = 1000u32;

        let modes = [
            ("None", PenaltyMode::None),
            ("GatedRefresh", PenaltyMode::GatedRefresh),
            ("SelfAdaptive", PenaltyMode::SelfAdaptive),
            ("PFlipMixture", PenaltyMode::PFlipMixture),
        ];

        // Reference: None mode, 1 thread
        let ref_moves: Vec<usize> = positions
            .iter()
            .map(|s| {
                let qcfg = QuartzConfig {
                    penalty_mode: PenaltyMode::None,
                    halt_mode: HaltMode::Fixed { budget: budget * 2 },
                    check_interval: 50,
                    min_visits: 30,
                    ..Default::default()
                };
                let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::default())
                    .with_quartz(qcfg.clone());
                let eng = MctsEngine::new(s.clone(), eval.clone(), cfg);
                let mut ctrl = QuartzController::new(budget * 2, qcfg);
                eng.run_quartz(&mut ctrl);
                eng.best_move().map(|m| -> usize { m.into() }).unwrap_or(0)
            })
            .collect();

        eprintln!(
            "\n  {:>14} {:>7} {:>7} {:>8} {:>8} {:>8}",
            "Mode", "Agree", "Entrop", "P_flip", "σ_Q", "NPS"
        );
        eprintln!("  {}", "─".repeat(58));

        for &(name, penalty) in &modes {
            let mut agree = 0u32;
            let mut ents = Vec::new();
            let mut pflips = Vec::new();
            let mut sigmas = Vec::new();
            let mut nps_sum = 0.0f64;

            for (i, s) in positions.iter().enumerate() {
                let qcfg = QuartzConfig {
                    penalty_mode: penalty,
                    halt_mode: HaltMode::Fixed { budget },
                    check_interval: 50,
                    min_visits: 30,
                    hbar_penalty_cap: 0.3,
                    ..Default::default()
                };
                let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::default())
                    .with_quartz(qcfg.clone());
                let eng = MctsEngine::new(s.clone(), eval.clone(), cfg);
                let mut ctrl = QuartzController::new(budget, qcfg);
                let stats = eng.run_quartz(&mut ctrl);

                let mv: usize = eng.best_move().map(|m| m.into()).unwrap_or(0);
                if mv == ref_moves[i] {
                    agree += 1;
                }

                let edges = eng.root.edge_snapshot(eng.root.materialized_count());
                let total: f32 = edges
                    .iter()
                    .map(|e| e.n.load(Ordering::Relaxed) as f32)
                    .sum();
                let ent = if total > 1.0 {
                    edges
                        .iter()
                        .map(|e| {
                            let p = e.n.load(Ordering::Relaxed) as f32 / total;
                            if p > 1e-8 {
                                -p * p.ln()
                            } else {
                                0.0
                            }
                        })
                        .sum::<f32>()
                } else {
                    0.0
                };
                ents.push(ent);

                let qs = ctrl.last_stats();
                pflips.push(qs.p_flip);
                sigmas.push(qs.sigma_q);
                nps_sum += stats.nps;
            }

            let n = positions.len() as f32;
            eprintln!(
                "  {:>14} {:>6.1}% {:>7.3} {:>8.3} {:>8.3} {:>8.0}",
                name,
                100.0 * agree as f32 / n,
                ents.iter().sum::<f32>() / n,
                pflips.iter().sum::<f32>() / n,
                sigmas.iter().sum::<f32>() / n,
                nps_sum / n as f64
            );
        }
    }
}
