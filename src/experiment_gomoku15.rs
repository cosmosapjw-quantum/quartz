//! Gomoku15 QUARTZ experiments — tournament + σ₀ calibration
//!
//! 사용법 (main.rs에서):
//!   mod experiment_gomoku15;
//!   experiment_gomoku15::run_all();
//!
//! 또는 #[test]:
//!   cargo test --release -- experiment_gomoku15 --ignored --nocapture

use crate::experiment::{run_tournament, TournamentConfig};
use crate::game::{Evaluator, GameState};
use crate::games::gomoku15::{gomoku15_quartz, Gomoku15, GomokuVariant};
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::quartz::{QuartzConfig, QuartzController};
use crate::mcts::search::FixedIterations;
use crate::mcts::{MctsConfig, MctsEngine, PwConfig};
use std::sync::Arc;

// ─────────────────────────────────────────────
// § Gomoku15 QUARTZ vs Baseline 토너먼트
// ─────────────────────────────────────────────

fn tournament_variant(variant: GomokuVariant, n_games: u32, iters: u32) {
    let label = match variant {
        GomokuVariant::Freestyle => "Free15",
        GomokuVariant::Standard => "Std15",
        GomokuVariant::Omok => "Omok15",
        GomokuVariant::Renju => "Renju15",
        GomokuVariant::Caro => "Caro15",
    };
    println!("══════════════════════════════════════════════════");
    println!(
        "  {}: QUARTZ vs Baseline ({} games, {} iters)",
        label, n_games, iters
    );
    println!("══════════════════════════════════════════════════");

    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));

    let p1 = gomoku15_quartz(variant);
    let p2 = MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5));
    let qcfg = p1.quartz.clone();

    let result = run_tournament(
        Gomoku15::new(variant),
        eval,
        p1,
        p2,
        qcfg,
        TournamentConfig {
            n_games,
            iters,
            p1_label: format!("QUARTZ-{}", label),
            p2_label: format!("Baseline-{}", label),
        },
    );
    result.print();
    println!();
}

// ─────────────────────────────────────────────
// § σ₀ 캘리브레이션 스캔
// ─────────────────────────────────────────────

fn sigma0_scan(variant: GomokuVariant) {
    let label = match variant {
        GomokuVariant::Freestyle => "Free15",
        GomokuVariant::Standard => "Std15",
        GomokuVariant::Omok => "Omok15",
        GomokuVariant::Renju => "Renju15",
        GomokuVariant::Caro => "Caro15",
    };

    println!("══════════════════════════════════════════════════");
    println!("  {} σ₀ calibration scan", label);
    println!("══════════════════════════════════════════════════");

    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));

    // Generate 10 random mid-game positions
    let mut positions = Vec::new();
    {
        use rand::rngs::StdRng;
        use rand::{seq::SliceRandom, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let n = 10 + rng.gen::<usize>() % 30;
            let mut mvs: Vec<u16> = (0..225).collect();
            mvs.shuffle(&mut rng);
            mvs.truncate(n);
            let mut s = Gomoku15::new(variant);
            let mut ok = true;
            for &mv in &mvs {
                if s.is_terminal() {
                    ok = false;
                    break;
                }
                s = s.apply_move(mv);
            }
            if ok && !s.is_terminal() && s.legal_moves().len() >= 5 {
                positions.push(s);
                if positions.len() >= 10 {
                    break;
                }
            }
        }
    }

    let sigma_values = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.7];
    let budget = 300u32;

    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
        "σ₀", "σ_Q", "ħ_eff", "P_flip", "VOC", "converged"
    );
    println!("  {}", "-".repeat(58));

    for &sigma_0 in &sigma_values {
        let mut tot_sq = 0.0f32;
        let mut tot_hbar = 0.0f32;
        let mut tot_pf = 0.0f32;
        let mut tot_voc = 0.0f32;
        let mut tot_conv = 0u32;
        let n = positions.len() as f32;

        for state in &positions {
            let qcfg = QuartzConfig {
                sigma_0,
                min_visits: 30,
                check_interval: 30,
                ..Default::default()
            };
            let config = MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5))
                .with_quartz(qcfg.clone());
            let engine = MctsEngine::new(state.clone(), eval.clone(), config);
            let mut ctrl = QuartzController::new(budget, qcfg);
            engine.run_quartz(&mut ctrl);
            let s = ctrl.last_stats();

            tot_sq += s.sigma_q;
            tot_hbar += s.hbar_eff;
            tot_pf += s.p_flip;
            tot_voc += s.voc_focus;
            if s.converged {
                tot_conv += 1;
            }
        }

        println!(
            "  {:>6.2}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.5}  {:>5}/{:>3}",
            sigma_0,
            tot_sq / n,
            tot_hbar / n,
            tot_pf / n,
            tot_voc / n,
            tot_conv,
            positions.len()
        );
    }

    println!("\n  이상적: ħ_eff ≈ 1.0 (σ_Q ≈ σ₀) 인 σ₀ 값이 최적");
    println!();
}

// ─────────────────────────────────────────────
// § MCTS NPS 벤치마크 (variant별)
// ─────────────────────────────────────────────

fn nps_benchmark() {
    println!("══════════════════════════════════════════════════");
    println!("  Gomoku15 NPS Benchmark");
    println!("══════════════════════════════════════════════════");

    let iters = 3000u32;

    for (label, variant) in [
        ("Freestyle", GomokuVariant::Freestyle),
        ("Standard", GomokuVariant::Standard),
        ("Omok", GomokuVariant::Omok),
        ("Renju", GomokuVariant::Renju),
        ("Caro", GomokuVariant::Caro),
    ] {
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let config = gomoku15_quartz(variant);
        let engine = MctsEngine::new(Gomoku15::new(variant), eval, config);

        let t = std::time::Instant::now();
        engine.run(&mut FixedIterations::new(iters));
        let ms = t.elapsed().as_millis().max(1) as f64;
        let nps = iters as f64 / (ms / 1000.0);

        println!(
            "  {:8}: {:.0} NPS  ({} iters, {:.0}ms)",
            label, nps, iters, ms
        );
    }

    // Parallel (4 threads)
    println!("\n  4-thread parallel:");
    for (label, variant) in [
        ("Freestyle", GomokuVariant::Freestyle),
        ("Standard", GomokuVariant::Standard),
        ("Omok", GomokuVariant::Omok),
    ] {
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let config = gomoku15_quartz(variant);
        let engine = MctsEngine::new(Gomoku15::new(variant), eval, config);

        let t = std::time::Instant::now();
        let stats = engine.run_par(&FixedIterations::new(10000), 4);
        let ms = t.elapsed().as_millis().max(1) as f64;
        let nps = stats.root_visits as f64 / (ms / 1000.0);

        println!(
            "  {:8}: {:.0} NPS  ({} visits, {:.0}ms)",
            label, nps, stats.root_visits, ms
        );
    }
    println!();
}

// ─────────────────────────────────────────────
// § 공개 인터페이스
// ─────────────────────────────────────────────

pub fn run_all() {
    nps_benchmark();
    sigma0_scan(GomokuVariant::Standard);
    sigma0_scan(GomokuVariant::Omok);
    tournament_variant(GomokuVariant::Standard, 20, 200);
    tournament_variant(GomokuVariant::Omok, 20, 200);
}

// ─────────────────────────────────────────────
// § 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gomoku15_quartz_preset_smoke() {
        for variant in [
            GomokuVariant::Freestyle,
            GomokuVariant::Standard,
            GomokuVariant::Omok,
            GomokuVariant::Renju,
            GomokuVariant::Caro,
        ] {
            let config = gomoku15_quartz(variant);
            assert!(config.quartz.is_some());
            assert!(config.gvoc.is_some());
            assert!(config.pw.is_some());
        }
    }

    #[test]
    #[ignore]
    fn test_nps_benchmark() {
        nps_benchmark();
    }

    #[test]
    #[ignore]
    fn test_sigma0_scan() {
        sigma0_scan(GomokuVariant::Standard);
    }

    #[test]
    #[ignore]
    fn test_tournament_standard() {
        tournament_variant(GomokuVariant::Standard, 10, 100);
    }

    #[test]
    #[ignore]
    fn test_tournament_omok() {
        tournament_variant(GomokuVariant::Omok, 10, 100);
    }
}

