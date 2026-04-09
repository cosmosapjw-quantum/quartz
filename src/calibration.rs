//! Cross-game σ₀ calibration — measures σ_Q and ħ_eff across all game types
//! to determine optimal σ₀ per game.

use std::sync::Arc;
use std::time::Instant;

use crate::game::{Evaluator, GameState};
use crate::mcts::eval::{ShortRollout, UniformEval};
use crate::mcts::quartz::{QuartzConfig, QuartzController};
use crate::mcts::{MctsConfig, MctsEngine, PwConfig};

fn sigma_scan<G: GameState, E: Evaluator<G> + Send + Sync + 'static>(
    label: &str,
    positions: &[G],
    eval: Arc<E>,
    budget: u32,
    sigma_values: &[f32],
) {
    eprintln!(
        "  {:>20}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
        label, "σ₀", "σ_Q", "ħ_eff", "P_flip", "conv"
    );
    eprintln!("  {:>20}  {}", "", "-".repeat(50));

    for &sigma_0 in sigma_values {
        let n = positions.len() as f32;
        let mut tot_sq = 0.0f32;
        let mut tot_hbar = 0.0f32;
        let mut tot_pf = 0.0f32;
        let mut tot_conv = 0u32;

        for state in positions {
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
            if s.converged {
                tot_conv += 1;
            }
        }

        eprintln!(
            "  {:>20}  {:>6.2}  {:>8.4}  {:>8.4}  {:>8.4}  {:>4}/{}",
            "",
            sigma_0,
            tot_sq / n,
            tot_hbar / n,
            tot_pf / n,
            tot_conv,
            positions.len()
        );
    }
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::chess::Chess;
    use crate::games::go::Go;
    use crate::games::gomoku15::{Gomoku15, GomokuVariant};

    fn random_mid_positions<G: GameState>(
        initial: G,
        n: usize,
        moves_range: (usize, usize),
    ) -> Vec<G> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let mut positions = Vec::new();
        for _ in 0..n * 5 {
            let n_moves = moves_range.0 + rng.gen::<usize>() % (moves_range.1 - moves_range.0);
            let mut s = initial.clone();
            for _ in 0..n_moves {
                if s.is_terminal() {
                    break;
                }
                let legal = s.legal_moves();
                if legal.is_empty() {
                    break;
                }
                s = s.apply_move(legal[rng.gen::<usize>() % legal.len()]);
            }
            if !s.is_terminal() && s.legal_moves().len() >= 3 {
                positions.push(s);
                if positions.len() >= n {
                    break;
                }
            }
        }
        positions
    }

    #[test]
    #[ignore]
    fn cross_game_sigma_calibration() {
        let sigmas = [0.05f32, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];
        let budget = 300u32;

        eprintln!("\n{}", "═".repeat(75));
        eprintln!(
            "  Cross-Game σ₀ Calibration (UniformEval, {} iters, 8 positions each)",
            budget
        );
        eprintln!("  Target: ħ_eff ≈ 1.0 (σ_Q ≈ σ₀)");
        eprintln!("{}\n", "═".repeat(75));

        // Gomoku15 Standard
        let positions = random_mid_positions(Gomoku15::standard(), 8, (10, 30));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        sigma_scan("Gomoku15-Std", &positions, eval, budget, &sigmas);

        // Gomoku15 Omok
        let positions = random_mid_positions(Gomoku15::omok(), 8, (10, 30));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        sigma_scan("Gomoku15-Omok", &positions, eval, budget, &sigmas);

        // Chess
        let positions = random_mid_positions(Chess::standard(), 8, (6, 20));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        sigma_scan("Chess", &positions, eval, budget, &sigmas);

        // Go 9×9
        let positions = random_mid_positions(Go::new_9x9(), 8, (10, 30));
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        sigma_scan("Go-9x9", &positions, eval, budget, &sigmas);

        eprintln!("  Interpretation:");
        eprintln!("  - ħ_eff >> 1: σ₀ too low → QUARTZ over-corrects");
        eprintln!("  - ħ_eff << 1: σ₀ too high → QUARTZ under-corrects");
        eprintln!("  - ħ_eff ≈ 1: optimal → σ₀ matches actual evaluation noise\n");
    }

    #[test]
    #[ignore]
    fn cross_game_sigma_with_rollout() {
        let sigmas = [0.1f32, 0.2, 0.3, 0.5, 0.7];
        let budget = 300u32;

        eprintln!("\n{}", "═".repeat(75));
        eprintln!(
            "  Cross-Game σ₀ Calibration (ShortRollout(10), {} iters)",
            budget
        );
        eprintln!("{}\n", "═".repeat(75));

        let positions = random_mid_positions(Gomoku15::standard(), 8, (10, 30));
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(10));
        sigma_scan("Gomoku15-Std(SR)", &positions, eval, budget, &sigmas);

        let positions = random_mid_positions(Chess::standard(), 8, (6, 20));
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(10));
        sigma_scan("Chess(SR)", &positions, eval, budget, &sigmas);

        let positions = random_mid_positions(Go::new_9x9(), 8, (10, 30));
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(10));
        sigma_scan("Go-9x9(SR)", &positions, eval, budget, &sigmas);

        eprintln!();
    }
}
