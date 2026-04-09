//! Go QUARTZ experiments — NPS benchmark

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use crate::game::GameState;
use crate::games::go::{go_quartz, Go};
use crate::mcts::eval::UniformEval;
use crate::mcts::quartz::QuartzController;
use crate::mcts::search::FixedIterations;
use crate::mcts::MctsEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_quartz_preset_smoke() {
        let config = go_quartz(9);
        assert!(config.quartz.is_some());
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let engine = MctsEngine::new(Go::new_9x9(), eval, config.clone());
        let qcfg = config.quartz.unwrap();
        let mut ctrl = QuartzController::new(50, qcfg);
        let stats = engine.run_quartz(&mut ctrl);
        assert!(stats.iterations > 0);
    }

    #[test]
    #[ignore]
    fn bench_go_nps() {
        eprintln!("\n=== Go NPS Benchmark ===\n");

        // UniformEval baseline
        for size in [9, 13, 19] {
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let config = go_quartz(size);
            let engine = MctsEngine::new(Go::new(size, 7.5), eval, config);
            let iters = if size <= 13 { 3000u32 } else { 1000 };
            let t = Instant::now();
            engine.run(&mut FixedIterations::new(iters));
            let ms = t.elapsed().as_millis().max(1) as f64;
            eprintln!(
                "  {:>2}×{:<2} UniformEval:   {:>7.0} NPS ({:.0}ms)",
                size,
                size,
                iters as f64 / (ms / 1000.0),
                ms
            );
        }

        // GoFastRollout
        eprintln!();
        for size in [9, 13, 19] {
            let eval = Arc::new(crate::games::go::GoFastRollout::new(200));
            let config = go_quartz(size);
            let engine = MctsEngine::new(Go::new(size, 7.5), eval, config);
            let iters = if size <= 13 { 2000u32 } else { 2000 };
            let t = Instant::now();
            engine.run(&mut FixedIterations::new(iters));
            let ms = t.elapsed().as_millis().max(1) as f64;
            eprintln!(
                "  {:>2}×{:<2} GoFastRollout: {:>7.0} NPS ({:.0}ms)",
                size,
                size,
                iters as f64 / (ms / 1000.0),
                ms
            );
        }

        // Parallel 9×9 with GoFastRollout
        eprintln!();
        let eval = Arc::new(crate::games::go::GoFastRollout::new(200));
        let config = go_quartz(9);
        let engine = MctsEngine::new(Go::new_9x9(), eval, config);
        let t = Instant::now();
        engine.run_par(&FixedIterations::new(10_000), 4);
        let ms = t.elapsed().as_millis().max(1) as f64;
        let root_n = engine.root.n_total.load(Ordering::Relaxed);
        eprintln!(
            "  9×9 GoFastRollout 4T: {:.0} NPS ({} visits, {:.0}ms)",
            root_n as f64 / (ms / 1000.0),
            root_n,
            ms
        );
        eprintln!();
    }

    #[test]
    #[ignore]
    fn selfplay_go_9x9() {
        eprintln!("\n=== Go 9×9 Self-Play (QUARTZ, 100 iters/move, max 120 plies) ===\n");
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let mut state = Go::new_9x9();
        let mut move_n = 0u32;
        let mut consecutive_passes = 0u8;
        let game_start = Instant::now();

        while !state.is_terminal() && move_n < 120 {
            let legal = state.legal_moves();
            if legal.is_empty() {
                break;
            }
            let config = go_quartz(9);
            let qcfg = config.quartz.clone().unwrap();
            let engine = MctsEngine::new(state.clone(), eval.clone(), config);
            let mut ctrl = QuartzController::new(100, qcfg);
            engine.run_quartz(&mut ctrl);
            let best = engine.best_move().expect("should find move");
            let who = if state.current_player() > 0 { "B" } else { "W" };

            if best == state.num_actions() as u16 - 1 {
                consecutive_passes += 1;
                if move_n < 6 || consecutive_passes > 0 {
                    eprintln!(
                        "  {:>3}  pass    {:>3}  ({} legal)",
                        move_n,
                        who,
                        legal.len()
                    );
                }
            } else {
                consecutive_passes = 0;
                let r = best as usize / 9;
                let c = best as usize % 9;
                if move_n < 6 || move_n % 15 == 0 {
                    eprintln!(
                        "  {:>3}  ({},{})   {:>3}  ({} legal)",
                        move_n,
                        r,
                        c,
                        who,
                        legal.len()
                    );
                }
            }

            state = state.apply_move(best);
            move_n += 1;
        }

        let ms = game_start.elapsed().as_millis();
        let (bs, ws) = state.tromp_taylor_score();
        let result = if !state.is_terminal() {
            "incomplete".to_string()
        } else {
            format!("B={:.0} W={:.0} (komi included)", bs, ws)
        };
        eprintln!(
            "\n  Result: {} after {} plies ({:.1}s, {:.0}ms/ply)\n",
            result,
            move_n,
            ms as f64 / 1000.0,
            ms as f64 / move_n.max(1) as f64
        );
        assert!(move_n > 0);
    }
}
