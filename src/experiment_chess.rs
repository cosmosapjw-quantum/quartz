//! Chess QUARTZ experiments — NPS benchmark + MCTS self-play

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use crate::game::GameState;
use crate::games::chess::{chess_quartz, Chess};
use crate::mcts::eval::UniformEval;
use crate::mcts::quartz::QuartzController;
use crate::mcts::search::FixedIterations;
use crate::mcts::MctsEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chess_quartz_preset_smoke() {
        let config = chess_quartz();
        assert!(config.quartz.is_some());
        assert!(config.gvoc.is_some());
        assert!(config.pw.is_some());

        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let engine = MctsEngine::new(Chess::standard(), eval, config.clone());
        let qcfg = config.quartz.unwrap();
        let mut ctrl = QuartzController::new(100, qcfg);
        let stats = engine.run_quartz(&mut ctrl);
        assert!(stats.iterations > 0);
        assert!(engine.best_move().is_some());
    }

    #[test]
    #[ignore]
    fn bench_chess_nps() {
        eprintln!("\n=== Chess NPS Benchmark ===\n");

        // Single thread, UniformEval
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let config = chess_quartz();
        let engine = MctsEngine::new(Chess::standard(), eval.clone(), config);
        let t = Instant::now();
        engine.run(&mut FixedIterations::new(3000));
        let ms = t.elapsed().as_millis().max(1) as f64;
        eprintln!(
            "  1T UniformEval: {:.0} NPS (3K iters, {:.0}ms)",
            3000.0 / (ms / 1000.0),
            ms
        );

        // 4-thread parallel
        let config2 = chess_quartz();
        let engine2 = MctsEngine::new(Chess::standard(), eval, config2);
        let t = Instant::now();
        let stats = engine2.run_par(&FixedIterations::new(10_000), 4);
        let ms = t.elapsed().as_millis().max(1) as f64;
        let root_n = engine2.root.n_total.load(Ordering::Relaxed);
        // Phase 7 C: lock-free slab read.
        let edges = engine2.root.read_edges();
        let vl: i64 = edges
            .iter()
            .map(|e| e.virtual_losses.load(Ordering::Relaxed) as i64)
            .sum();
        eprintln!(
            "  4T parallel:    {:.0} NPS ({} visits, {:.0}ms, VL={})",
            root_n as f64 / (ms / 1000.0),
            root_n,
            ms,
            vl
        );
        assert_eq!(vl, 0, "VL leak");

        // Movegen
        let c = Chess::standard();
        let t = Instant::now();
        let n = 100_000u32;
        for _ in 0..n {
            let _ = c.generate_legal_moves();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        eprintln!("  Movegen:        {:.2} μs/call", us);
        eprintln!("  Copy size:      {} bytes\n", std::mem::size_of::<Chess>());
    }

    #[test]
    #[ignore]
    fn selfplay_chess() {
        eprintln!("\n=== Chess Self-Play (QUARTZ, 200 iters/move, max 80 plies) ===\n");
        let eval: Arc<UniformEval> = Arc::new(UniformEval);
        let mut state = Chess::standard();
        let mut move_n = 0u32;
        let game_start = Instant::now();

        eprintln!(
            "  {:>3}  {:>6}  {:>3}  {:>8}  {:>8}  {:>5}",
            "ply", "move", "who", "P_flip", "sigma_Q", "legal"
        );

        while !state.is_terminal() && move_n < 80 {
            let legal = state.legal_moves();
            if legal.is_empty() {
                break;
            }
            let config = chess_quartz();
            let qcfg = config.quartz.clone().unwrap();
            let engine = MctsEngine::new(state.clone(), eval.clone(), config);
            let mut ctrl = QuartzController::new(200, qcfg);
            engine.run_quartz(&mut ctrl);
            let qs = ctrl.last_stats();
            let best = engine.best_move().expect("should find move");
            let who = if state.current_player() > 0 { "W" } else { "B" };

            if move_n < 6 || move_n % 10 == 0 || legal.len() < 10 {
                eprintln!(
                    "  {:>3}  {:>6}  {:>3}  {:>8.4}  {:>8.4}  {:>5}",
                    move_n,
                    best.to_uci(),
                    who,
                    qs.p_flip,
                    qs.sigma_q,
                    legal.len()
                );
            }

            state = state.apply_move(best);
            move_n += 1;
        }

        let ms = game_start.elapsed().as_millis();
        let result = if !state.is_terminal() {
            "incomplete"
        } else if state.outcome() == 0.0 {
            "Draw"
        } else if state.current_player() == 1 {
            "White loses"
        } else {
            "Black loses"
        };
        eprintln!(
            "\n  Result: {} after {} plies ({:.1}s, {:.0}ms/ply)\n",
            result,
            move_n,
            ms as f64 / 1000.0,
            ms as f64 / move_n.max(1) as f64
        );
    }
}
