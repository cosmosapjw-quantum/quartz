#[cfg(test)]
mod greedy_experiment {
    use std::sync::Arc;
    use crate::game::{GameState, Evaluator};
    use crate::games::Gomoku;
    use crate::mcts::{MctsConfig, MctsEngine};
    use crate::mcts::quartz::*;
    use crate::mcts::search::SearchController;
    use crate::experiment::*;

    #[test]
    fn greedy_benchmark() {
        let eval: Arc<dyn Evaluator<Gomoku> + Send + Sync> =
            Arc::new(crate::mcts_server::GreedyGomokuEval::new(7, 4));
        let positions = generate_benchmark_positions(7, 4, 20, 4, 14, 42);
        let n = positions.len();
        let max_budget = 400u32;
        let replay_budget = 1600u32;

        println!("GREEDY EXPERIMENT: {} positions, budget={}", n, max_budget);

        let modes: Vec<(&str, HaltMode, CostMode)> = vec![
            ("VOC+TimeDriven",   HaltMode::VOC,             CostMode::TimeDriven),
            ("VOC+Legacy",       HaltMode::VOC,             CostMode::Legacy),
            ("Simple+TimeDriven",HaltMode::SimpleThreshold,  CostMode::TimeDriven),
            ("Fixed",            HaltMode::Fixed { budget: max_budget }, CostMode::Legacy),
        ];

        for (label, halt, cost) in &modes {
            let mut total_iters = 0u32;
            let mut early = 0u32;
            let mut flips = 0u32;
            let mut pflips: Vec<f32> = Vec::new();
            let mut flipped: Vec<f32> = Vec::new();
            let mut hbars: Vec<f32> = Vec::new();

            for state in &positions {
                let qcfg = QuartzConfig {
                    min_visits: 30, check_interval: 20,
                    halt_mode: halt.clone(), cost_mode: cost.clone(),
                    ctm_budget_ms: 500, ..Default::default()
                };
                let config = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
                let engine = MctsEngine::new(state.clone(), eval.clone(), config.clone());
                let mut ctrl = QuartzController::new(max_budget, qcfg);
                engine.run_quartz(&mut ctrl);
                let stats = ctrl.last_stats();
                let stop = ctrl.stop_reason();
                let snap = take_root_snapshot(&engine, state, &stats, &stop, halt);
                let replay = replay_and_compare(state, &snap, replay_budget, eval.clone(), &config);

                total_iters += snap.iterations_used;
                if snap.iterations_used < max_budget { early += 1; }
                if replay.did_flip { flips += 1; }
                pflips.push(snap.p_flip);
                flipped.push(if replay.did_flip { 1.0 } else { 0.0 });
                hbars.push(stats.hbar_eff);
            }

            let nf = n as f32;
            let avg_iters = total_iters as f32 / nf;
            let save = (1.0 - avg_iters / max_budget as f32) * 100.0;
            let flip_r = flips as f32 / nf * 100.0;
            let avg_hbar = hbars.iter().sum::<f32>() / nf;

            // Simple ECE (3 bins)
            let mut ece = 0.0f32;
            for b in 0..3 {
                let lo = b as f32 / 3.0;
                let hi = (b + 1) as f32 / 3.0;
                let (sp, sa, c) = pflips.iter().zip(flipped.iter())
                    .filter(|(&p, _)| p >= lo && (p < hi || (b == 2 && p <= hi)))
                    .fold((0f32, 0f32, 0u32), |(s1, s2, c), (&p, &a)| (s1+p, s2+a, c+1));
                if c > 0 {
                    ece += (c as f32 / nf) * ((sp/c as f32) - (sa/c as f32)).abs();
                }
            }

            println!("  {:<20}: avg_iter={:>6.1} save={:>5.1}% early={:>2}/{} flip={:>5.1}% ECE={:.3} hbar={:.3}",
                     label, avg_iters, save, early, n, flip_r, ece, avg_hbar);
        }

        // Selection ablation with Greedy
        println!("\nSELECTION ABLATION (Greedy):");
        let sel_configs = [
            ("Full_QUARTZ",  true,  true),
            ("Fisher_Only",  true,  false),
            ("OneLoop_Only", false, true),
            ("StdPUCT",      false, false),
        ];
        for (label, fisher, oneloop) in &sel_configs {
            let mut total_iters = 0u32;
            let mut flips = 0u32;
            for state in &positions {
                let qcfg = QuartzConfig {
                    min_visits: 30, check_interval: 20,
                    halt_mode: HaltMode::VOC, cost_mode: CostMode::TimeDriven,
                    ctm_budget_ms: 500,
                    enable_fisher_puct: *fisher, enable_one_loop: *oneloop,
                    ..Default::default()
                };
                let config = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
                let engine = MctsEngine::new(state.clone(), eval.clone(), config.clone());
                let mut ctrl = QuartzController::new(max_budget, qcfg);
                engine.run_quartz(&mut ctrl);
                let stats = ctrl.last_stats();
                let stop = ctrl.stop_reason();
                let snap = take_root_snapshot(&engine, state, &stats, &stop, &HaltMode::VOC);
                let replay = replay_and_compare(state, &snap, replay_budget, eval.clone(), &config);
                total_iters += snap.iterations_used;
                if replay.did_flip { flips += 1; }
            }
            let nf = n as f32;
            println!("  {:<16}: avg_iter={:>6.1} save={:>5.1}% flip={:>5.1}%",
                     label, total_iters as f32 / nf,
                     (1.0 - total_iters as f32 / nf / max_budget as f32) * 100.0,
                     flips as f32 / nf * 100.0);
        }
    }
}
