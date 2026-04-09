//! 실험 프레임워크 — QUARTZ vs Baseline 자가대국 토너먼트
//!
//! # 목적
//! - 알고리즘 기여도 분리 측정 (baseline fairness contract §1.4)
//! - Fisher PUCT / FEP modulation / CTM cost 개별 ablation
//! - wimp(s) 기반 GVOC 효과 측정
//!
//! # 설계
//! - 동일 게임 / 동일 iteration budget → 알고리즘 차이만
//! - 반복 대국 (N=100) → 통계적 유의성
//! - 결과: win/loss/draw + Elo 추정

use std::sync::Arc;

use crate::game::{Evaluator, GameState};
use crate::mcts::quartz::QuartzStats;
use crate::mcts::search::FixedIterations;
use crate::mcts::{MctsConfig, MctsEngine, PwConfig, QuartzConfig, QuartzController};

// ─────────────────────────────────────────────
// § TournamentResult
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct TournamentResult {
    pub games: u32,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub p1_label: String,
    pub p2_label: String,
    /// 평균 p1 VOC (QUARTZ 통계)
    pub avg_voc_focus: f32,
    pub avg_sigma_delta: f32,
    pub avg_fep_mod: f32,
}

impl TournamentResult {
    pub fn win_rate_p1(&self) -> f32 {
        if self.games == 0 {
            return 0.5;
        }
        (self.p1_wins as f32 + 0.5 * self.draws as f32) / self.games as f32
    }

    /// Elo 추정 (logistic model)
    pub fn elo_diff(&self) -> f32 {
        let wr = self.win_rate_p1().clamp(0.001, 0.999);
        -400.0 * (1.0 / wr - 1.0).log10()
    }

    pub fn print(&self) {
        println!("╔══ Tournament: {} vs {} ══", self.p1_label, self.p2_label);
        println!(
            "║  games={}  P1: W={} L={} D={}",
            self.games, self.p1_wins, self.p2_wins, self.draws
        );
        println!(
            "║  P1 win-rate={:.3}  Elo-diff={:+.1}",
            self.win_rate_p1(),
            self.elo_diff()
        );
        println!(
            "║  avg VOCfocus={:.4}  σ_Δ={:.4}  fep_mod={:.3}",
            self.avg_voc_focus, self.avg_sigma_delta, self.avg_fep_mod
        );
        println!("╚══");
    }
}

// ─────────────────────────────────────────────
// § 단일 대국
// ─────────────────────────────────────────────

/// 두 엔진의 단일 자가 대국 결과 (1=p1 win, -1=p2 win, 0=draw)
fn play_one_game<G, E>(
    init: G,
    eval1: Arc<E>,
    eval2: Arc<E>,
    config1: MctsConfig,
    config2: MctsConfig,
    iters: u32,
    quartz_cfg: Option<QuartzConfig>,
) -> (i8, Vec<QuartzStats>)
where
    G: GameState,
    E: Evaluator<G> + Send + Sync + 'static,
{
    let mut state = init;
    let mut move_count = 0usize;
    let mut stats_log = Vec::new();

    loop {
        if state.is_terminal() {
            break;
        }
        let is_p1_turn = move_count % 2 == 0;

        let mv = if is_p1_turn {
            // Player 1 — QUARTZ (if configured)
            let engine = MctsEngine::new(state.clone(), eval1.clone(), config1.clone());
            if let Some(ref qcfg) = quartz_cfg {
                let mut ctrl = QuartzController::new(iters, qcfg.clone());
                let _ = engine.run_quartz(&mut ctrl);
                if let Some(s) = engine.current_quartz_stats() {
                    stats_log.push(s);
                }
            } else {
                engine.run(&mut FixedIterations::new(iters));
            }
            engine.best_move()
        } else {
            // Player 2 — baseline
            let engine = MctsEngine::new(state.clone(), eval2.clone(), config2.clone());
            engine.run(&mut FixedIterations::new(iters));
            engine.best_move()
        };

        match mv {
            Some(m) => state = state.apply_move(m),
            None => break,
        }
        move_count += 1;
        if move_count > 500 {
            break;
        } // timeout → draw
    }

    // 최종 outcome (P1 관점)
    let outcome = if state.is_terminal() {
        let raw = state.outcome(); // current player 관점
                                   // negamax: move_count번 진행됐으면 현재 플레이어는 (move_count%2==0 → P1, 1 → P2)
        let flip = if move_count % 2 == 0 { 1.0_f32 } else { -1.0 };
        let p1_val = raw * flip;
        if p1_val > 0.5 {
            1
        } else if p1_val < -0.5 {
            -1
        } else {
            0
        }
    } else {
        0 // timeout → draw
    };

    (outcome, stats_log)
}

// ─────────────────────────────────────────────
// § 토너먼트
// ─────────────────────────────────────────────

pub struct TournamentConfig {
    pub n_games: u32,
    pub iters: u32,
    pub p1_label: String,
    pub p2_label: String,
}

pub fn run_tournament<G, E>(
    init: G,
    evaluator: Arc<E>,
    p1_config: MctsConfig, // QUARTZ
    p2_config: MctsConfig, // baseline
    quartz_cfg: Option<QuartzConfig>,
    cfg: TournamentConfig,
) -> TournamentResult
where
    G: GameState + Clone,
    E: Evaluator<G> + Send + Sync + 'static,
{
    let mut result = TournamentResult {
        p1_label: cfg.p1_label.clone(),
        p2_label: cfg.p2_label.clone(),
        ..Default::default()
    };

    let mut total_voc = 0.0f32;
    let mut total_sigma = 0.0f32;
    let mut total_fep = 0.0f32;
    let mut n_stats = 0u32;

    for game_n in 0..cfg.n_games {
        let (outcome, stats_log) = play_one_game(
            init.clone(),
            Arc::clone(&evaluator),
            Arc::clone(&evaluator),
            p1_config.clone(),
            p2_config.clone(),
            cfg.iters,
            quartz_cfg.clone(),
        );

        result.games += 1;
        match outcome {
            1 => result.p1_wins += 1,
            -1 => result.p2_wins += 1,
            _ => result.draws += 1,
        }

        for s in &stats_log {
            total_voc += s.voc_focus;
            total_sigma += s.sigma_delta;
            total_fep += s.envar_delta;
            n_stats += 1;
        }

        // 진행 상황 출력 (10게임마다)
        if (game_n + 1) % 10 == 0 {
            println!(
                "  [{}/{}] P1: W={} L={} D={}  wr={:.3}",
                game_n + 1,
                cfg.n_games,
                result.p1_wins,
                result.p2_wins,
                result.draws,
                result.win_rate_p1()
            );
        }
    }

    if n_stats > 0 {
        result.avg_voc_focus = total_voc / n_stats as f32;
        result.avg_sigma_delta = total_sigma / n_stats as f32;
        result.avg_fep_mod = total_fep / n_stats as f32;
    }

    result
}

// ─────────────────────────────────────────────
// § wimp(s) 분석 — 탐색 중 비루트 노드 wimp 분포
// ─────────────────────────────────────────────

/// 탐색 후 루트 자식들의 wimp 분포를 계산
pub fn analyze_wimp<G, E>(state: G, evaluator: Arc<E>, config: MctsConfig, iters: u32)
where
    G: GameState,
    E: Evaluator<G> + Send + Sync + 'static,
{
    let engine = MctsEngine::new(state, evaluator, config);
    engine.run(&mut FixedIterations::new(iters));

    let root_n = engine
        .root
        .n_total
        .load(std::sync::atomic::Ordering::Acquire);
    let n_mat = engine.root.materialized_count();
    let edges = engine.root.edge_snapshot(n_mat);

    println!("╔══ wimp(s) 분포 (root N={}) ══", root_n);
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}",
        "edge", "N_child", "wimp", "VOC_proxy"
    );
    println!("  {}", "-".repeat(40));

    let mut wimps: Vec<(f32, f32, f32)> = Vec::new();
    for e in &edges {
        let n_child = e.child.n_total.load(std::sync::atomic::Ordering::Acquire);
        let w = e.child.wimp(root_n);
        let q = e.q();
        wimps.push((w, q, n_child as f32));
    }
    wimps.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    for (i, &(w, q, nc)) in wimps.iter().take(8).enumerate() {
        println!(
            "  {:>6}  {:>8.0}  {:>8.4}  {:>8.4}",
            i + 1,
            nc,
            w,
            w * q.abs()
        );
    }
    println!("╚══");
}
