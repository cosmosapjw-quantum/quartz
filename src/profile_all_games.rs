use std::env;
use std::hint::black_box;
use std::mem;
use std::sync::Arc;
use std::time::Instant;

use crate::game::GameState;
use crate::games::chess::{chess_quartz, Chess};
use crate::games::go::{go_quartz, Go};
use crate::games::gomoku::Gomoku;
use crate::games::gomoku15::{gomoku15_quartz, Gomoku15, GomokuVariant};
use crate::games::tictactoe::TicTacToe;
use crate::mcts::eval::UniformEval;
use crate::mcts::node::{edge_lock_contention_snapshot, reset_edge_lock_contention_counters};
use crate::mcts::profiling::hot_path_metrics_enabled;
use crate::mcts::search::FixedIterations;
use crate::mcts::{
    engine_phase_snapshot, parallel::AutoThreadPolicy, reset_engine_phase_counters, MctsConfig,
    MctsEngine,
};

#[derive(Clone, Copy)]
struct GameOpsProfile {
    legal_us: f64,
    apply_undo_us: f64,
    encode_us: f64,
    legal_count: usize,
}

#[derive(Clone, Copy)]
struct MctsProfileSummary {
    nps: f64,
    elapsed_ms: f64,
    select_ns_iter: f64,
    expand_ns_iter: f64,
    backprop_ns_iter: f64,
    tt_lock_wait_ns_iter: f64,
    tt_read_lock_wait_ns_iter: f64,
    tt_write_lock_wait_ns_iter: f64,
    edge_lock_wait_ns_iter: f64,
    edge_busy_skip_iter: f64,
}

fn profile_game_ops<G: GameState>(state: &G, reps: usize) -> GameOpsProfile {
    let reps = reps.max(1);

    let start = Instant::now();
    let mut legal_count = 0usize;
    for _ in 0..reps {
        let moves = state.legal_moves();
        legal_count = legal_count.saturating_add(black_box(moves.len()));
    }
    let legal_us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let moves = state.legal_moves();
    let apply_undo_us = if moves.is_empty() {
        0.0
    } else {
        let mut probe = state.clone();
        let start = Instant::now();
        for i in 0..reps {
            let undo = probe.apply_move_in_place(moves[i % moves.len()]);
            black_box(probe.tt_hash());
            probe.undo_move(undo);
        }
        start.elapsed().as_secs_f64() * 1e6 / reps as f64
    };

    let mut scratch = Vec::new();
    let start = Instant::now();
    for _ in 0..reps {
        state.encode_planes_into(&mut scratch);
        black_box(&scratch);
    }
    let encode_us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;

    GameOpsProfile {
        legal_us,
        apply_undo_us,
        encode_us,
        legal_count: legal_count / reps,
    }
}

fn scaled_iters(default: u32) -> u32 {
    if let Ok(raw) = env::var("QUARTZ_PROFILE_ITERS") {
        if let Ok(parsed) = raw.parse::<u32>() {
            return parsed.max(1);
        }
    }
    let scale = env::var("QUARTZ_PROFILE_SCALE")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0);
    ((default as f64 * scale).round() as u32).max(1)
}

fn scaled_parallel_iters(default: u32) -> u32 {
    if let Ok(raw) = env::var("QUARTZ_PROFILE_PAR_ITERS") {
        if let Ok(parsed) = raw.parse::<u32>() {
            return parsed.max(1);
        }
    }
    scaled_iters(default)
}

fn scaled_reps(default: usize) -> usize {
    if let Ok(raw) = env::var("QUARTZ_PROFILE_OPS_REPS") {
        if let Ok(parsed) = raw.parse::<usize>() {
            return parsed.max(1);
        }
    }
    default.max(1)
}

fn profile_threads(default: usize) -> usize {
    if let Ok(raw) = env::var("QUARTZ_PROFILE_THREADS") {
        if let Ok(parsed) = raw.parse::<usize>() {
            return parsed.max(1);
        }
    }
    default.max(1)
}

fn profile_auto_thread_policy() -> Option<AutoThreadPolicy> {
    let raw = env::var("QUARTZ_PROFILE_THREADS").ok()?;
    match raw.as_str() {
        "auto" | "auto-throughput" | "throughput" => Some(AutoThreadPolicy::throughput()),
        "auto-quality" | "quality" => Some(AutoThreadPolicy::quality()),
        _ => None,
    }
}

fn profile_repeats() -> usize {
    if let Ok(raw) = env::var("QUARTZ_PROFILE_REPEATS") {
        if let Ok(parsed) = raw.parse::<usize>() {
            return parsed.max(1);
        }
    }
    1
}

fn skip_ops() -> bool {
    matches!(
        env::var("QUARTZ_PROFILE_SKIP_OPS").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn min_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, v| acc.min(v))
}

fn max_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(v))
}

fn print_profile_summary(
    kind: &str,
    label: &str,
    threads: Option<usize>,
    summaries: &[MctsProfileSummary],
) {
    if summaries.len() < 2 {
        return;
    }

    let nps: Vec<f64> = summaries.iter().map(|s| s.nps).collect();
    let elapsed: Vec<f64> = summaries.iter().map(|s| s.elapsed_ms).collect();
    let select_ns: Vec<f64> = summaries.iter().map(|s| s.select_ns_iter).collect();
    let expand_ns: Vec<f64> = summaries.iter().map(|s| s.expand_ns_iter).collect();
    let backprop_ns: Vec<f64> = summaries.iter().map(|s| s.backprop_ns_iter).collect();
    let tt_wait_ns: Vec<f64> = summaries.iter().map(|s| s.tt_lock_wait_ns_iter).collect();
    let tt_read_wait_ns: Vec<f64> = summaries
        .iter()
        .map(|s| s.tt_read_lock_wait_ns_iter)
        .collect();
    let tt_write_wait_ns: Vec<f64> = summaries
        .iter()
        .map(|s| s.tt_write_lock_wait_ns_iter)
        .collect();
    let edge_wait_ns: Vec<f64> = summaries.iter().map(|s| s.edge_lock_wait_ns_iter).collect();
    let edge_busy_skip: Vec<f64> = summaries.iter().map(|s| s.edge_busy_skip_iter).collect();
    let threads_field = threads
        .map(|n| format!("\tthreads={n}"))
        .unwrap_or_default();
    println!(
        "{kind}\tgame={label}{threads_field}\trepeats={}\tnps_min={:.0}\tnps_median={:.0}\tnps_mean={:.0}\tnps_max={:.0}\telapsed_ms_median={:.3}\tselect_ns_iter_median={:.1}\texpand_ns_iter_median={:.1}\tbackprop_ns_iter_median={:.1}\ttt_lock_wait_ns_iter_median={:.1}\ttt_read_lock_wait_ns_iter_median={:.1}\ttt_write_lock_wait_ns_iter_median={:.1}\tedge_lock_wait_ns_iter_median={:.1}\tedge_busy_skip_iter_median={:.3}",
        summaries.len(),
        min_value(&nps),
        median(&nps),
        mean(&nps),
        max_value(&nps),
        median(&elapsed),
        median(&select_ns),
        median(&expand_ns),
        median(&backprop_ns),
        median(&tt_wait_ns),
        median(&tt_read_wait_ns),
        median(&tt_write_wait_ns),
        median(&edge_wait_ns),
        median(&edge_busy_skip),
    );
}

fn profile_mcts<G: GameState>(
    label: &str,
    state: G,
    config: MctsConfig,
    iters: u32,
    ops_reps: usize,
) {
    let repeats = profile_repeats();
    let mut summaries = Vec::with_capacity(repeats);
    for repeat_idx in 1..=repeats {
        summaries.push(profile_mcts_once(
            label,
            state.clone(),
            config.clone(),
            iters,
            ops_reps,
            repeat_idx,
            repeats,
        ));
    }
    print_profile_summary("PROFILE_MCTS_SUMMARY", label, None, &summaries);
}

fn profile_mcts_once<G: GameState>(
    label: &str,
    state: G,
    config: MctsConfig,
    iters: u32,
    ops_reps: usize,
    repeat_idx: usize,
    repeats: usize,
) -> MctsProfileSummary {
    let iters = scaled_iters(iters);
    let ops_reps = scaled_reps(ops_reps);
    let has_pw = config.pw.is_some();
    let has_quartz = config.quartz.is_some();
    let has_gvoc = config.gvoc.is_some();
    let select_scratch = G::uses_reusable_select_scratch();
    let state_bytes = mem::size_of::<G>();
    let ops = if skip_ops() {
        GameOpsProfile {
            legal_us: 0.0,
            apply_undo_us: 0.0,
            encode_us: 0.0,
            legal_count: state.legal_move_count(),
        }
    } else {
        profile_game_ops(&state, ops_reps)
    };

    let eval = Arc::new(UniformEval);
    let engine = MctsEngine::new(state, eval, config);
    reset_edge_lock_contention_counters();
    reset_engine_phase_counters();
    let start = Instant::now();
    let stats = engine.run(&mut FixedIterations::new(iters));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let phase = engine_phase_snapshot();
    let phase_iters = phase.iterate_calls.max(1) as f64;
    let root_edges = engine.root.materialized_count();
    let best = engine.best_move();
    let vl = engine.par_ctrl.telemetry.snapshot();
    let tt_locks = engine.tt.contention_snapshot();
    let edge_locks = edge_lock_contention_snapshot();
    let select_ns_iter = phase.select_time_nanos as f64 / phase_iters;
    let expand_ns_iter = phase.expand_eval_time_nanos as f64 / phase_iters;
    let backprop_ns_iter = phase.backprop_time_nanos as f64 / phase_iters;
    let tt_lock_wait_ns_iter = tt_locks.lock_wait_nanos as f64 / phase_iters;
    let tt_read_lock_wait_ns_iter = tt_locks.read_lock_wait_nanos as f64 / phase_iters;
    let tt_write_lock_wait_ns_iter = tt_locks.write_lock_wait_nanos as f64 / phase_iters;
    let edge_lock_wait_ns_iter = edge_locks.wait_nanos as f64 / phase_iters;
    let edge_busy_skip_iter = edge_locks.busy_skips as f64 / phase_iters;

    println!(
        "PROFILE_MCTS\tgame={label}\trepeat={repeat_idx}\trepeats={repeats}\titers={iters}\telapsed_ms={elapsed_ms:.3}\tnps={:.0}\troot_visits={}\troot_edges={root_edges}\ttt_hit_rate={:.4}\ttt_size={}\tstate_bytes={state_bytes}\tselect_scratch={select_scratch}\tlegal0={}\tlegal_us={:.3}\tapply_undo_us={:.3}\tencode_us={:.3}\tselect_ns_iter={select_ns_iter:.1}\texpand_ns_iter={expand_ns_iter:.1}\tbackprop_ns_iter={backprop_ns_iter:.1}\ttt_goc_iter={:.3}\ttt_get_iter={:.3}\ttt_lock_wait_ns_iter={tt_lock_wait_ns_iter:.1}\ttt_read_lock_wait_ns_iter={tt_read_lock_wait_ns_iter:.1}\ttt_write_lock_wait_ns_iter={tt_write_lock_wait_ns_iter:.1}\ttt_max_lock_wait_ns={}\ttt_max_read_lock_wait_ns={}\ttt_max_write_lock_wait_ns={}\tedge_lock_calls_iter={:.3}\tedge_lock_wait_ns_iter={edge_lock_wait_ns_iter:.1}\tedge_max_lock_wait_ns={}\tedge_busy_skip_iter={:.3}\thotpath_metrics={}\tvl_selects={}\tvl_max_pending={}\tvl_dup_rate={:.4}\tpw={has_pw}\tquartz={has_quartz}\tgvoc={has_gvoc}\tbest={:?}",
        stats.nps,
        stats.root_visits,
        stats.tt_hit_rate,
        stats.tt_size,
        ops.legal_count,
        ops.legal_us,
        ops.apply_undo_us,
        ops.encode_us,
        tt_locks.get_or_create_calls as f64 / phase_iters,
        tt_locks.get_calls as f64 / phase_iters,
        tt_locks.max_lock_wait_nanos,
        tt_locks.max_read_lock_wait_nanos,
        tt_locks.max_write_lock_wait_nanos,
        edge_locks.calls as f64 / phase_iters,
        edge_locks.max_wait_nanos,
        edge_busy_skip_iter,
        hot_path_metrics_enabled(),
        vl.total_selects,
        vl.max_pending,
        vl.dup_rate,
        best,
    );

    MctsProfileSummary {
        nps: stats.nps,
        elapsed_ms,
        select_ns_iter,
        expand_ns_iter,
        backprop_ns_iter,
        tt_lock_wait_ns_iter,
        tt_read_lock_wait_ns_iter,
        tt_write_lock_wait_ns_iter,
        edge_lock_wait_ns_iter,
        edge_busy_skip_iter,
    }
}

fn profile_mcts_parallel<G: GameState>(
    label: &str,
    state: G,
    config: MctsConfig,
    iters: u32,
    ops_reps: usize,
    threads: usize,
) {
    let repeats = profile_repeats();
    let auto_policy = profile_auto_thread_policy();
    let threads = profile_threads(threads);
    let mut summaries = Vec::with_capacity(repeats);
    for repeat_idx in 1..=repeats {
        summaries.push(profile_mcts_parallel_once(
            label,
            state.clone(),
            config.clone(),
            iters,
            ops_reps,
            threads,
            auto_policy,
            repeat_idx,
            repeats,
        ));
    }
    print_profile_summary(
        "PROFILE_MCTS_PAR_SUMMARY",
        label,
        if auto_policy.is_some() {
            None
        } else {
            Some(threads)
        },
        &summaries,
    );
}

fn profile_mcts_parallel_once<G: GameState>(
    label: &str,
    state: G,
    config: MctsConfig,
    iters: u32,
    ops_reps: usize,
    threads: usize,
    auto_policy: Option<AutoThreadPolicy>,
    repeat_idx: usize,
    repeats: usize,
) -> MctsProfileSummary {
    let iters = scaled_parallel_iters(iters);
    let ops_reps = scaled_reps(ops_reps);
    let has_pw = config.pw.is_some();
    let has_quartz = config.quartz.is_some();
    let has_gvoc = config.gvoc.is_some();
    let select_scratch = G::uses_reusable_select_scratch();
    let state_bytes = mem::size_of::<G>();
    let ops = if skip_ops() {
        GameOpsProfile {
            legal_us: 0.0,
            apply_undo_us: 0.0,
            encode_us: 0.0,
            legal_count: state.legal_move_count(),
        }
    } else {
        profile_game_ops(&state, ops_reps)
    };

    let eval = Arc::new(UniformEval);
    let engine = MctsEngine::new(state, eval, config);
    let mut ctrl = FixedIterations::new(iters);
    reset_edge_lock_contention_counters();
    reset_engine_phase_counters();
    let start = Instant::now();
    let (stats, effective_threads, thread_policy, auto_reason) = if let Some(policy) = auto_policy {
        let (stats, decision) = engine.run_auto(&mut ctrl, policy);
        (
            stats,
            decision.threads,
            match policy.mode {
                crate::mcts::parallel::AutoThreadMode::Throughput => "auto-throughput",
                crate::mcts::parallel::AutoThreadMode::Quality => "auto-quality",
            },
            format!("{:?}", decision.reason),
        )
    } else {
        (
            engine.run_par(&ctrl, threads),
            threads,
            "fixed",
            "Manual".to_string(),
        )
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let phase = engine_phase_snapshot();
    let phase_iters = phase.iterate_calls.max(1) as f64;
    let root_edges = engine.root.materialized_count();
    let best = engine.best_move();
    let vl = engine.par_ctrl.telemetry.snapshot();
    let tt_locks = engine.tt.contention_snapshot();
    let edge_locks = edge_lock_contention_snapshot();
    let select_ns_iter = phase.select_time_nanos as f64 / phase_iters;
    let expand_ns_iter = phase.expand_eval_time_nanos as f64 / phase_iters;
    let backprop_ns_iter = phase.backprop_time_nanos as f64 / phase_iters;
    let tt_lock_wait_ns_iter = tt_locks.lock_wait_nanos as f64 / phase_iters;
    let tt_read_lock_wait_ns_iter = tt_locks.read_lock_wait_nanos as f64 / phase_iters;
    let tt_write_lock_wait_ns_iter = tt_locks.write_lock_wait_nanos as f64 / phase_iters;
    let edge_lock_wait_ns_iter = edge_locks.wait_nanos as f64 / phase_iters;
    let edge_busy_skip_iter = edge_locks.busy_skips as f64 / phase_iters;

    println!(
        "PROFILE_MCTS_PAR\tgame={label}\tthreads={effective_threads}\tthread_policy={thread_policy}\tauto_reason={auto_reason}\trepeat={repeat_idx}\trepeats={repeats}\titers={iters}\telapsed_ms={elapsed_ms:.3}\tnps={:.0}\troot_visits={}\troot_edges={root_edges}\ttt_hit_rate={:.4}\ttt_size={}\tstate_bytes={state_bytes}\tselect_scratch={select_scratch}\tlegal0={}\tlegal_us={:.3}\tapply_undo_us={:.3}\tencode_us={:.3}\tselect_ns_iter={select_ns_iter:.1}\texpand_ns_iter={expand_ns_iter:.1}\tbackprop_ns_iter={backprop_ns_iter:.1}\ttt_goc_iter={:.3}\ttt_get_iter={:.3}\ttt_lock_wait_ns_iter={tt_lock_wait_ns_iter:.1}\ttt_read_lock_wait_ns_iter={tt_read_lock_wait_ns_iter:.1}\ttt_write_lock_wait_ns_iter={tt_write_lock_wait_ns_iter:.1}\ttt_max_lock_wait_ns={}\ttt_max_read_lock_wait_ns={}\ttt_max_write_lock_wait_ns={}\tedge_lock_calls_iter={:.3}\tedge_lock_wait_ns_iter={edge_lock_wait_ns_iter:.1}\tedge_max_lock_wait_ns={}\tedge_busy_skip_iter={:.3}\thotpath_metrics={}\tvl_selects={}\tvl_max_pending={}\tvl_dup_rate={:.4}\tvl_avg_vvalue={:.4}\tpw={has_pw}\tquartz={has_quartz}\tgvoc={has_gvoc}\tbest={:?}",
        stats.nps,
        stats.root_visits,
        stats.tt_hit_rate,
        stats.tt_size,
        ops.legal_count,
        ops.legal_us,
        ops.apply_undo_us,
        ops.encode_us,
        tt_locks.get_or_create_calls as f64 / phase_iters,
        tt_locks.get_calls as f64 / phase_iters,
        tt_locks.max_lock_wait_nanos,
        tt_locks.max_read_lock_wait_nanos,
        tt_locks.max_write_lock_wait_nanos,
        edge_locks.calls as f64 / phase_iters,
        edge_locks.max_wait_nanos,
        edge_busy_skip_iter,
        hot_path_metrics_enabled(),
        vl.total_selects,
        vl.max_pending,
        vl.dup_rate,
        vl.avg_vvalue,
        best,
    );

    MctsProfileSummary {
        nps: stats.nps,
        elapsed_ms,
        select_ns_iter,
        expand_ns_iter,
        backprop_ns_iter,
        tt_lock_wait_ns_iter,
        tt_read_lock_wait_ns_iter,
        tt_write_lock_wait_ns_iter,
        edge_lock_wait_ns_iter,
        edge_busy_skip_iter,
    }
}

fn run_tictactoe() {
    profile_mcts(
        "tictactoe",
        TicTacToe::initial(),
        MctsConfig::evaluation(2.0),
        50_000,
        20_000,
    );
}

fn run_tictactoe_parallel() {
    profile_mcts_parallel(
        "tictactoe",
        TicTacToe::initial(),
        MctsConfig::evaluation(2.0),
        50_000,
        20_000,
        4,
    );
}

fn run_gomoku7() {
    profile_mcts(
        "gomoku7",
        Gomoku::new(7),
        MctsConfig::evaluation(2.0),
        15_000,
        10_000,
    );
}

fn run_gomoku7_parallel() {
    profile_mcts_parallel(
        "gomoku7",
        Gomoku::new(7),
        MctsConfig::evaluation(2.0),
        15_000,
        10_000,
        4,
    );
}

fn run_gomoku15_variant(label: &str, variant: GomokuVariant) {
    profile_mcts(
        label,
        Gomoku15::new(variant),
        gomoku15_quartz(variant),
        30_000,
        3_000,
    );
}

fn run_gomoku15_variant_parallel(label: &str, variant: GomokuVariant) {
    profile_mcts_parallel(
        label,
        Gomoku15::new(variant),
        gomoku15_quartz(variant),
        30_000,
        3_000,
        4,
    );
}

fn run_chess() {
    profile_mcts("chess", Chess::standard(), chess_quartz(), 8_000, 2_000);
}

fn run_chess_parallel() {
    profile_mcts_parallel("chess", Chess::standard(), chess_quartz(), 8_000, 2_000, 4);
}

fn run_go9() {
    profile_mcts("go9", Go::new_9x9(), go_quartz(9), 8_000, 1_500);
}

fn run_go9_parallel() {
    profile_mcts_parallel("go9", Go::new_9x9(), go_quartz(9), 8_000, 1_500, 4);
}

fn run_go13() {
    profile_mcts("go13", Go::new_13x13(), go_quartz(13), 4_000, 800);
}

fn run_go13_parallel() {
    profile_mcts_parallel("go13", Go::new_13x13(), go_quartz(13), 4_000, 800, 4);
}

fn run_go19() {
    profile_mcts("go19", Go::new_19x19(), go_quartz(19), 1_500, 300);
}

fn run_go19_parallel() {
    profile_mcts_parallel("go19", Go::new_19x19(), go_quartz(19), 1_500, 300, 4);
}

#[test]
#[ignore]
fn profile_mcts_all_supported_games() {
    run_tictactoe();
    run_gomoku7();
    run_gomoku15_variant("gomoku15_freestyle", GomokuVariant::Freestyle);
    run_gomoku15_variant("gomoku15_standard", GomokuVariant::Standard);
    run_gomoku15_variant("gomoku15_omok", GomokuVariant::Omok);
    run_gomoku15_variant("gomoku15_renju", GomokuVariant::Renju);
    run_gomoku15_variant("gomoku15_caro", GomokuVariant::Caro);
    run_chess();
    run_go9();
    run_go13();
    run_go19();
}

#[test]
#[ignore]
fn profile_mcts_parallel_all_supported_games() {
    run_tictactoe_parallel();
    run_gomoku7_parallel();
    run_gomoku15_variant_parallel("gomoku15_freestyle", GomokuVariant::Freestyle);
    run_gomoku15_variant_parallel("gomoku15_standard", GomokuVariant::Standard);
    run_gomoku15_variant_parallel("gomoku15_omok", GomokuVariant::Omok);
    run_gomoku15_variant_parallel("gomoku15_renju", GomokuVariant::Renju);
    run_gomoku15_variant_parallel("gomoku15_caro", GomokuVariant::Caro);
    run_chess_parallel();
    run_go9_parallel();
    run_go13_parallel();
    run_go19_parallel();
}

#[test]
#[ignore]
fn profile_mcts_tictactoe() {
    run_tictactoe();
}

#[test]
#[ignore]
fn profile_mcts_parallel_tictactoe() {
    run_tictactoe_parallel();
}

#[test]
#[ignore]
fn profile_mcts_gomoku7() {
    run_gomoku7();
}

#[test]
#[ignore]
fn profile_mcts_parallel_gomoku7() {
    run_gomoku7_parallel();
}

#[test]
#[ignore]
fn profile_mcts_gomoku15_freestyle() {
    run_gomoku15_variant("gomoku15_freestyle", GomokuVariant::Freestyle);
}

#[test]
#[ignore]
fn profile_mcts_parallel_gomoku15_freestyle() {
    run_gomoku15_variant_parallel("gomoku15_freestyle", GomokuVariant::Freestyle);
}

#[test]
#[ignore]
fn profile_mcts_gomoku15_standard() {
    run_gomoku15_variant("gomoku15_standard", GomokuVariant::Standard);
}

#[test]
#[ignore]
fn profile_mcts_parallel_gomoku15_standard() {
    run_gomoku15_variant_parallel("gomoku15_standard", GomokuVariant::Standard);
}

#[test]
#[ignore]
fn profile_mcts_gomoku15_omok() {
    run_gomoku15_variant("gomoku15_omok", GomokuVariant::Omok);
}

#[test]
#[ignore]
fn profile_mcts_parallel_gomoku15_omok() {
    run_gomoku15_variant_parallel("gomoku15_omok", GomokuVariant::Omok);
}

#[test]
#[ignore]
fn profile_mcts_gomoku15_renju() {
    run_gomoku15_variant("gomoku15_renju", GomokuVariant::Renju);
}

#[test]
#[ignore]
fn profile_mcts_parallel_gomoku15_renju() {
    run_gomoku15_variant_parallel("gomoku15_renju", GomokuVariant::Renju);
}

#[test]
#[ignore]
fn profile_mcts_gomoku15_caro() {
    run_gomoku15_variant("gomoku15_caro", GomokuVariant::Caro);
}

#[test]
#[ignore]
fn profile_mcts_parallel_gomoku15_caro() {
    run_gomoku15_variant_parallel("gomoku15_caro", GomokuVariant::Caro);
}

#[test]
#[ignore]
fn profile_mcts_chess() {
    run_chess();
}

#[test]
#[ignore]
fn profile_mcts_parallel_chess() {
    run_chess_parallel();
}

#[test]
#[ignore]
fn profile_mcts_go9() {
    run_go9();
}

#[test]
#[ignore]
fn profile_mcts_parallel_go9() {
    run_go9_parallel();
}

#[test]
#[ignore]
fn profile_mcts_go13() {
    run_go13();
}

#[test]
#[ignore]
fn profile_mcts_parallel_go13() {
    run_go13_parallel();
}

#[test]
#[ignore]
fn profile_mcts_go19() {
    run_go19();
}

#[test]
#[ignore]
fn profile_mcts_parallel_go19() {
    run_go19_parallel();
}
