//! Multi-game MCTS JSON-line server — trajectory + move protocols
//!
//! Protocol 1 (single move): {"cmd":"move","game":"chess","fen":"...","iters":200}
//! Protocol 2 (self-play):   {"cmd":"selfplay","game":"gomoku15","iters":400,"n_games":1,"temp_threshold":15}
//!   → full game trajectory with (state_planes, policy, player, outcome)

use std::io::{self, BufRead, Write};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::game::GameState;
use crate::mcts::eval::{AsyncEvalTicket, BatchStdioEval, GlobalBroker, ShortRollout};
use crate::mcts::node::edge_lock_contention_snapshot;
use crate::mcts::quartz::{PenaltyMode, QuartzConfig, QuartzController};
use crate::mcts::search::FixedIterations;
use crate::mcts::{engine_phase_snapshot, MctsConfig, MctsEngine, PreparedIteration};

use crate::games::chess::{chess_quartz, Chess, ChessMove};
use crate::games::go::{go_quartz, Go, GoRuleset, GoScoring};
use crate::games::gomoku15::{gomoku15_quartz, Gomoku15, GomokuVariant};
use crate::games::{Gomoku, TicTacToe};

const CHESS_POLICY_ACTIONS: usize = 4096;

fn rust_server_trace_path() -> Option<&'static str> {
    static TRACE_PATH: OnceLock<Option<String>> = OnceLock::new();
    TRACE_PATH
        .get_or_init(|| {
            std::env::var("QUARTZ_RUST_SERVER_TRACE")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .as_deref()
}

fn rust_server_trace(event: &str, fields: serde_json::Value) {
    let Some(path) = rust_server_trace_path() else {
        return;
    };
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let mut obj = serde_json::Map::new();
    obj.insert("ts".to_string(), serde_json::json!(ts));
    obj.insert("pid".to_string(), serde_json::json!(std::process::id()));
    obj.insert("event".to_string(), serde_json::json!(event));
    if let serde_json::Value::Object(map) = fields {
        for (k, v) in map {
            obj.insert(k, v);
        }
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let mut line = serde_json::Value::Object(obj).to_string().into_bytes();
        line.push(b'\n');
        let _ = file.write_all(&line);
    }
}

fn emit_json_message(payload: &serde_json::Value) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{}", payload);
    let _ = out.flush();
}

fn parse_result_value(resp: String) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(&resp)
        .ok()
        .and_then(|v| v.get("result").cloned())
        .unwrap_or_else(|| serde_json::json!({}))
}

fn jstr<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start(); // skip whitespace after colon
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(&inner[..end])
    } else {
        None
    }
}
fn jint(s: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}
fn jarr(s: &str, key: &str) -> Vec<i64> {
    let pat = format!("\"{}\":[", key);
    if let Some(start) = s.find(&pat) {
        let rest = &s[start + pat.len()..];
        if let Some(end) = rest.find(']') {
            return rest[..end]
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect();
        }
    }
    vec![]
}
fn jfloat(s: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}
fn jbool(s: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{}\":", key);
    let start = s.find(&pat)? + pat.len();
    let rest = s[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}
fn f_or(v: f32, d: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        d
    }
}

fn chess_policy_index(mv: ChessMove) -> usize {
    (mv.from_sq() as usize) * 64 + mv.to_sq() as usize
}

fn chess_outcome_for_white(state: &Chess) -> f32 {
    let raw = state.outcome();
    raw * state.current_player() as f32
}

fn find_chess_move_by_uci(state: &Chess, uci: &str) -> Option<ChessMove> {
    state
        .generate_legal_moves()
        .into_iter()
        .find(|mv| mv.to_uci() == uci)
}

fn handle_chess_state(line: &str, default_960: bool) -> String {
    let state = chess_state_from_request(line, default_960);
    let legal = state.generate_legal_moves();
    let legal_moves = legal
        .iter()
        .map(|mv| format!("\"{}\"", mv.to_uci()))
        .collect::<Vec<_>>()
        .join(",");
    let legal_actions = legal
        .iter()
        .map(|mv| chess_policy_index(*mv).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let terminal = state.is_terminal();
    let outcome = if terminal {
        chess_outcome_for_white(&state)
    } else {
        0.0
    };
    format!(
        concat!(
            "{{\"status\":\"ok\",",
            "\"fen\":\"{}\",",
            "\"side_to_move\":\"{}\",",
            "\"terminal\":{},",
            "\"outcome_white\":{:.4},",
            "\"legal_moves\":[{}],",
            "\"legal_actions\":[{}]",
            "}}"
        ),
        state.to_fen(),
        if state.current_player() > 0 { "w" } else { "b" },
        if terminal { "true" } else { "false" },
        outcome,
        legal_moves,
        legal_actions
    )
}

fn handle_chess_apply(line: &str, default_960: bool) -> String {
    let state = chess_state_from_request(line, default_960);
    let move_uci = match jstr(line, "move_uci") {
        Some(v) => v,
        None => return "{\"status\":\"error\",\"error\":\"missing move_uci\"}".into(),
    };
    let Some(mv) = find_chess_move_by_uci(&state, move_uci) else {
        return format!(
            "{{\"status\":\"error\",\"error\":\"illegal move: {}\"}}",
            move_uci
        );
    };
    let next = state.apply_move(mv);
    let legal = next.generate_legal_moves();
    let legal_moves = legal
        .iter()
        .map(|m| format!("\"{}\"", m.to_uci()))
        .collect::<Vec<_>>()
        .join(",");
    let legal_actions = legal
        .iter()
        .map(|m| chess_policy_index(*m).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let terminal = next.is_terminal();
    let outcome = if terminal {
        chess_outcome_for_white(&next)
    } else {
        0.0
    };
    format!(
        concat!(
            "{{\"status\":\"ok\",",
            "\"applied_move\":\"{}\",",
            "\"fen\":\"{}\",",
            "\"side_to_move\":\"{}\",",
            "\"terminal\":{},",
            "\"outcome_white\":{:.4},",
            "\"legal_moves\":[{}],",
            "\"legal_actions\":[{}]",
            "}}"
        ),
        mv.to_uci(),
        next.to_fen(),
        if next.current_player() > 0 { "w" } else { "b" },
        if terminal { "true" } else { "false" },
        outcome,
        legal_moves,
        legal_actions
    )
}

#[derive(Clone, Copy)]
struct SearchOverrides {
    penalty_mode: PenaltyMode,
    hbar_penalty_cap: Option<f32>,
    c_puct: Option<f32>,
    sigma_0: Option<f32>,
    min_visits: Option<u32>,
    check_interval: Option<u32>,
    prior_refresh_rate: Option<f32>,
    prior_refresh_temp: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchProfile {
    Quartz,
    Baseline,
    BaselineStrict,
}

fn parse_search_overrides(line: &str) -> SearchOverrides {
    SearchOverrides {
        penalty_mode: parse_penalty_mode(jstr(line, "penalty_mode").unwrap_or("GatedRefresh")),
        hbar_penalty_cap: jfloat(line, "hbar_penalty_cap")
            .map(|v| v as f32)
            .filter(|v| *v > 0.0),
        c_puct: jfloat(line, "c_puct")
            .map(|v| v as f32)
            .filter(|v| *v > 0.0),
        sigma_0: jfloat(line, "sigma_0")
            .map(|v| v as f32)
            .filter(|v| *v > 0.0),
        min_visits: jint(line, "min_visits").map(|v| v.max(1) as u32),
        check_interval: jint(line, "check_interval").map(|v| v.max(1) as u32),
        prior_refresh_rate: jfloat(line, "prior_refresh_rate")
            .map(|v| v as f32)
            .filter(|v| *v >= 0.0),
        prior_refresh_temp: jfloat(line, "prior_refresh_temp")
            .map(|v| v as f32)
            .filter(|v| *v > 0.0),
    }
}

fn parse_search_profile(line: &str) -> SearchProfile {
    match jstr(line, "search_profile").unwrap_or("quartz") {
        "baseline" => SearchProfile::Baseline,
        "baseline_strict" => SearchProfile::BaselineStrict,
        _ => SearchProfile::Quartz,
    }
}

fn search_profile_name(profile: SearchProfile) -> &'static str {
    match profile {
        SearchProfile::Quartz => "quartz",
        SearchProfile::Baseline => "baseline_shared_substrate",
        SearchProfile::BaselineStrict => "baseline_strict",
    }
}

fn apply_search_profile(mut cfg: MctsConfig, profile: SearchProfile) -> MctsConfig {
    match profile {
        SearchProfile::Quartz => {}
        SearchProfile::Baseline => {
            cfg.quartz = None;
            cfg.vl_mode = crate::mcts::parallel::VlMode::Disabled;
        }
        SearchProfile::BaselineStrict => {
            cfg.quartz = None;
            cfg.gvoc = None;
            cfg.vl_mode = crate::mcts::parallel::VlMode::Disabled;
            cfg.root_forced_win = false;
            cfg.exact_terminal_value = false;
            cfg.fpu_reduction = 0.0;
        }
    }
    cfg
}

fn apply_search_overrides(mut cfg: MctsConfig, ov: SearchOverrides) -> MctsConfig {
    cfg = override_penalty(cfg, ov.penalty_mode, ov.hbar_penalty_cap.unwrap_or(0.0));
    if let Some(c_puct) = ov.c_puct {
        cfg.c_puct = c_puct;
    }
    if let Some(ref mut q) = cfg.quartz {
        if let Some(sigma_0) = ov.sigma_0 {
            q.sigma_0 = sigma_0;
        }
        if let Some(min_visits) = ov.min_visits {
            q.min_visits = min_visits;
        }
        if let Some(check_interval) = ov.check_interval {
            q.check_interval = check_interval;
        }
        if let Some(prior_refresh_rate) = ov.prior_refresh_rate {
            q.prior_refresh_rate = prior_refresh_rate;
        }
        if let Some(prior_refresh_temp) = ov.prior_refresh_temp {
            q.prior_refresh_temp = prior_refresh_temp;
        }
    }
    cfg
}

/// Run one self-play game with tree reuse via advance_root().
fn selfplay_one<G: GameState>(
    initial: G,
    config_fn: impl Fn() -> MctsConfig,
    iters: u32,
    temp_thresh: usize,
    num_actions: usize,
) -> String
where
    G::Move: Into<usize> + PartialEq,
{
    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
    let first_player = initial.current_player();
    let mut positions = Vec::new();
    let mut move_n = 0usize;

    // Create engine ONCE, reuse tree across moves
    let config = config_fn();
    let qcfg_template = config.quartz.clone().unwrap_or_default();
    let mut engine = MctsEngine::new(initial, eval.clone(), config);

    while !engine.root_state().is_terminal() && move_n < 500 {
        let legal = engine.root_state().legal_moves();
        if legal.is_empty() {
            break;
        }

        // Search from current root (reusing subtree from previous move)
        let mut ctrl = QuartzController::new(iters, qcfg_template.clone());
        engine.run_quartz(&mut ctrl);

        // Visit distribution
        let guard = engine.root.edges.read().unwrap();
        let mut visits = vec![0u32; num_actions];
        let mut total = 0u32;
        for e in guard.iter() {
            let idx: usize = e.mv.into();
            let n = e.n.load(Ordering::Relaxed);
            if idx < num_actions {
                visits[idx] = n;
                total += n;
            }
        }
        drop(guard);

        let policy: Vec<f32> = visits
            .iter()
            .map(|&n| {
                if total > 0 {
                    n as f32 / total as f32
                } else {
                    0.0
                }
            })
            .collect();
        let planes = engine.root_state().encode_planes();

        let pol_str: Vec<String> = policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-6)
            .map(|(i, p)| format!("\"{}:{:.4}\"", i, p))
            .collect();
        let board_str: String = planes
            .iter()
            .map(|v| if *v > 0.5 { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join("");

        // Controller telemetry per position (Doc 23 patch)
        let stats = ctrl.last_stats();
        let pf = f_or(stats.p_flip, 0.0);
        let sq = f_or(stats.sigma_q, 0.0);
        let hb = f_or(stats.hbar_eff, 0.0);

        positions.push(format!(
            "{{\"pl\":{},\"bd\":\"{}\",\"pol\":[{}],\"pf\":{:.4},\"sq\":{:.4},\"hb\":{:.4}}}",
            engine.root_state().current_player(),
            board_str,
            pol_str.join(","),
            pf,
            sq,
            hb
        ));

        // Move selection
        let chosen = if move_n < temp_thresh {
            let seed = engine
                .root_state()
                .hash()
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let r = (seed >> 33) as f32 / (1u64 << 31) as f32;
            let mut cum = 0.0f32;
            let mut sel = legal[0];
            for &mv in &legal {
                let idx: usize = mv.into();
                if idx < num_actions {
                    cum += policy[idx];
                    if cum >= r {
                        sel = mv;
                        break;
                    }
                }
            }
            sel
        } else {
            let mut best = legal[0];
            let mut bn = 0u32;
            for &mv in &legal {
                let i: usize = mv.into();
                if i < num_actions && visits[i] > bn {
                    bn = visits[i];
                    best = mv;
                }
            }
            best
        };

        // Tree reuse: advance root to chosen child
        if !engine.advance_root(chosen) {
            // Fallback: if advance fails, rebuild engine from current state
            let new_state = engine.root_state().apply_move(chosen);
            let cfg = config_fn();
            engine = MctsEngine::new(new_state, eval.clone(), cfg);
        }
        move_n += 1;
    }

    let outcome = if engine.root_state().is_terminal() {
        let raw = engine.root_state().outcome();
        if engine.root_state().current_player() == first_player {
            raw
        } else {
            -raw
        }
    } else {
        0.0
    };

    format!(
        "{{\"outcome\":{:.4},\"n_moves\":{},\"positions\":[{}]}}",
        outcome,
        move_n,
        positions.join(",")
    )
}

fn parse_penalty_mode(s: &str) -> PenaltyMode {
    match s {
        "None" => PenaltyMode::None,
        "Legacy" => PenaltyMode::Legacy,
        "EffectiveV2" => PenaltyMode::EffectiveV2,
        "SelfAdaptive" => PenaltyMode::SelfAdaptive,
        "GatedRefresh" => PenaltyMode::GatedRefresh,
        "PFlipMixture" => PenaltyMode::PFlipMixture,
        _ => PenaltyMode::GatedRefresh, // default
    }
}

fn override_penalty(mut cfg: MctsConfig, mode: PenaltyMode, cap: f32) -> MctsConfig {
    if let Some(ref mut q) = cfg.quartz {
        q.penalty_mode = mode;
        if cap > 0.0 {
            q.hbar_penalty_cap = cap;
        }
    }
    cfg
}

fn parse_gomoku15_variant(game: &str) -> Option<GomokuVariant> {
    match game {
        "gomoku15" | "gomoku15_free" | "freestyle" => Some(GomokuVariant::Freestyle),
        "gomoku15_std" | "standard" => Some(GomokuVariant::Standard),
        "gomoku15_omok" | "omok" => Some(GomokuVariant::Omok),
        "gomoku15_renju" | "renju" => Some(GomokuVariant::Renju),
        "gomoku15_caro" | "caro" => Some(GomokuVariant::Caro),
        _ => None,
    }
}

fn parse_go_game(game: &str) -> Option<(usize, GoRuleset)> {
    match game {
        "go9" | "go9_cn" => Some((9, GoRuleset::Chinese)),
        "go9_jp" => Some((9, GoRuleset::Japanese)),
        "go9_kr" => Some((9, GoRuleset::Korean)),
        "go13" | "go13_cn" => Some((13, GoRuleset::Chinese)),
        "go13_jp" => Some((13, GoRuleset::Japanese)),
        "go13_kr" => Some((13, GoRuleset::Korean)),
        "go19" | "go19_cn" => Some((19, GoRuleset::Chinese)),
        "go19_jp" => Some((19, GoRuleset::Japanese)),
        "go19_kr" => Some((19, GoRuleset::Korean)),
        _ => None,
    }
}

fn is_chess_game_name(game: &str) -> bool {
    matches!(game, "chess" | "chess960")
}

fn parse_chess960_index(line: &str) -> Option<u16> {
    jint(line, "chess960_index")
        .map(|v| v.clamp(0, 959) as u16)
}

fn chess_state_from_request(line: &str, default_960: bool) -> Chess {
    let fallback = || {
        if default_960 {
            if let Some(idx) = parse_chess960_index(line) {
                Chess::from_960(idx)
            } else if jbool(line, "chess960_random_start").unwrap_or(false) {
                Chess::from_960(rand::random::<u16>() % 960)
            } else {
                Chess::from_960(518)
            }
        } else if let Some(idx) = parse_chess960_index(line) {
            Chess::from_960(idx)
        } else {
            Chess::standard()
        }
    };
    if let Some(fen) = jstr(line, "fen") {
        Chess::from_fen(fen).unwrap_or_else(|_| fallback())
    } else {
        fallback()
    }
}

fn parse_go_ruleset(line: &str, fallback: GoRuleset) -> GoRuleset {
    match jstr(line, "go_ruleset").unwrap_or("") {
        "japanese" | "jp" => GoRuleset::Japanese,
        "korean" | "kr" => GoRuleset::Korean,
        "chinese" | "cn" => GoRuleset::Chinese,
        _ => fallback,
    }
}

fn parse_go_scoring(line: &str, fallback: GoScoring) -> GoScoring {
    match jstr(line, "go_scoring").unwrap_or("") {
        "territory" => GoScoring::Territory,
        "area" => GoScoring::Area,
        _ => fallback,
    }
}

fn parse_go_komi(line: &str, fallback: f32) -> f32 {
    jfloat(line, "go_komi").map(|v| v as f32).unwrap_or(fallback)
}

fn parse_go_allow_suicide(line: &str, fallback: bool) -> bool {
    if line.contains("\"go_allow_suicide\":true") {
        true
    } else if line.contains("\"go_allow_suicide\":false") {
        false
    } else {
        fallback
    }
}

fn handle_selfplay(line: &str) -> String {
    let game = jstr(line, "game").unwrap_or("gomoku15");
    let iters = jint(line, "iters").unwrap_or(200) as u32;
    let n = jint(line, "n_games").unwrap_or(1) as usize;
    let tt = jint(line, "temp_threshold").unwrap_or(15) as usize;
    let overrides = parse_search_overrides(line);

    let mut games = Vec::new();
    for _ in 0..n {
        let g = match game {
            "gomoku7" => {
                let cfg = MctsConfig::evaluation(overrides.c_puct.unwrap_or(2.0)).with_quartz(
                    QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    },
                );
                let cfg = apply_search_overrides(cfg, overrides);
                selfplay_one(
                    Gomoku::new_with_win(7, 4),
                    move || cfg.clone(),
                    iters,
                    tt,
                    49,
                )
            }
            _ if parse_gomoku15_variant(game).is_some() => {
                let variant = parse_gomoku15_variant(game).unwrap();
                let cfg = apply_search_overrides(gomoku15_quartz(variant), overrides);
                selfplay_one(Gomoku15::new(variant), move || cfg.clone(), iters, tt, 225)
            }
            _ if parse_go_game(game).is_some() => {
                let (size, default_ruleset) = parse_go_game(game).unwrap();
                let ruleset = parse_go_ruleset(line, default_ruleset);
                let scoring = parse_go_scoring(line, ruleset.scoring());
                let komi = parse_go_komi(line, if size == 19 { 7.5 } else { 7.5 });
                let allow_suicide = parse_go_allow_suicide(line, false);
                let cfg = apply_search_overrides(go_quartz(size), overrides);
                selfplay_one(
                    Go::new_with_options(size, komi, ruleset, scoring, allow_suicide),
                    move || cfg.clone(),
                    iters,
                    tt,
                    size * size + 1,
                )
            }
            "tictactoe" => {
                let cfg = apply_search_overrides(
                    MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                    overrides,
                );
                selfplay_one(TicTacToe::initial(), move || cfg.clone(), iters, tt, 9)
            }
            game if is_chess_game_name(game) => {
                let cfg = apply_search_overrides(chess_quartz(), overrides);
                selfplay_one(
                    chess_state_from_request(line, game == "chess960"),
                    move || cfg.clone(),
                    iters,
                    tt,
                    CHESS_POLICY_ACTIONS,
                )
            }
            _ => {
                let cfg =
                    apply_search_overrides(gomoku15_quartz(GomokuVariant::Freestyle), overrides);
                selfplay_one(Gomoku15::freestyle(), move || cfg.clone(), iters, tt, 225)
            }
        };
        games.push(g);
    }
    format!("[{}]", games.join(","))
}

fn search_gomoku15(line: &str, variant: GomokuVariant, iters: u32) -> String {
    let board_raw = jarr(line, "board");
    let player = jint(line, "player").unwrap_or(1) as i8;
    let state = if board_raw.len() == 225 {
        Gomoku15::from_board(
            &board_raw.iter().map(|&v| v as i8).collect::<Vec<_>>(),
            player,
            variant,
        )
    } else {
        Gomoku15::new(variant)
    };
    let config = gomoku15_quartz(variant);
    let qcfg = config.quartz.clone().unwrap();
    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
    let engine = MctsEngine::new(state, eval, config);
    let mut ctrl = QuartzController::new(iters, qcfg);
    engine.run_quartz(&mut ctrl);
    let best = engine.best_move().unwrap_or(0);
    let s = ctrl.last_stats();
    let it = engine.root.n_total.load(Ordering::Relaxed);
    format!(
        "{{\"move\":{},\"move_str\":\"({},{})\",\"hbar_eff\":{:.4},\"p_flip\":{:.4},\"iters\":{}}}",
        best,
        best as usize / 15,
        best as usize % 15,
        f_or(s.hbar_eff, 0.0),
        f_or(s.p_flip, 0.0),
        it
    )
}

fn search_chess(line: &str, default_960: bool, iters: u32) -> String {
    let state = chess_state_from_request(line, default_960);
    let config = chess_quartz();
    let qcfg = config.quartz.clone().unwrap();
    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
    let engine = MctsEngine::new(state, eval, config);
    let mut ctrl = QuartzController::new(iters, qcfg);
    engine.run_quartz(&mut ctrl);
    let best = engine
        .best_move()
        .unwrap_or_else(|| crate::games::chess::ChessMove::new(0, 0, 0));
    let s = ctrl.last_stats();
    let it = engine.root.n_total.load(Ordering::Relaxed);
    format!(
        "{{\"move\":{},\"move_str\":\"{}\",\"hbar_eff\":{:.4},\"p_flip\":{:.4},\"iters\":{}}}",
        best.0,
        best.to_uci(),
        f_or(s.hbar_eff, 0.0),
        f_or(s.p_flip, 0.0),
        it
    )
}

fn search_go(line: &str, size: usize, default_ruleset: GoRuleset, iters: u32) -> String {
    let board_raw = jarr(line, "board");
    let player = jint(line, "player").unwrap_or(1) as u8;
    let board: Vec<u8> = board_raw.iter().map(|&v| v as u8).collect();
    let ruleset = parse_go_ruleset(line, default_ruleset);
    let scoring = parse_go_scoring(line, ruleset.scoring());
    let komi = parse_go_komi(line, 7.5);
    let allow_suicide = parse_go_allow_suicide(line, false);
    let passes = jint(line, "passes").unwrap_or(0).clamp(0, 2) as u8;
    let ko_point_raw = jint(line, "ko_point").unwrap_or(-1);
    let ko_point = if ko_point_raw >= 0 { Some(ko_point_raw as u16) } else { None };
    let black_caps = jint(line, "black_caps").unwrap_or(0).max(0) as u16;
    let white_caps = jint(line, "white_caps").unwrap_or(0).max(0) as u16;
    let state = if board.is_empty() {
        Go::new_with_options(size, komi, ruleset, scoring, allow_suicide)
    } else {
        Go::from_board_with_options(
            size, komi, &board, player, ruleset, scoring, allow_suicide,
            passes, ko_point, black_caps, white_caps)
    };
    let config = go_quartz(size);
    let qcfg = config.quartz.clone().unwrap();
    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
    let engine = MctsEngine::new(state.clone(), eval, config);
    let mut ctrl = QuartzController::new(iters, qcfg);
    engine.run_quartz(&mut ctrl);
    let best = engine.best_move().unwrap_or(state.pass_action());
    let s = ctrl.last_stats();
    let it = engine.root.n_total.load(Ordering::Relaxed);
    let n2 = size * size;
    let ms = if best as usize == n2 {
        "pass".into()
    } else {
        format!("({},{})", best as usize / size, best as usize % size)
    };
    format!(
        "{{\"move\":{},\"move_str\":\"{}\",\"hbar_eff\":{:.4},\"p_flip\":{:.4},\"iters\":{}}}",
        best,
        ms,
        f_or(s.hbar_eff, 0.0),
        f_or(s.p_flip, 0.0),
        it
    )
}

fn search_tictactoe(line: &str, iters: u32) -> String {
    use crate::mcts::search::FixedIterations;

    let board_raw = jarr(line, "board");
    let player = jint(line, "player").unwrap_or(1) as i8;
    let state = if board_raw.len() == 9 {
        TicTacToe::from_board(
            &board_raw.iter().map(|&v| v as i8).collect::<Vec<_>>(),
            player,
        )
    } else {
        TicTacToe::initial()
    };
    let config = MctsConfig::evaluation(1.4);
    let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
    let engine = MctsEngine::new(state, eval, config);
    engine.run(&mut FixedIterations::new(iters));
    let best = engine.best_move().unwrap_or(0);
    format!(
        "{{\"move\":{},\"move_str\":\"({},{})\"}}",
        best,
        best / 3,
        best % 3
    )
}

pub fn serve() {
    eprintln!("MCTS server ready (selfplay + move + search_nn protocols)");
    // NOTE: We do NOT hold stdin/stdout locks across the loop.
    // search_nn needs direct stdio access for bidirectional eval protocol.
    loop {
        let mut line = String::new();
        {
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let cmd = jstr(&line, "cmd").unwrap_or("move");
        if cmd == "quit" {
            break;
        }

        if cmd == "search_nn" {
            let resp = handle_search_nn(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }
        if cmd == "search_nn_multi" {
            let resp = handle_search_nn_multi(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }
        if cmd == "search_nn_multi_session_open" {
            let resp = handle_search_nn_multi_session_open(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }
        if cmd == "search_nn_multi_session_step" {
            let resp = handle_search_nn_multi_session_step(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }
        if cmd == "search_nn_multi_session_close" {
            let resp = handle_search_nn_multi_session_close(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }
        if cmd == "eval_nn_run" {
            let resp = handle_eval_nn_run(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }
        if cmd == "selfplay_nn_run" {
            let resp = handle_selfplay_nn_run(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            continue;
        }

        let resp = match cmd {
            "selfplay" => handle_selfplay(&line),
            "chess_state" => handle_chess_state(&line, jstr(&line, "game").unwrap_or("chess960") == "chess960"),
            "chess_apply" => handle_chess_apply(&line, jstr(&line, "game").unwrap_or("chess960") == "chess960"),
            _ => {
                let game = jstr(&line, "game").unwrap_or("gomoku15");
                let iters = jint(&line, "iters").unwrap_or(200) as u32;
                match game {
                    _ if parse_gomoku15_variant(game).is_some() => {
                        search_gomoku15(&line, parse_gomoku15_variant(game).unwrap(), iters)
                    }
            game if is_chess_game_name(game) => search_chess(&line, game == "chess960", iters),
                    _ if parse_go_game(game).is_some() => {
                        let (size, default_ruleset) = parse_go_game(game).unwrap();
                        search_go(&line, size, parse_go_ruleset(&line, default_ruleset), iters)
                    }
                    "tictactoe" => search_tictactoe(&line, iters),
                    _ => format!("{{\"error\":\"unknown game: {}\"}}", game),
                }
            }
        };
        {
            let mut out = io::stdout().lock();
            let _ = writeln!(out, "{}", resp);
            let _ = out.flush();
        }
    }
}

#[derive(Clone)]
struct EvalRunnerSession<G: GameState> {
    game_id: String,
    state: G,
    black_tag: u32,
    white_tag: u32,
    opening: Vec<usize>,
    seed: Option<u64>,
    ply: usize,
    total_time_ms: f64,
    done: bool,
    error: Option<String>,
}

impl<G: GameState> EvalRunnerSession<G> {
    fn active_model_tag(&self) -> u32 {
        if self.state.current_player() > 0 {
            self.black_tag
        } else {
            self.white_tag
        }
    }
}

fn terminal_black_score<G: GameState>(state: &G) -> Option<f64> {
    if !state.is_terminal() {
        return None;
    }
    let outcome = state.outcome();
    if outcome > 0.0 {
        Some(if state.current_player() > 0 { 1.0 } else { 0.0 })
    } else if outcome < 0.0 {
        Some(if state.current_player() > 0 { 0.0 } else { 1.0 })
    } else {
        Some(0.5)
    }
}

fn build_eval_record_json<G: GameState>(sess: &EvalRunnerSession<G>) -> serde_json::Value {
    let (outcome, score_black) = match terminal_black_score(&sess.state) {
        Some(1.0) => ("black_win", Some(1.0)),
        Some(0.0) => ("white_win", Some(0.0)),
        Some(_) => ("draw", Some(0.5)),
        None => ("draw", Some(0.5)),
    };
    serde_json::json!({
        "game_id": sess.game_id,
        "black_tag": sess.black_tag,
        "white_tag": sess.white_tag,
        "outcome": outcome,
        "score_black": score_black,
        "move_count": sess.ply,
        "total_time_ms": sess.total_time_ms,
        "opening": sess.opening,
        "seed": sess.seed,
        "error": sess.error,
        "is_void": sess.error.is_some(),
    })
}

#[derive(Clone)]
struct GomokuSelfplaySession {
    state: Gomoku,
    rng: StdRng,
    moves: usize,
    finished: bool,
    winner: f64,
    board_history: Vec<Vec<i64>>,
    player_history: Vec<i8>,
    policy_history: Vec<Vec<String>>,
    trace_history: Vec<serde_json::Value>,
}

fn choose_selfplay_action(
    rng: &mut StdRng,
    state: &Gomoku,
    policy_entries: &[String],
    move_count: usize,
    temp_threshold: usize,
    fallback_best: usize,
) -> Option<usize> {
    let legal = state.legal_moves();
    if legal.is_empty() {
        return None;
    }
    let mut policy = vec![0.0f64; 49];
    for entry in policy_entries {
        if let Some((idx_raw, val_raw)) = entry.split_once(':') {
            if let (Ok(idx), Ok(val)) = (idx_raw.parse::<usize>(), val_raw.parse::<f64>()) {
                if idx < policy.len() {
                    policy[idx] = val.max(0.0);
                }
            }
        }
    }
    if move_count < temp_threshold {
        let weights = legal.iter().map(|&a| policy[a]).collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        if total > 1e-12 {
            let mut r = rng.gen::<f64>() * total;
            for (&action, &w) in legal.iter().zip(weights.iter()) {
                r -= w;
                if r <= 0.0 {
                    return Some(action);
                }
            }
            return legal.last().copied();
        }
        let idx = rng.gen_range(0..legal.len());
        return Some(legal[idx]);
    }
    if legal.contains(&fallback_best) {
        return Some(fallback_best);
    }
    legal
        .into_iter()
        .max_by(|&a, &b| policy[a].partial_cmp(&policy[b]).unwrap_or(std::cmp::Ordering::Equal))
}

/// NN-backed single-move search using bidirectional eval protocol.
/// Python sends board state, Rust does MCTS with eval callbacks to Python NN.
/// Batch MCTS search: select K leaves → 1 batch eval → K expand+backprop.
/// Throughput: ~K× fewer IPC round-trips, ~K× better GPU utilization.
/// Run search with appropriate parallelism, then extract result JSON.
fn build_result_json<G: GameState>(
    engine: &MctsEngine<G>,
    n_act: usize,
    iterations: u32,
    stop_reason: String,
    p_flip: f32,
    value: f32,
    sigma_q: f32,
    hbar_eff: f32,
) -> String
where
    usize: From<G::Move>,
{
    let best: usize = engine.best_move().map(|m| m.into()).unwrap_or(0);
    let tt = engine.tt.contention_snapshot();
    let guard = engine.root.edges.read().unwrap();
    let mut visits = vec![0u32; n_act];
    for e in guard.iter() {
        let i: usize = e.mv.into();
        if i < n_act {
            visits[i] = e.n.load(Ordering::Relaxed);
        }
    }
    drop(guard);
    let total: u32 = visits.iter().sum();
    let pol: Vec<String> = visits
        .iter()
        .enumerate()
        .filter(|(_, &n)| n > 0)
        .map(|(i, &n)| format!("\"{}:{:.4}\"", i, n as f32 / total.max(1) as f32))
        .collect();
    let par = engine.par_ctrl.telemetry.snapshot();
    format!(
        concat!(
            "{{\"result\":{{",
            "\"best_move\":{},\"policy\":[{}],",
            "\"p_flip\":{:.4},\"value\":{:.4},",
            "\"sigma_q\":{:.4},\"hbar_eff\":{:.4},",
            "\"stop_reason\":\"{}\",\"iterations\":{},",
            "\"dup_rate\":{:.4},\"max_pending\":{},\"avg_vvalue\":{:.4},",
            "\"tt_hit_rate\":{:.6},\"tt_size\":{},",
            "\"tt_get_or_create_calls\":{},\"tt_get_calls\":{},",
            "\"tt_lock_wait_ms\":{:.6},\"tt_max_lock_wait_ms\":{:.6}",
            "}}}}"
        ),
        best,
        pol.join(","),
        f_or(p_flip, 0.0),
        f_or(value, 0.0),
        f_or(sigma_q, 0.0),
        f_or(hbar_eff, 0.0),
        stop_reason,
        iterations.max(total),
        par.dup_rate,
        par.max_pending,
        par.avg_vvalue,
        engine.tt.hit_rate(),
        engine.tt.size(),
        tt.get_or_create_calls,
        tt.get_calls,
        tt.lock_wait_nanos as f64 / 1_000_000.0,
        tt.max_lock_wait_nanos as f64 / 1_000_000.0
    )
}

fn run_and_extract<G: GameState>(
    engine: &MctsEngine<G>,
    n_threads: usize,
    n_act: usize,
    iters: u32,
    qcfg: Option<QuartzConfig>,
    profile: SearchProfile,
) -> String
where
    usize: From<G::Move>,
{
    let phase_before = engine_phase_snapshot();
    let edge_before = edge_lock_contention_snapshot();
    match profile {
        SearchProfile::Quartz => {
            let mut ctrl = QuartzController::new(iters, qcfg.unwrap_or_default());
            if n_threads > 1 {
                engine.run_par_quartz(&mut ctrl, n_threads);
            } else {
                engine.run_quartz(&mut ctrl);
            }
            let s = ctrl.last_stats();
            let out = build_result_json(
                engine,
                n_act,
                engine.root.n_total.load(Ordering::Relaxed),
                format!("{:?}", ctrl.last_stop_reason()),
                s.p_flip,
                s.mean_q,
                s.sigma_q,
                s.hbar_eff,
            );
            let tt = engine.tt.contention_snapshot();
            let phase_after = engine_phase_snapshot();
            let edge_after = edge_lock_contention_snapshot();
            rust_server_trace(
                "search_result_stats",
                serde_json::json!({
                    "profile": search_profile_name(SearchProfile::Quartz),
                    "n_threads": n_threads,
                    "iters": iters,
                    "tt_hit_rate": engine.tt.hit_rate(),
                    "tt_size": engine.tt.size(),
                    "tt_get_or_create_calls": tt.get_or_create_calls,
                    "tt_get_calls": tt.get_calls,
                    "tt_lock_wait_ms": tt.lock_wait_nanos as f64 / 1_000_000.0,
                    "tt_max_lock_wait_ms": tt.max_lock_wait_nanos as f64 / 1_000_000.0,
                    "iterate_calls": phase_after.iterate_calls.saturating_sub(phase_before.iterate_calls),
                    "select_time_ms": (phase_after.select_time_nanos.saturating_sub(phase_before.select_time_nanos)) as f64 / 1_000_000.0,
                    "expand_eval_time_ms": (phase_after.expand_eval_time_nanos.saturating_sub(phase_before.expand_eval_time_nanos)) as f64 / 1_000_000.0,
                    "backprop_time_ms": (phase_after.backprop_time_nanos.saturating_sub(phase_before.backprop_time_nanos)) as f64 / 1_000_000.0,
                    "edges_lock_calls": edge_after.calls.saturating_sub(edge_before.calls),
                    "edges_lock_wait_ms": (edge_after.wait_nanos.saturating_sub(edge_before.wait_nanos)) as f64 / 1_000_000.0,
                    "edges_lock_max_wait_ms": edge_after.max_wait_nanos as f64 / 1_000_000.0,
                }),
            );
            out
        }
        SearchProfile::Baseline | SearchProfile::BaselineStrict => {
            let ctrl = FixedIterations::new(iters);
            let stats = if n_threads > 1 {
                engine.run_par(&ctrl, n_threads)
            } else {
                engine.run(&mut FixedIterations::new(iters))
            };
            let out = build_result_json(
                engine,
                n_act,
                stats.iterations,
                format!("{:?}", stats.stop_reason),
                0.0,
                0.0,
                0.0,
                0.0,
            );
            let tt = engine.tt.contention_snapshot();
            let phase_after = engine_phase_snapshot();
            let edge_after = edge_lock_contention_snapshot();
            rust_server_trace(
                "search_result_stats",
                serde_json::json!({
                    "profile": search_profile_name(profile),
                    "n_threads": n_threads,
                    "iters": iters,
                    "tt_hit_rate": engine.tt.hit_rate(),
                    "tt_size": engine.tt.size(),
                    "tt_get_or_create_calls": tt.get_or_create_calls,
                    "tt_get_calls": tt.get_calls,
                    "tt_lock_wait_ms": tt.lock_wait_nanos as f64 / 1_000_000.0,
                    "tt_max_lock_wait_ms": tt.max_lock_wait_nanos as f64 / 1_000_000.0,
                    "iterate_calls": phase_after.iterate_calls.saturating_sub(phase_before.iterate_calls),
                    "select_time_ms": (phase_after.select_time_nanos.saturating_sub(phase_before.select_time_nanos)) as f64 / 1_000_000.0,
                    "expand_eval_time_ms": (phase_after.expand_eval_time_nanos.saturating_sub(phase_before.expand_eval_time_nanos)) as f64 / 1_000_000.0,
                    "backprop_time_ms": (phase_after.backprop_time_nanos.saturating_sub(phase_before.backprop_time_nanos)) as f64 / 1_000_000.0,
                    "edges_lock_calls": edge_after.calls.saturating_sub(edge_before.calls),
                    "edges_lock_wait_ms": (edge_after.wait_nanos.saturating_sub(edge_before.wait_nanos)) as f64 / 1_000_000.0,
                    "edges_lock_max_wait_ms": edge_after.max_wait_nanos as f64 / 1_000_000.0,
                }),
            );
            out
        }
    }
}

fn make_eval<G: GameState>(
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
    n_actions: usize,
    force_batch: bool,
) -> Arc<dyn crate::game::Evaluator<G>>
where
    usize: From<G::Move>,
{
    use crate::mcts::eval::{BatchConfig, BatchStdioEval, StdioCallbackEval};
    if force_batch || n_threads > 1 {
        let cfg = BatchConfig {
            max_batch_size: batch_size.max(n_threads),
            timeout_us: batch_timeout_us,
        };
        Arc::new(BatchStdioEval::<<G as GameState>::Move>::new(n_actions, cfg))
            as Arc<dyn crate::game::Evaluator<G>>
    } else {
        Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>
    }
}

fn make_eval_pair<G: GameState>(
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
    n_actions: usize,
    force_batch: bool,
    dual_model: bool,
) -> (
    Arc<dyn crate::game::Evaluator<G>>,
    Option<Arc<dyn crate::game::Evaluator<G>>>,
)
where
    usize: From<G::Move>,
{
    use crate::mcts::eval::{BatchConfig, BatchStdioEval, StdioCallbackEval};
    if force_batch || n_threads > 1 {
        let cfg = BatchConfig {
            max_batch_size: batch_size.max(n_threads),
            timeout_us: batch_timeout_us,
        };
        if dual_model {
            let (eval_a, eval_b) =
                BatchStdioEval::<<G as GameState>::Move>::new_shared_pair(n_actions, cfg, 0, 1);
            (
                Arc::new(eval_a) as Arc<dyn crate::game::Evaluator<G>>,
                Some(Arc::new(eval_b) as Arc<dyn crate::game::Evaluator<G>>),
            )
        } else {
            (
                Arc::new(BatchStdioEval::<<G as GameState>::Move>::new(n_actions, cfg))
                    as Arc<dyn crate::game::Evaluator<G>>,
                None,
            )
        }
    } else if dual_model {
        (
            Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>,
            Some(Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>),
        )
    } else {
        (
            Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>,
            None,
        )
    }
}

fn bounded_host_workers(job_count: usize, n_threads: usize) -> usize {
    if job_count <= 1 {
        return job_count;
    }
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let per_search = n_threads.max(1);
    let cap = (avail / per_search).max(1);
    job_count.min(cap)
}

fn cap_search_threads(requested: usize) -> usize {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    requested.max(1).min(avail.max(1))
}

fn available_host_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

fn async_multi_search_enabled(job_count: usize) -> bool {
    if job_count <= 1 {
        return false;
    }
    match std::env::var("QUARTZ_ASYNC_MULTI_SEARCH") {
        Ok(raw) => {
            let v = raw.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "" | "0" | "false" | "no" | "off")
        }
        Err(_) => true,
    }
}

fn multi_job_execution_plan(job_count: usize, requested_threads: usize) -> (usize, usize, bool) {
    let avail = available_host_threads();
    if async_multi_search_enabled(job_count) {
        let engine_threads = requested_threads.max(1).min(avail);
        let worker_count = job_count.min((avail / engine_threads.max(1)).max(1));
        (engine_threads.max(1), worker_count.max(1), true)
    } else {
        let engine_threads = requested_threads.max(1).min(avail);
        let worker_count = bounded_host_workers(job_count, engine_threads);
        // Hard cap: total threads (workers × engine_threads) must not exceed available cores
        let total = worker_count * engine_threads;
        let (final_workers, final_threads) = if total > avail {
            // Reduce workers first, keeping engine_threads as requested
            let capped_workers = (avail / engine_threads).max(1);
            (capped_workers.min(job_count), engine_threads)
        } else {
            (worker_count, engine_threads)
        };
        (final_threads, final_workers, false)
    }
}

struct AsyncBatchPending<G: GameState> {
    selection: crate::mcts::AsyncPendingIteration<G>,
    ticket: AsyncEvalTicket<G::Move>,
}

struct AsyncBatchJob<G: GameState> {
    slot_idx: usize,
    model_tag: u32,
    engine: MctsEngine<G>,
    launched: u32,
    completed: u32,
    pending: Vec<AsyncBatchPending<G>>,
}

fn build_async_result_value<G: GameState>(
    engine: &MctsEngine<G>,
    completed: u32,
    n_actions: usize,
    search_profile: SearchProfile,
) -> serde_json::Value
where
    usize: From<G::Move>,
{
    let (p_flip, value, sigma_q, hbar_eff) = match search_profile {
        SearchProfile::Quartz => match engine.current_quartz_stats() {
            Some(stats) => (stats.p_flip, stats.mean_q, stats.sigma_q, stats.hbar_eff),
            None => (0.0, 0.0, 0.0, 0.0),
        },
        SearchProfile::Baseline | SearchProfile::BaselineStrict => (0.0, 0.0, 0.0, 0.0),
    };
    parse_result_value(build_result_json(
        engine,
        n_actions,
        completed,
        "BudgetExhausted".to_string(),
        p_flip,
        value,
        sigma_q,
        hbar_eff,
    ))
}

fn run_multi_async_batch_tags<G: GameState>(
    tagged_states: &[(usize, G, u32)],
    eval_a: BatchStdioEval<G::Move>,
    eval_b: Option<BatchStdioEval<G::Move>>,
    base_cfg: &MctsConfig,
    _qcfg: Option<QuartzConfig>,
    iters: u32,
    n_threads: usize,
    n_actions: usize,
    search_profile: SearchProfile,
) -> Vec<serde_json::Value>
where
    usize: From<G::Move>,
{
    if tagged_states.is_empty() {
        return vec![];
    }
    let max_index = tagged_states.iter().map(|(idx, _, _)| *idx).max().unwrap_or(0);
    let max_inflight_per_job = n_threads.max(1);
    let mut jobs = tagged_states
        .iter()
        .cloned()
        .map(|(slot_idx, state, model_tag)| AsyncBatchJob {
            slot_idx,
            model_tag,
            engine: MctsEngine::new(
                state,
                Arc::new(if model_tag == 1 {
                    eval_b.clone().unwrap_or_else(|| eval_a.clone())
                } else {
                    eval_a.clone()
                }),
                base_cfg.clone(),
            ),
            launched: 0,
            completed: 0,
            pending: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut results = vec![serde_json::Value::Null; max_index + 1];
    let mut idle_spins = 0u64;
    let mut immediate_terminal = 0u64;
    let mut immediate_tt_cap = 0u64;

    rust_server_trace(
        "run_multi_async_batch_start",
        serde_json::json!({
            "jobs": jobs.len(),
            "iters": iters,
            "max_inflight_per_job": max_inflight_per_job,
            "search_profile": search_profile_name(search_profile),
        }),
    );

    while jobs
        .iter()
        .any(|job| job.completed < iters || !job.pending.is_empty())
    {
        let mut made_progress = false;

        // Load shedding: skip launching new work when aggregate pending is high
        let total_pending: usize = jobs.iter().map(|j| j.pending.len()).sum();
        let aggregate_cap = max_inflight_per_job * jobs.len();
        let load_shed = aggregate_cap > 0 && total_pending >= aggregate_cap * 3 / 4;

        for job in jobs.iter_mut() {
            if load_shed && !job.pending.is_empty() {
                continue; // drain existing pending first
            }
            while job.launched < iters && job.pending.len() < max_inflight_per_job {
                match job.engine.prepare_iteration_async() {
                    PreparedIteration::Immediate { path, value, reason } => {
                        job.engine.apply_iteration_value_async(path, value);
                        job.launched += 1;
                        job.completed += 1;
                        job.engine.refresh_async_runtime(job.completed);
                        made_progress = true;
                        match reason {
                            crate::mcts::ImmediateReason::TtCapHit => {
                                immediate_tt_cap += 1;
                            }
                            crate::mcts::ImmediateReason::TerminalNode => {
                                immediate_terminal += 1;
                            }
                        }
                    }
                    PreparedIteration::Pending(selection) => {
                        let ticket = if job.model_tag == 1 {
                            eval_b
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| eval_a.clone())
                                .submit(&selection.leaf_state)
                        } else {
                            eval_a.clone().submit(&selection.leaf_state)
                        };
                        job.pending.push(AsyncBatchPending { selection, ticket });
                        job.launched += 1;
                        made_progress = true;
                    }
                }
            }
        }

        for job in jobs.iter_mut() {
            let mut idx = 0usize;
            while idx < job.pending.len() {
                if let Some(result) = job.pending[idx].ticket.try_take() {
                    let pending = job.pending.swap_remove(idx);
                    job.engine.complete_iteration_async(pending.selection, result);
                    job.completed += 1;
                    job.engine.refresh_async_runtime(job.completed);
                    made_progress = true;
                } else {
                    idx += 1;
                }
            }
        }

        if !made_progress {
            idle_spins += 1;
            // Adaptive backoff: spin briefly, then yield, then sleep with increasing delay
            let sleep_us = if idle_spins <= 4 {
                0 // pure spin
            } else if idle_spins <= 16 {
                std::thread::yield_now();
                0
            } else if idle_spins <= 128 {
                50 // 50µs
            } else {
                200 // 200µs for extended idle
            };
            if idle_spins % 2048 == 0 {
                rust_server_trace(
                    "run_multi_async_batch_idle",
                    serde_json::json!({
                        "jobs": jobs.len(),
                        "pending_eval": jobs.iter().map(|j| j.pending.len()).sum::<usize>(),
                        "completed": jobs.iter().map(|j| j.completed as usize).sum::<usize>(),
                    }),
                );
            }
            if sleep_us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(sleep_us));
            }
        } else {
            idle_spins = 0;
        }
    }

    for job in jobs.iter() {
        if job.slot_idx < results.len() {
            results[job.slot_idx] = build_async_result_value(
                &job.engine,
                job.completed,
                n_actions,
                search_profile,
            );
        }
    }

    rust_server_trace(
        "run_multi_async_batch_done",
        serde_json::json!({
            "results_len": results.len(),
            "null_results": results.iter().filter(|v| v.is_null()).count(),
            "immediate_terminal": immediate_terminal,
            "immediate_tt_cap": immediate_tt_cap,
            "idle_spins": idle_spins,
        }),
    );

    results
}

fn run_multi_with_eval<G: GameState>(
    states: &[Option<G>],
    eval: Arc<dyn crate::game::Evaluator<G>>,
    base_cfg: &MctsConfig,
    qcfg: Option<QuartzConfig>,
    iters: u32,
    n_threads: usize,
    n_actions: usize,
    search_profile: SearchProfile,
) -> Vec<serde_json::Value>
where
    usize: From<G::Move>,
{
    rust_server_trace(
        "run_multi_with_eval_start",
        serde_json::json!({
            "states_len": states.len(),
            "iters": iters,
            "n_threads": n_threads,
            "n_actions": n_actions,
            "search_profile": search_profile_name(search_profile),
        }),
    );
    let active_states = states
        .iter()
        .enumerate()
        .filter_map(|(idx, state)| state.clone().map(|st| (idx, st)))
        .collect::<Vec<_>>();
    if active_states.is_empty() {
        rust_server_trace(
            "run_multi_with_eval_empty",
            serde_json::json!({ "states_len": states.len() }),
        );
        return vec![serde_json::Value::Null; states.len()];
    }
    let results: Vec<Mutex<serde_json::Value>> = (0..states.len())
        .map(|_| Mutex::new(serde_json::Value::Null))
        .collect();
    let results = Arc::new(results);
    let next_job = AtomicUsize::new(0);
    let (engine_threads, worker_count, async_mode) =
        multi_job_execution_plan(active_states.len(), n_threads);
    rust_server_trace(
        "run_multi_with_eval_workers",
        serde_json::json!({
            "active_states": active_states.len(),
            "worker_count": worker_count,
            "n_threads": n_threads,
            "engine_threads": engine_threads,
            "async_mode": async_mode,
        }),
    );
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let next_job_ref = &next_job;
        let active_states_ref = &active_states;
        for _ in 0..worker_count {
            let eval = eval.clone();
            let cfg_template = base_cfg.clone();
            let qcfg = qcfg.clone();
            let results = results.clone();
            handles.push(scope.spawn(move || {
                rust_server_trace(
                    "run_multi_worker_start",
                    serde_json::json!({
                        "worker_n_threads": engine_threads,
                        "requested_threads": n_threads,
                        "async_mode": async_mode,
                        "search_profile": search_profile_name(search_profile),
                    }),
                );
                loop {
                    let job_ix = next_job_ref.fetch_add(1, Ordering::Relaxed);
                    if job_ix >= active_states_ref.len() {
                        break;
                    }
                    let (idx, state) = active_states_ref[job_ix].clone();
                    rust_server_trace(
                        "run_multi_job_start",
                        serde_json::json!({
                            "job_ix": job_ix,
                            "slot_idx": idx,
                        }),
                    );
                    let engine = MctsEngine::new(state, eval.clone(), cfg_template.clone());
                    let value = parse_result_value(run_and_extract(
                        &engine,
                        engine_threads,
                        n_actions,
                        iters,
                        qcfg.clone(),
                        search_profile,
                    ));
                    rust_server_trace(
                        "run_multi_job_done",
                        serde_json::json!({
                            "job_ix": job_ix,
                            "slot_idx": idx,
                            "is_null": value.is_null(),
                        }),
                    );
                    if idx < results.len() {
                        if let Ok(mut slot) = results[idx].lock() {
                            *slot = value;
                        }
                    }
                }
                rust_server_trace("run_multi_worker_done", serde_json::json!({}));
            }));
        }
        for handle in handles {
            let _ = handle.join();
        }
    });
    let final_results: Vec<serde_json::Value> = match Arc::try_unwrap(results) {
        Ok(vec_of_mutex) => vec_of_mutex
            .into_iter()
            .map(|m| m.into_inner().unwrap_or(serde_json::Value::Null))
            .collect(),
        Err(arc) => arc
            .iter()
            .map(|m| m.lock().map(|g| g.clone()).unwrap_or(serde_json::Value::Null))
            .collect(),
    };
    rust_server_trace(
        "run_multi_with_eval_done",
        serde_json::json!({
            "results_len": final_results.len(),
            "null_results": final_results.iter().filter(|v| v.is_null()).count(),
        }),
    );
    final_results
}

fn run_multi_with_eval_tags<G: GameState>(
    tagged_states: &[(usize, G, u32)],
    eval_a: Arc<dyn crate::game::Evaluator<G>>,
    eval_b: Option<Arc<dyn crate::game::Evaluator<G>>>,
    base_cfg: &MctsConfig,
    qcfg: Option<QuartzConfig>,
    iters: u32,
    n_threads: usize,
    n_actions: usize,
    search_profile: SearchProfile,
) -> Vec<serde_json::Value>
where
    usize: From<G::Move>,
{
    if tagged_states.is_empty() {
        return vec![];
    }
    let max_index = tagged_states
        .iter()
        .map(|(idx, _, _)| *idx)
        .max()
        .unwrap_or(0);
    let results: Vec<Mutex<serde_json::Value>> = (0..=max_index)
        .map(|_| Mutex::new(serde_json::Value::Null))
        .collect();
    let results = Arc::new(results);
    let next_job = AtomicUsize::new(0);
    let (engine_threads, worker_count, _async_mode) =
        multi_job_execution_plan(tagged_states.len(), n_threads);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let next_job_ref = &next_job;
        let tagged_states_ref = tagged_states;
        for _ in 0..worker_count {
            let eval_a = eval_a.clone();
            let eval_b = eval_b.clone();
            let cfg_template = base_cfg.clone();
            let qcfg = qcfg.clone();
            let results = results.clone();
            handles.push(scope.spawn(move || loop {
                let job_ix = next_job_ref.fetch_add(1, Ordering::Relaxed);
                if job_ix >= tagged_states_ref.len() {
                    break;
                }
                let (idx, state, model_tag) = tagged_states_ref[job_ix].clone();
                let eval = if model_tag == 1 {
                    eval_b.clone().unwrap_or_else(|| eval_a.clone())
                } else {
                    eval_a.clone()
                };
                let engine = MctsEngine::new(state, eval, cfg_template.clone());
                let value = parse_result_value(run_and_extract(
                    &engine,
                    engine_threads,
                    n_actions,
                    iters,
                    qcfg.clone(),
                    search_profile,
                ));
                if idx < results.len() {
                    if let Ok(mut slot) = results[idx].lock() {
                        *slot = value;
                    }
                }
            }));
        }
        for handle in handles {
            let _ = handle.join();
        }
    });
    match Arc::try_unwrap(results) {
        Ok(vec_of_mutex) => vec_of_mutex
            .into_iter()
            .map(|m| m.into_inner().unwrap_or(serde_json::Value::Null))
            .collect(),
        Err(arc) => arc
            .iter()
            .map(|m| m.lock().map(|g| g.clone()).unwrap_or(serde_json::Value::Null))
            .collect(),
    }
}

struct SearchSession<G: GameState> {
    states: Vec<Option<G>>,
    eval: Arc<dyn crate::game::Evaluator<G>>,
    cfg: MctsConfig,
    qcfg: Option<QuartzConfig>,
    iters: u32,
    n_threads: usize,
    n_actions: usize,
    search_profile: SearchProfile,
}

impl<G: GameState> SearchSession<G>
where
    usize: From<G::Move>,
{
    fn search(&self) -> Vec<serde_json::Value> {
        rust_server_trace(
            "search_session_search_start",
            serde_json::json!({
                "states_len": self.states.len(),
                "active_states": self.states.iter().filter(|s| s.is_some()).count(),
                "iters": self.iters,
                "n_threads": self.n_threads,
                "n_actions": self.n_actions,
                "search_profile": search_profile_name(self.search_profile),
            }),
        );
        let results = run_multi_with_eval(
            &self.states,
            self.eval.clone(),
            &self.cfg,
            self.qcfg.clone(),
            self.iters,
            self.n_threads,
            self.n_actions,
            self.search_profile,
        );
        rust_server_trace(
            "search_session_search_done",
            serde_json::json!({
                "results_len": results.len(),
                "null_results": results.iter().filter(|v| v.is_null()).count(),
            }),
        );
        results
    }

    fn deactivate(&mut self, slot: usize) {
        if let Some(state) = self.states.get_mut(slot) {
            *state = None;
        }
    }

    fn replace(&mut self, slot: usize, state: G) {
        if let Some(dst) = self.states.get_mut(slot) {
            *dst = Some(state);
        }
    }

    fn apply_action_idx(&mut self, slot: usize, action: usize) -> Result<(), String> {
        let Some(state) = self.states.get_mut(slot).and_then(Option::take) else {
            return Ok(());
        };
        let Some(mv) = state.idx_to_move(action) else {
            self.states[slot] = Some(state);
            return Err(format!("invalid action {} for slot {}", action, slot));
        };
        self.states[slot] = Some(state.apply_move(mv));
        Ok(())
    }
}

struct Gomoku15Session {
    inner: SearchSession<Gomoku15>,
    variant: GomokuVariant,
}

struct GoSession {
    inner: SearchSession<Go>,
    size: usize,
    ruleset: GoRuleset,
    scoring: GoScoring,
    komi: f32,
    allow_suicide: bool,
}

struct ChessSession {
    inner: SearchSession<Chess>,
    default_960: bool,
}

enum SearchSessionAny {
    Gomoku(SearchSession<Gomoku>),
    Gomoku15(Gomoku15Session),
    Go(GoSession),
    Chess(ChessSession),
    TicTacToe(SearchSession<TicTacToe>),
}

impl SearchSessionAny {
    fn search(&self) -> Vec<serde_json::Value> {
        match self {
            SearchSessionAny::Gomoku(inner) => inner.search(),
            SearchSessionAny::Gomoku15(inner) => inner.inner.search(),
            SearchSessionAny::Go(inner) => inner.inner.search(),
            SearchSessionAny::Chess(inner) => inner.inner.search(),
            SearchSessionAny::TicTacToe(inner) => inner.search(),
        }
    }

    fn apply_updates(&mut self, updates: &[serde_json::Value]) -> Result<(), String> {
        match self {
            SearchSessionAny::Gomoku(inner) => apply_updates_gomoku(inner, updates),
            SearchSessionAny::Gomoku15(inner) => apply_updates_gomoku15(inner, updates),
            SearchSessionAny::Go(inner) => apply_updates_go(inner, updates),
            SearchSessionAny::Chess(inner) => apply_updates_chess(inner, updates),
            SearchSessionAny::TicTacToe(inner) => apply_updates_tictactoe(inner, updates),
        }
    }
}

fn parse_gomoku7_job(job: &serde_json::Value) -> Gomoku {
    let board_raw = job
        .get("board")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
    let player_12: u8 = if player == 1 { 1 } else { 2 };
    let board_12: Vec<i64> = if board_raw.len() == 49 {
        board_raw
            .iter()
            .map(|v| match v.as_i64().unwrap_or(0) {
                1 => 1,
                -1 => 2,
                _ => 0,
            })
            .collect()
    } else {
        vec![0i64; 49]
    };
    Gomoku::from_board_12(7, 4, &board_12, player_12)
}

fn parse_gomoku15_job(job: &serde_json::Value, variant: GomokuVariant) -> Gomoku15 {
    let board_raw = job
        .get("board")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
    if board_raw.len() == 225 {
        Gomoku15::from_board(
            &board_raw
                .iter()
                .map(|v| v.as_i64().unwrap_or(0) as i8)
                .collect::<Vec<_>>(),
            player,
            variant,
        )
    } else {
        Gomoku15::new(variant)
    }
}

fn parse_go_job(
    job: &serde_json::Value,
    size: usize,
    ruleset: GoRuleset,
    scoring: GoScoring,
    komi: f32,
    allow_suicide: bool,
) -> Go {
    let board_raw = job
        .get("board")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1);
    let side: u8 = if player == 1 { 1 } else { 2 };
    let passes = job
        .get("passes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        .min(2) as u8;
    let ko_point = job
        .get("ko_point")
        .and_then(|v| v.as_i64())
        .and_then(|v| if v >= 0 { Some(v as u16) } else { None });
    let black_caps = job.get("black_caps").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
    let white_caps = job.get("white_caps").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
    let n2 = size * size;
    if board_raw.len() == n2 {
        let board_12: Vec<u8> = board_raw
            .iter()
            .map(|v| match v.as_i64().unwrap_or(0) {
                1 => 1,
                2 | -1 => 2,
                _ => 0,
            })
            .collect();
        Go::from_board_with_options(
            size,
            komi,
            &board_12,
            side,
            ruleset,
            scoring,
            allow_suicide,
            passes,
            ko_point,
            black_caps,
            white_caps,
        )
    } else {
        Go::new_with_options(size, komi, ruleset, scoring, allow_suicide)
    }
}

fn parse_tictactoe_job(job: &serde_json::Value) -> TicTacToe {
    let board_raw = job
        .get("board")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
    if board_raw.len() == 9 {
        TicTacToe::from_board(
            &board_raw
                .iter()
                .map(|v| v.as_i64().unwrap_or(0) as i8)
                .collect::<Vec<_>>(),
            player,
        )
    } else {
        TicTacToe::initial()
    }
}

fn parse_chess_job(job: &serde_json::Value, default_960: bool) -> Chess {
    let fallback = || {
        if default_960 {
            if let Some(idx) = job
                .get("chess960_index")
                .and_then(|v| v.as_u64())
                .map(|v| v.min(959) as u16)
            {
                Chess::from_960(idx)
            } else if job
                .get("chess960_random_start")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                Chess::from_960(rand::random::<u16>() % 960)
            } else {
                Chess::from_960(518)
            }
        } else if let Some(idx) = job
            .get("chess960_index")
            .and_then(|v| v.as_u64())
            .map(|v| v.min(959) as u16)
        {
            Chess::from_960(idx)
        } else {
            Chess::standard()
        }
    };
    if let Some(fen) = job.get("fen").and_then(|v| v.as_str()) {
        Chess::from_fen(fen).unwrap_or_else(|_| fallback())
    } else {
        fallback()
    }
}

fn apply_updates_gomoku(
    session: &mut SearchSession<Gomoku>,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session.replace(slot, parse_gomoku7_job(replace));
        } else if update.get("deactivate").and_then(|v| v.as_bool()).unwrap_or(false) {
            session.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_gomoku15(
    session: &mut Gomoku15Session,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session
                .inner
                .replace(slot, parse_gomoku15_job(replace, session.variant));
        } else if update.get("deactivate").and_then(|v| v.as_bool()).unwrap_or(false) {
            session.inner.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.inner.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_go(session: &mut GoSession, updates: &[serde_json::Value]) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session.inner.replace(
                slot,
                parse_go_job(
                    replace,
                    session.size,
                    session.ruleset,
                    session.scoring,
                    session.komi,
                    session.allow_suicide,
                ),
            );
        } else if update.get("deactivate").and_then(|v| v.as_bool()).unwrap_or(false) {
            session.inner.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.inner.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_tictactoe(
    session: &mut SearchSession<TicTacToe>,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session.replace(slot, parse_tictactoe_job(replace));
        } else if update.get("deactivate").and_then(|v| v.as_bool()).unwrap_or(false) {
            session.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_chess(session: &mut ChessSession, updates: &[serde_json::Value]) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session
                .inner
                .replace(slot, parse_chess_job(replace, session.default_960));
        } else if update.get("deactivate").and_then(|v| v.as_bool()).unwrap_or(false) {
            session.inner.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.inner.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn search_sessions() -> &'static Mutex<HashMap<u64, SearchSessionAny>> {
    static SESSIONS: OnceLock<Mutex<HashMap<u64, SearchSessionAny>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_search_session_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn default_batch_timeout_us(n_threads: usize, batch_size: usize, job_count: usize) -> u64 {
    let thread_count = n_threads.max(1);
    let jobs = job_count.max(1);
    let mut timeout_us = if thread_count <= 1 {
        if jobs > 1 {
            1800 + 250 * jobs.saturating_sub(1) as u64
        } else {
            1500
        }
    } else {
        (1200 + 250 * std::cmp::max(thread_count, batch_size / 2)) as u64
    };
    if jobs >= 4 {
        timeout_us += 250;
    }
    timeout_us.clamp(500, 6000)
}

fn handle_search_nn(line: &str) -> String {
    use crate::mcts::eval::{BatchConfig, BatchStdioEval, StdioCallbackEval};

    let game = jstr(line, "game").unwrap_or("gomoku15");
    let iters = jint(line, "iters").unwrap_or(200) as u32;
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let n_threads = cap_search_threads(jint(line, "n_threads").unwrap_or(1) as usize);
    let batch_size = (jint(line, "batch_size").unwrap_or(8) as usize).max(1);
    let batch_timeout_us =
        jint(line, "batch_timeout_us").unwrap_or(default_batch_timeout_us(n_threads, batch_size, 1) as i64)
            as u64;

    let n_actions: usize = match game {
        "gomoku7" => 49,
        _ if parse_gomoku15_variant(game).is_some() => 225,
        _ if parse_go_game(game).is_some() => {
            let (size, _) = parse_go_game(game).unwrap();
            size * size + 1
        }
        "tictactoe" => 9,
        game if is_chess_game_name(game) => CHESS_POLICY_ACTIONS,
        _ => 225,
    };

    // Macro to create the right evaluator type (avoids dynamic dispatch boxing issues)
    macro_rules! make_eval {
        ($game_type:ty) => {
            if n_threads > 1 {
                let cfg = BatchConfig {
                    max_batch_size: batch_size.max(n_threads),
                    timeout_us: batch_timeout_us,
                };
                Arc::new(BatchStdioEval::<<$game_type as GameState>::Move>::new(
                    n_actions, cfg,
                )) as Arc<dyn crate::game::Evaluator<$game_type>>
            } else {
                Arc::new(StdioCallbackEval::new(n_actions))
                    as Arc<dyn crate::game::Evaluator<$game_type>>
            }
        };
    }

    match game {
        "gomoku7" => {
            let board_raw = jarr(line, "board");
            let player = jint(line, "player").unwrap_or(1) as i8;
            let player_12: u8 = if player == 1 { 1 } else { 2 };
            let board_12: Vec<i64> = if board_raw.len() == 49 {
                board_raw
                    .iter()
                    .map(|&v| match v {
                        1 => 1,
                        -1 => 2,
                        _ => 0,
                    })
                    .collect()
            } else {
                vec![0i64; 49]
            };
            let state = Gomoku::from_board_12(7, 4, &board_12, player_12);
            let eval = make_eval!(Gomoku);
            let cfg = apply_search_profile(apply_search_overrides(
                MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                }),
                overrides,
            ), search_profile);
            let engine = MctsEngine::new(state, eval, cfg);
            run_and_extract(&engine, n_threads, 49, iters, engine.config.quartz.clone(), search_profile)
        }
        _ if parse_gomoku15_variant(game).is_some() => {
            let board_raw = jarr(line, "board");
            let player = jint(line, "player").unwrap_or(1) as i8;
            let variant = parse_gomoku15_variant(game).unwrap();
            let state = if board_raw.len() == 225 {
                Gomoku15::from_board(
                    &board_raw.iter().map(|&v| v as i8).collect::<Vec<_>>(),
                    player,
                    variant,
                )
            } else {
                Gomoku15::new(variant)
            };
            let eval = make_eval!(Gomoku15);
            let cfg = apply_search_profile(apply_search_overrides(gomoku15_quartz(variant), overrides), search_profile);
            let engine = MctsEngine::new(state, eval, cfg);
            run_and_extract(&engine, n_threads, 225, iters, engine.config.quartz.clone(), search_profile)
        }
        _ if parse_go_game(game).is_some() => {
            let (size, default_ruleset) = parse_go_game(game).unwrap();
            let board_raw = jarr(line, "board");
            let player = jint(line, "player").unwrap_or(1);
            let side: u8 = if player == 1 { 1 } else { 2 };
            let ruleset = parse_go_ruleset(line, default_ruleset);
            let scoring = parse_go_scoring(line, ruleset.scoring());
            let komi = parse_go_komi(line, 7.5);
            let allow_suicide = parse_go_allow_suicide(line, false);
            let passes = jint(line, "passes").unwrap_or(0).clamp(0, 2) as u8;
            let ko_point_raw = jint(line, "ko_point").unwrap_or(-1);
            let ko_point = if ko_point_raw >= 0 { Some(ko_point_raw as u16) } else { None };
            let black_caps = jint(line, "black_caps").unwrap_or(0).max(0) as u16;
            let white_caps = jint(line, "white_caps").unwrap_or(0).max(0) as u16;
            let n2 = size * size;
            let state = if board_raw.len() == n2 {
                let board_12: Vec<u8> = board_raw
                    .iter()
                    .map(|&v| match v {
                        1 => 1,
                        2 | -1 => 2,
                        _ => 0,
                    })
                    .collect();
                Go::from_board_with_options(
                    size, komi, &board_12, side, ruleset, scoring, allow_suicide,
                    passes, ko_point, black_caps, white_caps)
            } else {
                Go::new_with_options(size, komi, ruleset, scoring, allow_suicide)
            };
            let eval = make_eval!(Go);
            let cfg = apply_search_profile(apply_search_overrides(go_quartz(size), overrides), search_profile);
            let engine = MctsEngine::new(state, eval, cfg);
            run_and_extract(&engine, n_threads, n_actions, iters, engine.config.quartz.clone(), search_profile)
        }
        "tictactoe" => {
            let board_raw = jarr(line, "board");
            let player = jint(line, "player").unwrap_or(1) as i8;
            let state = if board_raw.len() == 9 {
                TicTacToe::from_board(
                    &board_raw.iter().map(|&v| v as i8).collect::<Vec<_>>(),
                    player,
                )
            } else {
                TicTacToe::initial()
            };
            let eval = make_eval!(TicTacToe);
            let cfg = apply_search_profile(apply_search_overrides(
                MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                overrides,
            ), search_profile);
            let engine = MctsEngine::new(state, eval, cfg);
            run_and_extract(&engine, n_threads, 9, iters, engine.config.quartz.clone(), search_profile)
        }
        game if is_chess_game_name(game) => {
            let state = chess_state_from_request(line, game == "chess960");
            let eval = make_eval::<Chess>(n_threads, batch_size, batch_timeout_us, n_actions, false);
            let cfg = apply_search_profile(apply_search_overrides(chess_quartz(), overrides), search_profile);
            let engine = MctsEngine::new(state, eval, cfg);
            let (iterations, stop_reason, p_flip, value, sigma_q, hbar_eff) = match search_profile {
                SearchProfile::Quartz => {
                    let mut ctrl = QuartzController::new(iters, engine.config.quartz.clone().unwrap_or_default());
                    if n_threads > 1 {
                        engine.run_par_quartz(&mut ctrl, n_threads);
                    } else {
                        engine.run_quartz(&mut ctrl);
                    }
                    let s = ctrl.last_stats();
                    (
                        engine.root.n_total.load(Ordering::Relaxed),
                        format!("{:?}", ctrl.last_stop_reason()),
                        s.p_flip,
                        s.mean_q,
                        s.sigma_q,
                        s.hbar_eff,
                    )
                }
                SearchProfile::Baseline | SearchProfile::BaselineStrict => {
                    let stats = if n_threads > 1 {
                        engine.run_par(&FixedIterations::new(iters), n_threads)
                    } else {
                        engine.run(&mut FixedIterations::new(iters))
                    };
                    (
                        stats.iterations,
                        format!("{:?}", stats.stop_reason),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                }
            };

            // Chess has custom result extraction (includes result_fen)
            let best: usize = engine.best_move().map(chess_policy_index).unwrap_or(0);
            let guard = engine.root.edges.read().unwrap();
            let mut visits = vec![0u32; CHESS_POLICY_ACTIONS];
            for e in guard.iter() {
                let i = chess_policy_index(e.mv);
                if i < CHESS_POLICY_ACTIONS {
                    visits[i] = visits[i].saturating_add(e.n.load(Ordering::Relaxed));
                }
            }
            drop(guard);
            let total: u32 = visits.iter().sum();
            let pol: Vec<String> = visits
                .iter()
                .enumerate()
                .filter(|(_, &n)| n > 0)
                .map(|(i, &n)| format!("\"{}:{:.4}\"", i, n as f32 / total.max(1) as f32))
                .collect();
            let best_move_uci = engine.best_move().map(|mv| mv.to_uci()).unwrap_or_default();
            let result_fen = if let Some(mv) = engine.best_move() {
                let new_state = engine.root_state().apply_move(mv);
                new_state.to_fen()
            } else {
                engine.root_state().to_fen()
            };

            let par = engine.par_ctrl.telemetry.snapshot();
            format!(
                concat!(
                    "{{\"result\":{{",
                    "\"best_move\":{},\"policy\":[{}],",
                    "\"p_flip\":{:.4},\"value\":{:.4},",
                    "\"sigma_q\":{:.4},\"hbar_eff\":{:.4},",
                    "\"stop_reason\":\"{}\",\"iterations\":{},",
                    "\"dup_rate\":{:.4},\"max_pending\":{},\"avg_vvalue\":{:.4},",
                    "\"best_move_uci\":\"{}\",",
                    "\"result_fen\":\"{}\"",
                    "}}}}"
                ),
                best,
                pol.join(","),
                f_or(p_flip, 0.0),
                f_or(value, 0.0),
                f_or(sigma_q, 0.0),
                f_or(hbar_eff, 0.0),
                stop_reason,
                iterations.max(total),
                par.dup_rate,
                par.max_pending,
                par.avg_vvalue,
                best_move_uci,
                result_fen
            )
        }
        _ => format!("{{\"error\":\"search_nn not yet supported for {}\"}}", game),
    }
}

fn handle_search_nn_multi(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let game = root
        .get("game")
        .and_then(|v| v.as_str())
        .unwrap_or("gomoku15");
    let jobs = root
        .get("jobs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if jobs.is_empty() {
        return "{\"results\":[]}".to_string();
    }

    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let n_threads = cap_search_threads(root
        .get("n_threads")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize);
    let batch_size = root
        .get("batch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .max(1) as usize;
    let batch_timeout_us = root
        .get("batch_timeout_us")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| default_batch_timeout_us(n_threads, batch_size, jobs.len()));

    let n_actions: usize = match game {
        "gomoku7" => 49,
        _ if parse_gomoku15_variant(game).is_some() => 225,
        _ if parse_go_game(game).is_some() => {
            let (size, _) = parse_go_game(game).unwrap();
            size * size + 1
        }
        "tictactoe" => 9,
        game if is_chess_game_name(game) => CHESS_POLICY_ACTIONS,
        _ => 225,
    };

    macro_rules! run_multi_generic {
        ($game_type:ty, $states:expr, $cfg:expr, $n_act:expr) => {{
            let dual_model = $states.iter().any(|(_, _, tag)| *tag != 0);
            let force_batch = $states.len() > 1 || dual_model;
            let (eval_a, eval_b) = make_eval_pair::<$game_type>(
                n_threads,
                batch_size,
                batch_timeout_us,
                n_actions,
                force_batch,
                dual_model,
            );
            let base_cfg = $cfg;
            let qcfg = base_cfg.quartz.clone();
            let results = run_multi_with_eval_tags(
                &$states,
                eval_a,
                eval_b,
                &base_cfg,
                qcfg,
                iters,
                n_threads,
                $n_act,
                search_profile,
            );
            serde_json::json!({ "results": results }).to_string()
        }};
    }

    match game {
        "gomoku7" => {
            let mut states = Vec::with_capacity(jobs.len());
            for (idx, job) in jobs.into_iter().enumerate() {
                let board_raw = job
                    .get("board")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let model_tag = job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
                let player_12: u8 = if player == 1 { 1 } else { 2 };
                let board_12: Vec<i64> = if board_raw.len() == 49 {
                    board_raw
                        .iter()
                        .map(|v| match v.as_i64().unwrap_or(0) {
                            1 => 1,
                            -1 => 2,
                            _ => 0,
                        })
                        .collect()
                } else {
                    vec![0i64; 49]
                };
                states.push((idx, Gomoku::from_board_12(7, 4, &board_12, player_12), model_tag));
            }
            let cfg = apply_search_profile(apply_search_overrides(
                MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                }),
                overrides,
            ), search_profile);
            run_multi_generic!(Gomoku, states, cfg, 49)
        }
        _ if parse_gomoku15_variant(game).is_some() => {
            let variant = parse_gomoku15_variant(game).unwrap();
            let mut states = Vec::with_capacity(jobs.len());
            for (idx, job) in jobs.into_iter().enumerate() {
                let board_raw = job
                    .get("board")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let model_tag = job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
                let state = if board_raw.len() == 225 {
                    Gomoku15::from_board(
                        &board_raw.iter().map(|v| v.as_i64().unwrap_or(0) as i8).collect::<Vec<_>>(),
                        player,
                        variant,
                    )
                } else {
                    Gomoku15::new(variant)
                };
                states.push((idx, state, model_tag));
            }
            let cfg = apply_search_profile(apply_search_overrides(gomoku15_quartz(variant), overrides), search_profile);
            run_multi_generic!(Gomoku15, states, cfg, 225)
        }
        _ if parse_go_game(game).is_some() => {
            let (size, default_ruleset) = parse_go_game(game).unwrap();
            let ruleset = parse_go_ruleset(line, default_ruleset);
            let scoring = parse_go_scoring(line, ruleset.scoring());
            let komi = parse_go_komi(line, 7.5);
            let allow_suicide = parse_go_allow_suicide(line, false);
            let mut states = Vec::with_capacity(jobs.len());
            for (idx, job) in jobs.into_iter().enumerate() {
                let board_raw = job
                    .get("board")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let model_tag = job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1);
                let side: u8 = if player == 1 { 1 } else { 2 };
                let passes = job.get("passes").and_then(|v| v.as_u64()).unwrap_or(0).min(2) as u8;
                let ko_point = job.get("ko_point").and_then(|v| v.as_i64()).and_then(|v| if v >= 0 { Some(v as u16) } else { None });
                let black_caps = job.get("black_caps").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                let white_caps = job.get("white_caps").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                let n2 = size * size;
                let state = if board_raw.len() == n2 {
                    let board_12: Vec<u8> = board_raw
                        .iter()
                        .map(|v| match v.as_i64().unwrap_or(0) {
                            1 => 1,
                            2 | -1 => 2,
                            _ => 0,
                        })
                        .collect();
                    Go::from_board_with_options(
                        size, komi, &board_12, side, ruleset, scoring, allow_suicide,
                        passes, ko_point, black_caps, white_caps
                    )
                } else {
                    Go::new_with_options(size, komi, ruleset, scoring, allow_suicide)
                };
                states.push((idx, state, model_tag));
            }
            let cfg = apply_search_profile(apply_search_overrides(go_quartz(size), overrides), search_profile);
            run_multi_generic!(Go, states, cfg, n_actions)
        }
        "tictactoe" => {
            let mut states = Vec::with_capacity(jobs.len());
            for (idx, job) in jobs.into_iter().enumerate() {
                let board_raw = job
                    .get("board")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let model_tag = job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let player = job.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
                let state = if board_raw.len() == 9 {
                    TicTacToe::from_board(
                        &board_raw.iter().map(|v| v.as_i64().unwrap_or(0) as i8).collect::<Vec<_>>(),
                        player,
                    )
                } else {
                    TicTacToe::initial()
                };
                states.push((idx, state, model_tag));
            }
            let cfg = apply_search_profile(apply_search_overrides(
                MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                overrides,
            ), search_profile);
            run_multi_generic!(TicTacToe, states, cfg, 9)
        }
        game if is_chess_game_name(game) => {
            let dual_model = jobs.iter().any(|job| job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) != 0);
            let (eval_a, eval_b) = make_eval_pair::<Chess>(
                n_threads,
                batch_size,
                batch_timeout_us,
                n_actions,
                jobs.len() > 1 || dual_model,
                dual_model,
            );
            let base_cfg = apply_search_profile(apply_search_overrides(chess_quartz(), overrides), search_profile);
            let qcfg = base_cfg.quartz.clone().unwrap_or_default();
            let results = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(jobs.len());
                for job in jobs {
                    let model_tag = job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let eval = if model_tag == 1 {
                        eval_b.clone().unwrap_or_else(|| eval_a.clone())
                    } else {
                        eval_a.clone()
                    };
                    let cfg = base_cfg.clone();
                    let qcfg = qcfg.clone();
                    handles.push(scope.spawn(move || {
                        let state = if let Some(fen) = job.get("fen").and_then(|v| v.as_str()) {
                            Chess::from_fen(fen).unwrap_or_else(|_| chess_state_from_request(line, game == "chess960"))
                        } else {
                            chess_state_from_request(line, game == "chess960")
                        };
                        let engine = MctsEngine::new(state, eval, cfg);
                        let (iterations, stop_reason, p_flip, value, sigma_q, hbar_eff) = match search_profile {
                            SearchProfile::Quartz => {
                                let mut ctrl = QuartzController::new(iters, qcfg);
                                if n_threads > 1 {
                                    engine.run_par_quartz(&mut ctrl, n_threads);
                                } else {
                                    engine.run_quartz(&mut ctrl);
                                }
                                let s = ctrl.last_stats();
                                (
                                    engine.root.n_total.load(Ordering::Relaxed),
                                    format!("{:?}", ctrl.last_stop_reason()),
                                    s.p_flip,
                                    s.mean_q,
                                    s.sigma_q,
                                    s.hbar_eff,
                                )
                            }
                            SearchProfile::Baseline | SearchProfile::BaselineStrict => {
                                let stats = if n_threads > 1 {
                                    engine.run_par(&FixedIterations::new(iters), n_threads)
                                } else {
                                    engine.run(&mut FixedIterations::new(iters))
                                };
                                (
                                    stats.iterations,
                                    format!("{:?}", stats.stop_reason),
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                )
                            }
                        };
                        let best: usize = engine.best_move().map(chess_policy_index).unwrap_or(0);
                        let guard = engine.root.edges.read().unwrap();
                        let mut visits = vec![0u32; CHESS_POLICY_ACTIONS];
                        for e in guard.iter() {
                            let i = chess_policy_index(e.mv);
                            if i < CHESS_POLICY_ACTIONS {
                                visits[i] = visits[i].saturating_add(e.n.load(Ordering::Relaxed));
                            }
                        }
                        drop(guard);
                        let total: u32 = visits.iter().sum();
                        let pol: Vec<String> = visits
                            .iter()
                            .enumerate()
                            .filter(|(_, &n)| n > 0)
                            .map(|(i, &n)| format!("{}:{:.4}", i, n as f32 / total.max(1) as f32))
                            .collect();
                        let best_move_uci = engine.best_move().map(|mv| mv.to_uci()).unwrap_or_default();
                        let result_fen = if let Some(mv) = engine.best_move() {
                            engine.root_state().apply_move(mv).to_fen()
                        } else {
                            engine.root_state().to_fen()
                        };
                        serde_json::json!({
                            "best_move": best,
                            "policy": pol,
                            "p_flip": f_or(p_flip, 0.0),
                            "value": f_or(value, 0.0),
                            "sigma_q": f_or(sigma_q, 0.0),
                            "hbar_eff": f_or(hbar_eff, 0.0),
                            "stop_reason": stop_reason,
                            "iterations": iterations.max(total),
                            "dup_rate": engine.par_ctrl.telemetry.snapshot().dup_rate,
                            "max_pending": engine.par_ctrl.telemetry.snapshot().max_pending,
                            "avg_vvalue": engine.par_ctrl.telemetry.snapshot().avg_vvalue,
                            "best_move_uci": best_move_uci,
                            "result_fen": result_fen,
                        })
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| serde_json::json!({})))
                    .collect::<Vec<_>>()
            });
            serde_json::json!({ "results": results }).to_string()
        }
        _ => format!("{{\"error\":\"search_nn_multi not yet supported for {}\"}}", game),
    }
}

fn handle_search_nn_multi_session_open(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let game = root
        .get("game")
        .and_then(|v| v.as_str())
        .unwrap_or("gomoku15");
    let jobs = root
        .get("jobs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if jobs.is_empty() {
        return "{\"error\":\"jobs required\"}".to_string();
    }
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let n_threads = cap_search_threads(root
        .get("n_threads")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize);
    let batch_size = root
        .get("batch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .max(1) as usize;
    let job_count = jobs.len();
    let force_batch = job_count > 1;
    let batch_timeout_us = root
        .get("batch_timeout_us")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| default_batch_timeout_us(n_threads, batch_size, job_count));
    rust_server_trace(
        "session_open_start",
        serde_json::json!({
            "game": game,
            "job_count": job_count,
            "iters": iters,
            "n_threads": n_threads,
            "batch_size": batch_size,
            "batch_timeout_us": batch_timeout_us,
            "force_batch": force_batch,
            "search_profile": search_profile_name(search_profile),
        }),
    );
    let session = match game {
        "gomoku7" => {
            let cfg = apply_search_profile(apply_search_overrides(
                MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                }),
                overrides,
            ), search_profile);
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::Gomoku(SearchSession {
                states: jobs.into_iter().map(|job| Some(parse_gomoku7_job(&job))).collect(),
                eval: make_eval::<Gomoku>(n_threads, batch_size, batch_timeout_us, 49, force_batch),
                cfg,
                qcfg,
                iters,
                n_threads,
                n_actions: 49,
                search_profile,
            })
        }
        _ if parse_gomoku15_variant(game).is_some() => {
            let variant = parse_gomoku15_variant(game).unwrap();
            let cfg = apply_search_profile(apply_search_overrides(gomoku15_quartz(variant), overrides), search_profile);
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::Gomoku15(Gomoku15Session {
                inner: SearchSession {
                    states: jobs
                        .into_iter()
                        .map(|job| Some(parse_gomoku15_job(&job, variant)))
                        .collect(),
                    eval: make_eval::<Gomoku15>(n_threads, batch_size, batch_timeout_us, 225, force_batch),
                    cfg,
                    qcfg,
                    iters,
                    n_threads,
                    n_actions: 225,
                    search_profile,
                },
                variant,
            })
        }
        _ if parse_go_game(game).is_some() => {
            let (size, default_ruleset) = parse_go_game(game).unwrap();
            let ruleset = parse_go_ruleset(line, default_ruleset);
            let scoring = parse_go_scoring(line, ruleset.scoring());
            let komi = parse_go_komi(line, 7.5);
            let allow_suicide = parse_go_allow_suicide(line, false);
            let n_actions = size * size + 1;
            let cfg = apply_search_profile(apply_search_overrides(go_quartz(size), overrides), search_profile);
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::Go(GoSession {
                inner: SearchSession {
                    states: jobs
                        .into_iter()
                        .map(|job| {
                            Some(parse_go_job(
                                &job,
                                size,
                                ruleset,
                                scoring,
                                komi,
                                allow_suicide,
                            ))
                        })
                        .collect(),
                    eval: make_eval::<Go>(n_threads, batch_size, batch_timeout_us, n_actions, force_batch),
                    cfg,
                    qcfg,
                    iters,
                    n_threads,
                    n_actions,
                    search_profile,
                },
                size,
                ruleset,
                scoring,
                komi,
                allow_suicide,
            })
        }
        game if is_chess_game_name(game) => {
            let default_960 = game == "chess960";
            let cfg = apply_search_profile(
                apply_search_overrides(chess_quartz(), overrides),
                search_profile,
            );
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::Chess(ChessSession {
                inner: SearchSession {
                    states: jobs
                        .into_iter()
                        .map(|job| Some(parse_chess_job(&job, default_960)))
                        .collect(),
                    eval: make_eval::<Chess>(
                        n_threads,
                        batch_size,
                        batch_timeout_us,
                        CHESS_POLICY_ACTIONS,
                        force_batch,
                    ),
                    cfg,
                    qcfg,
                    iters,
                    n_threads,
                    n_actions: CHESS_POLICY_ACTIONS,
                    search_profile,
                },
                default_960,
            })
        }
        "tictactoe" => {
            let cfg = apply_search_profile(apply_search_overrides(
                MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                overrides,
            ), search_profile);
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::TicTacToe(SearchSession {
                states: jobs.into_iter().map(|job| Some(parse_tictactoe_job(&job))).collect(),
                eval: make_eval::<TicTacToe>(n_threads, batch_size, batch_timeout_us, 9, force_batch),
                cfg,
                qcfg,
                iters,
                n_threads,
                n_actions: 9,
                search_profile,
            })
        }
        _ => {
            return format!(
                "{{\"error\":\"search_nn_multi_session not yet supported for {}\"}}",
                game
            )
        }
    };

    rust_server_trace(
        "session_open_built",
        serde_json::json!({
            "game": game,
            "job_count": job_count,
        }),
    );
    let results = session.search();
    let session_id = next_search_session_id();
    search_sessions().lock().unwrap().insert(session_id, session);
    rust_server_trace(
        "session_open_reply",
        serde_json::json!({
            "session_id": session_id,
            "results_len": results.len(),
            "null_results": results.iter().filter(|v| v.is_null()).count(),
        }),
    );
    serde_json::json!({ "session_id": session_id, "results": results }).to_string()
}

fn handle_search_nn_multi_session_step(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let Some(session_id) = root.get("session_id").and_then(|v| v.as_u64()) else {
        return "{\"error\":\"session_id required\"}".to_string();
    };
    let updates = root
        .get("updates")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut sessions = search_sessions().lock().unwrap();
    let Some(session) = sessions.get_mut(&session_id) else {
        return "{\"error\":\"unknown session_id\"}".to_string();
    };
    if let Err(err) = session.apply_updates(&updates) {
        return serde_json::json!({ "error": err }).to_string();
    }
    let results = session.search();
    serde_json::json!({ "session_id": session_id, "results": results }).to_string()
}

fn handle_search_nn_multi_session_close(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let Some(session_id) = root.get("session_id").and_then(|v| v.as_u64()) else {
        return "{\"error\":\"session_id required\"}".to_string();
    };
    let removed = search_sessions().lock().unwrap().remove(&session_id).is_some();
    serde_json::json!({ "ok": removed, "session_id": session_id }).to_string()
}

fn parse_eval_runner_sessions_gomoku(
    sessions: &[serde_json::Value],
) -> Vec<EvalRunnerSession<Gomoku>> {
    sessions
        .iter()
        .map(|session| EvalRunnerSession {
            game_id: session
                .get("game_id")
                .and_then(|v| v.as_str())
                .unwrap_or("g0000")
                .to_string(),
            state: parse_gomoku7_job(session),
            black_tag: session
                .get("black_tag")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            white_tag: session
                .get("white_tag")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u32,
            opening: session
                .get("opening")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            seed: session.get("seed").and_then(|v| v.as_u64()),
            ply: session.get("ply").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            total_time_ms: session
                .get("total_time_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            done: session
                .get("done")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            error: None,
        })
        .collect()
}

fn handle_eval_nn_run_gomoku(
    sessions_json: &[serde_json::Value],
    iters: u32,
    max_moves: usize,
    overrides: SearchOverrides,
    search_profile: SearchProfile,
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
) -> String {
    let mut sessions = parse_eval_runner_sessions_gomoku(sessions_json);
    let dual_model = sessions
        .iter()
        .any(|sess| sess.black_tag != 0 || sess.white_tag != 0);
    let cfg = apply_search_profile(
        apply_search_overrides(
            MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                min_visits: 15,
                check_interval: 20,
                ..Default::default()
            }),
            overrides,
        ),
        search_profile,
    );
    let qcfg = cfg.quartz.clone();
    let batch_cfg = crate::mcts::eval::BatchConfig {
        max_batch_size: batch_size.max(n_threads),
        timeout_us: batch_timeout_us,
    };
    let broker = GlobalBroker::<usize>::new(49, batch_cfg);
    let eval_a = BatchStdioEval::<usize>::from_broker(&broker, 0);
    let eval_b = if dual_model {
        Some(BatchStdioEval::<usize>::from_broker(&broker, 1))
    } else {
        None
    };
    let started = Instant::now();
    let progress_every = (sessions.len() / 10).clamp(1, 25);
    let mut last_reported = 0usize;
    rust_server_trace(
        "eval_runner_start",
        serde_json::json!({
            "game": "gomoku7",
            "session_count": sessions.len(),
            "iters": iters,
            "max_moves": max_moves,
            "n_threads": n_threads,
            "batch_size": batch_size,
            "batch_timeout_us": batch_timeout_us,
            "search_profile": search_profile_name(search_profile),
        }),
    );

    loop {
        let active = sessions
            .iter()
            .enumerate()
            .filter_map(|(idx, sess)| {
                if sess.done || sess.state.is_terminal() || sess.ply >= max_moves {
                    None
                } else {
                    Some((idx, sess.state.clone(), sess.active_model_tag()))
                }
            })
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        let active_count = active.len();
        let completed_before = sessions.iter().filter(|sess| sess.done).count();
        let batch_started = Instant::now();
        let results = run_multi_async_batch_tags(
            &active,
            eval_a.clone(),
            eval_b.clone(),
            &cfg,
            qcfg.clone(),
            iters,
            n_threads,
            49,
            search_profile,
        );
        let batch_elapsed_ms = batch_started.elapsed().as_secs_f64() * 1000.0;
        let share_ms = batch_elapsed_ms / active_count.max(1) as f64;
        let mut wave_errors = 0usize;
        let mut wave_nulls = 0usize;
        for (idx, _, _) in active {
            let Some(sess) = sessions.get_mut(idx) else {
                continue;
            };
            if sess.done || sess.state.is_terminal() || sess.ply >= max_moves {
                sess.done = true;
                continue;
            }
            let result = results.get(idx).cloned().unwrap_or(serde_json::Value::Null);
            if result.is_null() {
                wave_nulls += 1;
            }
            if let Some(err) = result.get("error").and_then(|v| v.as_str()) {
                sess.error = Some(err.to_string());
                sess.done = true;
                wave_errors += 1;
                continue;
            }
            let Some(action) = result.get("best_move").and_then(|v| v.as_u64()) else {
                sess.error = Some("missing best_move".to_string());
                sess.done = true;
                wave_errors += 1;
                continue;
            };
            let Some(mv) = sess.state.idx_to_move(action as usize) else {
                sess.error = Some(format!("invalid action {} for eval runner", action));
                sess.done = true;
                wave_errors += 1;
                continue;
            };
            let move_time_ms = result
                .get("time_used_ms")
                .and_then(|v| v.as_f64())
                .filter(|v| *v > 0.0)
                .unwrap_or(share_ms);
            sess.total_time_ms += move_time_ms;
            sess.state = sess.state.apply_move(mv);
            sess.ply += 1;
            if sess.state.is_terminal() || sess.ply >= max_moves {
                sess.done = true;
            }
        }
        let completed = sessions.iter().filter(|sess| sess.done).count();
        rust_server_trace(
            "eval_runner_wave",
            serde_json::json!({
                "game": "gomoku7",
                "active_games": active_count,
                "completed_before": completed_before,
                "completed_after": completed,
                "newly_completed": completed.saturating_sub(completed_before),
                "wave_errors": wave_errors,
                "wave_nulls": wave_nulls,
                "batch_elapsed_ms": batch_elapsed_ms,
                "share_ms": share_ms,
            }),
        );
        if completed >= last_reported + progress_every || completed == sessions.len() {
            last_reported = completed;
            rust_server_trace(
                "eval_runner_progress",
                serde_json::json!({
                    "game": "gomoku7",
                    "completed_games": completed,
                    "total_games": sessions.len(),
                    "active_games": sessions.len().saturating_sub(completed),
                }),
            );
        }
    }

    let records = sessions
        .iter()
        .map(build_eval_record_json)
        .collect::<Vec<_>>();
    rust_server_trace(
        "eval_runner_done",
        serde_json::json!({
            "game": "gomoku7",
            "completed_games": records.len(),
            "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
            "errors": records.iter().filter(|r| r.get("error").and_then(|v| v.as_str()).is_some()).count(),
        }),
    );
    serde_json::json!({
        "valid_eval": true,
        "game": "gomoku7",
        "records": records,
        "completed_games": sessions.len(),
        "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
    })
    .to_string()
}

fn handle_eval_nn_run(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let game = root.get("game").and_then(|v| v.as_str()).unwrap_or("gomoku7");
    let sessions = root
        .get("sessions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if sessions.is_empty() {
        return "{\"error\":\"sessions required\"}".to_string();
    }
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
    let max_moves = root
        .get("max_moves")
        .and_then(|v| v.as_u64())
        .unwrap_or(500) as usize;
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let n_threads = cap_search_threads(
        root.get("n_threads")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as usize,
    );
    let batch_size = root
        .get("batch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .max(1) as usize;
    let batch_timeout_us = root
        .get("batch_timeout_us")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| default_batch_timeout_us(n_threads, batch_size, sessions.len()));
    match game {
        "gomoku7" => handle_eval_nn_run_gomoku(
            &sessions,
            iters,
            max_moves,
            overrides,
            search_profile,
            n_threads,
            batch_size,
            batch_timeout_us,
        ),
        _ => serde_json::json!({
            "error": format!("eval_nn_run not yet supported for {}", game)
        })
        .to_string(),
    }
}

fn handle_selfplay_nn_run_gomoku(
    num_games: usize,
    parallel: usize,
    iters: u32,
    temp_threshold: usize,
    overrides: SearchOverrides,
    search_profile: SearchProfile,
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
    base_seed: u64,
) -> String {
    let cfg = apply_search_profile(
        apply_search_overrides(
            MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                min_visits: 15,
                check_interval: 20,
                ..Default::default()
            }),
            overrides,
        ),
        search_profile,
    );
    let qcfg = cfg.quartz.clone();
    let batch_cfg = crate::mcts::eval::BatchConfig {
        max_batch_size: batch_size.max(n_threads),
        timeout_us: batch_timeout_us,
    };
    let broker = GlobalBroker::<usize>::new(49, batch_cfg);
    let eval_a = BatchStdioEval::<usize>::from_broker(&broker, 0);
    let slot_count = parallel.max(batch_size).max(1).min(num_games.max(1));
    let mut games_done = 0usize;
    let mut games_started = 0usize;
    let started = Instant::now();
    let progress_every = (num_games / 10).clamp(1, 25);
    let mut last_reported = 0usize;
    let mut sessions = (0..slot_count)
        .map(|slot| {
            let seed = base_seed.wrapping_add(slot as u64).wrapping_add(1);
            games_started += 1;
            GomokuSelfplaySession {
                state: Gomoku::new_with_win(7, 4),
                rng: StdRng::seed_from_u64(seed),
                moves: 0,
                finished: false,
                winner: 0.0,
                board_history: Vec::new(),
                player_history: Vec::new(),
                policy_history: Vec::new(),
                trace_history: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    rust_server_trace(
        "selfplay_runner_start",
        serde_json::json!({
            "game": "gomoku7",
            "num_games": num_games,
            "parallel": parallel,
            "slot_count": slot_count,
            "batch_size": batch_size,
            "iters": iters,
            "temp_threshold": temp_threshold,
            "search_profile": search_profile_name(search_profile),
        }),
    );

    while games_done < num_games {
        let active = sessions
            .iter()
            .enumerate()
            .filter_map(|(idx, sess)| {
                if sess.finished || sess.state.is_terminal() {
                    None
                } else {
                    Some((idx, sess.state.clone(), 0u32))
                }
            })
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        let active_count = active.len();
        let finished_before = games_done;
        let games_started_before = games_started;
        let wave_started = Instant::now();
        let mut emitted_chunk_games = Vec::new();
        let results = run_multi_async_batch_tags(
            &active,
            eval_a.clone(),
            None,
            &cfg,
            qcfg.clone(),
            iters,
            n_threads,
            49,
            search_profile,
        );
        let mut wave_finished = 0usize;
        let mut wave_nulls = 0usize;
        let mut wave_positions_emitted = 0usize;
        for (idx, _, _) in active {
            let Some(sess) = sessions.get_mut(idx) else {
                continue;
            };
            let result = results.get(idx).cloned().unwrap_or(serde_json::Value::Null);
            if result.is_null() {
                wave_nulls += 1;
            }
            let policy_entries = result
                .get("policy")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if policy_entries.is_empty() {
                sess.finished = true;
                sess.winner = terminal_black_score(&sess.state).unwrap_or(0.0);
            } else {
                sess.board_history.push(sess.state.board_as_12());
                sess.player_history.push(sess.state.current_player());
                sess.policy_history.push(policy_entries.clone());
                wave_positions_emitted += 1;
                sess.trace_history.push(serde_json::json!({
                    "p_flip": result.get("p_flip").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "value": result.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "sigma_q": result.get("sigma_q").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "stop_reason": result.get("stop_reason").and_then(|v| v.as_str()).unwrap_or(""),
                    "hbar_eff": result.get("hbar_eff").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "iterations": result.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0),
                    "dup_rate": result.get("dup_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "max_pending": result.get("max_pending").and_then(|v| v.as_u64()).unwrap_or(0),
                    "avg_vvalue": result.get("avg_vvalue").and_then(|v| v.as_f64()).unwrap_or(0.0),
                }));
                let fallback_best = result.get("best_move").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(action) = choose_selfplay_action(
                    &mut sess.rng,
                    &sess.state,
                    &policy_entries,
                    sess.moves,
                    temp_threshold,
                    fallback_best,
                ) {
                    if let Some(mv) = sess.state.idx_to_move(action) {
                        sess.state = sess.state.apply_move(mv);
                        sess.moves += 1;
                        if sess.state.is_terminal() {
                            sess.finished = true;
                            sess.winner = terminal_black_score(&sess.state).unwrap_or(0.0);
                        }
                    } else {
                        sess.finished = true;
                    }
                } else {
                    sess.finished = true;
                }
            }
            if sess.finished {
                let outcome = sess.winner;
                emitted_chunk_games.push(serde_json::json!({
                    "states": sess.board_history,
                    "players": sess.player_history,
                    "policies": sess.policy_history,
                    "outcome": outcome,
                    "trace": sess.trace_history,
                }));
                games_done += 1;
                wave_finished += 1;
                if games_started < num_games {
                    let seed = base_seed.wrapping_add(games_started as u64).wrapping_add(1);
                    *sess = GomokuSelfplaySession {
                        state: Gomoku::new_with_win(7, 4),
                        rng: StdRng::seed_from_u64(seed),
                        moves: 0,
                        finished: false,
                        winner: 0.0,
                        board_history: Vec::new(),
                        player_history: Vec::new(),
                        policy_history: Vec::new(),
                        trace_history: Vec::new(),
                    };
                    games_started += 1;
                }
            }
        }
        rust_server_trace(
            "selfplay_runner_wave",
            serde_json::json!({
                "game": "gomoku7",
                "active_games": active_count,
                "frontier_slots": active_count,
                "completed_before": finished_before,
                "completed_after": games_done,
                "newly_completed": wave_finished,
                "wave_nulls": wave_nulls,
                "wave_positions_emitted": wave_positions_emitted,
                "replenished_slots": games_started.saturating_sub(games_started_before),
                "batch_elapsed_ms": wave_started.elapsed().as_secs_f64() * 1000.0,
                "games_started": games_started,
            }),
        );
        if !emitted_chunk_games.is_empty() {
            emit_json_message(&serde_json::json!({
                "selfplay_chunk": {
                    "games": emitted_chunk_games,
                }
            }));
        }
        if games_done >= last_reported + progress_every || games_done == num_games {
            last_reported = games_done;
            rust_server_trace(
                "selfplay_runner_progress",
                serde_json::json!({
                    "completed_games": games_done,
                    "total_games": num_games,
                }),
            );
            emit_json_message(&serde_json::json!({
                "selfplay_progress": {
                    "completed_games": games_done,
                    "total_games": num_games,
                }
            }));
        }
    }
    rust_server_trace(
        "selfplay_runner_done",
        serde_json::json!({
            "completed_games": games_done,
            "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
        }),
    );
    serde_json::json!({
        "game": "gomoku7",
        "selfplay_done": {
            "completed_games": games_done,
            "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
        }
    })
    .to_string()
}

fn handle_selfplay_nn_run(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let game = root.get("game").and_then(|v| v.as_str()).unwrap_or("gomoku7");
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
    let num_games = root.get("n_games").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let parallel = root.get("parallel").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let temp_threshold = root
        .get("temp_threshold")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as usize;
    let seed = root.get("seed").and_then(|v| v.as_u64()).unwrap_or(0xC0FFEE);
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let n_threads = cap_search_threads(
        root.get("n_threads")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as usize,
    );
    let batch_size = root
        .get("batch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .max(1) as usize;
    let batch_timeout_us = root
        .get("batch_timeout_us")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| default_batch_timeout_us(n_threads, batch_size, parallel.max(1)));
    match game {
        "gomoku7" => handle_selfplay_nn_run_gomoku(
            num_games,
            parallel,
            iters,
            temp_threshold,
            overrides,
            search_profile,
            n_threads,
            batch_size,
            batch_timeout_us,
            seed,
        ),
        _ => serde_json::json!({
            "error": format!("selfplay_nn_run not yet supported for {}", game)
        }).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_json() {
        assert_eq!(jstr(r#"{"cmd":"selfplay"}"#, "cmd"), Some("selfplay"));
    }
    #[test]
    fn test_chess() {
        let r = search_chess(r#"{}"#, false, 30);
        assert!(r.contains("move_str"));
    }
    #[test]
    fn test_chess960() {
        let r = search_chess(r#"{"chess960_index":0}"#, true, 30);
        assert!(r.contains("move_str"));
    }
    #[test]
    fn test_chess_state_lists_legal_moves() {
        let r = handle_chess_state(r#"{"game":"chess"}"#, false);
        assert!(r.contains("\"status\":\"ok\""));
        assert!(r.contains("\"legal_moves\""));
        assert!(r.contains("e2e4"));
    }
    #[test]
    fn test_chess_apply_advances_fen() {
        let r = handle_chess_apply(r#"{"game":"chess","move_uci":"e2e4"}"#, false);
        assert!(r.contains("\"applied_move\":\"e2e4\""));
        assert!(r.contains("\"fen\":\"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\""));
    }
    #[test]
    fn test_go9() {
        let r = search_go(r#"{}"#, 9, GoRuleset::Chinese, 30);
        assert!(r.contains("move"));
    }
    #[test]
    fn test_go13() {
        let r = search_go(r#"{}"#, 13, GoRuleset::Chinese, 30);
        assert!(r.contains("move"));
    }
    #[test]
    fn test_gomoku15() {
        let r = search_gomoku15(r#"{}"#, GomokuVariant::Standard, 30);
        assert!(r.contains("move"));
    }
    #[test]
    fn test_tictactoe() {
        let r = search_tictactoe(r#"{}"#, 30);
        assert!(r.contains("move"));
    }

    #[test]
    fn test_chess_session_updates_apply_action_and_deactivate() {
        let state = Chess::standard();
        let action = chess_policy_index(
            state
                .legal_moves()
                .into_iter()
                .find(|mv| mv.to_uci() == "e2e4")
                .unwrap(),
        );
        let cfg = chess_quartz();
        let qcfg = cfg.quartz.clone();
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(4));
        let mut session = ChessSession {
            inner: SearchSession {
                states: vec![Some(state), Some(Chess::standard())],
                eval,
                cfg,
                qcfg,
                iters: 8,
                n_threads: 1,
                n_actions: CHESS_POLICY_ACTIONS,
                search_profile: SearchProfile::Quartz,
            },
            default_960: false,
        };

        apply_updates_chess(
            &mut session,
            &[
                serde_json::json!({"action": action}),
                serde_json::json!({"deactivate": true}),
            ],
        )
        .unwrap();

        let moved = session.inner.states[0].as_ref().unwrap();
        assert_eq!(
            moved.to_fen(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        );
        assert!(session.inner.states[1].is_none());
    }

    #[test]
    fn test_parse_gomoku15_variant_aliases() {
        assert_eq!(
            parse_gomoku15_variant("gomoku15"),
            Some(GomokuVariant::Freestyle)
        );
        assert_eq!(
            parse_gomoku15_variant("gomoku15_std"),
            Some(GomokuVariant::Standard)
        );
        assert_eq!(
            parse_gomoku15_variant("gomoku15_omok"),
            Some(GomokuVariant::Omok)
        );
        assert_eq!(
            parse_gomoku15_variant("gomoku15_renju"),
            Some(GomokuVariant::Renju)
        );
        assert_eq!(
            parse_gomoku15_variant("gomoku15_caro"),
            Some(GomokuVariant::Caro)
        );
    }

    #[test]
    fn test_parse_go_game_aliases() {
        assert_eq!(parse_go_game("go9"), Some((9, GoRuleset::Chinese)));
        assert_eq!(parse_go_game("go9_jp"), Some((9, GoRuleset::Japanese)));
        assert_eq!(parse_go_game("go19_kr"), Some((19, GoRuleset::Korean)));
    }
}
