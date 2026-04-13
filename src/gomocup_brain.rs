use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::game::{Evaluator, GameState};
use crate::gomocup_bundle::{apply_bundle_search_config, load_bundle, LoadedGomocupBundle};
use crate::games::gomoku15::{gomoku15_quartz_timed, Gomoku15, GomokuVariant};
use crate::games::Gomoku;
use crate::mcts::eval::ShortRollout;
use crate::mcts::quartz::{QuartzConfig, QuartzController};
use crate::mcts::search::FixedIterations;
use crate::mcts::{MctsConfig, MctsEngine, PwConfig};

const DEFAULT_BUDGET_MS: u64 = 1_000;
const MIN_BUDGET_MS: u64 = 10;
const SAFETY_MARGIN_MS: u64 = 20;
const MAX_VISITS: u32 = 50_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrainRule {
    Freestyle,
    Standard,
    Renju,
    Caro,
    Omok,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeKind {
    Gomoku15(GomokuVariant),
    Freestyle { size: usize },
}

#[derive(Clone, Debug, Default)]
pub struct BrainInfo {
    pub timeout_turn_ms: u64,
    pub timeout_match_ms: u64,
    pub time_left_ms: Option<u64>,
    pub time_increment_ms: u64,
    pub max_memory_mb: Option<u64>,
    pub thread_num: Option<u32>,
    pub folder: Option<String>,
    pub rule_code: Option<i32>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum HandleResult {
    NoResponse,
    Response(String),
    Quit,
}

pub struct GomocupBrain {
    info: BrainInfo,
    rule: BrainRule,
    size: Option<usize>,
    runtime: Option<RuntimeKind>,
    board: Vec<u8>,
    board_mode: Option<BoardParseMode>,
    bundle: Option<LoadedGomocupBundle>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoardParseMode {
    Xy,
    Yx,
}

impl Default for GomocupBrain {
    fn default() -> Self {
        let mut brain = Self {
            info: BrainInfo::default(),
            rule: BrainRule::Freestyle,
            size: None,
            runtime: None,
            board: Vec::new(),
            board_mode: None,
            bundle: None,
        };
        brain.refresh_bundle();
        brain
    }
}

pub fn parse_rule_code(code: i32) -> Option<BrainRule> {
    match code {
        0 => Some(BrainRule::Freestyle),
        1 => Some(BrainRule::Standard),
        4 => Some(BrainRule::Renju),
        8 | 9 => Some(BrainRule::Caro),
        104 => Some(BrainRule::Omok),
        _ => None,
    }
}

pub fn compute_budget_ms(info: &BrainInfo, remaining_moves: usize) -> u64 {
    let remaining = remaining_moves.max(1) as u64;

    let turn_cap = if info.timeout_turn_ms > 0 {
        Some(info.timeout_turn_ms.saturating_sub(SAFETY_MARGIN_MS))
    } else {
        None
    };

    let match_cap = if let Some(left) = info.time_left_ms {
        let usable = left
            .saturating_add(info.time_increment_ms)
            .saturating_sub(SAFETY_MARGIN_MS)
            .max(MIN_BUDGET_MS);
        let alloc = (usable / remaining).saturating_mul(2).max(MIN_BUDGET_MS);
        Some(alloc.min(usable))
    } else if info.timeout_match_ms > 0 {
        let alloc = (info.timeout_match_ms / remaining)
            .max(MIN_BUDGET_MS)
            .min(DEFAULT_BUDGET_MS);
        Some(alloc)
    } else {
        None
    };

    let budget = match (turn_cap, match_cap) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => DEFAULT_BUDGET_MS,
    };

    budget.max(MIN_BUDGET_MS)
}

impl GomocupBrain {
    fn bundle_search_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        if let Ok(path) = std::env::var("QUARTZ_GOMOCUP_BUNDLE_DIR") {
            if !path.trim().is_empty() {
                roots.push(PathBuf::from(path));
            }
        }
        if let Some(folder) = self.info.folder.as_ref() {
            if !folder.trim().is_empty() {
                roots.push(PathBuf::from(folder));
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                roots.push(parent.to_path_buf());
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            roots.push(cwd);
        }
        roots
    }

    fn refresh_bundle(&mut self) {
        self.bundle = load_bundle(&self.bundle_search_roots());
    }

    fn bundle_budget_ms(&self) -> Option<u64> {
        self.bundle.as_ref().and_then(|bundle| bundle.budget_ms())
    }

    fn bundle_max_visits(&self) -> u32 {
        self.bundle
            .as_ref()
            .and_then(|bundle| bundle.max_visits())
            .unwrap_or(MAX_VISITS)
    }

    pub fn handle_line(&mut self, line: &str) -> HandleResult {
        let line = line.trim();
        if line.is_empty() {
            return HandleResult::NoResponse;
        }

        if self.board_mode.is_some() {
            return self.handle_board_line(line);
        }

        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or_default().to_ascii_uppercase();
        match cmd.as_str() {
            "START" => self.handle_start(parts.next()),
            "RECTSTART" => self.handle_rectstart(line),
            "RESTART" => self.handle_restart(),
            "BEGIN" => self.handle_begin(),
            "TURN" => self.handle_turn(parts.next()),
            "BOARD" => self.handle_board_start(BoardParseMode::Xy),
            "YXBOARD" => self.handle_board_start(BoardParseMode::Yx),
            "TAKEBACK" => self.handle_takeback(parts.next()),
            "INFO" => self.handle_info(line),
            "ABOUT" => HandleResult::Response(self.about()),
            "END" => HandleResult::Quit,
            _ => HandleResult::Response("UNKNOWN".to_string()),
        }
    }

    fn handle_start(&mut self, arg: Option<&str>) -> HandleResult {
        let Some(raw) = arg else {
            return HandleResult::Response("ERROR missing board size".to_string());
        };
        let Ok(size) = raw.trim().parse::<usize>() else {
            return HandleResult::Response(format!("ERROR invalid board size: {raw}"));
        };
        match self.init_runtime(size) {
            Ok(()) => HandleResult::Response("OK".to_string()),
            Err(err) => HandleResult::Response(format!("ERROR {err}")),
        }
    }

    fn handle_rectstart(&mut self, line: &str) -> HandleResult {
        let payload = line
            .split_once(char::is_whitespace)
            .map(|(_, rest)| rest.trim())
            .unwrap_or("");
        let dims: Vec<_> = payload
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .collect();
        if dims.len() != 2 {
            return HandleResult::Response("ERROR RECTSTART expects width,height".to_string());
        }
        let Ok(w) = dims[0].parse::<usize>() else {
            return HandleResult::Response("ERROR invalid RECTSTART width".to_string());
        };
        let Ok(h) = dims[1].parse::<usize>() else {
            return HandleResult::Response("ERROR invalid RECTSTART height".to_string());
        };
        if w != h {
            return HandleResult::Response("ERROR only square boards are supported".to_string());
        }
        match self.init_runtime(w) {
            Ok(()) => HandleResult::Response("OK".to_string()),
            Err(err) => HandleResult::Response(format!("ERROR {err}")),
        }
    }

    fn handle_restart(&mut self) -> HandleResult {
        let Some(size) = self.size else {
            return HandleResult::Response("ERROR board is not initialized".to_string());
        };
        match self.init_runtime(size) {
            Ok(()) => HandleResult::Response("OK".to_string()),
            Err(err) => HandleResult::Response(format!("ERROR {err}")),
        }
    }

    fn handle_begin(&mut self) -> HandleResult {
        match self.think_and_play() {
            Ok(resp) => HandleResult::Response(resp),
            Err(err) => HandleResult::Response(format!("ERROR {err}")),
        }
    }

    fn handle_turn(&mut self, arg: Option<&str>) -> HandleResult {
        let Some(raw) = arg else {
            return HandleResult::Response("ERROR TURN expects x,y".to_string());
        };
        let (x, y) = match parse_xy(raw) {
            Some(v) => v,
            None => return HandleResult::Response("ERROR invalid TURN coordinates".to_string()),
        };
        if let Err(err) = self.apply_coord(x, y, None) {
            return HandleResult::Response(format!("ERROR {err}"));
        }
        match self.think_and_play() {
            Ok(resp) => HandleResult::Response(resp),
            Err(err) => HandleResult::Response(format!("ERROR {err}")),
        }
    }

    fn handle_board_start(&mut self, mode: BoardParseMode) -> HandleResult {
        let Some(size) = self.size else {
            return HandleResult::Response("ERROR board is not initialized".to_string());
        };
        self.board = vec![0; size * size];
        self.board_mode = Some(mode);
        HandleResult::NoResponse
    }

    fn handle_board_line(&mut self, line: &str) -> HandleResult {
        if line.eq_ignore_ascii_case("DONE") {
            self.board_mode = None;
            return match self.think_and_play() {
                Ok(resp) => HandleResult::Response(resp),
                Err(err) => HandleResult::Response(format!("ERROR {err}")),
            };
        }
        let mode = self.board_mode.unwrap_or(BoardParseMode::Xy);
        let Some((x, y, player)) = parse_board_triplet(line) else {
            self.board_mode = None;
            return HandleResult::Response("ERROR invalid BOARD entry".to_string());
        };
        let (bx, by) = match mode {
            BoardParseMode::Xy => (x, y),
            BoardParseMode::Yx => (y, x),
        };
        match self.apply_coord(bx, by, Some(player)) {
            Ok(()) => HandleResult::NoResponse,
            Err(err) => {
                self.board_mode = None;
                HandleResult::Response(format!("ERROR {err}"))
            }
        }
    }

    fn handle_takeback(&mut self, arg: Option<&str>) -> HandleResult {
        let Some(raw) = arg else {
            return HandleResult::Response("ERROR TAKEBACK expects x,y".to_string());
        };
        let Some((x, y)) = parse_xy(raw) else {
            return HandleResult::Response("ERROR invalid TAKEBACK coordinates".to_string());
        };
        let Some(size) = self.size else {
            return HandleResult::Response("ERROR board is not initialized".to_string());
        };
        if x >= size || y >= size {
            return HandleResult::Response("ERROR TAKEBACK out of bounds".to_string());
        }
        let idx = y * size + x;
        if self.board.get(idx).copied().unwrap_or(0) == 0 {
            return HandleResult::Response("ERROR TAKEBACK empty cell".to_string());
        }
        self.board[idx] = 0;
        HandleResult::Response("OK".to_string())
    }

    fn handle_info(&mut self, line: &str) -> HandleResult {
        let mut parts = line.splitn(3, char::is_whitespace);
        let _ = parts.next();
        let Some(key) = parts.next() else {
            return HandleResult::NoResponse;
        };
        let value = parts.next().unwrap_or("").trim();
        match key.to_ascii_lowercase().as_str() {
            "rule" => {
                if let Ok(code) = value.parse::<i32>() {
                    self.info.rule_code = Some(code);
                    if let Some(rule) = parse_rule_code(code) {
                        self.rule = rule;
                        if let Some(size) = self.size {
                            if self.board.iter().all(|&v| v == 0) {
                                let _ = self.init_runtime(size);
                            } else {
                                self.runtime = runtime_for(size, self.rule).ok();
                            }
                        }
                    }
                }
            }
            "timeout_turn" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.info.timeout_turn_ms = ms;
                }
            }
            "timeout_match" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.info.timeout_match_ms = ms;
                }
            }
            "time_left" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.info.time_left_ms = Some(ms);
                }
            }
            "time_increment" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.info.time_increment_ms = ms;
                }
            }
            "max_memory" => {
                if let Ok(mb) = value.parse::<u64>() {
                    self.info.max_memory_mb = Some(mb);
                }
            }
            "thread_num" => {
                if let Ok(n) = value.parse::<u32>() {
                    self.info.thread_num = Some(n);
                }
            }
            "folder" => {
                if !value.is_empty() {
                    self.info.folder = Some(value.to_string());
                    self.refresh_bundle();
                }
            }
            _ => {}
        }
        HandleResult::NoResponse
    }

    fn init_runtime(&mut self, size: usize) -> Result<(), String> {
        let runtime = runtime_for(size, self.rule)?;
        self.size = Some(size);
        self.runtime = Some(runtime);
        self.board = vec![0; size * size];
        self.board_mode = None;
        self.refresh_bundle();
        Ok(())
    }

    fn apply_coord(&mut self, x: usize, y: usize, forced_player: Option<u8>) -> Result<(), String> {
        let Some(size) = self.size else {
            return Err("board is not initialized".to_string());
        };
        if x >= size || y >= size {
            return Err("move out of bounds".to_string());
        }
        let idx = y * size + x;
        if self.board[idx] != 0 {
            return Err("cell already occupied".to_string());
        }
        let player = forced_player.unwrap_or(self.infer_player_12()?);
        if !matches!(player, 1 | 2) {
            return Err("invalid player value".to_string());
        }
        self.board[idx] = player;
        Ok(())
    }

    fn infer_player_12(&self) -> Result<u8, String> {
        let black = self.board.iter().filter(|&&v| v == 1).count();
        let white = self.board.iter().filter(|&&v| v == 2).count();
        if black == white {
            Ok(1)
        } else if black == white + 1 {
            Ok(2)
        } else {
            Err("invalid board parity".to_string())
        }
    }

    fn remaining_moves(&self) -> usize {
        let played = self.board.iter().filter(|&&v| v != 0).count();
        match self.runtime {
            Some(RuntimeKind::Gomoku15(GomokuVariant::Renju)) => 200usize.saturating_sub(played),
            _ => self.board.len().saturating_sub(played),
        }
    }

    fn think_and_play(&mut self) -> Result<String, String> {
        let size = self
            .size
            .ok_or_else(|| "board is not initialized".to_string())?;
        let runtime = self
            .runtime
            .ok_or_else(|| "runtime is not configured".to_string())?;

        let center = (size / 2, size / 2);
        let center_idx = center.1 * size + center.0;
        if self.board.iter().all(|&v| v == 0) && self.board.get(center_idx) == Some(&0) {
            self.apply_coord(center.0, center.1, None)?;
            return Ok(format!("{},{}", center.0, center.1));
        }

        let mut budget_ms = compute_budget_ms(&self.info, self.remaining_moves());
        if let Some(bundle_budget) = self.bundle_budget_ms() {
            budget_ms = budget_ms.min(bundle_budget.max(MIN_BUDGET_MS));
        }
        let started = Instant::now();
        let mv = match runtime {
            RuntimeKind::Gomoku15(variant) => self.search_gomoku15(variant, budget_ms)?,
            RuntimeKind::Freestyle { size } => self.search_freestyle(size, budget_ms)?,
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;
        if let Some(left) = self.info.time_left_ms.as_mut() {
            *left = left.saturating_sub(elapsed_ms);
        }

        let x = mv % size;
        let y = mv / size;
        self.apply_coord(x, y, None)?;
        Ok(format!("{},{}", x, y))
    }

    fn search_gomoku15(&self, variant: GomokuVariant, budget_ms: u64) -> Result<usize, String> {
        let player = self.infer_player_12()?;
        let side = if player == 1 { 1 } else { -1 };
        let board_i8: Vec<i8> = self.board.iter().map(|&v| v as i8).collect();
        let state = Gomoku15::from_board(&board_i8, side, variant);
        if state.is_terminal() {
            return Err("position is terminal".to_string());
        }

        let legal = state.legal_moves();
        if legal.is_empty() {
            return Err("no legal moves".to_string());
        }

        let config = apply_bundle_search_config(
            gomoku15_quartz_timed(variant, budget_ms),
            self.bundle.as_ref(),
        );
        let qcfg = config.quartz.clone().unwrap_or_default();
        #[cfg(feature = "onnx")]
        let eval: Arc<dyn Evaluator<Gomoku15> + Send + Sync> = self
            .bundle
            .as_ref()
            .and_then(|bundle| bundle.evaluator_for_variant(variant))
            .unwrap_or_else(|| Arc::new(ShortRollout::new(12)));
        #[cfg(not(feature = "onnx"))]
        let eval: Arc<dyn Evaluator<Gomoku15> + Send + Sync> = Arc::new(ShortRollout::new(12));
        let engine = MctsEngine::new(state, eval, config);
        if self
            .bundle
            .as_ref()
            .map(|bundle| bundle.uses_quartz_controller())
            .unwrap_or(true)
        {
            let mut ctrl = QuartzController::new(self.bundle_max_visits(), qcfg);
            engine.run_quartz(&mut ctrl);
        } else {
            let mut ctrl = FixedIterations::new(self.bundle_max_visits());
            engine.run(&mut ctrl);
        }
        Ok(engine
            .best_move()
            .or_else(|| legal.first().copied())
            .unwrap_or(legal[0]) as usize)
    }

    fn search_freestyle(&self, size: usize, budget_ms: u64) -> Result<usize, String> {
        let player = self.infer_player_12()?;
        let board_i64: Vec<i64> = self.board.iter().map(|&v| v as i64).collect();
        let state = Gomoku::from_board_12(size, 5, &board_i64, player);
        if state.is_terminal() {
            return Err("position is terminal".to_string());
        }

        let legal = state.legal_moves();
        if legal.is_empty() {
            return Err("no legal moves".to_string());
        }

        let config = apply_bundle_search_config(freestyle_quartz_timed(budget_ms), self.bundle.as_ref());
        let qcfg = config.quartz.clone().unwrap_or_default();
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(12));
        let engine = MctsEngine::new(state, eval, config);
        if self
            .bundle
            .as_ref()
            .map(|bundle| bundle.uses_quartz_controller())
            .unwrap_or(true)
        {
            let mut ctrl = QuartzController::new(self.bundle_max_visits(), qcfg);
            engine.run_quartz(&mut ctrl);
        } else {
            let mut ctrl = FixedIterations::new(self.bundle_max_visits());
            engine.run(&mut ctrl);
        }
        Ok(engine
            .best_move()
            .or_else(|| legal.first().copied())
            .unwrap_or(legal[0]))
    }

    fn about(&self) -> String {
        self.bundle
            .as_ref()
            .map(|bundle| bundle.about_line())
            .unwrap_or_else(|| {
                "name=QUARTZ-Gomocup, version=0.2, author=cosmosapjw+Codex, country=KR".to_string()
            })
    }
}

pub fn should_run_gomocup_mode<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    let Some(program) = args.next() else {
        return false;
    };
    let program = program.as_ref();
    let exe_name = std::path::Path::new(program)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(program)
        .to_ascii_lowercase();

    exe_name.starts_with("pbrain")
        || args
            .into_iter()
            .any(|arg| arg.as_ref().eq_ignore_ascii_case("--gomocup"))
}

fn runtime_for(size: usize, rule: BrainRule) -> Result<RuntimeKind, String> {
    match (size, rule) {
        (15, BrainRule::Freestyle) => Ok(RuntimeKind::Gomoku15(GomokuVariant::Freestyle)),
        (15, BrainRule::Standard) => Ok(RuntimeKind::Gomoku15(GomokuVariant::Standard)),
        (15, BrainRule::Renju) => Ok(RuntimeKind::Gomoku15(GomokuVariant::Renju)),
        (15, BrainRule::Caro) => Ok(RuntimeKind::Gomoku15(GomokuVariant::Caro)),
        (15, BrainRule::Omok) => Ok(RuntimeKind::Gomoku15(GomokuVariant::Omok)),
        (20, BrainRule::Freestyle) => Ok(RuntimeKind::Freestyle { size: 20 }),
        (size, BrainRule::Freestyle) if size == 15 => Ok(RuntimeKind::Freestyle { size }),
        (20, _) => Err("board size 20 only supports freestyle".to_string()),
        _ => Err(format!(
            "unsupported Gomocup board size/rule combination: {size}"
        )),
    }
}

fn freestyle_quartz_timed(budget_ms: u64) -> MctsConfig {
    MctsConfig::evaluation_with_pw(2.0, PwConfig::default_gomoku()).with_quartz(QuartzConfig {
        sigma_0: 0.3,
        min_visits: 30,
        check_interval: 50,
        ctm_budget_ms: budget_ms,
        ..Default::default()
    })
}

fn parse_xy(raw: &str) -> Option<(usize, usize)> {
    let parts: Vec<_> = raw
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 2 {
        return None;
    }
    let x = parts[0].parse::<usize>().ok()?;
    let y = parts[1].parse::<usize>().ok()?;
    Some((x, y))
}

fn parse_board_triplet(raw: &str) -> Option<(usize, usize, u8)> {
    let parts: Vec<_> = raw
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 3 {
        return None;
    }
    let x = parts[0].parse::<usize>().ok()?;
    let y = parts[1].parse::<usize>().ok()?;
    let player = parts[2].parse::<u8>().ok()?;
    if !matches!(player, 1 | 2) {
        return None;
    }
    Some((x, y, player))
}

pub fn serve() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut brain = GomocupBrain::default();
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        match brain.handle_line(&line) {
            HandleResult::NoResponse => {}
            HandleResult::Response(resp) => {
                let _ = writeln!(stdout, "{}", resp);
                let _ = stdout.flush();
            }
            HandleResult::Quit => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_of(brain: &mut GomocupBrain, line: &str) -> String {
        match brain.handle_line(line) {
            HandleResult::Response(resp) => resp,
            other => panic!("expected response for `{line}`, got {other:?}"),
        }
    }

    #[test]
    fn test_rule_code_mapping() {
        assert_eq!(parse_rule_code(0), Some(BrainRule::Freestyle));
        assert_eq!(parse_rule_code(1), Some(BrainRule::Standard));
        assert_eq!(parse_rule_code(4), Some(BrainRule::Renju));
        assert_eq!(parse_rule_code(8), Some(BrainRule::Caro));
        assert_eq!(parse_rule_code(9), Some(BrainRule::Caro));
        assert_eq!(parse_rule_code(104), Some(BrainRule::Omok));
    }

    #[test]
    fn test_budget_respects_timeout_turn() {
        let info = BrainInfo {
            timeout_turn_ms: 2_000,
            timeout_match_ms: 20_000,
            time_left_ms: Some(6_000),
            time_increment_ms: 0,
            max_memory_mb: None,
            thread_num: None,
            folder: None,
            rule_code: None,
        };
        let budget = compute_budget_ms(&info, 10);
        assert!(budget > 0);
        assert!(budget <= 2_000);
    }

    #[test]
    fn test_budget_uses_time_increment() {
        let no_inc = BrainInfo {
            timeout_turn_ms: 0,
            timeout_match_ms: 0,
            time_left_ms: Some(200),
            time_increment_ms: 0,
            max_memory_mb: None,
            thread_num: None,
            folder: None,
            rule_code: None,
        };
        let with_inc = BrainInfo {
            time_increment_ms: 500,
            ..no_inc.clone()
        };
        assert!(compute_budget_ms(&with_inc, 20) > compute_budget_ms(&no_inc, 20));
    }

    #[test]
    fn test_start_rejects_invalid_size_rule_combo() {
        let mut brain = GomocupBrain::default();
        assert_eq!(brain.handle_line("INFO rule 4"), HandleResult::NoResponse);
        let resp = response_of(&mut brain, "START 20");
        assert!(resp.starts_with("ERROR"));
    }

    #[test]
    fn test_begin_returns_center_on_empty_board() {
        let mut brain = GomocupBrain::default();
        assert_eq!(response_of(&mut brain, "START 15"), "OK");
        let resp = response_of(&mut brain, "BEGIN");
        assert_eq!(resp, "7,7");
    }

    #[test]
    fn test_board_done_returns_valid_move() {
        let mut brain = GomocupBrain::default();
        assert_eq!(response_of(&mut brain, "START 15"), "OK");
        assert_eq!(brain.handle_line("BOARD"), HandleResult::NoResponse);
        assert_eq!(brain.handle_line("7,7,1"), HandleResult::NoResponse);
        let resp = response_of(&mut brain, "DONE");
        let parts: Vec<_> = resp.split(',').collect();
        assert_eq!(parts.len(), 2);
        let x: usize = parts[0].parse().expect("x");
        let y: usize = parts[1].parse().expect("y");
        assert!(x < 15 && y < 15);
        assert_ne!((x, y), (7, 7));
    }

    #[test]
    fn test_yxboard_done_returns_valid_move() {
        let mut brain = GomocupBrain::default();
        assert_eq!(response_of(&mut brain, "START 15"), "OK");
        assert_eq!(brain.handle_line("YXBOARD"), HandleResult::NoResponse);
        assert_eq!(brain.handle_line("7,6,1"), HandleResult::NoResponse);
        let resp = response_of(&mut brain, "DONE");
        let parts: Vec<_> = resp.split(',').collect();
        assert_eq!(parts.len(), 2);
        let x: usize = parts[0].parse().expect("x");
        let y: usize = parts[1].parse().expect("y");
        assert!(x < 15 && y < 15);
        assert_ne!((x, y), (6, 7));
    }

    #[test]
    fn test_takeback_clears_cell() {
        let mut brain = GomocupBrain::default();
        assert_eq!(response_of(&mut brain, "START 15"), "OK");
        assert_eq!(response_of(&mut brain, "BEGIN"), "7,7");
        assert_eq!(response_of(&mut brain, "TAKEBACK 7,7"), "OK");
        assert_eq!(brain.board[7 * 15 + 7], 0);
    }

    #[test]
    fn test_info_parses_folder_and_thread_num() {
        let mut brain = GomocupBrain::default();
        assert_eq!(
            brain.handle_line("INFO folder C:\\gomocup\\work"),
            HandleResult::NoResponse
        );
        assert_eq!(
            brain.handle_line("INFO thread_num 1"),
            HandleResult::NoResponse
        );
        assert_eq!(brain.info.folder.as_deref(), Some("C:\\gomocup\\work"));
        assert_eq!(brain.info.thread_num, Some(1));
    }

    #[test]
    fn test_should_run_gomocup_mode_detects_pbrain_name() {
        assert!(should_run_gomocup_mode(["pbrain-quartz.exe"]));
        assert!(should_run_gomocup_mode([
            "target/release/mcts_demo",
            "--gomocup",
        ]));
        assert!(!should_run_gomocup_mode(["target/release/mcts_demo"]));
    }

    #[test]
    fn test_about_format() {
        let mut brain = GomocupBrain::default();
        let resp = response_of(&mut brain, "ABOUT");
        assert!(resp.contains("name="));
        assert!(resp.contains("version="));
        assert!(resp.contains("author="));
    }
}
