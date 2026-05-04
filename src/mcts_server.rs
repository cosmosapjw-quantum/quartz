//! Multi-game MCTS JSON-line server — trajectory + move protocols
//!
//! Protocol 1 (single move): {"cmd":"move","game":"chess","fen":"...","iters":200}
//! Protocol 2 (self-play):   {"cmd":"selfplay","game":"gomoku15","iters":400,"n_games":1,"temp_threshold":15}
//!   → full game trajectory with (state_planes, policy, player, outcome)

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Read, Write};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::game::{EvalResult, Evaluator, GameState};
use crate::mcts::eval::{
    global_ring_buffer, AsyncEvalTicket, BatchStdioEval, GlobalBroker, ShortRollout, SHM_MSG_JSON,
    SHM_MSG_SEARCH_RESP,
};
use crate::mcts::node::edge_lock_contention_snapshot;
use crate::mcts::parallel::{AutoThreadMode, AutoThreadPolicy};
use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController};
use crate::mcts::search::FixedIterations;
use crate::mcts::{engine_phase_snapshot, MctsConfig, MctsEngine, PreparedIteration};

use crate::games::chess::{chess_quartz, Chess, ChessMove, CHESS_POLICY_ACTIONS};
use crate::games::go::{go_quartz, Go, GoRuleset, GoScoring};
use crate::games::gomoku15::{gomoku15_quartz, Gomoku15, GomokuVariant};
use crate::games::{Gomoku, TicTacToe};

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
    // Try ring buffer first — avoids stdout contention with broker
    if let Some(ring) = global_ring_buffer() {
        let json_bytes = payload.to_string().into_bytes();
        let epoch = ring.epoch();
        // Reclaim DONE slots so we have room to write
        ring.r2p_reclaim();
        if ring.r2p_try_write(SHM_MSG_JSON, &json_bytes, epoch, 0) {
            return;
        }
        // Still full after reclaim — fall through to stdout
    }
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{}", payload);
    let _ = out.flush();
}

fn emit_stdout_json_value(payload: &serde_json::Value) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{}", payload);
    let _ = out.flush();
}

fn emit_ring_message(msg_type: u8, payload: &[u8]) -> bool {
    let Some(ring) = global_ring_buffer() else {
        return false;
    };
    ring.r2p_reclaim();
    ring.r2p_try_write(msg_type, payload, ring.epoch(), 0)
}

const SEARCH_RESP_SINGLE: u8 = 1;
const SEARCH_RESP_MULTI: u8 = 2;
const SEARCH_RESP_SESSION: u8 = 3;
const SHM_MSG_ARENA_EVAL_RESP: u8 = 5;
const QIPC_ARENA_EVAL_RESP: u8 = 9;
const QIPC_ARENA_EVAL_REQ: u8 = 10;
const QIPC_MAGIC: &[u8; 4] = b"QIPC";
const QIPC_HEADER_SIZE: usize = 9;
const ARENA_EVAL_RESP_VERSION: u8 = 2;
const ARENA_OUTCOME_DRAW: u8 = 0;
const ARENA_OUTCOME_BLACK_WIN: u8 = 1;
const ARENA_OUTCOME_WHITE_WIN: u8 = 2;
const ARENA_EVAL_REQ_VERSION: u8 = 2;
const ARENA_STATE_BOARD: u8 = 0;
const ARENA_STATE_GO: u8 = 1;
const ARENA_STATE_CHESS: u8 = 2;

enum SearchResponsePayload {
    Single {
        result: serde_json::Value,
    },
    Multi {
        results: Vec<serde_json::Value>,
    },
    Session {
        session_id: u64,
        results: Vec<serde_json::Value>,
    },
}

impl SearchResponsePayload {
    fn kind(&self) -> u8 {
        match self {
            SearchResponsePayload::Single { .. } => SEARCH_RESP_SINGLE,
            SearchResponsePayload::Multi { .. } => SEARCH_RESP_MULTI,
            SearchResponsePayload::Session { .. } => SEARCH_RESP_SESSION,
        }
    }

    fn session_id(&self) -> u64 {
        match self {
            SearchResponsePayload::Session { session_id, .. } => *session_id,
            SearchResponsePayload::Single { .. } | SearchResponsePayload::Multi { .. } => 0,
        }
    }

    fn results(&self) -> &[serde_json::Value] {
        match self {
            SearchResponsePayload::Single { result } => std::slice::from_ref(result),
            SearchResponsePayload::Multi { results } => results,
            SearchResponsePayload::Session { results, .. } => results,
        }
    }

    fn to_json_value(&self) -> serde_json::Value {
        match self {
            SearchResponsePayload::Single { result } => serde_json::json!({ "result": result }),
            SearchResponsePayload::Multi { results } => serde_json::json!({ "results": results }),
            SearchResponsePayload::Session {
                session_id,
                results,
            } => serde_json::json!({
                "session_id": session_id,
                "results": results,
            }),
        }
    }
}

enum SearchCommandReply {
    Search(SearchResponsePayload),
    Json(serde_json::Value),
}

enum EvalCommandReply {
    Json(String),
    Binary(Vec<u8>),
}

enum ServerCommandInput {
    Json(String),
    Frame(u8, Vec<u8>),
    Invalid(String),
}

struct ArenaEvalFrameRequest {
    game: String,
    search_options: ArenaEvalSearchOptions,
    iters: u32,
    max_moves: usize,
    sessions: Vec<ArenaEvalSessionSpec>,
}

#[derive(Clone)]
struct ArenaEvalSearchOptions {
    search_profile: SearchProfile,
    overrides: SearchOverrides,
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
}

#[derive(Clone)]
enum ArenaEvalStateSpec {
    Board {
        player: i8,
        board: Vec<i8>,
    },
    Go {
        player: i8,
        board: Vec<u8>,
        ruleset: GoRuleset,
        scoring: GoScoring,
        komi: f32,
        allow_suicide: bool,
        passes: u8,
        ko_point: Option<u16>,
        black_caps: u16,
        white_caps: u16,
    },
    Chess {
        fen: String,
        history_hashes: Vec<u64>,
    },
}

#[derive(Clone)]
struct ArenaEvalSessionSpec {
    game_id: String,
    black_tag: u32,
    white_tag: u32,
    opening: Vec<usize>,
    seed: Option<u64>,
    ply: usize,
    total_time_ms: f64,
    done: bool,
    state: ArenaEvalStateSpec,
}

fn push_u8(buf: &mut Vec<u8>, value: u8) {
    buf.push(value);
}

fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(buf: &mut Vec<u8>, value: f32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(buf: &mut Vec<u8>, value: f64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_string(buf: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    push_u32(buf, bytes.len().min(u32::MAX as usize) as u32);
    buf.extend_from_slice(bytes);
}

fn encode_search_result_entry(buf: &mut Vec<u8>, result: &serde_json::Value) {
    let Some(obj) = result.as_object() else {
        push_u8(buf, 0b10);
        return;
    };
    let error = obj.get("error").and_then(|v| v.as_str()).unwrap_or("");
    if !error.is_empty() {
        push_u8(buf, 0b01);
        push_string(buf, error);
        return;
    }

    push_u8(buf, 0);
    push_u32(
        buf,
        obj.get("best_move").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
    );
    push_u32(
        buf,
        obj.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
    );
    push_u32(
        buf,
        obj.get("max_pending").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
    );
    push_f32(
        buf,
        obj.get("p_flip").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
    );
    push_f32(
        buf,
        obj.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
    );
    push_f32(
        buf,
        obj.get("sigma_q").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
    );
    push_f32(
        buf,
        obj.get("hbar_eff").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
    );
    push_f32(
        buf,
        obj.get("dup_rate").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
    );
    push_f32(
        buf,
        obj.get("avg_vvalue")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
    );

    let policy = parse_sparse_policy_value(obj.get("policy").unwrap_or(&serde_json::Value::Null));
    push_u32(buf, policy.len().min(u32::MAX as usize) as u32);
    for (idx, prob) in policy {
        push_u32(buf, idx as u32);
        push_f32(buf, prob);
    }

    let history_hashes = obj
        .get("result_history_hashes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| {
                    entry
                        .as_u64()
                        .or_else(|| entry.as_i64().and_then(|v| (v >= 0).then_some(v as u64)))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    push_u32(buf, history_hashes.len().min(u32::MAX as usize) as u32);
    for hash in history_hashes {
        push_u64(buf, hash);
    }

    push_string(
        buf,
        obj.get("stop_reason")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    );
    push_string(
        buf,
        obj.get("best_move_uci")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    );
    push_string(
        buf,
        obj.get("result_fen").and_then(|v| v.as_str()).unwrap_or(""),
    );
    push_string(
        buf,
        &obj.get("search_manifest")
            .and_then(|v| (!v.is_null()).then_some(v))
            .and_then(|v| serde_json::to_string(v).ok())
            .unwrap_or_default(),
    );
    push_string(
        buf,
        &obj.get("realized_budget")
            .and_then(|v| (!v.is_null()).then_some(v))
            .and_then(|v| serde_json::to_string(v).ok())
            .unwrap_or_default(),
    );
    push_string(
        buf,
        &obj.get("controller_summary")
            .and_then(|v| (!v.is_null()).then_some(v))
            .and_then(|v| serde_json::to_string(v).ok())
            .unwrap_or_default(),
    );
}

fn encode_search_response_payload(response: &SearchResponsePayload) -> Vec<u8> {
    let mut payload = Vec::new();
    push_u8(&mut payload, response.kind());
    push_u64(&mut payload, response.session_id());
    push_u32(
        &mut payload,
        response.results().len().min(u32::MAX as usize) as u32,
    );
    for result in response.results() {
        encode_search_result_entry(&mut payload, result);
    }
    payload
}

fn emit_search_response_payload(response: &SearchResponsePayload) -> bool {
    let payload = encode_search_response_payload(response);
    emit_ring_message(SHM_MSG_SEARCH_RESP, &payload)
}

fn emit_search_command_reply(reply: SearchCommandReply) {
    match reply {
        SearchCommandReply::Search(response) => {
            if !emit_search_response_payload(&response) {
                emit_stdout_json_value(&response.to_json_value());
            }
        }
        SearchCommandReply::Json(value) => emit_stdout_json_value(&value),
    }
}

fn emit_binary_frame(frame_kind: u8, payload: &[u8]) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = out.write_all(b"QIPC");
    let _ = out.write_all(&[frame_kind]);
    let _ = out.write_all(&(payload.len().min(u32::MAX as usize) as u32).to_le_bytes());
    if !payload.is_empty() {
        let _ = out.write_all(payload);
    }
    let _ = out.flush();
}

fn emit_eval_command_reply(reply: EvalCommandReply) {
    match reply {
        EvalCommandReply::Json(value) => {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            let _ = writeln!(out, "{}", value);
            let _ = out.flush();
        }
        EvalCommandReply::Binary(payload) => {
            if !emit_ring_message(SHM_MSG_ARENA_EVAL_RESP, &payload) {
                emit_binary_frame(QIPC_ARENA_EVAL_RESP, &payload);
            }
        }
    }
}

fn read_stdin_command() -> Option<ServerCommandInput> {
    loop {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let mut first = [0u8; 1];
        if reader.read(&mut first).ok()? == 0 {
            return None;
        }
        if matches!(first[0], b'\n' | b'\r') {
            continue;
        }
        if matches!(first[0], b'{' | b'[') {
            let mut rest = Vec::new();
            if reader.read_until(b'\n', &mut rest).is_err() {
                return Some(ServerCommandInput::Invalid(
                    "failed to read json command".to_string(),
                ));
            }
            let mut raw = vec![first[0]];
            raw.extend_from_slice(&rest);
            let line = String::from_utf8_lossy(&raw).trim().to_string();
            if line.is_empty() {
                continue;
            }
            return Some(ServerCommandInput::Json(line));
        }
        if first[0] != b'Q' {
            return Some(ServerCommandInput::Invalid(format!(
                "unexpected stdin command prefix: {}",
                first[0]
            )));
        }
        let mut header_rest = [0u8; QIPC_HEADER_SIZE - 1];
        if reader.read_exact(&mut header_rest).is_err() {
            return Some(ServerCommandInput::Invalid(
                "failed to read QIPC header".to_string(),
            ));
        }
        let mut header = [0u8; QIPC_HEADER_SIZE];
        header[0] = first[0];
        header[1..].copy_from_slice(&header_rest);
        if &header[..4] != QIPC_MAGIC {
            return Some(ServerCommandInput::Invalid(
                "unexpected IPC frame magic".to_string(),
            ));
        }
        let frame_kind = header[4];
        let payload_len = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;
        let mut payload = vec![0u8; payload_len];
        if reader.read_exact(&mut payload).is_err() {
            return Some(ServerCommandInput::Invalid(
                "failed to read QIPC payload".to_string(),
            ));
        }
        return Some(ServerCommandInput::Frame(frame_kind, payload));
    }
}

type SparsePolicyEntry = (usize, f32);

fn sparse_policy_from_visits(visits: &[u32]) -> (Vec<SparsePolicyEntry>, u32) {
    let total: u32 = visits.iter().sum();
    let denom = total.max(1) as f32;
    let policy = visits
        .iter()
        .enumerate()
        .filter_map(|(idx, &n)| {
            if n > 0 {
                Some((idx, n as f32 / denom))
            } else {
                None
            }
        })
        .collect();
    (policy, total)
}

fn collect_sparse_policy<G: GameState>(
    engine: &MctsEngine<G>,
    n_actions: usize,
) -> (usize, Vec<SparsePolicyEntry>, u32) {
    let best = engine
        .best_move()
        .map(|mv| engine.root_state().move_to_idx(mv))
        .unwrap_or(0);
    // Phase 7 C (2026-04-26): lock-free slab read.
    let edges = engine.root.read_edges();
    let mut visits = vec![0u32; n_actions];
    for edge in edges.iter() {
        let idx = engine.root_state().move_to_idx(edge.mv);
        if idx < n_actions {
            visits[idx] = visits[idx].saturating_add(edge.n.load(Ordering::Relaxed));
        }
    }
    let (policy, total) = sparse_policy_from_visits(&visits);
    (best, policy, total)
}

fn parse_sparse_policy_entry(value: &serde_json::Value) -> Option<SparsePolicyEntry> {
    match value {
        serde_json::Value::String(entry) => {
            let (idx_raw, prob_raw) = entry.split_once(':')?;
            let idx = idx_raw.parse::<usize>().ok()?;
            let prob = prob_raw.parse::<f32>().ok()?;
            Some((idx, prob))
        }
        serde_json::Value::Array(entry) if entry.len() >= 2 => {
            let idx = entry.first()?.as_u64()? as usize;
            let prob = entry.get(1)?.as_f64()? as f32;
            Some((idx, prob))
        }
        _ => None,
    }
}

fn parse_sparse_policy_value(value: &serde_json::Value) -> Vec<SparsePolicyEntry> {
    value
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(parse_sparse_policy_entry)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

// Q10 (audit_codex_20260428.md W'6): JSON parsing helpers were extracted
// to `crate::mcts_server_parsers`. They are pure str→Option transforms
// with no engine-state dependencies; the new home gives them their own
// unit-test module and shrinks the mcts_server review surface.
use crate::mcts_server_parsers::{jarr, jbool, jfloat, jint, jstr};

fn frame_take<'a>(payload: &'a [u8], offset: &mut usize, n: usize) -> Result<&'a [u8], String> {
    if payload.len().saturating_sub(*offset) < n {
        return Err("truncated arena eval request payload".to_string());
    }
    let slice = &payload[*offset..*offset + n];
    *offset += n;
    Ok(slice)
}

fn frame_read_u8(payload: &[u8], offset: &mut usize) -> Result<u8, String> {
    Ok(frame_take(payload, offset, 1)?[0])
}

fn frame_read_u32(payload: &[u8], offset: &mut usize) -> Result<u32, String> {
    let raw = frame_take(payload, offset, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn frame_read_u64(payload: &[u8], offset: &mut usize) -> Result<u64, String> {
    let raw = frame_take(payload, offset, 8)?;
    Ok(u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

fn frame_read_i32(payload: &[u8], offset: &mut usize) -> Result<i32, String> {
    let raw = frame_take(payload, offset, 4)?;
    Ok(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn frame_read_f32(payload: &[u8], offset: &mut usize) -> Result<f32, String> {
    Ok(f32::from_bits(frame_read_u32(payload, offset)?))
}

fn frame_read_f64(payload: &[u8], offset: &mut usize) -> Result<f64, String> {
    Ok(f64::from_bits(frame_read_u64(payload, offset)?))
}

fn frame_read_string(payload: &[u8], offset: &mut usize) -> Result<String, String> {
    let byte_len = frame_read_u32(payload, offset)? as usize;
    let raw = frame_take(payload, offset, byte_len)?;
    String::from_utf8(raw.to_vec()).map_err(|_| "invalid utf-8 in arena eval request".to_string())
}

fn decode_arena_eval_request_payload(payload: &[u8]) -> Result<ArenaEvalFrameRequest, String> {
    let mut offset = 0usize;
    let version = frame_read_u8(payload, &mut offset)?;
    let game = frame_read_string(payload, &mut offset)?;
    let search_options_json_v1 = if version == 1 {
        Some(frame_read_string(payload, &mut offset)?)
    } else {
        None
    };
    let typed_search_options_v2 = if version == 1 {
        None
    } else if version == 2 || version == 3 {
        let search_profile = frame_read_string(payload, &mut offset)?;
        let penalty_mode = frame_read_string(payload, &mut offset)?;
        let vl_mode = frame_read_string(payload, &mut offset)?;
        let n_threads = frame_read_u32(payload, &mut offset)?;
        let batch_size = frame_read_u32(payload, &mut offset)?;
        let batch_timeout_us = frame_read_u32(payload, &mut offset)?;
        let hbar_penalty_cap = frame_read_f32(payload, &mut offset)?;
        let sigma_0 = frame_read_f32(payload, &mut offset)?;
        let min_visits = frame_read_u32(payload, &mut offset)?;
        let check_interval = frame_read_u32(payload, &mut offset)?;
        let prior_refresh_rate = frame_read_f32(payload, &mut offset)?;
        let prior_refresh_temp = frame_read_f32(payload, &mut offset)?;
        let c_puct = frame_read_f32(payload, &mut offset)?;
        let root_only_raw = frame_take(payload, &mut offset, 1)?[0] as i8;
        let tt_enabled_raw = frame_take(payload, &mut offset, 1)?[0] as i8;
        let seed_present = frame_read_u8(payload, &mut offset)? != 0;
        let seed = frame_read_u64(payload, &mut offset)?;
        Some(ArenaEvalSearchOptions {
            search_profile: match search_profile.as_str() {
                "baseline" => SearchProfile::Baseline,
                "baseline_strict" => SearchProfile::BaselineStrict,
                _ => SearchProfile::Quartz,
            },
            overrides: SearchOverrides {
                penalty_mode: parse_penalty_mode(&penalty_mode),
                hbar_penalty_cap: if hbar_penalty_cap > 0.0 {
                    Some(hbar_penalty_cap)
                } else {
                    None
                },
                c_puct: if c_puct > 0.0 { Some(c_puct) } else { None },
                sigma_0: if sigma_0 > 0.0 { Some(sigma_0) } else { None },
                min_visits: Some(min_visits.max(1)),
                check_interval: Some(check_interval.max(1)),
                prior_refresh_rate: if prior_refresh_rate >= 0.0 {
                    Some(prior_refresh_rate)
                } else {
                    None
                },
                prior_refresh_temp: if prior_refresh_temp >= 0.0 {
                    Some(prior_refresh_temp)
                } else {
                    None
                },
                root_only_shaping: if root_only_raw >= 0 {
                    Some(root_only_raw != 0)
                } else {
                    None
                },
                vl_mode: if vl_mode.is_empty() {
                    None
                } else {
                    Some(vl_mode)
                },
                tt_enabled: if tt_enabled_raw >= 0 {
                    Some(tt_enabled_raw != 0)
                } else {
                    None
                },
                seed: if seed_present { Some(seed) } else { None },
                halt_mode: if version >= 3 {
                    let halt_mode = frame_read_string(payload, &mut offset)?;
                    parse_halt_mode_override(&format!(r#"{{"halt_mode":"{}"}}"#, halt_mode))
                } else {
                    None
                },
            },
            n_threads: cap_search_threads(n_threads.max(1) as usize),
            batch_size: (batch_size as usize).max(1),
            batch_timeout_us: (batch_timeout_us.max(1)) as u64,
        })
    } else {
        return Err(format!(
            "unsupported arena eval request version: {}",
            version
        ));
    };
    let iters = frame_read_u32(payload, &mut offset)?;
    let max_moves = frame_read_u32(payload, &mut offset)? as usize;
    let session_count = frame_read_u32(payload, &mut offset)? as usize;
    let mut sessions = Vec::with_capacity(session_count);
    for _ in 0..session_count {
        let game_id = frame_read_string(payload, &mut offset)?;
        let black_tag = frame_read_u32(payload, &mut offset)?;
        let white_tag = frame_read_u32(payload, &mut offset)?;
        let seed_raw = frame_read_u64(payload, &mut offset)?;
        let ply = frame_read_u32(payload, &mut offset)? as usize;
        let total_time_ms = frame_read_f64(payload, &mut offset)?;
        let done = frame_read_u8(payload, &mut offset)? != 0;
        let opening_len = frame_read_u32(payload, &mut offset)? as usize;
        let mut opening = Vec::with_capacity(opening_len);
        for _ in 0..opening_len {
            opening.push(frame_read_u32(payload, &mut offset)? as usize);
        }
        let state_kind = frame_read_u8(payload, &mut offset)?;
        let state = match state_kind {
            ARENA_STATE_CHESS => {
                let fen = frame_read_string(payload, &mut offset)?;
                let history_len = frame_read_u32(payload, &mut offset)? as usize;
                let mut history_hashes = Vec::with_capacity(history_len);
                for _ in 0..history_len {
                    history_hashes.push(frame_read_u64(payload, &mut offset)?);
                }
                ArenaEvalStateSpec::Chess {
                    fen,
                    history_hashes,
                }
            }
            ARENA_STATE_GO => {
                let player = frame_read_i32(payload, &mut offset)? as i8;
                let board_len = frame_read_u32(payload, &mut offset)? as usize;
                let board = frame_take(payload, &mut offset, board_len)?.to_vec();
                let go_ruleset = match frame_read_string(payload, &mut offset)?.as_str() {
                    "japanese" | "jp" => GoRuleset::Japanese,
                    "korean" | "kr" => GoRuleset::Korean,
                    "chinese" | "cn" => GoRuleset::Chinese,
                    _ => GoRuleset::Chinese,
                };
                let go_scoring = match frame_read_string(payload, &mut offset)?.as_str() {
                    "territory" => GoScoring::Territory,
                    "area" => GoScoring::Area,
                    _ => go_ruleset.scoring(),
                };
                let go_komi = frame_read_f32(payload, &mut offset)?;
                let go_allow_suicide = frame_read_u8(payload, &mut offset)? != 0;
                let passes = frame_read_u32(payload, &mut offset)?.min(2) as u8;
                let ko_point_raw = frame_read_u32(payload, &mut offset)?;
                let black_caps = frame_read_u32(payload, &mut offset)? as u16;
                let white_caps = frame_read_u32(payload, &mut offset)? as u16;
                ArenaEvalStateSpec::Go {
                    player,
                    board,
                    ruleset: go_ruleset,
                    scoring: go_scoring,
                    komi: go_komi,
                    allow_suicide: go_allow_suicide,
                    passes,
                    ko_point: if ko_point_raw == u32::MAX {
                        None
                    } else {
                        Some(ko_point_raw as u16)
                    },
                    black_caps,
                    white_caps,
                }
            }
            ARENA_STATE_BOARD => {
                let player = frame_read_i32(payload, &mut offset)? as i8;
                let board_len = frame_read_u32(payload, &mut offset)? as usize;
                let board = frame_take(payload, &mut offset, board_len)?
                    .iter()
                    .map(|value| i8::from_le_bytes([*value]))
                    .collect::<Vec<_>>();
                ArenaEvalStateSpec::Board { player, board }
            }
            other => {
                return Err(format!("unknown arena eval state kind: {}", other));
            }
        };
        sessions.push(ArenaEvalSessionSpec {
            game_id,
            black_tag,
            white_tag,
            opening,
            seed: if seed_raw == u64::MAX {
                None
            } else {
                Some(seed_raw)
            },
            ply,
            total_time_ms,
            done,
            state,
        });
    }
    if offset != payload.len() {
        return Err("arena eval request trailing bytes".to_string());
    }
    let search_options = if let Some(options_json) = search_options_json_v1 {
        arena_eval_search_options_from_json(&game, &options_json, session_count)
    } else {
        typed_search_options_v2.expect("v2 typed arena options must exist")
    };
    Ok(ArenaEvalFrameRequest {
        game,
        search_options,
        iters,
        max_moves,
        sessions,
    })
}

fn json_u64ish(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|v| (v >= 0).then_some(v as u64)))
        .or_else(|| {
            value.as_str().and_then(|raw| {
                let trimmed = raw.trim();
                if let Some(hex) = trimmed
                    .strip_prefix("0x")
                    .or_else(|| trimmed.strip_prefix("0X"))
                {
                    u64::from_str_radix(hex, 16).ok()
                } else {
                    trimmed.parse::<u64>().ok()
                }
            })
        })
}

fn f_or(v: f32, d: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        d
    }
}

fn chess_policy_index(state: &Chess, mv: ChessMove) -> usize {
    state.move_to_idx(mv)
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
        .map(|mv| chess_policy_index(&state, *mv).to_string())
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
            "\"history_hashes\":{},",
            "\"legal_moves\":[{}],",
            "\"legal_actions\":[{}]",
            "}}"
        ),
        state.to_fen(),
        if state.current_player() > 0 { "w" } else { "b" },
        if terminal { "true" } else { "false" },
        outcome,
        serde_json::json!(state.history_hashes()).to_string(),
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
        .map(|m| chess_policy_index(&next, *m).to_string())
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
            "\"history_hashes\":{},",
            "\"legal_moves\":[{}],",
            "\"legal_actions\":[{}]",
            "}}"
        ),
        mv.to_uci(),
        next.to_fen(),
        if next.current_player() > 0 { "w" } else { "b" },
        if terminal { "true" } else { "false" },
        outcome,
        serde_json::json!(next.history_hashes()).to_string(),
        legal_moves,
        legal_actions
    )
}

#[derive(Clone)]
struct SearchOverrides {
    penalty_mode: PenaltyMode,
    hbar_penalty_cap: Option<f32>,
    c_puct: Option<f32>,
    sigma_0: Option<f32>,
    min_visits: Option<u32>,
    check_interval: Option<u32>,
    prior_refresh_rate: Option<f32>,
    prior_refresh_temp: Option<f32>,
    root_only_shaping: Option<bool>,
    vl_mode: Option<String>,
    tt_enabled: Option<bool>,
    seed: Option<u64>,
    /// P7 (audit_codex_20260425.md W2): per-request halt_mode override.
    /// "fixed" disables all adaptive halts (P_flip / VOC / ConfAdaptive)
    /// so attribution presets can ensure same-budget fairness across
    /// rows that vary penalty mode.
    halt_mode: Option<HaltMode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchProfile {
    Quartz,
    Baseline,
    BaselineStrict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchThreadSpec {
    requested_threads: usize,
    auto_policy: Option<AutoThreadPolicy>,
}

impl SearchThreadSpec {
    fn explicit(n_threads: usize) -> Self {
        SearchThreadSpec {
            requested_threads: cap_search_threads(n_threads.max(1)),
            auto_policy: None,
        }
    }

    fn policy_name(&self) -> &'static str {
        match self.auto_policy.map(|p| p.mode) {
            Some(AutoThreadMode::Throughput) => "auto-throughput",
            Some(AutoThreadMode::Quality) => "auto-quality",
            None => "explicit",
        }
    }
}

fn parse_auto_thread_policy_name(raw: &str) -> Option<AutoThreadPolicy> {
    match raw {
        "auto" | "throughput" | "auto-throughput" => Some(AutoThreadPolicy::throughput()),
        "quality" | "auto-quality" => Some(AutoThreadPolicy::quality()),
        _ => None,
    }
}

fn parse_search_thread_spec(line: &str) -> SearchThreadSpec {
    let auto_policy = jstr(line, "n_threads")
        .and_then(parse_auto_thread_policy_name)
        .or_else(|| jstr(line, "thread_policy").and_then(parse_auto_thread_policy_name))
        .or_else(|| jstr(line, "auto_thread_policy").and_then(parse_auto_thread_policy_name));

    if let Some(mut policy) = auto_policy {
        if let Some(cap) = jint(line, "thread_cap")
            .or_else(|| jint(line, "max_threads"))
            .or_else(|| jint(line, "n_threads_cap"))
            .map(|v| v.max(1) as usize)
        {
            policy = policy.with_max_threads(cap_search_threads(cap));
        }
        let requested_threads = policy.max_threads.unwrap_or_else(available_host_threads);
        SearchThreadSpec {
            requested_threads: cap_search_threads(requested_threads),
            auto_policy: Some(policy),
        }
    } else {
        SearchThreadSpec::explicit(jint(line, "n_threads").unwrap_or(1).max(1) as usize)
    }
}

fn parse_halt_mode_override(line: &str) -> Option<HaltMode> {
    // P7 (audit W2): map JSON `halt_mode` strings to a controller halt
    // selection. Only the modes that make sense as ablation-row pins are
    // accepted — `ConfAdaptive` requires its own theta_init / target etc.
    // and is intentionally not exposed through this surface.
    let raw: &str = &jstr(line, "halt_mode")?;
    match raw {
        "fixed" => Some(HaltMode::Fixed { budget: u32::MAX }),
        "simple_threshold" | "SimpleThreshold" => Some(HaltMode::SimpleThreshold),
        "voc" | "VOC" => Some(HaltMode::VOC),
        _ => None,
    }
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
            .filter(|v| *v >= 0.0),
        root_only_shaping: jbool(line, "root_only_shaping"),
        vl_mode: jstr(line, "vl_mode").map(|s| s.to_string()),
        tt_enabled: jbool(line, "tt_enabled"),
        seed: jint(line, "seed").map(|v| v.max(0) as u64),
        halt_mode: parse_halt_mode_override(line),
    }
}

fn parse_search_profile(line: &str) -> SearchProfile {
    match jstr(line, "search_profile").unwrap_or("quartz") {
        "baseline" => SearchProfile::Baseline,
        "baseline_strict" => SearchProfile::BaselineStrict,
        _ => SearchProfile::Quartz,
    }
}

fn arena_eval_search_options_from_json(
    _game: &str,
    line: &str,
    session_count: usize,
) -> ArenaEvalSearchOptions {
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let n_threads = cap_search_threads(jint(line, "n_threads").unwrap_or(1).max(1) as usize);
    let batch_size = (jint(line, "batch_size").unwrap_or(8) as usize).max(1);
    let batch_timeout_us = jint(line, "batch_timeout_us")
        .map(|v| v.max(1) as u64)
        .unwrap_or_else(|| default_batch_timeout_us(n_threads, batch_size, session_count));
    ArenaEvalSearchOptions {
        search_profile,
        overrides,
        n_threads,
        batch_size,
        batch_timeout_us,
    }
}

fn arena_eval_session_specs_from_json(
    _game: &str,
    sessions: &[serde_json::Value],
) -> Vec<ArenaEvalSessionSpec> {
    let go_defaults = parse_go_game(_game).map(|(_, default_ruleset)| default_ruleset);
    let chess_default_960 = _game == "chess960";
    sessions
        .iter()
        .map(|session| {
            let game_id = session
                .get("game_id")
                .and_then(|v| v.as_str())
                .unwrap_or("g0000")
                .to_string();
            let black_tag = session
                .get("black_tag")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let white_tag = session
                .get("white_tag")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u32;
            let opening = session
                .get("opening")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let seed = session.get("seed").and_then(|v| v.as_u64());
            let ply = session.get("ply").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let total_time_ms = session
                .get("total_time_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let done = session
                .get("done")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let state = if is_chess_game_name(_game) || session.get("fen").is_some() {
                let fallback = if chess_default_960 {
                    Chess::from_960(518)
                } else {
                    Chess::standard()
                };
                let default_fen = fallback.to_fen();
                let history_hashes = session
                    .get("chess_history_hashes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>())
                    .unwrap_or_default();
                ArenaEvalStateSpec::Chess {
                    fen: session
                        .get("fen")
                        .and_then(|v| v.as_str())
                        .unwrap_or(default_fen.as_str())
                        .to_string(),
                    history_hashes,
                }
            } else if go_defaults.is_some() || session.get("go_ruleset").is_some() {
                let board = session
                    .get("board")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|v| match v.as_i64().unwrap_or(0) {
                                1 => 1,
                                2 | -1 => 2,
                                _ => 0,
                            } as u8)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let player = session.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
                let default_ruleset = go_defaults.unwrap_or(GoRuleset::Chinese);
                let ruleset = match session
                    .get("go_ruleset")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                {
                    "japanese" | "jp" => GoRuleset::Japanese,
                    "korean" | "kr" => GoRuleset::Korean,
                    "chinese" | "cn" => GoRuleset::Chinese,
                    _ => default_ruleset,
                };
                let scoring = match session
                    .get("go_scoring")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                {
                    "territory" => GoScoring::Territory,
                    "area" => GoScoring::Area,
                    _ => ruleset.scoring(),
                };
                let komi = session
                    .get("go_komi")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(7.5);
                let allow_suicide = session
                    .get("go_allow_suicide")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let passes = session
                    .get("passes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
                    .min(2) as u8;
                let ko_point = session
                    .get("ko_point")
                    .and_then(|v| v.as_i64())
                    .and_then(|v| if v >= 0 { Some(v as u16) } else { None });
                let black_caps = session
                    .get("black_caps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u16;
                let white_caps = session
                    .get("white_caps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u16;
                ArenaEvalStateSpec::Go {
                    player,
                    board,
                    ruleset,
                    scoring,
                    komi,
                    allow_suicide,
                    passes,
                    ko_point,
                    black_caps,
                    white_caps,
                }
            } else {
                let board = session
                    .get("board")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|v| v.as_i64().unwrap_or(0) as i8)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let player = session.get("player").and_then(|v| v.as_i64()).unwrap_or(1) as i8;
                ArenaEvalStateSpec::Board { player, board }
            };
            ArenaEvalSessionSpec {
                game_id,
                black_tag,
                white_tag,
                opening,
                seed,
                ply,
                total_time_ms,
                done,
                state,
            }
        })
        .collect()
}

fn search_profile_name(profile: SearchProfile) -> &'static str {
    match profile {
        SearchProfile::Quartz => "quartz",
        SearchProfile::Baseline => "baseline_shared_substrate",
        SearchProfile::BaselineStrict => "baseline_strict",
    }
}

fn should_use_batch_eval(profile: SearchProfile, n_threads: usize, force_batch: bool) -> bool {
    // When the SHM ring buffer is configured, Python's shm_eval_loop only
    // services the ring — it does not write QIPC frames back over stdin.
    // Falling back to StdioCallbackEval here would deadlock the search
    // worker against Python (the worker reads from stdin, Python is parked
    // on the ring). So whenever the ring is live, force the broker-backed
    // BatchStdioEval path; the broker selects ring vs. stdio internally
    // (mcts/eval.rs:1556 use_ring), so this is correct in both modes.
    force_batch
        || n_threads > 1
        || matches!(profile, SearchProfile::BaselineStrict)
        || global_ring_buffer().is_some()
}

fn note_serial_eval_fallback(profile: SearchProfile, n_threads: usize, n_actions: usize) {
    static WARNED: OnceLock<()> = OnceLock::new();
    rust_server_trace(
        "serial_eval_fallback",
        serde_json::json!({
            "search_profile": search_profile_name(profile),
            "n_threads": n_threads,
            "n_actions": n_actions,
            "reason": "stdio_callback_eval_serializes_all_requests",
        }),
    );
    if WARNED.set(()).is_ok() {
        eprintln!(
            "[warn] using serialized StdioCallbackEval fallback; this path is exploratory and not benchmark-safe"
        );
    }
}

fn search_evaluator_path(
    profile: SearchProfile,
    n_threads: usize,
    force_batch: bool,
) -> &'static str {
    if should_use_batch_eval(profile, n_threads, force_batch) {
        "batch_stdio"
    } else {
        "serial_stdio"
    }
}

fn controller_actuator_coverage(qcfg: Option<&QuartzConfig>) -> serde_json::Value {
    let Some(cfg) = qcfg else {
        return serde_json::json!({
            "controller_present": false,
        });
    };
    let prior_refresh_rate_configured = cfg.prior_refresh_rate > 1e-6;
    let (
        prior_refresh_rate_consumed_by_mode,
        prior_refresh_temp_consumed_by_mode,
        prior_refresh_source,
    ) = match cfg.penalty_mode {
        PenaltyMode::Legacy | PenaltyMode::EffectiveV2 | PenaltyMode::None => (
            true,
            prior_refresh_rate_configured,
            "config_prior_refresh_rate",
        ),
        PenaltyMode::SelfAdaptive => (false, false, "self_adaptive_visit_bayes"),
        PenaltyMode::GatedRefresh => (false, false, "prior_q_divergence_gate"),
        PenaltyMode::GatedRefreshLegacy => (false, true, "pflip_q_refresh"),
        PenaltyMode::PFlipMixture => (false, true, "pflip_mixture"),
    };
    serde_json::json!({
        "controller_present": true,
        "penalty_mode_consumed_by_selection": true,
        "halt_mode_consumed_by_controller": true,
        "root_only_shaping_consumed_by_selection": true,
        "prior_refresh_rate_configured": prior_refresh_rate_configured,
        "prior_refresh_rate_consumed_by_mode": prior_refresh_rate_consumed_by_mode,
        "prior_refresh_rate_inert_for_mode": prior_refresh_rate_configured && !prior_refresh_rate_consumed_by_mode,
        "prior_refresh_temp_consumed_by_mode": prior_refresh_temp_consumed_by_mode,
        "prior_refresh_source": prior_refresh_source,
    })
}

fn attach_search_metadata(
    result: &mut serde_json::Value,
    profile: SearchProfile,
    requested_iteration_limit: u32,
    n_threads: usize,
    evaluator_path: &'static str,
    qcfg: Option<&QuartzConfig>,
) {
    let Some(obj) = result.as_object_mut() else {
        return;
    };
    let realized_iterations = obj.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let stop_reason = obj
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let halt_reason_hist = if stop_reason.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::json!({ stop_reason.clone(): 1 })
    };
    let penalty_mode = qcfg.map(|cfg| format!("{:?}", cfg.penalty_mode));
    let halt_mode = qcfg.map(|cfg| format!("{:?}", cfg.halt_mode));
    let root_only_shaping = qcfg.map(|cfg| cfg.root_only_shaping);
    let prior_refresh_rate = qcfg.map(|cfg| cfg.prior_refresh_rate);
    let prior_refresh_temp = qcfg.map(|cfg| cfg.prior_refresh_temp);
    let refresh_count = obj.get("refresh_count").and_then(|v| v.as_f64());
    let refresh_activated = obj.get("refresh_activated").and_then(|v| v.as_bool());
    let penalty_sum = obj.get("penalty_sum").and_then(|v| v.as_f64());
    let effective_prior_l1 = obj.get("effective_prior_l1").and_then(|v| v.as_f64());
    let selection_root_selects = obj.get("selection_root_selects").and_then(|v| v.as_u64());
    let selection_refresh_selected_count = obj
        .get("selection_refresh_selected_count")
        .and_then(|v| v.as_u64());
    let selection_penalty_abs_sum = obj
        .get("selection_penalty_abs_sum")
        .and_then(|v| v.as_f64());
    let selection_effective_prior_l1_sum = obj
        .get("selection_effective_prior_l1_sum")
        .and_then(|v| v.as_f64());
    let selection_mean_candidate_count = obj
        .get("selection_mean_candidate_count")
        .and_then(|v| v.as_f64());
    let selection_max_candidate_count = obj
        .get("selection_max_candidate_count")
        .and_then(|v| v.as_u64());
    let prior_q_divergence = obj.get("prior_q_divergence").and_then(|v| v.as_f64());
    let requested_threads = obj
        .get("requested_threads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_threads as u64) as usize;
    let effective_threads = obj
        .get("effective_threads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_threads as u64) as usize;
    let thread_policy = obj
        .get("thread_policy")
        .and_then(|v| v.as_str())
        .unwrap_or("explicit")
        .to_string();
    let auto_thread_reason = obj
        .get("auto_thread_reason")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let mut telemetry_missing_fields = Vec::new();
    if refresh_count.is_none() {
        telemetry_missing_fields.push("refresh_count");
    }
    if penalty_sum.is_none() {
        telemetry_missing_fields.push("penalty_sum");
    }
    if selection_root_selects.is_none() {
        telemetry_missing_fields.push("selection_trace");
    }
    // P6 (audit_codex_20260425.md W8): VOC channel decomposition is the
    // mechanism-level signal callers need to falsify "controller helps"
    // claims. The fields below were already populated upstream (engine
    // root_search_summary attaches them), so this block just plumbs them
    // through the controller_summary view.
    let voc_total = obj.get("voc_total").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let voc_focus = obj.get("voc_focus").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let voc_expand = obj
        .get("voc_expand")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let voc_merge = obj.get("voc_merge").and_then(|v| v.as_f64()).unwrap_or(0.0);
    // Q3 (audit_codex_20260428.md W'1): forward the per-game argmax
    // histogram into the controller_summary view so the Python aggregator
    // (replay.py:ReplayMetrics.controller_telemetry_summary) can build a
    // study-wide histogram over rows without reparsing the search-result
    // top level. An empty object means the row never recorded a halt-check
    // (older Rust binary on schema_version 1, or non-Quartz profile).
    let voc_argmax_channel_hist = obj
        .get("voc_argmax_channel_hist")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    // P01: build the `controller_summary.extended` object by reading the
    // flat keys placed by `build_result_value`. We map [u64; 7] →
    // {<PenaltyMode name>: count} and [u32; 10] → {<HaltReason name>: count}
    // so consumers see stable string keys instead of raw indices.
    let pm_invoke_arr = obj
        .get("selection_penalty_mode_invoke_count")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let halt_reason_arr = obj
        .get("halt_reason_count")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let refresh_eligible = obj
        .get("selection_refresh_eligible_count")
        .and_then(|v| v.as_u64());
    let refresh_active = obj
        .get("selection_refresh_active_count")
        .and_then(|v| v.as_u64());
    let mean_prior_refresh_rate: Option<f64> = match (refresh_active, refresh_eligible) {
        (Some(active), Some(eligible)) if eligible > 0 => Some(active as f64 / eligible as f64),
        (Some(_), Some(_)) => None,
        _ => None,
    };
    let mut pm_invoke_map = serde_json::Map::new();
    for (i, key) in crate::mcts::quartz::PENALTY_MODE_KEYS.iter().enumerate() {
        let v = pm_invoke_arr
            .get(i)
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        pm_invoke_map.insert((*key).to_string(), serde_json::json!(v));
    }
    let mut halt_reason_map = serde_json::Map::new();
    let halt_reason_keys: [&'static str; crate::mcts::quartz::HALT_REASON_COUNT] = [
        crate::mcts::quartz::HaltReason::PFlipConverged.as_key(),
        crate::mcts::quartz::HaltReason::VOCNonPositive.as_key(),
        crate::mcts::quartz::HaltReason::FixedBudget.as_key(),
        crate::mcts::quartz::HaltReason::KLLUCBStop.as_key(),
        crate::mcts::quartz::HaltReason::MaxVisits.as_key(),
        crate::mcts::quartz::HaltReason::MaxTime.as_key(),
        crate::mcts::quartz::HaltReason::MinVisitsNotMet.as_key(),
        crate::mcts::quartz::HaltReason::GLRCertified.as_key(),
        crate::mcts::quartz::HaltReason::PolicyConverged.as_key(),
        crate::mcts::quartz::HaltReason::EmpBernsteinSep.as_key(),
    ];
    for (i, key) in halt_reason_keys.iter().enumerate() {
        let v = halt_reason_arr
            .get(i)
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        halt_reason_map.insert((*key).to_string(), serde_json::json!(v));
    }
    let extended_block = serde_json::json!({
        "schema_version": 1,
        "refresh_active_count": refresh_active,
        "refresh_eligible_count": refresh_eligible,
        "mean_prior_refresh_rate": mean_prior_refresh_rate,
        "controller_penalty_mode_counts": serde_json::Value::Object(pm_invoke_map),
        "halt_reason_count": serde_json::Value::Object(halt_reason_map),
    });

    let controller_summary = serde_json::json!({
        // P6+Q3: schema_version pins the wire format. v1 had no argmax
        // channel field; v2 adds `voc_argmax_channel_hist`; v3 adds root
        // snapshot refresh/penalty/effective-prior diagnostics; v4 adds
        // actual root-selection path telemetry; v5 adds controller actuator
        // coverage so configured-but-inert knobs are visible.
        // P01: v6 adds `extended` block carrying
        // controller_penalty_mode_counts, mean_prior_refresh_rate,
        // halt_reason_count — fields previously claimed in README but
        // never emitted from Rust. Readers must tolerate unknown extra
        // fields and degrade gracefully when a newer field is missing on
        // an older Rust binary.
        "schema_version": 6,
        "p_flip": obj.get("p_flip").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "value": obj.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "sigma_q": obj.get("sigma_q").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "hbar_eff": obj.get("hbar_eff").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "voc_total": voc_total,
        "voc_focus": voc_focus,
        "voc_expand": voc_expand,
        "voc_merge": voc_merge,
        "voc_argmax_channel_hist": voc_argmax_channel_hist,
        "dup_rate": obj.get("dup_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "max_pending": obj.get("max_pending").and_then(|v| v.as_u64()).unwrap_or(0),
        "avg_vvalue": obj.get("avg_vvalue").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "stop_reason": stop_reason,
        "halt_reason_hist": halt_reason_hist,
        "penalty_mode": penalty_mode,
        "halt_mode": halt_mode,
        "root_only_shaping": root_only_shaping,
        "prior_refresh_rate": prior_refresh_rate,
        "prior_refresh_temp": prior_refresh_temp,
        "prior_q_divergence": prior_q_divergence,
        "refresh_count": refresh_count,
        "refresh_activated": refresh_activated,
        "penalty_sum": penalty_sum,
        "effective_prior_l1": effective_prior_l1,
        "selection_trace": {
            "root_selects": selection_root_selects,
            "refresh_selected_count": selection_refresh_selected_count,
            "selected_penalty_abs_sum": selection_penalty_abs_sum,
            "selected_effective_prior_l1_sum": selection_effective_prior_l1_sum,
            "selected_mean_candidate_count": selection_mean_candidate_count,
            "selected_max_candidate_count": selection_max_candidate_count,
        },
        "refresh_metric_present": refresh_count.is_some(),
        "penalty_metric_present": penalty_sum.is_some(),
        "telemetry_partial": !telemetry_missing_fields.is_empty(),
        "telemetry_missing_fields": telemetry_missing_fields,
        "actuator_coverage": controller_actuator_coverage(qcfg),
        // P01: see the schema_version comment above for rationale.
        "extended": extended_block,
    });
    obj.insert(
        "search_manifest".to_string(),
        serde_json::json!({
            "profile": search_profile_name(profile),
            "requested_iteration_limit": requested_iteration_limit,
            "n_threads": n_threads,
            "requested_threads": requested_threads,
            "effective_threads": effective_threads,
            "thread_policy": thread_policy,
            "auto_thread_reason": auto_thread_reason,
            "evaluator_path": evaluator_path,
            "benchmark_safe": evaluator_path != "serial_stdio",
        }),
    );
    obj.insert(
        "realized_budget".to_string(),
        serde_json::json!({
            "requested_iteration_limit": requested_iteration_limit,
            "realized_iterations": realized_iterations,
            "stop_reason": stop_reason,
        }),
    );
    obj.insert("controller_summary".to_string(), controller_summary);
}

fn apply_search_profile(mut cfg: MctsConfig, profile: SearchProfile) -> MctsConfig {
    match profile {
        SearchProfile::Quartz => {}
        SearchProfile::Baseline => {
            cfg.quartz = None;
            cfg.gvoc = None; // GVOC is part of the QUARTZ search controller bundle
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
    // P05: auto-load σ₀ calibration from $QUARTZ_CALIBRATION_DIR if set.
    // Only applies when the profile retains a Quartz config — Baseline*
    // profiles disabled it above. Diagnostics are eprintln'd so they
    // surface in the train_log without artifact inspection.
    if let Some(quartz_cfg) = cfg.quartz.take() {
        let cal_dir = std::env::var_os("QUARTZ_CALIBRATION_DIR")
            .map(std::path::PathBuf::from);
        let game_label = std::env::var_os("QUARTZ_CALIBRATION_GAME")
            .and_then(|v| v.into_string().ok());
        let strength = std::env::var_os("QUARTZ_CALIBRATION_STRENGTH")
            .and_then(|v| v.into_string().ok())
            .map(|s| match s.as_str() {
                "weak" => crate::mcts::quartz::EvalStrength::Weak,
                "medium" => crate::mcts::quartz::EvalStrength::Medium,
                _ => crate::mcts::quartz::EvalStrength::Strong,
            });
        let updated = if let Some(dir) = cal_dir {
            let (next, diags) = quartz_cfg.with_calibration(
                &dir,
                game_label.as_deref(),
                strength,
                2.0,
            );
            for d in diags {
                match d {
                    crate::mcts::quartz::CalibrationDiagnostic::Info(msg) => {
                        eprintln!("[quartz][calibration] INFO: {msg}");
                    }
                    crate::mcts::quartz::CalibrationDiagnostic::Warn(msg) => {
                        eprintln!("[quartz][calibration] WARN: {msg}");
                    }
                }
            }
            next
        } else {
            quartz_cfg
        };
        cfg.quartz = Some(updated);
    }
    cfg
}

fn apply_search_overrides(mut cfg: MctsConfig, ov: &SearchOverrides) -> MctsConfig {
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
        if let Some(root_only_shaping) = ov.root_only_shaping {
            q.root_only_shaping = root_only_shaping;
        }
        if let Some(halt_mode) = ov.halt_mode.clone() {
            q.halt_mode = halt_mode;
        }
    }
    // VL mode override (independent of search profile)
    if let Some(ref vl) = ov.vl_mode {
        use crate::mcts::parallel::VlMode;
        cfg.vl_mode = match vl.as_str() {
            "disabled" => VlMode::Disabled,
            "fixed" => VlMode::Fixed,
            "adaptive" => VlMode::Adaptive,
            "vvisit_only" => VlMode::VvisitOnly,
            "vvalue_only" => VlMode::VvalueOnly,
            _ => cfg.vl_mode,
        };
    }
    if let Some(tt_enabled) = ov.tt_enabled {
        cfg.tt_enabled = tt_enabled;
    }
    if let Some(seed) = ov.seed {
        cfg.seed = Some(seed);
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
    G::Move: PartialEq,
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

        // Visit distribution. Phase 7 C: lock-free slab read.
        let edges = engine.root.read_edges();
        let mut visits = vec![0u32; num_actions];
        let mut total = 0u32;
        for e in edges.iter() {
            let idx = engine.root_state().move_to_idx(e.mv);
            let n = e.n.load(Ordering::Relaxed);
            if idx < num_actions {
                visits[idx] = n;
                total += n;
            }
        }

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

        let sparse_policy: Vec<SparsePolicyEntry> = policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-6)
            .map(|(i, p)| (i, *p))
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

        positions.push(
            serde_json::json!({
                "pl": engine.root_state().current_player(),
                "bd": board_str,
                "pol": sparse_policy,
                "pf": pf,
                "sq": sq,
                "hb": hb,
            })
            .to_string(),
        );

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
                let idx = engine.root_state().move_to_idx(mv);
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
                let i = engine.root_state().move_to_idx(mv);
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
        "GatedRefreshLegacy" => PenaltyMode::GatedRefreshLegacy,
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
    jint(line, "chess960_index").map(|v| v.clamp(0, 959) as u16)
}

fn apply_chess_history_from_json(state: &mut Chess, value: &serde_json::Value) {
    if let Some(history) = value.get("chess_history_hashes").and_then(|v| v.as_array()) {
        let hashes = history.iter().filter_map(json_u64ish).collect::<Vec<_>>();
        state.set_history_hashes(&hashes);
        return;
    }
    if let Some(history) = value.get("chess_history_keys").and_then(|v| v.as_array()) {
        let keys = history
            .iter()
            .filter_map(|entry| entry.as_str().map(str::to_string))
            .collect::<Vec<_>>();
        if !keys.is_empty() {
            let _ = state.set_history_keys(&keys);
        }
    }
}

fn chess_state_from_json(root: &serde_json::Value, default_960: bool) -> Chess {
    let fallback = || {
        if default_960 {
            if let Some(idx) = root
                .get("chess960_index")
                .and_then(json_u64ish)
                .map(|v| v.min(959) as u16)
            {
                Chess::from_960(idx)
            } else if root
                .get("chess960_random_start")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                Chess::from_960(rand::random::<u16>() % 960)
            } else {
                Chess::from_960(518)
            }
        } else if let Some(idx) = root
            .get("chess960_index")
            .and_then(json_u64ish)
            .map(|v| v.min(959) as u16)
        {
            Chess::from_960(idx)
        } else {
            Chess::standard()
        }
    };
    let mut state = if let Some(fen) = root.get("fen").and_then(|v| v.as_str()) {
        Chess::from_fen(fen).unwrap_or_else(|_| fallback())
    } else {
        fallback()
    };
    apply_chess_history_from_json(&mut state, root);
    state
}

fn chess_state_from_request(line: &str, default_960: bool) -> Chess {
    if let Ok(root) = serde_json::from_str::<serde_json::Value>(line) {
        chess_state_from_json(&root, default_960)
    } else if default_960 {
        if let Some(idx) = parse_chess960_index(line) {
            Chess::from_960(idx)
        } else {
            Chess::from_960(518)
        }
    } else {
        Chess::standard()
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
    jfloat(line, "go_komi")
        .map(|v| v as f32)
        .unwrap_or(fallback)
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
                let cfg = apply_search_overrides(cfg, &overrides);
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
                let cfg = apply_search_overrides(gomoku15_quartz(variant), &overrides);
                selfplay_one(Gomoku15::new(variant), move || cfg.clone(), iters, tt, 225)
            }
            _ if parse_go_game(game).is_some() => {
                let (size, default_ruleset) = parse_go_game(game).unwrap();
                let ruleset = parse_go_ruleset(line, default_ruleset);
                let scoring = parse_go_scoring(line, ruleset.scoring());
                let komi = parse_go_komi(line, if size == 19 { 7.5 } else { 7.5 });
                let allow_suicide = parse_go_allow_suicide(line, false);
                let cfg = apply_search_overrides(go_quartz(size), &overrides);
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
                    &overrides,
                );
                selfplay_one(TicTacToe::initial(), move || cfg.clone(), iters, tt, 9)
            }
            game if is_chess_game_name(game) => {
                let cfg = apply_search_overrides(chess_quartz(), &overrides);
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
                    apply_search_overrides(gomoku15_quartz(GomokuVariant::Freestyle), &overrides);
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
    // Honor search_profile / penalty_mode / vl_mode / etc. overrides so the
    // regression test can pin per-mode behavior from the JSON request.
    let overrides = parse_search_overrides(line);
    let config = apply_search_overrides(gomoku15_quartz(variant), &overrides);
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
    let ko_point = if ko_point_raw >= 0 {
        Some(ko_point_raw as u16)
    } else {
        None
    };
    let black_caps = jint(line, "black_caps").unwrap_or(0).max(0) as u16;
    let white_caps = jint(line, "white_caps").unwrap_or(0).max(0) as u16;
    let state = if board.is_empty() {
        Go::new_with_options(size, komi, ruleset, scoring, allow_suicide)
    } else {
        Go::from_board_with_options(
            size,
            komi,
            &board,
            player,
            ruleset,
            scoring,
            allow_suicide,
            passes,
            ko_point,
            black_caps,
            white_caps,
        )
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
        let Some(input) = read_stdin_command() else {
            break;
        };

        // Bump ring buffer epoch at command start so Python ignores stale slots
        if let Some(ring) = global_ring_buffer() {
            ring.bump_epoch();
        }

        let line = match input {
            ServerCommandInput::Invalid(err) => {
                emit_stdout_json_value(&serde_json::json!({ "error": err }));
                if let Some(ring) = global_ring_buffer() {
                    ring.set_cmd_done(true);
                }
                continue;
            }
            ServerCommandInput::Frame(frame_kind, payload) => {
                if frame_kind == QIPC_ARENA_EVAL_REQ {
                    emit_eval_command_reply(handle_eval_nn_run_frame(&payload));
                } else {
                    emit_stdout_json_value(&serde_json::json!({
                        "error": format!("unsupported QIPC command frame kind: {}", frame_kind)
                    }));
                }
                if let Some(ring) = global_ring_buffer() {
                    ring.set_cmd_done(true);
                }
                continue;
            }
            ServerCommandInput::Json(line) => line,
        };

        let cmd = jstr(&line, "cmd").unwrap_or("move");
        if cmd == "quit" {
            break;
        }

        if cmd == "search_nn" {
            emit_search_command_reply(handle_search_nn(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "search_nn_multi" {
            emit_search_command_reply(handle_search_nn_multi(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "search_nn_multi_session_open" {
            emit_search_command_reply(handle_search_nn_multi_session_open(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "search_nn_multi_engine_session_open" {
            emit_search_command_reply(handle_search_nn_multi_engine_session_open(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "search_nn_multi_session_step" {
            emit_search_command_reply(handle_search_nn_multi_session_step(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "search_nn_multi_engine_session_step" {
            emit_search_command_reply(handle_search_nn_multi_engine_session_step(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
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
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "search_nn_multi_engine_session_close" {
            let resp = handle_search_nn_multi_session_close(&line);
            {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{}", resp);
                let _ = out.flush();
            }
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }
        if cmd == "eval_nn_run" {
            emit_eval_command_reply(handle_eval_nn_run(&line));
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
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
            if let Some(ring) = global_ring_buffer() {
                ring.set_cmd_done(true);
            }
            continue;
        }

        let resp = match cmd {
            "selfplay" => handle_selfplay(&line),
            "chess_state" => handle_chess_state(
                &line,
                jstr(&line, "game").unwrap_or("chess960") == "chess960",
            ),
            "chess_apply" => handle_chess_apply(
                &line,
                jstr(&line, "game").unwrap_or("chess960") == "chess960",
            ),
            _ => {
                let game = jstr(&line, "game").unwrap_or("gomoku15");
                let iters = jint(&line, "iters").unwrap_or(200) as u32;
                match game {
                    _ if parse_gomoku15_variant(game).is_some() => {
                        search_gomoku15(&line, parse_gomoku15_variant(game).unwrap(), iters)
                    }
                    game if is_chess_game_name(game) => {
                        search_chess(&line, game == "chess960", iters)
                    }
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
        if let Some(ring) = global_ring_buffer() {
            ring.set_cmd_done(true);
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
    search_root_visits: Vec<u32>,
    search_p_flip: Vec<f32>,
    search_halt_reasons: std::collections::BTreeMap<String, u32>,
    selection_root_selects: u64,
    selection_refresh_selected_count: u64,
    selection_penalty_abs_sum: f64,
    selection_effective_prior_l1_sum: f64,
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
            "search_summary": eval_session_search_summary(sess),
    })
}

fn eval_session_search_summary<G: GameState>(sess: &EvalRunnerSession<G>) -> serde_json::Value {
    let count = sess.search_root_visits.len();
    let mean_root_visits = if count > 0 {
        sess.search_root_visits
            .iter()
            .map(|v| *v as f64)
            .sum::<f64>()
            / count as f64
    } else {
        0.0
    };
    let max_root_visits = sess.search_root_visits.iter().copied().max().unwrap_or(0);
    let mean_p_flip = if sess.search_p_flip.is_empty() {
        0.0
    } else {
        sess.search_p_flip.iter().map(|v| *v as f64).sum::<f64>() / sess.search_p_flip.len() as f64
    };
    serde_json::json!({
        "moves": count,
        "root_visits": {
            "mean": mean_root_visits,
            "max": max_root_visits,
            "samples": &sess.search_root_visits,
        },
        "mean_p_flip": mean_p_flip,
        "halt_reason_hist": &sess.search_halt_reasons,
        "selection_trace": {
            "root_selects": sess.selection_root_selects,
            "refresh_selected_count": sess.selection_refresh_selected_count,
            "selected_penalty_abs_sum": sess.selection_penalty_abs_sum,
            "selected_effective_prior_l1_sum": sess.selection_effective_prior_l1_sum,
        },
    })
}

fn arena_eval_outcome<G: GameState>(sess: &EvalRunnerSession<G>) -> (u8, f32) {
    match terminal_black_score(&sess.state) {
        Some(1.0) => (ARENA_OUTCOME_BLACK_WIN, 1.0),
        Some(0.0) => (ARENA_OUTCOME_WHITE_WIN, 0.0),
        Some(_) => (ARENA_OUTCOME_DRAW, 0.5),
        None => (ARENA_OUTCOME_DRAW, 0.5),
    }
}

fn encode_arena_eval_record<G: GameState>(buf: &mut Vec<u8>, sess: &EvalRunnerSession<G>) {
    let (outcome_code, score_black) = arena_eval_outcome(sess);
    push_string(buf, &sess.game_id);
    push_u32(buf, sess.black_tag);
    push_u32(buf, sess.white_tag);
    push_u8(buf, outcome_code);
    push_u8(buf, if sess.error.is_some() { 1 } else { 0 });
    push_f32(buf, score_black);
    push_u32(buf, sess.ply.min(u32::MAX as usize) as u32);
    push_f64(buf, sess.total_time_ms);
    push_u64(buf, sess.seed.unwrap_or(u64::MAX));
    push_u32(buf, sess.opening.len().min(u32::MAX as usize) as u32);
    for &mv in &sess.opening {
        push_u32(buf, mv.min(u32::MAX as usize) as u32);
    }
    push_string(buf, sess.error.as_deref().unwrap_or(""));
    push_string(buf, &eval_session_search_summary(sess).to_string());
}

fn encode_arena_eval_response_payload<G: GameState>(
    game_name: &str,
    sessions: &[EvalRunnerSession<G>],
    duration_ms: f64,
) -> Vec<u8> {
    let mut payload = Vec::new();
    push_u8(&mut payload, ARENA_EVAL_RESP_VERSION);
    push_u8(&mut payload, 1);
    push_u32(&mut payload, sessions.len().min(u32::MAX as usize) as u32);
    push_f64(&mut payload, duration_ms);
    push_string(&mut payload, game_name);
    push_u32(&mut payload, sessions.len().min(u32::MAX as usize) as u32);
    for sess in sessions {
        encode_arena_eval_record(&mut payload, sess);
    }
    payload
}

/// Lookup action-space size for a game name.
fn game_n_actions(game: &str) -> usize {
    match game {
        "gomoku7" => 49,
        _ if parse_gomoku15_variant(game).is_some() => 225,
        _ if parse_go_game(game).is_some() => {
            let (size, _) = parse_go_game(game).unwrap();
            size * size + 1
        }
        "tictactoe" => 9,
        _ if is_chess_game_name(game) => CHESS_POLICY_ACTIONS,
        _ => 225, // default to gomoku15
    }
}

#[derive(Clone)]
struct SelfplaySession<G: GameState> {
    state: G,
    rng: StdRng,
    moves: usize,
    finished: bool,
    winner: f64,
    board_history: Vec<Vec<i64>>,
    player_history: Vec<i8>,
    policy_history: Vec<Vec<SparsePolicyEntry>>,
    trace_history: Vec<serde_json::Value>,
}

fn choose_selfplay_action_generic<G: GameState>(
    rng: &mut StdRng,
    state: &G,
    policy_entries: &[SparsePolicyEntry],
    move_count: usize,
    temp_threshold: usize,
    fallback_best: usize,
    n_actions: usize,
) -> Option<usize> {
    let legal: Vec<usize> = state
        .legal_moves()
        .into_iter()
        .map(|m| state.move_to_idx(m))
        .collect();
    if legal.is_empty() {
        return None;
    }
    let mut policy = vec![0.0f64; n_actions];
    for &(idx, prob) in policy_entries {
        if idx < policy.len() {
            policy[idx] = f64::from(prob.max(0.0));
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
    legal.into_iter().max_by(|&a, &b| {
        policy[a]
            .partial_cmp(&policy[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// NN-backed single-move search using bidirectional eval protocol.
/// Python sends board state, Rust does MCTS with eval callbacks to Python NN.
/// Batch MCTS search: select K leaves → 1 batch eval → K expand+backprop.
/// Throughput: ~K× fewer IPC round-trips, ~K× better GPU utilization.
/// Run search with appropriate parallelism, then extract result JSON.
fn build_result_value<G: GameState>(
    engine: &MctsEngine<G>,
    n_act: usize,
    outcome: &SearchExecutionOutcome,
    iterations: u32,
    stop_reason: String,
    p_flip: f32,
    value: f32,
    sigma_q: f32,
    hbar_eff: f32,
    prior_q_divergence: Option<f32>,
) -> serde_json::Value {
    let (best, policy, total) = collect_sparse_policy(engine, n_act);
    let tt = engine.tt.contention_snapshot();
    let par = engine.par_ctrl.telemetry.snapshot();
    // Q3: serialize the argmax histogram as a plain JSON object so the
    // Python aggregator (replay.py) can read it directly.
    let argmax_hist_json: serde_json::Map<String, serde_json::Value> = outcome
        .voc_argmax_channel_hist
        .iter()
        .map(|(k, v)| ((*k).to_string(), serde_json::json!(*v)))
        .collect();
    serde_json::json!({
        "best_move": best,
        "policy": policy,
        "p_flip": f_or(p_flip, 0.0),
        "value": f_or(value, 0.0),
        "sigma_q": f_or(sigma_q, 0.0),
        "hbar_eff": f_or(hbar_eff, 0.0),
        "prior_q_divergence": prior_q_divergence.map(|v| f_or(v, 0.0)),
        "stop_reason": stop_reason,
        "iterations": iterations.max(total),
        "requested_threads": outcome.requested_threads,
        "effective_threads": outcome.effective_threads,
        "n_threads": outcome.effective_threads,
        "thread_policy": outcome.thread_policy,
        "auto_thread_reason": outcome.auto_thread_reason.clone(),
        "dup_rate": par.dup_rate,
        "max_pending": par.max_pending,
        "avg_vvalue": par.avg_vvalue,
        "tt_hit_rate": engine.tt.hit_rate(),
        "tt_size": engine.tt.size(),
        "tt_get_or_create_calls": tt.get_or_create_calls,
        "tt_get_calls": tt.get_calls,
        "tt_lock_wait_ms": tt.lock_wait_nanos as f64 / 1_000_000.0,
        "tt_max_lock_wait_ms": tt.max_lock_wait_nanos as f64 / 1_000_000.0,
        // Q3 (audit_codex_20260428.md W'1): VOC channel decomposition is now
        // populated for SearchProfile::Quartz; baseline profiles emit zeros.
        "voc_total": f_or(outcome.voc_total, 0.0),
        "voc_focus": f_or(outcome.voc_focus, 0.0),
        "voc_expand": f_or(outcome.voc_expand, 0.0),
        "voc_merge": f_or(outcome.voc_merge, 0.0),
        "voc_argmax_channel_hist": serde_json::Value::Object(argmax_hist_json),
        "refresh_count": outcome.refresh_count,
        "refresh_activated": outcome.refresh_activated,
        "penalty_sum": f_or(outcome.penalty_sum, 0.0),
        "effective_prior_l1": f_or(outcome.effective_prior_l1, 0.0),
        "selection_root_selects": outcome.selection_root_selects,
        "selection_refresh_selected_count": outcome.selection_refresh_selected_count,
        "selection_penalty_abs_sum": outcome.selection_penalty_abs_sum,
        "selection_effective_prior_l1_sum": outcome.selection_effective_prior_l1_sum,
        "selection_mean_candidate_count": outcome.selection_mean_candidate_count,
        "selection_max_candidate_count": outcome.selection_max_candidate_count,
        // P01: extended controller telemetry — emitted as flat keys here so
        // attach_search_metadata can promote them into
        // controller_summary.extended without a second pass over the engine.
        // `mean_prior_refresh_rate` is computed in attach_search_metadata
        // since it's a derived ratio.
        "selection_penalty_mode_invoke_count": outcome.selection_penalty_mode_invoke_count.to_vec(),
        "selection_refresh_eligible_count": outcome.selection_refresh_eligible_count,
        "selection_refresh_active_count": outcome.selection_refresh_active_count,
        "halt_reason_count": outcome.halt_reason_count.to_vec(),
    })
}

struct SearchExecutionOutcome {
    iterations: u32,
    stop_reason: String,
    requested_threads: usize,
    effective_threads: usize,
    thread_policy: &'static str,
    auto_thread_reason: Option<String>,
    p_flip: f32,
    value: f32,
    sigma_q: f32,
    hbar_eff: f32,
    prior_q_divergence: Option<f32>,
    /// Q3 (audit_codex_20260428.md W'1): VOC channel decomposition from
    /// the controller's last_stats. Populated only for SearchProfile::Quartz;
    /// other profiles report 0.0 (no controller present).
    voc_total: f32,
    voc_focus: f32,
    voc_expand: f32,
    voc_merge: f32,
    refresh_count: u32,
    refresh_activated: bool,
    penalty_sum: f32,
    effective_prior_l1: f32,
    selection_root_selects: u64,
    selection_refresh_selected_count: u64,
    selection_penalty_abs_sum: f64,
    selection_effective_prior_l1_sum: f64,
    selection_mean_candidate_count: f64,
    selection_max_candidate_count: u64,
    /// Q3: per-game histogram of the argmax channel across every halt-check
    /// the controller recorded. Empty for non-Quartz profiles. Lets readers
    /// see whether VOC-halt is dominated by one channel or genuinely
    /// distributed across focus / expand / merge.
    voc_argmax_channel_hist: std::collections::BTreeMap<&'static str, u32>,
    /// P01: per-`PenaltyMode` invocation count, indexed identically to
    /// `quartz::PENALTY_MODE_KEYS`. Sampled at root selects only (one
    /// increment per MCTS root iteration). Empty / all-zero for non-Quartz.
    selection_penalty_mode_invoke_count: [u64; crate::mcts::quartz::PENALTY_MODE_COUNT],
    /// P01: # of root selects where the active mode's refresh path was
    /// eligible to fire. Used to compute `mean_prior_refresh_rate`.
    selection_refresh_eligible_count: u64,
    /// P01: # of root selects where refresh actually fired (effective_prior
    /// drifted from raw prior). Always ≤ `selection_refresh_eligible_count`.
    selection_refresh_active_count: u64,
    /// P01: per-`HaltReason` terminal counts from `QuartzController`.
    /// Indexed identically to `quartz::HaltReason` (offsets 0..HALT_REASON_COUNT).
    /// All-zero for non-Quartz profiles (no controller).
    halt_reason_count: [u32; crate::mcts::quartz::HALT_REASON_COUNT],
}

#[derive(Default)]
struct ControllerDiagnostics {
    refresh_count: u32,
    refresh_activated: bool,
    penalty_sum: f32,
    effective_prior_l1: f32,
}

fn blended_prior_diag(prior: f32, signal: f32, rho: f32) -> f32 {
    ((1.0 - rho) * prior.max(1e-8).ln() + rho * signal.max(1e-8).ln())
        .exp()
        .max(1e-8)
}

fn controller_diagnostics<G: GameState>(
    engine: &MctsEngine<G>,
    stats: &crate::mcts::quartz::QuartzStats,
    qcfg: &QuartzConfig,
) -> ControllerDiagnostics
where
    G::Move: Copy + Send + Sync + 'static,
{
    use crate::mcts::quartz::PenaltyMode;

    let mut diag = ControllerDiagnostics::default();
    let n = engine.root.materialized_count();
    let edges = engine.root.edge_snapshot(n);
    if edges.is_empty() {
        return diag;
    }
    let root_visits = (stats.root_visits as f32).max(1.0);
    let k = (stats.n_visible as f32).max(2.0);
    let n_avg = root_visits / k;
    for edge in edges {
        let n_raw = edge.n;
        let prior = edge.p.max(1e-8);
        let q_eff = edge.q();
        let mut effective_prior = prior;
        let penalty = match qcfg.penalty_mode {
            PenaltyMode::SelfAdaptive => {
                if n_raw > 0 {
                    let n_a = n_raw as f32;
                    let alpha_a = n_a / (n_a + k);
                    let tau = (1.0 + n_avg).ln().max(0.1);
                    let log_visit = (1.0 + n_a).ln();
                    effective_prior = ((1.0 - alpha_a) * prior.ln() + alpha_a * log_visit / tau)
                        .exp()
                        .max(1e-8);
                    -stats.sigma_q.max(0.001) / (1.0 + n_a)
                } else {
                    0.0
                }
            }
            PenaltyMode::GatedRefresh => {
                if n_raw > 0 && stats.root_visits > 0 {
                    let gate = stats.epsilon_t.max(1e-6);
                    let divergence = stats.prior_q_divergence.max(0.0);
                    if divergence > gate {
                        let rho = ((divergence - gate) / divergence.max(gate)).clamp(0.0, 1.0);
                        let visit_share = (n_raw as f32 / root_visits).max(1e-8);
                        effective_prior = blended_prior_diag(prior, visit_share, rho);
                    }
                }
                -qcfg.hbar_penalty_cap.min(stats.hbar_eff) * (n_raw as f32 / root_visits)
            }
            PenaltyMode::GatedRefreshLegacy => {
                if n_raw > 0 {
                    let rho = 0.3_f32 * (stats.p_flip / 0.159_f32).min(1.0);
                    if rho > 1e-4 {
                        let tau = qcfg.prior_refresh_temp.max(1e-6);
                        effective_prior = ((1.0 - rho) * prior.ln() + rho * q_eff / tau)
                            .exp()
                            .max(1e-8);
                    }
                }
                crate::mcts::quartz::effective_penalty_v2(n_raw, 0, qcfg.hbar_penalty_cap)
            }
            PenaltyMode::PFlipMixture => {
                if n_raw > 0 {
                    let flip_thresh = 0.159_f32;
                    let rho_max = 0.3_f32;
                    let p_ratio = (stats.p_flip / flip_thresh).min(2.0);
                    let mut rho_q = rho_max * p_ratio.min(1.0);
                    let mut rho_vf = rho_max * (1.0 - p_ratio).max(0.0);
                    if qcfg.pflip_mixture_divergence_gate
                        && stats.prior_q_divergence <= stats.epsilon_t.max(1e-6)
                    {
                        rho_q = 0.0;
                        rho_vf = 0.0;
                    }
                    if rho_q + rho_vf > 1e-4 {
                        let tau_q = qcfg.prior_refresh_temp.max(1e-6);
                        let tau_vf = (1.0 + n_avg).ln().max(0.1);
                        let vf_signal = (1.0 + n_raw as f32).ln() / tau_vf;
                        effective_prior = ((1.0 - rho_q - rho_vf) * prior.ln()
                            + rho_q * q_eff / tau_q
                            + rho_vf * vf_signal)
                            .exp()
                            .max(1e-8);
                    }
                    -qcfg.hbar_penalty_cap.max(stats.sigma_q.max(0.001)) / (1.0 + n_raw as f32)
                } else {
                    0.0
                }
            }
            PenaltyMode::Legacy => {
                if qcfg.enable_one_loop && n_raw > 0 {
                    -qcfg.hbar_penalty_cap.min(stats.hbar_eff) / (1.0 + n_raw as f32)
                } else {
                    0.0
                }
            }
            PenaltyMode::EffectiveV2 => {
                crate::mcts::quartz::effective_penalty_v2(n_raw, 0, qcfg.hbar_penalty_cap)
            }
            PenaltyMode::None => 0.0,
        };
        let delta = (effective_prior - prior).abs();
        if delta > 1e-6 {
            diag.refresh_count = diag.refresh_count.saturating_add(1);
            diag.refresh_activated = true;
            diag.effective_prior_l1 += delta;
        }
        diag.penalty_sum += penalty.abs();
    }
    diag
}

#[derive(Clone)]
struct EvalSearchStepResult {
    best_move: usize,
    iterations: u32,
    root_visits: u32,
    time_used_ms: f64,
    p_flip: f32,
    stop_reason: String,
    selection_root_selects: u64,
    selection_refresh_selected_count: u64,
    selection_penalty_abs_sum: f64,
    selection_effective_prior_l1_sum: f64,
}

#[derive(Clone)]
struct EvalSearchStepCompact {
    best_move: usize,
    iterations: u32,
    root_visits: u32,
    time_used_ms: f64,
    p_flip: f32,
    stop_reason: String,
    selection_root_selects: u64,
    selection_refresh_selected_count: u64,
    selection_penalty_abs_sum: f64,
    selection_effective_prior_l1_sum: f64,
}

fn execute_search<G: GameState>(
    engine: &MctsEngine<G>,
    thread_spec: SearchThreadSpec,
    iters: u32,
    qcfg: Option<QuartzConfig>,
    profile: SearchProfile,
) -> SearchExecutionOutcome
where
    usize: From<G::Move>,
    G::Move: Copy + Send + Sync + 'static,
{
    match profile {
        SearchProfile::Quartz => {
            let mut ctrl = QuartzController::new(iters, qcfg.clone().unwrap_or_default());
            let (effective_threads, auto_thread_reason) =
                if let Some(policy) = thread_spec.auto_policy {
                    let (_stats, decision) = engine.run_quartz_auto(&mut ctrl, policy);
                    (decision.threads, Some(format!("{:?}", decision.reason)))
                } else if thread_spec.requested_threads > 1 {
                    engine.run_par_quartz(&mut ctrl, thread_spec.requested_threads);
                    (thread_spec.requested_threads, None)
                } else {
                    engine.run_quartz(&mut ctrl);
                    (1, None)
                };
            let s = ctrl.last_stats();
            // Q3 (audit_codex_20260428.md W'1): aggregate per-halt-check
            // argmax-channel labels into a histogram so the per-game artifact
            // shows whether VOC halts are channel-distributed or
            // single-channel-dominated.
            let mut argmax_hist: std::collections::BTreeMap<&'static str, u32> =
                std::collections::BTreeMap::new();
            for check in ctrl.halt_telemetry() {
                *argmax_hist.entry(check.voc_argmax_channel).or_insert(0) += 1;
            }
            let diag = qcfg
                .as_ref()
                .map(|cfg| controller_diagnostics(engine, &s, cfg))
                .unwrap_or_default();
            let selection_trace = engine.selection_telemetry.snapshot();
            SearchExecutionOutcome {
                iterations: engine.root.n_total.load(Ordering::Relaxed),
                stop_reason: format!("{:?}", ctrl.last_stop_reason()),
                requested_threads: thread_spec.requested_threads,
                effective_threads,
                thread_policy: thread_spec.policy_name(),
                auto_thread_reason,
                p_flip: s.p_flip,
                value: s.mean_q,
                sigma_q: s.sigma_q,
                hbar_eff: s.hbar_eff,
                prior_q_divergence: Some(s.prior_q_divergence),
                voc_total: s.unified.voc_total,
                voc_focus: s.unified.voc_focus,
                voc_expand: s.unified.voc_expand,
                voc_merge: s.unified.voc_merge,
                refresh_count: diag.refresh_count,
                refresh_activated: diag.refresh_activated,
                penalty_sum: diag.penalty_sum,
                effective_prior_l1: diag.effective_prior_l1,
                selection_root_selects: selection_trace.root_selects,
                selection_refresh_selected_count: selection_trace.refresh_selected_count,
                selection_penalty_abs_sum: selection_trace.selected_penalty_abs_sum,
                selection_effective_prior_l1_sum: selection_trace.selected_effective_prior_l1_sum,
                selection_mean_candidate_count: selection_trace.selected_mean_candidate_count,
                selection_max_candidate_count: selection_trace.selected_max_candidate_count,
                voc_argmax_channel_hist: argmax_hist,
                // P01
                selection_penalty_mode_invoke_count: selection_trace.penalty_mode_invoke_count,
                selection_refresh_eligible_count: selection_trace.refresh_eligible_count,
                selection_refresh_active_count: selection_trace.refresh_active_count,
                halt_reason_count: ctrl.halt_reason_count_snapshot(),
            }
        }
        SearchProfile::Baseline | SearchProfile::BaselineStrict => {
            let (stats, effective_threads, auto_thread_reason) =
                if let Some(policy) = thread_spec.auto_policy {
                    let mut ctrl = FixedIterations::new(iters);
                    let (stats, decision) = engine.run_auto(&mut ctrl, policy);
                    (
                        stats,
                        decision.threads,
                        Some(format!("{:?}", decision.reason)),
                    )
                } else if thread_spec.requested_threads > 1 {
                    let ctrl = FixedIterations::new(iters);
                    (
                        engine.run_par(&ctrl, thread_spec.requested_threads),
                        thread_spec.requested_threads,
                        None,
                    )
                } else {
                    (engine.run(&mut FixedIterations::new(iters)), 1, None)
                };
            let selection_trace = engine.selection_telemetry.snapshot();
            SearchExecutionOutcome {
                iterations: stats.iterations,
                stop_reason: format!("{:?}", stats.stop_reason),
                requested_threads: thread_spec.requested_threads,
                effective_threads,
                thread_policy: thread_spec.policy_name(),
                auto_thread_reason,
                p_flip: 0.0,
                value: 0.0,
                sigma_q: 0.0,
                hbar_eff: 0.0,
                prior_q_divergence: None,
                voc_total: 0.0,
                voc_focus: 0.0,
                voc_expand: 0.0,
                voc_merge: 0.0,
                refresh_count: 0,
                refresh_activated: false,
                penalty_sum: 0.0,
                effective_prior_l1: 0.0,
                selection_root_selects: selection_trace.root_selects,
                selection_refresh_selected_count: selection_trace.refresh_selected_count,
                selection_penalty_abs_sum: selection_trace.selected_penalty_abs_sum,
                selection_effective_prior_l1_sum: selection_trace.selected_effective_prior_l1_sum,
                selection_mean_candidate_count: selection_trace.selected_mean_candidate_count,
                selection_max_candidate_count: selection_trace.selected_max_candidate_count,
                voc_argmax_channel_hist: std::collections::BTreeMap::new(),
                // P01: baseline profiles run no QuartzController, so the
                // controller-side counters are all-zero.
                selection_penalty_mode_invoke_count: selection_trace.penalty_mode_invoke_count,
                selection_refresh_eligible_count: selection_trace.refresh_eligible_count,
                selection_refresh_active_count: selection_trace.refresh_active_count,
                halt_reason_count: [0u32; crate::mcts::quartz::HALT_REASON_COUNT],
            }
        }
    }
}

fn run_eval_search_step<G: GameState>(
    engine: &MctsEngine<G>,
    n_threads: usize,
    iters: u32,
    qcfg: Option<QuartzConfig>,
    profile: SearchProfile,
) -> Option<EvalSearchStepResult>
where
    usize: From<G::Move>,
    G::Move: Copy + Send + Sync + 'static,
{
    let outcome = execute_search(
        engine,
        SearchThreadSpec::explicit(n_threads),
        iters,
        qcfg.clone(),
        profile,
    );
    let best_move = engine
        .best_move()
        .map(|mv| engine.root_state().move_to_idx(mv))?;
    Some(EvalSearchStepResult {
        best_move,
        iterations: engine.root.n_total.load(Ordering::Relaxed),
        root_visits: engine.root.n_total.load(Ordering::Relaxed),
        time_used_ms: 0.0,
        p_flip: outcome.p_flip,
        stop_reason: outcome.stop_reason,
        selection_root_selects: outcome.selection_root_selects,
        selection_refresh_selected_count: outcome.selection_refresh_selected_count,
        selection_penalty_abs_sum: outcome.selection_penalty_abs_sum,
        selection_effective_prior_l1_sum: outcome.selection_effective_prior_l1_sum,
    })
}

fn run_and_extract<G: GameState>(
    engine: &MctsEngine<G>,
    thread_spec: SearchThreadSpec,
    n_act: usize,
    iters: u32,
    qcfg: Option<QuartzConfig>,
    profile: SearchProfile,
) -> serde_json::Value
where
    usize: From<G::Move>,
    G::Move: Copy + Send + Sync + 'static,
{
    let phase_before = engine_phase_snapshot();
    let edge_before = edge_lock_contention_snapshot();
    let outcome = execute_search(engine, thread_spec, iters, qcfg.clone(), profile);
    let effective_threads = outcome.effective_threads;
    let out = build_result_value(
        engine,
        n_act,
        &outcome,
        outcome.iterations,
        outcome.stop_reason.clone(),
        outcome.p_flip,
        outcome.value,
        outcome.sigma_q,
        outcome.hbar_eff,
        outcome.prior_q_divergence,
    );
    let mut out = out;
    attach_search_metadata(
        &mut out,
        profile,
        iters,
        effective_threads,
        search_evaluator_path(profile, thread_spec.requested_threads, false),
        qcfg.as_ref(),
    );
    let tt = engine.tt.contention_snapshot();
    let phase_after = engine_phase_snapshot();
    let edge_after = edge_lock_contention_snapshot();
    rust_server_trace(
        "search_result_stats",
        serde_json::json!({
            "profile": search_profile_name(profile),
            "requested_threads": outcome.requested_threads,
            "effective_threads": outcome.effective_threads,
            "n_threads": outcome.effective_threads,
            "thread_policy": outcome.thread_policy,
            "auto_thread_reason": outcome.auto_thread_reason,
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

fn make_eval<G: GameState>(
    profile: SearchProfile,
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
    if should_use_batch_eval(profile, n_threads, force_batch) {
        let cfg = BatchConfig {
            max_batch_size: batch_size.max(n_threads),
            timeout_us: batch_timeout_us,
        };
        Arc::new(BatchStdioEval::<<G as GameState>::Move>::new(
            n_actions, cfg,
        )) as Arc<dyn crate::game::Evaluator<G>>
    } else {
        note_serial_eval_fallback(profile, n_threads, n_actions);
        Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>
    }
}

fn make_eval_pair<G: GameState>(
    profile: SearchProfile,
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
    if should_use_batch_eval(profile, n_threads, force_batch) {
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
                Arc::new(BatchStdioEval::<<G as GameState>::Move>::new(
                    n_actions, cfg,
                )) as Arc<dyn crate::game::Evaluator<G>>,
                None,
            )
        }
    } else if dual_model {
        note_serial_eval_fallback(profile, n_threads, n_actions);
        (
            Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>,
            Some(Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>),
        )
    } else {
        note_serial_eval_fallback(profile, n_threads, n_actions);
        (
            Arc::new(StdioCallbackEval::new(n_actions)) as Arc<dyn crate::game::Evaluator<G>>,
            None,
        )
    }
}

#[derive(Clone)]
struct TaggedSharedEvaluator<G: GameState> {
    current_tag: Arc<AtomicU32>,
    eval_a: Arc<dyn Evaluator<G>>,
    eval_b: Option<Arc<dyn Evaluator<G>>>,
}

impl<G: GameState> Evaluator<G> for TaggedSharedEvaluator<G> {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        let tag = self.current_tag.load(Ordering::Relaxed);
        if tag == 0 {
            self.eval_a.evaluate(state)
        } else {
            self.eval_b.as_ref().unwrap_or(&self.eval_a).evaluate(state)
        }
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

// ─── Global inflight credit system ───

/// Atomic credit counter for global inflight admission control.
/// Limits the total number of pending NN evaluations across all jobs.
struct GlobalInflightCredit {
    remaining: AtomicUsize,
    capacity: usize,
    peak_used: AtomicUsize,
}

impl GlobalInflightCredit {
    fn new(capacity: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(capacity),
            capacity,
            peak_used: AtomicUsize::new(0),
        }
    }

    /// Try to acquire one credit. Returns an RAII permit on success.
    fn try_acquire(&self) -> Option<CreditPermit<'_>> {
        let mut prev = self.remaining.load(Ordering::Acquire);
        loop {
            if prev == 0 {
                return None;
            }
            match self.remaining.compare_exchange_weak(
                prev,
                prev - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Update peak high-water mark
                    let used = self.capacity - (prev - 1);
                    let mut peak = self.peak_used.load(Ordering::Relaxed);
                    while used > peak {
                        match self.peak_used.compare_exchange_weak(
                            peak,
                            used,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(cur) => peak = cur,
                        }
                    }
                    return Some(CreditPermit { credit: self });
                }
                Err(cur) => prev = cur,
            }
        }
    }

    fn peak(&self) -> usize {
        self.peak_used.load(Ordering::Relaxed)
    }
}

/// RAII guard that returns one credit on drop. Prevents credit leaks on
/// early return, panic, or teardown.
struct CreditPermit<'a> {
    credit: &'a GlobalInflightCredit,
}

impl Drop for CreditPermit<'_> {
    fn drop(&mut self) {
        self.credit.remaining.fetch_add(1, Ordering::Release);
    }
}

// ─── Async batch job structures ───

struct AsyncBatchPending<'credit, G: GameState> {
    selection: crate::mcts::AsyncPendingIteration<G>,
    ticket: AsyncEvalTicket<G::Move>,
    _permit: CreditPermit<'credit>,
}

struct AsyncBatchJob<'credit, G: GameState> {
    slot_idx: usize,
    model_tag: u32,
    engine: MctsEngine<G>,
    launched: u32,
    completed: u32,
    pending: Vec<AsyncBatchPending<'credit, G>>,
}

fn build_async_result_value<G: GameState>(
    engine: &MctsEngine<G>,
    completed: u32,
    n_actions: usize,
    n_threads: usize,
    search_profile: SearchProfile,
) -> serde_json::Value
where
    usize: From<G::Move>,
{
    let (
        p_flip,
        value,
        sigma_q,
        hbar_eff,
        prior_q_divergence,
        voc_total,
        voc_focus,
        voc_expand,
        voc_merge,
    ) = match search_profile {
        SearchProfile::Quartz => match engine.current_quartz_stats() {
            Some(stats) => (
                stats.p_flip,
                stats.mean_q,
                stats.sigma_q,
                stats.hbar_eff,
                Some(stats.prior_q_divergence),
                stats.unified.voc_total,
                stats.unified.voc_focus,
                stats.unified.voc_expand,
                stats.unified.voc_merge,
            ),
            None => (0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0),
        },
        SearchProfile::Baseline | SearchProfile::BaselineStrict => {
            (0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0)
        }
    };
    // Q3: async result paths don't accumulate per-halt-check telemetry
    // (only the live `current_quartz_stats` snapshot is available), so the
    // argmax histogram is empty here. The non-async path
    // (`run_and_extract`) does carry the full per-game histogram.
    let selection_trace = engine.selection_telemetry.snapshot();
    let synth_outcome = SearchExecutionOutcome {
        iterations: completed,
        stop_reason: "BudgetExhausted".to_string(),
        requested_threads: n_threads,
        effective_threads: n_threads,
        thread_policy: "explicit",
        auto_thread_reason: None,
        p_flip,
        value,
        sigma_q,
        hbar_eff,
        prior_q_divergence,
        voc_total,
        voc_focus,
        voc_expand,
        voc_merge,
        refresh_count: 0,
        refresh_activated: false,
        penalty_sum: 0.0,
        effective_prior_l1: 0.0,
        selection_root_selects: selection_trace.root_selects,
        selection_refresh_selected_count: selection_trace.refresh_selected_count,
        selection_penalty_abs_sum: selection_trace.selected_penalty_abs_sum,
        selection_effective_prior_l1_sum: selection_trace.selected_effective_prior_l1_sum,
        selection_mean_candidate_count: selection_trace.selected_mean_candidate_count,
        selection_max_candidate_count: selection_trace.selected_max_candidate_count,
        voc_argmax_channel_hist: std::collections::BTreeMap::new(),
        // P01: async synth path doesn't have direct controller access here.
        // Selection trace is real (engine.selection_telemetry); halt_reason
        // counts are intentionally zeroed since the live controller is
        // outside this function's scope.
        selection_penalty_mode_invoke_count: selection_trace.penalty_mode_invoke_count,
        selection_refresh_eligible_count: selection_trace.refresh_eligible_count,
        selection_refresh_active_count: selection_trace.refresh_active_count,
        halt_reason_count: [0u32; crate::mcts::quartz::HALT_REASON_COUNT],
    };
    build_result_value(
        engine,
        n_actions,
        &synth_outcome,
        completed,
        "BudgetExhausted".to_string(),
        p_flip,
        value,
        sigma_q,
        hbar_eff,
        prior_q_divergence,
    )
}

/// Per-tick counters accumulated during job processing.
struct TickCounters {
    immediate_terminal: u64,
    immediate_tt_cap: u64,
}

/// Process one tick of a single async batch job: gather new iterations + reap completed results.
/// Returns true if any progress was made (launch or reap).
fn process_job_tick<'credit, G: GameState>(
    job: &mut AsyncBatchJob<'credit, G>,
    iters: u32,
    per_job_soft_cap: usize,
    credit: &'credit GlobalInflightCredit,
    eval_a: &BatchStdioEval<G::Move>,
    eval_b: &Option<BatchStdioEval<G::Move>>,
    counters: &mut TickCounters,
) -> bool
where
    usize: From<G::Move>,
{
    let mut made_progress = false;

    // --- Gather: launch new iterations (gated by per-job soft cap + global credit) ---
    while job.launched < iters && job.pending.len() < per_job_soft_cap {
        // Immediate iterations don't consume credit (no NN eval needed)
        match job.engine.prepare_iteration_async() {
            PreparedIteration::Immediate {
                path,
                value,
                reason,
            } => {
                job.engine.apply_iteration_value_async(path, value);
                job.launched += 1;
                job.completed += 1;
                job.engine.refresh_async_runtime(job.completed);
                made_progress = true;
                match reason {
                    crate::mcts::ImmediateReason::TtCapHit => {
                        counters.immediate_tt_cap += 1;
                    }
                    crate::mcts::ImmediateReason::TerminalNode => {
                        counters.immediate_terminal += 1;
                    }
                }
            }
            PreparedIteration::Pending(selection) => {
                // Acquire global credit before submitting to broker
                let Some(permit) = credit.try_acquire() else {
                    break; // global budget exhausted — drain first
                };
                let ticket = if job.model_tag == 1 {
                    eval_b
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| eval_a.clone())
                        .submit(&selection.leaf_state)
                } else {
                    eval_a.clone().submit(&selection.leaf_state)
                };
                job.pending.push(AsyncBatchPending {
                    selection,
                    ticket,
                    _permit: permit,
                });
                job.launched += 1;
                made_progress = true;
            }
        }
    }

    // --- Reap: poll completed results (credit auto-released via RAII on drop) ---
    let mut idx = 0usize;
    while idx < job.pending.len() {
        if let Some(result) = job.pending[idx].ticket.try_take() {
            let pending = job.pending.swap_remove(idx);
            // pending._permit dropped here → credit auto-released
            job.engine
                .complete_iteration_async(pending.selection, result);
            job.completed += 1;
            job.engine.refresh_async_runtime(job.completed);
            made_progress = true;
        } else {
            idx += 1;
        }
    }

    made_progress
}

/// Adaptive backoff for idle spins: spin → yield → 50µs → 200µs.
fn adaptive_backoff(idle_spins: u64) {
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
    if sleep_us > 0 {
        std::thread::sleep(std::time::Duration::from_micros(sleep_us));
    }
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
    let max_index = tagged_states
        .iter()
        .map(|(idx, _, _)| *idx)
        .max()
        .unwrap_or(0);
    let max_inflight_per_job = n_threads.max(1);
    let job_count = tagged_states.len();

    // Global inflight credit: scale with both job concurrency and n_threads.
    // Use generous capacity to ensure the broker can fill large batches.
    let aggregate_cap = max_inflight_per_job * job_count;
    let credit_capacity = aggregate_cap.max(job_count);
    let credit = GlobalInflightCredit::new(credit_capacity);
    let per_job_soft_cap = (credit_capacity / job_count.max(1)).max(1) + 2;

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
    let mut counters = TickCounters {
        immediate_terminal: 0,
        immediate_tt_cap: 0,
    };

    // Compute worker thread count independently from n_threads (which means inflight-per-job).
    let worker_threads = {
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        (avail / 2).max(1).min(job_count)
    };

    rust_server_trace(
        "run_multi_async_batch_start",
        serde_json::json!({
            "jobs": job_count,
            "iters": iters,
            "max_inflight_per_job": max_inflight_per_job,
            "worker_threads_effective": worker_threads,
            "inflight_per_job_effective": per_job_soft_cap,
            "credit_capacity": credit_capacity,
            "search_profile": search_profile_name(search_profile),
        }),
    );

    if worker_threads <= 1 {
        // Single-threaded path (backward compatible)
        while jobs
            .iter()
            .any(|job| job.completed < iters || !job.pending.is_empty())
        {
            let mut made_progress = false;
            for job in jobs.iter_mut() {
                made_progress |= process_job_tick(
                    job,
                    iters,
                    per_job_soft_cap,
                    &credit,
                    &eval_a,
                    &eval_b,
                    &mut counters,
                );
            }
            if !made_progress {
                idle_spins += 1;
                if idle_spins % 2048 == 0 {
                    rust_server_trace(
                        "run_multi_async_batch_idle",
                        serde_json::json!({
                            "jobs": job_count,
                            "pending_eval": jobs.iter().map(|j| j.pending.len()).sum::<usize>(),
                            "completed": jobs.iter().map(|j| j.completed as usize).sum::<usize>(),
                            "credit_in_use": credit.capacity - credit.remaining.load(Ordering::Relaxed),
                        }),
                    );
                }
                adaptive_backoff(idle_spins);
            } else {
                idle_spins = 0;
            }
        }
    } else {
        // Multi-threaded path: partition jobs across worker threads
        let mut worker_buckets: Vec<Vec<AsyncBatchJob<'_, G>>> =
            (0..worker_threads).map(|_| Vec::new()).collect();
        for (i, job) in jobs.into_iter().enumerate() {
            worker_buckets[i % worker_threads].push(job);
        }

        let worker_results: Vec<(Vec<AsyncBatchJob<'_, G>>, TickCounters, u64)> =
            std::thread::scope(|s| {
                let credit_ref = &credit;
                let eval_a_ref = &eval_a;
                let eval_b_ref = &eval_b;
                let handles: Vec<_> = worker_buckets
                    .into_iter()
                    .enumerate()
                    .map(|(wid, mut my_jobs)| {
                        let my_job_count = my_jobs.len();
                        s.spawn(move || {
                            let mut telem = TickCounters {
                                immediate_terminal: 0,
                                immediate_tt_cap: 0,
                            };
                            let mut local_idle_spins = 0u64;
                            while my_jobs
                                .iter()
                                .any(|j| j.completed < iters || !j.pending.is_empty())
                            {
                                let mut made_progress = false;
                                for job in my_jobs.iter_mut() {
                                    made_progress |= process_job_tick(
                                        job,
                                        iters,
                                        per_job_soft_cap,
                                        credit_ref,
                                        eval_a_ref,
                                        eval_b_ref,
                                        &mut telem,
                                    );
                                }
                                if !made_progress {
                                    local_idle_spins += 1;
                                    adaptive_backoff(local_idle_spins);
                                } else {
                                    local_idle_spins = 0;
                                }
                            }
                            rust_server_trace(
                                "worker_done",
                                serde_json::json!({
                                    "worker_id": wid,
                                    "jobs_count": my_job_count,
                                    "iterations_completed": my_jobs.iter().map(|j| j.completed as u64).sum::<u64>(),
                                    "idle_spins": local_idle_spins,
                                }),
                            );
                            (my_jobs, telem, local_idle_spins)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

        // Reassemble jobs from all workers
        jobs = Vec::new();
        for (worker_jobs, worker_counters, worker_idle) in worker_results {
            counters.immediate_terminal += worker_counters.immediate_terminal;
            counters.immediate_tt_cap += worker_counters.immediate_tt_cap;
            idle_spins += worker_idle;
            jobs.extend(worker_jobs);
        }
    }

    // Track which slots were targeted by jobs vs structurally inactive
    let mut targeted_slots = vec![false; results.len()];
    for job in jobs.iter() {
        if job.slot_idx < results.len() {
            targeted_slots[job.slot_idx] = true;
            results[job.slot_idx] = build_async_result_value(
                &job.engine,
                job.completed,
                n_actions,
                n_threads,
                search_profile,
            );
        }
    }

    // Split null metrics: inactive-slot (structural) vs result-miss (actual failure)
    let mut null_inactive_slot = 0usize;
    let mut null_result_miss = 0usize;
    for (i, v) in results.iter().enumerate() {
        if v.is_null() {
            if i < targeted_slots.len() && targeted_slots[i] {
                null_result_miss += 1;
            } else {
                null_inactive_slot += 1;
            }
        }
    }

    rust_server_trace(
        "run_multi_async_batch_done",
        serde_json::json!({
            "results_len": results.len(),
            "null_inactive_slot": null_inactive_slot,
            "null_result_miss": null_result_miss,
            "immediate_terminal": counters.immediate_terminal,
            "immediate_tt_cap": counters.immediate_tt_cap,
            "idle_spins": idle_spins,
            "worker_threads": worker_threads,
            "credit_capacity": credit_capacity,
            "peak_inflight": credit.peak(),
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
                    let value = run_and_extract(
                        &engine,
                        SearchThreadSpec::explicit(engine_threads),
                        n_actions,
                        iters,
                        qcfg.clone(),
                        search_profile,
                    );
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
            .map(|m| {
                m.lock()
                    .map(|g| g.clone())
                    .unwrap_or(serde_json::Value::Null)
            })
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
                let value = run_and_extract(
                    &engine,
                    SearchThreadSpec::explicit(engine_threads),
                    n_actions,
                    iters,
                    qcfg.clone(),
                    search_profile,
                );
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
            .map(|m| {
                m.lock()
                    .map(|g| g.clone())
                    .unwrap_or(serde_json::Value::Null)
            })
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

struct EngineSearchSession<G: GameState> {
    engines: Vec<Option<MctsEngine<G>>>,
    cumulative_iters: Vec<u32>,
    n_threads: usize,
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

impl<G: GameState> EngineSearchSession<G>
where
    usize: From<G::Move>,
    G::Move: PartialEq + Copy + Send + Sync + 'static,
{
    fn search_with_iters_compact(&mut self, iters: u32) -> Vec<Option<EvalSearchStepCompact>> {
        self.engines
            .iter_mut()
            .enumerate()
            .map(|(idx, engine_opt)| {
                let Some(engine) = engine_opt.as_mut() else {
                    return None;
                };
                let mut result = run_eval_search_step(
                    engine,
                    self.n_threads,
                    iters,
                    engine.config.quartz.clone(),
                    self.search_profile,
                )?;
                if let Some(total) = self.cumulative_iters.get_mut(idx) {
                    *total = total.saturating_add(iters);
                    result.iterations = *total;
                }
                Some(EvalSearchStepCompact {
                    best_move: result.best_move,
                    iterations: result.iterations,
                    root_visits: result.root_visits,
                    time_used_ms: result.time_used_ms,
                    p_flip: result.p_flip,
                    stop_reason: result.stop_reason,
                    selection_root_selects: result.selection_root_selects,
                    selection_refresh_selected_count: result.selection_refresh_selected_count,
                    selection_penalty_abs_sum: result.selection_penalty_abs_sum,
                    selection_effective_prior_l1_sum: result.selection_effective_prior_l1_sum,
                })
            })
            .collect()
    }

    fn search_with_iters(&mut self, iters: u32) -> Vec<serde_json::Value> {
        self.search_with_iters_compact(iters)
            .into_iter()
            .map(|result| {
                let Some(result) = result else {
                    return serde_json::Value::Null;
                };
                serde_json::json!({
                    "best_move": result.best_move,
                    "time_used_ms": result.time_used_ms,
                    "iterations": result.iterations,
                    "root_visits": result.root_visits,
                    "p_flip": result.p_flip,
                    "stop_reason": result.stop_reason,
                    "selection_root_selects": result.selection_root_selects,
                    "selection_refresh_selected_count": result.selection_refresh_selected_count,
                    "selection_penalty_abs_sum": result.selection_penalty_abs_sum,
                    "selection_effective_prior_l1_sum": result.selection_effective_prior_l1_sum,
                })
            })
            .collect()
    }

    fn deactivate(&mut self, slot: usize) {
        if let Some(engine) = self.engines.get_mut(slot) {
            *engine = None;
        }
    }

    fn insert_engine(&mut self, slot: usize, engine: MctsEngine<G>) {
        if let Some(dst) = self.engines.get_mut(slot) {
            *dst = Some(engine);
        }
    }

    fn apply_action_idx(&mut self, slot: usize, action: usize) -> Result<(), String> {
        let Some(engine) = self.engines.get_mut(slot).and_then(Option::as_mut) else {
            return Ok(());
        };
        engine
            .apply_action_idx_root(action)
            .map_err(|err| format!("{} for slot {}", err, slot))
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

struct EngineGomoku15Session {
    inner: EngineSearchSession<Gomoku15>,
    variant: GomokuVariant,
}

struct EngineGoSession {
    inner: EngineSearchSession<Go>,
    size: usize,
    ruleset: GoRuleset,
    scoring: GoScoring,
    komi: f32,
    allow_suicide: bool,
}

struct EngineChessSession {
    inner: EngineSearchSession<Chess>,
    default_960: bool,
}

impl ChessSession {
    fn search(&self) -> Vec<serde_json::Value> {
        let mut results = self.inner.search();
        for (state, result) in self.inner.states.iter().zip(results.iter_mut()) {
            if let Some(state) = state.as_ref() {
                enrich_chess_result(state, result);
            }
        }
        results
    }
}

impl EngineChessSession {
    fn search_with_iters(&mut self, iters: u32) -> Vec<serde_json::Value> {
        let mut results = self.inner.search_with_iters(iters);
        for (engine, result) in self.inner.engines.iter().zip(results.iter_mut()) {
            if let Some(engine) = engine.as_ref() {
                enrich_chess_result(engine.root_state(), result);
            }
        }
        results
    }
}

enum SearchSessionAny {
    Gomoku(SearchSession<Gomoku>),
    Gomoku15(Gomoku15Session),
    Go(GoSession),
    Chess(ChessSession),
    TicTacToe(SearchSession<TicTacToe>),
    EngineGomoku(EngineSearchSession<Gomoku>),
    EngineGomoku15(EngineGomoku15Session),
    EngineGo(EngineGoSession),
    EngineChess(EngineChessSession),
    EngineTicTacToe(EngineSearchSession<TicTacToe>),
}

impl SearchSessionAny {
    fn search(&self) -> Vec<serde_json::Value> {
        match self {
            SearchSessionAny::Gomoku(inner) => inner.search(),
            SearchSessionAny::Gomoku15(inner) => inner.inner.search(),
            SearchSessionAny::Go(inner) => inner.inner.search(),
            SearchSessionAny::Chess(inner) => inner.search(),
            SearchSessionAny::TicTacToe(inner) => inner.search(),
            SearchSessionAny::EngineGomoku(_)
            | SearchSessionAny::EngineGomoku15(_)
            | SearchSessionAny::EngineGo(_)
            | SearchSessionAny::EngineChess(_)
            | SearchSessionAny::EngineTicTacToe(_) => vec![],
        }
    }

    fn search_with_iters(&mut self, iters: u32) -> Vec<serde_json::Value> {
        match self {
            SearchSessionAny::Gomoku(inner) => {
                inner.iters = iters;
                inner.search()
            }
            SearchSessionAny::Gomoku15(inner) => {
                inner.inner.iters = iters;
                inner.inner.search()
            }
            SearchSessionAny::Go(inner) => {
                inner.inner.iters = iters;
                inner.inner.search()
            }
            SearchSessionAny::Chess(inner) => {
                inner.inner.iters = iters;
                inner.search()
            }
            SearchSessionAny::TicTacToe(inner) => {
                inner.iters = iters;
                inner.search()
            }
            SearchSessionAny::EngineGomoku(inner) => inner.search_with_iters(iters),
            SearchSessionAny::EngineGomoku15(inner) => inner.inner.search_with_iters(iters),
            SearchSessionAny::EngineGo(inner) => inner.inner.search_with_iters(iters),
            SearchSessionAny::EngineChess(inner) => inner.search_with_iters(iters),
            SearchSessionAny::EngineTicTacToe(inner) => inner.search_with_iters(iters),
        }
    }

    fn apply_updates(&mut self, updates: &[serde_json::Value]) -> Result<(), String> {
        match self {
            SearchSessionAny::Gomoku(inner) => apply_updates_gomoku(inner, updates),
            SearchSessionAny::Gomoku15(inner) => apply_updates_gomoku15(inner, updates),
            SearchSessionAny::Go(inner) => apply_updates_go(inner, updates),
            SearchSessionAny::Chess(inner) => apply_updates_chess(inner, updates),
            SearchSessionAny::TicTacToe(inner) => apply_updates_tictactoe(inner, updates),
            SearchSessionAny::EngineGomoku(inner) => apply_updates_engine_gomoku(inner, updates),
            SearchSessionAny::EngineGomoku15(inner) => {
                apply_updates_engine_gomoku15(inner, updates)
            }
            SearchSessionAny::EngineGo(inner) => apply_updates_engine_go(inner, updates),
            SearchSessionAny::EngineChess(inner) => apply_updates_engine_chess(inner, updates),
            SearchSessionAny::EngineTicTacToe(inner) => {
                apply_updates_engine_tictactoe(inner, updates)
            }
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
    let ko_point = job.get("ko_point").and_then(|v| v.as_i64()).and_then(|v| {
        if v >= 0 {
            Some(v as u16)
        } else {
            None
        }
    });
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
    chess_state_from_json(job, default_960)
}

fn enrich_chess_result(state: &Chess, result: &mut serde_json::Value) {
    let Some(obj) = result.as_object_mut() else {
        return;
    };
    if obj.get("error").and_then(|v| v.as_str()).is_some() {
        return;
    }
    let best_move = obj
        .get("best_move")
        .and_then(|v| v.as_u64())
        .and_then(|idx| state.idx_to_move(idx as usize));
    let next_state = best_move
        .map(|mv| state.apply_move(mv))
        .unwrap_or_else(|| state.clone());
    obj.insert(
        "best_move_uci".to_string(),
        serde_json::json!(best_move.map(|mv| mv.to_uci()).unwrap_or_default()),
    );
    obj.insert(
        "result_fen".to_string(),
        serde_json::json!(next_state.to_fen()),
    );
    obj.insert(
        "result_history_hashes".to_string(),
        serde_json::json!(next_state.history_hashes()),
    );
}

fn apply_updates_gomoku(
    session: &mut SearchSession<Gomoku>,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session.replace(slot, parse_gomoku7_job(replace));
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
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
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
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
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
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
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            session.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_chess(
    session: &mut ChessSession,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            session
                .inner
                .replace(slot, parse_chess_job(replace, session.default_960));
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            session.inner.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.inner.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_engine_gomoku(
    session: &mut EngineSearchSession<Gomoku>,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            let Some(engine) = session.engines.get(slot).and_then(|v| v.as_ref()) else {
                return Err(format!("replace requested for inactive slot {}", slot));
            };
            session.insert_engine(
                slot,
                MctsEngine::new(
                    parse_gomoku7_job(replace),
                    engine.evaluator.clone(),
                    engine.config.clone(),
                ),
            );
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            session.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_engine_gomoku15(
    session: &mut EngineGomoku15Session,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            let Some(engine) = session.inner.engines.get(slot).and_then(|v| v.as_ref()) else {
                return Err(format!("replace requested for inactive slot {}", slot));
            };
            session.inner.insert_engine(
                slot,
                MctsEngine::new(
                    parse_gomoku15_job(replace, session.variant),
                    engine.evaluator.clone(),
                    engine.config.clone(),
                ),
            );
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            session.inner.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.inner.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_engine_go(
    session: &mut EngineGoSession,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            let Some(engine) = session.inner.engines.get(slot).and_then(|v| v.as_ref()) else {
                return Err(format!("replace requested for inactive slot {}", slot));
            };
            session.inner.insert_engine(
                slot,
                MctsEngine::new(
                    parse_go_job(
                        replace,
                        session.size,
                        session.ruleset,
                        session.scoring,
                        session.komi,
                        session.allow_suicide,
                    ),
                    engine.evaluator.clone(),
                    engine.config.clone(),
                ),
            );
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            session.inner.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.inner.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_engine_tictactoe(
    session: &mut EngineSearchSession<TicTacToe>,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            let Some(engine) = session.engines.get(slot).and_then(|v| v.as_ref()) else {
                return Err(format!("replace requested for inactive slot {}", slot));
            };
            session.insert_engine(
                slot,
                MctsEngine::new(
                    parse_tictactoe_job(replace),
                    engine.evaluator.clone(),
                    engine.config.clone(),
                ),
            );
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            session.deactivate(slot);
        } else if let Some(action) = update.get("action").and_then(|v| v.as_u64()) {
            session.apply_action_idx(slot, action as usize)?;
        }
    }
    Ok(())
}

fn apply_updates_engine_chess(
    session: &mut EngineChessSession,
    updates: &[serde_json::Value],
) -> Result<(), String> {
    for (slot, update) in updates.iter().enumerate() {
        if let Some(replace) = update.get("replace") {
            let Some(engine) = session.inner.engines.get(slot).and_then(|v| v.as_ref()) else {
                return Err(format!("replace requested for inactive slot {}", slot));
            };
            session.inner.insert_engine(
                slot,
                MctsEngine::new(
                    parse_chess_job(replace, session.default_960),
                    engine.evaluator.clone(),
                    engine.config.clone(),
                ),
            );
        } else if update
            .get("deactivate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
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
    // Large batches need more fill time
    if batch_size > 64 {
        timeout_us += (batch_size as u64 - 64) * 10;
    }
    timeout_us.clamp(500, 8000)
}

fn handle_search_nn(line: &str) -> SearchCommandReply {
    use crate::mcts::eval::{BatchConfig, BatchStdioEval, StdioCallbackEval};

    let game = jstr(line, "game").unwrap_or("gomoku15");
    let iters = jint(line, "iters").unwrap_or(200) as u32;
    let overrides = parse_search_overrides(line);
    let search_profile = parse_search_profile(line);
    let thread_spec = parse_search_thread_spec(line);
    let n_threads = thread_spec.requested_threads;
    let batch_size = (jint(line, "batch_size").unwrap_or(8) as usize).max(1);
    let batch_timeout_us = jint(line, "batch_timeout_us")
        .unwrap_or(default_batch_timeout_us(n_threads, batch_size, 1) as i64)
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
            if should_use_batch_eval(search_profile, n_threads, false) {
                let cfg = BatchConfig {
                    max_batch_size: batch_size.max(n_threads),
                    timeout_us: batch_timeout_us,
                };
                Arc::new(BatchStdioEval::<<$game_type as GameState>::Move>::new(
                    n_actions, cfg,
                )) as Arc<dyn crate::game::Evaluator<$game_type>>
            } else {
                note_serial_eval_fallback(search_profile, n_threads, n_actions);
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
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    }),
                    &overrides,
                ),
                search_profile,
            );
            let engine = MctsEngine::new(state, eval, cfg);
            SearchCommandReply::Search(SearchResponsePayload::Single {
                result: run_and_extract(
                    &engine,
                    thread_spec,
                    49,
                    iters,
                    engine.config.quartz.clone(),
                    search_profile,
                ),
            })
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
            let cfg = apply_search_profile(
                apply_search_overrides(gomoku15_quartz(variant), &overrides),
                search_profile,
            );
            let engine = MctsEngine::new(state, eval, cfg);
            SearchCommandReply::Search(SearchResponsePayload::Single {
                result: run_and_extract(
                    &engine,
                    thread_spec,
                    225,
                    iters,
                    engine.config.quartz.clone(),
                    search_profile,
                ),
            })
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
            let ko_point = if ko_point_raw >= 0 {
                Some(ko_point_raw as u16)
            } else {
                None
            };
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
            };
            let eval = make_eval!(Go);
            let cfg = apply_search_profile(
                apply_search_overrides(go_quartz(size), &overrides),
                search_profile,
            );
            let engine = MctsEngine::new(state, eval, cfg);
            SearchCommandReply::Search(SearchResponsePayload::Single {
                result: run_and_extract(
                    &engine,
                    thread_spec,
                    n_actions,
                    iters,
                    engine.config.quartz.clone(),
                    search_profile,
                ),
            })
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
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                    &overrides,
                ),
                search_profile,
            );
            let engine = MctsEngine::new(state, eval, cfg);
            SearchCommandReply::Search(SearchResponsePayload::Single {
                result: run_and_extract(
                    &engine,
                    thread_spec,
                    9,
                    iters,
                    engine.config.quartz.clone(),
                    search_profile,
                ),
            })
        }
        game if is_chess_game_name(game) => {
            let state = chess_state_from_request(line, game == "chess960");
            let eval = make_eval::<Chess>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                n_actions,
                false,
            );
            let cfg = apply_search_profile(
                apply_search_overrides(chess_quartz(), &overrides),
                search_profile,
            );
            let engine = MctsEngine::new(state, eval, cfg);
            // Chess has custom result extraction (includes result_fen)
            let outcome = execute_search(
                &engine,
                thread_spec,
                iters,
                engine.config.quartz.clone(),
                search_profile,
            );
            let mut result = build_result_value(
                &engine,
                CHESS_POLICY_ACTIONS,
                &outcome,
                outcome.iterations,
                outcome.stop_reason.clone(),
                outcome.p_flip,
                outcome.value,
                outcome.sigma_q,
                outcome.hbar_eff,
                outcome.prior_q_divergence,
            );
            enrich_chess_result(engine.root_state(), &mut result);
            attach_search_metadata(
                &mut result,
                search_profile,
                iters,
                outcome.effective_threads,
                search_evaluator_path(search_profile, thread_spec.requested_threads, false),
                engine.config.quartz.as_ref(),
            );
            SearchCommandReply::Search(SearchResponsePayload::Single { result })
        }
        _ => SearchCommandReply::Json(serde_json::json!({
            "error": format!("search_nn not yet supported for {}", game)
        })),
    }
}

fn handle_search_nn_multi(line: &str) -> SearchCommandReply {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "invalid json" }));
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
        return SearchCommandReply::Search(SearchResponsePayload::Multi { results: vec![] });
    }

    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
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
                search_profile,
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
            SearchCommandReply::Search(SearchResponsePayload::Multi { results })
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
                states.push((
                    idx,
                    Gomoku::from_board_12(7, 4, &board_12, player_12),
                    model_tag,
                ));
            }
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    }),
                    &overrides,
                ),
                search_profile,
            );
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
                        &board_raw
                            .iter()
                            .map(|v| v.as_i64().unwrap_or(0) as i8)
                            .collect::<Vec<_>>(),
                        player,
                        variant,
                    )
                } else {
                    Gomoku15::new(variant)
                };
                states.push((idx, state, model_tag));
            }
            let cfg = apply_search_profile(
                apply_search_overrides(gomoku15_quartz(variant), &overrides),
                search_profile,
            );
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
                let passes = job
                    .get("passes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
                    .min(2) as u8;
                let ko_point = job.get("ko_point").and_then(|v| v.as_i64()).and_then(|v| {
                    if v >= 0 {
                        Some(v as u16)
                    } else {
                        None
                    }
                });
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
                };
                states.push((idx, state, model_tag));
            }
            let cfg = apply_search_profile(
                apply_search_overrides(go_quartz(size), &overrides),
                search_profile,
            );
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
                        &board_raw
                            .iter()
                            .map(|v| v.as_i64().unwrap_or(0) as i8)
                            .collect::<Vec<_>>(),
                        player,
                    )
                } else {
                    TicTacToe::initial()
                };
                states.push((idx, state, model_tag));
            }
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                    &overrides,
                ),
                search_profile,
            );
            run_multi_generic!(TicTacToe, states, cfg, 9)
        }
        game if is_chess_game_name(game) => {
            let dual_model = jobs
                .iter()
                .any(|job| job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) != 0);
            let force_batch = jobs.len() > 1 || dual_model;
            let (eval_a, eval_b) = make_eval_pair::<Chess>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                n_actions,
                force_batch,
                dual_model,
            );
            let base_cfg = apply_search_profile(
                apply_search_overrides(chess_quartz(), &overrides),
                search_profile,
            );
            let qcfg = base_cfg.quartz.clone().unwrap_or_default();
            let results = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(jobs.len());
                for job in jobs {
                    let model_tag =
                        job.get("model_tag").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let eval = if model_tag == 1 {
                        eval_b.clone().unwrap_or_else(|| eval_a.clone())
                    } else {
                        eval_a.clone()
                    };
                    let cfg = base_cfg.clone();
                    let qcfg = qcfg.clone();
                    handles.push(scope.spawn(move || {
                        let state = if let Some(fen) = job.get("fen").and_then(|v| v.as_str()) {
                            let mut parsed = Chess::from_fen(fen).unwrap_or_else(|_| {
                                chess_state_from_request(line, game == "chess960")
                            });
                            apply_chess_history_from_json(&mut parsed, &job);
                            parsed
                        } else {
                            chess_state_from_json(&job, game == "chess960")
                        };
                        let engine = MctsEngine::new(state, eval, cfg);
                        let (iterations, stop_reason, p_flip, value, sigma_q, hbar_eff) =
                            match search_profile {
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
                        let (best, policy, total) =
                            collect_sparse_policy(&engine, CHESS_POLICY_ACTIONS);
                        let mut result = serde_json::json!({
                            "best_move": best,
                            "policy": policy,
                            "p_flip": f_or(p_flip, 0.0),
                            "value": f_or(value, 0.0),
                            "sigma_q": f_or(sigma_q, 0.0),
                            "hbar_eff": f_or(hbar_eff, 0.0),
                            "stop_reason": stop_reason,
                            "iterations": iterations.max(total),
                            "dup_rate": engine.par_ctrl.telemetry.snapshot().dup_rate,
                            "max_pending": engine.par_ctrl.telemetry.snapshot().max_pending,
                            "avg_vvalue": engine.par_ctrl.telemetry.snapshot().avg_vvalue,
                        });
                        enrich_chess_result(engine.root_state(), &mut result);
                        attach_search_metadata(
                            &mut result,
                            search_profile,
                            iters,
                            n_threads,
                            search_evaluator_path(search_profile, n_threads, force_batch),
                            engine.config.quartz.as_ref(),
                        );
                        result
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| serde_json::json!({})))
                    .collect::<Vec<_>>()
            });
            SearchCommandReply::Search(SearchResponsePayload::Multi { results })
        }
        _ => SearchCommandReply::Json(serde_json::json!({
            "error": format!("search_nn_multi not yet supported for {}", game)
        })),
    }
}

fn handle_search_nn_multi_session_open(line: &str) -> SearchCommandReply {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "invalid json" }));
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
        return SearchCommandReply::Json(serde_json::json!({ "error": "jobs required" }));
    }
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
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
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    }),
                    &overrides,
                ),
                search_profile,
            );
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::Gomoku(SearchSession {
                states: jobs
                    .into_iter()
                    .map(|job| Some(parse_gomoku7_job(&job)))
                    .collect(),
                eval: make_eval::<Gomoku>(
                    search_profile,
                    n_threads,
                    batch_size,
                    batch_timeout_us,
                    49,
                    force_batch,
                ),
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
            let cfg = apply_search_profile(
                apply_search_overrides(gomoku15_quartz(variant), &overrides),
                search_profile,
            );
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::Gomoku15(Gomoku15Session {
                inner: SearchSession {
                    states: jobs
                        .into_iter()
                        .map(|job| Some(parse_gomoku15_job(&job, variant)))
                        .collect(),
                    eval: make_eval::<Gomoku15>(
                        search_profile,
                        n_threads,
                        batch_size,
                        batch_timeout_us,
                        225,
                        force_batch,
                    ),
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
            let cfg = apply_search_profile(
                apply_search_overrides(go_quartz(size), &overrides),
                search_profile,
            );
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
                    eval: make_eval::<Go>(
                        search_profile,
                        n_threads,
                        batch_size,
                        batch_timeout_us,
                        n_actions,
                        force_batch,
                    ),
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
                apply_search_overrides(chess_quartz(), &overrides),
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
                        search_profile,
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
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                    &overrides,
                ),
                search_profile,
            );
            let qcfg = cfg.quartz.clone();
            SearchSessionAny::TicTacToe(SearchSession {
                states: jobs
                    .into_iter()
                    .map(|job| Some(parse_tictactoe_job(&job)))
                    .collect(),
                eval: make_eval::<TicTacToe>(
                    search_profile,
                    n_threads,
                    batch_size,
                    batch_timeout_us,
                    9,
                    force_batch,
                ),
                cfg,
                qcfg,
                iters,
                n_threads,
                n_actions: 9,
                search_profile,
            })
        }
        _ => {
            return SearchCommandReply::Json(serde_json::json!({
                "error": format!("search_nn_multi_session not yet supported for {}", game)
            }))
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
    search_sessions()
        .lock()
        .unwrap()
        .insert(session_id, session);
    rust_server_trace(
        "session_open_reply",
        serde_json::json!({
            "session_id": session_id,
            "results_len": results.len(),
            "null_results": results.iter().filter(|v| v.is_null()).count(),
        }),
    );
    SearchCommandReply::Search(SearchResponsePayload::Session {
        session_id,
        results,
    })
}

fn handle_search_nn_multi_engine_session_open(line: &str) -> SearchCommandReply {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "invalid json" }));
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
        return SearchCommandReply::Json(serde_json::json!({ "error": "jobs required" }));
    }
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
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
    let job_count = jobs.len();
    let force_batch = job_count > 1;
    let batch_timeout_us = root
        .get("batch_timeout_us")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| default_batch_timeout_us(n_threads, batch_size, job_count));

    let session = match game {
        "gomoku7" => {
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                        min_visits: 15,
                        check_interval: 20,
                        ..Default::default()
                    }),
                    &overrides,
                ),
                search_profile,
            );
            let eval = make_eval::<Gomoku>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                49,
                force_batch,
            );
            let engines = jobs
                .into_iter()
                .map(|job| {
                    Some(MctsEngine::new(
                        parse_gomoku7_job(&job),
                        eval.clone(),
                        cfg.clone(),
                    ))
                })
                .collect();
            SearchSessionAny::EngineGomoku(EngineSearchSession {
                engines,
                cumulative_iters: vec![0; job_count],
                n_threads,
                search_profile,
            })
        }
        _ if parse_gomoku15_variant(game).is_some() => {
            let variant = parse_gomoku15_variant(game).unwrap();
            let cfg = apply_search_profile(
                apply_search_overrides(gomoku15_quartz(variant), &overrides),
                search_profile,
            );
            let eval = make_eval::<Gomoku15>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                225,
                force_batch,
            );
            let engines = jobs
                .into_iter()
                .map(|job| {
                    Some(MctsEngine::new(
                        parse_gomoku15_job(&job, variant),
                        eval.clone(),
                        cfg.clone(),
                    ))
                })
                .collect();
            SearchSessionAny::EngineGomoku15(EngineGomoku15Session {
                inner: EngineSearchSession {
                    engines,
                    cumulative_iters: vec![0; job_count],
                    n_threads,
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
            let cfg = apply_search_profile(
                apply_search_overrides(go_quartz(size), &overrides),
                search_profile,
            );
            let eval = make_eval::<Go>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                n_actions,
                force_batch,
            );
            let engines = jobs
                .into_iter()
                .map(|job| {
                    Some(MctsEngine::new(
                        parse_go_job(&job, size, ruleset, scoring, komi, allow_suicide),
                        eval.clone(),
                        cfg.clone(),
                    ))
                })
                .collect();
            SearchSessionAny::EngineGo(EngineGoSession {
                inner: EngineSearchSession {
                    engines,
                    cumulative_iters: vec![0; job_count],
                    n_threads,
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
                apply_search_overrides(chess_quartz(), &overrides),
                search_profile,
            );
            let eval = make_eval::<Chess>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                CHESS_POLICY_ACTIONS,
                force_batch,
            );
            let engines = jobs
                .into_iter()
                .map(|job| {
                    Some(MctsEngine::new(
                        parse_chess_job(&job, default_960),
                        eval.clone(),
                        cfg.clone(),
                    ))
                })
                .collect();
            SearchSessionAny::EngineChess(EngineChessSession {
                inner: EngineSearchSession {
                    engines,
                    cumulative_iters: vec![0; job_count],
                    n_threads,
                    search_profile,
                },
                default_960,
            })
        }
        "tictactoe" => {
            let cfg = apply_search_profile(
                apply_search_overrides(
                    MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                    &overrides,
                ),
                search_profile,
            );
            let eval = make_eval::<TicTacToe>(
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                9,
                force_batch,
            );
            let engines = jobs
                .into_iter()
                .map(|job| {
                    Some(MctsEngine::new(
                        parse_tictactoe_job(&job),
                        eval.clone(),
                        cfg.clone(),
                    ))
                })
                .collect();
            SearchSessionAny::EngineTicTacToe(EngineSearchSession {
                engines,
                cumulative_iters: vec![0; job_count],
                n_threads,
                search_profile,
            })
        }
        _ => {
            return SearchCommandReply::Json(serde_json::json!({
                "error": format!("search_nn_multi_engine_session not yet supported for {}", game)
            }))
        }
    };
    let mut session = session;
    let results = session.search_with_iters(iters);
    let session_id = next_search_session_id();
    search_sessions()
        .lock()
        .unwrap()
        .insert(session_id, session);
    SearchCommandReply::Search(SearchResponsePayload::Session {
        session_id,
        results,
    })
}

fn handle_search_nn_multi_session_step(line: &str) -> SearchCommandReply {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "invalid json" }));
    };
    let Some(session_id) = root.get("session_id").and_then(|v| v.as_u64()) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "session_id required" }));
    };
    let updates = root
        .get("updates")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut session = {
        let mut sessions = search_sessions().lock().unwrap();
        let Some(session) = sessions.remove(&session_id) else {
            return SearchCommandReply::Json(serde_json::json!({ "error": "unknown session_id" }));
        };
        session
    };
    if let Err(err) = session.apply_updates(&updates) {
        search_sessions()
            .lock()
            .unwrap()
            .insert(session_id, session);
        return SearchCommandReply::Json(serde_json::json!({ "error": err }));
    }
    let results = session.search();
    search_sessions()
        .lock()
        .unwrap()
        .insert(session_id, session);
    SearchCommandReply::Search(SearchResponsePayload::Session {
        session_id,
        results,
    })
}

fn handle_search_nn_multi_engine_session_step(line: &str) -> SearchCommandReply {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "invalid json" }));
    };
    let Some(session_id) = root.get("session_id").and_then(|v| v.as_u64()) else {
        return SearchCommandReply::Json(serde_json::json!({ "error": "session_id required" }));
    };
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let updates = root
        .get("updates")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut session = {
        let mut sessions = search_sessions().lock().unwrap();
        let Some(session) = sessions.remove(&session_id) else {
            return SearchCommandReply::Json(serde_json::json!({ "error": "unknown session_id" }));
        };
        session
    };
    if let Err(err) = session.apply_updates(&updates) {
        search_sessions()
            .lock()
            .unwrap()
            .insert(session_id, session);
        return SearchCommandReply::Json(serde_json::json!({ "error": err }));
    }
    let results = session.search_with_iters(iters);
    search_sessions()
        .lock()
        .unwrap()
        .insert(session_id, session);
    SearchCommandReply::Search(SearchResponsePayload::Session {
        session_id,
        results,
    })
}

fn handle_search_nn_multi_session_close(line: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return "{\"error\":\"invalid json\"}".to_string();
    };
    let Some(session_id) = root.get("session_id").and_then(|v| v.as_u64()) else {
        return "{\"error\":\"session_id required\"}".to_string();
    };
    let removed = search_sessions()
        .lock()
        .unwrap()
        .remove(&session_id)
        .is_some();
    serde_json::json!({ "ok": removed, "session_id": session_id }).to_string()
}

fn build_eval_runner_sessions_generic<G: GameState>(
    sessions: &[ArenaEvalSessionSpec],
    parse_state: impl Fn(&ArenaEvalSessionSpec) -> G,
) -> Vec<EvalRunnerSession<G>> {
    sessions
        .iter()
        .map(|session| EvalRunnerSession {
            game_id: session.game_id.clone(),
            state: parse_state(session),
            black_tag: session.black_tag,
            white_tag: session.white_tag,
            opening: session.opening.clone(),
            seed: session.seed,
            ply: session.ply,
            total_time_ms: session.total_time_ms,
            done: session.done,
            error: None,
            search_root_visits: Vec::new(),
            search_p_flip: Vec::new(),
            search_halt_reasons: std::collections::BTreeMap::new(),
            selection_root_selects: 0,
            selection_refresh_selected_count: 0,
            selection_penalty_abs_sum: 0.0,
            selection_effective_prior_l1_sum: 0.0,
        })
        .collect()
}

fn parse_gomoku7_eval_session(session: &ArenaEvalSessionSpec) -> Gomoku {
    let ArenaEvalStateSpec::Board { player, board } = &session.state else {
        return Gomoku::new(7);
    };
    let player_12: u8 = if *player == 1 { 1 } else { 2 };
    let board_12: Vec<i64> = if board.len() == 49 {
        board
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
    Gomoku::from_board_12(7, 4, &board_12, player_12)
}

fn parse_gomoku15_eval_session(session: &ArenaEvalSessionSpec, variant: GomokuVariant) -> Gomoku15 {
    let ArenaEvalStateSpec::Board { player, board } = &session.state else {
        return Gomoku15::new(variant);
    };
    if board.len() == 225 {
        Gomoku15::from_board(board, *player, variant)
    } else {
        Gomoku15::new(variant)
    }
}

fn parse_go_eval_session(session: &ArenaEvalSessionSpec, size: usize) -> Go {
    let ArenaEvalStateSpec::Go {
        player,
        board,
        ruleset,
        scoring,
        komi,
        allow_suicide,
        passes,
        ko_point,
        black_caps,
        white_caps,
    } = &session.state
    else {
        return Go::new_with_options(size, 7.5, GoRuleset::Chinese, GoScoring::Area, false);
    };
    let side: u8 = if *player == 1 { 1 } else { 2 };
    if board.len() == size * size {
        Go::from_board_with_options(
            size,
            *komi,
            board,
            side,
            *ruleset,
            *scoring,
            *allow_suicide,
            *passes,
            *ko_point,
            *black_caps,
            *white_caps,
        )
    } else {
        Go::new_with_options(size, *komi, *ruleset, *scoring, *allow_suicide)
    }
}

fn parse_tictactoe_eval_session(session: &ArenaEvalSessionSpec) -> TicTacToe {
    let ArenaEvalStateSpec::Board { player, board } = &session.state else {
        return TicTacToe::initial();
    };
    if board.len() == 9 {
        TicTacToe::from_board(board, *player)
    } else {
        TicTacToe::initial()
    }
}

fn parse_chess_eval_session(session: &ArenaEvalSessionSpec, default_960: bool) -> Chess {
    let ArenaEvalStateSpec::Chess {
        fen,
        history_hashes,
    } = &session.state
    else {
        return if default_960 {
            Chess::from_960(518)
        } else {
            Chess::standard()
        };
    };
    let mut json = serde_json::Map::new();
    json.insert("fen".to_string(), serde_json::json!(fen));
    json.insert(
        "chess_history_hashes".to_string(),
        serde_json::json!(history_hashes),
    );
    chess_state_from_json(&serde_json::Value::Object(json), default_960)
}

fn handle_eval_nn_run_generic<G: GameState>(
    game_name: &str,
    sessions_spec: &[ArenaEvalSessionSpec],
    parse_state: impl Fn(&ArenaEvalSessionSpec) -> G,
    n_actions: usize,
    iters: u32,
    max_moves: usize,
    cfg: MctsConfig,
    search_profile: SearchProfile,
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
    typed_response: bool,
) -> EvalCommandReply
where
    usize: From<G::Move>,
{
    let mut sessions = build_eval_runner_sessions_generic(sessions_spec, parse_state);
    let dual_model = sessions
        .iter()
        .any(|sess| sess.black_tag != 0 || sess.white_tag != 0);
    let batch_cfg = crate::mcts::eval::BatchConfig {
        max_batch_size: batch_size.max(n_threads),
        timeout_us: batch_timeout_us,
    };
    let broker = GlobalBroker::<G::Move>::new(n_actions, batch_cfg);
    let eval_a = BatchStdioEval::<G::Move>::from_broker(&broker, 0);
    let eval_b = if dual_model {
        Some(BatchStdioEval::<G::Move>::from_broker(&broker, 1))
    } else {
        None
    };
    let eval_a_shared: Arc<dyn Evaluator<G>> = Arc::new(eval_a.clone());
    let eval_b_shared: Option<Arc<dyn Evaluator<G>>> = eval_b
        .as_ref()
        .map(|eval| Arc::new(eval.clone()) as Arc<dyn Evaluator<G>>);
    let model_tags = sessions
        .iter()
        .map(|sess| Arc::new(AtomicU32::new(sess.active_model_tag())))
        .collect::<Vec<_>>();
    let mut engine_session = EngineSearchSession {
        engines: sessions
            .iter()
            .enumerate()
            .map(|(idx, sess)| {
                if sess.done || sess.state.is_terminal() || sess.ply >= max_moves {
                    None
                } else {
                    Some(MctsEngine::new(
                        sess.state.clone(),
                        Arc::new(TaggedSharedEvaluator {
                            current_tag: model_tags[idx].clone(),
                            eval_a: eval_a_shared.clone(),
                            eval_b: eval_b_shared.clone(),
                        }),
                        cfg.clone(),
                    ))
                }
            })
            .collect(),
        cumulative_iters: vec![0; sessions.len()],
        n_threads,
        search_profile,
    };
    let started = Instant::now();
    let progress_every = (sessions.len() / 10).clamp(1, 25);
    let mut last_reported = 0usize;
    rust_server_trace(
        "eval_runner_start",
        serde_json::json!({
            "game": game_name,
            "session_count": sessions.len(),
            "iters": iters,
            "max_moves": max_moves,
            "n_threads": n_threads,
            "batch_size": batch_size,
            "batch_timeout_us": batch_timeout_us,
            "search_profile": search_profile_name(search_profile),
            "runner_impl": "persistent_engine_session",
        }),
    );

    loop {
        let active_count = sessions
            .iter()
            .enumerate()
            .filter(|(idx, sess)| {
                !sess.done
                    && !sess.state.is_terminal()
                    && sess.ply < max_moves
                    && engine_session
                        .engines
                        .get(*idx)
                        .and_then(|e| e.as_ref())
                        .is_some()
            })
            .count();
        if active_count == 0 {
            break;
        }
        let completed_before = sessions.iter().filter(|sess| sess.done).count();
        let batch_started = Instant::now();
        let results = engine_session.search_with_iters_compact(iters);
        let batch_elapsed_ms = batch_started.elapsed().as_secs_f64() * 1000.0;
        let share_ms = batch_elapsed_ms / active_count.max(1) as f64;
        let mut wave_errors = 0usize;
        let mut wave_nulls = 0usize;
        for idx in 0..sessions.len() {
            let Some(sess) = sessions.get_mut(idx) else {
                continue;
            };
            if sess.done || sess.state.is_terminal() || sess.ply >= max_moves {
                sess.done = true;
                engine_session.deactivate(idx);
                continue;
            }
            let result = results.get(idx).cloned().flatten();
            if result.is_none() {
                wave_nulls += 1;
            }
            let Some(result) = result else {
                sess.error = Some("missing best_move".to_string());
                sess.done = true;
                engine_session.deactivate(idx);
                wave_errors += 1;
                continue;
            };
            sess.search_root_visits.push(result.root_visits);
            sess.search_p_flip.push(result.p_flip);
            *sess
                .search_halt_reasons
                .entry(result.stop_reason.clone())
                .or_insert(0) += 1;
            sess.selection_root_selects = sess
                .selection_root_selects
                .saturating_add(result.selection_root_selects);
            sess.selection_refresh_selected_count = sess
                .selection_refresh_selected_count
                .saturating_add(result.selection_refresh_selected_count);
            sess.selection_penalty_abs_sum += result.selection_penalty_abs_sum;
            sess.selection_effective_prior_l1_sum += result.selection_effective_prior_l1_sum;
            let action = result.best_move;
            let Some(mv) = sess.state.idx_to_move(action as usize) else {
                sess.error = Some(format!("invalid action {} for eval runner", action));
                sess.done = true;
                wave_errors += 1;
                continue;
            };
            let move_time_ms = if result.time_used_ms > 0.0 {
                result.time_used_ms
            } else {
                share_ms
            };
            sess.total_time_ms += move_time_ms;
            sess.state = sess.state.apply_move(mv);
            sess.ply += 1;
            if let Err(err) = engine_session.apply_action_idx(idx, action) {
                sess.error = Some(err);
                sess.done = true;
                wave_errors += 1;
                engine_session.deactivate(idx);
                continue;
            }
            if sess.state.is_terminal() || sess.ply >= max_moves {
                sess.done = true;
                engine_session.deactivate(idx);
            } else {
                model_tags[idx].store(sess.active_model_tag(), Ordering::Relaxed);
            }
        }
        let completed = sessions.iter().filter(|sess| sess.done).count();
        rust_server_trace(
            "eval_runner_wave",
            serde_json::json!({
                "game": game_name,
                "active_games": active_count,
                "completed_before": completed_before,
                "completed_after": completed,
                "newly_completed": completed.saturating_sub(completed_before),
                "wave_errors": wave_errors,
                "wave_nulls": wave_nulls,
                "batch_elapsed_ms": batch_elapsed_ms,
                "share_ms": share_ms,
                "runner_impl": "persistent_engine_session",
            }),
        );
        if completed >= last_reported + progress_every || completed == sessions.len() {
            last_reported = completed;
            rust_server_trace(
                "eval_runner_progress",
                serde_json::json!({
                    "game": game_name,
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
            "game": game_name,
            "completed_games": records.len(),
            "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
            "errors": records.iter().filter(|r| r.get("error").and_then(|v| v.as_str()).is_some()).count(),
        }),
    );
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    if typed_response {
        EvalCommandReply::Binary(encode_arena_eval_response_payload(
            game_name,
            &sessions,
            duration_ms,
        ))
    } else {
        EvalCommandReply::Json(
            serde_json::json!({
                "valid_eval": true,
                "game": game_name,
                "records": records,
                "completed_games": sessions.len(),
                "duration_ms": duration_ms,
            })
            .to_string(),
        )
    }
}

fn handle_eval_nn_run_parts(
    game: &str,
    sessions: Vec<ArenaEvalSessionSpec>,
    iters: u32,
    max_moves: usize,
    options: &ArenaEvalSearchOptions,
    typed_response: bool,
) -> EvalCommandReply {
    if sessions.is_empty() {
        return EvalCommandReply::Json("{\"error\":\"sessions required\"}".to_string());
    }
    let overrides = options.overrides.clone();
    let search_profile = options.search_profile;
    let n_threads = cap_search_threads(options.n_threads.max(1));
    let batch_size = options.batch_size.max(1);
    let batch_timeout_us = if options.batch_timeout_us > 0 {
        options.batch_timeout_us
    } else {
        default_batch_timeout_us(n_threads, batch_size, sessions.len())
    };
    let n_actions = game_n_actions(game);

    macro_rules! dispatch_eval {
        ($parse_state:expr, $game_name:expr, $cfg:expr) => {
            handle_eval_nn_run_generic(
                $game_name,
                &sessions,
                $parse_state,
                n_actions,
                iters,
                max_moves,
                apply_search_profile($cfg, search_profile),
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                typed_response,
            )
        };
    }

    match game {
        "gomoku7" => dispatch_eval!(
            |session| parse_gomoku7_eval_session(session),
            "gomoku7",
            apply_search_overrides(
                MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                }),
                &overrides
            )
        ),
        g if parse_gomoku15_variant(g).is_some() => {
            let variant = parse_gomoku15_variant(g).unwrap();
            dispatch_eval!(
                move |session| parse_gomoku15_eval_session(session, variant),
                g,
                apply_search_overrides(gomoku15_quartz(variant), &overrides)
            )
        }
        g if parse_go_game(g).is_some() => {
            let (size, _default_ruleset) = parse_go_game(g).unwrap();
            dispatch_eval!(
                move |session| parse_go_eval_session(session, size),
                g,
                apply_search_overrides(go_quartz(size), &overrides)
            )
        }
        "tictactoe" => dispatch_eval!(
            |session| parse_tictactoe_eval_session(session),
            "tictactoe",
            apply_search_overrides(
                MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                &overrides
            )
        ),
        g if is_chess_game_name(g) => {
            let default_960 = g == "chess960";
            dispatch_eval!(
                move |session| parse_chess_eval_session(session, default_960),
                g,
                apply_search_overrides(chess_quartz(), &overrides)
            )
        }
        _ => EvalCommandReply::Json(
            serde_json::json!({
                "error": format!("eval_nn_run not supported for {}", game)
            })
            .to_string(),
        ),
    }
}

fn handle_eval_nn_run(line: &str) -> EvalCommandReply {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(line) else {
        return EvalCommandReply::Json("{\"error\":\"invalid json\"}".to_string());
    };
    let game = root
        .get("game")
        .and_then(|v| v.as_str())
        .unwrap_or("gomoku7");
    let sessions = root
        .get("sessions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
    let max_moves = root
        .get("max_moves")
        .and_then(|v| v.as_u64())
        .unwrap_or(500) as usize;
    let typed_response = root
        .get("_typed_arena_eval_resp")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let options = arena_eval_search_options_from_json(game, line, sessions.len());
    let session_specs = arena_eval_session_specs_from_json(game, &sessions);
    handle_eval_nn_run_parts(
        game,
        session_specs,
        iters,
        max_moves,
        &options,
        typed_response,
    )
}

fn handle_eval_nn_run_frame(payload: &[u8]) -> EvalCommandReply {
    let request = match decode_arena_eval_request_payload(payload) {
        Ok(request) => request,
        Err(err) => return EvalCommandReply::Json(serde_json::json!({ "error": err }).to_string()),
    };
    handle_eval_nn_run_parts(
        &request.game,
        request.sessions,
        request.iters,
        request.max_moves,
        &request.search_options,
        true,
    )
}

fn handle_selfplay_nn_run_generic<G: GameState>(
    game_name: &str,
    init_state: G,
    n_actions: usize,
    num_games: usize,
    parallel: usize,
    iters: u32,
    temp_threshold: usize,
    cfg: MctsConfig,
    search_profile: SearchProfile,
    n_threads: usize,
    batch_size: usize,
    batch_timeout_us: u64,
    base_seed: u64,
) -> String
where
    usize: From<G::Move>,
{
    let batch_cfg = crate::mcts::eval::BatchConfig {
        max_batch_size: batch_size.max(n_threads),
        timeout_us: batch_timeout_us,
    };
    let broker = GlobalBroker::<G::Move>::new(n_actions, batch_cfg);
    let eval_a = BatchStdioEval::<G::Move>::from_broker(&broker, 0);
    let slot_count = parallel.max(batch_size).max(1).min(num_games.max(1));
    let mut games_done = 0usize;
    let mut games_started = 0usize;
    let started = Instant::now();
    let progress_every = (num_games / 10).clamp(1, 25);
    let mut last_reported = 0usize;
    let mut sessions: Vec<SelfplaySession<G>> = (0..slot_count)
        .map(|slot| {
            let seed = base_seed.wrapping_add(slot as u64).wrapping_add(1);
            games_started += 1;
            SelfplaySession {
                state: init_state.clone(),
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
        .collect();
    rust_server_trace(
        "selfplay_runner_start",
        serde_json::json!({
            "game": game_name,
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
        // Check cancel flag at wave boundary (cooperative cancellation from Python)
        if let Some(ring) = global_ring_buffer() {
            if ring.cancel_requested() {
                rust_server_trace(
                    "selfplay_runner_cancelled",
                    serde_json::json!({
                        "completed_games": games_done,
                        "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
                    }),
                );
                break; // exit wave loop → emit normal selfplay_done below
            }
        }

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
            cfg.quartz.clone(),
            iters,
            n_threads,
            n_actions,
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
            let policy_entries =
                parse_sparse_policy_value(result.get("policy").unwrap_or(&serde_json::Value::Null));
            if policy_entries.is_empty() {
                sess.finished = true;
                sess.winner = terminal_black_score(&sess.state).unwrap_or(0.0);
            } else {
                sess.board_history.push(sess.state.board_state_record());
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
                    "search_manifest": result.get("search_manifest").cloned().unwrap_or(serde_json::json!({})),
                    "realized_budget": result.get("realized_budget").cloned().unwrap_or(serde_json::json!({})),
                    "controller_summary": result.get("controller_summary").cloned().unwrap_or(serde_json::json!({})),
                }));
                let fallback_best = result
                    .get("best_move")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if let Some(action) = choose_selfplay_action_generic(
                    &mut sess.rng,
                    &sess.state,
                    &policy_entries,
                    sess.moves,
                    temp_threshold,
                    fallback_best,
                    n_actions,
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
                    *sess = SelfplaySession {
                        state: init_state.clone(),
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
                "game": game_name,
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
        "game": game_name,
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
    let game = root
        .get("game")
        .and_then(|v| v.as_str())
        .unwrap_or("gomoku7");
    let iters = root.get("iters").and_then(|v| v.as_u64()).unwrap_or(200) as u32;
    let num_games = root.get("n_games").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let parallel = root.get("parallel").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let temp_threshold = root
        .get("temp_threshold")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as usize;
    let seed = root
        .get("seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0xC0FFEE);
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
    let n_actions = game_n_actions(game);

    macro_rules! dispatch_selfplay {
        ($init_state:expr, $game_name:expr, $cfg:expr) => {
            handle_selfplay_nn_run_generic(
                $game_name,
                $init_state,
                n_actions,
                num_games,
                parallel,
                iters,
                temp_threshold,
                apply_search_profile($cfg, search_profile),
                search_profile,
                n_threads,
                batch_size,
                batch_timeout_us,
                seed,
            )
        };
    }

    match game {
        "gomoku7" => dispatch_selfplay!(
            Gomoku::new_with_win(7, 4),
            "gomoku7",
            apply_search_overrides(
                MctsConfig::evaluation(2.0).with_quartz(QuartzConfig {
                    min_visits: 15,
                    check_interval: 20,
                    ..Default::default()
                }),
                &overrides
            )
        ),
        g if parse_gomoku15_variant(g).is_some() => {
            let variant = parse_gomoku15_variant(g).unwrap();
            dispatch_selfplay!(
                Gomoku15::new(variant),
                g,
                apply_search_overrides(gomoku15_quartz(variant), &overrides)
            )
        }
        g if parse_go_game(g).is_some() => {
            let (size, default_ruleset) = parse_go_game(g).unwrap();
            let ruleset = parse_go_ruleset(line, default_ruleset);
            let scoring = parse_go_scoring(line, ruleset.scoring());
            let komi = parse_go_komi(line, 7.5);
            let allow_suicide = parse_go_allow_suicide(line, false);
            dispatch_selfplay!(
                Go::new_with_options(size, komi, ruleset, scoring, allow_suicide),
                g,
                apply_search_overrides(go_quartz(size), &overrides)
            )
        }
        "tictactoe" => dispatch_selfplay!(
            TicTacToe::initial(),
            "tictactoe",
            apply_search_overrides(
                MctsConfig::evaluation(1.4).with_quartz(QuartzConfig::default()),
                &overrides
            )
        ),
        g if is_chess_game_name(g) => dispatch_selfplay!(
            chess_state_from_request(line, g == "chess960"),
            g,
            apply_search_overrides(chess_quartz(), &overrides)
        ),
        _ => serde_json::json!({
            "error": format!("selfplay_nn_run not supported for {}", game)
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::eval::UniformEval;

    #[test]
    fn test_json() {
        assert_eq!(jstr(r#"{"cmd":"selfplay"}"#, "cmd"), Some("selfplay"));
    }

    #[test]
    fn test_parse_search_thread_spec_explicit() {
        let spec = parse_search_thread_spec(r#"{"n_threads":3}"#);
        assert_eq!(spec.requested_threads, cap_search_threads(3));
        assert!(spec.auto_policy.is_none());
        assert_eq!(spec.policy_name(), "explicit");
    }

    #[test]
    fn test_parse_search_thread_spec_auto_with_cap() {
        let spec = parse_search_thread_spec(r#"{"n_threads":"auto","thread_cap":8}"#);
        assert!(spec.auto_policy.is_some());
        assert_eq!(spec.policy_name(), "auto-throughput");
        assert!(spec.requested_threads >= 1);
        assert!(spec.requested_threads <= cap_search_threads(8));
    }

    #[test]
    fn test_parse_search_thread_spec_quality_alias() {
        let spec = parse_search_thread_spec(r#"{"thread_policy":"quality","max_threads":16}"#);
        assert!(spec.auto_policy.is_some());
        assert_eq!(spec.policy_name(), "auto-quality");
        assert!(spec.requested_threads >= 1);
        assert!(spec.requested_threads <= cap_search_threads(16));
    }

    #[test]
    fn test_execute_search_auto_threads_records_metadata() {
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let cfg = MctsConfig::evaluation(1.4);
        let engine = MctsEngine::new(TicTacToe::initial(), eval, cfg);
        let thread_spec = SearchThreadSpec {
            requested_threads: cap_search_threads(8),
            auto_policy: Some(AutoThreadPolicy::quality().with_max_threads(8)),
        };

        let outcome = execute_search(&engine, thread_spec, 128, None, SearchProfile::Baseline);

        assert_eq!(outcome.thread_policy, "auto-quality");
        assert_eq!(outcome.requested_threads, thread_spec.requested_threads);
        assert!(outcome.effective_threads >= 1);
        assert!(outcome.effective_threads <= outcome.requested_threads);
        assert_eq!(outcome.auto_thread_reason.as_deref(), Some("TinyBudget"));
    }

    #[test]
    fn test_attach_search_metadata_preserves_auto_thread_manifest() {
        let mut result = serde_json::json!({
            "iterations": 128,
            "stop_reason": "BudgetExhausted",
            "requested_threads": 8,
            "effective_threads": 2,
            "thread_policy": "auto-quality",
            "auto_thread_reason": "TinyBudget"
        });

        attach_search_metadata(
            &mut result,
            SearchProfile::Baseline,
            128,
            2,
            "batch_stdio",
            None,
        );

        let manifest = &result["search_manifest"];
        assert_eq!(manifest["n_threads"], 2);
        assert_eq!(manifest["requested_threads"], 8);
        assert_eq!(manifest["effective_threads"], 2);
        assert_eq!(manifest["thread_policy"], "auto-quality");
        assert_eq!(manifest["auto_thread_reason"], "TinyBudget");
        assert_eq!(manifest["evaluator_path"], "batch_stdio");
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
        assert!(
            r.contains("\"fen\":\"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\"")
        );
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
    fn test_p7_parse_halt_mode_override_fixed() {
        // P7 (audit_codex_20260425.md W2): "fixed" maps to
        // HaltMode::Fixed with a sentinel max-budget that effectively
        // disables adaptive halt branches.
        let parsed = parse_halt_mode_override(r#"{"halt_mode":"fixed"}"#);
        assert!(matches!(parsed, Some(HaltMode::Fixed { budget }) if budget == u32::MAX));
    }

    #[test]
    fn test_p7_parse_halt_mode_override_voc() {
        let parsed = parse_halt_mode_override(r#"{"halt_mode":"voc"}"#);
        assert!(matches!(parsed, Some(HaltMode::VOC)));
        let parsed_pascal = parse_halt_mode_override(r#"{"halt_mode":"VOC"}"#);
        assert!(matches!(parsed_pascal, Some(HaltMode::VOC)));
    }

    #[test]
    fn test_p7_parse_halt_mode_override_unknown_returns_none() {
        assert!(parse_halt_mode_override(r#"{"halt_mode":"bogus"}"#).is_none());
        assert!(parse_halt_mode_override(r#"{}"#).is_none());
    }

    #[test]
    fn test_p7_apply_search_overrides_pins_halt_mode() {
        let mut cfg = MctsConfig::evaluation(2.0).with_quartz(QuartzConfig::default());
        let ov = parse_search_overrides(r#"{"halt_mode":"fixed"}"#);
        cfg = apply_search_overrides(cfg, &ov);
        let q = cfg.quartz.expect("quartz config retained");
        assert!(matches!(q.halt_mode, HaltMode::Fixed { budget } if budget == u32::MAX));
    }

    #[test]
    fn test_controller_summary_marks_inert_prior_refresh_rate() {
        let mut qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            prior_refresh_rate: 0.5,
            ..Default::default()
        };
        let mut result = serde_json::json!({
            "iterations": 32,
            "stop_reason": "BudgetExhausted"
        });

        attach_search_metadata(
            &mut result,
            SearchProfile::Quartz,
            32,
            2,
            "batch_stdio",
            Some(&qcfg),
        );

        let coverage = &result["controller_summary"]["actuator_coverage"];
        // P01: schema bumped from 5 to 6 when the `extended` block was
        // added. The block carries controller_penalty_mode_counts,
        // mean_prior_refresh_rate, and halt_reason_count — fields that
        // were claimed in README but never emitted before P01.
        assert_eq!(result["controller_summary"]["schema_version"], 6);
        let extended = &result["controller_summary"]["extended"];
        assert_eq!(extended["schema_version"], 1);
        assert!(extended["controller_penalty_mode_counts"].is_object());
        assert!(extended["halt_reason_count"].is_object());
        assert_eq!(coverage["prior_refresh_rate_configured"], true);
        assert_eq!(coverage["prior_refresh_rate_consumed_by_mode"], false);
        assert_eq!(coverage["prior_refresh_rate_inert_for_mode"], true);
        assert_eq!(coverage["prior_refresh_source"], "prior_q_divergence_gate");

        qcfg.penalty_mode = PenaltyMode::EffectiveV2;
        let mut result = serde_json::json!({
            "iterations": 32,
            "stop_reason": "BudgetExhausted"
        });
        attach_search_metadata(
            &mut result,
            SearchProfile::Quartz,
            32,
            2,
            "batch_stdio",
            Some(&qcfg),
        );

        let coverage = &result["controller_summary"]["actuator_coverage"];
        assert_eq!(coverage["prior_refresh_rate_consumed_by_mode"], true);
        assert_eq!(coverage["prior_refresh_rate_inert_for_mode"], false);
        assert_eq!(
            coverage["prior_refresh_source"],
            "config_prior_refresh_rate"
        );
    }

    /// P01: schema_version 6 must emit `extended` with the full key set
    /// the README claimed but never delivered before P01. This is the
    /// regression guard against telemetry-claim drift.
    #[test]
    fn test_controller_summary_extended_block_has_all_keys() {
        use crate::mcts::quartz::{HaltReason, PenaltyMode, PENALTY_MODE_KEYS};

        let qcfg = QuartzConfig {
            sigma_0: 0.3,
            min_visits: 50,
            ctm_budget_ms: 0,
            penalty_mode: PenaltyMode::Legacy,
            ..Default::default()
        };
        // Synthesize a SearchExecutionOutcome-shaped result by hand-feeding
        // the obj that build_result_value normally emits. This isolates the
        // attach_search_metadata pipeline from the live engine.
        let zero_hist: Vec<u32> = vec![0; crate::mcts::quartz::HALT_REASON_COUNT];
        let mut counts: Vec<u64> = vec![0; PENALTY_MODE_KEYS.len()];
        // Synthesize a non-trivial signal: pretend Legacy mode fired 7 times
        // and PFlipMixture fired 3 times. Aggregator should reflect both.
        counts[0] = 7;
        counts[6] = 3;
        let mut halt_hist: Vec<u32> = zero_hist.clone();
        halt_hist[HaltReason::MaxVisits as usize] = 1;
        halt_hist[HaltReason::PFlipConverged as usize] = 4;
        let mut result = serde_json::json!({
            "iterations": 32,
            "stop_reason": "BudgetExhausted",
            "selection_penalty_mode_invoke_count": counts,
            "selection_refresh_eligible_count": 50_u64,
            "selection_refresh_active_count": 17_u64,
            "halt_reason_count": halt_hist,
        });

        attach_search_metadata(
            &mut result,
            SearchProfile::Quartz,
            32,
            2,
            "batch_stdio",
            Some(&qcfg),
        );

        assert_eq!(result["controller_summary"]["schema_version"], 6);
        let ext = &result["controller_summary"]["extended"];
        assert_eq!(ext["schema_version"], 1);
        assert_eq!(ext["refresh_active_count"], 17);
        assert_eq!(ext["refresh_eligible_count"], 50);
        // Bit-exact float check would be brittle; cap at 4 decimals.
        let measured = ext["mean_prior_refresh_rate"].as_f64().unwrap();
        assert!((measured - 0.34).abs() < 1e-6, "measured rate = {measured}");
        let pm = ext["controller_penalty_mode_counts"].as_object().unwrap();
        assert_eq!(pm["Legacy"], 7);
        assert_eq!(pm["PFlipMixture"], 3);
        assert_eq!(pm["EffectiveV2"], 0);
        let hr = ext["halt_reason_count"].as_object().unwrap();
        assert_eq!(hr["MaxVisits"], 1);
        assert_eq!(hr["PFlipConverged"], 4);
        // Reserved keys still present with zero count — important for
        // downstream aggregators that index by key, not position.
        assert_eq!(hr["KLLUCBStop"], 0);
        assert_eq!(hr["GLRCertified"], 0);
    }

    /// P01: zero-eligible-count must NOT divide by zero — the JSON
    /// reports `null` instead of NaN/Infinity. Empty refresh path
    /// (e.g. PenaltyMode::None on a non-Quartz profile) is the
    /// standard case where this matters.
    #[test]
    fn test_controller_summary_extended_handles_zero_eligible() {
        use crate::mcts::quartz::PenaltyMode;
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::None,
            ..Default::default()
        };
        let zero_pm: Vec<u64> = vec![0; crate::mcts::quartz::PENALTY_MODE_COUNT];
        let zero_hr: Vec<u32> = vec![0; crate::mcts::quartz::HALT_REASON_COUNT];
        let mut result = serde_json::json!({
            "iterations": 0,
            "stop_reason": "BudgetExhausted",
            "selection_penalty_mode_invoke_count": zero_pm,
            "selection_refresh_eligible_count": 0_u64,
            "selection_refresh_active_count": 0_u64,
            "halt_reason_count": zero_hr,
        });
        attach_search_metadata(
            &mut result,
            SearchProfile::Quartz,
            32,
            2,
            "batch_stdio",
            Some(&qcfg),
        );
        let ext = &result["controller_summary"]["extended"];
        assert_eq!(ext["refresh_eligible_count"], 0);
        assert!(ext["mean_prior_refresh_rate"].is_null());
    }

    #[test]
    fn test_parse_sparse_policy_value_accepts_legacy_and_numeric_pairs() {
        let policy = parse_sparse_policy_value(&serde_json::json!([
            "1:0.25",
            [3, 0.75],
            {"bad": true},
        ]));

        assert_eq!(policy, vec![(1, 0.25), (3, 0.75)]);
    }

    #[test]
    fn test_parse_chess_job_preserves_history_hashes_for_exact_tt() {
        let state = Chess::standard()
            .apply_move(
                Chess::standard()
                    .legal_moves()
                    .into_iter()
                    .find(|mv| mv.to_uci() == "g1f3")
                    .unwrap(),
            )
            .apply_move(
                Chess::from_fen("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1")
                    .unwrap()
                    .legal_moves()
                    .into_iter()
                    .find(|mv| mv.to_uci() == "g8f6")
                    .unwrap(),
            );
        let parsed = parse_chess_job(
            &serde_json::json!({
                "fen": state.to_fen(),
                "chess_history_hashes": state.history_hashes(),
            }),
            false,
        );

        assert_eq!(parsed.tt_hash(), state.tt_hash());
    }

    #[test]
    fn test_encode_search_response_payload_supports_single_result_wrapper() {
        let payload = encode_search_response_payload(&SearchResponsePayload::Single {
            result: serde_json::json!({
                "best_move": 17,
                "policy": [[3, 0.6], [5, 0.4]],
                "iterations": 321,
                "result_history_hashes": [101, 202],
                "best_move_uci": "e2e4",
                "result_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            }),
        });

        assert_eq!(payload[0], SEARCH_RESP_SINGLE);
        assert!(!payload.is_empty());
    }

    #[test]
    fn test_chess_session_updates_apply_action_and_deactivate() {
        let state = Chess::standard();
        let action = chess_policy_index(
            &state,
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
    fn test_baseline_strict_forces_batched_eval_even_single_thread() {
        assert!(should_use_batch_eval(
            SearchProfile::BaselineStrict,
            1,
            false
        ));
        assert!(should_use_batch_eval(SearchProfile::Baseline, 4, false));
        assert!(!should_use_batch_eval(SearchProfile::Baseline, 1, false));
    }

    #[test]
    fn test_parse_go_game_aliases() {
        assert_eq!(parse_go_game("go9"), Some((9, GoRuleset::Chinese)));
        assert_eq!(parse_go_game("go9_jp"), Some((9, GoRuleset::Japanese)));
        assert_eq!(parse_go_game("go19_kr"), Some((19, GoRuleset::Korean)));
    }

    #[test]
    fn test_engine_session_step_iters_are_delta_not_absolute() {
        let state = Gomoku::new(7);
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(4));
        let cfg = MctsConfig::evaluation(2.0);
        let mut session = EngineSearchSession {
            engines: vec![Some(MctsEngine::new(state, eval, cfg))],
            cumulative_iters: vec![0],
            n_threads: 1,
            search_profile: SearchProfile::Baseline,
        };

        let first = session.search_with_iters(8);
        let first_iters = first[0]
            .get("iterations")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let second = session.search_with_iters(8);
        let second_iters = second[0]
            .get("iterations")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        assert!(
            second_iters >= first_iters + 8,
            "engine session steps must add visits cumulatively: first={}, second={}",
            first_iters,
            second_iters
        );
    }

    #[test]
    fn test_engine_session_multi_engine_matches_serial_reference() {
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(4));
        let cfg = MctsConfig::evaluation(2.0);
        let state_a = Gomoku::new(7);
        let state_b = state_a.apply_move(state_a.idx_to_move(24).unwrap());

        let mut session = EngineSearchSession {
            engines: vec![
                Some(MctsEngine::new(state_a.clone(), eval.clone(), cfg.clone())),
                Some(MctsEngine::new(state_b.clone(), eval.clone(), cfg.clone())),
            ],
            cumulative_iters: vec![0, 0],
            n_threads: 1,
            search_profile: SearchProfile::Baseline,
        };

        let ref_engine_a = MctsEngine::new(state_a, eval.clone(), cfg.clone());
        let ref_engine_b = MctsEngine::new(state_b, eval, cfg);
        let ref_a = run_eval_search_step(
            &ref_engine_a,
            1,
            32,
            ref_engine_a.config.quartz.clone(),
            SearchProfile::Baseline,
        )
        .unwrap();
        let ref_b = run_eval_search_step(
            &ref_engine_b,
            1,
            32,
            ref_engine_b.config.quartz.clone(),
            SearchProfile::Baseline,
        )
        .unwrap();

        let got = session.search_with_iters(32);
        let got_a = &got[0];
        let got_b = &got[1];

        assert_eq!(
            got_a.get("best_move").and_then(|v| v.as_u64()),
            Some(ref_a.best_move as u64)
        );
        assert_eq!(
            got_b.get("best_move").and_then(|v| v.as_u64()),
            Some(ref_b.best_move as u64)
        );
        assert_eq!(
            got_a.get("iterations").and_then(|v| v.as_u64()),
            Some(ref_a.iterations as u64)
        );
        assert_eq!(
            got_b.get("iterations").and_then(|v| v.as_u64()),
            Some(ref_b.iterations as u64)
        );
    }

    #[test]
    fn test_engine_session_compact_step_iters_are_delta_not_absolute() {
        let state = Gomoku::new(7);
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(4));
        let cfg = MctsConfig::evaluation(2.0);
        let mut session = EngineSearchSession {
            engines: vec![Some(MctsEngine::new(state, eval, cfg))],
            cumulative_iters: vec![0],
            n_threads: 1,
            search_profile: SearchProfile::Baseline,
        };

        let first = session.search_with_iters_compact(8);
        let first_iters = first[0].as_ref().map(|v| v.iterations).unwrap_or(0);
        let second = session.search_with_iters_compact(8);
        let second_iters = second[0].as_ref().map(|v| v.iterations).unwrap_or(0);

        assert!(
            second_iters >= first_iters + 8,
            "compact engine session steps must add visits cumulatively: first={}, second={}",
            first_iters,
            second_iters
        );
    }

    #[test]
    fn test_engine_session_compact_matches_json_reference() {
        let eval: Arc<ShortRollout> = Arc::new(ShortRollout::new(4));
        let cfg = MctsConfig::evaluation(2.0);
        let state = Gomoku::new(7);

        let mut compact_session = EngineSearchSession {
            engines: vec![Some(MctsEngine::new(
                state.clone(),
                eval.clone(),
                cfg.clone(),
            ))],
            cumulative_iters: vec![0],
            n_threads: 1,
            search_profile: SearchProfile::Baseline,
        };
        let mut json_session = EngineSearchSession {
            engines: vec![Some(MctsEngine::new(state, eval, cfg))],
            cumulative_iters: vec![0],
            n_threads: 1,
            search_profile: SearchProfile::Baseline,
        };

        let compact = compact_session.search_with_iters_compact(16);
        let json = json_session.search_with_iters(16);
        let compact = compact[0].as_ref().expect("compact result");
        let json = &json[0];

        assert_eq!(
            json.get("best_move").and_then(|v| v.as_u64()),
            Some(compact.best_move as u64)
        );
        assert_eq!(
            json.get("iterations").and_then(|v| v.as_u64()),
            Some(compact.iterations as u64)
        );
        assert_eq!(
            json.get("p_flip").and_then(|v| v.as_f64()),
            Some(compact.p_flip as f64)
        );
    }

    #[test]
    fn test_decode_arena_eval_request_payload_v1_restores_board_session() {
        let mut payload = Vec::new();
        push_u8(&mut payload, 1);
        push_string(&mut payload, "gomoku7");
        push_string(
            &mut payload,
            r#"{"search_profile":"baseline","n_threads":1,"batch_size":8}"#,
        );
        push_u32(&mut payload, 32);
        push_u32(&mut payload, 64);
        push_u32(&mut payload, 1);
        push_string(&mut payload, "m0::g0000");
        push_u32(&mut payload, 7);
        push_u32(&mut payload, 11);
        push_u64(&mut payload, u64::MAX);
        push_u32(&mut payload, 2);
        push_f64(&mut payload, 12.5);
        push_u8(&mut payload, 0);
        push_u32(&mut payload, 2);
        push_u32(&mut payload, 3);
        push_u32(&mut payload, 5);
        push_u8(&mut payload, ARENA_STATE_BOARD);
        payload.extend_from_slice(&(1_i32).to_le_bytes());
        push_u32(&mut payload, 4);
        payload.extend_from_slice(&[1u8, 0u8, 255u8, 0u8]);

        let request = decode_arena_eval_request_payload(&payload).unwrap();

        assert_eq!(request.game, "gomoku7");
        assert!(matches!(
            request.search_options.search_profile,
            SearchProfile::Baseline
        ));
        assert_eq!(request.search_options.n_threads, 1);
        assert_eq!(request.search_options.batch_size, 8);
        assert_eq!(request.iters, 32);
        assert_eq!(request.max_moves, 64);
        assert_eq!(request.sessions.len(), 1);
        let session = &request.sessions[0];
        assert_eq!(session.game_id, "m0::g0000");
        assert_eq!(session.black_tag, 7);
        assert_eq!(session.white_tag, 11);
        assert_eq!(session.seed, None);
        assert_eq!(session.ply, 2);
        assert_eq!(session.opening, vec![3, 5]);
        match &session.state {
            ArenaEvalStateSpec::Board { player, board } => {
                assert_eq!(*player, 1);
                assert_eq!(board, &vec![1, 0, -1, 0]);
            }
            other => panic!(
                "expected board state, got {:?}",
                std::mem::discriminant(other)
            ),
        }
    }

    #[test]
    fn test_decode_arena_eval_request_payload_v2_restores_typed_options_and_board_session() {
        let mut payload = Vec::new();
        push_u8(&mut payload, ARENA_EVAL_REQ_VERSION);
        push_string(&mut payload, "gomoku7");
        push_string(&mut payload, "baseline");
        push_string(&mut payload, "GatedRefresh");
        push_string(&mut payload, "adaptive");
        push_u32(&mut payload, 3);
        push_u32(&mut payload, 16);
        push_u32(&mut payload, 2400);
        push_f32(&mut payload, 0.25);
        push_f32(&mut payload, 0.45);
        push_u32(&mut payload, 12);
        push_u32(&mut payload, 33);
        push_f32(&mut payload, 0.5);
        push_f32(&mut payload, 1.25);
        push_f32(&mut payload, 1.8);
        payload.extend_from_slice(&(1_i8).to_le_bytes());
        payload.extend_from_slice(&(0_i8).to_le_bytes());
        push_u8(&mut payload, 1);
        push_u64(&mut payload, 777);
        push_u32(&mut payload, 32);
        push_u32(&mut payload, 64);
        push_u32(&mut payload, 1);
        push_string(&mut payload, "m0::g0000");
        push_u32(&mut payload, 7);
        push_u32(&mut payload, 11);
        push_u64(&mut payload, u64::MAX);
        push_u32(&mut payload, 2);
        push_f64(&mut payload, 12.5);
        push_u8(&mut payload, 0);
        push_u32(&mut payload, 2);
        push_u32(&mut payload, 3);
        push_u32(&mut payload, 5);
        push_u8(&mut payload, ARENA_STATE_BOARD);
        payload.extend_from_slice(&(1_i32).to_le_bytes());
        push_u32(&mut payload, 4);
        payload.extend_from_slice(&[1u8, 0u8, 255u8, 0u8]);

        let request = decode_arena_eval_request_payload(&payload).unwrap();

        assert_eq!(request.game, "gomoku7");
        assert!(matches!(
            request.search_options.search_profile,
            SearchProfile::Baseline
        ));
        assert_eq!(request.search_options.n_threads, 3);
        assert_eq!(request.search_options.batch_size, 16);
        assert_eq!(request.search_options.batch_timeout_us, 2400);
        assert!(matches!(
            request.search_options.overrides.penalty_mode,
            PenaltyMode::GatedRefresh
        ));
        assert_eq!(
            request.search_options.overrides.vl_mode.as_deref(),
            Some("adaptive")
        );
        assert_eq!(request.search_options.overrides.min_visits, Some(12));
        assert_eq!(request.search_options.overrides.check_interval, Some(33));
        assert_eq!(
            request.search_options.overrides.root_only_shaping,
            Some(true)
        );
        assert_eq!(request.search_options.overrides.tt_enabled, Some(false));
        assert_eq!(request.search_options.overrides.seed, Some(777));
        assert_eq!(request.iters, 32);
        assert_eq!(request.max_moves, 64);
        assert_eq!(request.sessions.len(), 1);
        let session = &request.sessions[0];
        assert_eq!(session.game_id, "m0::g0000");
        assert_eq!(session.black_tag, 7);
        assert_eq!(session.white_tag, 11);
        assert_eq!(session.seed, None);
        assert_eq!(session.ply, 2);
        assert_eq!(session.opening, vec![3, 5]);
        match &session.state {
            ArenaEvalStateSpec::Board { player, board } => {
                assert_eq!(*player, 1);
                assert_eq!(board, &vec![1, 0, -1, 0]);
            }
            other => panic!(
                "expected board state, got {:?}",
                std::mem::discriminant(other)
            ),
        }
    }

    #[test]
    fn test_tagged_shared_evaluator_switches_model_by_atomic_tag() {
        #[derive(Clone)]
        struct MarkerEval {
            value: f32,
        }

        impl Evaluator<Gomoku> for MarkerEval {
            fn evaluate(&self, state: &Gomoku) -> EvalResult<<Gomoku as GameState>::Move> {
                let legal = state.legal_moves();
                EvalResult::uniform(&legal, self.value)
            }
        }

        let tag = Arc::new(AtomicU32::new(0));
        let eval = TaggedSharedEvaluator::<Gomoku> {
            current_tag: tag.clone(),
            eval_a: Arc::new(MarkerEval { value: 0.25 }),
            eval_b: Some(Arc::new(MarkerEval { value: -0.5 })),
        };
        let state = Gomoku::new(7);

        let first = eval.evaluate(&state);
        assert!((first.value - 0.25).abs() < 1e-6);

        tag.store(1, Ordering::Relaxed);
        let second = eval.evaluate(&state);
        assert!((second.value + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_search_overrides_apply_seed_to_mcts_config() {
        let overrides = parse_search_overrides(r#"{"seed":1234}"#);
        let cfg = apply_search_overrides(MctsConfig::default(), &overrides);
        assert_eq!(cfg.seed, Some(1234));
    }

    #[test]
    fn test_global_inflight_credit_basic() {
        let credit = GlobalInflightCredit::new(3);
        // Acquire all 3
        let p1 = credit.try_acquire();
        let p2 = credit.try_acquire();
        let p3 = credit.try_acquire();
        assert!(p1.is_some());
        assert!(p2.is_some());
        assert!(p3.is_some());
        // 4th should fail
        assert!(credit.try_acquire().is_none());
        assert_eq!(credit.peak(), 3);
        // Drop one → can acquire again
        drop(p2);
        let p4 = credit.try_acquire();
        assert!(p4.is_some());
        assert!(credit.try_acquire().is_none());
        // Drop all → remaining restored
        drop(p1);
        drop(p3);
        drop(p4);
        assert_eq!(
            credit.remaining.load(std::sync::atomic::Ordering::Relaxed),
            3
        );
    }

    #[test]
    fn test_credit_permit_raii_on_panic() {
        let credit = std::sync::Arc::new(GlobalInflightCredit::new(5));
        let c2 = credit.clone();
        let handle = std::thread::spawn(move || {
            let _p1 = c2.try_acquire().unwrap();
            let _p2 = c2.try_acquire().unwrap();
            panic!("intentional panic to test RAII");
        });
        let _ = handle.join(); // join panicked thread
                               // Both permits should have been released via Drop
        assert_eq!(
            credit.remaining.load(std::sync::atomic::Ordering::Relaxed),
            5
        );
    }

    #[test]
    fn test_credits_in_equals_credits_out() {
        use std::sync::Arc;
        let credit = Arc::new(GlobalInflightCredit::new(100));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let c = credit.clone();
            let b = barrier.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                for _ in 0..1000 {
                    if let Some(permit) = c.try_acquire() {
                        // Hold briefly then release via drop
                        std::hint::black_box(&permit);
                        drop(permit);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            credit.remaining.load(std::sync::atomic::Ordering::Relaxed),
            100,
            "credits leaked: remaining != capacity"
        );
    }
}
