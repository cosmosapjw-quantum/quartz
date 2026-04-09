//! JSONL 구조화 로거
//!
//! docs/LOGGING_METRICS.md 요구사항 구현:
//!   - 매 착수: move_time_ms, nps, root_visits, root_entropy, best_move,
//!              tt_hit_rate, tt_size, difficulty_bucket
//!   - 게임 종료: outcome, move_count, stage_id
//!   - 공통 메타: timestamp, stage_id, engine_mode
//!
//! 포맷: JSONL (1 line = 1 event)
//! 출력: File 또는 Stderr (None이면 비활성)

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

// ─────────────────────────────────────────────
// § 이벤트 타입
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineMode {
    Baseline,
    Quartz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DifficultyBucket {
    Easy,
    Average,
    Hard,
}

/// root entropy H(π) 기반 난이도 분류
pub fn classify_difficulty(entropy: f32) -> DifficultyBucket {
    if entropy < 0.5 {
        DifficultyBucket::Easy
    } else if entropy < 1.5 {
        DifficultyBucket::Average
    } else {
        DifficultyBucket::Hard
    }
}

/// 매 착수 로그
#[derive(Debug, Serialize)]
pub struct MoveEvent {
    pub event: &'static str, // "move"
    pub timestamp_ms: u64,
    pub stage_id: String,
    pub engine_mode: EngineMode,
    pub move_idx: u32,
    pub player: i8,
    pub best_move: usize,
    pub move_time_ms: u64,
    pub root_visits: u32,
    pub nps: f64,
    pub root_entropy: f32,
    pub root_value: f32,
    pub tt_hit_rate: f64,
    pub tt_size: usize,
    pub difficulty_bucket: DifficultyBucket,
}

/// 게임 종료 로그
#[derive(Debug, Serialize)]
pub struct GameEvent {
    pub event: &'static str, // "game"
    pub timestamp_ms: u64,
    pub stage_id: String,
    pub engine_mode: EngineMode,
    pub outcome: f32, // +1 / -1 / 0 (최초 플레이어 관점)
    pub move_count: u32,
}

// ─────────────────────────────────────────────
// § Logger
// ─────────────────────────────────────────────

enum LogDest {
    File(BufWriter<File>),
    Buffer(Vec<String>), // 테스트용 in-memory 버퍼
}

pub struct Logger {
    inner: Mutex<LogDest>,
    pub active: bool,
}

impl Logger {
    /// 파일로 출력하는 로거
    pub fn to_file(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Logger {
            inner: Mutex::new(LogDest::File(BufWriter::new(file))),
            active: true,
        })
    }

    /// 테스트용 in-memory 버퍼 로거
    pub fn in_memory() -> Self {
        Logger {
            inner: Mutex::new(LogDest::Buffer(Vec::new())),
            active: true,
        }
    }

    /// 비활성 로거 (no-op)
    pub fn null() -> Self {
        Logger {
            inner: Mutex::new(LogDest::Buffer(Vec::new())),
            active: false,
        }
    }

    /// 버퍼 내용 반환 (테스트용)
    pub fn drain_buffer(&self) -> Vec<String> {
        let mut dest = self.inner.lock().unwrap();
        match &mut *dest {
            LogDest::Buffer(buf) => {
                let out = buf.clone();
                buf.clear();
                out
            }
            _ => vec![],
        }
    }

    /// 이벤트를 JSONL로 기록
    pub fn log<T: Serialize>(&self, event: &T) {
        if !self.active {
            return;
        }
        let line = match serde_json::to_string(event) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[logger] serialize error: {}", e);
                return;
            }
        };
        let mut dest = self.inner.lock().unwrap();
        match &mut *dest {
            LogDest::File(writer) => {
                let _ = writeln!(writer, "{}", line);
                let _ = writer.flush();
            }
            LogDest::Buffer(buf) => {
                buf.push(line);
            }
        }
    }

    /// 착수 이벤트 헬퍼
    pub fn log_move(
        &self,
        stage_id: &str,
        engine_mode: EngineMode,
        move_idx: u32,
        player: i8,
        best_move: usize,
        move_time_ms: u64,
        root_visits: u32,
        nps: f64,
        root_entropy: f32,
        root_value: f32,
        tt_hit_rate: f64,
        tt_size: usize,
    ) {
        let bucket = classify_difficulty(root_entropy);
        self.log(&MoveEvent {
            event: "move",
            timestamp_ms: unix_ms(),
            stage_id: stage_id.to_string(),
            engine_mode,
            move_idx,
            player,
            best_move,
            move_time_ms,
            root_visits,
            nps,
            root_entropy,
            root_value,
            tt_hit_rate,
            tt_size,
            difficulty_bucket: bucket,
        });
    }

    /// 게임 종료 이벤트 헬퍼
    pub fn log_game(&self, stage_id: &str, engine_mode: EngineMode, outcome: f32, move_count: u32) {
        self.log(&GameEvent {
            event: "game",
            timestamp_ms: unix_ms(),
            stage_id: stage_id.to_string(),
            engine_mode,
            outcome,
            move_count,
        });
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────
// § root_value 추출 헬퍼
// ─────────────────────────────────────────────

/// 루트 엣지의 가중 평균 Q (루트 value 추정)
pub fn root_value_estimate(visit_counts: &[u32], q_values: &[f32]) -> f32 {
    let total: u32 = visit_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    visit_counts
        .iter()
        .zip(q_values.iter())
        .map(|(&n, &q)| n as f32 * q)
        .sum::<f32>()
        / total as f32
}
