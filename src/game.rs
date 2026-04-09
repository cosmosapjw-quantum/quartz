//! specs/010-game-api.md 구현
//!
//! GameState 트레이트: 게임-무관(game-agnostic) 인터페이스
//! Evaluator 트레이트: NN / random playout 양쪽을 동일 인터페이스로 사용
//!
//! 핵심 설계 원칙:
//!   - apply_move: pure function (입력 불변, 새 상태 반환)
//!   - outcome: negamax convention (현재 플레이어 관점, +1/-1/0)
//!   - current_player: +1 or -1

use std::fmt::Debug;
use std::hash::Hash;

// ─────────────────────────────────────────────
// § 1. GameState 트레이트
// ─────────────────────────────────────────────

#[allow(dead_code)]
pub trait GameState: Clone + Send + Sync + 'static {
    /// 이 게임의 착수(이동) 타입
    type Move: Copy + Eq + Hash + Send + Sync + Debug + 'static;

    /// 게임 초기 상태 생성
    fn initial() -> Self;

    /// 현재 차례 플레이어 (+1 or -1)
    fn current_player(&self) -> i8;

    /// 현재 상태에서 합법 착수 목록
    fn legal_moves(&self) -> Vec<Self::Move>;

    /// 착수를 적용해 새 상태 반환 (pure function, self 불변)
    fn apply_move(&self, mv: Self::Move) -> Self;

    /// 게임 종료 여부
    fn is_terminal(&self) -> bool;

    /// 게임 결과 — negamax convention:
    ///   현재 플레이어(current_player) 관점으로 +1(승) / -1(패) / 0(무)
    ///   terminal이 아닌 상태에서는 호출하지 않는다.
    fn outcome(&self) -> f32;

    /// Zobrist 해시 (TT 키)
    fn hash(&self) -> u64;

    /// 전체 행동 공간 크기 (NN 정책 출력 차원)
    fn num_actions(&self) -> usize;

    /// 착수 → 행동 인덱스
    fn move_to_idx(&self, mv: Self::Move) -> usize;

    /// 행동 인덱스 → 착수 (불합법이면 None)
    fn idx_to_move(&self, idx: usize) -> Option<Self::Move>;

    /// NN 입력용 feature plane (flat, channel-first)
    /// 기본 구현: 빈 vec (NN stub 사용 시)
    fn encode_planes(&self) -> Vec<f32> {
        vec![]
    }

    /// [OPT] Check if move wins for current player without full apply_move.
    /// Default: apply_move + check terminal (expensive).
    /// Override for O(1) win check (e.g., Gomoku line scan).
    fn is_winning_move(&self, mv: Self::Move) -> bool {
        let next = self.apply_move(mv);
        next.is_terminal() && next.outcome() != 0.0
    }

    /// [OPT] Pick a random legal move without Vec allocation.
    /// Default: legal_moves() + choose (allocates Vec).
    /// Override for O(n) board scan with no allocation.
    fn random_legal_move(&self, rand_idx: usize) -> Option<Self::Move> {
        let moves = self.legal_moves();
        if moves.is_empty() {
            None
        } else {
            Some(moves[rand_idx % moves.len()])
        }
    }

    /// [OPT] Count of legal moves without allocating Vec.
    fn legal_move_count(&self) -> usize {
        self.legal_moves().len()
    }
}

// ─────────────────────────────────────────────
// § 2. Evaluator 트레이트 및 결과 타입
// ─────────────────────────────────────────────

/// Leaf 평가 결과
#[derive(Debug, Clone)]
pub struct EvalResult<M> {
    /// (착수, 사전 확률) 쌍 — 합 ≈ 1.0, 합법수만 포함
    pub policy: Vec<(M, f32)>,

    /// 현재 플레이어 관점 가치: [-1.0, 1.0]
    pub value: f32,
}

impl<M: Copy + Eq + Hash + Debug> EvalResult<M> {
    /// 균등 사전 확률 + 지정 가치로 생성
    pub fn uniform(legal_moves: &[M], value: f32) -> Self {
        let n = legal_moves.len();
        let p = if n > 0 { 1.0 / n as f32 } else { 0.0 };
        EvalResult {
            policy: legal_moves.iter().map(|&m| (m, p)).collect(),
            value,
        }
    }
}

/// Leaf 평가기 트레이트
/// NN 추론, 랜덤 플레이아웃, 균등 사전 확률 등을 동일 인터페이스로 사용
pub trait Evaluator<G: GameState>: Send + Sync {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move>;
}
