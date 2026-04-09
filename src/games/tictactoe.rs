//! TicTacToe 3×3 — GameState 구현체
//!
//! 설계:
//!   - Board: [i8; 9]  (+1=X, -1=O, 0=Empty)
//!   - current_player: +1 or -1
//!   - outcome: negamax — 현재 플레이어 관점
//!   - Zobrist hash: 고정 시드, 증분 XOR

use crate::game::GameState;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

// ─────────────────────────────────────────────
// § Zobrist 테이블 (thread-local, 프로그램 시작시 1회 초기화)
// ─────────────────────────────────────────────

struct ZobTable {
    piece: [[u64; 9]; 2], // [player_idx 0=X 1=O][square]
    side: u64,            // O 차례일 때 XOR
}

impl ZobTable {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut piece = [[0u64; 9]; 2];
        for p in 0..2 {
            for sq in 0..9 {
                piece[p][sq] = rng.gen();
            }
        }
        ZobTable {
            piece,
            side: rng.gen(),
        }
    }
    fn piece_hash(&self, player: i8, sq: usize) -> u64 {
        let pi = if player > 0 { 0 } else { 1 };
        self.piece[pi][sq]
    }
}

thread_local! {
    static ZOB: ZobTable = ZobTable::new(0xDEAD_BEEF_CAFE_1234);
}

// ─────────────────────────────────────────────
// § TicTacToe 상태
// ─────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct TicTacToe {
    board: [i8; 9],     // +1=X, -1=O, 0=empty
    current_player: i8, // +1 or -1
    hash: u64,          // Zobrist 증분 해시
}

impl TicTacToe {
    pub fn from_board(board: &[i8], player: i8) -> Self {
        let mut state = TicTacToe {
            board: [0; 9],
            current_player: if player >= 0 { 1 } else { -1 },
            hash: 0,
        };
        for i in 0..9.min(board.len()) {
            state.board[i] = match board[i] {
                v if v > 0 => 1,
                v if v < 0 => -1,
                _ => 0,
            };
        }
        ZOB.with(|z| {
            for (sq, piece) in state.board.iter().copied().enumerate() {
                if piece != 0 {
                    state.hash ^= z.piece_hash(piece, sq);
                }
            }
            if state.current_player < 0 {
                state.hash ^= z.side;
            }
        });
        state
    }

    fn check_winner(&self) -> i8 {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];
        for [a, b, c] in &LINES {
            let v = self.board[*a];
            if v != 0 && v == self.board[*b] && v == self.board[*c] {
                return v; // +1 (X) 또는 -1 (O)
            }
        }
        0
    }
}

impl std::fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = |v: i8| match v {
            1 => 'X',
            -1 => 'O',
            _ => '.',
        };
        let b = &self.board;
        writeln!(f, " {} | {} | {}", s(b[0]), s(b[1]), s(b[2]))?;
        writeln!(f, "---+---+---")?;
        writeln!(f, " {} | {} | {}", s(b[3]), s(b[4]), s(b[5]))?;
        writeln!(f, "---+---+---")?;
        write!(f, " {} | {} | {}", s(b[6]), s(b[7]), s(b[8]))
    }
}

// ─────────────────────────────────────────────
// § GameState 구현
// ─────────────────────────────────────────────

impl GameState for TicTacToe {
    type Move = usize; // 0..8 (보드 위치)

    fn initial() -> Self {
        TicTacToe {
            board: [0; 9],
            current_player: 1,
            hash: 0,
        }
    }

    fn current_player(&self) -> i8 {
        self.current_player
    }

    fn legal_moves(&self) -> Vec<usize> {
        if self.is_terminal() {
            return vec![];
        }
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn apply_move(&self, mv: usize) -> Self {
        let mut next = self.clone();
        // 보드 갱신
        next.board[mv] = self.current_player;
        // Zobrist 증분 갱신: 기물 배치 + 차례 전환
        ZOB.with(|z| {
            next.hash ^= z.piece_hash(self.current_player, mv);
            next.hash ^= z.side;
        });
        // 차례 전환
        next.current_player = -self.current_player;
        next
    }

    fn is_terminal(&self) -> bool {
        self.check_winner() != 0 || self.board.iter().all(|&v| v != 0)
    }

    /// negamax convention: 현재 플레이어 관점
    /// X가 방금 이겼으면 current_player = O → outcome = -1 (O 패배)
    fn outcome(&self) -> f32 {
        let winner = self.check_winner();
        if winner == self.current_player {
            1.0
        } else if winner != 0 {
            -1.0
        } else {
            0.0
        } // draw
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn num_actions(&self) -> usize {
        9
    }

    fn move_to_idx(&self, mv: usize) -> usize {
        mv
    }

    fn idx_to_move(&self, idx: usize) -> Option<usize> {
        if idx < 9 {
            Some(idx)
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────
// § 단위 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negamax_outcome() {
        //  X 승리: X(0) O(3) X(1) O(4) X(2)
        let mut s = TicTacToe::initial();
        for mv in [0, 3, 1, 4, 2] {
            s = s.apply_move(mv);
        }
        // X가 이겼음, current_player = O
        assert!(s.is_terminal());
        assert_eq!(s.current_player(), -1); // O 차례
        assert_eq!(s.outcome(), -1.0); // O 관점에서 패배
    }

    #[test]
    fn test_draw() {
        // 무승부: X O X / X X O / O X O  (시퀀스 0→1→2→6→4→8→3→5→7)
        let mut s = TicTacToe::initial();
        for mv in [0, 1, 2, 6, 4, 8, 3, 5, 7] {
            s = s.apply_move(mv);
        }
        // 보드가 가득 찼고 승자 없음
        assert!(s.is_terminal());
        assert_eq!(s.outcome(), 0.0);
    }

    #[test]
    fn test_hash_transposition() {
        // 두 가지 순서로 같은 보드 → 해시 동일
        let mut a = TicTacToe::initial();
        for mv in [0, 4, 2, 6] {
            a = a.apply_move(mv);
        }

        let mut b = TicTacToe::initial();
        for mv in [2, 6, 0, 4] {
            b = b.apply_move(mv);
        }

        assert_eq!(a.board, b.board);
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn test_apply_move_pure() {
        let s = TicTacToe::initial();
        let s2 = s.apply_move(4);
        assert_eq!(s.board, [0i8; 9]); // 원본 불변
        assert_eq!(s2.board[4], 1);
    }
}
