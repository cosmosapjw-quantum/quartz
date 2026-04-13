//! Gomoku (오목) — GameState 구현체
//!
//! 규칙: 표준 5목 (5개 이상 일렬이면 승리)
//!   이후 Renju(렌주) 확장 시 overline/double-3/double-4 금수 추가 예정
//!
//! 지원 보드 크기: 9×9 / 13×13 / 15×15 (런타임 파라미터)
//!
//! Zobrist 해시:
//!   piece[player_idx][square], side — 최대 19×19=361 칸 대응 전역 테이블
//!
//! encode_planes (NN 입력):
//!   채널 0: 현재 플레이어 기물 위치 (1.0)
//!   채널 1: 상대 플레이어 기물 위치 (1.0)
//!   채널 2: 현재 플레이어 색상 (black=1.0, white=0.0)

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::fmt;

use crate::game::GameState;

// ─────────────────────────────────────────────
// § Zobrist 테이블 (최대 19×19)
// ─────────────────────────────────────────────

const MAX_SQ: usize = 19 * 19; // 361

struct GomokuZob {
    piece: [[u64; MAX_SQ]; 2], // [player_idx 0=black(+1) 1=white(-1)][square]
    side: u64,
}

impl GomokuZob {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut piece = [[0u64; MAX_SQ]; 2];
        for p in 0..2 {
            for sq in 0..MAX_SQ {
                piece[p][sq] = rng.gen();
            }
        }
        GomokuZob {
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
    static GOB: GomokuZob = GomokuZob::new(0xFEED_FACE_CAFE_BABE);
}

// ─────────────────────────────────────────────
// § Gomoku 상태
// ─────────────────────────────────────────────

/// History depth for AlphaZero-style encoding (T=8 timesteps including current).
const GOMOKU_HISTORY_LEN: usize = 8;

#[derive(Clone, Debug)]
pub struct Gomoku {
    pub size: usize,    // 보드 크기 (7/9/13/15)
    pub win_len: usize, // 승리 조건 (연속 몇 개: 4 or 5)
    board: Vec<i8>,     // +1=black, -1=white, 0=empty  (row-major)
    current_player: i8, // +1=black, -1=white
    hash: u64,          // Zobrist 증분 해시
    move_count: u32,
    winner: i8, // 0=없음, +1=black 승, -1=white 승
    last_move: Option<usize>,
    /// Past board snapshots for AlphaZero-style history encoding (most recent last).
    board_history: Vec<Vec<i8>>,
}

impl Gomoku {
    pub fn new(size: usize) -> Self {
        Self::new_with_win(size, 5)
    }

    /// 보드 크기와 승리 조건을 지정하여 생성
    /// e.g. Gomoku::new_with_win(7, 4) — 7×7 board, 4-in-a-row
    pub fn new_with_win(size: usize, win_len: usize) -> Self {
        assert!(size <= 19, "board size > 19 not supported");
        assert!(
            win_len >= 3 && win_len <= size,
            "win_len must be in [3, size]"
        );
        Gomoku {
            size,
            win_len,
            board: vec![0; size * size],
            current_player: 1, // black 선수
            hash: 0,
            move_count: 0,
            winner: 0,
            last_move: None,
            board_history: Vec::new(),
        }
    }

    /// JSON board (0/1/2 encoding) + player (1 or 2)로부터 상태 복원
    /// mcts_server용: 외부 프로토콜과 내부 엔진 사이의 변환
    pub fn from_board_12(size: usize, win_len: usize, board_12: &[i64], player_12: u8) -> Self {
        let mut g = Self::new_with_win(size, win_len);
        let mut mc = 0u32;
        for (i, &v) in board_12.iter().enumerate() {
            match v {
                1 => {
                    g.board[i] = 1;
                    mc += 1;
                    GOB.with(|z| g.hash ^= z.piece_hash(1, i));
                }
                2 => {
                    g.board[i] = -1;
                    mc += 1;
                    GOB.with(|z| g.hash ^= z.piece_hash(-1, i));
                }
                _ => {}
            }
        }
        g.move_count = mc;
        g.current_player = if player_12 == 1 { 1 } else { -1 };
        // side hash: black's turn = base, white's turn = xor side
        if g.current_player == -1 {
            GOB.with(|z| g.hash ^= z.side);
        }
        // Check if already terminal (winner from last move)
        for i in 0..g.board.len() {
            let v = g.board[i];
            if v != 0 && g.check_win_at(i, v) {
                g.winner = v;
                break;
            }
        }
        g
    }

    /// 내부 board를 0/1/2 encoding으로 변환 (JSON 응답용)
    pub fn board_as_12(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.board.len());
        for &v in &self.board {
            out.push(match v {
                1 => 1,
                -1 => 2,
                _ => 0,
            });
        }
        out
    }

    /// 현재 보드 상태에서 pos에 player를 놓으면 승리하는지 체크 (착수 전 검사)
    pub fn wins_if(&self, pos: usize, player: i8) -> bool {
        if self.board[pos] != 0 {
            return false;
        }
        // [OPT] No clone — check_win_at only reads neighbors, and we count
        // the center cell as 1 implicitly. Neighbors already on the board.
        self.check_win_at_hypothetical(pos, player)
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.size + col
    }

    /// Check if placing `player` at `pos` would form win_len in a row.
    /// Does NOT require the piece to actually be on the board.
    fn check_win_at_hypothetical(&self, pos: usize, player: i8) -> bool {
        let row = (pos / self.size) as i32;
        let col = (pos % self.size) as i32;
        let sz = self.size as i32;
        let target = self.win_len as i32;

        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

        for &(dr, dc) in &DIRS {
            let mut cnt = 1; // count the hypothetical piece at pos

            let (mut r, mut c) = (row + dr, col + dc);
            while r >= 0
                && r < sz
                && c >= 0
                && c < sz
                && self.board[(r as usize) * self.size + c as usize] == player
            {
                cnt += 1;
                r += dr;
                c += dc;
            }

            let (mut r, mut c) = (row - dr, col - dc);
            while r >= 0
                && r < sz
                && c >= 0
                && c < sz
                && self.board[(r as usize) * self.size + c as usize] == player
            {
                cnt += 1;
                r -= dr;
                c -= dc;
            }

            if cnt >= target {
                return true;
            }
        }
        false
    }

    /// pos에 player 기물을 놓았을 때 win_len목 달성 여부
    fn check_win_at(&self, pos: usize, player: i8) -> bool {
        let row = (pos / self.size) as i32;
        let col = (pos % self.size) as i32;
        let sz = self.size as i32;
        let target = self.win_len as i32;

        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

        for &(dr, dc) in &DIRS {
            let mut cnt = 1;

            let (mut r, mut c) = (row + dr, col + dc);
            while r >= 0
                && r < sz
                && c >= 0
                && c < sz
                && self.board[(r as usize) * self.size + c as usize] == player
            {
                cnt += 1;
                r += dr;
                c += dc;
            }

            let (mut r, mut c) = (row - dr, col - dc);
            while r >= 0
                && r < sz
                && c >= 0
                && c < sz
                && self.board[(r as usize) * self.size + c as usize] == player
            {
                cnt += 1;
                r -= dr;
                c -= dc;
            }

            if cnt >= target {
                return true;
            }
        }
        false
    }
}

impl fmt::Display for Gomoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 좌표 헤더
        write!(f, "   ")?;
        for c in 0..self.size {
            write!(f, "{:2}", c)?;
        }
        writeln!(f)?;
        for r in 0..self.size {
            write!(f, "{:2} ", r)?;
            for c in 0..self.size {
                let sym = match self.board[self.idx(r, c)] {
                    1 => " X",
                    -1 => " O",
                    _ => " .",
                };
                write!(f, "{}", sym)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────
// § GameState 구현
// ─────────────────────────────────────────────

impl GameState for Gomoku {
    type Move = usize; // flat index (0..size*size)

    fn initial() -> Self {
        Gomoku::new(9)
    } // 기본 9×9

    fn current_player(&self) -> i8 {
        self.current_player
    }

    fn legal_moves(&self) -> Vec<usize> {
        if self.is_terminal() {
            return vec![];
        }
        let mut moves = Vec::with_capacity(self.size * self.size - self.move_count as usize);
        for (i, &v) in self.board.iter().enumerate() {
            if v == 0 {
                moves.push(i);
            }
        }
        moves
    }

    fn apply_move(&self, mv: usize) -> Self {
        let mut next = self.clone();
        let player = self.current_player;

        // Save current board to history before applying move
        next.board_history.push(self.board.clone());
        if next.board_history.len() > GOMOKU_HISTORY_LEN - 1 {
            next.board_history.drain(0..next.board_history.len() - (GOMOKU_HISTORY_LEN - 1));
        }

        // Zobrist 갱신
        GOB.with(|z| {
            next.hash ^= z.piece_hash(player, mv);
            next.hash ^= z.side;
        });

        next.board[mv] = player;
        next.move_count += 1;
        next.last_move = Some(mv);

        // 승리 판정 (착수 위치에서만 검사 → O(1) 아닌 O(n) 이지만 충분히 빠름)
        if next.check_win_at(mv, player) {
            next.winner = player;
        }

        next.current_player = -player;
        next
    }

    fn is_terminal(&self) -> bool {
        self.winner != 0 || self.move_count as usize >= self.size * self.size
    }

    /// negamax convention: current_player 관점
    /// 방금 상대가 이겼으면 current_player가 진 것 → -1.0
    fn outcome(&self) -> f32 {
        if self.winner == self.current_player {
            1.0
        }
        // 이 상태는 정상적으로 나타나지 않음
        else if self.winner != 0 {
            -1.0
        }
        // 상대(직전 착수자)가 이겼음
        else {
            0.0
        } // 무승부
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn num_actions(&self) -> usize {
        self.size * self.size
    }

    fn move_to_idx(&self, mv: usize) -> usize {
        mv
    }

    fn idx_to_move(&self, idx: usize) -> Option<usize> {
        if idx < self.size * self.size {
            Some(idx)
        } else {
            None
        }
    }

    /// NN 입력 feature planes (AlphaZero-style: [17, H, W])
    ///   planes 0-1:   현재(t=0) 내 돌, 상대 돌
    ///   planes 2-3:   t=1 (1수 전) 내 돌, 상대 돌
    ///   ...
    ///   planes 14-15: t=7 (7수 전) 내 돌, 상대 돌
    ///   plane 16:     현재 플레이어 색상 (black=1, white=0)
    fn encode_planes(&self) -> Vec<f32> {
        let n = self.size * self.size;
        let total_planes = GOMOKU_HISTORY_LEN * 2 + 1; // 17
        let mut out = vec![0.0f32; total_planes * n];
        let cp = self.current_player;

        // t=0: current board
        for (i, &v) in self.board.iter().enumerate() {
            if v == cp { out[i] = 1.0; }
            else if v != 0 { out[n + i] = 1.0; }
        }

        // t=1..7: history (most recent = last element in board_history)
        for (k, hist_board) in self.board_history.iter().rev().enumerate() {
            let t = k + 1;
            if t >= GOMOKU_HISTORY_LEN { break; }
            let base = t * 2 * n;
            for (i, &v) in hist_board.iter().enumerate() {
                if v == cp { out[base + i] = 1.0; }
                else if v != 0 { out[base + n + i] = 1.0; }
            }
        }

        // Color plane
        let color_val = if cp == 1 { 1.0 } else { 0.0 };
        let color_base = (total_planes - 1) * n;
        for i in 0..n { out[color_base + i] = color_val; }
        out
    }

    fn board_state_record(&self) -> Vec<i64> {
        self.board.iter().map(|&v| match v { 1 => 1, -1 => 2, _ => 0 }).collect()
    }

    /// [OPT] O(win_len) win check — no state clone, no hash update.
    fn is_winning_move(&self, mv: usize) -> bool {
        self.wins_if(mv, self.current_player)
    }

    /// [OPT] Random legal move without Vec allocation.
    /// Scans board for the (rand_idx % n_empty)-th empty cell.
    fn random_legal_move(&self, rand_idx: usize) -> Option<usize> {
        if self.is_terminal() {
            return None;
        }
        let n_empty = self.size * self.size - self.move_count as usize;
        if n_empty == 0 {
            return None;
        }
        let target = rand_idx % n_empty;
        let mut count = 0usize;
        for (i, &v) in self.board.iter().enumerate() {
            if v == 0 {
                if count == target {
                    return Some(i);
                }
                count += 1;
            }
        }
        None // unreachable if n_empty > 0
    }

    /// [OPT] Count empty cells without Vec.
    fn legal_move_count(&self) -> usize {
        if self.is_terminal() {
            return 0;
        }
        self.size * self.size - self.move_count as usize
    }
}

// ─────────────────────────────────────────────
// § 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(size: usize, moves: &[(usize, usize)]) -> Gomoku {
        // moves: (row, col) 쌍, 교대로 착수
        let mut s = Gomoku::new(size);
        for &(r, c) in moves {
            s = s.apply_move(r * size + c);
        }
        s
    }

    #[test]
    fn test_initial_state() {
        let s = Gomoku::new(9);
        assert_eq!(s.legal_moves().len(), 81);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_horizontal_win() {
        // Black: (0,0)(0,1)(0,2)(0,3)(0,4) — 5 in a row
        // White: (1,0)(1,1)(1,2)(1,3)
        let s = make_state(
            9,
            &[
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
            ],
        );
        assert!(s.is_terminal());
        assert_eq!(s.current_player(), -1); // white's turn (black just won)
        assert_eq!(s.outcome(), -1.0); // white perspective: lost
    }

    #[test]
    fn test_diagonal_win() {
        // Black: (0,0)(1,1)(2,2)(3,3)(4,4)
        let s = make_state(
            9,
            &[
                (0, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (3, 3),
                (0, 4),
                (4, 4),
            ],
        );
        assert!(s.is_terminal());
        assert_eq!(s.outcome(), -1.0);
    }

    #[test]
    fn test_no_win_four_in_row() {
        // Black: 4 in a row (not 5) — should not be terminal
        let s = make_state(9, &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_hash_transposition() {
        let size = 9;
        // Same board via different move orders
        let mut a = Gomoku::new(size);
        for mv in [0 * size + 0, 1 * size + 0, 0 * size + 1, 1 * size + 1] {
            a = a.apply_move(mv);
        }

        let mut b = Gomoku::new(size);
        for mv in [0 * size + 1, 1 * size + 1, 0 * size + 0, 1 * size + 0] {
            b = b.apply_move(mv);
        }

        assert_eq!(a.board, b.board);
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn test_apply_move_pure() {
        let s = Gomoku::new(9);
        let before_board = s.board.clone();
        let s2 = s.apply_move(40); // center of 9×9
        assert_eq!(s.board, before_board); // 원본 불변
        assert_eq!(s2.board[40], 1);
    }

    #[test]
    fn test_encode_planes() {
        let s = make_state(9, &[(4, 4), (0, 0)]);
        let planes = s.encode_planes();
        let n = 9 * 9;
        assert_eq!(planes.len(), 17 * n);
        // current_player = black (+1) after 2 moves
        assert_eq!(planes[4 * 9 + 4], 1.0); // plane 0: black's piece at (4,4)
        assert_eq!(planes[n + 0], 1.0); // plane 1: white's piece at (0,0)
    }

    // ── win_len=4 tests (7×7 server compatibility) ──

    fn make_state_7x4(moves: &[(usize, usize)]) -> Gomoku {
        let mut s = Gomoku::new_with_win(7, 4);
        for &(r, c) in moves {
            s = s.apply_move(r * 7 + c);
        }
        s
    }

    #[test]
    fn test_7x4_four_in_row_wins() {
        // Black: (0,0)(0,1)(0,2)(0,3) — 4 in a row on 7×7 w/ win_len=4
        // White: (1,0)(1,1)(1,2)
        let s = make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]);
        assert!(s.is_terminal(), "4-in-row should win with win_len=4");
    }

    #[test]
    fn test_7x4_three_no_win() {
        // Black: 3 in a row — not enough for win_len=4
        let s = make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]);
        assert!(!s.is_terminal(), "3-in-row should NOT win with win_len=4");
    }

    #[test]
    fn test_9x5_four_no_win() {
        // 9×9 standard: 4 in a row should NOT win (need 5)
        let s = make_state(9, &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]);
        assert!(!s.is_terminal(), "4-in-row should NOT win on standard 9×9");
    }

    #[test]
    fn test_from_board_12() {
        // 7×7 board with 1=black, 2=white, 0=empty
        let mut board = vec![0i64; 49];
        board[0] = 1; // black at (0,0)
        board[1] = 1; // black at (0,1)
        board[7] = 2; // white at (1,0)
        board[8] = 2; // white at (1,1)
        let s = Gomoku::from_board_12(7, 4, &board, 1);
        assert_eq!(s.current_player(), 1); // black's turn
        assert_eq!(s.board[0], 1); // internal: +1
        assert_eq!(s.board[7], -1); // internal: -1
        assert!(!s.is_terminal());
        assert_eq!(s.move_count, 4);
    }

    #[test]
    fn test_wins_if() {
        // Black has 3 in a row at (0,0)(0,1)(0,2) on 7×7 w/ win_len=4
        let s = make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
        // Black placing at (0,3) should win
        assert!(
            s.wins_if(0 * 7 + 3, 1),
            "black should win by placing at (0,3)"
        );
        // White placing at (0,3) should NOT win (only 1 white piece at row 1)
        assert!(!s.wins_if(0 * 7 + 3, -1));
    }

    // ── PR-4A: Gomoku 15×15 tests ──────────────────────

    fn make_state_15(moves: &[(usize, usize)]) -> Gomoku {
        let mut s = Gomoku::new(15); // 15×15, win_len=5 (standard)
        for &(r, c) in moves {
            s = s.apply_move(r * 15 + c);
        }
        s
    }

    // TEST-4A-1: legal_moves on 15×15
    #[test]
    fn test_4a1_gomoku15_legal_moves() {
        let s = Gomoku::new(15);
        assert_eq!(s.legal_moves().len(), 225, "15×15 = 225 cells");
        assert!(!s.is_terminal());
        assert_eq!(s.size, 15);
        assert_eq!(s.win_len, 5);
    }

    #[test]
    fn test_4a1_gomoku15_after_moves() {
        let s = make_state_15(&[(7, 7), (0, 0), (7, 8)]);
        assert_eq!(s.legal_moves().len(), 222); // 225 - 3
        assert_eq!(s.current_player(), -1); // white's turn (3 moves = black just played)
    }

    // TEST-4A-2: win detection all directions on 15×15
    #[test]
    fn test_4a2_gomoku15_horizontal_win() {
        // Black: (7,3)(7,4)(7,5)(7,6)(7,7) = 5 in a row horizontal
        let s = make_state_15(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6),
            (0, 3),
            (7, 7),
        ]);
        assert!(s.is_terminal(), "5-in-row horizontal should win on 15×15");
    }

    #[test]
    fn test_4a2_gomoku15_vertical_win() {
        // Black: (3,7)(4,7)(5,7)(6,7)(7,7) = 5 in a row vertical
        let s = make_state_15(&[
            (3, 7),
            (0, 0),
            (4, 7),
            (0, 1),
            (5, 7),
            (0, 2),
            (6, 7),
            (0, 3),
            (7, 7),
        ]);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_4a2_gomoku15_diagonal_win() {
        // Black: (3,3)(4,4)(5,5)(6,6)(7,7) = diagonal
        let s = make_state_15(&[
            (3, 3),
            (0, 0),
            (4, 4),
            (0, 1),
            (5, 5),
            (0, 2),
            (6, 6),
            (0, 3),
            (7, 7),
        ]);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_4a2_gomoku15_four_no_win() {
        // 4 in a row should NOT win (need 5) on 15×15
        let s = make_state_15(&[(7, 3), (0, 0), (7, 4), (0, 1), (7, 5), (0, 2), (7, 6)]);
        assert!(!s.is_terminal(), "4-in-row should NOT win on 15×15");
    }

    // TEST-4A-3: Rust MctsEngine integration with 15×15
    #[test]
    fn test_4a3_gomoku15_engine_integration() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Gomoku::new(15);
        let eval: Arc<dyn crate::game::Evaluator<Gomoku> + Send + Sync> = Arc::new(UniformEval);
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(state, eval, config);
        engine.run(&mut FixedIterations::new(100));

        let best = engine.best_move();
        assert!(best.is_some(), "should find a move on 15×15");
        let mv = best.unwrap();
        assert!(mv < 225, "move should be valid index");
    }

    // TEST-4A: 15×15 hash/transposition
    #[test]
    fn test_4a_gomoku15_hash() {
        let mut a = Gomoku::new(15);
        for mv in [7 * 15 + 7, 0, 7 * 15 + 8, 1] {
            a = a.apply_move(mv);
        }

        let mut b = Gomoku::new(15);
        for mv in [7 * 15 + 8, 1, 7 * 15 + 7, 0] {
            b = b.apply_move(mv);
        }

        assert_eq!(
            a.hash(),
            b.hash(),
            "same board via different order should have same hash"
        );
    }
}
