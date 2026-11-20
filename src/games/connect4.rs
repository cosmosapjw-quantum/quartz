//! Standard gravity Connect Four, 7 columns by 6 rows, four in a row.
//! All move indices are zero-based *columns*; board index is (row_from_bottom * 7 + col).
//! No neural evaluation is assumed by this GameState adapter.
use crate::game::GameState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connect4 {
    board: [i8; 42],
    heights: [u8; 7],
    next: i8,
    winner: i8,
    ply: u8,
}

impl Connect4 {
    /// Parse the Pascal Pons solver's 1-based column sequence, rejecting
    /// invalid columns, overfull columns and any play after terminal.
    pub fn from_history(moves_one_based: &str) -> Result<Self, String> {
        let mut state = Self::initial();
        for (ply, b) in moves_one_based.bytes().enumerate() {
            if state.is_terminal() {
                return Err(format!("history continues after game ended at ply {ply}"));
            }
            if !(b'1'..=b'7').contains(&b) {
                return Err(format!("invalid Connect4 column at ply {}", ply + 1));
            }
            let col = (b - b'1') as usize;
            if state.heights[col] >= 6 {
                return Err(format!("column {} is full at ply {}", col + 1, ply + 1));
            }
            state.apply_move_in_place_no_undo(col);
        }
        Ok(state)
    }

    pub fn board_flat(&self) -> [i8; 42] {
        self.board
    }
    pub fn ply_count(&self) -> usize {
        self.ply as usize
    }
    pub fn heights(&self) -> [u8; 7] {
        self.heights
    }

    fn check_from_cell(&self, x: usize, y: usize, color: i8) -> bool {
        for (dx, dy) in [(1i32, 0i32), (0, 1), (1, 1), (1, -1)] {
            let mut count = 1;
            for sign in [-1, 1] {
                let mut step = 1i32;
                loop {
                    let nx = x as i32 + sign * step * dx;
                    let ny = y as i32 + sign * step * dy;
                    if !(0..7).contains(&nx) || !(0..6).contains(&ny) {
                        break;
                    }
                    if self.board[ny as usize * 7 + nx as usize] != color {
                        break;
                    }
                    count += 1;
                    step += 1;
                }
            }
            if count >= 4 {
                return true;
            }
        }
        false
    }

    #[inline]
    fn can_play(&self, col: usize) -> bool {
        col < 7 && self.heights[col] < 6 && !self.is_terminal()
    }
}

impl GameState for Connect4 {
    type Move = usize;
    // Only 42 cells, so full clone-and-undo is an explicit, bounded baseline.
    type Undo = Self;

    fn initial() -> Self {
        Self {
            board: [0; 42],
            heights: [0; 7],
            next: 1,
            winner: 0,
            ply: 0,
        }
    }
    fn current_player(&self) -> i8 {
        self.next
    }
    fn legal_moves(&self) -> Vec<Self::Move> {
        if self.is_terminal() {
            return Vec::new();
        }
        (0..7).filter(|&col| self.heights[col] < 6).collect()
    }
    fn apply_move(&self, col: Self::Move) -> Self {
        assert!(self.can_play(col), "illegal or post-terminal Connect4 move");
        let mut state = self.clone();
        state.apply_move_in_place_no_undo(col);
        state
    }
    fn apply_move_in_place(&mut self, col: Self::Move) -> Self {
        let prior = self.clone();
        self.apply_move_in_place_no_undo(col);
        prior
    }
    fn apply_move_in_place_no_undo(&mut self, col: Self::Move) {
        assert!(self.can_play(col), "illegal or post-terminal Connect4 move");
        let row = self.heights[col] as usize;
        let color = self.next;
        self.board[row * 7 + col] = color;
        self.heights[col] += 1;
        self.ply += 1;
        if self.check_from_cell(col, row, color) {
            self.winner = color;
        }
        self.next = -color;
    }
    fn undo_move(&mut self, undo: Self) {
        *self = undo;
    }
    fn is_terminal(&self) -> bool {
        self.winner != 0 || self.ply == 42
    }
    fn outcome(&self) -> f32 {
        if self.winner == 0 {
            0.0
        } else if self.winner == self.next {
            1.0
        } else {
            -1.0
        }
    }

    /// Deterministic board+side fingerprint for transposition table lookup.
    /// No use of randomized standard-library HashMap seeds.
    fn hash(&self) -> u64 {
        let mut value = 0xcbf29ce484222325u64;
        for &cell in &self.board {
            value ^= (cell as i16 + 2) as u64;
            value = value.wrapping_mul(0x100000001b3);
        }
        value ^= (self.next as i16 + 2) as u64;
        value = value.wrapping_mul(0x100000001b3);
        value
    }
    fn num_actions(&self) -> usize {
        7
    }
    fn move_to_idx(&self, mv: Self::Move) -> usize {
        mv
    }
    fn idx_to_move(&self, idx: usize) -> Option<Self::Move> {
        if self.can_play(idx) {
            Some(idx)
        } else {
            None
        }
    }
    fn is_winning_move(&self, col: Self::Move) -> bool {
        if !self.can_play(col) {
            return false;
        }
        let mut next = self.clone();
        next.apply_move_in_place_no_undo(col);
        next.winner == self.next
    }
    fn legal_move_count(&self) -> usize {
        if self.is_terminal() {
            0
        } else {
            self.heights.iter().filter(|&&height| height < 6).count()
        }
    }
    fn board_state_record(&self) -> Vec<i64> {
        self.board
            .iter()
            .map(|&cell| match cell {
                1 => 1,
                -1 => 2,
                _ => 0,
            })
            .collect()
    }
    /// Minimal [current-player pieces, opponent pieces, color] planes,
    /// each of shape [6,7]. Not interchangeable with 17-plane trained models.
    fn encode_planes_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.resize(3 * 42, 0.0);
        for (idx, &piece) in self.board.iter().enumerate() {
            if piece == self.next {
                out[idx] = 1.0;
            } else if piece == -self.next {
                out[42 + idx] = 1.0;
            }
        }
        if self.next > 0 {
            out[84..126].fill(1.0);
        }
    }
    fn eval_encoder_revision(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_and_column_actions_reconstruct_sequence() {
        let s = Connect4::from_history("1212").unwrap();
        assert_eq!(s.board[0], 1);
        assert_eq!(s.board[1], -1);
        assert_eq!(s.board[7], 1);
        assert_eq!(s.board[8], -1);
        assert_eq!(s.current_player(), 1);
        assert_eq!(s.legal_moves(), vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(s.ply_count(), 4);
    }

    #[test]
    fn horizontal_and_vertical_wins_have_negamax_sign() {
        let horizontal = Connect4::from_history("1122334").unwrap();
        assert!(horizontal.is_terminal());
        assert_eq!(horizontal.outcome(), -1.0);
        assert_eq!(horizontal.current_player(), -1);
        assert!(horizontal.legal_moves().is_empty());
        let vertical = Connect4::from_history("1212121").unwrap();
        assert!(vertical.is_terminal());
        assert_eq!(vertical.outcome(), -1.0);
    }

    #[test]
    fn diagonal_winning_trajectory_is_terminal() {
        // Independently constructed valid diagonal line, white wins at ply 26.
        let state = Connect4::from_history("42723243732565313116365445").unwrap();
        assert!(state.is_terminal());
        assert_eq!(state.outcome(), -1.0);
        assert_eq!(state.current_player(), 1);
    }

    #[test]
    fn invalid_column_and_post_terminal_play_rejected() {
        for seq in ["8", "0", "1111111", "11223345"] {
            assert!(Connect4::from_history(seq).is_err(), "accepted {seq}");
        }
    }

    #[test]
    fn apply_undo_restores_full_identity() {
        let mut state = Connect4::from_history("112233").unwrap();
        let before = state.clone();
        let undo = state.apply_move_in_place(3);
        assert!(state.is_terminal());
        state.undo_move(undo);
        assert_eq!(state, before);
        assert_eq!(state.hash(), before.hash());
        assert_eq!(state.idx_to_move(7), None);
    }

    #[test]
    fn full_column_is_not_a_legal_move() {
        let state = Connect4::from_history("111111").unwrap();
        assert!(!state.legal_moves().contains(&0));
        assert_eq!(state.idx_to_move(0), None);
        assert_eq!(state.legal_moves().len(), 6);
    }
}
