//! C++ FFI adapter stub
//! 실제 C++ 연결 전까지 Rust Gomoku로 위임하는 stub

use crate::game::GameState;
use crate::games::Gomoku;

#[derive(Clone, Debug)]
pub struct CppGameAdapter {
    inner: Gomoku,
}

impl CppGameAdapter {
    pub fn gomoku_9x9() -> Self {
        CppGameAdapter {
            inner: Gomoku::new(9),
        }
    }
}

impl GameState for CppGameAdapter {
    type Move = usize;

    fn initial() -> Self {
        Self::gomoku_9x9()
    }
    fn current_player(&self) -> i8 {
        self.inner.current_player()
    }
    fn legal_moves(&self) -> Vec<usize> {
        self.inner.legal_moves()
    }
    fn apply_move(&self, mv: usize) -> Self {
        CppGameAdapter {
            inner: self.inner.apply_move(mv),
        }
    }
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }
    fn outcome(&self) -> f32 {
        self.inner.outcome()
    }
    fn hash(&self) -> u64 {
        self.inner.hash()
    }
    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }
    fn move_to_idx(&self, mv: usize) -> usize {
        self.inner.move_to_idx(mv)
    }
    fn idx_to_move(&self, idx: usize) -> Option<usize> {
        self.inner.idx_to_move(idx)
    }
    fn encode_planes(&self) -> Vec<f32> {
        self.inner.encode_planes()
    }
}
