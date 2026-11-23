//! C26 RED scaffold: intentionally absent root-frame/metareasoning transformations.
//! This is not a defect claim against the imported negamax GameState API.
#![allow(dead_code)]
#[path = "../game.rs"]
mod game;
#[path = "../games/tictactoe.rs"]
mod tictactoe;
use game::GameState;
use tictactoe::TicTacToe;

fn root_value(state: &TicTacToe, _root: i8) -> i8 { state.outcome() as i8 }
fn positive_excess(_p: [f64;3], _b: f64) -> f64 { 0.0 }
fn reservation_index(_p: [f64;3], d: f64) -> f64 { -d }
fn softmax(q: &[f64], _tau:f64) -> Vec<f64> {
    let mut p=vec![0.0;q.len()]; p[0]=1.0; p
}
fn main() {}

#[cfg(test)]
mod c26_tests {
 use super::*;
 #[test]
 fn red_root_frame_retains_actual_x_win() {
  let mut s=TicTacToe::initial();
  for m in [0,3,1,4,2] { s=s.apply_move(m); }
  assert!(s.is_terminal());
  assert_eq!(s.outcome(),-1.0); // existing API is correct
  assert_eq!(root_value(&s,1),1,"missing root-player sign conversion");
 }
 #[test]
 fn red_positive_excess_three_point_literal() {
  assert!((positive_excess([0.2,0.4,0.4],0.5)-0.2).abs()<1e-14);
 }
 #[test]
 fn red_reservation_first_branch_literal() {
  assert!((reservation_index([0.2,0.4,0.4],0.1)-0.75).abs()<1e-14);
 }
 #[test]
 fn red_gibbs_nonpositive_focus_has_probability() {
  let p=softmax(&[0.0,-0.03125*3.0_f64.ln()],0.03125);
  assert!((p[1]-0.25).abs()<1e-14,"Gibbs must sample negative-Q FOCUS");
 }
}
