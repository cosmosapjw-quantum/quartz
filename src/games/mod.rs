//! Game implementations
#![allow(unused_imports)]

pub mod chess;
pub mod go;
pub mod gomoku;
pub mod gomoku15;
pub mod tictactoe;

pub use chess::Chess;
pub use go::Go;
pub use gomoku::Gomoku;
pub use gomoku15::Gomoku15;
pub use tictactoe::TicTacToe;
