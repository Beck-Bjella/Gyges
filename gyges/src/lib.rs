//! A library for the board game Gygès.
//!
//! Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. 
//! You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces it must move. 
//! If it lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
//! It can also displace that piece to any open space. 
//! 
//! # Usage
//! Check out the git repo for this project [here](https://github.com/Beck-Bjella/Gyges).
//! 
//! # Acknowledgements
//! This project and its formating was inspired by the incredible rust chess program [Pleco](https://github.com/pleco-rs/Pleco).
//! 

#![feature(test)]
#![allow(dead_code)]

pub mod board;
pub mod core;
pub mod moves;
pub mod tools;

pub use board::{BoardState, STARTING_BOARD, BENCH_BOARD};
pub use board::bitboard::BitBoard;
pub use core::{Piece, Player, SQ};
