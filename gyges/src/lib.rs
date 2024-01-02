//! A library for the board game Gygès.
//!
//! Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row. The catch is that no one owns the pieces. 
//! You can only move a piece in the row nearest you. Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces it must move. 
//! If it lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
//! It can also displace that piece to any open space. 
//! 
//! Offical rule book: [Rules](https://s3.amazonaws.com/geekdo-files.com/bgg32746?response-content-disposition=inline%3B%20filename%3D%22gyges_rules.pdf%22&response-content-type=application%2Fpdf&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJYFNCT7FKCE4O6TA%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T031405Z&X-Amz-SignedHeaders=host&X-Amz-Expires=120&X-Amz-Signature=e7c322bed070e101346483b70e896133e22967568a021c530add36ef698b99d0)
//! 
//! # Usage
//!
//! # Acknowledgements
//! This project and its formating was inspired by the incridible rust chess program [Pleco](https://github.com/pleco-rs/Pleco).
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
