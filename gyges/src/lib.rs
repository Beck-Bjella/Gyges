//! A library for the board game Gygès.
//!
//! Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row.
//! The catch is that no one owns the pieces. You can only move a piece in the row nearest you. 
//! Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces they must move. 
//! If a piece lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
//! It can also displace that piece to any open space. 
//! 
//! # Usage 
//! This crate is available on [crates.io](https://crates.io/crates/gyges) and can be used by adding `gyges` to your `Cargo.toml` file. 
//! For more examples check out the [github page](https://github.com/Beck-Bjella/Gyges) for the project.
//! 
//! # Platforms
//! This project has only been tested on a x86_64 architecture, and no guarantees are made for other platforms.
//! 
//! ## Examples
//! 
//! ### Setting up a starting board position
//! 
//! This is one specific starting position built into the library. Other constant board positions can be loaded as well.
//! ```rust 
//! use gyges::board::*;
//! 
//! // Load Board
//! let board = BoardState::from(STARTING_BOARD);
//! 
//! ```
//! 
//! ### Loading a specific Board
//! 
//! Boards can be created using this notation, as shown below. Each set of 6 numbers represents a row on the board, starting from your side of the board and going left to right. The orientation of the board is subjective and is based on how the board is inputted.
//! ```rust
//! use gyges::board::*;
//! 
//! // Load Board
//! let board = BoardState::from("321123/000000/000000/000000/000000/321123");
//! 
//! ```
//! 
//! ### Applying and generating moves
//! 
//! Move generation is done in a two-step process. You must generate a `RawMoveList` and then extract the moves from that list into a `Vec<Move>`. This is done to improve efficiency and reduce unnecessary processing. When making a move, the `make_move` function will return a copy of the board with the move applied.
//! ```rust
//! use gyges::board::*;
//! use gyges::moves::*;
//! 
//! // Load Board
//! let mut board = BoardState::from(STARTING_BOARD);
//! 
//! // Define a player
//! let player = Player::One;
//! 
//! // Generate moves
//! let mut movelist = unsafe{ valid_moves(&mut board, player) }; // Create a MoveList
//! let moves = movelist.moves(&mut board); // Extract the moves
//! 
//! // Make a move
//! board.make_move(&moves[0]);
//! 
//! ```
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
