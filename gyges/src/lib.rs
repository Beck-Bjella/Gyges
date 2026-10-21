//! A library for the board game Gygès.
//!
//! Gygès is an abstract strategy game for two players. The object of Gygès is to move a piece to your opponent's last row.
//! The catch is that no one owns the pieces. You can only move a piece in the row nearest you. 
//! Pieces are made up of 1, 2, or 3 rings; this number is also the number of spaces they must move. 
//! If a piece lands on another piece, it can move the number of spaces equal to that piece's number of rings. 
//! It can also displace that piece to any open space. 
//! 
//! # Platforms
//! The library builds for any target Rust supports, including WebAssembly. On x86-64 CPUs with BMI2 (enabled with `-C target-cpu=x86-64-v3` or `-C target-feature=+bmi2`), move generation uses the hardware `pext` instruction. Every other target uses a portable fallback that gives identical results.
//! 
//! # Basic Library Usage
//! 
//! This crate is available on [crates.io](https://crates.io/crates/gyges) and can be used by adding `gyges` to your `Cargo.toml` file. 
//! For more examples check out the [github page](https://github.com/Beck-Bjella/Gyges) for the project.
//! 
//! ### Setting up a Starting Board Position
//! 
//! The library provides predefined starting board positions. Use the following code to load the default board:
//! 
//! ```rust
//! use gyges::*;
//! 
//! // Load the starting board position
//! let board: BoardState = BoardState::from(STARTING_BOARD);
//! ```
//! 
//! ### Loading a Specific Board Configuration
//! 
//! To load a custom board configuration, provide a string where each row is represented by a series of numbers. Rows are inputted from your side of the board to the opponent's, and pieces are numbered based on their ring count:
//! 
//! ```rust
//! use gyges::*;
//! 
//! // Load a custom board configuration
//! let board: BoardState = BoardState::from("321123/000000/000000/000000/000000/321123");
//! ```
//! 
//! ### Generating Moves
//! 
//! The `MoveGen` structure is the core for generating moves in Gygès. It provides a flexible interface for generating and calculating move-related data tailored to specific needs using generic parameters.
//! 
//! #### How It Works
//! 
//! `MoveGen` uses two key generic parameters:
//! 
//! - **`GenType`**: Specifies the type of data to generate:
//!   - `GenMoves`: Generates all possible legal moves.
//!   - `GenMoveCount`: Counts the total number of possible moves.
//!   - `GenThreatCount`: Counts the number of threats on the board.
//!   - `GenControlMoveCount`: Combines control analysis and move counting.
//! - **`QuitType`**: Controls when generation stops:
//!   - `NoQuit`: Completes the full generation process.
//!   - `QuitOnThreat`: Stops generation immediately if a threat is found. This is particularly useful for saving computation in scenarios where you need both the data and the guarantee that no threats exist. If a threat is found, you can handle it separately.
//! 
//! #### Examples
//! 
//! ##### 1. Generate All Moves
//! 
//! ```rust
//! use gyges::*;
//! 
//! // Setup
//! let mut board: BoardState = BoardState::from(STARTING_BOARD);
//! let player: Player = Player::One;
//! let mut move_gen: MoveGen = MoveGen::default();
//! 
//! // Generate
//! let data: GenResult = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, player) };
//! let mut movelist: RawMoveList = data.move_list;
//! 
//! let moves: Vec<Move> = movelist.moves(&mut board);
//! println!("Generated moves: {:?}", moves);
//! ```
//! 
//! ##### 2. Generate Move Count, Stopping if a Threat is Found
//! 
//! ```rust
//! use gyges::*;
//! 
//! // Setup
//! let mut board: BoardState = BoardState::from(STARTING_BOARD);
//! let player: Player = Player::One;
//! let mut move_gen: MoveGen = MoveGen::default();
//! 
//! // Generate
//! let data: GenResult = unsafe { move_gen.gen::<GenMoveCount, QuitOnThreat>(&mut board, player) };
//! let move_count: usize = data.move_count;
//! println!("Move count: {}", move_count);
//! ```
//! 
//! ### Making a Move
//! 
//! Use the `make_move` method on the `BoardState` struct to make a move. It applies the move to the board in place, and `unmake_move` reverses it:
//! 
//! ```rust
//! use gyges::*;
//!  
//! // Setup
//! let mut board: BoardState = BoardState::from(STARTING_BOARD);
//! let player: Player = Player::One;
//! 
//! let mut move_gen: MoveGen = MoveGen::default();
//! 
//! // Generate moves
//! let data: GenResult = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, player) };
//! let mut movelist: RawMoveList = data.move_list;
//! 
//! let moves: Vec<Move> = movelist.moves(&mut board);
//! 
//! // Make a move
//! println!("Original board: {}", board);
//! println!("Move: {:?}", moves[0]);
//! 
//! board.make_move(&moves[0]);
//! println!("New board: {}", board);
//!
//! // Undo it
//! board.unmake_move(&moves[0]);
//! println!("Restored board: {}", board);
//! ```
//! 
//! 
//! # Acknowledgements
//! This project and its formating was inspired by the incridible rust chess program [Pleco](https://github.com/pleco-rs/Pleco).
//! 

#![allow(dead_code)]

pub mod board;
pub mod core;
pub mod moves;
pub mod tools;

pub use board::{BoardState, STARTING_BOARD, BENCH_BOARD};
pub use board::bitboard::BitBoard;
pub use core::{Piece, Player, SQ};
pub use moves::{Move, movegen::*, move_list::RawMoveList};
