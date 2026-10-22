//! A Gygès engine: a neural network evaluation inside a parallel alpha-beta search.
//!
//! - **Evaluation:** an MLP with 5,886 sparse binary inputs (piece-square features plus every pair
//!   of pieces), 1,024 hidden units (ReLU), and one output (tanh). Only 78 inputs are ever active,
//!   so an evaluation takes about 8 µs. The weights are embedded in the binary.
//! - **Search:** negamax alpha-beta with principal variation search, iterative deepening with
//!   aspiration windows, a lockless transposition table, and threat-filtered move ordering.
//! - **Parallelism:** the Young Brothers Wait Concept, with split points anywhere in the tree,
//!   reaching a 50.5x speedup on 96 cores.
//!
//! The engine is a standalone program that communicates over the UGI protocol; it is not intended
//! as a dependency. See the [engine README](https://github.com/Beck-Bjella/Gyges/tree/main/gyges_engine)
//! for details and benchmarks, and the [gyges](https://crates.io/crates/gyges) crate for the game
//! library.

#![allow(dead_code)]

pub mod search;
pub mod consts;
pub mod ugi;
