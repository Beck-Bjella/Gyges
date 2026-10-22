//! Other tools used for the game.
//! 
//! The most important tool in this module is the [TranspositionTable](tt), a fast hash table that can be accessed concurrently by multiple threads.
//! 

pub mod tt;
pub mod zobrist;
