//! Other tools used for the game.
//! 
//! The most important tool in this module is the [TranspositionTable](tt), a super fast hash table that can be concurently accessed by multiple threads.
//! 

pub mod tt;
pub mod zobrist;
