use crate::tt::*;

use std::mem;

const TT_ALLOC_SIZE: usize = mem::size_of::<TranspositionTable>();

// Gloabal TT
pub static mut TT_TABLE: DummyTranspositionTable = [0; TT_ALLOC_SIZE];

pub type DummyTranspositionTable = [u8; TT_ALLOC_SIZE];
