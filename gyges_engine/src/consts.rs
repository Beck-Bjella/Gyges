use gyges::tools::tt::*;

use std::mem;
use std::ptr;

// TT (Transposition Table)
const TT_ALLOC_SIZE: usize = mem::size_of::<TranspositionTable>();
pub type DummyTranspositionTable = [u8; TT_ALLOC_SIZE];
pub static mut TT_TABLE: DummyTranspositionTable = [0; TT_ALLOC_SIZE];

/// Returns acess to the global transposition table.
pub fn tt() -> &'static TranspositionTable {
    unsafe { &*(&mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable) }

}

/// Initalizes the global transposition table.
/// Size must be a power of 2
pub fn init_tt(size: usize) {
    unsafe {
        let tt = &mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable;
        ptr::write(tt, TranspositionTable::new(size));

    }

}

// Boards
pub const STARTING_BOARD: [usize; 38] = [
    3, 2, 1, 1, 2, 3,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    3, 2, 1, 1, 2, 3,
    0, 0

];

pub const BENCH_BOARD: [usize; 38] = [
    2, 0, 2, 1, 1, 0,
    0, 0, 0, 3, 3, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 3, 0, 1, 2, 3,
    0, 0

];