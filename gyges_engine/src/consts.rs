//! General constants.
//! 

use gyges::tools::tt::*;
use gyges::moves::new_movegen::MoveGen;

use std::mem;
use std::ptr;
use std::ptr::addr_of_mut;
use std::cell::RefCell;

thread_local! {
    /// A thread local move generator. 
    pub static THREAD_LOCAL_MOVEGEN: RefCell<MoveGen> = RefCell::new(MoveGen::default());

}

// TT (Transposition Table)
const TT_ALLOC_SIZE: usize = mem::size_of::<TranspositionTable>();
pub type DummyTranspositionTable = [u8; TT_ALLOC_SIZE];
pub static mut TT_TABLE: DummyTranspositionTable = [0; TT_ALLOC_SIZE];

/// Returns acess to the global transposition table.
pub fn tt() -> &'static TranspositionTable {
    unsafe { &*(addr_of_mut!(TT_TABLE) as *mut DummyTranspositionTable as *mut TranspositionTable) }

}

/// Initalizes the global transposition table.
/// Size must be a power of 2
pub fn init_tt(size: usize) {
    unsafe {
        let tt = addr_of_mut!(TT_TABLE) as *mut DummyTranspositionTable as *mut TranspositionTable;
        ptr::write(tt, TranspositionTable::new(size));

    }

}
