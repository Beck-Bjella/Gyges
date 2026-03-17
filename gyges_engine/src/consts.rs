//! General constants.
//! 

use gyges::tools::tt::*;
use gyges::moves::movegen::MoveGen;

use std::mem::MaybeUninit;
use std::cell::RefCell;

thread_local! {
    /// A thread local move generator.
    pub static THREAD_LOCAL_MOVEGEN: RefCell<MoveGen> = RefCell::new(MoveGen::default());

}

// TT (Transposition Table)
static mut TT_TABLE: MaybeUninit<TranspositionTable> = MaybeUninit::uninit();

/// Returns acess to the global transposition table.
pub fn tt() -> &'static TranspositionTable {
    unsafe { TT_TABLE.assume_init_ref() }

}

/// Initalizes the global transposition table.
/// Size must be a power of 2
pub fn init_tt(size: usize) {
    unsafe {
        TT_TABLE.write(TranspositionTable::new(size));

    }

}
