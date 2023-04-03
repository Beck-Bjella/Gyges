use crate::tt::*;

use std::ptr;
use std::mem;


const TT_ALLOC_SIZE: usize = mem::size_of::<TranspositionTable>();

// Gloabal TT
pub static mut TT_TABLE: DummyTranspositionTable = [0; TT_ALLOC_SIZE];


pub type DummyTranspositionTable = [u8; TT_ALLOC_SIZE];

pub fn tt() -> &'static TranspositionTable {
    return unsafe { &*(&mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable) };

}

pub fn init_tt() {
    unsafe {
        let tt = &mut TT_TABLE as *mut DummyTranspositionTable as *mut TranspositionTable;
        ptr::write(tt, TranspositionTable::new(2usize.pow(25)));

    }

}
