use crate::move_gen::*;

pub const TRANSPOSITION_TABLE_SIZE: usize = 2_usize.pow(24) + 2;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum TTEntryType {
    ExactValue,
    UpperBound,
    LowerBound,
    None

}


#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    pub key: u64,
    pub value: f64,
    pub bestmove: Move,
    pub flag: TTEntryType,
    pub depth: i8,
    pub empty: bool

}

impl TTEntry {
    pub fn new() -> TTEntry {
        return TTEntry {
            key: 0,
            value: 0.0,
            bestmove: Move::new_null(),
            flag: TTEntryType::None,
            depth: 0,
            empty: true

        };

    }

    pub fn is_empty(&self) -> bool {
        return self.empty

    }

}

pub static mut REPLACEMENTS: usize = 0;
pub static mut COLLISIONS: usize = 0;

pub struct TranspositionTable {
    table: Vec<TTEntry>

}

impl TranspositionTable {
    pub fn new() -> TranspositionTable{
        return TranspositionTable {
            table: vec![TTEntry::new(); TRANSPOSITION_TABLE_SIZE],

        }

    }

    pub fn insert(&mut self, key: u64, new_entry: TTEntry) {
        let index = key as usize % TRANSPOSITION_TABLE_SIZE;

        let entry = self.table[index];
        if entry.is_empty() {
            self.table[index] = new_entry;

        } else {
            // if new_entry.depth >= entry.depth {
                unsafe{REPLACEMENTS+=1};

                self.table[index] = new_entry;

            // }

        }

    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let index = key as usize % TRANSPOSITION_TABLE_SIZE;

        let entry = self.table[index];
        if entry.is_empty() {
            return None;

        } 

        if entry.key == key {
            return Some(entry);

        }

        if entry.key != key && entry.key != 0 {
            unsafe{COLLISIONS+=1}

        }
            
        return None;

    }

}