use crate::move_gen::*;

use primes::{Sieve, PrimeSet};

pub const TRANSPOSTION_TABLE_DEFAULT_SIZE_MB: usize = 8000;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum TTEntryType {
    ExactValue,
    UpperBound,
    LowerBound,
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    pub key: u64,
    pub value: f64,
    pub flag: TTEntryType,
    pub depth: i8,
    pub empty: bool,

}

impl TTEntry {
    pub fn new() -> TTEntry {
        return TTEntry {
            key: 0,
            value: 0.0,
            flag: TTEntryType::None,
            depth: 0,
            empty: true,
            
        };

    }

    pub fn is_empty(&self) -> bool {
        return self.empty;

    }

}

pub static mut TT_EMPTY_INSERTS: usize = 0;
pub static mut TT_SAFE_INSERTS: usize = 0;
pub static mut TT_UNSAFE_INSERTS: usize = 0;
pub static mut TT_LOOKUP_COLLISIONS: usize = 0;

pub struct TranspositionTable {
    table: Vec<TTEntry>,
    pub entrys: usize,

}

impl TranspositionTable {
    pub fn new_from_entrys(entrys: u64) -> TranspositionTable {
        let mut pset = Sieve::new();
        let prime_entrys = pset.find(entrys).1 as usize;

        return TranspositionTable {
            table: vec![TTEntry::new(); prime_entrys],
            entrys: prime_entrys,

        };

    }

    pub fn new_from_mb(mb: usize) -> TranspositionTable {
        let entrys = (mb as f64 / 0.000088) as u64;

        let mut pset = Sieve::new();
        let prime_entrys = pset.find(entrys).1 as usize;

        return TranspositionTable {
            table: vec![TTEntry::new(); prime_entrys],
            entrys: prime_entrys,

        };

    }

    pub fn mb_size(&self) -> f64 {
        return (88 * self.entrys) as f64 * 0.000001;

    }

    pub fn insert(&mut self, key: u64, new_entry: TTEntry) {
        let index = key as usize % self.entrys;

        let entry = self.table[index];
        if entry.is_empty() {
            self.table[index] = new_entry;
            unsafe { TT_EMPTY_INSERTS += 1 }

        } else {
            if new_entry.depth >= entry.depth {
                if entry.key == new_entry.key {
                    self.table[index] = new_entry;
                    unsafe { TT_SAFE_INSERTS += 1 }

                }    

                if entry.key != new_entry.key {
                    self.table[index] = new_entry;
                    unsafe { TT_UNSAFE_INSERTS += 1 }
    
                }    
                
            }

        }

    }

    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let index = key as usize % self.entrys;

        let entry = self.table[index];
        if entry.is_empty() {
            return None;

        }

        if entry.key == key {
            return Some(entry);

        }

        if entry.key != key && entry.key != 0 {
            unsafe { TT_LOOKUP_COLLISIONS += 1 }

        }

        return None;

    }
    
}
