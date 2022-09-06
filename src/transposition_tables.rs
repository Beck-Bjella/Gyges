use std::collections::HashMap;

pub struct EvaluationTable {
    table: HashMap<u64, f64>

}

impl EvaluationTable {
    pub fn new() -> EvaluationTable{
        return EvaluationTable {
            table: HashMap::new(),

        }

    }

    pub fn insert(&mut self, key: u64, item: f64) {
        self.table.insert(key, item);

    }

    pub fn get(&self, key: &u64) -> Option<&f64> {
        return self.table.get(key);

    }

}

#[derive(PartialEq)]
pub enum TTEntryType {
    ExactValue,
    UpperBound,
    LowerBound,
    None

}

pub struct TTEntry {
    pub value: f64,
    pub flag: TTEntryType,
    pub depth: i8

}
pub struct TranspositionTable {
    table: HashMap<u64, TTEntry>

}

impl TranspositionTable {
    pub fn new() -> TranspositionTable{
        return TranspositionTable {
            table: HashMap::new(),

        }

    }

    pub fn insert(&mut self, key: u64, item: TTEntry) {
        self.table.insert(key, item);

    }

    pub fn get(&self, key: &u64) -> Option<&TTEntry> {
        return self.table.get(key);

    }

}