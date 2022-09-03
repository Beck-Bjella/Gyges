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