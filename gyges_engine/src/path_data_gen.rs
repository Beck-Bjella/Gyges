// ── PathData structs ──────────────────────────────────────────────────────────
// One struct per piece type, sized to their own max paths.
// All fields co-located so one cache line fetch gets everything needed.

use std::fs::File;
use std::io::{BufWriter, Write};

use gyges::moves::one_dir_tables::*;
use gyges::moves::two_dir_tables::*;
use gyges::moves::three_dir_tables::*;

#[derive(Copy, Clone)]
pub struct OnePathData {
    pub end_bits:  u64,
    pub goal_bits: u64,
    pub count:     u8,
    pub ends:      [u8;  ONE_MAX_PATHS],
    pub backs:     [u64; ONE_MAX_PATHS],
}

#[derive(Copy, Clone)]
pub struct TwoPathData {
    pub end_bits:  u64,
    pub goal_bits: u64,
    pub count:     u8,
    pub ends:      [u8;  TWO_MAX_PATHS],
    pub backs:     [u64; TWO_MAX_PATHS],
}

#[derive(Copy, Clone)]
pub struct ThreePathData {
    pub end_bits:  u64,
    pub goal_bits: u64,
    pub count:     u8,
    pub ends:      [u8;  THREE_MAX_PATHS],
    pub backs:     [u64; THREE_MAX_PATHS],
}

// ── Generator functions ───────────────────────────────────────────────────────
// These compress the separate old arrays into the combined PathData format.
// Simple to understand — just loop and pack. Add new fields here if needed.

pub fn generate_one_path_data() -> Vec<OnePathData> {
    (0..ONE_PATH_COUNT.len()).map(|idx| {
        let count = ONE_PATH_COUNT[idx] as usize;
        let mut ends  = [0u8;  ONE_MAX_PATHS];
        let mut backs = [0u64; ONE_MAX_PATHS];
        for p in 0..count {
            ends[p]  = ONE_PATH_ENDS[idx][p];
            backs[p] = ONE_PATH_BACKS[idx][p];
        }
        OnePathData {
            end_bits:  ONE_PATH_END_BITS[idx],
            goal_bits: ONE_PATH_GOAL_BITS[idx],
            count:     count as u8,
            ends,
            backs,
        }
    }).collect()
}

pub fn generate_two_path_data() -> Vec<TwoPathData> {
    (0..TWO_PATH_COUNT.len()).map(|idx| {
        let count = TWO_PATH_COUNT[idx] as usize;
        let mut ends  = [0u8;  TWO_MAX_PATHS];
        let mut backs = [0u64; TWO_MAX_PATHS];
        for p in 0..count {
            ends[p]  = TWO_PATH_ENDS[idx][p];
            backs[p] = TWO_PATH_BACKS[idx][p];
        }
        TwoPathData {
            end_bits:  TWO_PATH_END_BITS[idx],
            goal_bits: TWO_PATH_GOAL_BITS[idx],
            count:     count as u8,
            ends,
            backs,
        }
    }).collect()
}

pub fn generate_three_path_data() -> Vec<ThreePathData> {
    (0..THREE_PATH_COUNT.len()).map(|idx| {
        let count = THREE_PATH_COUNT[idx] as usize;
        let mut ends  = [0u8;  THREE_MAX_PATHS];
        let mut backs = [0u64; THREE_MAX_PATHS];
        for p in 0..count {
            ends[p]  = THREE_PATH_ENDS[idx][p];
            backs[p] = THREE_PATH_BACKS[idx][p];
        }
        ThreePathData {
            end_bits:  THREE_PATH_END_BITS[idx],
            goal_bits: THREE_PATH_GOAL_BITS[idx],
            count:     count as u8,
            ends,
            backs,
        }
    }).collect()
}

// ── Writer ────────────────────────────────────────────────────────────────────
// Writes the combined PathData arrays to a file.

pub fn write_path_data(out_path: &str) {
    let file = File::create(out_path).expect("failed to create output file");
    let mut w = BufWriter::new(file);

    writeln!(w, "// Auto-generated combined PathData tables").unwrap();
    writeln!(w, "// DO NOT EDIT — regenerate with write_path_data\n").unwrap();

    // ── ONE ───────────────────────────────────────────────────────────────────
    let one_data = generate_one_path_data();
    let n = one_data.len();
    writeln!(w, "pub static ONE_PATH_DATA: [OnePathData; {n}] = [").unwrap();
    for d in &one_data {
        let ends:  Vec<String> = d.ends.iter().map(|v| v.to_string()).collect();
        let backs: Vec<String> = d.backs.iter().map(|v| format!("0x{v:016X}")).collect();
        writeln!(w, "    OnePathData {{ end_bits: 0x{:016X}, goal_bits: 0x{:016X}, count: {}, ends: [{}], backs: [{}] }},",
            d.end_bits, d.goal_bits, d.count,
            ends.join(","), backs.join(",")
        ).unwrap();
    }
    writeln!(w, "];\n").unwrap();

    // ── TWO ───────────────────────────────────────────────────────────────────
    let two_data = generate_two_path_data();
    let n = two_data.len();
    writeln!(w, "pub static TWO_PATH_DATA: [TwoPathData; {n}] = [").unwrap();
    for d in &two_data {
        let ends:  Vec<String> = d.ends.iter().map(|v| v.to_string()).collect();
        let backs: Vec<String> = d.backs.iter().map(|v| format!("0x{v:016X}")).collect();
        writeln!(w, "    TwoPathData {{ end_bits: 0x{:016X}, goal_bits: 0x{:016X}, count: {}, ends: [{}], backs: [{}] }},",
            d.end_bits, d.goal_bits, d.count,
            ends.join(","), backs.join(",")
        ).unwrap();
    }
    writeln!(w, "];\n").unwrap();

    // ── THREE ─────────────────────────────────────────────────────────────────
    let three_data = generate_three_path_data();
    let n = three_data.len();
    writeln!(w, "pub static THREE_PATH_DATA: [ThreePathData; {n}] = [").unwrap();
    for d in &three_data {
        let ends:  Vec<String> = d.ends.iter().map(|v| v.to_string()).collect();
        let backs: Vec<String> = d.backs.iter().map(|v| format!("0x{v:016X}")).collect();
        writeln!(w, "    ThreePathData {{ end_bits: 0x{:016X}, goal_bits: 0x{:016X}, count: {}, ends: [{}], backs: [{}] }},",
            d.end_bits, d.goal_bits, d.count,
            ends.join(","), backs.join(",")
        ).unwrap();
    }
    writeln!(w, "];").unwrap();

    println!("Written to {out_path}");
}
