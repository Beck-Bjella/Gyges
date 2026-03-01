use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use gyges::moves::movegen_consts::OnePathList;

fn h_bit(row: usize, col: usize) -> u64 { 1u64 << (row * 11 + col) }
fn v_bit(row: usize, col: usize) -> u64 { 1u64 << (row * 11 + 5 + col) }

// For ONE: each direction has exactly 1 intercept bit
// No intermediate squares exist, so no piece conversion needed
// blocked = backtrack_board only
fn dir_mask_one(sq: usize, dir: u8) -> u64 {
    if sq >= 36 { return 0; }
    let row = sq / 6;
    let col = sq % 6;
    match dir {
        0 => if row < 5 { v_bit(row, col) } else { 0 }, // north — goal has no intercept
        1 => if row > 0 { v_bit(row - 1, col) } else { 0 }, // south — goal has no intercept
        2 => if col < 5 { h_bit(row, col) } else { 0 },     // east
        3 => if col > 0 { h_bit(row, col - 1) } else { 0 }, // west
        _ => unreachable!(),
    }
}

fn expand_combo(combo: usize, mask: u64) -> u64 {
    let mut blocked = 0u64;
    let mut bit_idx = 0usize;
    let mut m = mask;
    while m != 0 {
        let b = m.trailing_zeros() as usize;
        if (combo >> bit_idx) & 1 != 0 { blocked |= 1u64 << b; }
        m &= m - 1;
        bit_idx += 1;
    }
    blocked
}

fn get_or_insert(
    paths: &[(u8, u64)],
    registry: &mut HashMap<Vec<(u8, u64)>, u16>,
    list_data: &mut Vec<Vec<(u8, u64)>>,
) -> u16 {
    let mut key = paths.to_vec();
    key.sort();
    if let Some(&idx) = registry.get(&key) { return idx; }
    let idx = list_data.len() as u16;
    registry.insert(key, idx);
    list_data.push(paths.to_vec());
    idx
}

// ONE: squares = [start, end], back = single intercept
// Filter: squares[0] == sq, direction determined by squares[1] vs sq
fn get_valid_paths_one(
    sq: usize,
    init_dir: u8,
    blocked: u64,
    all_lists: &[OnePathList],
) -> Vec<(u8, u64)> {
    // expected end square for this direction
    let row = sq / 6;
    let col = sq % 6;
    let expected_end = match init_dir {
        0 => if row < 5 { Some(sq + 6) } else { Some(37) }, // north — could be goal
        1 => if row > 0 { Some(sq - 6) } else { Some(36) }, // south — could be goal
        2 => if col < 5 { Some(sq + 1) } else { return vec![] },
        3 => if col > 0 { Some(sq - 1) } else { return vec![] },
        _ => unreachable!(),
    };
    let expected_end = match expected_end {
        Some(v) => v,
        None => return vec![],
    };

    let mut paths = Vec::new();
    for list in all_lists {
        for p in 0..list.count as usize {
            let (squares, back) = list.paths[p];
            if squares[0] as usize != sq { continue; }
            if squares[1] as usize != expected_end { continue; }
            if back & blocked == 0 {
                paths.push((squares[1], back));
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

pub fn generate_one_tables(out_path: &str, all_lists: &[OnePathList]) {
    let dir_names = ["NORTH", "SOUTH", "EAST", "WEST"];

    let mut registry: HashMap<Vec<(u8, u64)>, u16> = HashMap::new();
    let mut list_data: Vec<Vec<(u8, u64)>> = Vec::new();

    let mut all_masks: [[u64; 36]; 4] = [[0; 36]; 4];
    let mut all_index_tables: Vec<Vec<Vec<u16>>> = Vec::new();

    for dir_idx in 0..4u8 {
        println!("Generating ONE {}...", dir_names[dir_idx as usize]);

        let masks: [u64; 36] = std::array::from_fn(|sq| dir_mask_one(sq, dir_idx));
        all_masks[dir_idx as usize] = masks;

        let mut index_table: Vec<Vec<u16>> = Vec::with_capacity(36);

        for sq in 0..36usize {
            let mask = masks[sq];
            let bits = mask.count_ones() as usize;
            let real_combos = if bits == 0 { 1 } else { 1usize << bits }; // 0 or 1 bit → 1 or 2 combos

            let mut sq_indices: Vec<u16> = Vec::with_capacity(real_combos);
            for combo in 0..real_combos {
                let blocked = expand_combo(combo, mask);
                let paths = get_valid_paths_one(sq, dir_idx, blocked, all_lists);
                let idx = get_or_insert(&paths, &mut registry, &mut list_data);
                sq_indices.push(idx);
            }

            // pad to 2 (max combos for ONE — 1 bit per direction)
            let pad = sq_indices[0];
            sq_indices.resize(2, pad);
            index_table.push(sq_indices);
        }

        all_index_tables.push(index_table);
    }

    println!("Total unique ONE path lists: {}", list_data.len());
    println!("Max paths in any ONE list: {}", list_data.iter().map(|p| p.len()).max().unwrap_or(0));

    let file = File::create(out_path).expect("failed to create output file");
    let mut w = BufWriter::new(file);

    writeln!(w, "// Auto-generated ONE directional path tables\n").unwrap();

    // masks
    for dir_idx in 0..4usize {
        let name = dir_names[dir_idx];
        let vals: Vec<String> = (0..36).map(|sq| format!("0x{:016X}", all_masks[dir_idx][sq])).collect();
        writeln!(w, "pub static ONE_{name}_MASK: [u64; 36] = [{}];", vals.join(", ")).unwrap();
    }
    writeln!(w).unwrap();

    // index tables — [[u16; 2]; 36] since max 1 bit per direction
    for dir_idx in 0..4usize {
        let name = dir_names[dir_idx];
        writeln!(w, "pub static ONE_{name}_TABLE: [[u16; 2]; 36] = [").unwrap();
        for sq in 0..36 {
            let row = &all_index_tables[dir_idx][sq];
            writeln!(w, "    [{},{}],", row[0], row[1]).unwrap();
        }
        writeln!(w, "];\n").unwrap();
    }

    let n = list_data.len();
    let max_paths = list_data.iter().map(|p| p.len()).max().unwrap_or(0);
    writeln!(w, "// {n} unique ONE path lists, max {max_paths} paths each\n").unwrap();

    let counts: Vec<String> = list_data.iter().map(|p| p.len().to_string()).collect();
    writeln!(w, "pub static ONE_PATH_COUNT: [u8; {n}] = [{}];", counts.join(",")).unwrap();

    let end_bits: Vec<String> = list_data.iter().map(|paths| {
        let bits: u64 = paths.iter().filter(|&&(sq, _)| sq < 36).fold(0, |a, &(sq, _)| a | (1u64 << sq));
        format!("0x{bits:016X}")
    }).collect();
    writeln!(w, "pub static ONE_PATH_END_BITS: [u64; {n}] = [{}];", end_bits.join(",")).unwrap();

    let goal_bits: Vec<String> = list_data.iter().map(|paths| {
        let bits: u64 = paths.iter().filter(|&&(sq, _)| sq >= 36).fold(0, |a, &(sq, _)| a | (1u64 << sq));
        format!("0x{bits:016X}")
    }).collect();
    writeln!(w, "pub static ONE_PATH_GOAL_BITS: [u64; {n}] = [{}];", goal_bits.join(",")).unwrap();

    writeln!(w, "pub static ONE_PATH_ENDS: [[u8; {max_paths}]; {n}] = [").unwrap();
    for paths in &list_data {
        let mut row = vec![0u8; max_paths];
        for (i, &(sq, _)) in paths.iter().enumerate() { row[i] = sq; }
        writeln!(w, "[{}],", row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")).unwrap();
    }
    writeln!(w, "];").unwrap();

    writeln!(w, "pub static ONE_PATH_BACKS: [[u64; {max_paths}]; {n}] = [").unwrap();
    for paths in &list_data {
        let mut row = vec![0u64; max_paths];
        for (i, &(_, back)) in paths.iter().enumerate() { row[i] = back; }
        writeln!(w, "[{}],", row.iter().map(|v| format!("0x{v:016X}")).collect::<Vec<_>>().join(",")).unwrap();
    }
    writeln!(w, "];").unwrap();
    writeln!(w, "pub const ONE_MAX_PATHS: usize = {max_paths};").unwrap();

    println!("Written to {out_path}");
}
