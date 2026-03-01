use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use gyges::moves::movegen_consts::TwoPathList;

fn h_bit(row: usize, col: usize) -> u64 { 1u64 << (row * 11 + col) }
fn v_bit(row: usize, col: usize) -> u64 { 1u64 << (row * 11 + 5 + col) }

fn step(sq: usize, dir: u8) -> Option<(usize, u64)> {
    if sq >= 36 { return None; }
    let row = sq / 6;
    let col = sq % 6;
    match dir {
        0 => if row < 5 { Some((sq + 6, v_bit(row, col))) } else { Some((37, 0)) },
        1 => if row > 0 { Some((sq - 6, v_bit(row - 1, col))) } else { Some((36, 0)) },
        2 => if col < 5 { Some((sq + 1, h_bit(row, col))) } else { None },
        3 => if col > 0 { Some((sq - 1, h_bit(row, col - 1))) } else { None },
        _ => unreachable!(),
    }
}

fn opposite(dir: u8) -> u8 { match dir { 0=>1, 1=>0, 2=>3, 3=>2, _=>unreachable!() } }

fn secondary_dirs(dir: u8) -> [u8; 3] {
    let back = opposite(dir);
    let mut result = [0u8; 3];
    let mut i = 0;
    for d in 0u8..4 { if d != back { result[i] = d; i += 1; } }
    result
}

// TWO: 2 steps, can turn at step 2
// dir_mask = i1 (step1 intercept) + all possible i2 intercepts
fn dir_mask_two(sq: usize, init_dir: u8) -> u64 {
    let mut mask = 0u64;
    let (sq1, i1) = match step(sq, init_dir) {
        Some(v) => v,
        None => return 0,
    };
    mask |= i1;
    if sq1 >= 36 { return mask; } // goal at step 1 — no step 2
    for &d2 in &secondary_dirs(init_dir) {
        if let Some((_sq2, i2)) = step(sq1, d2) {
            mask |= i2;
        }
    }
    mask
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

// TWO: squares = [start, step1, end], back = i1 | i2
// Filter: squares[0] == sq, squares[1] == expected_first (init_dir step)
fn get_valid_paths_two(
    sq: usize,
    init_dir: u8,
    blocked: u64,
    all_lists: &[TwoPathList],
) -> Vec<(u8, u64)> {
    let row = sq / 6;
    let col = sq % 6;
    let expected_first = match init_dir {
        0 => if row < 5 { Some(sq + 6) } else { return vec![] }, // goals too early for TWO
        1 => if row > 0 { Some(sq - 6) } else { return vec![] },
        2 => if col < 5 { Some(sq + 1) } else { return vec![] },
        3 => if col > 0 { Some(sq - 1) } else { return vec![] },
        _ => unreachable!(),
    };
    let expected_first = match expected_first {
        Some(v) => v,
        None => return vec![],
    };

    let mut paths = Vec::new();
    for list in all_lists {
        for p in 0..list.count as usize {
            let (squares, back) = list.paths[p];
            if squares[0] as usize != sq { continue; }
            if squares[1] as usize != expected_first { continue; }
            // goal paths: end is sq >= 36, back only has i1 (step into goal can't be blocked)
            if back & blocked == 0 {
                paths.push((squares[2], back));
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

// TWO_PIECE_INTERCEPTS[sq][s]: intercepts blocked by a piece at s for a TWO at sq
// For TWO, only adjacent squares can be intermediates (step 1)
// So only adjacent cases matter — single intercept between sq and s
pub fn generate_two_piece_intercepts() -> [[u64; 36]; 36] {
    let mut table = [[0u64; 36]; 36];
    for sq in 0..36usize {
        let sq_row = sq / 6;
        let sq_col = sq % 6;
        for s in 0..36usize {
            let s_row = s / 6;
            let s_col = s % 6;
            let dr = s_row as i32 - sq_row as i32;
            let dc = s_col as i32 - sq_col as i32;
            table[sq][s] = match (dr, dc) {
                ( 1,  0) => v_bit(sq_row, sq_col),       // s is north
                (-1,  0) => v_bit(s_row,  sq_col),       // s is south
                ( 0,  1) => h_bit(sq_row, sq_col),       // s is east
                ( 0, -1) => h_bit(sq_row, s_col),        // s is west
                _        => 0,                            // not adjacent — not an intermediate
            };
        }
    }
    table
}

// ALL_TWO_INTERCEPTS[sq]: squares that can be intermediates for TWO paths from sq
// = the 4 adjacent squares (step 1 only, step 2 is always an endpoint)
pub fn generate_all_two_intercepts() -> [u64; 36] {
    let mut table = [0u64; 36];
    for sq in 0..36usize {
        let row = sq / 6;
        let col = sq % 6;
        let mut bb = 0u64;
        if row < 5 { bb |= 1u64 << (sq + 6); } // north
        if row > 0 { bb |= 1u64 << (sq - 6); } // south
        if col < 5 { bb |= 1u64 << (sq + 1); } // east
        if col > 0 { bb |= 1u64 << (sq - 1); } // west
        table[sq] = bb;
    }
    table
}

pub fn generate_two_tables(out_path: &str, all_lists: &[TwoPathList]) {
    let dir_names = ["NORTH", "SOUTH", "EAST", "WEST"];

    let mut registry: HashMap<Vec<(u8, u64)>, u16> = HashMap::new();
    let mut list_data: Vec<Vec<(u8, u64)>> = Vec::new();

    let mut all_masks: [[u64; 36]; 4] = [[0; 36]; 4];
    let mut all_index_tables: Vec<Vec<Vec<u16>>> = Vec::new();

    for dir_idx in 0..4u8 {
        println!("Generating TWO {}...", dir_names[dir_idx as usize]);

        let masks: [u64; 36] = std::array::from_fn(|sq| dir_mask_two(sq, dir_idx));
        all_masks[dir_idx as usize] = masks;

        let mut index_table: Vec<Vec<u16>> = Vec::with_capacity(36);

        for sq in 0..36usize {
            let mask = masks[sq];
            let bits = mask.count_ones() as usize;
            let real_combos = if bits == 0 { 1 } else { 1usize << bits };

            let mut sq_indices: Vec<u16> = Vec::with_capacity(real_combos);
            for combo in 0..real_combos {
                let blocked = expand_combo(combo, mask);
                let paths = get_valid_paths_two(sq, dir_idx, blocked, all_lists);
                let idx = get_or_insert(&paths, &mut registry, &mut list_data);
                sq_indices.push(idx);
            }

            // pad to 16 (max combos for TWO — up to 4 bits: 1 step1 + 3 step2)
            let pad = sq_indices[0];
            sq_indices.resize(16, pad);
            index_table.push(sq_indices);
        }

        all_index_tables.push(index_table);
    }

    println!("Total unique TWO path lists: {}", list_data.len());
    println!("Max paths in any TWO list: {}", list_data.iter().map(|p| p.len()).max().unwrap_or(0));

    let file = File::create(out_path).expect("failed to create output file");
    let mut w = BufWriter::new(file);

    writeln!(w, "// Auto-generated TWO directional path tables\n").unwrap();

    // masks
    for dir_idx in 0..4usize {
        let name = dir_names[dir_idx];
        let vals: Vec<String> = (0..36).map(|sq| format!("0x{:016X}", all_masks[dir_idx][sq])).collect();
        writeln!(w, "pub static TWO_{name}_MASK: [u64; 36] = [{}];", vals.join(", ")).unwrap();
    }
    writeln!(w).unwrap();

    // index tables — [[u16; 16]; 36]
    for dir_idx in 0..4usize {
        let name = dir_names[dir_idx];
        writeln!(w, "pub static TWO_{name}_TABLE: [[u16; 16]; 36] = [").unwrap();
        for sq in 0..36 {
            let row = &all_index_tables[dir_idx][sq];
            let vals: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            writeln!(w, "    [{}],", vals.join(",")).unwrap();
        }
        writeln!(w, "];\n").unwrap();
    }

    let n = list_data.len();
    let max_paths = list_data.iter().map(|p| p.len()).max().unwrap_or(0);
    writeln!(w, "// {n} unique TWO path lists, max {max_paths} paths each\n").unwrap();

    let counts: Vec<String> = list_data.iter().map(|p| p.len().to_string()).collect();
    writeln!(w, "pub static TWO_PATH_COUNT: [u8; {n}] = [{}];", counts.join(",")).unwrap();

    let end_bits: Vec<String> = list_data.iter().map(|paths| {
        let bits: u64 = paths.iter().filter(|&&(sq, _)| sq < 36).fold(0, |a, &(sq, _)| a | (1u64 << sq));
        format!("0x{bits:016X}")
    }).collect();
    writeln!(w, "pub static TWO_PATH_END_BITS: [u64; {n}] = [{}];", end_bits.join(",")).unwrap();

    let goal_bits: Vec<String> = list_data.iter().map(|paths| {
        let bits: u64 = paths.iter().filter(|&&(sq, _)| sq >= 36).fold(0, |a, &(sq, _)| a | (1u64 << sq));
        format!("0x{bits:016X}")
    }).collect();
    writeln!(w, "pub static TWO_PATH_GOAL_BITS: [u64; {n}] = [{}];", goal_bits.join(",")).unwrap();

    writeln!(w, "pub static TWO_PATH_ENDS: [[u8; {max_paths}]; {n}] = [").unwrap();
    for paths in &list_data {
        let mut row = vec![0u8; max_paths];
        for (i, &(sq, _)) in paths.iter().enumerate() { row[i] = sq; }
        writeln!(w, "[{}],", row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")).unwrap();
    }
    writeln!(w, "];").unwrap();

    writeln!(w, "pub static TWO_PATH_BACKS: [[u64; {max_paths}]; {n}] = [").unwrap();
    for paths in &list_data {
        let mut row = vec![0u64; max_paths];
        for (i, &(_, back)) in paths.iter().enumerate() { row[i] = back; }
        writeln!(w, "[{}],", row.iter().map(|v| format!("0x{v:016X}")).collect::<Vec<_>>().join(",")).unwrap();
    }
    writeln!(w, "];").unwrap();
    writeln!(w, "pub const TWO_MAX_PATHS: usize = {max_paths};").unwrap();

    // also write the piece intercepts and all_intercepts tables
    let two_piece_intercepts = generate_two_piece_intercepts();
    writeln!(w, "\npub static TWO_PIECE_INTERCEPTS: [[u64; 36]; 36] = [").unwrap();
    for sq in 0..36 {
        let vals: Vec<String> = two_piece_intercepts[sq].iter().map(|v| format!("0x{v:016X}")).collect();
        writeln!(w, "    [{}], // sq{sq}", vals.join(", ")).unwrap();
    }
    writeln!(w, "];").unwrap();

    let all_two_intercepts = generate_all_two_intercepts();
    let vals: Vec<String> = all_two_intercepts.iter().map(|v| format!("0x{v:016X}")).collect();
    writeln!(w, "\npub static ALL_TWO_INTERCEPTS: [u64; 36] = [{}];", vals.join(", ")).unwrap();

    println!("Written to {out_path}");
}
