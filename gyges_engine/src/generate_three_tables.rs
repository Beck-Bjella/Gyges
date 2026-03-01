// use std::collections::HashMap;
// use std::fs::File;
// use std::io::{BufWriter, Write};

// // ── BOARD LAYOUT ─────────────────────────────────────────────────────────────
// //
// // 6x6 board, squares numbered 0-35:
// //   row 5 (top):    30 31 32 33 34 35
// //   row 4:          24 25 26 27 28 29
// //   row 3:          18 19 20 21 22 23
// //   row 2:          12 13 14 15 16 17
// //   row 1:           6  7  8  9 10 11
// //   row 0 (bottom):  0  1  2  3  4  5
// //
// // Goal squares: 36 = south goal (off bottom), 37 = north goal (off top)
// //
// // ── INTERCEPT BIT LAYOUT ─────────────────────────────────────────────────────
// //
// // 60 total intercept bits in a u64, laid out as:
// //   For each row 0..5:
// //     5 horizontal intercepts (between adjacent squares in same row):
// //       h_bit(row, col) = row * 11 + col       (col 0..4)
// //     6 vertical intercepts (between this row and row above):
// //       v_bit(row, col) = row * 11 + 5 + col   (col 0..5, row 0..4 only)
// //
// // Example for row 0:
// //   bits 0-4:  horizontal intercepts between sq0-1, sq1-2, sq2-3, sq3-4, sq4-5
// //   bits 5-10: vertical intercepts between row0 and row1, columns 0-5
// //
// // A path "crosses" an intercept when it moves through that edge between squares.
// // If that intercept bit is set in `blocked`, the path cannot be taken.

// fn h_bit(row: usize, col: usize) -> u64 {
//     // horizontal intercept between (row,col) and (row,col+1)
//     debug_assert!(col < 5, "no h_bit for rightmost column");
//     1u64 << (row * 11 + col)
// }

// fn v_bit(row: usize, col: usize) -> u64 {
//     // vertical intercept between (row,col) and (row+1,col)
//     debug_assert!(row < 5, "no v_bit above top row");
//     1u64 << (row * 11 + 5 + col)
// }

// // ── STEP FUNCTION ────────────────────────────────────────────────────────────
// //
// // Move one step from sq in direction dir.
// // Returns (new_square, intercept_bit_crossed).
// // Returns None if the move goes off the board sideways (east/west walls).
// // Goal squares (36, 37) are returned for north/south off-board moves.
// // Intercept bit is 0 for goal moves — nothing can block the step off the board.
// //
// // Directions: 0=north, 1=south, 2=east, 3=west

// fn step(sq: usize, dir: u8) -> Option<(usize, u64)> {
//     if sq >= 36 {
//         // can't step from a goal square — paths end there
//         return None;
//     }
//     let row = sq / 6;
//     let col = sq % 6;
//     match dir {
//         0 => if row < 5 {
//             // normal north step — cross the vertical intercept above this square
//             Some((sq + 6, v_bit(row, col)))
//         } else {
//             // stepping off the top — goal 37, no intercept bit (nothing blocks it)
//             Some((37, 0))
//         },
//         1 => if row > 0 {
//             // normal south step — cross the vertical intercept below this square
//             Some((sq - 6, v_bit(row - 1, col)))
//         } else {
//             // stepping off the bottom — goal 36, no intercept bit
//             Some((36, 0))
//         },
//         2 => if col < 5 {
//             // east step — cross the horizontal intercept to the right
//             Some((sq + 1, h_bit(row, col)))
//         } else {
//             None // wall on east side
//         },
//         3 => if col > 0 {
//             // west step — cross the horizontal intercept to the left
//             Some((sq - 1, h_bit(row, col - 1)))
//         } else {
//             None // wall on west side
//         },
//         _ => unreachable!(),
//     }
// }

// fn opposite(dir: u8) -> u8 {
//     // you can't go back the way you came
//     match dir { 0 => 1, 1 => 0, 2 => 3, 3 => 2, _ => unreachable!() }
// }

// fn secondary_dirs(dir: u8) -> [u8; 3] {
//     // all directions except the one you just came from
//     let back = opposite(dir);
//     let mut result = [0u8; 3];
//     let mut i = 0;
//     for d in 0u8..4 {
//         if d != back {
//             result[i] = d;
//             i += 1;
//         }
//     }
//     result
// }

// // ── DIRECTION MASK ───────────────────────────────────────────────────────────
// //
// // For a given square and initial direction, compute the set of ALL intercept
// // bits that could possibly affect paths going in that direction.
// // This is used as the pext mask — only these bits are relevant for the lookup key.

// fn dir_mask(sq: usize, init_dir: u8) -> u64 {
//     let mut mask = 0u64;

//     // step 1: must go init_dir first
//     let (sq1, i1) = match step(sq, init_dir) {
//         Some(v) => v,
//         None => return 0, // can't go this direction at all
//     };
//     mask |= i1; // i1 is the intercept crossed at step 1

//     if sq1 >= 36 {
//         return mask; // hit goal at step 1 — no further steps possible
//     }

//     // step 2: can go any direction except back
//     for &d2 in &secondary_dirs(init_dir) {
//         if let Some((sq2, i2)) = step(sq1, d2) {
//             mask |= i2; // i2 is the intercept crossed at step 2

//             if sq2 >= 36 {
//                 continue; // hit goal at step 2 — no step 3
//             }

//             // step 3: can go any direction except back from step 2
//             for &d3 in &secondary_dirs(d2) {
//                 if let Some((_sq3, i3)) = step(sq2, d3) {
//                     mask |= i3; // i3 is the intercept crossed at step 3
//                 }
//             }
//         }
//     }

//     mask
// }

// // ── PATH WALKING ─────────────────────────────────────────────────────────────
// //
// // For a given square, initial direction, and set of blocked intercepts,
// // enumerate all valid THREE paths.
// //
// // A path is valid if none of its 3 intercepts are in `blocked`.
// // Returns list of (end_sq, back_mask) where:
// //   end_sq   = the destination square (0-35 normal, 36/37 goal)
// //   back_mask = the intercept bits this path crossed (used for backtracking)
// //
// // For goal paths, back_mask only includes steps 1 and 2 (not step 3),
// // because step 3 goes off the board and nothing can block it —
// // including it would cause the goal to disappear when unrelated intercepts are blocked.

// /// Brute force — scan all 8389 path lists, collect paths that:
// /// 1. Start from sq going init_dir (first intermediate square matches)
// /// 2. Are not blocked by any intercept in `blocked`
// /// 3. For goal paths, only check back & blocked for steps 1+2 (not step 3)
// fn get_valid_paths(
//     sq: usize,
//     init_dir: u8,
//     blocked: u64,
//     all_lists: &[ThreePathList],
// ) -> Vec<(u8, u64)> {
//     // what square should the first intermediate step land on?
//     let expected_first = match init_dir {
//         0 => if sq / 6 < 5 { Some(sq + 6) } else { None }, // north
//         1 => if sq / 6 > 0 { Some(sq - 6) } else { None }, // south
//         2 => if sq % 6 < 5 { Some(sq + 1) } else { None }, // east
//         3 => if sq % 6 > 0 { Some(sq - 1) } else { None }, // west
//         _ => unreachable!(),
//     };
//     let expected_first = match expected_first {
//         Some(v) => v,
//         None => return Vec::new(), // can't go this direction from this square
//     };

//     let mut paths = Vec::new();

//     for list in all_lists {
//         for p in 0..list.count as usize {
//             let (squares, back) = list.paths[p];

//             // check this path starts from sq going init_dir
//             if squares[0] as usize != sq { continue; }    // must start from current sq
//             if squares[1] as usize != expected_first { continue; } // must go init_dir first

//             let end = squares[3] as u8; // final destination

//             if end >= 36 {
//                 // goal path — only check back for steps 1+2
//                 // back already has step 3 excluded (from old table generation)
//                 if back & blocked == 0 {
//                     paths.push((end, back));
//                 }
//             } else {
//                 // normal path — check full back mask
//                 if back & blocked == 0 {
//                     paths.push((end, back));
//                 }
//             }
//         }
//     }

//     // deduplicate — same path may appear in multiple lists
//     paths.sort();
//     paths.dedup();

//     paths
// }

// // ── COMBO EXPANSION ──────────────────────────────────────────────────────────
// //
// // This is the inverse of pext (parallel bits extract).
// // Given a dense combo index (0..2^n) and a mask with n set bits,
// // expand the combo back into the positions of the mask bits.
// //
// // Example: mask = 0b10100, combo = 0b01
// //   bit 0 of combo -> bit 2 of mask -> result bit 2 = 0
// //   bit 1 of combo -> bit 4 of mask -> result bit 4 = 1
// //   result = 0b10000
// //
// // This lets us enumerate every possible combination of blocked intercepts
// // for a given mask, in the same order that pext would compress them.

// pub fn expand_combo(combo: usize, mask: u64) -> u64 {
//     let mut blocked = 0u64;
//     let mut bit_idx = 0usize; // which bit of combo we're reading
//     let mut m = mask;
//     while m != 0 {
//         let b = m.trailing_zeros() as usize; // position of lowest set bit in mask
//         if (combo >> bit_idx) & 1 != 0 {
//             blocked |= 1u64 << b; // this intercept is blocked in this combo
//         }
//         m &= m - 1; // clear lowest set bit
//         bit_idx += 1;
//     }
//     blocked
// }

// // ── PATH LIST DEDUPLICATION ──────────────────────────────────────────────────
// //
// // Many (sq, direction, combo) combinations produce identical path lists.
// // We store each unique path list once and use a u16 index to reference it.
// // This keeps the tables small and cache-friendly.



// fn get_or_insert(
//     paths: &[(u8, u64)],
//     registry: &mut HashMap<Vec<(u8, u64)>, u16>,
//     list_data: &mut Vec<Vec<(u8, u64)>>,
// ) -> u16 {
//     // sort so that identical path lists with different orderings hash the same
//     let mut key = paths.to_vec();
//     key.sort();
//     if let Some(&idx) = registry.get(&key) {
//         return idx; // already seen this path list — reuse its index
//     }
//     let idx = list_data.len() as u16;
//     registry.insert(key, idx);
//     list_data.push(paths.to_vec());
//     idx
// }

// // ── MAIN GENERATION ──────────────────────────────────────────────────────────
// //
// // For each of the 4 directions, for each of the 36 squares:
// //   1. Compute dir_mask — which intercept bits matter for this direction
// //   2. For each possible combination of those bits being blocked (0..2^n):
// //      a. expand_combo — convert combo index to actual blocked intercept bits
// //      b. walk_paths — find all valid paths given those blocked intercepts
// //      c. get_or_insert — store the path list, get its index
// //   3. Store the index in THREE_{DIR}_TABLE[sq][combo]
// //
// // At runtime, lookup works as:
// //   blocked = backtrack_board | piece_intercepts  (in intercept bit space)
// //   key = pext(blocked, THREE_NORTH_MASK[sq])     (compress to dense index)
// //   idx = THREE_NORTH_TABLE[sq][key]              (get path list index)
// //   paths = THREE_PATH_ENDS[idx], THREE_PATH_BACKS[idx], etc.

// use gyges::moves::movegen_consts::ThreePathList;

// pub fn generate_three_tables(out_path: &str, all_lists: &[ThreePathList]) {
//     let dir_names = ["NORTH", "SOUTH", "EAST", "WEST"];

//     // shared path list registry across all directions and squares
//     let mut registry: HashMap<Vec<(u8, u64)>, u16> = HashMap::new();
//     let mut list_data: Vec<Vec<(u8, u64)>> = Vec::new();

//     let mut all_masks:        [[u64; 36]; 4]     = [[0; 36]; 4];
//     let mut all_index_tables: Vec<Vec<Vec<u16>>> = Vec::new();

//     for dir_idx in 0..4u8 {
//         println!("Generating {}...", dir_names[dir_idx as usize]);

//         // compute pext mask for every square in this direction
//         let masks: [u64; 36] = std::array::from_fn(|sq| dir_mask(sq, dir_idx));
//         all_masks[dir_idx as usize] = masks;

//         let mut index_table: Vec<Vec<u16>> = Vec::with_capacity(36);

//         for sq in 0..36usize {
//             let mask = masks[sq];
//             let bits = mask.count_ones() as usize;
//             let real_combos = 1usize << bits; // e.g. 13 bits -> 8192 combos

//             let mut sq_indices: Vec<u16> = Vec::with_capacity(real_combos);

//             for combo in 0..real_combos {
//                 // expand this combo index into actual blocked intercept bits
//                 let blocked = expand_combo(combo, mask);

//                 // find all valid paths given these blocked intercepts
//                 let paths = get_valid_paths(sq, dir_idx, blocked, all_lists);

//                 // store path list, get its index
//                 let idx = get_or_insert(&paths, &mut registry, &mut list_data);
//                 sq_indices.push(idx);
//             }

//             // no padding needed — we use real_combos exactly

//             index_table.push(sq_indices);
//         }

//         all_index_tables.push(index_table);
//     }

//     println!("Total unique path lists: {}", list_data.len());
//     println!("Max paths in any list: {}", list_data.iter().map(|p| p.len()).max().unwrap_or(0));

//     // ── WRITE OUTPUT ──────────────────────────────────────────────────────────

//     let file = File::create(out_path).expect("failed to create output file");
//     let mut w = BufWriter::new(file);

//     writeln!(w, "// Auto-generated THREE directional path tables").unwrap();
//     writeln!(w, "// DO NOT EDIT — regenerate with generate_three_tables\n").unwrap();

//     // pext masks — one array per direction
//     for dir_idx in 0..4usize {
//         let name = dir_names[dir_idx];
//         let vals: Vec<String> = (0..36)
//             .map(|sq| format!("0x{:016X}", all_masks[dir_idx][sq]))
//             .collect();
//         writeln!(w, "pub static THREE_{name}_MASK: [u64; 36] = [{}];", vals.join(", ")).unwrap();
//     }
//     writeln!(w).unwrap();

//     // index tables — THREE_{DIR}_TABLE[sq][pext_key] -> u16 index into path lists
//     for dir_idx in 0..4usize {
//         let name = dir_names[dir_idx];
//         writeln!(w, "pub static THREE_{name}_TABLE: [[u16; 8192]; 36] = [").unwrap();
//         for sq in 0..36usize {
//             let row = &all_index_tables[dir_idx][sq];
//             // pad to 8192 with combo-0 value
//             let pad = row[0];
//             let mut padded = row.clone();
//             padded.resize(8192, pad);
//             let vals: Vec<String> = padded.chunks(64)
//                 .map(|c| c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
//                 .collect();
//             writeln!(w, "    [{}],", vals.join(",")).unwrap();
//         }
//         writeln!(w, "];
// ").unwrap();
//     }

//     // unique path list data
//     let n = list_data.len();
//     let max_paths = list_data.iter().map(|p| p.len()).max().unwrap_or(0);
//     writeln!(w, "// {n} unique path lists, max {max_paths} paths each\n").unwrap();

//     // count of valid paths per list
//     let counts: Vec<String> = list_data.iter().map(|p| p.len().to_string()).collect();
//     writeln!(w, "pub static THREE_PATH_COUNT: [u8; {n}] = [{}];", counts.join(",")).unwrap();

//     // bitboard of non-goal end squares per list
//     let end_bits_vals: Vec<String> = list_data.iter().map(|paths| {
//         let bits: u64 = paths.iter()
//             .filter(|&&(sq, _)| sq < 36)
//             .fold(0, |a, &(sq, _)| a | (1u64 << sq));
//         format!("0x{bits:016X}")
//     }).collect();
//     writeln!(w, "pub static THREE_PATH_END_BITS: [u64; {n}] = [{}];", end_bits_vals.join(",")).unwrap();

//     // bitboard of goal end squares per list
//     let goal_bits_vals: Vec<String> = list_data.iter().map(|paths| {
//         let bits: u64 = paths.iter()
//             .filter(|&&(sq, _)| sq >= 36)
//             .fold(0, |a, &(sq, _)| a | (1u64 << sq));
//         format!("0x{bits:016X}")
//     }).collect();
//     writeln!(w, "pub static THREE_PATH_GOAL_BITS: [u64; {n}] = [{}];", goal_bits_vals.join(",")).unwrap();

//     // end squares padded to max_paths with 0
//     writeln!(w, "pub static THREE_PATH_ENDS: [[u8; {max_paths}]; {n}] = [").unwrap();
//     for paths in &list_data {
//         let mut row = vec![0u8; max_paths];
//         for (i, &(sq, _)) in paths.iter().enumerate() { row[i] = sq; }
//         let s: Vec<String> = row.iter().map(|v| v.to_string()).collect();
//         writeln!(w, "[{}],", s.join(",")).unwrap();
//     }
//     writeln!(w, "];").unwrap();

//     // back masks padded to max_paths with 0
//     writeln!(w, "pub static THREE_PATH_BACKS: [[u64; {max_paths}]; {n}] = [").unwrap();
//     for paths in &list_data {
//         let mut row = vec![0u64; max_paths];
//         for (i, &(_, back)) in paths.iter().enumerate() { row[i] = back; }
//         let s: Vec<String> = row.iter().map(|v| format!("0x{v:016X}")).collect();
//         writeln!(w, "[{}],", s.join(",")).unwrap();
//     }
//     writeln!(w, "];").unwrap();

//     writeln!(w, "pub const THREE_MAX_PATHS: usize = {max_paths};").unwrap();

//     println!("Written to {out_path}");
// }

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use gyges::moves::movegen_consts::ThreePathList;

// ── BOARD LAYOUT ─────────────────────────────────────────────────────────────
//
// 6x6 board, squares numbered 0-35:
//   row 5 (top):    30 31 32 33 34 35
//   row 4:          24 25 26 27 28 29
//   row 3:          18 19 20 21 22 23
//   row 2:          12 13 14 15 16 17
//   row 1:           6  7  8  9 10 11
//   row 0 (bottom):  0  1  2  3  4  5
//
// Goal squares: 36 = south goal (off bottom), 37 = north goal (off top)
//
// ── INTERCEPT BIT LAYOUT ─────────────────────────────────────────────────────
//
// 60 total intercept bits in a u64, laid out as:
//   For each row 0..5:
//     5 horizontal intercepts (between adjacent squares in same row):
//       h_bit(row, col) = row * 11 + col       (col 0..4)
//     6 vertical intercepts (between this row and row above):
//       v_bit(row, col) = row * 11 + 5 + col   (col 0..5, row 0..4 only)
//
// Example for row 0:
//   bits 0-4:  horizontal intercepts between sq0-1, sq1-2, sq2-3, sq3-4, sq4-5
//   bits 5-10: vertical intercepts between row0 and row1, columns 0-5
//
// A path "crosses" an intercept when it moves through that edge between squares.
// If that intercept bit is set in `blocked`, the path cannot be taken.

fn h_bit(row: usize, col: usize) -> u64 {
    // horizontal intercept between (row,col) and (row,col+1)
    debug_assert!(col < 5, "no h_bit for rightmost column");
    1u64 << (row * 11 + col)
}

fn v_bit(row: usize, col: usize) -> u64 {
    // vertical intercept between (row,col) and (row+1,col)
    debug_assert!(row < 5, "no v_bit above top row");
    1u64 << (row * 11 + 5 + col)
}

// ── STEP FUNCTION ────────────────────────────────────────────────────────────
//
// Move one step from sq in direction dir.
// Returns (new_square, intercept_bit_crossed).
// Returns None if the move goes off the board sideways (east/west walls).
// Goal squares (36, 37) are returned for north/south off-board moves.
// Intercept bit is 0 for goal moves — nothing can block the step off the board.
//
// Directions: 0=north, 1=south, 2=east, 3=west

fn step(sq: usize, dir: u8) -> Option<(usize, u64)> {
    if sq >= 36 {
        // can't step from a goal square — paths end there
        return None;
    }
    let row = sq / 6;
    let col = sq % 6;
    match dir {
        0 => if row < 5 {
            // normal north step — cross the vertical intercept above this square
            Some((sq + 6, v_bit(row, col)))
        } else {
            // stepping off the top — goal 37, no intercept bit (nothing blocks it)
            Some((37, 0))
        },
        1 => if row > 0 {
            // normal south step — cross the vertical intercept below this square
            Some((sq - 6, v_bit(row - 1, col)))
        } else {
            // stepping off the bottom — goal 36, no intercept bit
            Some((36, 0))
        },
        2 => if col < 5 {
            // east step — cross the horizontal intercept to the right
            Some((sq + 1, h_bit(row, col)))
        } else {
            None // wall on east side
        },
        3 => if col > 0 {
            // west step — cross the horizontal intercept to the left
            Some((sq - 1, h_bit(row, col - 1)))
        } else {
            None // wall on west side
        },
        _ => unreachable!(),
    }
}

fn opposite(dir: u8) -> u8 {
    // you can't go back the way you came
    match dir { 0 => 1, 1 => 0, 2 => 3, 3 => 2, _ => unreachable!() }
}

fn secondary_dirs(dir: u8) -> [u8; 3] {
    // all directions except the one you just came from
    let back = opposite(dir);
    let mut result = [0u8; 3];
    let mut i = 0;
    for d in 0u8..4 {
        if d != back {
            result[i] = d;
            i += 1;
        }
    }
    result
}

// ── DIRECTION MASK ───────────────────────────────────────────────────────────
//
// For a given square and initial direction, compute the set of ALL intercept
// bits that could possibly affect paths going in that direction.
// This is used as the pext mask — only these bits are relevant for the lookup key.

fn dir_mask(sq: usize, init_dir: u8) -> u64 {
    let mut mask = 0u64;

    // step 1: must go init_dir first
    let (sq1, i1) = match step(sq, init_dir) {
        Some(v) => v,
        None => return 0, // can't go this direction at all
    };
    mask |= i1; // i1 is the intercept crossed at step 1

    if sq1 >= 36 {
        return mask; // hit goal at step 1 — no further steps possible
    }

    // step 2: can go any direction except back
    for &d2 in &secondary_dirs(init_dir) {
        if let Some((sq2, i2)) = step(sq1, d2) {
            mask |= i2; // i2 is the intercept crossed at step 2

            if sq2 >= 36 {
                continue; // hit goal at step 2 — no step 3
            }

            // step 3: can go any direction except back from step 2
            for &d3 in &secondary_dirs(d2) {
                if let Some((_sq3, i3)) = step(sq2, d3) {
                    mask |= i3; // i3 is the intercept crossed at step 3
                }
            }
        }
    }

    mask
}

// ── PATH WALKING ─────────────────────────────────────────────────────────────
//
// For a given square, initial direction, and set of blocked intercepts,
// enumerate all valid THREE paths.
//
// A path is valid if none of its 3 intercepts are in `blocked`.
// Returns list of (end_sq, back_mask) where:
//   end_sq   = the destination square (0-35 normal, 36/37 goal)
//   back_mask = the intercept bits this path crossed (used for backtracking)
//
// For goal paths, back_mask only includes steps 1 and 2 (not step 3),
// because step 3 goes off the board and nothing can block it —
// including it would cause the goal to disappear when unrelated intercepts are blocked.

/// Brute force — scan all 8389 path lists, collect paths that:
/// 1. Start from sq going init_dir (first intermediate square matches)
/// 2. Are not blocked by any intercept in `blocked`
/// 3. For goal paths, only check back & blocked for steps 1+2 (not step 3)
fn get_valid_paths(
    sq: usize,
    init_dir: u8,
    blocked: u64,
    all_lists: &[ThreePathList],
) -> Vec<(u8, u64)> {
    // what square should the first intermediate step land on?
    let expected_first = match init_dir {
        0 => if sq / 6 < 5 { Some(sq + 6) } else { None }, // north
        1 => if sq / 6 > 0 { Some(sq - 6) } else { None }, // south
        2 => if sq % 6 < 5 { Some(sq + 1) } else { None }, // east
        3 => if sq % 6 > 0 { Some(sq - 1) } else { None }, // west
        _ => unreachable!(),
    };
    let expected_first = match expected_first {
        Some(v) => v,
        None => return Vec::new(), // can't go this direction from this square
    };

    let mut paths = Vec::new();

    for list in all_lists {
        for p in 0..list.count as usize {
            let (squares, back) = list.paths[p];

            // check this path starts from sq going init_dir
            if squares[0] as usize != sq { continue; }    // must start from current sq
            if squares[1] as usize != expected_first { continue; } // must go init_dir first

            let end = squares[3] as u8; // final destination

            if end >= 36 {
                // goal path — only check back for steps 1+2
                // back already has step 3 excluded (from old table generation)
                if back & blocked == 0 {
                    paths.push((end, back));
                }
            } else {
                // normal path — check full back mask
                if back & blocked == 0 {
                    paths.push((end, back));
                }
            }
        }
    }

    // verify back is always subset of directional mask
    let dmask = dir_mask(sq, init_dir);
    for &(end, back) in &paths {
        if back & !dmask != 0 {
            println!("WARN sq{} dir{} end{}: back has bits outside dir_mask! extra={:064b}", sq, init_dir, end, back & !dmask);
        }
    }

    // deduplicate — same path may appear in multiple lists
    paths.sort();
    paths.dedup();

    paths
}

// ── COMBO EXPANSION ──────────────────────────────────────────────────────────
//
// This is the inverse of pext (parallel bits extract).
// Given a dense combo index (0..2^n) and a mask with n set bits,
// expand the combo back into the positions of the mask bits.
//
// Example: mask = 0b10100, combo = 0b01
//   bit 0 of combo -> bit 2 of mask -> result bit 2 = 0
//   bit 1 of combo -> bit 4 of mask -> result bit 4 = 1
//   result = 0b10000
//
// This lets us enumerate every possible combination of blocked intercepts
// for a given mask, in the same order that pext would compress them.

fn expand_combo(combo: usize, mask: u64) -> u64 {
    let mut blocked = 0u64;
    let mut bit_idx = 0usize; // which bit of combo we're reading
    let mut m = mask;
    while m != 0 {
        let b = m.trailing_zeros() as usize; // position of lowest set bit in mask
        if (combo >> bit_idx) & 1 != 0 {
            blocked |= 1u64 << b; // this intercept is blocked in this combo
        }
        m &= m - 1; // clear lowest set bit
        bit_idx += 1;
    }
    blocked
}

// ── PATH LIST DEDUPLICATION ──────────────────────────────────────────────────
//
// Many (sq, direction, combo) combinations produce identical path lists.
// We store each unique path list once and use a u16 index to reference it.
// This keeps the tables small and cache-friendly.



fn get_or_insert(
    paths: &[(u8, u64)],
    registry: &mut HashMap<Vec<(u8, u64)>, u16>,
    list_data: &mut Vec<Vec<(u8, u64)>>,
) -> u16 {
    // sort so that identical path lists with different orderings hash the same
    let mut key = paths.to_vec();
    key.sort();
    if let Some(&idx) = registry.get(&key) {
        return idx; // already seen this path list — reuse its index
    }
    let idx = list_data.len() as u16;
    registry.insert(key, idx);
    list_data.push(paths.to_vec());
    idx
}

// ── MAIN GENERATION ──────────────────────────────────────────────────────────
//
// For each of the 4 directions, for each of the 36 squares:
//   1. Compute dir_mask — which intercept bits matter for this direction
//   2. For each possible combination of those bits being blocked (0..2^n):
//      a. expand_combo — convert combo index to actual blocked intercept bits
//      b. walk_paths — find all valid paths given those blocked intercepts
//      c. get_or_insert — store the path list, get its index
//   3. Store the index in THREE_{DIR}_TABLE[sq][combo]
//
// At runtime, lookup works as:
//   blocked = backtrack_board | piece_intercepts  (in intercept bit space)
//   key = pext(blocked, THREE_NORTH_MASK[sq])     (compress to dense index)
//   idx = THREE_NORTH_TABLE[sq][key]              (get path list index)
//   paths = THREE_PATH_ENDS[idx], THREE_PATH_BACKS[idx], etc.

pub fn generate_three_tables(out_path: &str, all_lists: &[ThreePathList]) {
    let dir_names = ["NORTH", "SOUTH", "EAST", "WEST"];

    // shared path list registry across all directions and squares
    let mut registry: HashMap<Vec<(u8, u64)>, u16> = HashMap::new();
    let mut list_data: Vec<Vec<(u8, u64)>> = Vec::new();

    let mut all_masks:        [[u64; 36]; 4]     = [[0; 36]; 4];
    let mut all_index_tables: Vec<Vec<Vec<u16>>> = Vec::new();

    for dir_idx in 0..4u8 {
        println!("Generating {}...", dir_names[dir_idx as usize]);

        // compute pext mask for every square in this direction
        let masks: [u64; 36] = std::array::from_fn(|sq| dir_mask(sq, dir_idx));
        all_masks[dir_idx as usize] = masks;

        let mut index_table: Vec<Vec<u16>> = Vec::with_capacity(36);

        for sq in 0..36usize {
            let mask = masks[sq];
            let bits = mask.count_ones() as usize;
            let real_combos = 1usize << bits; // e.g. 13 bits -> 8192 combos

            let mut sq_indices: Vec<u16> = Vec::with_capacity(real_combos);

            for combo in 0..real_combos {
                // expand this combo index into actual blocked intercept bits
                let blocked = expand_combo(combo, mask);

                // find all valid paths given these blocked intercepts
                let paths = get_valid_paths(sq, dir_idx, blocked, all_lists);

                // store path list, get its index
                let idx = get_or_insert(&paths, &mut registry, &mut list_data);
                sq_indices.push(idx);
            }

            // no padding needed — we use real_combos exactly

            index_table.push(sq_indices);
        }

        all_index_tables.push(index_table);
    }

    println!("Total unique path lists: {}", list_data.len());
    println!("Max paths in any list: {}", list_data.iter().map(|p| p.len()).max().unwrap_or(0));

    // ── WRITE OUTPUT ──────────────────────────────────────────────────────────

    let file = File::create(out_path).expect("failed to create output file");
    let mut w = BufWriter::new(file);

    writeln!(w, "// Auto-generated THREE directional path tables").unwrap();
    writeln!(w, "// DO NOT EDIT — regenerate with generate_three_tables\n").unwrap();

    // pext masks — one array per direction
    for dir_idx in 0..4usize {
        let name = dir_names[dir_idx];
        let vals: Vec<String> = (0..36)
            .map(|sq| format!("0x{:016X}", all_masks[dir_idx][sq]))
            .collect();
        writeln!(w, "pub static THREE_{name}_MASK: [u64; 36] = [{}];", vals.join(", ")).unwrap();
    }
    writeln!(w).unwrap();

    // index tables — THREE_{DIR}_TABLE[sq][pext_key] -> u16 index into path lists
    for dir_idx in 0..4usize {
        let name = dir_names[dir_idx];
        writeln!(w, "pub static THREE_{name}_TABLE: [[u16; 8192]; 36] = [").unwrap();
        for sq in 0..36usize {
            let row = &all_index_tables[dir_idx][sq];
            // pad to 8192 with combo-0 value
            let pad = row[0];
            let mut padded = row.clone();
            padded.resize(8192, pad);
            let vals: Vec<String> = padded.chunks(64)
                .map(|c| c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
                .collect();
            writeln!(w, "    [{}],", vals.join(",")).unwrap();
        }
        writeln!(w, "];
").unwrap();
    }

    // unique path list data
    let n = list_data.len();
    let max_paths = list_data.iter().map(|p| p.len()).max().unwrap_or(0);
    writeln!(w, "// {n} unique path lists, max {max_paths} paths each\n").unwrap();

    // count of valid paths per list
    let counts: Vec<String> = list_data.iter().map(|p| p.len().to_string()).collect();
    writeln!(w, "pub static THREE_PATH_COUNT: [u8; {n}] = [{}];", counts.join(",")).unwrap();

    // bitboard of non-goal end squares per list
    let end_bits_vals: Vec<String> = list_data.iter().map(|paths| {
        let bits: u64 = paths.iter()
            .filter(|&&(sq, _)| sq < 36)
            .fold(0, |a, &(sq, _)| a | (1u64 << sq));
        format!("0x{bits:016X}")
    }).collect();
    writeln!(w, "pub static THREE_PATH_END_BITS: [u64; {n}] = [{}];", end_bits_vals.join(",")).unwrap();

    // bitboard of goal end squares per list
    let goal_bits_vals: Vec<String> = list_data.iter().map(|paths| {
        let bits: u64 = paths.iter()
            .filter(|&&(sq, _)| sq >= 36)
            .fold(0, |a, &(sq, _)| a | (1u64 << sq));
        format!("0x{bits:016X}")
    }).collect();
    writeln!(w, "pub static THREE_PATH_GOAL_BITS: [u64; {n}] = [{}];", goal_bits_vals.join(",")).unwrap();

    // end squares padded to max_paths with 0
    writeln!(w, "pub static THREE_PATH_ENDS: [[u8; {max_paths}]; {n}] = [").unwrap();
    for paths in &list_data {
        let mut row = vec![0u8; max_paths];
        for (i, &(sq, _)) in paths.iter().enumerate() { row[i] = sq; }
        let s: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        writeln!(w, "[{}],", s.join(",")).unwrap();
    }
    writeln!(w, "];").unwrap();

    // back masks padded to max_paths with 0
    writeln!(w, "pub static THREE_PATH_BACKS: [[u64; {max_paths}]; {n}] = [").unwrap();
    for paths in &list_data {
        let mut row = vec![0u64; max_paths];
        for (i, &(_, back)) in paths.iter().enumerate() { row[i] = back; }
        let s: Vec<String> = row.iter().map(|v| format!("0x{v:016X}")).collect();
        writeln!(w, "[{}],", s.join(",")).unwrap();
    }
    writeln!(w, "];").unwrap();

    writeln!(w, "pub const THREE_MAX_PATHS: usize = {max_paths};").unwrap();

    println!("Written to {out_path}");
}
