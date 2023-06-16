use crate::board::board::*;
use crate::board::bitboard::*;
use crate::consts::*;
use crate::moves::move_gen::*;

use std::fs::File;
use std::path::Path;
use std::io::{Write, BufWriter, Error};
use std::vec;

// pub fn gen_unique_three_paths() -> Result<(), Error> {
//     let path = Path::new("./unique_three_paths.txt");
//     let mut f = BufWriter::new(File::create(&path).unwrap());

//     let mut unique_three_paths = vec![];

//     let mut board = BoardState::new();
//     board.set(
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0],
//         PLAYER_1,

//     );
    
//     for pos in 0..36 {
//         let mut all_intercepts_bb = BitBoard(0);

//         // Generate all intercepts bitboard
//         for path_idx in 0..THREE_PATH_LENGTHS[pos] {
//             let path = THREE_PATHS[pos][path_idx];
    
//             all_intercepts_bb.set_bit(path[1]);
//             all_intercepts_bb.set_bit(path[2]);
    
//         }
    
//         let intercept_positions = all_intercepts_bb.clone().get_data();
    
//         let intercept_combos = generate_permutations(&intercept_positions);

//         for intercepts in intercept_combos {
//             let mut intercept_bb = ALL_INTERCEPTS[pos];
    
//             let mut all_paths = vec![];
    
//             // Check all paths and find vaild ones
//             for path_idx in 0..THREE_PATH_LENGTHS[pos] {
//                 let path = THREE_PATHS[pos][path_idx];
            
//                 if board.data[path[1]] != 0 {
//                     continue;
                    
//                 } else if board.data[path[2]] != 0 {
//                     continue;
                    
//                 }
        
//                 if path[3] == PLAYER_1_GOAL {
//                     continue;
        
//                 }

//                 let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];

//                 let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];

//                 let mut goal: bool = false;
//                 if path[3] == PLAYER_2_GOAL {
//                     goal = true;
        
//                 }
                
//                 let end_bb: u64 = 1 << path[3];
              
//                 let final_path = (small_path, backtrack_bb, goal);

//                 all_paths.push(final_path.clone());

//                 if !unique_three_paths.contains(&final_path) {
//                     unique_three_paths.push(final_path);
    
//                 }
        
//             }

//             // Undo intercepts
//             for pos in intercepts.clone() {
//                 board.data[*pos] = 0;
    
//             }
    
//         }

//     }

//     writeln!(f, "([100, 100, 100, 100], 0, false)")?;
//     for path in unique_three_paths {
//         writeln!(f, "({:?}, {}, {}),", path.0, path.1.0, path.2)?;

//     }
    

//     Ok(())

// }

// pub fn gen_unique_three_path_lists() -> Result<(), Error> {
//     let path = Path::new("./unique_three_path_lists.txt");
//     let mut f = BufWriter::new(File::create(&path).unwrap());

//     let mut unique_three_path_lists = vec![];

//     let mut board = BoardState::new();
//     board.set(
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0],
//         PLAYER_1,

//     );
    
//     for pos in 0..36 {
//         let mut all_intercepts_bb = BitBoard(0);

//         // Generate all intercepts bitboard
//         for path_idx in 0..THREE_PATH_LENGTHS[pos] {
//             let path = THREE_PATHS[pos][path_idx];
    
//             all_intercepts_bb.set_bit(path[1]);
//             all_intercepts_bb.set_bit(path[2]);
    
//         }
    
//         let intercept_positions = all_intercepts_bb.clone().get_data();
    
//         let intercept_combos = generate_permutations(&intercept_positions);

//         for intercepts in intercept_combos {
//             let mut intercept_bb = BitBoard(0);
    
//             // Place intercepts
//             for pos in intercepts.clone() {
//                 board.data[*pos] = 1;
//                 intercept_bb.set_bit(*pos);
    
//             }
    
//             let mut all_paths = vec![];

//             // Check all paths and find vaild ones
//             for path_idx in 0..THREE_PATH_LENGTHS[pos] {
//                 let path = THREE_PATHS[pos][path_idx];
            
//                 if board.data[path[1]] != 0 {
//                     continue;
                    
//                 } else if board.data[path[2]] != 0 {
//                     continue;
                    
//                 }
        
//                 if path[3] == PLAYER_1_GOAL {
//                     continue;
        
//                 }
        
//                 let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
//                 let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];
//                 let final_path = ThreePath {
//                     path: small_path,
//                     backtrack: backtrack_bb
        
//                 };

//                 all_paths.push(final_path.clone());

//             }

//             let mut all_array_paths = [ThreePath::empty(); 35];
//             for (i, path) in all_paths.iter().enumerate() {
//                 all_array_paths[i] = *path;

//             }

//             let mut array_idxs: [u16; 35] = [0; 35];
//             for ( i, p) in all_paths.iter().enumerate() {
//                 let tuple_path = (p.path, p.backtrack.0);
//                 let path_idx = UNIQUE_THREE_PATHS.iter().position(|p| p == &tuple_path).unwrap();
//                 array_idxs[i] = path_idx as u16;

//             }

//             let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
//             let three_path_array: [u16; 36] = whole.try_into().unwrap();

//             if !unique_three_path_lists.contains(&three_path_array) {
//                 unique_three_path_lists.push(three_path_array);

//             }

//             // Undo intercepts
//             for pos in intercepts.clone() {
//                 board.data[*pos] = 0;
    
//             }
    
//         }
        
//     }
    
//     for path in unique_three_path_lists {
//         writeln!(f, "{:?},", path)?;

//     }

//     Ok(())

// }

// pub fn gen_three_map() -> Result<(), Error> {
//     let path = Path::new("./gen_three_map.txt");
//     let mut f = BufWriter::new(File::create(&path).unwrap());

//     let mut board = BoardState::new();
//     board.set(
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0, 0, 0, 0, 0],
//         [0, 0],
//         PLAYER_1,

//     );
    
//     for pos in 0..36 {
//         let mut pos_data: Vec<u16> = vec![0; THREE_MAP_LEN];

//         let mut all_intercepts_bb = BitBoard(0);

//         // Generate all intercepts bitboard
//         for path_idx in 0..THREE_PATH_LENGTHS[pos] {
//             let path = THREE_PATHS[pos][path_idx];
    
//             all_intercepts_bb.set_bit(path[1]);
//             all_intercepts_bb.set_bit(path[2]);
    
//         }
    
//         let intercept_positions = all_intercepts_bb.clone().get_data();
    
//         let intercept_combos = generate_permutations(&intercept_positions);

//         for intercepts in intercept_combos {
//             let mut intercept_bb = BitBoard(0);
    
//             // Place intercepts
//             for pos in intercepts.clone() {
//                 board.data[*pos] = 1;
//                 intercept_bb.set_bit(*pos);
    
//             }
    
//             let mut all_paths = vec![];
    
//             // Check all paths and find vaild ones
//             for path_idx in 0..THREE_PATH_LENGTHS[pos] {
//                 let path = THREE_PATHS[pos][path_idx];
            
//                 if board.data[path[1]] != 0 {
//                     continue;
                    
//                 } else if board.data[path[2]] != 0 {
//                     continue;
                    
//                 }
        
//                 if path[3] == PLAYER_1_GOAL {
//                     continue;
        
//                 }
        
//                 let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
//                 let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];
//                 let final_path = ThreePath {
//                     path: small_path,
//                     backtrack: backtrack_bb
        
//                 };
    
//                 all_paths.push(final_path.clone());
        
//             }

//             let mut all_array_paths = [ThreePath::empty(); 35];
//             for (i, path) in all_paths.iter().enumerate() {
//                 all_array_paths[i] = *path;

//             }

//             let mut array_idxs: [u16; 35] = [0; 35];
//             for ( i, p) in all_paths.iter().enumerate() {
//                 let tuple_path = (p.path, p.backtrack.0);
//                 // println!("{:?}", tuple_path);
//                 let path_idx = UNIQUE_THREE_PATHS.iter().position(|p| p == &tuple_path).unwrap();
//                 array_idxs[i] = path_idx as u16;

//             }

//             let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
//             let path_list: [u16; 36] = whole.try_into().unwrap();
            
//             let path_list_idx = UNIQUE_THREE_PATH_LISTS.iter().position(|p| p == &path_list).unwrap();

//             let idx = intercept_bb.0 % 11007;
//             pos_data[idx as usize] = path_list_idx as u16;

//             // Undo intercepts
//             for pos in intercepts.clone() {
//                 board.data[*pos] = 0;
    
//             }
    
//         }

//         write!(f, "[")?;
//         for paths in pos_data {
//             write!(f, "{},", paths)?;

//         }

//         writeln!(f, "],")?;

//     }

//     Ok(())

// }
