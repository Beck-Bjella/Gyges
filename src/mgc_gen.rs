use crate::board::board::*;
use crate::board::bitboard::*;
use crate::consts::*;
use crate::moves::move_gen::*;

use std::fs::File;
use std::path::Path;
use std::io::{Write, BufWriter, Error};
use std::vec;

use itertools::Itertools;

pub fn generate_permutations<T: Clone>(list: &[T]) -> Vec<Vec<&T>> {
    let mut all_permutations = Vec::new();
    for r in 0..=list.len() {
        for combination in list.iter().combinations(r) {
            all_permutations.push(combination.clone());

        }

    }

    return all_permutations;

}

// Contains a path and its backtrack bb
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreePath {
    pub path: [u8; 4],
    pub backtrack: BitBoard

}

impl ThreePath {
    pub fn empty() -> ThreePath {
        return ThreePath {
            path: [NULL_U8; 4],
            backtrack: BitBoard(0)

        }

    }

}

// Contains all of the paths for a specific intercept bb
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreePaths {
    pub paths: [ThreePath; 35],
    pub len: u8,

}

impl ThreePaths {
    pub fn empty() -> ThreePaths {
        return ThreePaths {
            paths: [ThreePath::empty(); 35],
            len: 0

        }

    }

}


pub fn gen_unique_three_paths() -> Result<(), Error> {
    let path = Path::new("./unique_three_paths.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut unique_three_paths = vec![];

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..THREE_PATH_LENGTHS[pos] {
            let path = THREE_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
            all_intercepts_bb.set_bit(path[2]);
    
        }
    
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut all_paths = vec![];
    
            // Check all paths and find vaild ones
            for path_idx in 0..THREE_PATH_LENGTHS[pos] {
                let path = THREE_PATHS[pos][path_idx];
            
                if board.data[path[1]] != 0 {
                    continue;
                    
                } else if board.data[path[2]] != 0 {
                    continue;
                    
                }
        
                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];

                let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];

                let mut goal: bool = false;
                if path[3] == PLAYER_2_GOAL {
                    goal = true;
        
                }
                
                let end_bb: u64 = 1 << path[3];
              
                let final_path = (small_path, backtrack_bb);

                all_paths.push(final_path.clone());

                if !unique_three_paths.contains(&final_path) {
                    unique_three_paths.push(final_path);
    
                }
        
            }

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }

    }

    writeln!(f, "([100, 100, 100, 100], 0)")?;
    for path in unique_three_paths {
        writeln!(f, "({:?}, {}),", path.0, path.1.0)?;

    }
    
    Ok(())

}

pub fn gen_unique_two_paths() -> Result<(), Error> {
    let path = Path::new("./unique_two_paths.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut unique_two_paths = vec![];

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..THREE_PATH_LENGTHS[pos] {
            let path = THREE_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
            all_intercepts_bb.set_bit(path[2]);
    
        }
    
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut all_paths = vec![];
    
            // Check all paths and find vaild ones
            for path_idx in 0..TWO_PATH_LENGTHS[pos] {
                let path = TWO_PATHS[pos][path_idx];
            
                if board.data[path[1]] != 0 {
                    continue;
                    
                }

                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8];

                let backtrack_bb = TWO_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let final_path = (small_path, backtrack_bb);

                all_paths.push(final_path.clone());

                if !unique_two_paths.contains(&final_path) {
                    unique_two_paths.push(final_path);
    
                }
        
            }

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }

    }

    writeln!(f, "([100, 100, 100], 0)")?;
    for path in unique_two_paths {
        writeln!(f, "({:?}, {}),", path.0, path.1.0)?;

    }

    Ok(())

}

pub fn gen_unique_one_paths() -> Result<(), Error> {
    let path = Path::new("./unique_one_paths.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut unique_one_paths = vec![];

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..THREE_PATH_LENGTHS[pos] {
            let path = THREE_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
            all_intercepts_bb.set_bit(path[2]);
    
        }
    
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut all_paths = vec![];
    
            // Check all paths and find vaild ones
            for path_idx in 0..ONE_PATH_LEGNTHS[pos] {
                let path = ONE_PATHS[pos][path_idx];

                let small_path = [path[0] as u8, path[1] as u8];

                let backtrack_bb = ONE_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let final_path = (small_path, backtrack_bb);

                all_paths.push(final_path.clone());

                if !unique_one_paths.contains(&final_path) {
                    unique_one_paths.push(final_path);
    
                }
        
            }

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }

    }

    writeln!(f, "([100, 100], 0)")?;
    for path in unique_one_paths {
        writeln!(f, "({:?}, {}),", path.0, path.1.0)?;

    }

    Ok(())

}



pub fn gen_unique_three_path_lists() -> Result<(), Error> {
    let path = Path::new("./unique_three_path_lists.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut unique_three_path_lists = vec![];

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..THREE_PATH_LENGTHS[pos] {
            let path = THREE_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
            all_intercepts_bb.set_bit(path[2]);
    
        }
    
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut intercept_bb = BitBoard(0);
    
            // Place intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 1;
                intercept_bb.set_bit(*pos);
    
            }
    
            let mut all_paths = vec![];

            // Check all paths and find vaild ones
            for path_idx in 0..THREE_PATH_LENGTHS[pos] {
                let path = THREE_PATHS[pos][path_idx];
            
                if board.data[path[1]] != 0 {
                    continue;
                    
                } else if board.data[path[2]] != 0 {
                    continue;
                    
                }

                let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];
                let final_path = ThreePath {
                    path: small_path,
                    backtrack: backtrack_bb
        
                };

                all_paths.push(final_path.clone());

            }

            let mut all_array_paths = [ThreePath::empty(); 35];
            for (i, path) in all_paths.iter().enumerate() {
                all_array_paths[i] = *path;

            }

            let mut array_idxs: [u16; 35] = [0; 35];
            for ( i, p) in all_paths.iter().enumerate() {
                let tuple_path = (p.path, p.backtrack.0);
                let path_idx = UNIQUE_THREE_PATHS.iter().position(|p| p == &tuple_path).unwrap();
                array_idxs[i] = path_idx as u16;

            }

            let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
            let three_path_array: [u16; 36] = whole.try_into().unwrap();

            if !unique_three_path_lists.contains(&three_path_array) {
                unique_three_path_lists.push(three_path_array);

            }

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }
        
    }
    
    for path in unique_three_path_lists {
        writeln!(f, "{:?},", path)?;

    }

    Ok(())

}

pub fn gen_unique_two_path_lists() -> Result<(), Error> {
    let path = Path::new("./unique_two_path_lists.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut unique_two_path_lists = vec![];

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..TWO_PATH_LENGTHS[pos] {
            let path = TWO_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
            all_intercepts_bb.set_bit(path[2]);
    
        }
    
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut intercept_bb = BitBoard(0);
    
            // Place intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 1;
                intercept_bb.set_bit(*pos);
    
            }
    
            let mut all_paths = vec![];

            // Check all paths and find vaild ones
            for path_idx in 0..TWO_PATH_LENGTHS[pos] {
                let path = TWO_PATHS[pos][path_idx];
            
                if board.data[path[1]] != 0 {
                    continue;
                    
                }

                let backtrack_bb = TWO_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8];
                let final_path = (small_path, backtrack_bb.0);

                all_paths.push(final_path.clone());

            }

            let mut array_idxs: [u16; 12] = [0; 12];
            for ( i, p) in all_paths.iter().enumerate() {
                let path_idx = UNIQUE_TWO_PATHS.iter().position(|p1| p1 == p).unwrap();
                array_idxs[i] = path_idx as u16;

            }

            let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
            let two_path_array: [u16; 13] = whole.try_into().unwrap();

            if !unique_two_path_lists.contains(&two_path_array) {
                unique_two_path_lists.push(two_path_array);

            }

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }
        
    }
    
    for path in unique_two_path_lists {
        writeln!(f, "{:?},", path)?;

    }

    Ok(())

}

pub fn gen_unique_one_path_lists() -> Result<(), Error> {
    let path = Path::new("./unique_one_path_lists.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut unique_one_path_lists = vec![];

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut intercepts_bb = BitBoard(0);
    
        let mut all_paths = vec![];

        // Check all paths and find vaild ones
        for path_idx in 0..ONE_PATH_LEGNTHS[pos] {
            let path = ONE_PATHS[pos][path_idx];
        
            let backtrack_bb = ONE_PATH_BACKTRACK_CHECKS[pos][path_idx];
            
            let small_path = [path[0] as u8, path[1] as u8];
            let final_path = (small_path, backtrack_bb.0);

            all_paths.push(final_path.clone());

        }

        let mut array_idxs: [u16; 4] = [0; 4];
        for ( i, p) in all_paths.iter().enumerate() {
            let path_idx = UNIQUE_ONE_PATHS.iter().position(|p1| p1 == p).unwrap();
            array_idxs[i] = path_idx as u16;

        }

        let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
        let one_path_array: [u16; 5] = whole.try_into().unwrap();

        if !unique_one_path_lists.contains(&one_path_array) {
            unique_one_path_lists.push(one_path_array);

        }

    }
    
    for path in unique_one_path_lists {
        writeln!(f, "{:?},", path)?;

    }

    Ok(())

}







pub fn gen_three_map() -> Result<(), Error> {
    let path = Path::new("./three_map.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );
    
    for pos in 0..36 {
        let mut pos_data: Vec<u16> = vec![0; THREE_MAP_LEN];

        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..THREE_PATH_LENGTHS[pos] {
            let path = THREE_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
            all_intercepts_bb.set_bit(path[2]);
    
        }
    
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut intercept_bb = BitBoard(0);
    
            // Place intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 1;
                intercept_bb.set_bit(*pos);
    
            }
    
            let mut all_paths = vec![];
    
            // Check all paths and find vaild ones
            for path_idx in 0..THREE_PATH_LENGTHS[pos] {
                let path = THREE_PATHS[pos][path_idx];
            
                if board.data[path[1]] != 0 {
                    continue;
                    
                } else if board.data[path[2]] != 0 {
                    continue;
                    
                }
    
                let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];
                let final_path = ThreePath {
                    path: small_path,
                    backtrack: backtrack_bb
        
                };
    
                all_paths.push(final_path.clone());
        
            }

            let mut all_array_paths = [ThreePath::empty(); 35];
            for (i, path) in all_paths.iter().enumerate() {
                all_array_paths[i] = *path;

            }

            let mut array_idxs: [u16; 35] = [0; 35];
            for ( i, p) in all_paths.iter().enumerate() {
                let tuple_path = (p.path, p.backtrack.0);
                // println!("{:?}", tuple_path);
                let path_idx = UNIQUE_THREE_PATHS.iter().position(|p| p == &tuple_path).unwrap();
                array_idxs[i] = path_idx as u16;

            }

            let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
            let path_list: [u16; 36] = whole.try_into().unwrap();
            
            let path_list_idx = UNIQUE_THREE_PATH_LISTS.iter().position(|p| p == &path_list).unwrap();

            let idx = intercept_bb.0 % THREE_MAP_LEN as u64;
            pos_data[idx as usize] = path_list_idx as u16;

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }

        write!(f, "[")?;
        for paths in pos_data {
            write!(f, "{},", paths)?;

        }

        writeln!(f, "],")?;

    }

    Ok(())

}

pub fn gen_two_map() -> Result<(), Error> {
    let path = Path::new("./two_map.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );

    // for map_len in 1..100000 {
    let mut collisions = 0;

    for pos in 0..36 {
        // let mut used_idxs = vec![];
        let mut pos_data: Vec<u16> = vec![0; TWO_MAP_LEN];

        let mut all_intercepts_bb = BitBoard(ALL_TWO_INTERCEPTS[pos]);
        let intercept_positions = all_intercepts_bb.clone().get_data();
    
        let intercept_combos = generate_permutations(&intercept_positions);

        for intercepts in intercept_combos {
            let mut intercept_bb = BitBoard(0);
    
            // Place intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 1;
                intercept_bb.set_bit(*pos);
    
            }
    
            let mut all_paths = vec![];

            // Check all paths and find vaild ones
            for path_idx in 0..TWO_PATH_LENGTHS[pos] {
                let path = TWO_PATHS[pos][path_idx];
            
                if board.data[path[1]] != 0 {
                    continue;
                    
                }

                let backtrack_bb = TWO_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8];
                let final_path = (small_path, backtrack_bb.0);

                all_paths.push(final_path.clone());

            }

            let mut array_idxs: [u16; 12] = [0; 12];
            for ( i, p) in all_paths.iter().enumerate() {
                let path_idx = UNIQUE_TWO_PATHS.iter().position(|p1| p1 == p).unwrap();
                array_idxs[i] = path_idx as u16;

            }

            let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
            let path_list: [u16; 13] = whole.try_into().unwrap();

            
            let path_list_idx = UNIQUE_TWO_PATH_LISTS.iter().position(|p| p == &path_list).unwrap();

            let idx = intercept_bb.0 % TWO_MAP_LEN as u64;
            
            // if used_idxs.contains(&idx) {
            //     collisions += 1;

            // } 
            // used_idxs.push(idx);
            
            

            pos_data[idx as usize] = path_list_idx as u16;

            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }

        write!(f, "[")?;
        for paths in pos_data {
            write!(f, "{},", paths)?;

        }

        writeln!(f, "],")?;

    }

    // println!("{}: {}", map_len, collisions);
    // if collisions == 0 {
    //     break;

    // }

    // }
    
    Ok(())

}

pub fn gen_one_map() -> Result<(), Error> {
    let path = Path::new("./one_map.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());

    let mut board = BoardState::new();
    board.set(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0],
        PLAYER_1,

    );

    for pos in 0..36 {
        let mut pos_data: Vec<u16> = vec![0; ONE_MAP_LEN];

        let mut intercept_bb = BitBoard(0);
    
        let mut all_paths = vec![];

        // Check all paths and find vaild ones
        for path_idx in 0..ONE_PATH_LEGNTHS[pos] {
            let path = ONE_PATHS[pos][path_idx];
        
            let backtrack_bb = ONE_PATH_BACKTRACK_CHECKS[pos][path_idx];
            
            let small_path = [path[0] as u8, path[1] as u8];
            let final_path = (small_path, backtrack_bb.0);

            all_paths.push(final_path.clone());

        }

        let mut array_idxs: [u16; 4] = [0; 4];
        for ( i, p) in all_paths.iter().enumerate() {
            let path_idx = UNIQUE_ONE_PATHS.iter().position(|p1| p1 == p).unwrap();
            array_idxs[i] = path_idx as u16;

        }

        let whole: Vec<u16> = array_idxs.iter().copied().chain([all_paths.len() as u16].iter().copied()).collect();
        let path_list: [u16; 5] = whole.try_into().unwrap();

        let path_list_idx = UNIQUE_ONE_PATH_LISTS.iter().position(|p| p == &path_list).unwrap();

        let idx = intercept_bb.0 % ONE_MAP_LEN as u64;
        

        pos_data[idx as usize] = path_list_idx as u16;


        write!(f, "[")?;
        for paths in pos_data {
            write!(f, "{},", paths)?;

        }

        writeln!(f, "],")?;

    }
    
    Ok(())

}

pub fn gen_two_all_intercepts() -> Result<(), Error> {
    let path = Path::new("./two_all_intercepts.txt");
    let mut f = BufWriter::new(File::create(&path).unwrap());
    
    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..TWO_PATH_LENGTHS[pos] {
            let path = TWO_PATHS[pos][path_idx];
    
            all_intercepts_bb.set_bit(path[1]);
    
        }

        writeln!(f, "{:#064b},", all_intercepts_bb.0)?;

    }

    Ok(())
   
}
