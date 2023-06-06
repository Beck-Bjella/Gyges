mod board;
mod helper;
mod moves;
mod search;
mod tools;
mod consts;

use itertools::Itertools;

use crate::board::bitboard::*;
use crate::board::board::*;
use crate::consts::*;
use crate::search::searcher::*;
use crate::tools::tt::*;

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

use std::fs::OpenOptions;
use std::io::{Error, Write};

fn generate_permutations<T: Clone>(list: &[T]) -> Vec<Vec<&T>> {
    let mut all_permutations = Vec::new();
    for r in 0..=list.len() {
        for combination in list.iter().combinations(r) {
            all_permutations.push(combination.clone());

        }

    }

    return all_permutations;

}

#[derive(Debug, Clone, Copy)]
pub struct ThreePath {
    path: [u8; 4],
    backtrack: BitBoard

}

// Contains a path and its backtrack bb
impl ThreePath {
    pub fn empty() -> ThreePath {
        return ThreePath {
            path: [NULL_U8; 4],
            backtrack: BitBoard(0)

        }

    }

}

// Contains all of the paths for a specific intercept bb
#[derive(Debug, Clone)]
pub struct ThreePaths {
    paths: [ThreePath; 40],
    len: u8,

}

fn main() -> Result<(), Error> {

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

    let file_name = "output.txt";
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .append(false)
        .open(file_name)?;

    for pos in 0..36 {
        let mut all_intercepts_bb = BitBoard(0);

        // Generate all intercepts bitboard
        for path_idx in 0..THREE_PATH_LENGTHS[ pos] {
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
        
                if path[3] == PLAYER_1_GOAL {
                    continue;
        
                }
        
                let backtrack_bb = THREE_PATH_BACKTRACK_CHECKS[pos][path_idx];
                
                let small_path = [path[0] as u8, path[1] as u8, path[2] as u8, path[3] as u8];
                let final_path = ThreePath {
                    path: small_path,
                    backtrack: backtrack_bb
        
                };
    
                all_paths.push(final_path);
        
            }
        
            write!(f, "{}", format!("({}, ThreePaths {{ paths: [ ", intercept_bb.0) )?;
            let extras = 40 - all_paths.len();
            for path in all_paths.clone() {
                write!(f, "{}", format!("ThreePath {{ path: {:?}, backtrack: {:?} }}, ", path.path, path.backtrack) )?;

            }
            for i in 0..extras {
                write!(f, "ThreePath {{ path: [NULL_U8, NULL_U8, NULL_U8, NULL_U8], backtrack: BitBoard(0) }}, ")?;

            }
  
            writeln!(f, "{}", format!("], len: {}}} ),", all_paths.len()) )?;
    
            // Undo intercepts
            for pos in intercepts.clone() {
                board.data[*pos] = 0;
    
            }
    
        }

    }

    Ok(())  

    // let mut board2= BoardState::new();
    // board2.set(
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [3, 0, 0, 0, 0, 0],
    //     [0, 0],
    //     PLAYER_1,

    // );

    // let all_intercepts_bb = BitBoard(0b000000_000000_000000_000001_000011_000110);

    // let intercept_bb = board2.peice_board & all_intercepts_bb;

    // println!("{:?}", ALL_FINAL_PATHS.get(&intercept_bb.0));
    

    // init_tt();

    // println!("TT STATS: ");
    // println!("GB: {}", tt().size_gigabytes());
    // println!("Clusters: {}", tt().num_clusters());
    // println!("Entries: {}", tt().num_entrys());
    // println!("");

    // let (results_sender, results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();

    // let mut board = BoardState::new();

    // // TESTING BOARD
    // // board.set(
    // //     [0, 3, 0, 1, 2, 3],
    // //     [0, 0, 0, 1, 0, 0],
    // //     [0, 0, 0, 2, 0, 0],
    // //     [0, 0, 0, 0, 0, 0],
    // //     [0, 0, 0, 3, 3, 0],
    // //     [2, 0, 2, 1, 1, 0],
    // //     [0, 0],
    // //     PLAYER_1,

    // // );

    // // GAME
    // board.set(
    //     [3, 2, 1, 1, 2, 3],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [3, 2, 1, 1, 2, 3],
    //     [0, 0],
    //     PLAYER_1,
    // );

    // thread::spawn(move || {
    //     let mut searcher: Searcher = Searcher::new(results_sender);
    //     searcher.iterative_deepening_search(&mut board, 99);
        
    // });

    // loop {
    //     let results = results_reciver.try_recv();
    //     match results {
    //         Ok(_) => {
    //             let final_results = results.unwrap();
    //             println!("{}", final_results);
                
    //         }
    //         Err(TryRecvError::Disconnected) => {}
    //         Err(TryRecvError::Empty) => {}

    //     }

    // }

}
