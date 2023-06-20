mod board;
mod helper;
mod moves;
mod search;
mod tools;
mod consts;
mod mgc_gen;

use rayon::prelude::*;

use crate::board::bitboard::*;
use crate::board::board::*;
use crate::consts::*;
use crate::search::searcher::*;
use crate::tools::tt::*;
use crate::moves::move_gen::*;

use std::path;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;


fn main() {
    // let mut board = BoardState::new();
    // board.set(
    //     [0, 3, 0, 1, 2, 3],
    //     [0, 0, 0, 1, 0, 0],
    //     [0, 0, 0, 2, 0, 0],
    //     [0, 0, 0, 0, 0, 0],
    //     [0, 0, 0, 3, 3, 0],
    //     [2, 0, 2, 1, 1, 0],
    //     [0, 0],
    //     PLAYER_1,

    // );
  

    // println!("{}", unsafe{ valid_moves(&mut board, PLAYER_1) }.moves(&board).len());
    // println!("{}", unsafe{ valid_moves_2(&mut board, PLAYER_1) }.moves(&board).len());

    // let start_time = std::time::Instant::now();
    // for i in 0..100_000_00usize {
    //     unsafe{ valid_moves(&mut board, PLAYER_1) };
       
    // }
    // println!("Nanos 1: {}", start_time.elapsed().as_micros() as f64 / 100_000_00.0);

    // let start_time = std::time::Instant::now();
    // for i in 0..100_000_00usize {
    //     unsafe{ valid_moves_2(&mut board, PLAYER_1) };
        
    // }
    // println!("Nanos 2: {}", start_time.elapsed().as_micros() as f64 / 100_000_00.0);

    init_tt();

    println!("TT STATS: ");
    println!("GB: {}", tt().size_gigabytes());
    println!("Clusters: {}", tt().num_clusters());
    println!("Entries: {}", tt().num_entrys());
    println!("");

    let (results_sender, results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();

    let mut board = BoardState::new();

    // TESTING BOARD
    board.set(
        [0, 3, 0, 1, 2, 3],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 0],
        [2, 0, 2, 1, 1, 0],
        [0, 0],
        PLAYER_1,

    );

    let mut searcher: Searcher = Searcher::new(results_sender);
    searcher.iterative_deepening_search(&mut board, 5);

    // thread::spawn(move || {
    //     let mut searcher: Searcher = Searcher::new(results_sender);
    //     searcher.iterative_deepening_search(&mut board, 5);
        
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
