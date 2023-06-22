#![feature(test)]

mod board;
mod helper;
mod moves;
mod search;
mod tools;
mod consts;
mod mgc_gen;

use crate::board::board::*;
use crate::consts::*;
use crate::search::searcher::*;
use crate::tools::tt::*;

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;



fn main() {
    init_tt();

    println!("TT STATS: ");
    println!("GB: {}", tt().size_gigabytes());
    println!("Clusters: {}", tt().num_clusters());
    println!("Entries: {}", tt().num_entrys());
    println!("");

    let (rs, rr): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();

    // let mut board = BoardState::from(TEST_BOARD, PLAYER_1);

    let mut board = BoardState::new();
    board.set(
        [3, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0],
        [0, 3, 0, 3, 0, 0],
        [0, 2, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0], 
        [0, 0, 2, 0, 3, 0],
        [0, 0],
        PLAYER_1

    );

    println!("{board}");

    // SINGLE THREADED
    let mut searcher: Searcher = Searcher::new(rs);
    searcher.iterative_deepening_search(&mut board, 99);

    // MULTI THREADED
    // thread::spawn(move || {
    //     let mut searcher: Searcher = Searcher::new(rs);
    //     searcher.iterative_deepening_search(&mut board, 5);
        
    // });

    // loop {
    //     let results = rr.try_recv();
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
