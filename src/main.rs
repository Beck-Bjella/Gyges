#[macro_use]
mod macros;

mod bit_twiddles;
mod bitboard;
mod board;
mod evaluation;
mod move_gen;
mod engine;
mod zobrist;

mod tt;
mod consts;

use crate::board::*;
use crate::engine::*;
use crate::evaluation::*;
use crate::zobrist::*;
use crate::move_gen::*;
use crate::consts::*;
use crate::tt::*;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use rand::Rng;
use rayon::vec;


fn main() {
    init_tt();

    unsafe {
        println!("TT STATS: ");
        println!("GB: {}", tt().size_gigabytes());
        println!("Clusters: {}", tt().num_clusters());
        println!("Entries: {}", tt().num_entrys());
        println!("");

    }

    let mut dataouts = vec![];
    for i in 0usize..1 {
        let (worker_stop_sender, worker_stop_reciver): (Sender<bool>, Receiver<bool>) = mpsc::channel();
        let (worker_results_sender, worker_results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();

        dataouts.push(worker_results_reciver);

        let mut board = BoardState::new();
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

        thread::spawn(move || {
            let mut worker = Worker::new(worker_results_sender, i);
            worker.iterative_deepening_search(&mut board, 99);

        });

    }
    
    loop {
        for reciver in dataouts.iter() {
            let results = reciver.try_recv();
            match results {
                Ok(_) => {
                    let final_results = results.unwrap();
                    
                    println!("ID: {:?}", final_results.search_id);
                    println!("  - Depth: {:?}", final_results.depth);
                    println!("  - Best: {:?}", final_results.best_move);
                    println!("  - Time: {:?}", final_results.search_time);
                    println!("");
                    println!("  - Abf: {}", final_results.average_branching_factor);
                    println!("");
                    println!("  - Branchs: {}", final_results.branches);
                    println!("  - Bps: {}", final_results.bps);
                    println!("");
                    println!("  - Leafs: {}", final_results.leafs);
                    println!("  - Lps: {}", final_results.lps);
                    println!("");
                    println!("  - TT:");
                    println!("      - HITS: {:?}", final_results.tt_hits);
                    println!("      - EXACTS: {:?}", final_results.tt_exacts);
                    println!("      - CUTS: {:?}", final_results.tt_cuts);
                    println!("");
                    println!("      - EMPTY INSERTS: {}", unsafe { TT_EMPTY_INSERTS });
                    println!("      - UNSAFE INSERTS: {}", unsafe { TT_UNSAFE_INSERTS });
                    println!("");

                }
                Err(TryRecvError::Disconnected) => {}
                Err(TryRecvError::Empty) => {}
    
            }

        }

    }

}
