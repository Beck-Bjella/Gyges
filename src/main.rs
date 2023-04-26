#[macro_use]
mod macros;

mod bit_twiddles;
mod bitboard;
mod board;
mod consts;
mod engine;
mod evaluation;
mod move_gen;
mod move_list;
mod moves;
mod tt;
mod zobrist;

use move_gen::valid_moves;

use crate::board::*;
use crate::consts::*;
use crate::engine::*;
use crate::tt::*;
use crate::evaluation::*;

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

fn main() {
    init_tt();

    println!("TT STATS: ");
    println!("GB: {}", tt().size_gigabytes());
    println!("Clusters: {}", tt().num_clusters());
    println!("Entries: {}", tt().num_entrys());
    println!("");

    let (results_sender, results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();

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

    thread::spawn(move || {
        let mut searcher = Searcher::new(results_sender);
        searcher.iterative_deepening_search(&mut board, 99);
        
    });

    loop {
        let results = results_reciver.try_recv();
        match results {
            Ok(_) => {
                let final_results = results.unwrap();

                println!("Depth: {:?}", final_results.ply);
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
                println!("      - SAFE INSERTS: {}", unsafe { TT_SAFE_INSERTS });
                println!("      - UNSAFE INSERTS: {}", unsafe { TT_UNSAFE_INSERTS });
                println!("");
                println!("  - PV");
                for (i, e) in final_results.pv.iter().enumerate() {
                    println!("      - {}: {:?}", i, e.bestmove);

                }
                println!("");
            }
            Err(TryRecvError::Disconnected) => {}
            Err(TryRecvError::Empty) => {}

        }

    }

}
