#[macro_use]
mod macros;

mod bit_twiddles;
mod bitboard;
mod board;
mod evaluation;
mod move_gen;
mod transposition_table;

mod engine;
mod zobrist;

use crate::board::*;
use crate::engine::*;
use crate::transposition_table::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

fn main() {
    let mut board = BoardState::new();
    board.set(
        [3, 2, 1, 1, 2, 3],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [3, 2, 1, 1, 2, 3],
        [0, 0],

    );
    
    let search_input = SearchInput::new(board, 5);

    let (board_sender, board_reciver): (Sender<SearchInput>, Receiver<SearchInput>) = mpsc::channel();
    let (stop_sender, stop_reciver): (Sender<bool>, Receiver<bool>) = mpsc::channel();
    let (results_sender, results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();
    thread::spawn(move || {
        let mut engine = Engine::new(board_reciver, stop_reciver, results_sender);
        engine.start();

    });

    _ = board_sender.send(search_input);

    loop {
        let results = results_reciver.try_recv();
        match results {
            Ok(_) => {
                let final_results = results.unwrap();

                println!("DEPTH: {:?}", final_results.depth);
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
                println!("      - LOOKUP COLLISIONS: {}", unsafe { TT_LOOKUP_COLLISIONS });
                println!("      - EMPTY INSERTS: {}", unsafe { TT_EMPTY_INSERTS });
                println!("      - SAFE INSERTS: {}", unsafe { TT_SAFE_INSERTS });
                println!("      - UNSAFE INSERTS: {}", unsafe { TT_UNSAFE_INSERTS });
                println!("");
                println!("  - PV:");
                for (i, mv) in final_results.pv.iter().enumerate() {
                    println!("      - {}: {:?}", i, mv);

                }
                println!("");

            }
            Err(TryRecvError::Disconnected) => {}
            Err(TryRecvError::Empty) => {}

        }

    }

}
