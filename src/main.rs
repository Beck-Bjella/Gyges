#[macro_use]
mod macros;

mod board;
mod bitboard;
mod bit_twiddles;
mod move_gen;
mod evaluation;
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
    board.set(  [3, 2, 1, 1, 2, 3], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], 
                [3, 2, 1, 1, 2, 3], 
                [0, 0]);


    board.print();

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
        // _ = stop_sender.send(true);
        
        let results = results_reciver.try_recv();
        match results {
            Ok(_) => {
                let final_results = results.unwrap();

                println!("Depth: {:?}", final_results.depth);
                println!("  - Move: {:?}", final_results.best_move);
                println!("  - Time: {:?}", final_results.search_time);
                println!("  - Branchs: {:?}", final_results.branches);
                println!("  - Leafs: {:?}", final_results.leafs);
                println!("");
                println!("  - TT Hits: {:?}", final_results.tt_hits);
                println!("  - TT Exacts: {:?}", final_results.tt_exacts);
                println!("  - TT Cuts: {:?}", final_results.tt_cuts);
                println!("    TT Replacements: {}", unsafe{REPLACEMENTS});
                println!("    TT Collisions: {}", unsafe{COLLISIONS});
                println!("");
                println!("");
                println!("");
                println!("");
                println!("");
                
            },
            Err(TryRecvError::Disconnected) => {},
            Err(TryRecvError::Empty) => {}

        }

    }

}
