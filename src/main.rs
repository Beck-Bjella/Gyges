#[macro_use]
mod macros;

mod board;
mod bitboard;
mod bit_twiddles;
mod move_generation;
mod evaluation;
mod engine;

use crate::board::*;
use crate::engine::*;
use crate::move_generation::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;


fn main() {
    let mut board = BoardState::new();
    board.set(  [3, 2, 1, 1, 2, 3], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 3, 0], 
                [0, 0, 3, 0, 0, 0], 
                [3, 2, 1, 1, 2, 3], 
                [0, 0]);

    let search_input = SearchInput::new(board, MAX_SEARCH_PLY);

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
                println!("  - {:?}", final_results.best_move);
                println!("  - {:?}", final_results.search_time);
                println!("");
                
            },
            Err(TryRecvError::Disconnected) => {},
            Err(TryRecvError::Empty) => {}

        }

    }

}
