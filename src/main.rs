mod board;
mod helper;
mod moves;
mod search;
mod tools;
mod consts;

use crate::board::bitboard::*;
use crate::moves::moves::*;
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

    println!("{}", board);
    println!("{}", board.peice_board);

    let new_board = board.make_move(&Move::new([0, 0, 2, 2, 2, 0], MoveType::Drop));

    println!("{}", new_board);
    println!("{}", new_board.peice_board);

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
        let mut searcher: Searcher = Searcher::new(results_sender);
        searcher.iterative_deepening_search(&mut board, 5);
        
    });

    loop {
        let results = results_reciver.try_recv();
        match results {
            Ok(_) => {
                let final_results = results.unwrap();
                println!("{}", final_results);
                
            }
            Err(TryRecvError::Disconnected) => {}
            Err(TryRecvError::Empty) => {}

        }

    }

}
