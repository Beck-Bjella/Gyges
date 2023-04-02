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

mod tt;
mod consts;

use crate::transposition_table::*;
use crate::board::*;
use crate::engine::*;
use crate::evaluation::*;
use crate::zobrist::*;
use crate::move_gen::*;

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

use rand::Rng;

fn main() {
    // let mut rng = rand::thread_rng();

    // let mut keys = vec![];
    // let mut threads = vec![];

    // let tt = TranspositionTable::new_from_mb(100);

    // for i in 0..2 {
    //     let mut tt_clone = tt.clone();
    //     let key: u64 = rng.gen();

    //     let t = thread::spawn(move || {
    //         tt_clone.insert(key, TTEntry { key: key, value: 1.0, flag: TTEntryType::ExactValue, depth: 1, empty: false });
    //         let data = tt_clone.probe(key);

    //         println!("{:?}", data);
    //     });

    //     keys.push(key);
    //     threads.push(t);

    // }

    // for t in threads {
    //     let _ = t.join().unwrap();

    // }

    // println!("{:?}", keys);
    // println!("LOOKUP COLLISIONS: {}", unsafe { TT_LOOKUP_COLLISIONS });
    // println!("EMPTY INSERTS: {}", unsafe { TT_EMPTY_INSERTS });
    // println!("SAFE INSERTS: {}", unsafe { TT_SAFE_INSERTS });
    // println!("UNSAFE INSERTS: {}", unsafe { TT_UNSAFE_INSERTS });

    // let data = tt.probe(keys[0]);

    // println!("{:?}", data);

    let (board_sender, board_reciver): (Sender<SearchInput>, Receiver<SearchInput>) = mpsc::channel();
    let (stop_sender, stop_reciver): (Sender<bool>, Receiver<bool>) = mpsc::channel();
    let (results_sender, results_reciver): (Sender<SearchData>, Receiver<SearchData>) = mpsc::channel();
    thread::spawn(move || {
        let mut engine = Engine::new(board_reciver, stop_reciver, results_sender);
        engine.start();

    });

    let mut board = BoardState::new();
    board.set(
        [3, 2, 1, 1, 2, 3],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [3, 2, 1, 1, 2, 3],
        [0, 0],
        PLAYER_1,

    );

    // let mv1 = Move::new([0, 2, 1, 4, 2, 27], MoveType::Drop, 0.0);
    // let mv2 = Move::new([0, 5, 3, 4, 2, 8], MoveType::Drop, 0.0);
    // let mv3 = Move::new([0, 1, 2, 6, NULL, NULL], MoveType::Bounce, 0.0);
    // let mv4 = Move::new([0, 0, 3, 1, 2, 0], MoveType::Bounce, 0.0);

    // println!("{}", board.hash);
    // println!("{}", get_hash(&mut board, PLAYER_1));

    // board.make_move(&mv4);

    // println!("{}", board.hash);
    // println!("{}", get_hash(&mut board, PLAYER_1));

    // board.undo_move(&mv4);

    // println!("{}", board.hash);
    // println!("{}", get_hash(&mut board, PLAYER_1));



    // let boards = vec![board];
    // let board_count = boards.len();

    // let results = simulate_games(boards, board_sender, stop_sender, results_reciver);
    // println!("{:?}", results);

    // let mut p1_wins = 0;
    // let mut p2_wins = 0;
    // for data in results {
    //     if data.1 == PLAYER_1 {
    //         p1_wins += 1;

    //     } else if data.1 == PLAYER_2 {
    //         p2_wins += 1;

    //     }
        
    // }
    // let win_rates = (p1_wins / board_count, p2_wins / board_count);
    // println!("WIN PERCENTS: ");
    // println!("  - P1: {}", win_rates.0);
    // println!("  - P2: {}", win_rates.1);



    let search_input = SearchInput::new(board, 99, EvalType::Standard);
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

            }
            Err(TryRecvError::Disconnected) => {}
            Err(TryRecvError::Empty) => {}

        }

    }

}


// fn simulate_games(boards: Vec<BoardState>, board_sender: Sender<SearchInput>, stop_sender: Sender<bool>, results_reciver: Receiver<SearchData>) -> Vec<(i32, f64)> {
//     let max_ply = 3;

//     let mut outcomes = vec![];

//     for mut board in boards {
//         let mut current_player = PLAYER_1;
//         let mut game_depth = 1;
    
//         loop {
//             println!("========================");
            
//             let mut best_move = Move::new_null();
//             _ = stop_sender.send(true);

//             let eval_type: EvalType;
//             if current_player == 1.0 {
//                 eval_type = EvalType::Two;
//             } else {
//                 eval_type = EvalType::One;
//             }
//             _ = board_sender.send(SearchInput{board, max_ply, eval_type});

//             loop {
//                 let results = results_reciver.try_recv();
//                 match results {
//                     Ok(_) => {
//                         let final_results = results.unwrap();
//                         if final_results.depth >= max_ply {
//                             best_move = final_results.best_move;
//                             break;
    
//                         }
    
//                     }
//                     Err(TryRecvError::Disconnected) => {}
//                     Err(TryRecvError::Empty) => {}
    
//                 }
    
//             }
            
//             println!("{:?}: {}", best_move, current_player);
//             board.print();
    
//             if !best_move.is_null() {
//                 if best_move.score == f64::INFINITY {
//                     outcomes.push((game_depth, current_player));
//                     break;
    
//                 } else if best_move.score == f64::NEG_INFINITY {
//                     outcomes.push((game_depth, -current_player));
//                     break;
    
//                 }
    
//                 board.make_move(&best_move);
    
//             } else {
//                 panic!("TRIED TO MAKE A NULL MOVE!");
    
//             }
    
//             board.flip();
//             current_player *= -1.0;
//             game_depth += 1;
    
//         }

//     }

//     return outcomes;

// }
