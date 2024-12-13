extern crate gyges_engine;

use gyges_engine::ugi::Ugi;

pub fn main() {
    Ugi::new().start();

}

// use gyges::{
//     board::{BENCH_BOARD, STARTING_BOARD, TEST_BOARD},
//     moves::movegen::{has_threat, valid_moves},
//     BoardState, 
//     Player, 

// };

// use gyges_engine::new_movegen::*;

// fn main() {
//     let mut board1 = BoardState::from(TEST_BOARD); 
//     let mut board2 = BoardState::from(STARTING_BOARD);
//     let mut board3 = BoardState::from(BENCH_BOARD);
//     let mut board4 = BoardState::from([ // SPARCE CASE
//         0, 0, 0, 0, 2, 3,
//         0, 0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0, 1,
//         0, 0
//     ]);
//     let mut board5 = BoardState::from([ // REAL CASE
//         0, 3, 0, 0, 1, 3,
//         0, 0, 0, 0, 0, 2,
//         2, 0, 1, 0, 0, 0,
//         3, 0, 0, 0, 0, 0,
//         2, 0, 0, 0, 0, 0,
//         1, 0, 0, 1, 2, 3,
//         0, 0
//     ]);
//     let player = Player::One;

//     let mut mg = MoveGen::new();

//     // NEW
//     for i in 0..1 {
//         let mut boards: Vec<BoardState> = vec![];
//         for _ in 0..2000 {
//             boards.push(board1.clone());
//             boards.push(board2.clone());
//             boards.push(board3.clone());
//             boards.push(board4.clone());
//             boards.push(board5.clone());

//         }

//         let mut results = mg.batch(&boards, player, 0);

//         let mut count: usize = 0;
//         for i in 0..10000 {
//             count += results[i].moves(&boards[i], player).len();

//         }

//         println!("{}: New: {}", i, (count / 2000) as f64);
    
//         if (count / 2000) != 1662 {
//             break;

//         }

//     }
    
//     // NATIVE 
//     let mut moves1 = unsafe { valid_moves(&mut board1, player) }; 
//     let mut moves2 = unsafe { valid_moves(&mut board2, player) };
//     let mut moves3 = unsafe { valid_moves(&mut board3, player) };
//     let mut moves4 = unsafe { valid_moves(&mut board4, player) };
//     let mut moves5 = unsafe { valid_moves(&mut board5, player) };
//     let real1 = moves1.moves(&board1);
//     let real2 = moves2.moves(&board2);
//     let real3 = moves3.moves(&board3);
//     let real4 = moves4.moves(&board4);
//     let real5 = moves5.moves(&board5);
//     println!("Native: {}", (real1.len() + real2.len() + real3.len() + real4.len() + real5.len()) as f64);

//     // BENCHMARKS
//     unsafe {
//         let iters: i32 = 5000;

//         println!("NEW BENCHMARKS");
//         let genbatch_size = 1000;
//         let boards = vec![board1.clone(); genbatch_size];
//         for batch in 0..3 {
//             let mut num = 0;
//             let start: std::time::Instant = std::time::Instant::now();
//             for _ in 0..iters {
//                 let _moves: Vec<GenResult> = mg.batch(&boards, player, 0);
//                 num += 1;
               
//             }
            
//             let elapsed = start.elapsed().as_secs_f64();
//             let iter_time = elapsed / iters as f64;
//             let gens_per_sec = (1.0 / iter_time) * genbatch_size as f64;
//             println!("{}: {} g/s : {} s/iter", batch, gens_per_sec as usize, iter_time);

//         }

//         println!("NATIVE BENCHMARKS");
//         for batch in 0..3 {
//             let mut num = 0;
//             let start: std::time::Instant = std::time::Instant::now();
//             for _ in 0..iters {
//                 let _moves = valid_moves(&mut board1, player);
//                 num += 1;
               
//             }
            
//             let elapsed = start.elapsed().as_secs_f64();
//             let iter_time = elapsed / iters as f64;
//             let gens_per_sec = 1.0 / iter_time;
//             println!("{}: {} g/s : {} s/iter", batch, gens_per_sec as usize, iter_time);

//         }

//     }

// }
