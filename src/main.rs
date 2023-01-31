#[macro_use]
mod macros;
mod board;
mod bitboard;
mod bit_twiddles;
mod move_gen;
mod bitmove;

use crate::board::*;
use crate::move_gen::*;

use std::time::{Duration, Instant};

    
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let mut board = BoardState::new();

    board.set(  [3, 2, 1, 1, 2, 3], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 3, 0], 
                [0, 0, 3, 0, 0, 0], 
                [3, 2, 1, 1, 2, 3], 
                [0, 0]);

    benchmark_movegen(&mut board);

}

fn benchmark_movegen(board: &mut BoardState) {
    let mut average_time: Duration = Default::default();
    for i in 0..100_000 {
        let now = Instant::now();
    
        valid_moves(board, 1.0);

        let elapsed = now.elapsed();

        average_time = (average_time + elapsed) / 2;

    }
    println!("{:?} / iter", average_time);
    println!("move_count: {}", valid_moves(board, 1.0).len());

}
