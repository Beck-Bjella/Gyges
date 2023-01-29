#[macro_use]
mod macros;

mod board;
mod bitboard;
mod bit_twiddles;
mod move_generation;

use crate::board::*;
use crate::move_generation::*;

use std::time::Instant;
    

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    benchmark_movegen()

}

fn benchmark_movegen() {
    let mut board = BoardState::new();

    board.set(      [3, 2, 1, 1, 2, 3], 
                    [0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 3, 0], 
                    [0, 0, 3, 0, 0, 0], 
                    [3, 2, 1, 1, 2, 3], 
                    [0, 0]);


    let now = Instant::now();
    
    for _ in 0..1000 {
        valid_moves(&mut board, 1.0);

    }

    let elapsed = now.elapsed();

    println!("Elapsed: {:.3?}", elapsed);
    println!("move_count: {}", valid_moves(&mut board, 1.0).len());

}