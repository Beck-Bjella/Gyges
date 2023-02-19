#[macro_use]

mod macros;
mod board;
mod bitboard;
mod bit_twiddles;
mod move_gen;

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
    let iters = 100_000;

    let mut sum: Duration = Duration::from_secs(0);
    let mut lowest: Duration = Duration::from_secs(1);
    let mut highest: Duration = Duration::from_secs(0);

    for _ in 0..iters {
        let start = Instant::now();
        
        unsafe {valid_moves(board, PLAYER_1)};

        let elapsed = start.elapsed();

        sum += elapsed;

        if elapsed < lowest {
            lowest = elapsed

        }

        if elapsed > highest {
            highest = elapsed

        }

    }
    
    println!("+---------------------------------+");
    println!("");
    println!("highest: {:?} / iter", highest);
    println!("");
    println!("average: {:?} / iter" , sum/iters);
    println!("");
    println!("lowest: {:?} / iter", lowest);
    println!("");
    println!("+---------------------------------+");
    
}
