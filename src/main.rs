#[macro_use]
mod macros;

mod board;
mod bitboard;
mod bit_twiddles;
mod move_generation;
mod evaluation;
mod engine;
mod transposition_tables;
mod zobrist;

use crate::board::*;
use crate::engine::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let mut board = BoardState::new();
    board.set_rank([0, 1, 0 ,0, 0, 0], 5);
    board.set_rank([0 ,0 ,0 ,0, 0, 0], 4);
    board.set_rank([0 ,0 ,0 ,0, 0, 0], 3);
    board.set_rank([0 ,3 ,0 ,0, 0, 0], 2);
    board.set_rank([0 ,0, 0, 0, 0, 0], 1);
    board.set_rank([3 ,2 ,1 ,2, 3, 1], 0);
    board.set_goals([0, 0]);

    let mut engine = Engine::new();

    let results = engine.iterative_deepening_search(&mut board, 7, 100000.0);
        
    println!("");
    println!("==================== FINAL DATA ====================");
    println!("");
    for result in results {
        println!("{:?}", result);

    }

    println!("");
    println!("Evaluated {} nodes in {} seconds at a rate of {} NPS.", engine.nodes_evaluated, engine.search_time, engine.nps);
    println!("  - {} eval duplicates", engine.eval_duplicates);
    println!("");
    println!("====================================================");

}