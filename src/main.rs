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
    board.set_rank([1, 3, 2 ,1, 2, 3], 5);
    board.set_rank([0 ,0 ,0 ,0, 0, 0], 4);
    board.set_rank([0 ,0 ,0 ,0, 0, 0], 3);
    board.set_rank([0 ,0 ,0 ,0, 0, 0], 2);
    board.set_rank([0 ,0, 0, 0, 0, 0], 1);
    board.set_rank([2 ,3 ,1 ,3, 1, 2], 0);
    board.set_goals([0, 0]);

    let mut engine = Engine::new();

    let results = engine.iterative_deepening_search(&mut board, 4, 10000.0);
        
    println!("");
    println!("==================== FINAL DATA ====================");
    println!("");
    for result in results {
        println!("{:?}", result);

    }
    println!("");
    println!("Evaluated {} nodes in {} seconds at a rate of {} NPS.", engine.search_stats.nodes_evaluated, engine.search_stats.search_time, engine.search_stats.nps);
    println!("   - {} minimax hits", engine.search_stats.minimax_hits);
    println!("       - {} exact hits", engine.search_stats.exact_hits);
    println!("       - {} tt cuts", engine.search_stats.tt_cuts);
    println!("");
    println!("====================================================");

}
