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
    board.set_rank([3, 2, 1 ,1, 2, 3], 5);
    board.set_rank([0 ,0 ,0, 0, 0, 0], 4);
    board.set_rank([0 ,0 ,0, 0, 0, 0], 3);
    board.set_rank([0 ,0 ,0 ,0, 0, 0], 2);
    board.set_rank([0 ,0, 0, 0, 0, 0], 1);
    board.set_rank([3 ,2 ,1 ,2, 3, 1], 0);
    board.set_goals([0, 0]);

    let mut negamax = Negamax::new();
    let results = negamax.iterative_deepening_search(&mut board, 3);

    for result in results {
        println!("");
        println!("");
        println!("");
        println!("==================== FINAL DATA ====================");
        println!("");
        println!("Found the bestmove of {:?}", result.best_move);
        println!("    - In {} seconds", result.search_time);
        println!("    - Depth of {} ply", result.depth);
        println!("");
        println!("Searched {} total nodes with {} leaf nodes", result.nodes, result.leafs);
        println!("    - {} NPS", result.nps);
        println!("    - {} LPS", result.lps);
        println!("");
        println!("{} TT hits", result.tt_hits);
        println!("    - {} TT exacts", result.tt_exacts);
        println!("    - {} TT cuts", result.tt_cuts);
        println!("");
        println!("{} AlphaBeta cuts", result.beta_cuts);
        println!("");
        println!("====================================================");

    }

}
