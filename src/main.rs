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

use evaluation::get_positional_eval;

use crate::board::*;
use crate::engine::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let mut board = BoardState::new();
    board.set_rank([3, 0, 0 ,3, 0, 0], 5);
    board.set_rank([0 ,0 ,0 ,1, 1, 1], 4);
    board.set_rank([0 ,0 ,0 ,2, 3, 1], 3);
    board.set_rank([0 ,0 ,0 ,0, 2, 2], 2);
    board.set_rank([0 ,0, 3, 0, 0, 0], 1);
    board.set_rank([0 ,0 ,0 ,2, 0, 0], 0);
    board.set_goals([0, 0]);

    // board.flip();

    // board.set_rank([0, 0, 0 ,0, 0, 0], 5);
    // board.set_rank([0 ,0 ,0 ,0, 0, 0], 4);
    // board.set_rank([0 ,2 ,2 ,2, 0, 3], 3);
    // board.set_rank([0 ,3 ,1 ,3, 1, 2], 2);
    // board.set_rank([0 ,0, 1, 3, 0, 1], 1);
    // board.set_rank([0 ,0 ,0 ,0, 0, 0], 0);
    // board.set_goals([0, 0]);
    // println!("{}", get_positional_eval(&mut board));

    let mut negamax = Negamax::new();

    for result in negamax.iterative_deepening_search(&mut board, 5) {
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
