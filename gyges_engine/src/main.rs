extern crate gyges_engine;

use gyges::{BENCH_BOARD, BoardState, Move, MoveGen, Player, STARTING_BOARD, board::{self, TEST_BOARD}};
use gyges_engine::{search::evaluation::EvaluationContext, ugi::*};

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

    // let mut board = BoardState::from("231312000000000000000000000000231132");

    // let mut mg = MoveGen::default();
    // let eval = EvaluationContext::new(&mut board, &mut mg);

    // eval.print();

}
