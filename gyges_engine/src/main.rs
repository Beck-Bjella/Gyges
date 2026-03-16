extern crate gyges_engine;

use gyges::{BENCH_BOARD, BoardState, Move, MoveGen, Player, STARTING_BOARD, board::TEST_BOARD};
use gyges_engine::{search::evaluation::EvaluationContext, ugi::*};

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

    let mut board = BoardState::from(TEST_BOARD);
    let mut player = Player::One;
    let mut mg = MoveGen::default();
    
    let evaluation_ctx = EvaluationContext::new(&mut board, &mut mg);
    evaluation_ctx.print();

}
