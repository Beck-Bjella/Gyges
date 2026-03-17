extern crate gyges_engine;

use gyges::{BENCH_BOARD, BoardState, Move, MoveGen, Player, STARTING_BOARD, board::{self, TEST_BOARD}};
use gyges_engine::{search::evaluation::EvaluationContext, ugi::*};

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

    // // let mut board = BoardState::from("00230001001001002002003000030000312");
    // // let mut board = BoardState::from("0210100020300300000030000202000311");
    // let mut board = BoardState::from(BENCH_BOARD);

    // let mut player = Player::One;
    // let mut mg = MoveGen::default();
    
    // let evaluation_ctx = EvaluationContext::new(&mut board, &mut mg);
    // evaluation_ctx.print();

}
