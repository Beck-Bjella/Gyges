extern crate gyges_engine;

use gyges::{BENCH_BOARD, BoardState, Move, MoveGen, Player, STARTING_BOARD, board::{self, TEST_BOARD}};
use gyges_engine::{search::evaluation::EvaluationContext, ugi::*};

fn main() {
    let mut ugi = Ugi::new();
    ugi.start();

}
