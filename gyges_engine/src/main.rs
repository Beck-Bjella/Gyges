extern crate gyges_engine;

use gyges::{board::{self, TEST_BOARD}, moves::{movegen::{test_threat_or_movecount, threat_or_movecount, threat_or_movecount_simd, valid_move_count}, movegen_consts::{ALL_TWO_INTERCEPTS, ONE_MAP, ONE_PATH_COUNT_IDX, UNIQUE_ONE_PATHS, UNIQUE_ONE_PATH_LISTS}, Move, MoveType}, BoardState, Piece, Player, BENCH_BOARD, SQ, STARTING_BOARD};
use gyges::moves::movegen_consts::*;

use gyges_engine::ugi::*;

fn main() {
    // let mut ugi = Ugi::new();
    // ugi.start();

    unsafe {
        benchmark();

    }

}
