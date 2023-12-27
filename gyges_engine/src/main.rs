mod search;
mod consts;
mod ugi;

use consts::BENCH_BOARD;
use gyges::{board::board::BoardState, moves::movegen::valid_moves, core::player::Player};

use crate::ugi::*;

fn main() {
    let mut board = BoardState::from(BENCH_BOARD);

    let moves = unsafe{ valid_moves(&mut board, Player::One)}.moves(&mut board);

    for (i, mv) in moves.iter().enumerate() {
        println!("{}: {:?}", i, mv);

    }

    let mut ugi = Ugi::new();
    ugi.start();

}
