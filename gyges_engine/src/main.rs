extern crate gyges_engine;

use gyges::{moves::movegen::{MoveGen, MoveGenRequest, MoveGenType}, BoardState, Player, BENCH_BOARD};
use gyges_engine::ugi::Ugi;

fn main() {
    // let mut movegen = MoveGen::new();


    // for _ in 0..50 {
    //     // gen_single(&mut movegen);
    //     gen_batch(&mut movegen);

    // }
    
    let mut ugi = Ugi::new();
    ugi.start();

}

pub const ITERS: usize = 1000000;

pub fn gen_single(movegen: &mut MoveGen) {
    let mut boards = vec![];
    for _ in 0..ITERS {
        let board = BoardState::from(BENCH_BOARD);
        boards.push(board);

    }

    let start = std::time::Instant::now();
    for (i, board) in boards.iter().enumerate() {
        movegen.gen(&mut board.clone(), Player::One, MoveGenType::ValidMoves, i);

    }
    println!("movegen count: {}", movegen.gen_count);
    println!("Single: {:?}", start.elapsed().as_millis());

}

pub fn gen_batch(movegen: &mut MoveGen) {
    let mut boards = vec![];
    for _ in 0..ITERS {
        let board = BoardState::from(BENCH_BOARD);
        boards.push(board);

    }

    for (i, board) in boards.iter().enumerate() {
        movegen.queue(MoveGenRequest::new(board.clone(), Player::One, MoveGenType::ValidMoves, i));

    }

    let start = std::time::Instant::now();
    loop {
        movegen.get();
        if movegen.results.len() == ITERS {
            break;
            
        }

    }
    println!("movegen count: {}", movegen.gen_count);
    println!("Batch: {:?}", start.elapsed().as_millis());
    movegen.clear();


}
