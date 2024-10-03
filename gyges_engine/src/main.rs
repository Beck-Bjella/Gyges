extern crate gyges_engine;

use gyges::{moves::movegen::{self, MoveGen, MoveGenRequest, MoveGenType}, BoardState, Player, BENCH_BOARD, STARTING_BOARD};
// use gyges_engine::ugi::Ugi;

fn main() {
    let mut movegen = MoveGen::new();
    
    for i in 0..50 {
        gen_single(&mut movegen);
        gen_batch(&mut movegen);

    }
    
    // let mut ugi = Ugi::new();
    // ugi.start();

}

pub fn gen_single(movegen: &mut MoveGen) {
    let iters = 1000000;

    let mut boards = vec![];
    for _ in 0..iters {
        let board = BoardState::from(BENCH_BOARD);
        boards.push(board);

    }

    let start = std::time::Instant::now();

    for (i, board) in boards.iter().enumerate() {
        let moves = movegen.gen(&mut board.clone(), Player::One, MoveGenType::ValidMoves, i);
    }

    println!("Single: {:?}", start.elapsed().as_millis());

}

pub fn gen_batch(movegen: &mut MoveGen) {
    let iters = 1000000;

    let mut boards = vec![];
    for _ in 0..iters {
        let board = BoardState::from(BENCH_BOARD);
        boards.push(board);

    }

    for (i, board) in boards.iter().enumerate() {
        movegen.queue(MoveGenRequest::new(board.clone(), Player::One, MoveGenType::ValidMoves, i));

    }

    let start = std::time::Instant::now();

    loop {
        movegen.get();
        if movegen.results.len() == iters {
            break;
            
        }

    }
    
    println!("Batch: {:?}", start.elapsed().as_millis());
    movegen.clear();


}
