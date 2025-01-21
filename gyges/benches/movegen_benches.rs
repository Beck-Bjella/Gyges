use criterion::{black_box, criterion_group, Criterion};

use gyges::board::TEST_BOARD;
use gyges::moves::movegen::*;
use gyges::{BoardState, Player};

pub fn movegen_main(c: &mut Criterion) {
    let mut mg = MoveGen::default();
    let board = &mut BoardState::from(TEST_BOARD);
    let player = Player::One;

    let mut group = c.benchmark_group("movegen benchmarks");

    group.bench_function("move_count", |b| b.iter(|| unsafe {
        mg.gen::<GenMoveCount, NoQuit>(black_box(board), black_box(player));
    }));

    group.bench_function("moves", |b| b.iter(|| unsafe {
        mg.gen::<GenMoves, NoQuit>(black_box(board), black_box(player));
    }));

    group.finish();
    
}

criterion_group!(
    name = movegen_benches;
    config = Criterion::default()
        .sample_size(300)
        .warm_up_time(std::time::Duration::from_secs(1));
    targets = movegen_main

);
