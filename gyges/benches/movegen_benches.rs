use std::time::Duration;

use criterion::{black_box, criterion_group, Criterion};

use gyges::{BoardState, Player};
use gyges::board::TEST_BOARD;
use gyges::moves::movegen::*;

pub fn movegen_main(c: &mut Criterion) {
    let mut mg = MoveGen::default();
    let board = &mut BoardState::from(TEST_BOARD);
    let player = Player::One;

    let mut group = c.benchmark_group("movegen benchmarks");

    group.bench_function("move_count OLD", |b| b.iter(|| unsafe {
        mg.gen_old::<GenMoveCount, NoQuit>(black_box(board), black_box(player));
    }));

    group.bench_function("move_count", |b| b.iter(|| unsafe {
        mg.gen::<GenMoveCount, NoQuit>(black_box(board), black_box(player));
    }));

    group.bench_function("move_count NEW", |b| b.iter(|| unsafe {
        mg.gen_new::<GenMoveCount, NoQuit>(black_box(board), black_box(player));
    }));


    group.finish();
    
}

criterion_group!(
    name = movegen_benches;
    config = Criterion::default()
        .sample_size(1000)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(1))
        .nresamples(100000)
        .confidence_level(0.99)
        .significance_level(0.01);
    targets = movegen_main

);
