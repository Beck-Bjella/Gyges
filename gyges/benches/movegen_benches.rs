use criterion::{black_box, criterion_group, Criterion};

use gyges::moves::movegen::*;
use gyges::{BoardState, Player, BENCH_BOARD};

pub fn movegen_main(c: &mut Criterion) {
    let board = BoardState::from(BENCH_BOARD);
    let player = Player::One;

    let mut group = c.benchmark_group("movegen benchmarks");

    group.bench_function("valid_moves", |b| b.iter(|| unsafe {
        valid_moves(black_box(&mut board.clone()), black_box(player));
    }));

    group.bench_function("valid_move_count", |b| b.iter(|| unsafe {
        valid_move_count(black_box(&mut board.clone()), black_box(player));
    }));

    group.bench_function("valid_threat_count", |b| b.iter(|| unsafe {
        valid_threat_count(black_box(&mut board.clone()), black_box(player));
    }));
    
    group.bench_function("has_threat", |b| b.iter(|| unsafe {
        has_threat(black_box(&mut board.clone()), black_box(player));
    }));

    group.finish();
    
}

criterion_group!(
    name = movegen_benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(1));
    targets = movegen_main

);
