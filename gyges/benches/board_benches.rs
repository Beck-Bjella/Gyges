use criterion::{black_box, criterion_group, Criterion};

use gyges::{moves::Move, BoardState, Piece, BENCH_BOARD, SQ};

pub fn board_main(c: &mut Criterion) {
    let board = BoardState::from(BENCH_BOARD);
    let mv = Move::new([(Piece::None, SQ(0)), (Piece::Two, SQ(2)), (Piece::None, SQ::NONE)], gyges::moves::MoveType::Bounce);

    let mut group = c.benchmark_group("board benchmarks");

    group.bench_function("make_move", |b| b.iter(|| {
        black_box(board.clone().make_move(&mv));

    }));
    
    group.finish();

}

criterion_group!(
    name = board_benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(1));
    targets = board_main

);
