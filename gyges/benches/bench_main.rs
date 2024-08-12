use criterion::criterion_main;

mod board_benches;
mod movegen_benches;

criterion_main!(
    board_benches::board_benches,
    movegen_benches::movegen_benches,
    
);
