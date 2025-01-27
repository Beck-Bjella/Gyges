//! All logic and functions related to the evaluation of a board.
//! 

use gyges::board::*;
use gyges::board::bitboard::*;
use gyges::core::*;
use gyges::moves::movegen::{GenControlMoveCount, MoveGen, NoQuit};

pub const UNIQUE_PIECE_CONTROL_SCORES: [f64; 3] = [500.0, 100.0, 50.0];
pub const SHARED_PIECE_CONTROL_SCORES: [f64; 3] = [75.0, 50.0, 25.0];

pub const UNIQUE_SQUARE_CONTROL_SCORE: f64 = 10.0;
pub const SHARED_SQUARE_CONTROL_SCORE: f64 = 5.0;

// BEST EVALUATION FUNCTION
pub fn get_evalulation(board: &mut BoardState, mg: &mut MoveGen) -> f64 {
    let p1 = unsafe { mg.gen::<GenControlMoveCount, NoQuit>(board, Player::One) };
    let p2 = unsafe { mg.gen::<GenControlMoveCount, NoQuit>(board, Player::Two) };
    let control_squares = [p1.controlled_squares, p2.controlled_squares];
    let control_pieces = [p1.controlled_pieces, p2.controlled_pieces];
    let move_counts = [p1.move_count, p2.move_count];

    let mut eval = 0.0;

    eval += mobility_eval(Player::One, move_counts) - mobility_eval(Player::Two, move_counts);
    eval += (control_eval(board, Player::One, control_pieces, control_squares) - control_eval(board, Player::Two, control_pieces, control_squares)) * 3.0;

    eval

}

pub fn unique_controlled_pieces_score(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2]) -> f64 {
    let pieces = control_pieces[player as usize];
    let opp_pieces = control_pieces[player.other() as usize];
    
    let mut unique_controlled_pieces = pieces & !opp_pieces;

    let positions = unique_controlled_pieces.get_data();

    let mut score = 0.0;
    for pos in positions {
        let piece = board.data[pos];
        score += UNIQUE_PIECE_CONTROL_SCORES[piece as usize];

    }

    score

}

pub fn shared_controlled_pieces_score(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2]) -> f64 {
    let mut pieces = control_pieces[player as usize];

    let positions = pieces.get_data();

    let mut score = 0.0;
    for pos in positions {
        let piece = board.data[pos];
        score += SHARED_PIECE_CONTROL_SCORES[piece as usize];

    }

    score

}

pub fn unique_controlled_squares_score(player: Player, control_squares: [BitBoard; 2]) -> f64 {
    let squares = control_squares[player as usize];
    let opp_squares = control_squares[player.other() as usize];
    
    let unique_squares = squares & !opp_squares;

    unique_squares.pop_count() as f64 * UNIQUE_SQUARE_CONTROL_SCORE

}

pub fn shared_controlled_squares_score(player: Player, control_squares: [BitBoard; 2]) -> f64 {
    let squares = control_squares[player as usize];
    squares.pop_count() as f64 * SHARED_SQUARE_CONTROL_SCORE

}

pub fn mobility_eval(player: Player, move_counts: [usize; 2]) -> f64 {
    let mut eval = 0.0;
 
    eval += move_counts[player as usize] as f64;

    eval

}

pub fn control_eval(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2], control_squares: [BitBoard; 2]) -> f64 {
    let mut eval = 0.0;
    
    eval +=  unique_controlled_pieces_score(board, player, control_pieces);
    eval +=  unique_controlled_squares_score(player, control_squares);

    // eval +=  shared_controlled_pieces_score(board, player, control_pieces);
    // eval +=  shared_controlled_squares_score(player, control_squares);

    eval
    
}
