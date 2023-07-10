use std::f32::consts::E;

use crate::board::bitboard::BitBoard;
use crate::{consts::*, board};
use crate::board::board::*;
use crate::moves::move_gen::*;

fn in_bounds(pos: usize) -> bool {
    pos <= 35

}

fn on_top_edge(pos: usize) -> bool {
    pos + 6 > 35

}

fn on_bottom_edge(pos: usize) -> bool {
    pos - 6 > 35

}

fn on_right_edge(pos: usize) -> bool {
    pos == 5 || pos == 11 || pos == 17 || pos == 23 || pos == 29 || pos == 35

}

fn on_left_edge(pos: usize) -> bool {
    pos == 0 || pos == 6 || pos == 12 || pos == 18 || pos == 24 || pos == 30

}


pub fn unreaceable_positions(board: &mut BoardState) -> BitBoard {
    let mut piece_board = board.piece_bb;
    let piece_positions = piece_board.get_data();

    let mut reach_positions = EMPTY;

    for pos in piece_positions {
        if board.data[pos] == 1 {
            reach_positions |= ONE_ENDS[pos];

        } else if board.data[pos] == 2 {
            reach_positions |= TWO_ENDS[pos];

        } else if board.data[pos] == 3 {
            reach_positions |= THREE_ENDS[pos];

        }

    }

    !reach_positions

}

pub fn piece_cant_reach(board: &mut BoardState, pos: usize, piece: usize) -> bool {
    if piece == 3 {
        return (THREE_ENDS[pos] & board.piece_bb).is_empty();

    } else if piece == 2 {
        return (TWO_ENDS[pos] & board.piece_bb).is_empty();

    } else if piece == 1{
        return (ONE_ENDS[pos] & board.piece_bb).is_empty();

    }

    false

}

pub fn activeline_unreachable(board: &mut BoardState, player: f64) -> usize {
    let active_lines = board.get_active_lines();

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0]

    } else {
        active_lines[1]

    };

    let activeline_board = ROWS[active_line] & board.piece_bb;

    let unreach_pos = unreaceable_positions(board);

    (unreach_pos & activeline_board).pop_count()

}

pub fn activeline_cant_reach(board: &mut BoardState, player: f64) -> usize {
    let active_lines = board.get_active_lines();

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0]

    } else {
        active_lines[1]

    };

    let mut count = 0;
    for x in (active_line * 6)..((active_line * 6) + 6) {
        if piece_cant_reach(board, x, board.data[x]) {
            count += 1;

        }

    }

    count


}

// pub fn p1_positional_eval(board: &mut BoardState, p1_piece_control: BitBoard, p2_piece_control: BitBoard) -> f64 {
//     let mut one_conectivity = 0.0;
//     let mut one_safety = 0.0;
//     let mut two_protection = 0.0;

//     let mut p1_unique = p1_piece_control & !p2_piece_control;
//     let mut p2_unique = p2_piece_control & !p1_piece_control;

//     for pos in p1_piece_control.clone().get_data() {
//         let current_piece = board.data[pos];

//         let mut n_piece = NULL;
//         let mut ne_piece = NULL;
//         let mut e_piece = NULL;
//         let mut se_piece = NULL;
//         let mut s_piece = NULL;
//         let mut sw_piece = NULL;
//         let mut w_piece = NULL;
//         let mut nw_piece = NULL;

//         // Check if north is vaild
//         if !on_top_edge(pos) {
//             n_piece = board.data[pos + 6];

//         }
//         // Check if north-east is vaild
//         if !on_top_edge(pos) && !on_right_edge(pos) {
//             ne_piece = board.data[pos + 7];
            
//         }
//         // Check if east is vaild
//         if !on_right_edge(pos) {
//             e_piece = board.data[pos + 1];

//         }
//         // Check if south-east is vaild
//         if !on_right_edge(pos) && !on_bottom_edge(pos){
//             se_piece = board.data[pos - 5];

//         }
//         // Check if south is vaild
//         if !on_bottom_edge(pos){
//             s_piece = board.data[pos - 6];

//         }
//         // Check if south-west is vaild
//         if !on_bottom_edge(pos) && !on_left_edge(pos) {
//             sw_piece = board.data[pos - 7];

//         }
//         // Check if west is vaild
//         if !on_left_edge(pos) {
//             w_piece = board.data[pos - 1];

//         }
//         // Check if north-west is vaild
//         if !on_left_edge(pos) && !on_top_edge(pos) {
//             nw_piece = board.data[pos + 5];

//         }

//         if current_piece == 1 {
//             let unique = (p1_unique & 1 << pos).is_not_empty();
//             let unique_bonus = if unique {
//                 3.0

//             } else {
//                 1.0

//             };

//             // One Connectivity
//             if n_piece == 1 {
//                 one_conectivity += 1.0;

//             }
//             if e_piece == 1 {
//                 one_conectivity += 1.0;

//             }
//             if s_piece == 1 {
//                 one_conectivity += 1.0;

//             }
//             if w_piece == 1 {
//                 one_conectivity += 1.0;

//             }


//             // One Safety
//             if n_piece == 2 {
//                 one_safety += 2.0 * unique_bonus;

//             }

//             if nw_piece == 2 && (p2_piece_control & 1 << pos).is_empty() {
//                 one_safety += 1.0 * unique_bonus;

//             }
//             if ne_piece == 2 && (p2_piece_control & 1 << pos).is_empty() {
//                 one_safety += 1.0  * unique_bonus;

//             }

//         }

//         if current_piece == 2 {
//             let mut protection = 0.0;

//             if n_piece == 0 {
//                 protection += 3.0;

//             }

//             // Gather consecutive down pieces
//             let mut down_pieces = vec![];
//             let mut current_pos_clone = pos;
//             if in_bounds(current_pos_clone - 6)  {
//                 current_pos_clone -= 6;

//                 while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0 {
//                     down_pieces.push(board.data[current_pos_clone]);
//                     current_pos_clone -= 6;

//                 }

//             }

//             for piece in down_pieces.iter() {
//                 if piece == &1 {
//                     protection += 3.0;

//                 }
//                 if piece == &2 {
//                     protection -= 2.0;

//                 }
//                 if piece == &3 {
//                     protection += 1.0;

//                 }

//             }

//             if w_piece == 2 {
//                 protection += 2.0;

//             }
//             if e_piece == 2 {
//                 protection += 2.0;
                
//             }
        
        
//             two_protection += protection;

//         }

//     }

//     (two_protection * 300.0) + (one_conectivity * 20.0) + (one_safety * 80.0)

// }

// pub fn p2_positional_eval(board: &mut BoardState, p1_piece_control: BitBoard, p2_piece_control: BitBoard) -> f64 {
//     let mut one_conectivity = 0.0;
//     let mut one_safety = 0.0;
//     let mut two_protection = 0.0;

//     let mut p1_unique = p1_piece_control & !p2_piece_control;
//     let mut p2_unique = p2_piece_control & !p1_piece_control;

//     for pos in p2_piece_control.clone().get_data() {
//         let current_piece: usize = board.data[pos];

//         let mut n_piece = NULL;
//         let mut ne_piece = NULL;
//         let mut e_piece = NULL;
//         let mut se_piece = NULL;
//         let mut s_piece = NULL;
//         let mut sw_piece = NULL;
//         let mut w_piece = NULL;
//         let mut nw_piece = NULL;

//         // Check if north is vaild
//         if !on_top_edge(pos) {
//             n_piece = board.data[pos + 6];

//         }
//         // Check if north-east is vaild
//         if !on_top_edge(pos) && !on_right_edge(pos) {
//             ne_piece = board.data[pos + 7];
            
//         }
//         // Check if east is vaild
//         if !on_right_edge(pos) {
//             e_piece = board.data[pos + 1];

//         }
//         // Check if south-east is vaild
//         if !on_right_edge(pos) && !on_bottom_edge(pos){
//             se_piece = board.data[pos - 5];

//         }
//         // Check if south is vaild
//         if !on_bottom_edge(pos){
//             s_piece = board.data[pos - 6];

//         }
//         // Check if south-west is vaild
//         if !on_bottom_edge(pos) && !on_left_edge(pos) {
//             sw_piece = board.data[pos - 7];

//         }
//         // Check if west is vaild
//         if !on_left_edge(pos) {
//             w_piece = board.data[pos - 1];

//         }
//         // Check if north-west is vaild
//         if !on_left_edge(pos) && !on_top_edge(pos) {
//             nw_piece = board.data[pos + 5];

//         }

//         if current_piece == 1 {
//             let unique = (p1_unique & 1 << pos).is_not_empty();
//             let unique_bonus = if unique {
//                 3.0

//             } else {
//                 1.0

//             };

//             // One Connectivity
//             if n_piece == 1 {
//                 one_conectivity += 1.0;

//             }
//             if e_piece == 1 {
//                 one_conectivity += 1.0;

//             }
//             if s_piece == 1 {
//                 one_conectivity += 1.0;

//             }
//             if w_piece == 1 {
//                 one_conectivity += 1.0;

//             }


//             // One Safety
//             if s_piece == 2 {
//                 one_safety += 2.0 * unique_bonus;

//             }

//             if sw_piece == 2 && (p1_piece_control & 1 << pos).is_empty() {
//                 one_safety += 1.0 * unique_bonus;

//             }
//             if se_piece == 2 && (p1_piece_control & 1 << pos).is_empty() {
//                 one_safety += 1.0 * unique_bonus;

//             }

//         }

//         if current_piece == 2 {
//             let mut protection = 0.0;

//             if s_piece == 0 {
//                 protection += 3.0;

//             }

//             // Gather consecutive up pieces
//             let mut up_pieces = vec![];
//             let mut current_pos_clone = pos;
//             if in_bounds(current_pos_clone + 6) {
//                 current_pos_clone += 6;

//                 while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0{
//                     up_pieces.push(board.data[current_pos_clone]);
//                     current_pos_clone += 6;

//                 }

//             }

//             for piece in up_pieces.iter() {
//                 if piece == &1 {
//                     protection += 3.0;

//                 }
//                 if piece == &2 {
//                     protection -= 2.0;

//                 }
//                 if piece == &3 {
//                     protection += 1.0;

//                 }

//             }
        
//             two_protection += protection;

//         }

//     }

//     (two_protection * 300.0) + (one_conectivity * 20.0) + (one_safety * 80.0)

// }

pub const PIECE_CONTROL_SCORES: [f64; 3] = [300.0, 100.0, 50.0];

pub const SQUARE_CONTROL_SCORE: f64 = 20.0;

pub fn unique_controlled_pieces_score(board: &mut BoardState, player: f64) -> f64 {
    let pieces = unsafe{ controlled_pieces(board, player) };
    let opp_pieces = unsafe{ controlled_pieces(board, -player) };
    
    let mut unique_controlled_pieces = pieces & !opp_pieces;

    let positions = unique_controlled_pieces.get_data();

    let mut score = 0.0;
    for pos in positions {
        let piece = board.data[pos];
        score += PIECE_CONTROL_SCORES[piece - 1];

    }

    score

}

pub fn unique_controlled_squares_score(board: &mut BoardState, player: f64) -> f64 {
    let squares = unsafe{ controlled_squares(board, player) };
    let opp_squares = unsafe{ controlled_squares(board, -player) };
    
    let unique_squares = squares & !opp_squares;

    unique_squares.pop_count() as f64 * SQUARE_CONTROL_SCORE

}


pub const TEMPO_BONUS: f64 = 3000.0;

pub fn get_evalulation(board: &mut BoardState) -> f64 {
    let mut eval = 0.0;
    
    eval += unsafe{ valid_move_count(board, PLAYER_1) } as f64 - unsafe{ valid_move_count(board, PLAYER_2) } as f64;
    eval +=  unique_controlled_pieces_score(board, PLAYER_1) - unique_controlled_pieces_score(board, PLAYER_2);
    eval +=  unique_controlled_squares_score(board, PLAYER_1) - unique_controlled_squares_score(board, PLAYER_2);

    // eval += board.player * TEMPO_BONUS;

    eval

} 
