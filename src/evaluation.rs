use std::f32::consts::E;

use crate::board::*;
use crate::move_gen::*;

fn in_bounds(pos: usize) -> bool {
    if pos <= 35 {
        return true;

    } else {
        return false;

    }

}

fn on_top_edge(pos: usize) -> bool {
    if pos + 6 > 35 {
        return true;

    } else {
        return false;

    }

}

fn on_bottom_edge(pos: usize) -> bool {
    if pos - 6 > 35 {
        return true;

    } else {
        return false;

    }

}

fn on_right_edge(pos: usize) -> bool {
    if pos == 5 || pos == 11 || pos == 17 || pos == 23 || pos == 29 || pos == 35 {
        return true;

    } else {
        return false;

    }

}

fn on_left_edge(pos: usize) -> bool {
    if pos == 0 || pos == 6 || pos == 12 || pos == 18 || pos == 24 || pos == 30 {
        return true;

    } else {
        return false;

    }

}

pub fn get_positional_eval(board: &mut BoardState) -> f64 {
    let mut one_conectivity = 0;

    let mut total_two_goodness = 0.0;

    for current_pos in 0..36 {
        let current_piece = board.data[current_pos];

        let mut n_piece = NULL;
        let mut ne_piece = NULL;
        let mut e_piece = NULL;
        let mut se_piece = NULL;
        let mut s_piece = NULL;
        let mut sw_piece = NULL;
        let mut w_piece = NULL;
        let mut nw_piece = NULL;

        // Check if north is vaild
        if !on_top_edge(current_pos) {
            n_piece = board.data[current_pos + 6];

        }
        // Check if north-east is vaild
        if !on_top_edge(current_pos) && !on_right_edge(current_pos) {
            ne_piece = board.data[current_pos + 7];
            
        }
        // Check if east is vaild
        if !on_right_edge(current_pos) {
            e_piece = board.data[current_pos + 1];

        }
        // Check if south-east is vaild
        if !on_right_edge(current_pos) && !on_bottom_edge(current_pos){
            se_piece = board.data[current_pos - 5];

        }
        // Check if south is vaild
        if !on_bottom_edge(current_pos){
            s_piece = board.data[current_pos - 6];

        }
        // Check if south-west is vaild
        if !on_bottom_edge(current_pos) && !on_left_edge(current_pos) {
            sw_piece = board.data[current_pos - 7];

        }
        // Check if west is vaild
        if !on_left_edge(current_pos) {
            w_piece = board.data[current_pos - 1];

        }
        // Check if north-west is vaild
        if !on_left_edge(current_pos) && !on_top_edge(current_pos) {
            nw_piece = board.data[current_pos + 5];

        }

        let mut up_pieces = vec![];
        let mut down_pieces = vec![];

        // Gather consecutive up pieces
        let mut current_pos_clone = current_pos.clone();
        if in_bounds(current_pos_clone + 6) {
            current_pos_clone += 6;

            while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0{
                up_pieces.push(board.data[current_pos_clone]);
                current_pos_clone += 6;

            }

        }

        // Gather consecutive down pieces
        let mut current_pos_clone = current_pos.clone();
        if in_bounds(current_pos_clone - 6)  {
            current_pos_clone -= 6;

            while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0{
                down_pieces.push(board.data[current_pos_clone]);
                current_pos_clone -= 6;

            }

        }

        if current_piece == 1 {
            if n_piece == 1 {
                one_conectivity += 1;

            }
            if e_piece == 1 {
                one_conectivity += 1;

            }
            if s_piece == 1 {
                one_conectivity += 1;

            }
            if w_piece == 1 {
                one_conectivity += 1;

            }

        }

        if current_piece == 2 {
            let mut goodness = 0.0;

            if n_piece == 0 {
                goodness += 3.0;

            }

            if s_piece == 0 {
                goodness -= 3.0;

            }

            for piece in down_pieces.iter() {
                if piece == &1 {
                    goodness += 3.0;

                }
                if piece == &3 {
                    goodness += 1.0;

                }

            }

            for piece in up_pieces.iter() {
                if piece == &1 {
                    goodness -= 3.0;

                }
                if piece == &3 {
                    goodness -= 1.0;

                }

            }
            
            if e_piece == 2 {
                goodness *= 1.25

            }

            if w_piece == 2 {
                goodness *= 1.25

            }

            total_two_goodness += goodness as f64;

        }

    }

    return total_two_goodness;

}

pub fn get_standered_evalulation(board: &mut BoardState, player: f64) -> f64 {
    let mut move_score: f64 = 0.0;

    // move_score += unsafe{valid_move_count(board, player)} as f64;
    move_score -= unsafe{valid_move_count(board, -player)} as f64;

    return move_score;

}  

// pub fn is_quiet(board: &mut BoardState, player: f64) -> bool {
//     let orignal_opp_eval = unsafe{valid_move_count(board, -player)} as f64;

//     let mut move_list = unsafe{valid_moves(board, player)};
//     let current_player_moves = move_list.moves(board);

//     let mut opp_lowest_eval = f64::INFINITY;
//     for mv in current_player_moves.iter() {
//         board.make_move(&mv);

//         let opp_eval = unsafe{valid_move_count(board, -player)} as f64;
        
//         board.undo_move(&mv);

//         if opp_eval < opp_lowest_eval {
//             opp_lowest_eval = opp_eval;

//         }

//     }

//     if (opp_lowest_eval as f64 / orignal_opp_eval as f64) <= 0.01 {
//         return false;

//     } else {
//         return true;

//     }

// }
