use std::vec;

use crate::board::*;
use crate::move_generation::*;

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
    let mut score = 0.0;

    for current_pos in 0..board.data.len() {
        let current_piece = board.data[current_pos];

        if current_piece == 2 {
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

            let mut current_pos_clone = current_pos.clone();
            if in_bounds(current_pos_clone + 6) {
                current_pos_clone += 6;

                while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0{
                    up_pieces.push(board.data[current_pos_clone]);
                    current_pos_clone += 6;
    
                }

            }

            let mut current_pos_clone = current_pos.clone();
            if in_bounds(current_pos_clone - 6)  {
                current_pos_clone -= 6;

                while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0{
                    down_pieces.push(board.data[current_pos_clone]);
                    current_pos_clone -= 6;
    
                }

            }

            let mut goodness = 0.0;

            if up_pieces.len() == 0 {
                goodness += 3.0;

            }

            for piece in down_pieces {
                if piece == 1 {
                    goodness += 2.0;

                }

            }


            // if down_pieces[0] == 3 {
            //     if w_piece == 0 && sw_piece == 0 {
            //         goodness += 1

            //     } 

            //     if e_piece == 0 && se_piece == 0 {
            //         goodness += 1

            //     }

            // }
            

            if e_piece == 2 {
                goodness *= 1.25

            }
            if w_piece == 2 {
                goodness *= 1.25

            }


            println!("current_pos: {},   goodness: {}", current_pos, goodness);
            println!("");

            score += goodness as f64;

        }

    }

    return score;

}


pub fn get_evalulation(board: &mut BoardState) -> f64 {
    let mut score: f64 = 0.0;

    //  Move Counts

    let player_1_move_count = valid_move_count(board, 1) as f64;
    let player_2_moves_count = valid_move_count(board, 2) as f64;

    score += player_1_move_count * 1.5;
    score -= player_2_moves_count;

    return score;

}  
