use crate::consts::*;
use crate::board::board::*;
use crate::moves::move_gen::*;


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

// / Sums the value of every piece that the player can touch.
// pub fn control_score(board: &mut BoardState, player: f64) -> f64 {
//     let current_moves = unsafe{ valid_moves(board, player) };

//     let mut score = 0.0;
//     for board_pos in 0..36 {
//         if current_moves.piece_replaceable(board_pos) && board.data[board_pos] != 0 {
//             score += PIECE_SCORES[board.data[board_pos] - 1]
            
//         }

//     }

//     score

// }

pub fn piece_fully_stranded(board: &mut BoardState, pos: usize, piece: usize) -> bool {
    let mut stranded = true;

    for i in 0usize..36 {
        if i != pos {
            let x = (i % 6) as f64;
            let y = (i as f64 / 6 as f64).floor();
    
            let pos_x = (pos % 6) as f64;
            let pos_y = (pos as f64 / 6 as f64).floor();
    
            let dist = ((pos_y - y).powf(2.0) + (pos_x - x).powf(2.0)).sqrt();
    
            if board.data[i] != 0 && dist <= piece as f64 {
                stranded = false
    
            }

        }

    }

    return stranded;

}

pub fn fully_stranded_pieces(board: &mut BoardState, player: f64) -> Vec<usize> {
    let active_lines = board.get_active_lines();
    let active_line;
    if player == PLAYER_1 {
        active_line = active_lines[0]

    } else {
        active_line = active_lines[1]

    }

    let mut pieces = vec![];
    for board_pos in active_line..active_line + 6 {
        let piece = board.data[board_pos];

        if piece != 0 {
            if piece_fully_stranded(board, board_pos, piece) {
                pieces.push(piece);
                
            }

        }

    }

    return pieces;

}

pub fn get_evalulation(board: &mut BoardState) -> f64 {
    // Calculates the difference in move count between player 1 and player 2.
    let move_count_eval = unsafe{ valid_move_count(board, PLAYER_1) as f64 - valid_move_count(board, PLAYER_2) as f64 };
    
    // Determins the number of peices that are fully stranded on player 1 and player 2's active lines.
    let stranded_eval = 1000.0 * (fully_stranded_pieces(board, PLAYER_2).len() as f64 - fully_stranded_pieces(board, PLAYER_1).len()  as f64);

    let eval = move_count_eval + stranded_eval;

    return eval;

} 
