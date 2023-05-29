use crate::board::bitboard::BitBoard;
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

pub fn unreaceable_positions(board: &mut BoardState) -> BitBoard {
    let mut piece_board = board.peice_board.clone();
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

    return !reach_positions; 

}


pub fn piece_cant_reach(board: &mut BoardState, pos: usize, piece: usize) -> bool {
    if piece == 3 {
        return (THREE_ENDS[pos] & board.peice_board).is_empty();

    } else if piece == 2 {
        return (TWO_ENDS[pos] & board.peice_board).is_empty();

    } else if piece == 1{
        return (ONE_ENDS[pos] & board.peice_board).is_empty();

    }

    return false;

}

pub fn activeline_unreachable(board: &mut BoardState, player: f64) -> usize {
    let active_lines = board.get_active_lines();

    let active_line: usize;
    if player == PLAYER_1 {
        active_line = active_lines[0];

    } else {
        active_line = active_lines[1];

    }

    let activeline_board = ROWS[active_line] & board.peice_board;

    let unreach_pos = unreaceable_positions(board);

    return (unreach_pos & activeline_board).pop_count();

}

pub fn activeline_cant_reach(board: &mut BoardState, player: f64) -> usize {
    let active_lines = board.get_active_lines();

    let active_line: usize;
    if player == PLAYER_1 {
        active_line = active_lines[0];

    } else {
        active_line = active_lines[1];

    }

    let mut count = 0;
    for x in (active_line * 6)..((active_line * 6) + 6) {
        if piece_cant_reach(board, x, board.data[x]) {
            count += 1;

        }

    }

    return count;


}


pub fn get_evalulation(board: &mut BoardState) -> f64 {
    // Calculates the difference in move count between player 1 and player 2.
    let move_count_eval = unsafe{ valid_move_count(board, PLAYER_1) as f64 - valid_move_count(board, PLAYER_2) as f64 };
    
    // Determins the number of peices that cant theoriticaly reach anything on player 1 and player 2's active lines.
    // let cant_reach_eval = 1000.0 * (activeline_cant_reach(board, PLAYER_2) as f64 - activeline_cant_reach(board, PLAYER_1) as f64);

    // Determins the number of peices that are theoriticaly unreachable on player 1 and player 2's active lines.
    // let unreachable_eval = 500.0 * (activeline_unreachable(board, PLAYER_2) as f64 - activeline_unreachable(board, PLAYER_1) as f64);

    let eval = move_count_eval;// + cant_reach_eval + unreachable_eval;

    return eval;

} 
