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


// OLD TESTS
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

pub const CONNECETED_BONUSES: [f64; 3] = [15.0, 5.0, 5.0];

pub fn ones_connectivity_score(board: &mut BoardState, player: f64) -> f64 {
    let mut connectivity: f64 = 0.0;

    let mut pieces = unsafe{ controlled_pieces(board, player) };
    let positions = pieces.get_data();

    for pos in positions {
        let piece = board.data[pos];

        if piece == 1 {
            let mut adjecent_pieces = vec![];

            if !on_top_edge(pos) && board.data[pos + 6] != 0 { adjecent_pieces.push(board.data[pos + 6]) }; // N
            if !on_top_edge(pos) && !on_right_edge(pos) && board.data[pos + 7] != 0 { adjecent_pieces.push(board.data[pos + 7]) }; // NE
            if !on_right_edge(pos) && board.data[pos + 1] != 0  { adjecent_pieces.push(board.data[pos + 1]) }; // E
            if !on_right_edge(pos) && !on_bottom_edge(pos) && board.data[pos - 5] != 0 { adjecent_pieces.push(board.data[pos - 5]) }; // SE
            if !on_bottom_edge(pos) && board.data[pos - 6] != 0  { adjecent_pieces.push(board.data[pos - 6]) }; // S
            if !on_bottom_edge(pos) && !on_left_edge(pos) && board.data[pos - 7] != 0  { adjecent_pieces.push(board.data[pos - 7]) }; // SW
            if !on_left_edge(pos) && board.data[pos - 1] != 0 { adjecent_pieces.push(board.data[pos - 1]) }; // W
            if !on_left_edge(pos) && !on_top_edge(pos) && board.data[pos + 5] != 0 { adjecent_pieces.push(board.data[pos + 5]) }; // NW
            
            for adj_piece in adjecent_pieces {
                connectivity += CONNECETED_BONUSES[adj_piece - 1];
    
            }

        }

    }

    connectivity

}

pub fn ones_safety_score(board: &mut BoardState, player: f64) -> f64 {
    let mut safety = 0.0;

    safety

}

pub const PROTECTED_PIECE_SCORES: [f64; 3] = [250.0, 25.0, 100.0];

pub const HALF_OFFSET_SCORE: f64 = 300.0;

pub fn wall_depth_offset(board: &mut BoardState) -> f64 {
    let mut total_depth = 0.0;

    for pos in 0..36 {
        let piece = board.data[pos];

        if piece == 2 {
            let col = (pos as f64 / 6.0).floor();
            total_depth += col;

        }

    }

    (total_depth / 4.0) - 2.5

}

pub fn wall_strength(board: &mut BoardState) -> f64 {
    let mut strength = 0.0;

    for pos in 0..36 {
        let piece = board.data[pos];

        if piece == 2 {
            let ne_piece = if !on_top_edge(pos) && !on_right_edge(pos) { board.data[pos + 7] } else { NULL };
            let e_piece = if !on_right_edge(pos) { board.data[pos + 1] } else { NULL };
            let se_piece = if !on_right_edge(pos) && !on_bottom_edge(pos){ board.data[pos - 5] } else { NULL };
            
            let sw_piece = if !on_bottom_edge(pos) && !on_left_edge(pos) { board.data[pos - 7] } else { NULL };
            let w_piece = if !on_left_edge(pos) { board.data[pos - 1] } else { NULL };
            let nw_piece =  if !on_left_edge(pos) && !on_top_edge(pos) { board.data[pos + 5] } else { NULL };
        
            if ne_piece == 2 {
                strength += 1.0;

            } 
            if se_piece == 2 {
                strength += 1.0;

            } 
            if e_piece == 2 {
                strength += 2.0;

            }
            
            if nw_piece == 2 {
                strength += 1.0;

            } 
            if sw_piece == 2 {
                strength += 1.0;

            } 
            if w_piece == 2 {
                strength += 2.0;

            }

        }

    }

    strength

}

pub fn p1_wall_score(board: &mut BoardState) -> f64 {
    let mut protected_pieces_score: f64 = 0.0;
    let mut test = 0.0;

    for pos in 0..36 {
        let piece = board.data[pos];

        if piece == 2 {
            let mut down_pieces = vec![];
            let mut current_pos_clone = pos;
            if in_bounds(current_pos_clone - 6)  {
                current_pos_clone -= 6;

                while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0 {
                    down_pieces.push(board.data[current_pos_clone]);
                    current_pos_clone -= 6;

                }

            }

            for d_piece in down_pieces {
                protected_pieces_score += PROTECTED_PIECE_SCORES[d_piece - 1];

            }

        }

    }

    let wall_offset = wall_depth_offset(board);
    let offset_score = HALF_OFFSET_SCORE * wall_offset;

    protected_pieces_score + offset_score + test
    
}

pub fn p2_wall_score(board: &mut BoardState) -> f64 {
    let mut protected_pieces_score: f64 = 0.0;
    let mut test = 0.0;

    for pos in 0..36 {
        let piece = board.data[pos];

        if piece == 2 {
            let mut up_pieces = vec![];
            let mut current_pos_clone = pos;
            if in_bounds(current_pos_clone + 6) {
                current_pos_clone += 6;

                while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0{
                    up_pieces.push(board.data[current_pos_clone]);
                    current_pos_clone += 6;

                }

            }
            

            for u_piece in up_pieces {
                protected_pieces_score += PROTECTED_PIECE_SCORES[u_piece - 1];

            }

        }

    }

    let wall_offset = wall_depth_offset(board);
    let offset_score = HALF_OFFSET_SCORE * -wall_offset;

    protected_pieces_score + offset_score + test
    
}

pub const UNIQUE_PIECE_CONTROL_SCORES: [f64; 3] = [300.0, 100.0, 50.0];
pub const SHARED_PIECE_CONTROL_SCORES: [f64; 3] = [75.0, 50.0, 25.0];

pub const UNIQUE_SQUARE_CONTROL_SCORE: f64 = 20.0;
pub const SHARED_SQUARE_CONTROL_SCORE: f64 = 5.0;

pub fn unique_controlled_pieces_score(board: &mut BoardState, player: f64) -> f64 {
    let pieces = unsafe{ controlled_pieces(board, player) };
    let opp_pieces = unsafe{ controlled_pieces(board, -player) };
    
    let mut unique_controlled_pieces = pieces & !opp_pieces;

    let positions = unique_controlled_pieces.get_data();

    let mut score = 0.0;
    for pos in positions {
        let piece = board.data[pos];
        score += UNIQUE_PIECE_CONTROL_SCORES[piece - 1];

    }

    score

}

pub fn shared_controlled_pieces_score(board: &mut BoardState, player: f64) -> f64 {
    let mut pieces = unsafe{ controlled_pieces(board, player) };

    let positions = pieces.get_data();

    let mut score = 0.0;
    for pos in positions {
        let piece = board.data[pos];
        score += SHARED_PIECE_CONTROL_SCORES[piece - 1];

    }

    score

}

pub fn unique_controlled_squares_score(board: &mut BoardState, player: f64) -> f64 {
    let squares = unsafe{ controlled_squares(board, player) };
    let opp_squares = unsafe{ controlled_squares(board, -player) };
    
    let unique_squares = squares & !opp_squares;

    unique_squares.pop_count() as f64 * UNIQUE_SQUARE_CONTROL_SCORE

}

pub fn shared_controlled_squares_score(board: &mut BoardState, player: f64) -> f64 {
    let squares = unsafe{ controlled_squares(board, player) };
    squares.pop_count() as f64 * SHARED_SQUARE_CONTROL_SCORE

}


pub fn mobility_eval(board: &mut BoardState, player: f64) -> f64 {
    let mut eval = 0.0;
 
    eval += unsafe{ valid_move_count(board, player) } as f64;

    eval

}

pub fn control_eval(board: &mut BoardState, player: f64) -> f64 {
    let mut eval = 0.0;

    eval +=  unique_controlled_pieces_score(board, player);
    eval +=  unique_controlled_squares_score(board, player);

    eval +=  shared_controlled_pieces_score(board, player);
    eval +=  shared_controlled_squares_score(board, player);

    eval

}

pub fn ones_eval(board: &mut BoardState, player: f64) -> f64 {
    let mut eval = 0.0;

    eval += ones_connectivity_score(board, player);
    eval += ones_safety_score(board, player);

    eval

}

pub fn get_evalulation(board: &mut BoardState) -> f64 {
    let mut eval = 0.0;
    
    eval += mobility_eval(board, PLAYER_1) - mobility_eval(board, PLAYER_2);
    eval += control_eval(board, PLAYER_1) - control_eval(board, PLAYER_2);

    eval += (p1_wall_score(board) - p2_wall_score(board)) * wall_strength(board);

    eval += ones_eval(board, PLAYER_1) - ones_eval(board, PLAYER_2);

    eval

} 
