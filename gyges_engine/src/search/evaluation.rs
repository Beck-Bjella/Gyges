use gyges::board::board::*;
use gyges::board::bitboard::*;
use gyges::core::player::Player;
use gyges::moves::movegen::*;

pub const UNIQUE_PIECE_CONTROL_SCORES: [f64; 3] = [300.0, 100.0, 50.0];
pub const SHARED_PIECE_CONTROL_SCORES: [f64; 3] = [75.0, 50.0, 25.0];

pub const UNIQUE_SQUARE_CONTROL_SCORE: f64 = 20.0;
pub const SHARED_SQUARE_CONTROL_SCORE: f64 = 5.0;

pub const PROTECTED_PIECE_SCORES: [f64; 3] = [250.0, 25.0, 100.0];
pub const HALF_OFFSET_SCORE: f64 = 300.0;

// fn in_bounds(pos: usize) -> bool {
//     pos <= 35

// }

// fn on_top_edge(pos: usize) -> bool {
//     pos + 6 > 35

// }

// fn on_bottom_edge(pos: usize) -> bool {
//     pos - 6 > 35

// }

// fn on_right_edge(pos: usize) -> bool {
//     pos == 5 || pos == 11 || pos == 17 || pos == 23 || pos == 29 || pos == 35

// }

// fn on_left_edge(pos: usize) -> bool {
//     pos == 0 || pos == 6 || pos == 12 || pos == 18 || pos == 24 || pos == 30

// }

// pub const CONNECETED_BONUSES: [f64; 3] = [15.0, 5.0, 5.0];

// pub fn ones_connectivity_score(board: &mut BoardState, player: f64) -> f64 {
//     let mut connectivity: f64 = 0.0;

//     let mut pieces = unsafe{ controlled_pieces(board, player) };
//     let positions = pieces.get_data();

//     for pos in positions {
//         let piece = board.data[pos];

//         if piece == 1 {
//             let mut adjecent_pieces = vec![];

//             if !on_top_edge(pos) && board.data[pos + 6] != 0 { adjecent_pieces.push(board.data[pos + 6]) }; // N
//             if !on_top_edge(pos) && !on_right_edge(pos) && board.data[pos + 7] != 0 { adjecent_pieces.push(board.data[pos + 7]) }; // NE
//             if !on_right_edge(pos) && board.data[pos + 1] != 0  { adjecent_pieces.push(board.data[pos + 1]) }; // E
//             if !on_right_edge(pos) && !on_bottom_edge(pos) && board.data[pos - 5] != 0 { adjecent_pieces.push(board.data[pos - 5]) }; // SE
//             if !on_bottom_edge(pos) && board.data[pos - 6] != 0  { adjecent_pieces.push(board.data[pos - 6]) }; // S
//             if !on_bottom_edge(pos) && !on_left_edge(pos) && board.data[pos - 7] != 0  { adjecent_pieces.push(board.data[pos - 7]) }; // SW
//             if !on_left_edge(pos) && board.data[pos - 1] != 0 { adjecent_pieces.push(board.data[pos - 1]) }; // W
//             if !on_left_edge(pos) && !on_top_edge(pos) && board.data[pos + 5] != 0 { adjecent_pieces.push(board.data[pos + 5]) }; // NW
            
//             for adj_piece in adjecent_pieces {
//                 connectivity += CONNECETED_BONUSES[adj_piece - 1];
    
//             }

//         }

//     }

//     connectivity

// }

// pub fn wall_depth_offset(board: &mut BoardState) -> f64 {
//     let mut total_depth = 0.0;

//     for pos in 0..36 {
//         let piece = board.data[pos];

//         if piece == 2 {
//             let col = (pos as f64 / 6.0).floor();
//             total_depth += col;

//         }

//     }

//     (total_depth / 4.0) - 2.5

// }

// pub fn wall_strength(board: &mut BoardState) -> f64 {
//     let mut strength = 0.0;

//     for pos in 0..36 {
//         let piece = board.data[pos];

//         if piece == 2 {
//             let ne_piece = if !on_top_edge(pos) && !on_right_edge(pos) { board.data[pos + 7] } else { NULL };
//             let e_piece = if !on_right_edge(pos) { board.data[pos + 1] } else { NULL };
//             let se_piece = if !on_right_edge(pos) && !on_bottom_edge(pos){ board.data[pos - 5] } else { NULL };
            
//             let sw_piece = if !on_bottom_edge(pos) && !on_left_edge(pos) { board.data[pos - 7] } else { NULL };
//             let w_piece = if !on_left_edge(pos) { board.data[pos - 1] } else { NULL };
//             let nw_piece =  if !on_left_edge(pos) && !on_top_edge(pos) { board.data[pos + 5] } else { NULL };
        
//             if ne_piece == 2 {
//                 strength += 1.0;

//             } 
//             if se_piece == 2 {
//                 strength += 1.0;

//             } 
//             if e_piece == 2 {
//                 strength += 2.0;

//             }
            
//             if nw_piece == 2 {
//                 strength += 1.0;

//             } 
//             if sw_piece == 2 {
//                 strength += 1.0;

//             } 
//             if w_piece == 2 {
//                 strength += 2.0;

//             }

//         }

//     }

//     strength

// }

// pub fn p1_wall_score(board: &mut BoardState) -> f64 {
//     let mut protected_pieces_score: f64 = 0.0;

//     for pos in 0..36 {
//         let piece = board.data[pos];

//         if piece == 2 {
//             let mut down_pieces = vec![];
//             let mut current_pos_clone = pos;
//             if in_bounds(current_pos_clone - 6)  {
//                 current_pos_clone -= 6;

//                 while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0 {
//                     down_pieces.push(board.data[current_pos_clone]);
//                     current_pos_clone -= 6;

//                 }

//             }

//             for d_piece in down_pieces {
//                 protected_pieces_score += PROTECTED_PIECE_SCORES[d_piece - 1];

//             }

//         }

//     }

//     let wall_offset = wall_depth_offset(board);
//     let offset_score = HALF_OFFSET_SCORE * wall_offset;

//     protected_pieces_score + offset_score
    
// }

// pub fn p2_wall_score(board: &mut BoardState) -> f64 {
//     let mut protected_pieces_score: f64 = 0.0;

//     for pos in 0..36 {
//         let piece = board.data[pos];

//         if piece == 2 {
//             let mut up_pieces = vec![];
//             let mut current_pos_clone = pos;
//             if in_bounds(current_pos_clone + 6) {
//                 current_pos_clone += 6;

//                 while in_bounds(current_pos_clone) && board.data[current_pos_clone] != 0 {
//                     up_pieces.push(board.data[current_pos_clone]);
//                     current_pos_clone += 6;

//                 }

//             }
            
//             for u_piece in up_pieces {
//                 protected_pieces_score += PROTECTED_PIECE_SCORES[u_piece - 1];

//             }

//         }

//     }

//     let wall_offset = wall_depth_offset(board);
//     let offset_score = HALF_OFFSET_SCORE * -wall_offset;

//     protected_pieces_score + offset_score
    
// }

// pub fn unique_controlled_pieces_score_2(board: &mut BoardState, player: f64) -> f64 {
//     let pieces = unsafe{ controlled_pieces(board, player) };
//     let opp_pieces = unsafe{ controlled_pieces(board, -player) };
    
//     let mut unique_controlled_pieces = pieces & !opp_pieces;

//     let positions = unique_controlled_pieces.get_data();

//     let mut score = 0.0;
//     for pos in positions {
//         let piece = board.data[pos];
//         score += UNIQUE_PIECE_CONTROL_SCORES[piece - 1];

//     }

//     score

// }

// pub fn shared_controlled_pieces_score_2(board: &mut BoardState, player: f64) -> f64 {
//     let mut pieces = unsafe{ controlled_pieces(board, player) };

//     let positions = pieces.get_data();

//     let mut score = 0.0;
//     for pos in positions {
//         let piece = board.data[pos];
//         score += SHARED_PIECE_CONTROL_SCORES[piece - 1];

//     }

//     score

// }

// pub fn unique_controlled_squares_score_2(board: &mut BoardState, player: f64) -> f64 {
//     let squares = unsafe{ controlled_squares(board, player) };
//     let opp_squares = unsafe{ controlled_squares(board, -player) };
    
//     let unique_squares = squares & !opp_squares;

//     unique_squares.pop_count() as f64 * UNIQUE_SQUARE_CONTROL_SCORE

// }

// pub fn shared_controlled_squares_score_2(board: &mut BoardState, player: f64) -> f64 {
//     let squares = unsafe{ controlled_squares(board, player) };
//     squares.pop_count() as f64 * SHARED_SQUARE_CONTROL_SCORE

// }

// pub fn mobility_eval_2(board: &mut BoardState, player: f64) -> f64 {
//     let mut eval = 0.0;
 
//     eval += unsafe{ valid_move_count(board, player) } as f64;

//     eval

// }

// pub fn control_eval_2(board: &mut BoardState, player: f64) -> f64 {
//     let mut eval = 0.0;

//     eval +=  unique_controlled_pieces_score_2(board, player);
//     eval +=  unique_controlled_squares_score_2(board, player);

//     eval +=  shared_controlled_pieces_score_2(board, player);
//     eval +=  shared_controlled_squares_score_2(board, player);

//     eval

// }

// pub fn ones_eval(board: &mut BoardState, player: f64) -> f64 {
//     let mut eval = 0.0;

//     eval += ones_connectivity_score(board, player);

//     eval

// }

// // Experimental Evaluation Function
// pub fn get_evalulation_exp(board: &mut BoardState) -> f64 {
//     let mut eval = 0.0;

//     eval += mobility_eval_2(board, Player::One) - mobility_eval_2(board, Player::Two);
//     eval += (control_eval_2(board, Player::One) - control_eval_2(board, Player::Two)) * 3.0;
//     eval += (p1_wall_score(board) - p2_wall_score(board)) * (wall_strength(board) / 4.0);
//     eval += ones_eval(board, Player::One) - ones_eval(board, Player::Two);

//     eval

// }

// BEST EVALUATION FUNCTION
pub fn get_evalulation(board: &mut BoardState) -> f64 {
    let move_counts = [unsafe{ valid_move_count(board, Player::One) }, unsafe{ valid_move_count(board, Player::Two) }];
    let control_squares = [unsafe{ controlled_squares(board, Player::One) }, unsafe{ controlled_squares(board, Player::Two) } ];
    let control_pieces = [unsafe{ controlled_pieces(board, Player::One) }, unsafe{ controlled_pieces(board, Player::Two) } ];

    let mut eval = 0.0;

    eval += mobility_eval(Player::One, move_counts) - mobility_eval(Player::Two, move_counts);
    eval += (control_eval(board, Player::One, control_pieces, control_squares) - control_eval(board, Player::Two, control_pieces, control_squares)) * 3.0;

    eval

}

pub fn unique_controlled_pieces_score(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2]) -> f64 {
    let pieces = if player == Player::One { control_pieces[0] } else { control_pieces[1] };
    let opp_pieces = if player == Player::One { control_pieces[1] } else { control_pieces[0] };
    
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
    let mut pieces = if player == Player::One { control_pieces[0] } else { control_pieces[1] };

    let positions = pieces.get_data();

    let mut score = 0.0;
    for pos in positions {
        let piece = board.data[pos];
        score += SHARED_PIECE_CONTROL_SCORES[piece as usize];

    }

    score

}

pub fn unique_controlled_squares_score(player: Player, control_squares: [BitBoard; 2]) -> f64 {

    let squares = if player == Player::One { control_squares[0] } else { control_squares[1] };
    let opp_squares = if player == Player::One { control_squares[1] } else { control_squares[0] };
    
    let unique_squares = squares & !opp_squares;

    unique_squares.pop_count() as f64 * UNIQUE_SQUARE_CONTROL_SCORE

}

pub fn shared_controlled_squares_score(player: Player, control_squares: [BitBoard; 2]) -> f64 {
    let squares = if player == Player::One { control_squares[0] } else { control_squares[1] };
    squares.pop_count() as f64 * SHARED_SQUARE_CONTROL_SCORE

}

pub fn mobility_eval(player: Player, move_counts: [usize; 2]) -> f64 {
    let mut eval = 0.0;
 
    eval += if player == Player::One { move_counts[0] } else { move_counts[1] } as f64;

    eval

}

pub fn control_eval(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2], control_squares: [BitBoard; 2]) -> f64 {
    let mut eval = 0.0;
    
    eval +=  unique_controlled_pieces_score(board, player, control_pieces);
    eval +=  unique_controlled_squares_score(player, control_squares);

    eval +=  shared_controlled_pieces_score(board, player, control_pieces);
    eval +=  shared_controlled_squares_score(player, control_squares);

    eval

}
