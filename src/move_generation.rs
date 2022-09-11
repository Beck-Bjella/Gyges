use std::cmp::Ordering;

use crate::board::*;
use crate::bitboard::*;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Move(pub [usize; 6]);

pub fn sort_moves(mut evaluations: Vec<(Move, f64)>) -> Vec<Move> {
    evaluations.sort_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });


    let mut sorted_moves = vec![];

    for item in &evaluations {
        sorted_moves.push(item.0);
        
    }

    return sorted_moves;

}


pub fn valid_moves_2(board: &mut BoardState, player: i8) -> Vec<Move> {
    let active_lines = board.get_active_lines();

    let banned_positions: [bool; 36] = [false; 36];
    let backtrack_board = BitBoard(0);

    if player == 1 {
        let mut player_1_drops: Vec<usize> = board.get_drops(active_lines, 1);
        let mut player_1_moves: Vec<Move> = vec![];

        for x in 0..6 {
            if board.data[active_lines[0] + x] != 0 {
                let starting_piece: usize = active_lines[0] + x;
                let starting_piece_type: usize = board.data[starting_piece];

                player_1_drops.push(starting_piece);

                board.data[starting_piece] = 0;

                let mut piece_moves = get_piece_moves_2(board, backtrack_board, banned_positions, starting_piece, starting_piece_type, starting_piece, starting_piece_type, 1, &player_1_drops);
                player_1_moves.append(&mut piece_moves);

                board.data[starting_piece] = starting_piece_type;

                player_1_drops.pop();

            }

        }

        return player_1_moves;

    } else {
        let mut player_2_drops: Vec<usize> = board.get_drops(active_lines, 2);
        let mut player_2_moves: Vec<Move> = vec![];

        for x in 0..6 {
            if board.data[active_lines[1] + x] != 0 {
                let starting_piece: usize = active_lines[1] + x;
                let starting_piece_type: usize = board.data[starting_piece];

                player_2_drops.push(starting_piece);

                board.data[starting_piece] = 0;

                let mut piece_moves = get_piece_moves_2(board, backtrack_board, banned_positions, starting_piece, starting_piece_type, starting_piece, starting_piece_type,2, &player_2_drops);
                player_2_moves.append(&mut piece_moves);

                board.data[starting_piece] = starting_piece_type;

                player_2_drops.pop();

            }

        }

        return player_2_moves;

    }

}

fn get_piece_moves_2(board: &BoardState, mut backtrack_board: BitBoard, mut banned_positions: [bool; 36], current_piece: usize, current_piece_type: usize, starting_piece: usize, starting_piece_type: usize, player: i8, current_player_drops: &Vec<usize>) -> Vec<Move> {
    let mut final_moves: Vec<Move> = vec![];
    
    if current_piece_type == ONE_PIECE {
        for (path_idx, path) in board.one_moves[current_piece].iter().enumerate() {
            let end = path[1];
            let end_piece = board.data[end];

            let backtrack_path = board.one_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_moves.push(Move([0, starting_piece, starting_piece_type, PLAYER_1_GOAL, NULL, NULL]));
                continue;

            } else if end == PLAYER_2_GOAL {
                final_moves.push(Move([0, starting_piece, starting_piece_type, PLAYER_2_GOAL, NULL, NULL]));
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;

                    for drop_pos in current_player_drops.iter() {
                        final_moves.push(Move([0, starting_piece, starting_piece_type, end, end_piece, *drop_pos]));

                    }
                    
                    let mut moves = get_piece_moves_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player, current_player_drops);
                    final_moves.append(&mut moves);
                    
                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;

                }
                
            } else {
                final_moves.push(Move([0, starting_piece, starting_piece_type, end, NULL, NULL]));

            }

        }

    } if current_piece_type == TWO_PIECE {
        for (path_idx, path) in board.two_moves[current_piece].iter().enumerate() {
            let end = path[2];
            let end_piece = board.data[end];

            if board.data[path[1]] != 0 {
                continue;

            }

            let backtrack_path = board.two_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_moves.push(Move([0, starting_piece, starting_piece_type, PLAYER_1_GOAL, NULL, NULL]));
                continue;

            } else if end == PLAYER_2_GOAL {
                final_moves.push(Move([0, starting_piece, starting_piece_type, PLAYER_2_GOAL, NULL, NULL]));
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;
                    
                    for drop_pos in current_player_drops.iter() {
                        final_moves.push(Move([0, starting_piece, starting_piece_type, end, end_piece, *drop_pos]));

                    }
                    
                    let mut moves = get_piece_moves_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player, current_player_drops);
                    final_moves.append(&mut moves);
                    
                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;
            
                }
                
            } else {
                final_moves.push(Move([0, starting_piece, starting_piece_type, end, NULL, NULL]));

            }

        }

    } else if current_piece_type == THREE_PIECE {
        for (path_idx, path) in board.three_moves[current_piece].iter().enumerate() {
            let end = path[3];
            let end_piece = board.data[end];

            if board.data[path[1]] != 0 {
                continue;
                
            } else if board.data[path[2]] != 0 {
                continue;
                
            }
            
            let backtrack_path = board.three_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_moves.push(Move([0, starting_piece, starting_piece_type, PLAYER_1_GOAL, NULL, NULL]));
                continue;

            } else if end == PLAYER_2_GOAL {
                final_moves.push(Move([0, starting_piece, starting_piece_type, PLAYER_2_GOAL, NULL, NULL]));
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;

                    for drop_pos in current_player_drops.iter() {
                        final_moves.push(Move([0, starting_piece, starting_piece_type, end, end_piece, *drop_pos]));

                    }
                    
                    let mut moves = get_piece_moves_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player, current_player_drops);
                    final_moves.append(&mut moves);

                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;
            
                }
                
            } else {
                final_moves.push(Move([0, starting_piece, starting_piece_type, end, NULL, NULL]));

            }

        }

    }

    return final_moves;

}



pub fn valid_move_count_2(board: &mut BoardState, player: i8) -> usize {
    let active_lines = board.get_active_lines();

    let banned_positions: [bool; 36] = [false; 36];
    let backtrack_board = BitBoard(0);

    if player == 1 {
        let player_1_drop_count: usize = board.get_drops(active_lines, 1).len() + 1;
        let mut player_1_move_count: usize = 0;

        for x in 0..6 {
            if board.data[active_lines[0] + x] != 0 {
                let starting_piece: usize = active_lines[0] + x;
                let starting_piece_type: usize = board.data[starting_piece];

                board.data[starting_piece] = 0;

                let piece_move_count = get_piece_move_count_2(board, backtrack_board, banned_positions, starting_piece, starting_piece_type, starting_piece, starting_piece_type, 1, player_1_drop_count);
                player_1_move_count += piece_move_count;

                board.data[starting_piece] = starting_piece_type;

            }

        }

        return player_1_move_count;

    } else {
        let player_2_drop_count: usize = board.get_drops(active_lines, 2).len() + 1;
        let mut player_2_move_count: usize = 0;

        for x in 0..6 {
            if board.data[active_lines[1] + x] != 0 {
                let starting_piece: usize = active_lines[1] + x;
                let starting_piece_type: usize = board.data[starting_piece];

                board.data[starting_piece] = 0;

                let piece_move_count: usize = get_piece_move_count_2(board, backtrack_board, banned_positions, starting_piece, starting_piece_type, starting_piece, starting_piece_type,2, player_2_drop_count);
                player_2_move_count += piece_move_count;

                board.data[starting_piece] = starting_piece_type;

            }

        }

        return player_2_move_count;

    }

}

fn get_piece_move_count_2(board: &BoardState, mut backtrack_board: BitBoard, mut banned_positions: [bool; 36], current_piece: usize, current_piece_type: usize, starting_piece: usize, starting_piece_type: usize, player: i8, current_player_drop_count: usize) -> usize {
    let mut final_moves: usize = 0;
    
    if current_piece_type == ONE_PIECE {
        for (path_idx, path) in board.one_moves[current_piece].iter().enumerate() {
            let end = path[1];
            let end_piece = board.data[end];

            let backtrack_path = board.one_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_moves += 1;
                continue;

            } else if end == PLAYER_2_GOAL {
                final_moves += 1;
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;

                    final_moves += current_player_drop_count;
                    
                    let move_count: usize = get_piece_move_count_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player, current_player_drop_count);
                    final_moves += move_count;
                    
                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;

                }
                
            } else {
                final_moves += 1;

            }

        }

    } if current_piece_type == TWO_PIECE {
        for (path_idx, path) in board.two_moves[current_piece].iter().enumerate() {
            let end = path[2];
            let end_piece = board.data[end];

            if board.data[path[1]] != 0 {
                continue;

            }

            let backtrack_path = board.two_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_moves += 1;
                continue;

            } else if end == PLAYER_2_GOAL {
                final_moves += 1;
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;
                    
                    final_moves += current_player_drop_count;
                    
                    let move_count: usize = get_piece_move_count_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player, current_player_drop_count);
                    final_moves += move_count;
                    
                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;
            
                }
                
            } else {
                final_moves += 1;

            }

        }

    } else if current_piece_type == THREE_PIECE {
        for (path_idx, path) in board.three_moves[current_piece].iter().enumerate() {
            let end = path[3];
            let end_piece = board.data[end];

            if board.data[path[1]] != 0 {
                continue;
                
            } else if board.data[path[2]] != 0 {
                continue;
                
            }
            
            let backtrack_path = board.three_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_moves += 1;
                continue;

            } else if end == PLAYER_2_GOAL {
                final_moves += 1;
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;

                    final_moves += current_player_drop_count;
                    
                    let move_count: usize = get_piece_move_count_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player, current_player_drop_count);
                    final_moves += move_count;

                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;
            
                }
                
            } else {
                final_moves += 1;

            }

        }

    }

    return final_moves;

}


pub fn valid_threat_count_2(board: &mut BoardState, player: i8) -> usize {
    let active_lines = board.get_active_lines();

    let banned_positions: [bool; 36] = [false; 36];
    let backtrack_board = BitBoard(0);

    if player == 1 {
        let mut player_1_threat_count: usize = 0;

        for x in 0..6 {
            if board.data[active_lines[0] + x] != 0 {
                let starting_piece: usize = active_lines[0] + x;
                let starting_piece_type: usize = board.data[starting_piece];

                board.data[starting_piece] = 0;

                let piece_move_count = get_piece_threat_count_2(board, backtrack_board, banned_positions, starting_piece, starting_piece_type, starting_piece, starting_piece_type, 1);
                player_1_threat_count += piece_move_count;

                board.data[starting_piece] = starting_piece_type;

            }

        }

        return player_1_threat_count;

    } else {
        let mut player_2_threat_count: usize = 0;

        for x in 0..6 {
            if board.data[active_lines[1] + x] != 0 {
                let starting_piece: usize = active_lines[1] + x;
                let starting_piece_type: usize = board.data[starting_piece];

                board.data[starting_piece] = 0;

                let piece_move_count: usize = get_piece_threat_count_2(board, backtrack_board, banned_positions, starting_piece, starting_piece_type, starting_piece, starting_piece_type,2);
                player_2_threat_count += piece_move_count;

                board.data[starting_piece] = starting_piece_type;

            }

        }

        return player_2_threat_count;

    }

}

fn get_piece_threat_count_2(board: &BoardState, mut backtrack_board: BitBoard, mut banned_positions: [bool; 36], current_piece: usize, current_piece_type: usize, starting_piece: usize, starting_piece_type: usize, player: i8) -> usize {
    let mut final_threat_count: usize = 0;
    
    if current_piece_type == ONE_PIECE {
        for (path_idx, path) in board.one_moves[current_piece].iter().enumerate() {
            let end = path[1];
            let end_piece = board.data[end];

            let backtrack_path = board.one_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_threat_count += 1;
                continue;

            } else if end == PLAYER_2_GOAL {
                final_threat_count += 1;
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;
                    
                    let threat_count: usize = get_piece_threat_count_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player);
                    final_threat_count += threat_count;
                    
                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;

                }
                
            }

        }

    } if current_piece_type == TWO_PIECE {
        for (path_idx, path) in board.two_moves[current_piece].iter().enumerate() {
            let end = path[2];
            let end_piece = board.data[end];

            if board.data[path[1]] != 0 {
                continue;

            }

            let backtrack_path = board.two_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_threat_count += 1;
                continue;

            } else if end == PLAYER_2_GOAL {
                final_threat_count += 1;
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;
                    
                    let threat_count: usize = get_piece_threat_count_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player);
                    final_threat_count += threat_count;
                    
                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;
            
                }
                
            }

        }

    } else if current_piece_type == THREE_PIECE {
        for (path_idx, path) in board.three_moves[current_piece].iter().enumerate() {
            let end = path[3];
            let end_piece = board.data[end];

            if board.data[path[1]] != 0 {
                continue;
                
            } else if board.data[path[2]] != 0 {
                continue;
                
            }
            
            let backtrack_path = board.three_path_backtrack_checks[current_piece][path_idx];
            if (backtrack_board & backtrack_path).is_not_empty() {
                continue;

            }

            if end == PLAYER_2_GOAL && player == 2 || end == PLAYER_1_GOAL && player == 1 {
                continue;

            } 

            if end == PLAYER_1_GOAL {
                final_threat_count += 1;
                continue;

            } else if end == PLAYER_2_GOAL {
                final_threat_count += 1;
                continue;

            }

            if board.data[end] != 0 {
                if !banned_positions[end] {
                    banned_positions[end] = true;

                    backtrack_board ^= backtrack_path;

                    let threat_count: usize = get_piece_threat_count_2(board, backtrack_board, banned_positions, end, end_piece, starting_piece, starting_piece_type, player);
                    final_threat_count += threat_count;

                    banned_positions[end] = false;

                    backtrack_board ^= backtrack_path;
            
                }
                
            }

        }

    }

    return final_threat_count;

}

