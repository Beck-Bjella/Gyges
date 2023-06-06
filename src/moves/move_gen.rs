use std::path;

use crate::board::board::*;
use crate::board::bitboard::*;
use crate::moves::move_list::*;
use crate::consts::*;

#[derive(PartialEq)]
enum Action {
    Gen,
    Start,
    End

}

static mut STACK_BUFFER: Vec<(Action, BitBoard, BitBoard, usize, usize, usize, usize, usize, f64)> = Vec::new();

pub unsafe fn valid_moves(board: &mut BoardState, player: f64) -> RawMoveList {
    let active_lines = board.get_active_lines();
    let mut move_list: RawMoveList = RawMoveList::new(board.get_drops(active_lines, player));

    let active_line: usize;
    if player == PLAYER_1 {
        active_line = active_lines[0] * 6;

    } else {
        active_line = active_lines[1] * 6;

    }

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            move_list.add_start_index(x);
            move_list.set_start(x, starting_piece, starting_piece_type);

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    while let Some(data) = STACK_BUFFER.pop() {
        let action = data.0;
        let backtrack_board: BitBoard = data.1;
        let banned_positions: BitBoard = data.2;
        let current_piece: usize = data.3;
        let current_piece_type: usize = data.4;
        let starting_piece: usize = data.5;
        let starting_piece_type: usize = data.6;
        let active_line_idx: usize = data.7;
        let player: f64 = data.8;

        if action == Action::Start {
            board.data[starting_piece] = 0;
            continue;

        }
        
        if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            continue;
        }

        if current_piece_type == ONE_PIECE {
            for path_idx in 0..ONE_PATH_LEGNTHS[current_piece] {
                let path = ONE_PATHS[current_piece][path_idx];

                let backtrack_path = ONE_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;

                }

                let end = path[1];
   
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;

                    }
                    move_list.set_end_position(active_line_idx, PLAYER_1_GOAL);
                    continue;

                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;

                    }
                    move_list.set_end_position(active_line_idx, PLAYER_2_GOAL);
                    continue;

                }

                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        move_list.set_pickup_position(active_line_idx, end);
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    move_list.set_end_position(active_line_idx, end);

                }

            }

        } else if current_piece_type == TWO_PIECE {
            for path_idx in 0..TWO_PATH_LENGTHS[current_piece] {
                let path = TWO_PATHS[current_piece][path_idx];

                let backtrack_path = TWO_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;
    
                }
    
                if board.data[path[1]] != 0 {
                    continue;
    
                }
    
                let end = path[2];
    
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    move_list.set_end_position(active_line_idx, PLAYER_1_GOAL);
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    move_list.set_end_position(active_line_idx, PLAYER_2_GOAL);
                    continue;
    
                }

                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        move_list.set_pickup_position(active_line_idx, end);
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    move_list.set_end_position(active_line_idx, end);
    
                }
    
            }

        } else if current_piece_type == THREE_PIECE {
            for path_idx in 0..THREE_PATH_LENGTHS[current_piece] {
                let path = THREE_PATHS[current_piece][path_idx];

                let backtrack_path = THREE_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;
    
                }

                if board.data[path[1]] != 0 {
                    continue;
                    
                } else if board.data[path[2]] != 0 {
                    continue;
                    
                }
                
                let end = path[3];
               
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    move_list.set_end_position(active_line_idx, PLAYER_1_GOAL);
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    move_list.set_end_position(active_line_idx, PLAYER_2_GOAL);
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        move_list.set_pickup_position(active_line_idx, end);
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    move_list.set_end_position(active_line_idx, end);
    
                }
    
            }

        }

    }

    return move_list;

}


pub unsafe fn valid_move_count(board: &mut BoardState, player: f64) -> usize {
    let mut count = 0;

    let active_lines = board.get_active_lines();

    let active_line: usize;
    if player == PLAYER_1 {
        active_line = active_lines[0] * 6;

    } else {
        active_line = active_lines[1] * 6;

    }

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    while let Some(data) = STACK_BUFFER.pop() {
        let action = data.0;
        let backtrack_board: BitBoard = data.1;
        let banned_positions: BitBoard = data.2;
        let current_piece: usize = data.3;
        let current_piece_type: usize = data.4;
        let starting_piece: usize = data.5;
        let starting_piece_type: usize = data.6;
        let active_line_idx: usize = data.7;
        let player: f64 = data.8;

        if action == Action::Start {
            board.data[starting_piece] = 0;
            continue;

        }
        
        if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            continue;
        }

        if current_piece_type == ONE_PIECE {
            for path_idx in 0..ONE_PATH_LEGNTHS[current_piece] {
                let path = ONE_PATHS[current_piece][path_idx];

                let backtrack_path = ONE_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;

                }

                let end = path[1];
   
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;

                    }
                    count += 1;
                    continue;

                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;

                    }
                    count += 1;
                    continue;

                }

                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        count += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    count += 1;

                }

            }

        } else if current_piece_type == TWO_PIECE {
            for path_idx in 0..TWO_PATH_LENGTHS[current_piece] {
                let path = TWO_PATHS[current_piece][path_idx];

                let backtrack_path = TWO_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;
    
                }
    
                if board.data[path[1]] != 0 {
                    continue;
    
                }
    
                let end = path[2];
    
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                }

                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        count += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    count += 1;
    
                }
    
            }

        } else if current_piece_type == THREE_PIECE {
            for path_idx in 0..THREE_PATH_LENGTHS[current_piece] {
                let path = THREE_PATHS[current_piece][path_idx];

                let backtrack_path = THREE_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;
    
                }

                if board.data[path[1]] != 0 {
                    continue;
                    
                } else if board.data[path[2]] != 0 {
                    continue;
                    
                }
                
                let end = path[3];
               
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        count += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    count += 1;
    
                }
    
            }

        }

    }

    return count;

}


pub unsafe fn valid_threat_count(board: &mut BoardState, player: f64) -> usize {
    let mut count = 0;

    let active_lines = board.get_active_lines();

    let active_line: usize;
    if player == PLAYER_1 {
        active_line = active_lines[0] * 6;

    } else {
        active_line = active_lines[1] * 6;

    }

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    while let Some(data) = STACK_BUFFER.pop() {
        let action = data.0;
        let backtrack_board: BitBoard = data.1;
        let banned_positions: BitBoard = data.2;
        let current_piece: usize = data.3;
        let current_piece_type: usize = data.4;
        let starting_piece: usize = data.5;
        let starting_piece_type: usize = data.6;
        let active_line_idx: usize = data.7;
        let player: f64 = data.8;

        if action == Action::Start {
            board.data[starting_piece] = 0;
            continue;

        }
        
        if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            continue;
        }

        if current_piece_type == ONE_PIECE {
            for path_idx in 0..ONE_PATH_LEGNTHS[current_piece] {
                let path = ONE_PATHS[current_piece][path_idx];

                let backtrack_path = ONE_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;

                }

                let end = path[1];
   
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;

                    }
                    count += 1;
                    continue;

                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;

                    }
                    count += 1;
                    continue;

                }

                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                }

            }

        } else if current_piece_type == TWO_PIECE {
            for path_idx in 0..TWO_PATH_LENGTHS[current_piece] {
                let path = TWO_PATHS[current_piece][path_idx];

                let backtrack_path = TWO_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;
    
                }
    
                if board.data[path[1]] != 0 {
                    continue;
    
                }
    
                let end = path[2];
    
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                }

                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                }
    
            }

        } else if current_piece_type == THREE_PIECE {
            for path_idx in 0..THREE_PATH_LENGTHS[current_piece] {
                let path = THREE_PATHS[current_piece][path_idx];

                let backtrack_path = THREE_PATH_BACKTRACK_CHECKS[current_piece][path_idx];
                if (backtrack_board & backtrack_path).is_not_empty() {
                    continue;
    
                }

                if board.data[path[1]] != 0 {
                    continue;
                    
                } else if board.data[path[2]] != 0 {
                    continue;
                    
                }
                
                let end = path[3];
               
                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    count += 1;
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    let end_bit = 1 << end;
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ backtrack_path;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                }
    
            }

        }

    }

    return count;

}
