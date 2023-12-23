extern crate test;

use crate::board::board::*;
use crate::board::bitboard::*;
use crate::moves::move_list::*;
use crate::consts::*;

use super::moves::Move;
use super::moves::MoveType;

#[derive(PartialEq)]
enum Action {
    Gen,
    Start,
    End

}

type StackData = (Action, BitBoard, BitBoard, usize, usize, usize, usize, usize, f64);

static mut STACK_BUFFER: Vec<StackData> = Vec::new();

pub unsafe fn valid_moves(board: &mut BoardState, player: f64) -> RawMoveList {
    let active_lines = board.get_active_lines();
    let mut move_list: RawMoveList = RawMoveList::new(board.get_drops(active_lines, player));

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6

    };

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

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

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
            board.piece_bb ^= 1 << starting_piece;
            continue;

        } else if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            board.piece_bb ^= 1 << starting_piece;
            continue;
            
        } else if current_piece_type == ONE_PIECE {
            let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                let path_idx = valid_paths[i as usize];
                let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[1] as usize;
                let end_bit = 1 << end;
 
                if (move_list.end_positions[active_line_idx] & end_bit).is_not_empty() {
                    continue;

                }

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        move_list.set_pickup_position(active_line_idx, end);
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    move_list.set_end_position(active_line_idx, end);
    
                }
    
            }

        } else if current_piece_type == TWO_PIECE {
            let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_piece];

            let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                let path_idx = valid_paths[i as usize];
                let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[2] as usize;
                let end_bit = 1 << end;

                if (move_list.end_positions[active_line_idx] & end_bit).is_not_empty() {
                    continue;

                }

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        move_list.set_pickup_position(active_line_idx, end);
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    move_list.set_end_position(active_line_idx, end);
    
                }
    
            }
           
        } else if current_piece_type == THREE_PIECE {
            let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_piece];

            let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                let path_idx = valid_paths[i as usize];
                let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[3] as usize;
                let end_bit = 1 << end;

                if (move_list.end_positions[active_line_idx] & end_bit).is_not_empty() {
                    continue;

                }

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        move_list.set_pickup_position(active_line_idx, end);
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    move_list.set_end_position(active_line_idx, end);
    
                }
    
            }

        }

    }

    move_list

}

pub unsafe fn valid_move_count(board: &mut BoardState, player: f64) -> usize {
    let active_lines: [usize; 2] = board.get_active_lines();
    let mut count: usize = 0;

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6
        
    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

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
            board.piece_bb ^= 1 << starting_piece;
            continue;

        } else if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            board.piece_bb ^= 1 << starting_piece;
            continue;
            
        } else if current_piece_type == ONE_PIECE {
            let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(ONE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_ONE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[1] as usize;
                let end_bit = 1 << end;

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        count += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    count += 1;
    
                }
    
            }

        } else if current_piece_type == TWO_PIECE {
            let intercept_bb = board.piece_bb & *ALL_TWO_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(TWO_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_TWO_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[2] as usize;
                let end_bit = 1 << end;

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        count += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    count += 1;
    
                }
    
            }
           
        } else if current_piece_type == THREE_PIECE {
            let intercept_bb = board.piece_bb & *ALL_THREE_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(THREE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_THREE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[3] as usize;
                let end_bit = 1 << end;

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        count += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    count += 1;
    
                }
    
            }

        }

    }

    count

}

pub unsafe fn valid_threat_count(board: &mut BoardState, player: f64) -> usize {
    let active_lines: [usize; 2] = board.get_active_lines();
    let mut count: usize = 0;

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6
        
    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

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
            board.piece_bb ^= 1 << starting_piece;
            continue;

        } else if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            board.piece_bb ^= 1 << starting_piece;
            continue;
            
        } else if current_piece_type == ONE_PIECE {
            let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(ONE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_ONE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[1] as usize;
                let end_bit = 1 << end;

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                    
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
                    
                    }

                }
    
            }

        } else if current_piece_type == TWO_PIECE {
            let intercept_bb = board.piece_bb & *ALL_TWO_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(TWO_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_TWO_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[2] as usize;
                let end_bit = 1 << end;

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
                    
                    }

                }
    
            }
           
        } else if current_piece_type == THREE_PIECE {
            let intercept_bb = board.piece_bb & *ALL_THREE_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(THREE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_THREE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[3] as usize;
                let end_bit = 1 << end;

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
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                    
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
                    
                    }

                }
    
            }

        }

    }

    count

}

pub unsafe fn controlled_pieces(board: &mut BoardState, player: f64) -> BitBoard {
    let active_lines: [usize; 2] = board.get_active_lines();
    let mut controlled_pieces = BitBoard(0);

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6
        
    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            controlled_pieces |= 1 << active_line + x; 

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

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
            board.piece_bb ^= 1 << starting_piece;
            continue;

        } else if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            board.piece_bb ^= 1 << starting_piece;
            continue;
            
        } else if current_piece_type == ONE_PIECE {
            let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(ONE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_ONE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[1] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        controlled_pieces |= end_bit;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                }
    
            }

        } else if current_piece_type == TWO_PIECE {
            let intercept_bb = board.piece_bb & *ALL_TWO_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(TWO_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_TWO_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[2] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        controlled_pieces |= end_bit;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                }
    
            }
           
        } else if current_piece_type == THREE_PIECE {
            let intercept_bb = board.piece_bb & *ALL_THREE_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(THREE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_THREE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[3] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        controlled_pieces |= end_bit;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                }
    
            }

        }

    }

    controlled_pieces

}

pub unsafe fn controlled_squares(board: &mut BoardState, player: f64) -> BitBoard {
    let active_lines: [usize; 2] = board.get_active_lines();
    let mut controlled_squares = BitBoard(0);

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6
        
    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

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
            board.piece_bb ^= 1 << starting_piece;
            continue;

        } else if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            board.piece_bb ^= 1 << starting_piece;
            continue;
            
        } else if current_piece_type == ONE_PIECE {
            let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(ONE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_ONE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[1] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    controlled_squares |= end_bit;

                }
    
            }

        } else if current_piece_type == TWO_PIECE {
            let intercept_bb = board.piece_bb & *ALL_TWO_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(TWO_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_TWO_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[2] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    controlled_squares |= end_bit;

                }
    
            }
           
        } else if current_piece_type == THREE_PIECE {
            let intercept_bb = board.piece_bb & *ALL_THREE_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(THREE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_THREE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[3] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;

                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    controlled_squares |= end_bit;

                }
    
            }

        }

    }

    controlled_squares & !board.piece_bb

}

pub unsafe fn has_threat(board: &mut BoardState, player: f64) -> bool {
    let active_lines: [usize; 2] = board.get_active_lines();

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6
        
    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            STACK_BUFFER.push((Action::End, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));
            STACK_BUFFER.push((Action::Gen, BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard(0), BitBoard(0), 0, 0, starting_piece, starting_piece_type, 0, 0.0));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

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
            board.piece_bb ^= 1 << starting_piece;
            continue;

        } else if action == Action::End {
            board.data[starting_piece] = starting_piece_type;
            board.piece_bb ^= 1 << starting_piece;
            continue;
            
        } else if current_piece_type == ONE_PIECE {
            let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(ONE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_ONE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[1] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    STACK_BUFFER.clear();
                    return true;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    STACK_BUFFER.clear();
                    return true;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                    
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
                    
                    }

                }
    
            }

        } else if current_piece_type == TWO_PIECE {
            let intercept_bb = board.piece_bb & *ALL_TWO_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(TWO_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_TWO_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[2] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    STACK_BUFFER.clear();
                    return true;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    STACK_BUFFER.clear();
                    return true;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
                    
                    }

                }
    
            }
           
        } else if current_piece_type == THREE_PIECE {
            let intercept_bb = board.piece_bb & *ALL_THREE_INTERCEPTS.get_unchecked(current_piece);

            let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..*valid_paths.get_unchecked(THREE_PATH_COUNT_IDX) {
                let path_idx = valid_paths.get_unchecked(i as usize);
                let path = UNIQUE_THREE_PATHS.get_unchecked(*path_idx as usize);

                if (backtrack_board & path.1).is_not_empty() {
                    continue;
    
                }

                let end = path.0[3] as usize;
                let end_bit = 1 << end;

                if end == PLAYER_1_GOAL {
                    if player == PLAYER_1 {
                        continue;
                    }
                    STACK_BUFFER.clear();
                    return true;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    STACK_BUFFER.clear();
                    return true;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                    
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
                    
                    }

                }
    
            }

        }

    }

    false

}

// NEW FORMAT 
#[derive(PartialEq)]
enum NewAction {
    Gen(BitBoard, BitBoard, usize, usize, usize, usize, usize, f64),
    Start(usize),
    End(usize, usize)

}

static mut NEW_STACK_BUFFER: Vec<NewAction> = vec![];


pub unsafe fn new_valid_moves(board: &mut BoardState, player: f64) -> Vec<Move> {
    let active_lines = board.get_active_lines();
    let drops = board.get_drops(active_lines, player).get_data();

    let mut moves = Vec::with_capacity(1000);

    let mut end_pieces = BitBoard(0);

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6

    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            NEW_STACK_BUFFER.push(NewAction::End(starting_piece, starting_piece_type));
            NEW_STACK_BUFFER.push(NewAction::Gen(BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            NEW_STACK_BUFFER.push(NewAction::Start(starting_piece));

        }

    }

    loop {
        if NEW_STACK_BUFFER.is_empty() {
            break;
        }
        let action = NEW_STACK_BUFFER.pop().unwrap_unchecked();

        match action {
            NewAction::Start(starting_piece) => {
                board.data[starting_piece] = 0;
                board.piece_bb ^= 1 << starting_piece;
                end_pieces = BitBoard(0);

            },
            NewAction::End(starting_piece, starting_piece_type) => {
                board.data[starting_piece] = starting_piece_type;
                board.piece_bb ^= 1 << starting_piece;

            },
            NewAction::Gen(backtrack_board, banned_positions, current_piece, current_piece_type, starting_piece, starting_piece_type, active_line_idx, player) => {
                match current_piece_type {
                    1 => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = path.0[1] as usize;
                            let end_bit = 1 << end;
                            
                            if (end_pieces & end_bit).is_not_empty() {
                                continue;
            
                            }
            
                            if end == PLAYER_1_GOAL {
                                if player == PLAYER_1 {
                                    continue;
                                }
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                continue;
                
                            } else if end == PLAYER_2_GOAL {
                                if player == PLAYER_2 {
                                    continue;
                                }
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                continue;
                
                            }
                            
                            let end_piece = board.data[end];
                            if end_piece != 0 {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    for drop in &drops {
                                        moves.push(Move::new([0, starting_piece, starting_piece_type, end, end_piece, *drop], MoveType::Drop));
                                        continue;

                                    }
                                    
                                    NEW_STACK_BUFFER.push(NewAction::Gen(new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
            
                                }
                                
                            } else {
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                end_pieces |= end_bit;

                            }
                
                        }

                    },
                    2 => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_piece];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = path.0[2] as usize;
                            let end_bit = 1 << end;

                            if (end_pieces & end_bit).is_not_empty() {
                                continue;
            
                            }

                            if end == PLAYER_1_GOAL {
                                if player == PLAYER_1 {
                                    continue;
                                }
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                continue;
                
                            } else if end == PLAYER_2_GOAL {
                                if player == PLAYER_2 {
                                    continue;
                                }
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                continue;
                
                            }
                            
                            let end_piece = board.data[end];
                            if end_piece != 0 {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    for drop in &drops {
                                        moves.push(Move::new([0, starting_piece, starting_piece_type, end, end_piece, *drop], MoveType::Drop));
                                        continue;

                                    }
                                    
                                    NEW_STACK_BUFFER.push(NewAction::Gen(new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                                }
                                
                            } else {
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                end_pieces |= end_bit;

                            }
                
                        }
                    
                    },
                    3 => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_piece];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = path.0[3] as usize;
                            let end_bit = 1 << end;

                            if (end_pieces & end_bit).is_not_empty() {
                                continue;
            
                            }

                            if end == PLAYER_1_GOAL {
                                if player == PLAYER_1 {
                                    continue;
                                }
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                continue;
                
                            } else if end == PLAYER_2_GOAL {
                                if player == PLAYER_2 {
                                    continue;
                                }
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                continue;
                
                            }
                            
                            let end_piece = board.data[end];
                            if end_piece != 0 {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    for drop in &drops {
                                        moves.push(Move::new([0, starting_piece, starting_piece_type, end, end_piece, *drop], MoveType::Drop));
                                        continue;

                                    }
                                    
                                    NEW_STACK_BUFFER.push(NewAction::Gen(new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                                }
                                
                            } else {
                                moves.push(Move::new([0, starting_piece, starting_piece_type, end, NULL, NULL], MoveType::Bounce));
                                end_pieces |= end_bit;

                            }
                
                        }
                    
                    },
                    _ => {}

                }

            }
            
        }

    }

    moves

}

pub unsafe fn new_valid_move_count(board: &mut BoardState, player: f64) -> usize {
    let active_lines = board.get_active_lines();

    let mut count = 0;

    let active_line: usize = if player == PLAYER_1 {
        active_lines[0] * 6

    } else {
        active_lines[1] * 6

    };

    for x in 0..6 {
        if board.data[active_line + x] != 0 {
            let starting_piece: usize = active_line + x;
            let starting_piece_type: usize = board.data[starting_piece];

            NEW_STACK_BUFFER.push(NewAction::End(starting_piece, starting_piece_type));
            NEW_STACK_BUFFER.push(NewAction::Gen(BitBoard(0), BitBoard(0), starting_piece, starting_piece_type, starting_piece, starting_piece_type, x, player));
            NEW_STACK_BUFFER.push(NewAction::Start(starting_piece));

        }

    }

    loop {
        if NEW_STACK_BUFFER.is_empty() {
            break;
        }
        let action = NEW_STACK_BUFFER.pop().unwrap_unchecked();

        match action {
            NewAction::Start(starting_piece) => {
                board.data[starting_piece] = 0;
                board.piece_bb ^= 1 << starting_piece;

            },
            NewAction::End(starting_piece, starting_piece_type) => {
                board.data[starting_piece] = starting_piece_type;
                board.piece_bb ^= 1 << starting_piece;

            },
            NewAction::Gen(backtrack_board, banned_positions, current_piece, current_piece_type, starting_piece, starting_piece_type, active_line_idx, player) => {
                match current_piece_type {
                    1 => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_piece).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = path.0[1] as usize;
                            let end_bit = 1 << end;
                            
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
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    count += 25;
                                    
                                    NEW_STACK_BUFFER.push(NewAction::Gen(new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));
            
                                }
                                
                            } else {
                                count += 1;

                            }
                
                        }

                    },
                    2 => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_piece];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = path.0[2] as usize;
                            let end_bit = 1 << end;

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
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    count += 25;
                                    
                                    NEW_STACK_BUFFER.push(NewAction::Gen(new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                                }
                                
                            } else {
                                count += 1;

                            }
                
                        }
                    
                    },
                    3 => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_piece];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_piece).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = path.0[3] as usize;
                            let end_bit = 1 << end;

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
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    count += 25;
                                    
                                    NEW_STACK_BUFFER.push(NewAction::Gen(new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                                }
                                
                            } else {
                                count += 1;
  
                            }
                
                        }
                    
                    },
                    _ => {}

                }

            }
            
        }

    }

    count

}


#[cfg(test)]
mod move_gen_bench {
    use test::Bencher;
    
    use crate::consts::*;
    use crate::moves::move_gen::*;

    #[bench]
    fn bench_valid_moves(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ valid_moves(&mut board, PLAYER_1) }.moves(&board));

    }

    #[bench]
    fn bench_new_valid_moves(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ new_valid_moves(&mut board, PLAYER_1) });

    }

    #[bench]
    fn bench_valid_move_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ valid_move_count(&mut board, PLAYER_1) });

    }

    #[bench]
    fn bench_new_valid_move_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ new_valid_move_count(&mut board, PLAYER_1) });

    }


    #[bench]
    fn bench_threat_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ valid_threat_count(&mut board, PLAYER_1) });

    }

    #[bench]
    fn bench_has_threat(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ has_threat(&mut board, PLAYER_1) });

    }

}
