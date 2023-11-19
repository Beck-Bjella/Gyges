extern crate test;

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

pub unsafe fn per_piece_move_counts(board: &mut BoardState, player: f64) -> [usize; 6] {
    let active_lines: [usize; 2] = board.get_active_lines();
    let mut counts: [usize; 6] = [0; 6];

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
                    counts[active_line_idx] += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    counts[active_line_idx] += 1;
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        counts[active_line_idx] += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    counts[active_line_idx] += 1;
    
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
                    counts[active_line_idx] += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    counts[active_line_idx] += 1;
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        counts[active_line_idx] += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    counts[active_line_idx] += 1;
    
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
                    counts[active_line_idx] += 1;
                    continue;
    
                } else if end == PLAYER_2_GOAL {
                    if player == PLAYER_2 {
                        continue;
                    }
                    counts[active_line_idx] += 1;
                    continue;
    
                }
                
                let end_piece = board.data[end];
                if end_piece != 0 {
                    if (banned_positions & end_bit).is_empty() {
                        let new_banned_positions = banned_positions ^ end_bit;
                        let new_backtrack_board = backtrack_board ^ path.1;
                        
                        counts[active_line_idx] += 25;
                        
                        STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_piece, starting_piece_type, active_line_idx, player));

                    }
                    
                } else {
                    counts[active_line_idx] += 1;
    
                }
    
            }

        }

    }

    counts

}


#[cfg(test)]
mod move_gen_bench {
    use test::Bencher;

    use crate::moves::move_gen::*;

    #[bench]
    fn bench_valid_moves(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD, PLAYER_1);

        b.iter(|| unsafe{ valid_moves(&mut board, PLAYER_1) });

    }
    #[bench]
    fn bench_valid_move_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD, PLAYER_1);

        b.iter(|| unsafe{ valid_move_count(&mut board, PLAYER_1) });

    }

    #[bench]
    fn bench_threat_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD, PLAYER_1);

        b.iter(|| unsafe{ valid_threat_count(&mut board, PLAYER_1) });

    }

}
