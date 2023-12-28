extern crate test;

use crate::board::board::*;
use crate::board::bitboard::*;
use crate::core::piece::*;
use crate::core::player::*;
use crate::core::sq::*;
use crate::core::masks::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;

#[derive(PartialEq)]
enum Action {
    Gen,
    Start,
    End

}

type StackData = (Action, BitBoard, BitBoard, SQ, Piece, SQ, Piece, usize, Player);

static mut STACK_BUFFER: Vec<StackData> = Vec::new();

pub unsafe fn valid_moves(board: &mut BoardState, player: Player) -> RawMoveList {
    let active_lines = board.get_active_lines();
    let mut move_list: RawMoveList = RawMoveList::new(board.get_drops(active_lines, player));
    
    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        let starting_sq = active_line_sq + x;
        if board.piece_at(starting_sq) != Piece::None {
            let starting_piece = board.piece_at(starting_sq);

            move_list.add_start_index(x);
            move_list.set_start(x, starting_sq, starting_piece);

            STACK_BUFFER.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
            STACK_BUFFER.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

        let action = data.0;
        let backtrack_board = data.1;
        let banned_positions = data.2;
        let current_sq = data.3;
        let current_piece = data.4;
        let starting_sq = data.5;
        let starting_piece = data.6;
        let active_line_idx = data.7;
        let player: Player = data.8;

        match action {
            Action::Start => {
                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;

            },
            Action::End => {
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;
                
            },
            Action::Gen => {
                match current_piece {
                    Piece::One => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }
            
                            let end = SQ(path.0[1]);
                            let end_bit = end.bit();
             
                            if (move_list.end_positions[active_line_idx] & end_bit).is_not_empty() {
                                continue;
            
                            }
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                move_list.set_end_position(active_line_idx, end_bit);
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                move_list.set_end_position(active_line_idx, end_bit);
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    move_list.set_pickup_position(active_line_idx, end_bit);
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
            
                                }
                                
                            } else {
                                move_list.set_end_position(active_line_idx, end_bit);
                
                            }
                
                        }

                    },
                    Piece::Two => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);
      
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[2]);
                            let end_bit = end.bit();
            
                            if (move_list.end_positions[active_line_idx] & end_bit).is_not_empty() {
                                continue;

                            }

                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                move_list.set_end_position(active_line_idx, end_bit);
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                move_list.set_end_position(active_line_idx, end_bit);
                                continue;
                
                            }

                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    move_list.set_pickup_position(active_line_idx, end_bit);
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                }
                                
                            } else {
                                move_list.set_end_position(active_line_idx, end_bit);
                
                            }
                
                        }

                    },
                    Piece::Three => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[3]);
                            let end_bit = end.bit();
            
                            if (move_list.end_positions[active_line_idx] & end_bit).is_not_empty() {
                                continue;

                            }

                           if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                move_list.set_end_position(active_line_idx, end_bit);
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                move_list.set_end_position(active_line_idx, end_bit);
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    move_list.set_pickup_position(active_line_idx, end_bit);
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                }
                                
                            } else {
                                move_list.set_end_position(active_line_idx, end_bit);
                
                            }
                
                        }

                    },
                    Piece::None => {}

                }

            }

        }
        

    }

    move_list

}

pub unsafe fn valid_move_count(board: &mut BoardState, player: Player) -> usize {
    let active_lines = board.get_active_lines();

    let mut count = 0;

    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        let starting_sq = active_line_sq + x;
        if board.piece_at(starting_sq) != Piece::None {
            let starting_piece = board.piece_at(starting_sq);

            STACK_BUFFER.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
            STACK_BUFFER.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

        let action = data.0;
        let backtrack_board = data.1;
        let banned_positions = data.2;
        let current_sq = data.3;
        let current_piece = data.4;
        let starting_sq = data.5;
        let starting_piece = data.6;
        let active_line_idx = data.7;
        let player: Player = data.8;

        match action {
            Action::Start => {
                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;

            },
            Action::End => {
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;
                
            },
            Action::Gen => {
                match current_piece {
                    Piece::One => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }
            
                            let end = SQ(path.0[1]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    count += 25;
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
            
                                }
                                
                            } else {
                                count += 1;
                
                            }
                
                        }

                    },
                    Piece::Two => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[2]);
                            let end_bit = end.bit();

                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    count += 25;
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                }
                                
                            } else {
                                count += 1;
                
                            }
                
                        }

                    },
                    Piece::Three => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[3]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    count += 25;
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                }
                                
                            } else {
                                count += 1;
                
                            }
                
                        }

                    },
                    Piece::None => {}

                }

            }

        }
        

    }

    count

}

pub unsafe fn valid_threat_count(board: &mut BoardState, player: Player) -> usize {
    let active_lines = board.get_active_lines();

    let mut count = 0;

    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        let starting_sq = active_line_sq + x;
        if board.piece_at(starting_sq) != Piece::None {
            let starting_piece = board.piece_at(starting_sq);

            STACK_BUFFER.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
            STACK_BUFFER.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

        let action = data.0;
        let backtrack_board = data.1;
        let banned_positions = data.2;
        let current_sq = data.3;
        let current_piece = data.4;
        let starting_sq = data.5;
        let starting_piece = data.6;
        let active_line_idx = data.7;
        let player: Player = data.8;

        match action {
            Action::Start => {
                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;

            },
            Action::End => {
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;
                
            },
            Action::Gen => {
                match current_piece {
                    Piece::One => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }
            
                            let end = SQ(path.0[1]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            }

                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                
                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                
                            }
                
                        }

                    },
                    Piece::Two => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[2]);
                            let end_bit = end.bit();

                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            }
                            
                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                
                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                
                            }

                        }

                    },
                    Piece::Three => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[3]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                count += 1;
                                continue;
                
                            }
                            
                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                
                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                
                            }
                
                        }

                    },
                    Piece::None => {}

                }

            }

        }
        

    }

    count

}

pub unsafe fn controlled_pieces(board: &mut BoardState, player: Player) -> BitBoard {
    let active_lines = board.get_active_lines();

    let mut controlled_pieces = BitBoard::EMPTY;

    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        let starting_sq = active_line_sq + x;
        if board.piece_at(starting_sq) != Piece::None {
            let starting_piece = board.piece_at(starting_sq);

            controlled_pieces |= starting_sq.bit();

            STACK_BUFFER.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
            STACK_BUFFER.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

        let action = data.0;
        let backtrack_board = data.1;
        let banned_positions = data.2;
        let current_sq = data.3;
        let current_piece = data.4;
        let starting_sq = data.5;
        let starting_piece = data.6;
        let active_line_idx = data.7;
        let player: Player = data.8;

        match action {
            Action::Start => {
                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;

            },
            Action::End => {
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;
                
            },
            Action::Gen => {
                match current_piece {
                    Piece::One => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }
            
                            let end = SQ(path.0[1]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                continue;
                
                            }

                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                                
                                controlled_pieces |= end_bit;

                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                
                            }
                
                        }

                    },
                    Piece::Two => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[2]);
                            let end_bit = end.bit();

                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                continue;
                
                            }
                            
                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                                
                                controlled_pieces |= end_bit;

                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                
                            }

                        }

                    },
                    Piece::Three => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[3]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                continue;
                
                            }
                            
                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                                
                                controlled_pieces |= end_bit;

                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                
                            }
                
                        }

                    },
                    Piece::None => {}

                }

            }

        }
        

    }

    controlled_pieces

}

pub unsafe fn controlled_squares(board: &mut BoardState, player: Player) -> BitBoard {
    let active_lines = board.get_active_lines();

    let mut controlled_squares = BitBoard::EMPTY;

    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        let starting_sq = active_line_sq + x;
        if board.piece_at(starting_sq) != Piece::None {
            let starting_piece = board.piece_at(starting_sq);

            STACK_BUFFER.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
            STACK_BUFFER.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

        let action = data.0;
        let backtrack_board = data.1;
        let banned_positions = data.2;
        let current_sq = data.3;
        let current_piece = data.4;
        let starting_sq = data.5;
        let starting_piece = data.6;
        let active_line_idx = data.7;
        let player: Player = data.8;

        match action {
            Action::Start => {
                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;

            },
            Action::End => {
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;
                
            },
            Action::Gen => {
                match current_piece {
                    Piece::One => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }
            
                            let end = SQ(path.0[1]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;

                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
            
                                }
                                
                            } else {
                                controlled_squares |= end_bit;

                            }
                
                        }

                    },
                    Piece::Two => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[2]);
                            let end_bit = end.bit();

                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                }
                                
                            } else {
                                controlled_squares |= end_bit;

                            }

                        }

                    },
                    Piece::Three => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[3]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                continue;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                continue;
                
                            }
                            
                            let end_piece = board.piece_at(end);
                            if end_piece != Piece::None {
                                if (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                }
                                
                            } else {
                                controlled_squares |= end_bit;

                            }
                
                        }

                    },
                    Piece::None => {}

                }

            }

        }
        

    }

    controlled_squares & !board.piece_bb

}

pub unsafe fn has_threat(board: &mut BoardState, player: Player) -> bool {
    let active_lines = board.get_active_lines();
 
    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        if board.piece_at(active_line_sq + x) != Piece::None {
            let starting_sq = active_line_sq + x;
            let starting_piece = board.piece_at(starting_sq);

            STACK_BUFFER.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
            STACK_BUFFER.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
            STACK_BUFFER.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

        }

    }

    loop {
        if STACK_BUFFER.is_empty() {
            break;
        }
        let data = STACK_BUFFER.pop().unwrap_unchecked();

        let action = data.0;
        let backtrack_board = data.1;
        let banned_positions = data.2;
        let current_sq = data.3;
        let current_piece = data.4;
        let starting_sq = data.5;
        let starting_piece = data.6;
        let active_line_idx = data.7;
        let player: Player = data.8;

        match action {
            Action::Start => {
                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;

            },
            Action::End => {
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();
                continue;
                
            },
            Action::Gen => {
                match current_piece {
                    Piece::One => {
                        let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
            
                        for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
            
                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }
            
                            let end = SQ(path.0[1]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                return true;
 
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                return true;
  
                            }

                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                
                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
            
                            }
                
                        }

                    },
                    Piece::Two => {
                        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
                        let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[2]);
                            let end_bit = end.bit();

                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                return true;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                return true;
                
                            }
                            
                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                
                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
            
                            }

                        }

                    },
                    Piece::Three => {
                        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[current_sq.0 as usize];

                        let valid_paths_idx = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
                        let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                        for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                            let path_idx = valid_paths[i as usize];
                            let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                            if (backtrack_board & path.1).is_not_empty() {
                                continue;
                
                            }

                            let end = SQ(path.0[3]);
                            let end_bit = end.bit();
            
                            if end == SQ::P1_GOAL {
                                if player == Player::One {
                                    continue;
                                }
                                return true;
                
                            } else if end == SQ::P2_GOAL {
                                if player == Player::Two {
                                    continue;
                                }
                                return true;
                
                            }
                            
                            if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                let new_banned_positions = banned_positions ^ end_bit;
                                let new_backtrack_board = backtrack_board ^ path.1;
                
                                STACK_BUFFER.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
            
                            }
                
                        }

                    },
                    Piece::None => {}

                }

            }

        }
        

    }

    false

}


#[cfg(test)]
mod move_gen_bench {
    use test::Bencher;

    use crate::moves::movegen::*;

    #[bench]
    fn bench_valid_moves(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ valid_moves(&mut board, Player::One) }.moves(&board));

    }

    #[bench]
    fn bench_valid_move_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ valid_move_count(&mut board, Player::One) });

    }

    #[bench]
    fn bench_threat_count(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ valid_threat_count(&mut board, Player::One) });

    }

    #[bench]
    fn bench_controlled_pieces(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ controlled_pieces(&mut board, Player::One) });

    }

    #[bench]
    fn bench_controlled_squares(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ controlled_squares(&mut board, Player::One) });

    }

    #[bench]
    fn bench_has_threat(b: &mut Bencher) {
        let mut board = BoardState::from(BENCH_BOARD);

        b.iter(|| unsafe{ has_threat(&mut board, Player::One) });

    }

}