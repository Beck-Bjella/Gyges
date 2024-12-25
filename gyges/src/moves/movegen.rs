//! This module contains all of the functions for generating moves. 
//! 
//! All of the functions in this module are unsafe and cannot be run concurrently.
//!  

use std::cell::RefCell;

use crate::board::*;
use crate::board::bitboard::*;
use crate::core::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;

use super::Move;
use super::MoveType;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Action {
    Gen,
    Start,
    End

}

type StackData = (Action, BitBoard, BitBoard, SQ, Piece, SQ, Piece, usize, Player);

pub const STACK_BUFFER_SIZE: usize = 10000;
thread_local! {
    static STACK_BUFFER: RefCell<Vec<StackData>> = RefCell::new(Vec::with_capacity(STACK_BUFFER_SIZE));

}


//////////////////////////////////////////////////////////////////
///////////////////////////// SINGLE /////////////////////////////
//////////////////////////////////////////////////////////////////

/// Generates all of the legal moves for a player in the form of a [RawMoveList].
/// 
pub unsafe fn valid_moves(board: &mut BoardState, player: Player) -> RawMoveList {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();
        let mut move_list: RawMoveList = RawMoveList::new(board.get_drops(active_lines, player));
        
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                move_list.set_start(x, starting_sq, starting_piece);
                
                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
        
    })

}

/// Counts the number of moves that a player has on a board.
/// 
pub unsafe fn valid_move_count(board: &mut BoardState, player: Player) -> usize {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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

    })
    
}

/// Counts the number of threats that a player has on a board.
/// 
pub unsafe fn valid_threat_count(board: &mut BoardState, player: Player) -> usize {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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
                    
                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
                    
                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
                    
                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
                                }
                    
                            }

                        },
                        Piece::None => {}

                    }

                }

            }
            

        }

        count

    })

}

/// Generates a [BitBoard] for all of the pieces that a player can reach.
/// 
pub unsafe fn controlled_pieces(board: &mut BoardState, player: Player) -> BitBoard {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut controlled_pieces = BitBoard::EMPTY;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                controlled_pieces |= starting_sq.bit();

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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

                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
                                }
                    
                            }

                        },
                        Piece::None => {}

                    }

                }

            }
            

        }

        controlled_pieces
    
    })

}

/// Generates a [BitBoard] for all of the sqaures that a player can reach.
/// 
pub unsafe fn controlled_squares(board: &mut BoardState, player: Player) -> BitBoard {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut controlled_squares = BitBoard::EMPTY;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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

                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
    
    })

}

/// Returns true if there is a valid threat on the board.
/// 
pub unsafe fn has_threat(board: &mut BoardState, player: Player) -> bool {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return true;
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return true;
                    
                                }

                                if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                    
                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return true;
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return true;
                    
                                }

                                if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                    
                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return true;
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return true;
                    
                                }

                                if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                    
                                    stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
                                }
                    
                            }

                        },
                        Piece::None => {}

                    }

                }

            }
            

        }

        false
    
    })

}

////////////////////////////////////////////////////////////////////////
///////////////////////////// COMBONATIONS /////////////////////////////
////////////////////////////////////////////////////////////////////////

/// Generates the contolled squares, controlled pieces, and the move count for a player.
/// 
pub unsafe fn control_and_movecount(board: &mut BoardState, player: Player) -> (BitBoard, BitBoard, usize) {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut count = 0;
        let mut controlled_squares = BitBoard::EMPTY;
        let mut controlled_pieces = BitBoard::EMPTY;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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
                                        controlled_squares |= end_bit;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
                                    }
                                    
                                } else {
                                    count += 1;
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
                                        controlled_pieces |= end_bit;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                    }
                                    
                                } else {
                                    count += 1;
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
                                        controlled_pieces |= end_bit;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

                                    }
                                    
                                } else {
                                    count += 1;
                                    controlled_squares |= end_bit;
                    
                                }
                    
                            }

                        },
                        Piece::None => {}

                    }

                }

            }
            

        }

        (controlled_squares, controlled_pieces, count)

    })
    
}

/// Returns true if there is a valid threat on the board, else returns the move count.
/// 
pub unsafe fn threat_or_movecount(board: &mut BoardState, player: Player) -> (bool, usize) {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                    
                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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

        (false, count)

    })
    
}

/// Returns true if there is a valid threat on the board, else returns the valid moves.
/// 
pub unsafe fn threat_or_moves(board: &mut BoardState, player: Player) -> (bool, RawMoveList) {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();
        let mut move_list: RawMoveList = RawMoveList::new(board.get_drops(active_lines, player));
        
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                move_list.set_start(x, starting_sq, starting_piece);
                
                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, move_list);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }
                                    move_list.set_end_position(active_line_idx, end_bit);

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, move_list);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        move_list.set_pickup_position(active_line_idx, end_bit);
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, move_list);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }
                                    move_list.set_end_position(active_line_idx, end_bit);

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, move_list);
                    
                                }

                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        move_list.set_pickup_position(active_line_idx, end_bit);
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                   
                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, move_list);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }
                                    move_list.set_end_position(active_line_idx, end_bit);
                                    
                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, move_list);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        move_list.set_pickup_position(active_line_idx, end_bit);
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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

        (false, move_list)
        
    })

}



//////////////////////////////////////////////////////////////////////
///////////////////////////// SIMD TESTS /////////////////////////////
//////////////////////////////////////////////////////////////////////

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn process_paths_simd(backtrack_board: u64, paths: &[u64; 4]) -> [u64; 4] {
    unsafe {
        // Load `backtrack_board` into a SIMD register
        let backtrack_vec = _mm256_set1_epi64x(backtrack_board as i64);

        let paths_vec = _mm256_loadu_si256(paths.as_ptr() as *const __m256i);

        // Perform SIMD bitwise AND
        let and_result = _mm256_and_si256(backtrack_vec, paths_vec);

        // Return the results
        let mut results = [0; 4];
        _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, and_result);
        
        results

    }

}

/// Returns true if there is a valid threat on the board, else returns the move count.
/// 
pub unsafe fn threat_or_movecount_simd(board: &mut BoardState, player: Player) -> (bool, usize) {
    STACK_BUFFER.with_borrow_mut(|stack_buffer| {
        let active_lines = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if stack_buffer.is_empty() {
                break;
            }
            let data = stack_buffer.pop().unwrap_unchecked();

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
                        // Piece::One => {
                        //     let valid_paths_idx = ONE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(0);
                        //     let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
                        //     let path_count = valid_paths[ONE_PATH_COUNT_IDX] as usize;

                        //     // Part 1: SIMD Preprocessing
                        //     let mut paths = [([0; 2], 0); 4];
                        //     let mut buffer = [0; 4];
                        //     for i in 0..path_count {
                        //         let path_idx = valid_paths[i as usize];
                        //         paths[i] = *UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);
                        //         buffer[i] = paths[i].1;
      
                        //     }
                    
                        //     // Perform SIMD operation to filter valid paths
                        //     let results = process_paths_simd(backtrack_board.0, &buffer);
                            
                        //     // Part 2: Scalar Postprocessing for Valid Paths
                        //     for i in 0..path_count {
                        //         if results[i] != 0 {
                        //             continue;
                        //         }

                        //         let path = paths[i as usize];
                                
                        //         let end = SQ(path.0[1]);
                        //         let end_bit = end.bit();
                        
                        //         // Goal-check logic
                        //         if end == SQ::P1_GOAL {
                        //             if player == Player::One {
                        //                 continue;
                        //             }

                        //             board.place(starting_piece, starting_sq);
                        //             board.piece_bb ^= starting_sq.bit();
                        //             stack_buffer.clear();
                        //             return (true, 0);
                    
                        //         } else if end == SQ::P2_GOAL {
                        //             if player == Player::Two {
                        //                 continue;
                        //             }

                        //             board.place(starting_piece, starting_sq);
                        //             board.piece_bb ^= starting_sq.bit();
                        //             stack_buffer.clear();
                        //             return (true, 0);
                    
                        //         }
                                
                        
                        //         // Handle banned positions and stack updates
                        //         let end_piece = board.piece_at(end);
                        //         if end_piece != Piece::None {
                        //             if (banned_positions & end_bit).is_empty() {
                        //                 let new_banned_positions = banned_positions ^ end_bit;
                        //                 let new_backtrack_board = backtrack_board ^ path.1;
                        
                        //                 count += 25;
                        
                        //                 stack_buffer.push((
                        //                     Action::Gen,
                        //                     new_backtrack_board,
                        //                     new_banned_positions,
                        //                     end,
                        //                     end_piece,
                        //                     starting_sq,
                        //                     starting_piece,
                        //                     active_line_idx,
                        //                     player,
                        //                 ));
                        //             }
                        //         } else {
                                    
                        //         }
                        //     }
                        // },
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

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                    
                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack_buffer.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                            let path_count = valid_paths[THREE_PATH_COUNT_IDX] as usize;

                            let mut buffer = [0; 4];
                            for chunk_start in (0..path_count).step_by(4) {
                                let mut paths = [([0; 4], 0); 4];

                                // Load paths into the buffer
                                for i in 0..4 {
                                    if chunk_start + i < path_count {
                                        let path_idx = valid_paths[chunk_start + i];
                                        paths[i] = *UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);
                                        buffer[i] = paths[i].1;

                                    }

                                }
                        
                                // Process SIMD for the batch
                                let simd_results = process_paths_simd(backtrack_board.0, &buffer);
                        
                                // Store results back
                                for i in 0..4 {
                                    if chunk_start + i >= path_count {
                                        continue;

                                    }

                                    if simd_results[i] != 0 {
                                        continue;
                                    }
    
                                    let path = paths[i as usize];
    
                                    let end = SQ(path.0[3]);
                                    let end_bit = end.bit();
                    
                                    if end == SQ::P1_GOAL {
                                        if player == Player::One {
                                            continue;
                                        }
    
                                        board.place(starting_piece, starting_sq);
                                        board.piece_bb ^= starting_sq.bit();
                                        stack_buffer.clear();
                                        return (true, 0);
                        
                                    } else if end == SQ::P2_GOAL {
                                        if player == Player::Two {
                                            continue;
                                        }
    
                                        board.place(starting_piece, starting_sq);
                                        board.piece_bb ^= starting_sq.bit();
                                        stack_buffer.clear();
                                        return (true, 0);
                        
                                    }
                                    
                                    let end_piece = board.piece_at(end);
                                    if end_piece != Piece::None {
                                        if (banned_positions & end_bit).is_empty() {
                                            let new_banned_positions = banned_positions ^ end_bit;
                                            let new_backtrack_board = backtrack_board ^ path.1;
                                            
                                            count += 25;
                                            
                                            stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
    
                                        }
                                        
                                    } else {
                                        count += 1;
                        
                                    }

                                }

                            }

                        },
                        Piece::None => {}

                    }

                }

            }
            

        }

        (false, count)

    })
    
}

///////////////////////////////////////////////////////////////////////
///////////////////////////// OTHER TESTS /////////////////////////////
///////////////////////////////////////////////////////////////////////

const NEW_STACK_SIZE: usize = 1000;

thread_local! {
    static NEW_STACK: RefCell<FixedStack<StackData>> = RefCell::new(FixedStack::new(NEW_STACK_SIZE));
}


/// A fixed-size stack allocated on the heap.
pub struct FixedStack<T> {
    buffer: Box<[T]>,
    top: usize,
}

impl<T> FixedStack<T> {
    /// Creates a new fixed-size stack with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let buffer = unsafe {
            let mut vec = Vec::with_capacity(capacity);
            vec.set_len(capacity);
            vec.into_boxed_slice()
        };

        Self { buffer, top: 0 }

    }

    /// Pushes a value onto the top of the stack.
    pub unsafe fn push(&mut self, value: T) {
        if self.top >= self.buffer.len() {
            panic!("Stack overflow!");
        }

        *self.buffer.get_unchecked_mut(self.top) = value;
        self.top += 1;

    }

    /// Pops a value from the top of the stack.
    pub unsafe fn pop(&mut self) -> T {
        if self.top == 0 {
            panic!("Stack underflow!");
        }

        self.top -= 1;
        std::ptr::read(self.buffer.get_unchecked(self.top))

    }

    /// Returns true if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.top == 0

    }

    /// Clears the stack without modifying memory.
    pub fn clear(&mut self) {
        self.top = 0;

    }

}

impl<T> Drop for FixedStack<T> {
    // Custom drop to properly clean up memory
    fn drop(&mut self) {
        for i in 0..self.top {
            unsafe {
                std::ptr::drop_in_place(self.buffer.get_unchecked_mut(i));

            }

        }

    }

}


/// Returns true if there is a valid threat on the board, else returns the move count.
/// 
pub unsafe fn test_threat_or_movecount(board: &mut BoardState, player: Player) -> (bool, usize) {
    NEW_STACK.with_borrow_mut(|stack| {
        let active_lines: [usize; 2] = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                stack.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                stack.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                stack.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

       while !stack.is_empty() {
            let data = stack.pop();

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
                            let path_count = valid_paths[ONE_PATH_COUNT_IDX] as usize;

                            for i in 0..path_count {
                                let path_idx: u16 = valid_paths[i as usize];
                                let path = &UNIQUE_ONE_PATHS[path_idx as usize];
                
                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }
                
                                let end = SQ(path.0[1]);
                                let end_bit = end.bit();
                
                                if end == SQ::P1_GOAL {
                                    if player == Player::One {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                            let path_count = valid_paths[TWO_PATH_COUNT_IDX] as usize;

                            for i in 0..path_count {
                                let path_idx = valid_paths[i as usize];
                                let path = &UNIQUE_TWO_PATHS[path_idx as usize];

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                let end = SQ(path.0[2]);
                                let end_bit = end.bit();

                                if end == SQ::P1_GOAL {
                                    if player == Player::One {
                                        continue;
                                    }
                                    
                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                                        
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
                            let path_count = valid_paths[THREE_PATH_COUNT_IDX] as usize;

                            for i in 0..path_count {
                                let path_idx = valid_paths[i as usize];
                                let path = &UNIQUE_THREE_PATHS[path_idx as usize];

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                let end = SQ(path.0[3]);
                                let end_bit = end.bit();
                
                                if end == SQ::P1_GOAL {
                                    if player == Player::One {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack.clear();
                                    return (true, 0);
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    stack.clear();
                                    return (true, 0);
                    
                                }
                                
                                let end_piece = board.piece_at(end);
                                if end_piece != Piece::None {
                                    if (banned_positions & end_bit).is_empty() {
                                        let new_banned_positions = banned_positions ^ end_bit;
                                        let new_backtrack_board = backtrack_board ^ path.1;
                                        
                                        count += 25;
                                        
                                        stack.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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

        (false, count)

    })
    
}
