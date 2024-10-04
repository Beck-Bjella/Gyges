//! This module contains all of the functions for generating moves. 
//! 
//! All of the functions in this module are unsafe and cannot be run concurrently.
//!  

// use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crossbeam::channel::*;
use rayon::ThreadPoolBuilder;

// para iter stuff
use rayon::prelude::*;


use std::cell::RefCell;

use crate::board::*;
use crate::board::bitboard::*;
use crate::core::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;

pub const STACK_BUFFER_SIZE: usize = 10000;
thread_local! {
    static STACK_BUFFER: RefCell<Vec<StackData>> = RefCell::new(Vec::with_capacity(STACK_BUFFER_SIZE));

}

#[derive(PartialEq)]
pub enum Action {
    Gen,
    Start,
    End

}

type StackData = (Action, BitBoard, BitBoard, SQ, Piece, SQ, Piece, usize, Player);



#[derive(Debug, Clone)]
pub enum MoveGenType {
    ValidMoves,
    ValidMoveCount,
    ValidThreatCount,
    ControlledPieces,
    ControlledSquares,
    HasThreat

}

#[derive(Debug, Clone)]
pub struct MoveGenRequest {
    board: BoardState,
    player: Player,
    flag: MoveGenType,
    id: usize

}

impl MoveGenRequest {
    pub fn new(board: BoardState, player: Player, flag: MoveGenType, id: usize) -> Self {
        MoveGenRequest {
            board,
            player,
            flag,
            id

        }

    }

}

#[derive(Debug, Clone)]
pub struct MoveGenResult {
    pub id: usize,
    
    pub moves: RawMoveList,
    pub move_count: usize,
    pub threat_count: usize,
    pub controlled_pieces: BitBoard,
    pub controlled_squares: BitBoard,
    pub has_threat: bool

}

/// A struct for generating moves.
pub struct MoveGen {
    stop_sender: Sender<bool>,

    request_sender: Sender<MoveGenRequest>,
    result_receiver: Receiver<MoveGenResult>,

    pub results: Vec<MoveGenResult>,
    pub gen_count: usize

}

impl MoveGen {
    pub fn new() -> Self {
        let (ss, sr) = unbounded();
        let (request_sender_main, request_reciver_thread): (Sender<MoveGenRequest>, Receiver<MoveGenRequest>) = unbounded();
        let (result_sender_thread, result_reciver_main): (Sender<MoveGenResult>, Receiver<MoveGenResult>) = unbounded();

        init_movegen(sr, request_reciver_thread, result_sender_thread);

        MoveGen {
            stop_sender: ss,

            request_sender: request_sender_main,
            result_receiver: result_reciver_main,

            results: Vec::new(),
            gen_count: 0
            
        }

    }

    /// Gets all available results
    pub fn get(&mut self) {
        // Get all new results
        while let Ok(result) = self.result_receiver.try_recv() {
            self.results.push(result);
            self.gen_count += 1;

        }

    }

    /// Queues a request to generate moves.
    pub fn queue(&mut self, request: MoveGenRequest) {
        self.request_sender.send(request).unwrap();

    }

    /// Queries the results for a specific request id.
    pub fn query(&mut self, id: usize) -> Option<MoveGenResult> {
        // Get all new results
        self.get();

        // Find result with matching id EFFICIENTLY
        for i in 0..self.results.len() {
            if self.results[i].id == id {
                return Some(self.results.remove(i));

            }

        }

        None

    }

    /// Clears all of the results.
    pub fn clear(&mut self) {
        self.results.clear();

    }
   
    /// Generates the requested data, and waits until result is available.
    pub fn gen(&mut self, board: &mut BoardState, player: Player, flag: MoveGenType, id: usize) -> MoveGenResult {
        let request = MoveGenRequest::new(board.clone(), player, flag, id);
        self.request_sender.send(request).unwrap();

        // Wait for the result
        loop {
            self.get();
            if let Some(result) = self.query(id) {
                return result;

            }

        }

    }

}

/// Initializes the move generation thread.
pub fn init_movegen(stop_reciver: Receiver<bool>, request_receiver: Receiver<MoveGenRequest>, result_sender: Sender<MoveGenResult>) {
    // Setup rayon
    ThreadPoolBuilder::new().num_threads(4).build_global().expect("Failed to build movegen thread pool");
    
    // Start the movegen thread
    thread::spawn(move || {
        loop {
            // Collect all requests
            let mut requests: Vec<MoveGenRequest> = Vec::new();
            while let Ok(request) = request_receiver.try_recv() {
                requests.push(request);

            }

            // Process current requests
            requests.into_par_iter().for_each(|request| {
                let result = process_request(request);
                result_sender.send(result).unwrap();

            });

            // Check stop signal
            if stop_reciver.try_recv().is_ok() {
                break;
                
            }

        }

    });

}

/// Processes a move generation request.
pub fn process_request(mut request: MoveGenRequest) -> MoveGenResult {
    match request.flag {
        MoveGenType::ValidMoves => {
            let moves: RawMoveList = STACK_BUFFER.with(|stack| {
                unsafe { valid_moves(&mut request.board, request.player, &mut stack.borrow_mut()) }

            });

            let result = MoveGenResult {
                id: request.id,
                moves,
                move_count: 0,
                threat_count: 0,
                controlled_pieces: BitBoard::EMPTY,
                controlled_squares: BitBoard::EMPTY,
                has_threat: false

            };

            result

        },
        MoveGenType::ValidMoveCount => {
            let move_count: usize = STACK_BUFFER.with(|stack| {
                unsafe { valid_move_count(&mut request.board, request.player, &mut stack.borrow_mut()) }

            });

            let result = MoveGenResult {
                id: request.id,
                moves: RawMoveList::new(BitBoard::EMPTY),
                move_count,
                threat_count: 0,
                controlled_pieces: BitBoard::EMPTY,
                controlled_squares: BitBoard::EMPTY,
                has_threat: false

            };

            result

        },
        MoveGenType::ValidThreatCount => {
            let threat_count: usize = STACK_BUFFER.with(|stack| {
                unsafe { valid_threat_count(&mut request.board, request.player, &mut stack.borrow_mut()) }

            });

            let result = MoveGenResult {
                id: request.id,
                moves: RawMoveList::new(BitBoard::EMPTY),
                move_count: 0,
                threat_count,
                controlled_pieces: BitBoard::EMPTY,
                controlled_squares: BitBoard::EMPTY,
                has_threat: false

            };

            result

        },
        MoveGenType::ControlledPieces => {
            let controlled_pieces: BitBoard = STACK_BUFFER.with(|stack| {
                unsafe { controlled_pieces(&mut request.board, request.player, &mut stack.borrow_mut()) }

            });

            let result = MoveGenResult {
                id: request.id,
                moves: RawMoveList::new(BitBoard::EMPTY),
                move_count: 0,
                threat_count: 0,
                controlled_pieces,
                controlled_squares: BitBoard::EMPTY,
                has_threat: false

            };

            result

        },
        MoveGenType::ControlledSquares => {
            let controlled_squares: BitBoard = STACK_BUFFER.with(|stack| {
                unsafe { controlled_squares(&mut request.board, request.player, &mut stack.borrow_mut()) }

            });

            let result = MoveGenResult {
                id: request.id,
                moves: RawMoveList::new(BitBoard::EMPTY),
                move_count: 0,
                threat_count: 0,
                controlled_pieces: BitBoard::EMPTY,
                controlled_squares,
                has_threat: false

            };

            result

        },
        MoveGenType::HasThreat => {
            let has_threat: bool = STACK_BUFFER.with(|stack| {
                let mut stack = stack.borrow_mut();
                unsafe { has_threat(&mut request.board, request.player, &mut stack) }

            });

            let result = MoveGenResult {
                id: request.id,
                moves: RawMoveList::new(BitBoard::EMPTY),
                move_count: 0,
                threat_count: 0,
                controlled_pieces: BitBoard::EMPTY,
                controlled_squares: BitBoard::EMPTY,
                has_threat

            };

            result

        },
        
    }

}


/// Generates all of the legal moves for a player in the form of a [RawMoveList].
/// 
pub unsafe fn valid_moves(board: &mut BoardState, player: Player, stack_buffer: &mut Vec<StackData>) -> RawMoveList {
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

}

/// Counts the number of moves that a player has on a board.
/// 
pub unsafe fn valid_move_count(board: &mut BoardState, player: Player, stack_buffer: &mut Vec<StackData>) -> usize {
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

}

/// Counts the number of threats (ways into the opponents goal) that a player.
/// 
pub unsafe fn valid_threat_count(board: &mut BoardState, player: Player, stack_buffer: &mut Vec<StackData>) -> usize {
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

}

/// Generates a [BitBoard] for all of the pieces that a player can reach.
/// 
pub unsafe fn controlled_pieces(board: &mut BoardState, player: Player, stack_buffer: &mut Vec<StackData>) -> BitBoard {
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

}

/// Generates a [BitBoard] for all of the sqaures that a player can reach.
/// 
pub unsafe fn controlled_squares(board: &mut BoardState, player: Player, stack_buffer: &mut Vec<StackData>) -> BitBoard {
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

}

/// Returns true if there is a valid threat on the board.
/// 
pub unsafe fn has_threat(board: &mut BoardState, player: Player, stack_buffer: &mut Vec<StackData>) -> bool {
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

}
