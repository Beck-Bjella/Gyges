//! This module contains all of the functions for generating moves. 
//! 
//! All of the functions in this module are unsafe and cannot be run concurrently.
//!  

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::collections::VecDeque;


use crate::board::*;
use crate::board::bitboard::*;
use crate::core::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;

#[derive(PartialEq)]
enum Action {
    Gen,
    Start,
    End

}

type StackData = (Action, BitBoard, BitBoard, SQ, Piece, SQ, Piece, usize, Player);

pub const STACK_BUFFER_SIZE: usize = 1000;

pub enum MoveGenType {
    ValidMoves,
    ValidMoveCount,
    ValidThreatCount,
    ControlledPieces,
    ControlledSquares,
    HasThreat

}

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
    manager_thread: Option<thread::JoinHandle<()>>,
    manager_stop: Sender<bool>,

    request_sender: Sender<MoveGenRequest>,
    result_receiver: Receiver<MoveGenResult>,

    pub results: Vec<MoveGenResult>

}

impl MoveGen {
    pub fn new() -> Self {
        let (ss, sr): (Sender<bool>, Receiver<bool>) = mpsc::channel();
        let (request_sender_main, request_reciver_thread): (Sender<MoveGenRequest>, Receiver<MoveGenRequest>) = mpsc::channel();
        let (result_sender_thread, result_reciver_main): (Sender<MoveGenResult>, Receiver<MoveGenResult>) = mpsc::channel();

        let manager_thread = thread::spawn(move || {
            let mut manager = MoveGenManager::new(sr, request_reciver_thread, result_sender_thread);
            manager.start();

        });
            
        MoveGen {
            manager_thread: Some(manager_thread),
            manager_stop: ss,

            request_sender: request_sender_main,
            result_receiver: result_reciver_main,

            results: Vec::new()
            
        }

    }

    /// Gets all available results
    pub fn get(&mut self) {
        // Get all new results
        while let Ok(result) = self.result_receiver.try_recv() {
            self.results.push(result);

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

        // Result with the matching id
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

        // Wait for result
        loop {
            if let Some(result) = self.query(id) {
                return result;

            }

        }

    }

    pub fn stop(&mut self) {
        self.manager_stop.send(true).unwrap();
        self.manager_thread.take().unwrap().join().unwrap();

    }

}

pub const WORKER_COUNT: usize = 4;

pub struct MoveGenManager {
    request_queue: VecDeque<MoveGenRequest>,
    
    stop_receiver: Receiver<bool>,
    request_receiver: Receiver<MoveGenRequest>,
    result_sender: Sender<MoveGenResult>

}

impl MoveGenManager {
    pub fn new(stop_receiver_m: Receiver<bool>, request_receiver_m: Receiver<MoveGenRequest>, result_sender_m: Sender<MoveGenResult>) -> Self {
        MoveGenManager {
            request_queue: VecDeque::new(),

            stop_receiver: stop_receiver_m,
            request_receiver: request_receiver_m,
            result_sender: result_sender_m

        }

    }


    pub fn start(&mut self) {
        let mut worker_states: Vec<bool> = vec![false; WORKER_COUNT];
        let mut worker_threads = vec![];
        let mut worker_stops: Vec<Sender<bool>> = vec![];

        let mut worker_senders = vec![];
        let mut worker_receivers = vec![];

        for worker_id in 0..WORKER_COUNT {
            let (ss, sr): (Sender<bool>, Receiver<bool>) = mpsc::channel();
            worker_stops.push(ss);

            let (request_sender, request_receiver_w): (Sender<MoveGenRequest>, Receiver<MoveGenRequest>) = mpsc::channel();
            let (result_sender_w, result_receiver): (Sender<MoveGenResult>, Receiver<MoveGenResult>) = mpsc::channel();
            worker_senders.push(request_sender);
            worker_receivers.push(result_receiver);

            worker_threads.push(thread::spawn(move || {
                let mut worker = MoveGenWorker::new(worker_id, sr, request_receiver_w, result_sender_w);
                worker.start();

            }));
            
        }

        // Main manager loop
        loop {
            // Receive requests
            while let Some(request) = self.request_receiver.try_recv().ok() {
                self.request_queue.push_back(request); // Insert at the front 
                
            }

            // Give requests to workers
            for (thread_id, state) in worker_states.iter_mut().enumerate() {
                if !*state {
                    if let Some(request) = self.request_queue.pop_front() { // Pop from the back 
                        worker_senders[thread_id].send(request).unwrap();
                        *state = true;

                    }

                }

            }

            // Receive data from workers
            for (thread_id, state) in worker_states.iter_mut().enumerate() {
                if *state {
                    match worker_receivers[thread_id].try_recv() {
                        Ok(result) => {
                            self.result_sender.send(result).unwrap();
                            *state = false;

                        },
                        Err(_) => {}

                    }

                }

            }

            // Check for stop signal
            if self.stop_receiver.try_recv().is_ok() {
                break;
                
            }

        }

        // Stop workers
        for stop in worker_stops {
            stop.send(true).unwrap();

        }

        // Join threads
        for thread in worker_threads {
            thread.join().unwrap();

        }

    }

}


/// Contains all of the functions for generating moves.
pub struct MoveGenWorker {
    stack_buffer: Vec<StackData>,

    worker_id: usize,

    stop_receiver: Receiver<bool>,
    request_receiver: Receiver<MoveGenRequest>,
    result_sender: Sender<MoveGenResult>

}

impl MoveGenWorker {
    pub fn new(worker_id: usize, stop_receiver: Receiver<bool>, request_receiver: Receiver<MoveGenRequest>, result_sender: Sender<MoveGenResult>) -> Self {
        MoveGenWorker {
            stack_buffer: Vec::with_capacity(STACK_BUFFER_SIZE),

            worker_id,

            stop_receiver,
            request_receiver,
            result_sender

        }

    }

    pub fn start(&mut self) {
        println!("Worker {} running", self.worker_id);
        loop {
            // Handle most recent request
            while let Some(mut request) = self.request_receiver.try_recv().ok() {
                match request.flag {
                    MoveGenType::ValidMoves => {
                        let moves = unsafe { self.valid_moves(&mut request.board, request.player) };

                        let result = MoveGenResult {
                            id: request.id,
                            moves,
                            move_count: 0,
                            threat_count: 0,
                            controlled_pieces: BitBoard::EMPTY,
                            controlled_squares: BitBoard::EMPTY,
                            has_threat: false

                        };

                        self.result_sender.send(result).unwrap();

                    },
                    MoveGenType::ValidMoveCount => {

                    },
                    MoveGenType::ValidThreatCount => {

                    },
                    MoveGenType::ControlledPieces => {

                    },
                    MoveGenType::ControlledSquares => {

                    },
                    MoveGenType::HasThreat => {

                    }
                    
                }
                
            }

            // Check for stop signal
            if self.stop_receiver.try_recv().is_ok() {
                break;

            }

        }

    }

    /// Generates all of the legal moves for a player in the form of a [RawMoveList].
    /// 
    /// # Safety
    /// 
    /// Uses a static mutable buffer to process moves during generation. 
    /// The buffer is used to avoid recursion and is not thread safe. 
    /// Running this function in parallel will cause undefined behavior.
    /// 
    pub unsafe fn valid_moves(&mut self, board: &mut BoardState, player: Player) -> RawMoveList {
        let active_lines = board.get_active_lines();
        let mut move_list: RawMoveList = RawMoveList::new(board.get_drops(active_lines, player));
        
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                move_list.set_start(x, starting_sq, starting_piece);

                self.stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if self.stack_buffer.is_empty() {
                break;
            }
            let data = self.stack_buffer.pop().unwrap_unchecked();

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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
    /// # Safety
    /// 
    /// Uses a static mutable buffer to count moves.
    /// The buffer is used to avoid recursion and is not thread safe.
    /// Running this function in parallel will cause undefined behavior.
    /// 
    pub unsafe fn valid_move_count(&mut self, board: &mut BoardState, player: Player) -> usize {
        let active_lines = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                self.stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if self.stack_buffer.is_empty() {
                break;
            }
            let data = self.stack_buffer.pop().unwrap_unchecked();

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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
    /// # Safety
    /// 
    /// Uses a static mutable buffer to count threats.
    /// The buffer is used to avoid recursion and is not thread safe.
    /// Running this function in parallel will cause undefined behavior.
    /// 
    pub unsafe fn valid_threat_count(&mut self, board: &mut BoardState, player: Player) -> usize {
        let active_lines = board.get_active_lines();

        let mut count = 0;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                self.stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if self.stack_buffer.is_empty() {
                break;
            }
            let data = self.stack_buffer.pop().unwrap_unchecked();

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
                    
                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
                    
                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
                    
                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
    /// # Safety
    /// 
    /// Uses a static mutable buffer to obtain the controlled pieces.
    /// The buffer is used to avoid recursion and is not thread safe.
    /// Running this function in parallel will cause undefined behavior.
    /// 
    pub unsafe fn controlled_pieces(&mut self, board: &mut BoardState, player: Player) -> BitBoard {
        let active_lines = board.get_active_lines();

        let mut controlled_pieces = BitBoard::EMPTY;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                controlled_pieces |= starting_sq.bit();

                self.stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if self.stack_buffer.is_empty() {
                break;
            }
            let data = self.stack_buffer.pop().unwrap_unchecked();

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

                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
    /// # Safety
    /// 
    /// Uses a static mutable buffer to obtain the controlled squares.
    /// The buffer is used to avoid recursion and is not thread safe.
    /// Running this function in parallel will cause undefined behavior.
    /// 
    pub unsafe fn controlled_squares(&mut self, board: &mut BoardState, player: Player) -> BitBoard {
        let active_lines = board.get_active_lines();

        let mut controlled_squares = BitBoard::EMPTY;

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                self.stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if self.stack_buffer.is_empty() {
                break;
            }
            let data = self.stack_buffer.pop().unwrap_unchecked();

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

                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                
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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
                                        
                                        self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));

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
    /// # Safety
    /// 
    /// Uses a static mutable buffer to check for threats.
    /// The buffer is used to avoid recursion and is not thread safe.
    /// Running this function in parallel will cause undefined behavior.
    /// 
    pub unsafe fn has_threat(&mut self, board: &mut BoardState, player: Player) -> bool {
        let active_lines = board.get_active_lines();

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                self.stack_buffer.push((Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack_buffer.push((Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack_buffer.push((Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        loop {
            if self.stack_buffer.is_empty() {
                break;
            }
            let data = self.stack_buffer.pop().unwrap_unchecked();

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
                                    self.stack_buffer.clear();
                                    return true;
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    self.stack_buffer.clear();
                                    return true;
                    
                                }

                                if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                    
                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
                                    self.stack_buffer.clear();
                                    return true;
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    self.stack_buffer.clear();
                                    return true;
                    
                                }
                                
                                if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                    
                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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
                                    self.stack_buffer.clear();
                                    return true;
                    
                                } else if end == SQ::P2_GOAL {
                                    if player == Player::Two {
                                        continue;
                                    }

                                    board.place(starting_piece, starting_sq);
                                    board.piece_bb ^= starting_sq.bit();
                                    self.stack_buffer.clear();
                                    return true;
                    
                                }
                                
                                if (board.piece_bb & end_bit).is_not_empty() && (banned_positions & end_bit).is_empty() {
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                    
                                    self.stack_buffer.push((Action::Gen, new_backtrack_board, new_banned_positions, end, board.piece_at(end), starting_sq, starting_piece, active_line_idx, player));
                                    
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

}
