//! Main strutures, and concepts related to searching.
//! 

pub mod evaluation;

use core::f64;
use std::cmp::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::time::Instant;

use gyges::board::*;
use gyges::moves::*;
use gyges::moves::move_list::*;
use gyges::tools::tt::*;
use gyges::core::*;
use gyges::moves::movegen::*;

use crate::root_movelist::RootMoveList;
use crate::search::evaluation::*;
use crate::consts::*;
use crate::ugi;

// Constants
pub const MAXPLY: i8 = 99;

// Constants for the threads.
pub const THREAD_INCREMENTS: [i8; 6] = [1, 2, 3, 4, 5, 6];
pub const THREAD_START_PLY: [i8; 6] = [1, 1, 1, 1, 1, 1];
pub const THREAD_COUNT: usize = 6;

// Constants for move ordering.
pub const LOSE_MOVE_SCORE: f64 = -1000000.0;
pub const TT_MOVE_SCORE: f64 = 1000000.0;

/// Structure that holds all needed information to perform a search, and conatains all of the main searching functions. 
/// It designed to be used in a seperate thread.
pub struct ThreadSearcher {
    thread_id: usize,

    move_gen: MoveGen,

    options: SearchOptions,
    current_ply: i8,

    start_time: Instant,

    completed_searchs: Vec<SearchData>,
    search_data: SearchData,
    root_moves: RootMoveList,
    
    stop_recv: Receiver<bool>,
    stop: bool,

    data_sender: Sender<SearchData>,
    data_recv: Receiver<SearchData>,

    skip: bool,

}

impl ThreadSearcher {
    pub fn new(thread_id: usize, options: SearchOptions, stop_recv: Receiver<bool>, data_sender: Sender<SearchData>, data_recv: Receiver<SearchData>) -> ThreadSearcher {
        ThreadSearcher {
            thread_id,

            move_gen: MoveGen::new(),

            options,
            current_ply: 1,

            start_time: Instant::now(),

            completed_searchs: Vec::new(),
            search_data: SearchData::new(1),
            root_moves: RootMoveList::new(),

            stop_recv,
            stop: false,

            data_sender,
            data_recv,

            skip: false,

        }

    }

    // Checks for communication from main thread.
    pub fn check(&mut self) {
        // Check stop signal
        if self.stop_recv.try_recv().is_ok() {
            self.stop = true;

        }

        // Check for max time.
        if let Some(maxtime) = self.options.maxtime {
            if self.search_data.start_time.elapsed().as_secs_f64() >= maxtime {
                self.stop = true;

            }

        }

        // Check data channel.
        if let Ok(data) = self.data_recv.try_recv() {
            let mut updated: bool = false;
            while self.current_ply <= data.ply {
                self.current_ply += THREAD_INCREMENTS[self.thread_id % THREAD_INCREMENTS.len()];
                updated = true;

            }

            if updated {
                self.skip = true;
                self.root_moves.moves = data.all_moves.clone();
                self.completed_searchs.push(data.clone());

            }

        }

    }

    /// Updates the search data based on the collected data.
    pub fn update_search_stats(&mut self) {
        self.search_data.search_time = self.search_data.start_time.elapsed().as_secs_f64();
        self.search_data.nps = (self.search_data.nodes as f64 / self.search_data.search_time) as usize;
        self.search_data.average_branching_factor = (self.search_data.nodes as f64).powf(1.0 / self.search_data.ply as f64);

        self.root_moves.sort();
        self.search_data.best_move = self.root_moves.moves.first().unwrap_or(&RootMove::new_null()).clone();
        self.search_data.all_moves = self.root_moves.moves.clone();

        self.search_data.pv = get_pv(&mut self.options.board.clone(), self.current_ply);
        
        if self.search_data.best_move.score == f64::INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 1;

        } else if self.search_data.best_move.score == f64::NEG_INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 2;

        }
      
    } 

    /// Iterative deepening search.
    pub fn iterative_deepening_search(&mut self) {
        self.stop = false;

        let board = &mut self.options.board.clone();

        // Check if the game is already over
        for mv in unsafe{ self.move_gen.valid_moves(board, Player::One).moves(board) } {
            if mv.is_win() {
                self.search_data.best_move = RootMove::new(mv, f64::INFINITY, 1);
                self.completed_searchs.push(self.search_data.clone());

                let best_search_data = self.completed_searchs.last().unwrap().clone();
                ugi::info_output(best_search_data.clone(), self.thread_id);
                ugi::best_move_output(best_search_data);
                return;

            }

        }

        self.root_moves = RootMoveList::new();
        self.setup_rootmoves(board);

        self.start_time = Instant::now();

        self.current_ply = THREAD_START_PLY[self.thread_id % THREAD_START_PLY.len()];
        'iterative_deepening: while !self.search_data.game_over {
            self.search_data = SearchData::new(self.current_ply);

            let (mut alpha, mut beta) = if self.completed_searchs.len() > 0 {
                let prev_score = self.completed_searchs.last().unwrap().clone().best_move.score;
                (prev_score - 1000.0, prev_score + 1000.0)

            } else {
                (f64::NEG_INFINITY, f64::INFINITY)

            };

            'aspiration_windows: loop {
                let score = self.search(board, alpha, beta, Player::One, self.current_ply);
                
                if self.stop || self.skip || score == f64::INFINITY || score == f64::NEG_INFINITY {
                    break 'aspiration_windows;
        
                }

                if score <= alpha {
                    alpha -= 1000.0;

                } else if score >= beta {
                    beta += 1000.0;

                } else {
                    break 'aspiration_windows;

                }

            }

            if self.skip {
                self.skip = false;
                continue 'iterative_deepening;

            }
                
            if self.stop {
                break 'iterative_deepening;
    
            }

            self.update_search_stats();

            self.data_sender.send(self.search_data.clone()).unwrap(); // Output data
    
            self.completed_searchs.push(self.search_data.clone());

            // Prune root moves
            self.root_moves.moves = self.root_moves.moves.iter().filter(|mv| mv.score != f64::NEG_INFINITY).cloned().collect();

            self.current_ply += THREAD_INCREMENTS[self.thread_id % THREAD_INCREMENTS.len()];

            if self.current_ply > self.options.maxply {
                break 'iterative_deepening;

            }

        }

        // Wait for stop signal
        while !self.stop {
            self.check();

        }

    }

    /// Search to a specific ply
    fn search(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: Player, ply: i8) -> f64 {
        let is_root = ply == self.search_data.ply;
        let is_leaf = ply == 0;
        let board_hash = board.hash();

        // Check for communication
        if self.stop || self.skip {
            return 0.0;

        } else if self.search_data.nodes % 1000 == 0 {
            self.check();

        }
        
        self.search_data.nodes += 1;

        // Base case, if the node is a leaf node, return the evaluation.
        if is_leaf {
            let eval = get_evalulation(board, &mut self.move_gen) * player.eval_multiplier();
            return eval;

        }

        // Handle Transposition Table
        let mut tt_move: Option<Move> = None;
        if self.options.tt_enabled {
            let (valid, entry) = unsafe { tt().probe(board_hash) };
            if valid && entry.depth >= ply {
                tt_move = Some(entry.bestmove);

                match entry.bound {
                    NodeBound::ExactValue => {
                        return entry.score

                    },
                    NodeBound::LowerBound => {
                        alpha = entry.score

                    },
                    NodeBound::UpperBound => {
                        beta = entry.score
                    
                    }

                }

                if alpha >= beta {
                    return entry.score;

                }

            }

        }

        // Generate the Raw move list for this node.
        let mut move_list: RawMoveList = unsafe { self.move_gen.valid_moves(board, player) };

        // If there is the threat for the current player return INF, because that move would eventualy be picked as best.
        if move_list.has_threat(player) {
            return f64::INFINITY;
            
        }

        // Use previous ply search to order the moves, otherwise generate and order them.
        let current_player_moves: Vec<Move> = if is_root {
            self.root_moves.clone().into()
            
        } else {
            let moves = move_list.moves(board);
            self.order_moves(moves, board, player, tt_move)

        };

        // If there are no valid moves, return negative infinity.
        if current_player_moves.len() == 0 {
            return f64::NEG_INFINITY;

        }

        // Loop through valid moves and search them.
        let mut best_move = Move::new_null();
        let mut best_score: f64 = f64::NEG_INFINITY;
        for (_, mv) in current_player_moves.iter().enumerate() {
            let mut new_board = board.make_move(mv);

            let score: f64 = -self.search(&mut new_board, -beta, -alpha, player.other(), ply - 1);

            // Update the score of the rootnode.
            if is_root {
                self.root_moves.update_move(*mv, score, self.current_ply);

            }

            if score > best_score {
                best_score = score;
                best_move = *mv;

            }
            if best_score > alpha {
                alpha = best_score;

            }

            if alpha >= beta {
                self.search_data.beta_cuts += 1;

                break;

            }

        }

        if self.options.tt_enabled {
            let node_bound: NodeBound = if best_score >= beta {
                NodeBound::LowerBound

            } else if best_score <= alpha  {
                NodeBound::UpperBound

            } else {
                NodeBound::ExactValue

            };

            let new_entry = Entry::new(board_hash, best_score, ply, best_move, node_bound);
            unsafe { tt().insert(new_entry) };

        }
        
        best_score

    }

    /// Orders a list of moves.
    pub fn order_moves(&mut self, moves: Vec<Move>, board: &mut BoardState, player: Player, tt_move: Option<Move>) -> Vec<Move> {
        // For every move calculate a value to sort it by.
        let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
            let mut sort_val: f64 = 0.0;
            let mut new_board = board.make_move(&mv);

            // If the move is the TT sort it first.
            if tt_move.is_some() && mv == tt_move.unwrap() {
                return (mv, TT_MOVE_SCORE);

            }

            // If opponent has a threat then remove it as an option because the move would lose.
            if unsafe{ self.move_gen.has_threat(&mut new_board, player.other()) } {
                return (mv, LOSE_MOVE_SCORE);

            }

            // If the move has a threat then increase the sort value.
            if unsafe { self.move_gen.has_threat(&mut new_board, player) } {
                sort_val += 1000.0;
            }

            // Lower the moves sort value based on oppenent move count.
            sort_val -= unsafe{ self.move_gen.valid_move_count(&mut new_board, player.other())} as f64;

            (mv, sort_val)

        }).collect();

        // Sort the moves based on their predicted values.
        moves_to_sort.sort_by(|a, b| {
            if a.1 > b.1 {
                Ordering::Less
                
            } else if a.1 == b.1 {
                Ordering::Equal

            } else {
                Ordering::Greater

            }

        });

        // Collect the moves.
        let ordered_moves: Vec<Move> = moves_to_sort.into_iter()
            .filter(|mv| mv.1 != LOSE_MOVE_SCORE)
            .map(|x| x.0)
            .collect();

        ordered_moves
    
    }

    /// Setups up the RootMoveList from a [BoardState].
    /// 
    /// Generates all moves, sorts them, and calculates the number of threats that they each have.
    /// 
    pub fn setup_rootmoves(&mut self, board: &mut BoardState) {
        let raw_moves = unsafe { self.move_gen.valid_moves(board, Player::One) }.moves(board);
        let ordered_moves: Vec<Move> = self.order_moves(raw_moves, board, Player::One, None);
        
        let root_moves: Vec<RootMove> = ordered_moves.iter().map( |mv| {
            RootMove::new(*mv, 0.0, 0)

        }).collect();

        self.root_moves.moves = root_moves;

    }

}

pub struct Searcher {
    options: SearchOptions,

    stop_in: Receiver<bool>
    
}

impl Searcher {
    /// Creates a new searcher.
    pub fn new(stop_in: Receiver<bool>, options: SearchOptions) -> Searcher {
        Searcher {
            options,
            
            stop_in

        }

    }
    
    /// Root search function.
    pub fn search(&self) {
        let mut threads = Vec::new();

        let mut stop_senders = Vec::new();

        let mut data_senders = Vec::new();
        let mut data_receivers = Vec::new();

        // Start all threads.
        for i in 0..THREAD_COUNT {
            let options = self.options.clone();
            
            let (stop_sender, stop_receiver) = std::sync::mpsc::channel();
            stop_senders.push(stop_sender);

            let (main_data_sender, thread_data_receiver) = std::sync::mpsc::channel();
            let (thread_data_sender, main_data_receiver) = std::sync::mpsc::channel();
            data_senders.push(main_data_sender);
            data_receivers.push(main_data_receiver);

            threads.push(std::thread::spawn(move || {
                let mut thread_searcher = ThreadSearcher::new(i, options, stop_receiver, thread_data_sender, thread_data_receiver);
                thread_searcher.iterative_deepening_search();

            }));

        }
        
        let mut all_search_data: Vec<SearchData> = Vec::new();
        let mut best_search_data: Option<SearchData> = Option::None;
        'main: loop {
            // Stop signal
            if self.stop_in.try_recv().is_ok() {
                break;

            }

            // Get data from threads.
            for (_, receiver) in data_receivers.iter().enumerate() {
                let mut best_updated = false;

                if let Ok(recived_data) = receiver.try_recv() {
                    if let Some(best_data) = &best_search_data {
                        if recived_data.best_move.ply > best_data.ply {
                            best_search_data = Some(recived_data.clone());
                            best_updated = true;

                        } 

                    } else {
                        best_search_data = Some(recived_data.clone());
                        best_updated = true;

                    }

                    // Update threads with the best data.
                    if best_updated {
                        ugi::info_output(best_search_data.clone().unwrap(), 999);

                        for sender in data_senders.iter() {
                            sender.send(best_search_data.clone().unwrap()).unwrap();

                        }
                        
                    }

                    if let Some(best_data) = &best_search_data {
                        if best_data.best_move.score == f64::INFINITY || best_data.best_move.score == f64::NEG_INFINITY {
                            break 'main;

                        }

                    }
                    all_search_data.push(recived_data);

                }

            }

        }

        // Send stop signal to all threads.
        for sender in stop_senders {
            sender.send(true).unwrap();

        }

        // Join all threads.
        for thread in threads {
            thread.join().unwrap();

        }

        ugi::info_output(best_search_data.clone().unwrap(), 1000);
        ugi::best_move_output(best_search_data.clone().unwrap());

    }

}

// Calc PV from TT
pub fn get_pv(board: &BoardState, ply: i8) -> Vec<RootMove> {
    let mut pv: Vec<RootMove> = Vec::new();
    let mut current_ply = ply;

    let mut current_board = board.clone();
    while current_ply > 0 {
        let (valid, entry) = unsafe { tt().probe(current_board.hash()) };
        if valid {
            let root_move = RootMove::new(entry.bestmove, entry.score, entry.depth);
            pv.push(root_move);

            let new_board = current_board.make_move(&entry.bestmove);
            current_board = new_board.clone();

        } else {
            break;

        }

        current_ply -= 1;

    }

    pv

}

/// Structure that holds all of the informaion about a specific search ply.
#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: RootMove,
    pub all_moves: Vec<RootMove>,
    pub pv: Vec<RootMove>,

    pub start_time: Instant,
    pub search_time: f64,

    pub nodes: usize,
    pub average_branching_factor: f64,

    pub nps: usize,

    pub beta_cuts: usize,

    pub ply: i8,

    pub game_over: bool,
    pub winner: usize,

}

impl SearchData {
    pub fn new(ply: i8) -> SearchData {
        SearchData {
            best_move: RootMove::new_null(),
            all_moves: Vec::new(),
            pv: Vec::new(),

            start_time: std::time::Instant::now(),
            search_time: 0.0,

            nodes: 0,
            average_branching_factor: 0.0,

            nps: 0,
       
            beta_cuts: 0,

            ply,

            game_over: false,
            winner: 0,

        }

    }

}

/// Holds all of the settings for a specific search.
#[derive(Clone)]
pub struct SearchOptions {
    pub board: BoardState,
    pub maxply: i8,
    pub maxtime: Option<f64>,
    pub tt_enabled: bool

}

impl SearchOptions {
    pub fn new() -> SearchOptions {
        SearchOptions {
            board: BoardState::from(STARTING_BOARD),   
            maxply: MAXPLY,
            maxtime: Option::None,
            tt_enabled: true

        }

    }

}

impl Default for SearchOptions {
    fn default() -> Self {
        Self::new()

    }

}
