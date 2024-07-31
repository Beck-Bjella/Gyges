//! Main strutures, and concepts related to searching.
//! 

pub mod evaluation;

use std::sync::mpsc::Receiver;
use std::time::Instant;

use gyges::board::*;
use gyges::moves::*;
use gyges::moves::movegen::*;
use gyges::moves::move_list::*;
use gyges::tools::tt::*;
use gyges::core::*;

use crate::search::evaluation::*;
use crate::consts::*;
use crate::ugi;

pub const MAXPLY: i8 = 99;

/// Structure that holds all needed information to perform a search, and conatains all of the main searching functions.
pub struct Searcher {
    pub best_move: RootMove,
    pub current_ply: i8,
    pub start_time: Instant,

    pub completed_searchs: Vec<SearchData>,
    pub search_data: SearchData,
    pub root_moves: RootMoveList,

    pub options: SearchOptions,
    pub stop_in: Receiver<bool>,
    pub stop: bool

}

impl Searcher {
    /// Creates a new searcher.
    pub fn new(stop_in: Receiver<bool>, options: SearchOptions) -> Searcher {
        Searcher {
            best_move: RootMove::new_null(),
            current_ply: 0,
            start_time: Instant::now(),
            
            completed_searchs: vec![],
            search_data: SearchData::new(1),
            root_moves: RootMoveList::new(),

            options, 
            stop_in,
            stop: false

        }

    }

    // Checks to see if the engine should stop the search.
    pub fn check_stop(&mut self) {
        // Check if the stop signal has been sent.
        if self.stop_in.try_recv().is_ok() {
            self.stop = true;

        }

        // Check if the max time has been reached.
        if let Some(maxtime) = self.options.maxtime {
            if self.search_data.start_time.elapsed().as_secs_f64() >= maxtime {
                self.stop = true;

            }

        }

    }

    /// Updates the search data based on the collected data.
    pub fn update_search_stats(&mut self) {
        self.search_data.search_time = self.search_data.start_time.elapsed().as_secs_f64();
        self.search_data.nps = (self.search_data.nodes as f64 / self.search_data.search_time) as usize;
        self.search_data.average_branching_factor = (self.search_data.nodes as f64).powf(1.0 / self.search_data.ply as f64);

        self.root_moves.sort();
        self.search_data.best_move = self.root_moves.first();

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

        for mv in unsafe{ valid_moves(board, Player::One).moves(board) } {
            if mv.is_win() {
                self.search_data.best_move = RootMove::new(mv, f64::INFINITY, 1, 0);
                self.completed_searchs.push(self.search_data.clone());

                let best_search_data = self.completed_searchs.last().unwrap().clone();
                ugi::info_output(best_search_data.clone());
                ugi::best_move_output(best_search_data);
                return;

            }

        }

        self.root_moves = RootMoveList::new();
        self.root_moves.setup(board);

        self.start_time = Instant::now();

        self.current_ply = 1;
        'iterative_deepening: while !self.search_data.game_over {
            self.search_data = SearchData::new(self.current_ply);
         
            self.search(board, f64::NEG_INFINITY, f64::INFINITY, Player::One, self.current_ply);
      
            if self.stop {
                break 'iterative_deepening;
    
            }

            self.update_search_stats();
            ugi::info_output(self.search_data.clone());
            
            self.completed_searchs.push(self.search_data.clone());

            self.current_ply += 2;
            
            if self.current_ply > self.options.maxply {
                break 'iterative_deepening;

            }

        }
        
        let best_search_data = self.completed_searchs.last().unwrap().clone();
        ugi::info_output(best_search_data.clone());
        ugi::best_move_output(best_search_data);

    }

    /// Main search function.
    fn search(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: Player, ply: i8) -> f64 {
        let is_root = ply == self.search_data.ply;
        let is_leaf = ply == 0;
        let board_hash = board.hash();

        // Check if the search should stop.
        if self.stop {
            return 0.0;
    
        } else if self.search_data.nodes % 1000 == 0 {
            self.check_stop();

        }

        // Generate the Raw move list for this node.
        let mut move_list: RawMoveList = unsafe { valid_moves(board, player) };

        // If there is the threat for the current player return INF, because that move would eventualy be picked as best.
        if move_list.has_threat(player) {
            return f64::INFINITY;
            
        }

        self.search_data.nodes += 1;

        // Base case, if the node is a leaf node, return the evaluation.
        if is_leaf {
            let eval = get_evalulation(board) * player.eval_multiplier();
            return eval;

        }

        // Handle Transposition Table
        if self.options.tt_enabled {
            let (valid, entry) = unsafe { tt().probe(board_hash) };
            if valid && entry.depth >= ply {
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
        

        // Use previous ply search to order the moves, otherwise generate and order them.
        let current_player_moves: Vec<Move> = if is_root {
            self.root_moves.clone().into()
            
        } else {
            let moves = move_list.moves(board);
            order_moves(moves, board, player)

        };
        
        // Loop through valid moves and search them.
        let mut best_move = Move::new_null();
        let mut best_score: f64 = f64::NEG_INFINITY;
        for mv in current_player_moves.iter() {
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

}

/// Structure that holds all of the informaion about a specific search ply.
#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: RootMove,

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

/// Holds all of the settings for a spsific search.
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
