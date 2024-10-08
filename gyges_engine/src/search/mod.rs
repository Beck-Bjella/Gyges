//! Main strutures, and concepts related to searching.
//! 

pub mod evaluation;

use core::f64;
use std::cmp::Ordering;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use rayon::prelude::*;

use gyges::board::*;
use gyges::moves::*;
use gyges::moves::movegen::*;
use gyges::moves::move_list::*;
use gyges::tools::tt::*;
use gyges::core::*;

use crate::search::evaluation::*;
use crate::consts::*;
use crate::ugi;

// Constants
pub const MAXPLY: i8 = 99;

// Constants for move ordering.
pub const LOSE_MOVE_SCORE: f64 = -1000000.0;
pub const TT_MOVE_SCORE: f64 = 1000000.0;

/// Structure that holds all needed information to perform a search, and conatains all of the main searching functions.
pub struct Searcher {
    pub current_ply: i8,

    pub completed_searchs: Vec<SearchData>,
    pub search_data: SearchData,
    pub search_stats: SearchStats,
    pub root_moves: RootMoveList,

    pub options: SearchOptions,
    pub stop_in: Receiver<bool>,
    pub stop: bool

}

impl Searcher {
    /// Creates a new searcher.
    pub fn new(stop_in: Receiver<bool>, options: SearchOptions) -> Searcher {
        Searcher {
            current_ply: 0,
            
            completed_searchs: vec![],
            search_data: SearchData::new(1),
            search_stats: SearchStats::new(),
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
            if self.search_stats.start_time.elapsed().as_secs_f64() >= maxtime {
                self.stop = true;

            }

        }

    }

    /// Updates the search data based on the collected data.
    pub fn update_search_data(&mut self) {
        // Stats
        self.search_stats.search_time = self.search_stats.start_time.elapsed().as_secs_f64();
        self.search_stats.nps = (self.search_stats.nodes as f64 / self.search_stats.search_time) as usize;
        
        // Results
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
        // Configure rayon
        rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

        // Setup search
        self.stop = false;

        let board = &mut self.options.board.clone();

        for mv in unsafe { valid_moves(board, Player::One) }.moves(board) {
            if mv.is_win() {
                self.search_data.best_move = RootMove::new(mv, f64::INFINITY, 1, 0);
                self.completed_searchs.push(self.search_data.clone());

                let best_search_data = self.completed_searchs.last().unwrap().clone();
                ugi::info_output(best_search_data.clone(), self.search_stats.clone());
                ugi::best_move_output(best_search_data);
                return;

            }

        }

        self.root_moves = RootMoveList::new();
        self.setup_rootmoves(board);

        self.search_stats.start_time = Instant::now();

        // Start iterative deepening
        self.current_ply = 1;
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
                
                if self.stop || score == f64::INFINITY || score == f64::NEG_INFINITY {
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
                
            if self.stop {
                break 'iterative_deepening;
    
            }   

            self.update_search_data();
            ugi::info_output(self.search_data.clone(), self.search_stats.clone());
            
            self.completed_searchs.push(self.search_data.clone());
           
            self.current_ply += 2;

            // Prune the root moves list.
            self.root_moves.moves = self.root_moves.moves.iter().filter(|mv| mv.score != f64::NEG_INFINITY).cloned().collect();
            
            if self.current_ply > self.options.maxply {
                break 'iterative_deepening;

            }

        }
        
        let best_search_data = self.completed_searchs.last().unwrap().clone();
        ugi::info_output(best_search_data.clone(), self.search_stats.clone());
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
    
        } else if self.search_stats.nodes % 1000 == 0 {
            self.check_stop();

        }

        // Generate the Raw move list for this node.
        let mut move_list: RawMoveList = unsafe { valid_moves(board, player) };

        // If there is the threat for the current player return INF, because that move would eventualy be picked as best.
        if move_list.has_threat(player) {
            return f64::INFINITY;
            
        }

        self.search_stats.nodes += 1;

        // Base case, if the node is a leaf node, return the evaluation.
        if is_leaf {
            let eval = get_evalulation(board) * player.eval_multiplier();
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
        for (i, mv) in current_player_moves.iter().enumerate() {
            let mut new_board = board.make_move(mv);

            // Late Move Reduction
            let reduction = if i > (2 * (current_player_moves.len() / 3)) && ply > 5 {
                3
            } else if i > (current_player_moves.len() / 3) && ply > 5 {
                2
            } else {
                0
            };

            // Principal Variation Search
            let score: f64 = if i < 5 {
                -self.search(&mut new_board, -beta, -alpha, player.other(), ply - 1 - reduction) // Full search

            } else {
                let mut score = -self.search(&mut new_board, -alpha - 1.0, -alpha, player.other(), ply - 1 - reduction); // Null window search
                if score > alpha && score < beta { 
                    score = -self.search(&mut new_board, -beta, -alpha, player.other(), ply - 1 - reduction);

                }

                score

            };

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
                break;

            }

        }

        if self.options.tt_enabled && !self.stop {
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
        let mut moves_to_sort: Vec<(Move, f64)> = moves.into_par_iter().map(|mv| {
            let mut sort_val: f64 = 0.0;
            let mut new_board = board.make_move(&mv);

            // If the move is the TT sort it first.
            if tt_move.is_some() && mv == tt_move.unwrap() {
                return (mv, TT_MOVE_SCORE);

            }

            // If opponent has a threat then remove it as an option because the move would lose.
            if unsafe { has_threat(&mut new_board, player.other()) } {
                return (mv, LOSE_MOVE_SCORE);

            }

            // If the move has a threat then increase the sort value.
            if unsafe { has_threat(&mut new_board, player) } {
                sort_val += 1000.0;
            }

            // Lower the moves sort value based on oppenent move count.
            sort_val -= unsafe { valid_move_count(&mut new_board, player.other()) } as f64;

            // SLIGHTLY BETTER ORDERING, BUT BIGGER PERFORMANCE HIT
            // If a move has less then 5 threats then penalize it.
            // let threat_count = unsafe{ valid_threat_count(&mut new_board, player) };
            // if threat_count <= 5_usize {
            //     sort_val -= 1000.0 * (5 - threat_count) as f64;
            // }

            (mv, sort_val)

        }).filter(|mv| mv.1 != LOSE_MOVE_SCORE).collect();

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
            .map(|x| x.0)
            .collect();

        ordered_moves
    
    }


    /// Setups up the RootMoveList from a [BoardState].
    /// 
    /// Generates all moves, sorts them, and calculates the number of threats that they each have.
    /// 
    pub fn setup_rootmoves(&mut self, board: &mut BoardState) {
        let moves = unsafe { valid_moves(board, Player::One) }.moves(board);
        let ordered: Vec<Move> = self.order_moves(moves, board, Player::One, None);
        
        let root_moves: Vec<RootMove> = ordered.iter().map( |mv| {
            let mut new_board = board.make_move(mv);
            let threats: usize = unsafe { valid_threat_count(&mut new_board, Player::One) };

            RootMove::new(*mv, 0.0, 0, threats)

        }).collect();

        let mut rootmove_list = RootMoveList::new();
        rootmove_list.moves = root_moves;

        self.root_moves = rootmove_list;

    }


}

/// Structure that holds all of the results from a specific search ply.
#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: RootMove,
    pub ply: i8,

    pub game_over: bool,
    pub winner: usize,

}

impl SearchData {
    pub fn new(ply: i8) -> SearchData {
        SearchData {
            best_move: RootMove::new_null(),
            ply,

            game_over: false,
            winner: 0,

        }

    }

}

/// Structure that holds all statistics about a search.
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub nodes: usize,
    pub nps: usize,

    pub start_time: Instant,
    pub search_time: f64

}

impl SearchStats {
    pub fn new() -> SearchStats {
        SearchStats {
            nodes: 0,
            nps: 0,

            start_time: Instant::now(),
            search_time: 0.0

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
