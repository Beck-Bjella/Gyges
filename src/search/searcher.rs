use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::time::Instant;
use std::fmt::Display;

use crate::board::board::*;
use crate::consts::*;
use crate::search::evaluation::*;
use crate::moves::moves::*;
use crate::moves::move_gen::*;
use crate::moves::move_list::*;
use crate::tools::tt::*;

// pub const MULTI_CUT_REDUCTION: i8 = 1;
pub const NULL_MOVE_REDUCTION: i8 = 1;

/// Structure that holds all needed information to perform a search, and conatains all of the main searching functions.
pub struct Searcher {
    pub best_move: RootMove,
    pub pv: Vec<Entry>,
    pub current_ply: i8,

    pub search_data: SearchData,
    pub root_moves: RootMoveList,

    pub options: SearchOptions,
    pub stop_in: Receiver<bool>,
    pub stop: bool

}

impl Searcher {
    pub fn new(stop_in: Receiver<bool>, options: SearchOptions) -> Searcher {
        Searcher {
            best_move: RootMove::new_null(),
            pv: vec![],
            current_ply: 0,

            options, 
            search_data: SearchData::new(0),
            root_moves: RootMoveList::new(),

            stop_in,
            stop: false

        }

    }

    /// Sends search data over the dataout. This can be recived by the main thread.
    pub fn ugi_output(&mut self) {
        println!("info depth {} nodes {} time {} bestmove {} score {}", self.search_data.ply, self.search_data.branches + self.search_data.leafs, self.search_data.search_time, self.search_data.best_move.as_human(), self.search_data.best_move.score);
        
    }

    // Checks to see if the engine should stop the search
    pub fn check_stop(&mut self) {
        match self.stop_in.try_recv() {
            Ok(_) => {
                self.stop = true;

            }
            Err(_) => {}

        }

    }

    /// Updates the search data based on the collected data.
    pub fn update_search_stats(&mut self, board: &mut BoardState) {
        self.search_data.search_time = self.search_data.start_time.elapsed().as_secs_f64();
        self.search_data.bps =(self.search_data.branches as f64 / self.search_data.search_time) as usize;
        self.search_data.lps = (self.search_data.leafs as f64 / self.search_data.search_time) as usize;
        self.search_data.average_branching_factor = ((self.search_data.branches + self.search_data.leafs) as f64).powf(1.0 / self.search_data.ply as f64);

        self.root_moves.sort();
        self.search_data.best_move = self.root_moves.first();

        if self.search_data.best_move.score == f64::INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 1;

        } else if self.search_data.best_move.score == f64::NEG_INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 2;

        }

        let pv = calc_pv_tt(board, self.current_ply);
        self.pv = pv.clone();
        self.search_data.pv = pv;

    }

    /// Iterative deepening search.
    pub fn iterative_deepening_search(&mut self) {
        self.stop = false;

        let board = &mut self.options.board.clone();

        self.root_moves = RootMoveList::new();
        self.root_moves.setup(board);

        self.current_ply = 1;
        while !self.search_data.game_over {
            self.search_data = SearchData::new(self.current_ply);

            self.search::<PV>(board,f64::NEG_INFINITY, f64::INFINITY, PLAYER_1, self.current_ply, true);
            if self.stop {
                break;

            }

            self.update_search_stats(board);
            self.ugi_output();

            self.current_ply += 1;

        }

    }
    
    /// Main search function.
    fn search<N: Node>(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: i8, cut_node: bool) -> f64 {
        let is_root = depth == self.search_data.ply;
        let is_leaf = depth == 0;
        let is_pv: bool = N::is_pv();
        let board_hash = board.hash();

        if self.stop {
            return 0.0;
    
        } else {
            if self.search_data.leafs % 1000 == 0 {
                self.check_stop();
    
            }

        }
       
        if is_leaf {
            self.search_data.leafs += 1;

            let eval = get_evalulation(board) * player;

            return eval;

        }

        let (valid, entry) = unsafe { tt().probe(board_hash) };
        if !is_pv && valid && entry.depth >= depth {
            self.search_data.tt_hits += 1;

            if entry.bound == NodeBound::ExactValue {
                self.search_data.tt_exacts += 1;
                return entry.score;

            } else if entry.bound == NodeBound::LowerBound {
                if entry.score > alpha {
                    alpha = entry.score;
                }
                
            } else if entry.bound == NodeBound::UpperBound && entry.score < beta {
                beta = entry.score;

            }

            if alpha >= beta {
                self.search_data.tt_cuts += 1;

                return entry.score;

            }

        }

        self.search_data.branches += 1;

        // Generate the Raw move list for this node.
        let mut move_list: RawMoveList = unsafe { valid_moves(board, player) };

        // If there is the threat for the current player return INF, because that move would eventualy be picked as best.
        if move_list.has_threat(player) {
            return f64::INFINITY;
            
        }

        // Null Move Pruning
        let r_depth = depth - 1 - NULL_MOVE_REDUCTION;
        if r_depth > 0 {
            let mut null_move_board = board.make_null();
            let score = -self.search::<NonPV>(&mut null_move_board, -alpha - 1.0, -alpha, -player, r_depth, !cut_node);
            if score >= beta {
                return beta;
    
            }

        }

        // Use the previous search to order the moves, otherwise generate and order them.
        let current_player_moves: Vec<Move> = if is_root {
            self.root_moves.as_vec()
            // let moves = move_list.moves(board);
            // order_moves(moves, board, player, &self.pv)
            
        } else {
            let moves = move_list.moves(board);
            order_moves(moves, board, player, &self.pv)
            
        };
        
        // Multi Cut - Dosent Work
        // let r_depth = depth - 1 - 1;
        // if r_depth > 0 && cut_node {
        //     let mut cuts = 0;

        //     'test: for (i, mv) in current_player_moves.iter().enumerate() {
        //         if i >= 50 {
        //             break 'test;
        //         }
                
        //         let mut new_board = board.make_move(&mv);
     
        //         let score = -self.search::<NonPV>(&mut new_board, -alpha - 1.0, -alpha, -player, r_depth, !cut_node);
        //         if score >= beta {
        //             cuts += 1;
        //             if cuts >= 40 {
        //                 return beta; // mc-prune

        //             }
                      
        //         }
                
        //     }

        // }

        let mut best_move = Move::new_null();
        let mut best_score: f64 = f64::NEG_INFINITY;
        for (i, mv) in current_player_moves.iter().enumerate() {
            let mut new_board = board.make_move(mv);
            
            let mut lmr = 0;

            // Check for a LMR
            // if i > (current_player_moves.len() as f64 * 0.7) as usize && !is_pv && depth >= 3 {
            //     lmr += 1;
                
            // }

            let score: f64 = if i == 0 && is_pv {
                -self.search::<PV>(&mut new_board, -beta, -alpha, -player, depth - 1 - lmr, false)

            } else {
                -self.search::<NonPV>(&mut new_board, -beta, -alpha, -player, depth - 1 - lmr, !cut_node)

            };

            // Update the score of the corosponding rootnode.
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

        let node_bound: NodeBound = if best_score >= beta {
            NodeBound::LowerBound

        } else if is_pv && !best_move.is_null() {
            NodeBound::ExactValue

        } else {
            NodeBound::UpperBound

        };

        let new_entry = Entry::new(board_hash, best_score, depth, TTMove::from(best_move), node_bound);
        unsafe { tt().insert(new_entry) };

        best_score

    }

}

/// Structure that holds all of the informaion about a specific search ply.
#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: RootMove,
    pub pv: Vec<Entry>,

    pub start_time: Instant,
    pub search_time: f64,

    pub branches: usize,
    pub leafs: usize,
    pub average_branching_factor: f64,

    pub lps: usize,
    pub bps: usize,

    pub tt_hits: usize,
    pub tt_exacts: usize,
    pub tt_cuts: usize,

    pub beta_cuts: usize,

    pub ply: i8,

    pub game_over: bool,
    pub winner: usize,

}

impl SearchData {
    pub fn new(ply: i8) -> SearchData {
        SearchData {
            best_move: RootMove::new_null(),
            pv: vec![],

            start_time: std::time::Instant::now(),
            search_time: 0.0,

            branches: 0,
            leafs: 0,
            average_branching_factor: 0.0,

            lps: 0,
            bps: 0,

            tt_hits: 0,
            tt_exacts: 0,
            tt_cuts: 0,

            beta_cuts: 0,

            ply,

            game_over: false,
            winner: 0,

        }

    }

}

impl Display for SearchData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Depth: {:?}", self.ply)?;
        writeln!(f, "  - Best: {:?}", self.best_move)?;
        writeln!(f, "  - Time: {:?}", self.search_time)?;
        writeln!(f)?;
        writeln!(f, "  - Abf: {}", self.average_branching_factor)?;
        writeln!(f)?;
        writeln!(f, "  - Branchs: {}", self.branches)?;
        writeln!(f, "  - Bps: {}", self.bps)?;
        writeln!(f)?;
        writeln!(f, "  - Leafs: {}", self.leafs)?;
        writeln!(f, "  - Lps: {}", self.lps)?;
        writeln!(f)?;
        writeln!(f, "  - TT:")?;
        writeln!(f, "      - HITS: {:?}", self.tt_hits)?;
        writeln!(f, "      - EXACTS: {:?}", self.tt_exacts)?;
        writeln!(f, "      - CUTS: {:?}", self.tt_cuts)?;
        writeln!(f)?;
        writeln!(f, "      - SAFE INSERTS: {}", unsafe { TT_SAFE_INSERTS })?;
        writeln!(f, "      - UNSAFE INSERTS: {}", unsafe { TT_UNSAFE_INSERTS })?;
        writeln!(f)?;
        writeln!(f, "  - PV")?;
        for (i, e) in self.pv.iter().enumerate() {
            writeln!(f, "      - {}: {:?}, {}", i, e.bestmove, e.score)?;

        }
    
        Result::Ok(())

    }

}

#[derive(Clone)]
pub struct SearchOptions {
    pub board: BoardState,

}

impl SearchOptions {
    pub fn new() -> SearchOptions {
        SearchOptions {
            board: BoardState::from(TEST_BOARD, PLAYER_1)

        }

    }

}

/// Uses the trasposition table to calculate the PV.
pub fn calc_pv_tt(board: &mut BoardState, max_ply: i8) -> Vec<Entry> {
    let mut pv = vec![];

    let mut temp_board = *board;

    for d in 0..max_ply {
        let board_hash = temp_board.hash();
        let (vaild, entry) = unsafe { tt().probe(board_hash) };
        if vaild {
            pv.push(*entry);

            temp_board = temp_board.make_move(&Move::from(entry.bestmove));

            // if d == max_ply-1 {
            //     println!("{}", temp_board);

            // }

        } else {
            break;

        }

    }

    pv

}
