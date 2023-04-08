use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use crate::board::*;
use crate::move_gen::*;
use crate::evaluation::*;
use crate::zobrist::*;
use crate::consts::*;
use crate::tt::*;


pub struct Worker {
    pub root_node_moves: Vec<Move>,
    pub search_data: SearchData,

    pub dataout: Sender<SearchData>,
    pub id: usize

}

impl Worker {
    pub fn new(dataout: Sender<SearchData>, id: usize) -> Worker {
        return Worker {
            root_node_moves: vec![],
            search_data: SearchData::new(),

            dataout,
            id

        };

    }

    pub fn output_search_data(&mut self) {
        self.dataout.send(self.search_data.clone()).unwrap();
        
    }

    pub fn update_search_stats(&mut self) {
        self.search_data.search_time = self.search_data.start_time.elapsed().as_secs_f64();
        self.search_data.bps = (self.search_data.branches as f64 / self.search_data.search_time) as usize;
        self.search_data.lps = (self.search_data.leafs as f64 / self.search_data.search_time) as usize;
        self.search_data.average_branching_factor = ((self.search_data.branches + self.search_data.leafs) as f64).powf(1.0 / self.search_data.depth as f64);
            
        self.search_data.root_node_evals.sort_by(|a, b| {
            if a.score > b.score {
                Ordering::Less
                
            } else if a.score == b.score {
                Ordering::Equal
    
            } else {
                Ordering::Greater
    
            }
    
        });
        
        if self.search_data.root_node_evals.len() > 0 {
            self.search_data.best_move = self.search_data.root_node_evals[0];
    
        }

        if self.search_data.best_move.score == f64::INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 1;

        } else if self.search_data.best_move.score == f64::NEG_INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 2;

        }

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_ply: usize) {
        self.root_node_moves = unsafe{ valid_moves(board, PLAYER_1).moves(board) };
        
        let mut current_ply = 1;
        while !self.search_data.game_over {
            self.search_at_ply(board, current_ply);
        
            self.root_node_moves = sort_moves_highest_score_first(self.search_data.root_node_evals.clone());

            current_ply += 2;
            if current_ply > max_ply {
                break;
    
            }
    
        }
    
    }

    pub fn search_at_ply(&mut self, board: &mut BoardState, depth: usize) {
        self.search_data = SearchData::new();
        self.search_data.depth = depth;
        self.search_data.search_id = self.id;
        self.search_data.start_time = std::time::Instant::now();

        self.root_negamax(board, f64::NEG_INFINITY, f64::INFINITY, PLAYER_1, depth as i8);

        self.update_search_stats();

        self.output_search_data();

    }

    fn root_negamax(&mut self, board: &mut BoardState, mut alpha: f64, beta: f64, player: f64, depth: i8) -> f64 {
        self.search_data.branches += 1;
       
        let current_player_moves = self.root_node_moves.clone();
        
        let mut best_score = f64::NEG_INFINITY;
        let mut root_node_evals = vec![];
        for mv in current_player_moves.iter() {
            let mut new_board = board.make_move(&mv);

            let score = -self.negamax(&mut new_board, -beta, -alpha, -player, depth - 1);

            let mut scored_move = mv.clone();
            scored_move.score = score;
            root_node_evals.push(scored_move);

            if score > best_score {
                best_score = score;

            }

            if best_score > alpha {
                alpha = best_score;

            }

            if alpha >= beta {
                self.search_data.beta_cuts += 1;
                break;

            }

        }

        self.search_data.root_node_evals = root_node_evals.clone();

        return best_score;

    }

    fn negamax(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: i8) -> f64 {
        if depth == 0 {
            self.search_data.leafs += 1;

            let eval = old_evalulation(board, player);
            
            return eval;

        }

        let board_hash = board.hash();

        self.search_data.branches += 1;

        let original_alpha = alpha;
    
        let (result, entry) = unsafe{ tt().probe(board_hash) };
        if result {
            if entry.depth >= depth {
                self.search_data.tt_hits += 1;
                if entry.flag == TTEntryType::ExactValue {
                    self.search_data.tt_exacts += 1;
                    return entry.score;

                } else if entry.flag == TTEntryType::LowerBound {
                    if entry.score > alpha {
                        alpha = entry.score;

                    }

                } else if entry.flag == TTEntryType::UpperBound {
                    if entry.score < beta {
                        beta = entry.score;

                    }

                }
                    
                if alpha >= beta {
                    self.search_data.tt_cuts += 1;
                    return entry.score;

                }
            
            }

        }
     
        let mut move_list = unsafe{valid_moves(board, player)};
        if move_list.has_threat() {
            return f64::INFINITY;
    
        }

        let current_player_moves = order_moves(move_list.moves(board), board, player);

        let mut bestmove = Move::new_null();
        let mut best_score = f64::NEG_INFINITY;
        for mv in current_player_moves.iter() {
            let mut new_board = board.make_move(&mv);

            let score = -self.negamax(&mut new_board, -beta, -alpha, -player, depth - 1);

            if score > best_score {
                best_score = score;
                bestmove = *mv;

            }

            if best_score > alpha {
                alpha = best_score;

            }

            if alpha >= beta {
                self.search_data.beta_cuts += 1;
                break;

            }

        }
        
        let mut new_entry = Entry::new(board_hash, best_score, depth, bestmove, TTEntryType::ExactValue);
        
        if best_score <= original_alpha {
            new_entry.flag = TTEntryType::UpperBound;

        } else if best_score >= beta {
            new_entry.flag = TTEntryType::LowerBound;

        } else {
            new_entry.flag = TTEntryType::ExactValue;

        }

        let (result, entry) = unsafe{ tt().insert(board_hash) };
        entry.replace(new_entry);

        return best_score;

    }
    
}

#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: Move,

    pub root_node_evals: Vec<Move>,

    pub start_time: Instant,
    pub search_time: f64,

    pub branches: usize,
    pub leafs: usize,
    pub quiescence_nodes: usize,
    pub average_branching_factor: f64,

    pub lps: usize,
    pub bps: usize,

    pub tt_hits: usize,
    pub tt_exacts: usize,
    pub tt_cuts: usize,

    pub beta_cuts: usize,

    pub depth: usize,

    pub game_over: bool,
    pub winner: usize,

    pub search_id: usize

}

impl SearchData {
    pub fn new() -> SearchData {
        return SearchData {
            best_move: Move::new_null(),

            root_node_evals: vec![],

            start_time: std::time::Instant::now(),
            search_time: 0.0,

            branches: 0,
            leafs: 0,
            quiescence_nodes: 0,
            average_branching_factor: 0.0,

            lps: 0,
            bps: 0,

            tt_hits: 0,
            tt_exacts: 0,
            tt_cuts: 0,
            
            beta_cuts: 0,

            depth: 0,

            game_over: false,
            winner: 0,

            search_id: 0

        }

    }

}

pub fn tt_order_moves(moves: Vec<Move>, board: &mut BoardState, player: f64) -> Vec<Move> {
    let mut moves_to_sort: Vec<(Move, f64)> = Vec::with_capacity(moves.len());
    let mut ordered_moves: Vec<Move> = Vec::with_capacity(moves.len());
    
    for mv in moves {
        let mut sort_val: f64 = f64::NEG_INFINITY;

        let mut new_board = board.make_move(&mv);

        let board_hash = get_hash(&mut new_board, player);

        let (vaild, entry) = unsafe{ tt().probe(board_hash) };
        if vaild {
            sort_val = entry.score as f64
        
        }

        moves_to_sort.push((mv, sort_val));
        
    }

    moves_to_sort.sort_unstable_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    for item in &moves_to_sort {
        ordered_moves.push(item.0);
        
    }

    return ordered_moves;
    
}
