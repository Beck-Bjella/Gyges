use std::cmp::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;
use std::vec;

use crate::board::*;
use crate::move_generation::*;
use crate::evaluation::*;
use crate::transposition_tables::*;
use crate::zobrist::*;

const MAX_SEARCH_PLY: usize = 99;

pub struct Engine {
    pub root_node_moves: Vec<Move>,
    pub stop: bool,

    pub search_data: SearchData,

    pub transposition_table: TranspositionTable,
    pub eval_table: EvaluationTable,

    pub datain: Receiver<BoardState>,
    pub stopin: Receiver<bool>,
    pub dataout: Sender<SearchData>,
    
}

impl Engine {
    pub fn new(datain: Receiver<BoardState>, stopin: Receiver<bool>, dataout: Sender<SearchData>) -> Engine {
        return Engine {
            root_node_moves: vec![],
            stop: false,

            search_data: SearchData::new(),

            transposition_table: TranspositionTable::new(),
            eval_table: EvaluationTable::new(),

            datain: datain,
            stopin: stopin,
            dataout: dataout,

        };

    }

    pub fn start(&mut self) {
        loop {
            let recived = self.datain.try_recv();
            match recived {
                Ok(_) => {
                    let mut recived_board = recived.unwrap();
                    self.iterative_deepening_search(&mut recived_board, MAX_SEARCH_PLY);

                },
                Err(TryRecvError::Disconnected) => {
                    println!("QUITING");
                    break;
                    
                },
                Err(TryRecvError::Empty) => {}

            }
        
        }

    }

    pub fn check_stop(&mut self) {
        let quit = self.stopin.try_recv();
        match quit {
            Ok(true) => {
                self.stop = true;

            },
            Ok(false) => {
                self.stop = false;

            },
            Err(TryRecvError::Disconnected) => {},
            Err(TryRecvError::Empty) => {}

        }

    }

    pub fn reset(&mut self) {
        loop {
            let clear_data = self.stopin.try_recv();
            match clear_data {
                Ok(_) => {},
                Err(TryRecvError::Disconnected) => {},
                Err(TryRecvError::Empty) => {
                    break;

                }
    
            }

        }
        
        self.stop = false;
        self.root_node_moves = vec![];
        self.transposition_table = TranspositionTable::new();
        self.eval_table = EvaluationTable::new();
        self.search_data = SearchData::new();

    }

    pub fn update_search_stats(&mut self) {
        self.search_data.search_time = self.search_data.start_time.elapsed().as_secs_f64();
        self.search_data.nps = (self.search_data.nodes as f64 / self.search_data.search_time) as usize;
        self.search_data.lps = (self.search_data.leafs as f64 / self.search_data.search_time) as usize;
        self.search_data.average_branching_factor = (self.search_data.nodes as f64).powf(1.0 / self.search_data.depth as f64);
            
        self.search_data.root_node_evals.sort_unstable_by(|a, b| {
            if a.score > b.score {
                Ordering::Less
                
            } else if a.score == b.score {
                Ordering::Equal
    
            } else {
                Ordering::Greater
    
            }
    
        });
        
        self.search_data.best_move = self.search_data.root_node_evals[0];

    }

    pub fn output_search_data(&mut self) {
        self.dataout.send(self.search_data.clone()).unwrap();

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_ply: usize) {
        self.reset();

        if board.data[PLAYER_2_GOAL] == 0 && board.data[PLAYER_1_GOAL] == 0 {
            self.root_node_moves = order_moves(valid_moves(board, 1.0), board, 1.0);
            
            let mut current_ply = 1;
            'depth: loop {
                let search_data = self.search_at_ply(board, current_ply);

                if self.stop {
                    break;

                }

                if search_data.best_move.score == f64::INFINITY || search_data.best_move.score == f64::NEG_INFINITY {
                    self.search_data.game_over = true;
                    self.search_data.game_over_depth = current_ply;

                    self.output_search_data();
                    break;
                    
                }
                self.output_search_data();
            
                self.root_node_moves = sort_moves_highest_score_first(search_data.root_node_evals);

                current_ply += 2;
                if current_ply > max_ply {
                    break 'depth;
        
                }
        
            }

        } else {
            self.search_data.game_over = true;
            self.search_data.game_over_depth = 0;

            self.output_search_data();

        }
    
    }

    fn search_at_ply(&mut self, board: &mut BoardState, depth: usize) -> SearchData {
        self.search_data = SearchData::new();
        self.search_data.depth = depth;
        self.search_data.start_time = std::time::Instant::now();

        self.negamax(board, -f64::INFINITY, f64::INFINITY, 1.0, depth as i8, true);
        
        self.update_search_stats();

        return self.search_data.clone();

    }

    fn negamax(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: i8, root_node: bool) -> f64 {
        self.search_data.nodes += 1;

        if !root_node {
            if self.stop {
                return 0.0;

            } else {
                self.check_stop();

            }

        }

        let board_hash = get_hash(board);

        if depth == 0 {
            self.search_data.leafs += 1;
            
            if valid_threat_count(board, player) > 0  {
                return f64::INFINITY;
        
            }

            let look_up = self.eval_table.get(&board_hash);
            if look_up.is_some() {
                return *look_up.unwrap();

            } else {
                let score = get_evalulation(board) * player;
                
                self.eval_table.insert(board_hash, score);

                return score;

            }

        }

        let original_alpha = alpha;

        let look_up = self.transposition_table.get(&board_hash);
        if look_up.is_some() {
            let entry = look_up.unwrap();
            if entry.depth >= depth {
                self.search_data.tt_hits += 1;
                if entry.flag == TTEntryType::ExactValue {
                    self.search_data.tt_exacts += 1;
                    return entry.value;

                } else if entry.flag == TTEntryType::LowerBound {
                    if entry.value > alpha {
                        alpha = entry.value;

                    }

                } else if entry.flag == TTEntryType::UpperBound {
                    if entry.value < beta {
                        beta = entry.value;

                    }

                }
                    
                if alpha >= beta {
                    self.search_data.tt_cuts += 1;
                    return entry.value;

                }

            }

        }

        let current_player_moves: Vec<Move>;
        if root_node {
            current_player_moves = self.root_node_moves.clone();

        } else {
            if valid_threat_count(board, player) > 0  {
                return f64::INFINITY;
        
            }

            current_player_moves = order_moves(valid_moves(board, player), board, player);

        }

        let mut best_score = f64::NEG_INFINITY;
        let mut used_moves = vec![];
        let mut root_node_evals = vec![];
        for mv in current_player_moves.iter() {
            if used_moves.contains(mv) {
                continue;
            }
            used_moves.push(*mv);

            board.make_move(&mv);

            let score = -self.negamax(board, -beta, -alpha, -player, depth - 1, false);

            board.undo_move(&mv);

            if root_node {
                let mut scored_move = mv.clone();
                scored_move.score = score;
                root_node_evals.push(scored_move);

            }
            
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

        let mut entry = TTEntry {value: best_score, flag: TTEntryType::None, depth: depth};

        if best_score <= original_alpha {
            entry.flag = TTEntryType::UpperBound;

        } else if best_score >= beta {
            entry.flag = TTEntryType::LowerBound;

        } else {
            entry.flag = TTEntryType::ExactValue;

        }

        self.transposition_table.insert(board_hash, entry);

        if root_node {
            self.search_data.root_node_evals = root_node_evals;

        }

        return best_score;

    }

}

#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: Move,
    pub root_node_evals: Vec<Move>,

    pub start_time: Instant,
    pub search_time: f64,

    pub nodes: usize,
    pub leafs: usize,
    pub average_branching_factor: f64,

    pub lps: usize,
    pub nps: usize,

    pub tt_hits: usize,
    pub tt_exacts: usize,
    pub tt_cuts: usize,

    pub beta_cuts: usize,

    pub depth: usize,

    pub game_over: bool,
    pub game_over_depth: usize,

}

impl SearchData {
    pub fn new() -> SearchData {
        return SearchData {
            best_move: Move::new_worst(),
            root_node_evals: vec![],

            start_time: std::time::Instant::now(),
            search_time: 0.0,

            nodes: 0,
            leafs: 0,
            average_branching_factor: 0.0,

            nps: 0,
            lps: 0,

            tt_hits: 0,
            tt_exacts: 0,
            tt_cuts: 0,
            
            beta_cuts: 0,

            depth: 0,

            game_over: false,
            game_over_depth: 0,

        }

    }

}
