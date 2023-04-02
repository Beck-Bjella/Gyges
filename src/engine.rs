use std::cmp::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use crate::board::*;
use crate::move_gen::*;
use crate::evaluation::*;
use crate::transposition_table::*;
use crate::zobrist::*;

#[derive(Clone, PartialEq)]
pub enum EvalType {
    Standard,

}

pub struct Engine {
    pub root_node_moves: Vec<Move>,
    pub stop_search: bool,

    pub search_data: SearchData,
    pub tt: TranspositionTable,

    pub datain: Receiver<SearchInput>,
    pub stopin: Receiver<bool>,
    pub dataout: Sender<SearchData>,

    pub eval_type: EvalType
    
}

impl Engine {
    pub fn new(datain: Receiver<SearchInput>, stopin: Receiver<bool>, dataout: Sender<SearchData>) -> Engine {
        return Engine {
            root_node_moves: vec![],
            stop_search: false,

            search_data: SearchData::new(),
            tt: TranspositionTable::new_from_mb(TRANSPOSTION_TABLE_DEFAULT_SIZE_MB),

            datain: datain,
            stopin: stopin,
            dataout: dataout,

            eval_type: EvalType::Standard,

        };

    }

    pub fn start(&mut self) {
        println!("ENGINE STARTED");

        loop {
            let recived = self.datain.try_recv();
            match recived {
                Ok(_) => {
                    let search_input = recived.unwrap();
                    let mut board = search_input.board.clone();
                    let max_ply = search_input.max_ply;

                    self.eval_type = search_input.eval_type;

                    self.iterative_deepening_search(&mut board, max_ply);
           
                },
                Err(TryRecvError::Disconnected) => {
                    println!("ENGINE QUITING");
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
                self.stop_search = true;

            },
            Ok(false) => {
                self.stop_search = false;

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
        
        self.stop_search = false;
        self.root_node_moves = vec![];
        self.tt = TranspositionTable::new_from_mb(TRANSPOSTION_TABLE_DEFAULT_SIZE_MB);
        self.search_data = SearchData::new();

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

    }

    pub fn output_search_data(&mut self) {
        self.dataout.send(self.search_data.clone()).unwrap();

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_ply: usize) {
        self.reset();

        if board.data[PLAYER_2_GOAL] == 0 && board.data[PLAYER_1_GOAL] == 0 {
            self.root_node_moves = order_moves(unsafe{valid_moves(board, PLAYER_1).moves(board)}, board, PLAYER_1);
            
            let mut current_ply = 1;
            'depth: loop {
                self.search_at_ply(board, current_ply);

                if self.stop_search {
                    break 'depth;

                }

                if self.search_data.best_move.score == f64::INFINITY {
                    self.search_data.game_over = true;
                    self.search_data.winner = 1;

                    self.output_search_data();
                    break 'depth;

                } else if self.search_data.best_move.score == f64::NEG_INFINITY {
                    self.search_data.game_over = true;
                    self.search_data.winner = 2;

                    self.output_search_data();
                    break 'depth;

                }

                self.output_search_data();
            
                self.root_node_moves = sort_moves_highest_score_first(self.search_data.root_node_evals.clone());

                current_ply += 2;
                if current_ply > max_ply {
                    break 'depth;
        
                }
        
            }

        } else {
            self.search_data.game_over = true;

            if board.data[PLAYER_2_GOAL] != 0 {
                self.search_data.winner = 1;

            } else if board.data[PLAYER_1_GOAL] != 0 {
                self.search_data.winner = 2;

            }

            self.output_search_data();

        }
    
    }

    fn search_at_ply(&mut self, board: &mut BoardState, depth: usize) {
        self.search_data = SearchData::new();
        self.search_data.depth = depth;
        self.search_data.start_time = std::time::Instant::now();

        self.negamax(board, f64::NEG_INFINITY, f64::INFINITY, PLAYER_1, depth as i8, true);
        self.update_search_stats();
    
    }

    // fn quiesence_search(&mut self, board: &mut BoardState, player: f64, depth: i8) -> f64 {
        

    //     if is_quiet(board, player) || depth == 0 {
    //         return get_evalulation(board, player);

    //     }

    //     self.search_data.quiescence_nodes += 1;

    //     let mut move_list = unsafe{valid_moves(board, player)};
    //     let current_player_moves = move_list.moves(board);
        
    //     let mut best_score = f64::NEG_INFINITY;
    //     for mv in current_player_moves.iter() {
    //         board.make_move(&mv);

    //         let score = -self.quiesence_search(board, -player, depth - 1);
            
    //         board.undo_move(&mv);

    //         if score > best_score {
    //             best_score = score;

    //         }

    //     }

    //     return best_score;
    
    // }

    fn negamax(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: i8, root_node: bool) -> f64 {
        if !root_node {
            if self.stop_search {
                return 0.0;

            } else {
                self.check_stop();

            }

        }

        if depth == 0 {
            self.search_data.leafs += 1;

            let eval = get_standered_evalulation(board, player);
            
            return eval;

        }

        let board_hash = board.hash();

        self.search_data.branches += 1;

        let original_alpha = alpha;
        
        let probed = self.tt.probe(board_hash);

        if let Some(entry) = probed {
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
            let mut move_list = unsafe{valid_moves(board, player)};
            for mut mv in move_list.moves(board) {
                if mv.data[3] == PLAYER_2_GOAL {
                    mv.score = f64::INFINITY;
                    self.search_data.root_node_evals = vec![mv];

                    return f64::INFINITY;
    
                }
    
            }

            current_player_moves = self.root_node_moves.clone();

        } else {
            let mut move_list = unsafe{valid_moves(board, player)};
            if move_list.has_threat() {
                return f64::INFINITY;
        
            }
            current_player_moves = order_moves(move_list.moves(board), board, player);

        }

        let mut best_score = f64::NEG_INFINITY;
        let mut root_node_evals = vec![];
        for mv in current_player_moves.iter() {
            let mut new_board = board.make_move(&mv);

            let score = -self.negamax(&mut new_board, -beta, -alpha, -player, depth - 1, false);

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
        
        let mut entry = TTEntry {
            key: board_hash,
            value: best_score, 
            flag: TTEntryType::ExactValue, 
            depth, 
            empty: false

        };

        if best_score <= original_alpha {
            entry.flag = TTEntryType::UpperBound;

        } else if best_score >= beta {
            entry.flag = TTEntryType::LowerBound;

        } else {
            entry.flag = TTEntryType::ExactValue;

        }

        self.tt.insert(board_hash, entry);

        if root_node {
            self.search_data.root_node_evals = root_node_evals.clone();

        }

        return best_score;

    }

    // pub fn tt_order_moves(&mut self, moves: Vec<Move>, board: &mut BoardState, player: f64) -> Vec<Move> {
    //     let mut moves_to_sort: Vec<(Move, f64)> = Vec::with_capacity(moves.len());
    //     let mut ordered_moves: Vec<Move> = Vec::with_capacity(moves.len());
        
    //     for mv in moves {
    //         let mut sort_val: f64 = 0.0;

    //         board.make_move(&mv);

    //         let board_hash = get_hash(board, player);

    //         let probed = self.tt.probe(board_hash);
    //         if probed.is_some() {
    //             let entry = probed.unwrap();

    //             sort_val = f64::MAX - (entry.value * entry.depth as f64);
            
    //         }
           
    //         board.undo_move(&mv);
    
    //         moves_to_sort.push((mv, sort_val));
            
    //     }
    
    //     moves_to_sort.sort_by_cached_key(|m| m.1 as usize);
    
    //     for item in &moves_to_sort {
    //         ordered_moves.push(item.0);
            
    //     }
    
    //     return ordered_moves;
     
    // }
    
}

#[derive(Clone)]
pub struct SearchInput {
    pub board: BoardState,
    pub max_ply: usize,
    pub eval_type: EvalType

}

impl SearchInput {
    pub fn new(board: BoardState, max_ply: usize, eval_type: EvalType) -> SearchInput {
        return SearchInput {
            board: board.clone(),
            max_ply,
            eval_type
        }

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

        }

    }

}
