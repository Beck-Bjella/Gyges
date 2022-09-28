use std::vec;

use crate::board::*;
use crate::move_generation::*;
use crate::evaluation::*;
use crate::transposition_tables::*;
use crate::zobrist::*;

pub struct Negamax {
    pub root_node_moves: Vec<Move>,

    pub search_data: SearchData,

    pub transposition_table: TranspositionTable,
    pub eval_table: EvaluationTable,

    pub zobrist_hasher: ZobristHasher,
    
}

impl Negamax {
    pub fn new() -> Negamax {
        return Negamax {
            root_node_moves: vec![],

            search_data: SearchData::new(),

            transposition_table: TranspositionTable::new(),
            eval_table: EvaluationTable::new(),

            zobrist_hasher: ZobristHasher::new(),

        };

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_ply: usize) -> Vec<SearchData> {
        self.root_node_moves = valid_moves(board, 1);
        self.transposition_table = TranspositionTable::new();
        self.eval_table = EvaluationTable::new();

        let mut all_searchs: Vec<SearchData> = vec![];
    
        let mut current_ply = 1;
        'depth: loop {
            let search_data = self.search_at_ply(board, current_ply);
            all_searchs.push(search_data.clone());

            if search_data.best_move.score == f64::INFINITY || search_data.best_move.score == f64::NEG_INFINITY {
                break;
            }

            self.root_node_moves = sort_moves(search_data.root_node_evals);
    
            current_ply += 2;
            if current_ply > max_ply {
                break 'depth;
    
            }
    
        }

        return all_searchs;
    
    }

    fn search_at_ply(&mut self, board: &mut BoardState, depth: usize) -> SearchData {
        self.search_data = SearchData::new();
        self.search_data.depth = depth;
        let start_time = std::time::Instant::now();

        self.negamax(board, -f64::INFINITY, f64::INFINITY, 1.0, depth as i8, true);
        
        let search_time = start_time.elapsed().as_secs_f64();
        self.search_data.search_time = search_time;
       
        self.search_data.nps = (self.search_data.nodes as f64 / self.search_data.search_time) as usize;
        self.search_data.lps = (self.search_data.leafs as f64 / self.search_data.search_time) as usize;

        let mut best_move: Move = Move::new_worst();
        for mv in &self.search_data.root_node_evals {
            if mv.score > best_move.score {
                best_move = *mv;

            }

        }

        self.search_data.best_move = best_move;

        return self.search_data.clone();

    }

    fn negamax(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: i8, root_node: bool) -> f64 {
        self.search_data.nodes += 1;

        let board_hash = self.zobrist_hasher.get_hash_new(board, player);

        if depth == 0 {
            self.search_data.leafs += 1;
            
            if player == 1.0 {
                if valid_threat_count(board, 1) > 0  {
                    return f64::INFINITY;
            
                }
        
            } else {
                if valid_threat_count(board, 2) > 0  {
                    return f64::INFINITY;
            
                }
        
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
            if player == 1.0 {
                if valid_threat_count(board, 1) > 0 {
                    return f64::INFINITY;
            
                }

                current_player_moves = valid_moves(board, 1);
                
            } else {
                if valid_threat_count(board, 2) > 0 {
                    return f64::INFINITY;
            
                }

                current_player_moves = valid_moves(board, 2);
                
            }

        }

        let mut value = -f64::INFINITY;
        let mut used_moves = vec![];
        let mut moves_with_scores = vec![];
        for mv in current_player_moves.iter() {
            if used_moves.contains(mv) {
                continue;
            }
            used_moves.push(*mv);

            board.make_move(&mv);

            let score = -self.negamax(board, -beta, -alpha, -player, depth - 1, false);

            board.undo_move(&mv);

            let mut scored_move = mv.clone();
            scored_move.score = score;
            moves_with_scores.push(scored_move);

            if score > value {
                value = score;

            }

            if value > alpha {
                alpha = value;

            }

            if alpha >= beta {
                self.search_data.beta_cuts += 1;
                break;

            }

        }

        let mut entry = TTEntry {value: value, flag: TTEntryType::None, depth: depth};

        if value <= original_alpha {
            entry.flag = TTEntryType::UpperBound;

        } else if value >= beta {
            entry.flag = TTEntryType::LowerBound;

        } else {
            entry.flag = TTEntryType::ExactValue;

        }

        self.transposition_table.insert(board_hash, entry);

        if root_node {
            self.search_data.root_node_evals = moves_with_scores;

        }

        return value;

    }

}

#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: Move,
    pub root_node_evals: Vec<Move>,

    pub search_time: f64,

    pub nodes: usize,
    pub leafs: usize,

    pub lps: usize,
    pub nps: usize,

    pub beta_cuts: usize,

    pub tt_hits: usize,
    pub tt_exacts: usize,
    pub tt_cuts: usize,

    pub depth: usize

}

impl SearchData {
    fn new() -> SearchData {
        return SearchData {
            best_move: Move::new_worst(),
            root_node_evals: vec![],

            search_time: 0.0,

            nodes: 0,
            leafs: 0,

            nps: 0,
            lps: 0,

            beta_cuts: 0,

            tt_hits: 0,
            tt_exacts: 0,
            tt_cuts: 0,

            depth: 0

        }

    }

}
