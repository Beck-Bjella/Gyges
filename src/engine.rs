use crate::board::*;
use crate::move_generation::*;
use crate::evaluation::*;
use crate::transposition_tables::*;
use crate::zobrist::*;

pub struct Engine {
    pub nodes_evaluated: usize,
    pub search_time: f64,
    pub nps: usize,

    pub eval_duplicates: usize,
    pub eval_table: EvaluationTable,

    pub zobrist_hasher: ZobristHasher
    

}

impl Engine {
    pub fn new() -> Engine {
        return Engine {
            nodes_evaluated: 0,
            search_time: 0.0,
            nps: 0,

            eval_duplicates: 0,
            eval_table: EvaluationTable::new(),

            zobrist_hasher: ZobristHasher::new(),

        }

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_depth: i8, max_time: f64) -> Vec<(f64, [usize; 6], usize, usize, bool)> {
        self.nodes_evaluated = 0;
        self.search_time = 0.0;
        self.nps = 0;

        let start_time = std::time::Instant::now();
    
        let mut moves = valid_moves_2(board, 1);
    
        let mut all_bests: Vec<(f64, [usize; 6], usize, usize, bool)> = vec![];
    
        for mv in moves.iter() {
            if mv[3] == PLAYER_2_GOAL {
                all_bests.push((f64::INFINITY, *mv, 0, 0, true));
                return all_bests;
    
            }
    
        }
    
        let mut current_depth = 1;
        'depth: loop {
            let mut evaluations: Vec<(f64, [usize; 6], usize, usize, bool)> = vec![];
            let mut best_move = (f64::NEG_INFINITY, moves[0], 0, current_depth as usize, false);
            
            let mut alpha = f64::NEG_INFINITY;
            let beta = f64::INFINITY;
    
            let mut used_moves: Vec<[usize; 6]> = vec![];
            for mv in moves.iter() {
                if used_moves.contains(mv) {
                    continue;
    
                }
        
                board.make_move(&mv);
    
                used_moves.push(*mv);
                
                let minimax_eval = self.mini_max(board, alpha, beta, false, current_depth - 1);
                let eval: (f64, [usize; 6], usize, usize, bool) = (minimax_eval, *mv, valid_threat_count_2(board, 2), current_depth as usize, false);
    
                evaluations.push(eval);
    
                board.undo_move(&mv);
    
                if eval.0 > best_move.0 {
                    best_move = eval;
    
                } else if eval.0 == best_move.0 {
                    if eval.2 > best_move.2 {
                        best_move = eval;
    
                    }
    
                }
    
                if best_move.0 > alpha {
                    alpha = best_move.0;
    
                }
                if beta <= alpha {
                    best_move.4 = true;
    
                    all_bests.push(best_move);
                    break 'depth;
    
                }
    
                if start_time.elapsed().as_secs_f64() >= max_time {
                    all_bests.push(best_move);
    
                    break 'depth;
    
                }
    
            }
    
            best_move.4 = true;
            all_bests.push(best_move);
            println!("{:?}", best_move);
    
            moves = sort_moves(evaluations);
    
            current_depth += 2;
            if current_depth > max_depth {
                break 'depth;
    
            }
    
        }

        self.search_time = start_time.elapsed().as_secs_f64();
        self.nps = (self.nodes_evaluated  as f64 / self.search_time) as usize;
    
        return all_bests;
    
    }
    
    fn mini_max(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, is_maximisizing: bool, depth: i8) -> f64 {
        if depth <= 0 {
            self.nodes_evaluated += 1;

            let board_hash = self.zobrist_hasher.get_hash(board, is_maximisizing);
      
            let look_up = self.eval_table.get(&board_hash);
            if look_up.is_some() {
                self.eval_duplicates += 1;
                return *look_up.unwrap();

            } else {
                let score = get_evalulation(board);

                self.eval_table.insert(board_hash, score);

                return score;
                
            }

        }

        if is_maximisizing {
            if valid_threat_count_2(board, 1) > 0 {
                return f64::INFINITY;
        
            }
    
            let current_moves = valid_moves_2(board, 1);
    
            let mut max_eval = f64::NEG_INFINITY;
            let mut used_moves: Vec<[usize; 6]> = vec![];
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
                
                used_moves.push(*mv);
    
                board.make_move(&mv);
    
                let eval = self.mini_max(board, alpha, beta, false, depth - 1);
                if eval > max_eval {
                    max_eval = eval
                }
                
                board.undo_move(&mv);
                
                if max_eval > alpha {
                    alpha = max_eval
                }
                if beta <= alpha {
                    break
    
                }
    
            }
            
            return max_eval;
    
        } else {
            if valid_threat_count_2(board, 2) > 0 {
                return f64::NEG_INFINITY;
        
            }
    
            let current_moves = valid_moves_2(board, 2);
            
            let mut min_eval = f64::INFINITY;
            let mut used_moves: Vec<[usize; 6]> = vec![];
            for mv in current_moves.iter() {
                if used_moves.contains(mv) {
                    continue;
                }
    
                used_moves.push(*mv);
    
                board.make_move(&mv);
    
                let eval = self.mini_max(board, alpha, beta, true, depth - 1);
                if eval < min_eval {
                    min_eval = eval;
    
                }
    
                board.undo_move(&mv);
    
                if min_eval < beta {
                    beta = min_eval;
    
                }
                if beta <= alpha {
                    break
    
                }
    
            }
    
            return min_eval;
    
        }
    
    }
    

}
