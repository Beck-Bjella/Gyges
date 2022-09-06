use crate::board::*;
use crate::move_generation::*;
use crate::evaluation::*;
use crate::transposition_tables::*;
use crate::zobrist::*;

pub struct Engine {
    pub search_stats: SearchStats,

    pub transposition_table: TranspositionTable,
    pub evaluation_table: EvaluationTable,

    pub zobrist_hasher: ZobristHasher,

}

impl Engine {
    pub fn new() -> Engine {
        return Engine {
            search_stats: SearchStats::new(),
            
            transposition_table: TranspositionTable::new(),
            evaluation_table: EvaluationTable::new(),

            zobrist_hasher: ZobristHasher::new(),

        }

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_depth: i8, max_time: f64) -> Vec<(f64, Move, usize, usize, bool)> {
        self.search_stats.reset_stats();
        self.search_stats.start_timer();

        let start_time = std::time::Instant::now();
    
        let mut moves = valid_moves_2(board, 1);
    
        let mut all_bests: Vec<(f64, Move, usize, usize, bool)> = vec![];
    
        for mv in moves.iter() {
            if mv.0[3] == PLAYER_2_GOAL {
                all_bests.push((f64::INFINITY, *mv, 0, 0, true));
                return all_bests;
    
            }
    
        }
    
        let mut current_depth = 1;
        'depth: loop {
            let mut evaluations: Vec<(f64, Move, usize, usize, bool)> = vec![];
            let mut best_move = (f64::NEG_INFINITY, moves[0], 0, current_depth as usize, false);
            
            let mut alpha = f64::NEG_INFINITY;
            let beta = f64::INFINITY;
    
            let mut used_moves = vec![];
            for mv in moves.iter() {
                if used_moves.contains(mv) {
                    continue;
    
                }
        
                board.make_move(&mv);
    
                used_moves.push(*mv);
                
                let negscout_eval = -self.negascout(board, -beta, -alpha, -1.0, current_depth - 1);
                let eval: (f64, Move, usize, usize, bool) = (negscout_eval, *mv, valid_threat_count_2(board, 1), current_depth as usize, false);
                
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
    
            current_depth += 1;
            if current_depth > max_depth {
                break 'depth;
    
            }
    
        }

        self.search_stats.stop_timer();
    
        return all_bests;
    
    }
    
    pub fn negascout(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: i8) -> f64 {
        if depth == 0 {
            self.search_stats.nodes_evaluated += 1;

            let score = get_evalulation(board) * player;

            return score;

        }

        let original_alpha = alpha;

        let board_hash = self.zobrist_hasher.get_hash_new(board, player);
      
        let look_up = self.transposition_table.get(&board_hash);
        if look_up.is_some() {
            let entry = look_up.unwrap();
            if entry.depth >= depth {
                self.search_stats.minimax_hits += 1;
                if entry.flag == TTEntryType::ExactValue {
                    self.search_stats.exact_hits += 1;
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
                    self.search_stats.tt_cuts += 1;
                    return entry.value;

                }

            }

        }

        let current_player_moves: Vec<Move>;
        if player == 1.0 {
            if valid_threat_count_2(board, 1) > 0 {
                return f64::INFINITY;
        
            }

            current_player_moves = valid_moves_2(board, 1);
            
        } else {
            if valid_threat_count_2(board, 2) > 0 {
                return f64::INFINITY;
        
            }

            current_player_moves = valid_moves_2(board, 2);
            
        }

        let mut value = f64::NEG_INFINITY;
        let mut used_moves = vec![];
        for (mv_idx, mv) in current_player_moves.iter().enumerate() {
            if used_moves.contains(mv) {
                continue;
            }

            used_moves.push(*mv);

            board.make_move(&mv);

            let mut score: f64;

            if mv_idx > 0 {
                score = -self.negascout(board, -alpha - 1.0, -alpha, -player, depth - 1);
                if alpha < score && score < beta {
                    score = -self.negascout(board, -beta, -score, -player, depth - 1);

                }

            } else {
                score =  -self.negascout(board, -beta, -alpha, -player, depth - 1);

            }

            board.undo_move(&mv);

            if score > value {
                value = score;

            }

            if value > alpha {
                alpha = value;

            }

            if alpha >= beta {
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

        return value;

    }

}

pub struct SearchStats {
    pub nodes_evaluated: usize,

    pub search_time: f64,
    pub start_time: std::time::Instant,

    pub nps: usize,

    pub minimax_hits: usize,
    pub exact_hits: usize,
    pub tt_cuts: usize,

}

impl SearchStats {
    fn new() -> SearchStats {
        return SearchStats {
            nodes_evaluated: 0,

            search_time: 0.0,
            start_time: std::time::Instant::now(),

            nps: 0,

            minimax_hits: 0,
            exact_hits: 0,
            tt_cuts: 0,

        }

    }

    fn reset_stats(&mut self) {
        self.nodes_evaluated = 0;

        self.minimax_hits = 0;
        self.exact_hits = 0;
        self.tt_cuts = 0;

    }

    fn start_timer(&mut self) {
        self.start_time = std::time::Instant::now();

    }

    fn stop_timer(&mut self) {
        self.search_time = self.start_time.elapsed().as_secs_f64();
        self.nps = (self.nodes_evaluated as f64 / self.search_time) as usize

    }

}
