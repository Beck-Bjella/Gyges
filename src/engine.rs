use std::sync::mpsc::Sender;
use std::time::Instant;

use crate::board::*;
use crate::consts::*;
use crate::evaluation::*;
use crate::move_gen::*;
use crate::move_list::*;
use crate::moves::*;
use crate::tt::*;

pub struct Searcher {
    pub best_move: RootMove,
    pub pv: Vec<Entry>,
    pub current_ply: usize,

    pub search_data: SearchData,
    pub root_moves: RootMoveList,

    pub dataout: Sender<SearchData>,

}

impl Searcher {
    pub fn new(dataout: Sender<SearchData>) -> Searcher {
        return Searcher {
            best_move: RootMove::new_null(),
            pv: vec![],
            current_ply: 0,

            search_data: SearchData::new(0),
            root_moves: RootMoveList::new(),

            dataout,

        };

    }

    pub fn output_search_data(&mut self) {
        self.dataout.send(self.search_data.clone()).unwrap();
    }

    pub fn update_search_stats(&mut self, board: &mut BoardState) {
        self.search_data.search_time = self.search_data.start_time.elapsed().as_secs_f64();
        self.search_data.bps =(self.search_data.branches as f64 / self.search_data.search_time) as usize;
        self.search_data.lps = (self.search_data.leafs as f64 / self.search_data.search_time) as usize;
        self.search_data.average_branching_factor = ((self.search_data.branches + self.search_data.leafs) as f64).powf(1.0 / self.search_data.depth as f64);

        self.search_data.best_move = self.root_moves.first();

        if self.search_data.best_move.score == f64::INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 1;

        } else if self.search_data.best_move.score == f64::NEG_INFINITY {
            self.search_data.game_over = true;
            self.search_data.winner = 2;

        }

        let pv = calc_pv(board, self.current_ply);
        self.pv = pv.clone();
        self.search_data.pv = pv.clone();

    }

    pub fn iterative_deepening_search(&mut self, board: &mut BoardState, max_ply: usize) {
        self.root_moves = RootMoveList::new();
        self.root_moves.setup(board);

        self.current_ply = 1;
        while !self.search_data.game_over {
            self.search_data = SearchData::new(self.current_ply);

            self.search(board,f64::NEG_INFINITY, f64::INFINITY, PLAYER_1, self.current_ply);
            self.update_search_stats(board);

            self.output_search_data();

            self.current_ply += 2;
            if self.current_ply > max_ply {
                break;

            }

        }

    }

    fn search(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: f64, depth: usize) -> f64 {
        let is_root = depth == self.search_data.depth;

        if depth == 0 {
            self.search_data.leafs += 1;

            let eval = get_evalulation(board) * player;

            return eval;
        }

        let original_alpha = alpha;
        let board_hash = board.hash();

        let (result, entry) = unsafe { tt().probe(board_hash) };
        if result {
            if entry.depth >= depth as i8 {
                self.search_data.tt_hits += 1;
                if entry.bound == NodeBound::ExactValue {
                    self.search_data.tt_exacts += 1;
                    return entry.score;
                } else if entry.bound == NodeBound::None && depth == 0 {
                    return entry.score;
                } else if entry.bound == NodeBound::LowerBound {
                    if entry.score > alpha {
                        alpha = entry.score;
                    }
                } else if entry.bound == NodeBound::UpperBound {
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

        self.search_data.branches += 1;

        let mut move_list = unsafe { valid_moves(board, player) };
        if move_list.has_threat() {
            return f64::INFINITY;
            
        }

        // let current_player_moves: Vec<Move>;
        // if is_root {
        //     current_player_moves = self.root_moves.moves();

        // } else {
        let current_player_moves = order_moves(move_list.moves(board), board, player, &self.pv);

        // }

        let mut best_move = Move::new_null();
        let mut best_score = f64::NEG_INFINITY;
        for (i, mv) in current_player_moves.iter().enumerate() {
            let mut new_board = board.make_move(&mv);

            let mut score: f64;
            if i > 1 {
                score = -self.search(&mut new_board, -alpha - 1.0, -alpha, -player, depth - 1);

                if score > alpha && score < beta {
                    score = -self.search(&mut new_board, -beta, -score, -player, depth - 1);
                }

            } else {
                score = -self.search(&mut new_board, -beta, -alpha, -player, depth - 1);

            }

            if is_root {
                self.root_moves
                    .update_move(mv.clone(), score, self.current_ply);
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

        let node_bound: NodeBound = if best_score <= original_alpha {
            NodeBound::UpperBound

        } else if best_score >= beta {
            NodeBound::LowerBound

        } else {
            NodeBound::ExactValue

        };

        let new_entry = Entry::new(board_hash, best_score, depth as i8, best_move, node_bound);

        unsafe { tt().insert(new_entry) };

        return best_score;

    }

}

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

    pub depth: usize,

    pub game_over: bool,
    pub winner: usize,

}

impl SearchData {
    pub fn new(depth: usize) -> SearchData {
        return SearchData {
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

            depth,

            game_over: false,
            winner: 0,

        };

    }

}

pub fn calc_pv(board: &mut BoardState, max_ply: usize) -> Vec<Entry> {
    let mut pv = vec![];

    let mut temp_board = board.clone();

    for i in 0..max_ply {
        let board_hash = temp_board.hash();
        let (vaild, entry) = unsafe { tt().probe(board_hash) };
        if vaild {
            pv.push(entry.clone());

            temp_board = temp_board.make_move(&entry.bestmove);

        } else {
            break;

        }

    }

    return pv;

}
