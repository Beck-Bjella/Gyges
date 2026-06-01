//! Main strutures, and concepts related to searching.
//! 

pub mod evaluation;
pub mod eval_display;
pub mod network;

use core::f64;
use std::cmp::Ordering;
use std::ops::Add;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::thread;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;

use movegen::GenMoveCount;
use movegen::GenMoves;
use movegen::GenNone;
use movegen::GenResult;
use movegen::GenThreatCount;
use movegen::MoveGen;
use movegen::NoQuit;
use movegen::QuitOnThreat;

use gyges::board::*;
use gyges::moves::*;
use gyges::moves::move_list::*;
use gyges::tools::tt::*;
use gyges::core::*;

use crate::search::evaluation::*;
use crate::search::network::{get_evalulation_nn, network_loaded};
use crate::consts::*;
use crate::ugi;

// Win/loss score constants.
pub const WIN_SCORE: f64 = 100_000_000.0;
pub const LOSS_SCORE: f64 = -WIN_SCORE;
pub const WIN_THRESHOLD: f64 = WIN_SCORE - 10_000.0;
pub const LOSS_THRESHOLD: f64 = -WIN_SCORE + 10_000.0;
pub const DRAW_SCORE: f64 = 0.0;

/// Per-iteration root dispatch — main claims from `next_top`, helpers from `next_back`, all share `alpha_bits`.
pub struct RootDispatch {
    next_top: AtomicUsize,
    next_back: AtomicUsize,
    alpha_bits: AtomicU64,
    initial_alpha: f64,
    beta: f64,
    start_ply: i8,
}

impl RootDispatch {
    pub fn new(initial_alpha: f64, beta: f64, start_ply: i8, n_moves: usize) -> Self {
        Self {
            next_top: AtomicUsize::new(0),
            next_back: AtomicUsize::new(n_moves),
            alpha_bits: AtomicU64::new(initial_alpha.to_bits()),
            initial_alpha,
            beta,
            start_ply,
        }

    }

    pub fn current_alpha(&self) -> f64 {
        f64::from_bits(self.alpha_bits.load(AtomicOrdering::Acquire))
    }

    /// CAS-update the shared alpha if `score` improves it.
    pub fn try_update_alpha(&self, score: f64) -> bool {
        let mut current_bits = self.alpha_bits.load(AtomicOrdering::Acquire);
        loop {
            let current = f64::from_bits(current_bits);
            if score <= current {
                return false;

            }
            match self.alpha_bits.compare_exchange_weak(
                current_bits,
                score.to_bits(),
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current_bits = actual,
            }

        }

    }
}

/// Structure that holds all needed information to perform a search, and conatains all of the main searching functions.
pub struct Searcher {
    pub options: SearchOptions,
    pub stop_signal: Arc<AtomicBool>,
    pub stop: bool,
    /// Cross-worker node total — drives `maxnodes` honestly across the pool.
    pub shared_nodes: Arc<AtomicUsize>,

    pub completed_plys: Vec<SearchData>,
    pub search_stats: SearchStats,
    pub root_moves: RootMoveList,

    pub mg: MoveGen,

    pub history: HistoryTable,

    pub path: Vec<u64>,

}

impl Searcher {
    /// Creates a new searcher.
    pub fn new(stop_signal: Arc<AtomicBool>, shared_nodes: Arc<AtomicUsize>, options: SearchOptions) -> Searcher {
        Searcher {
            options,
            stop_signal,
            stop: false,
            shared_nodes,

            completed_plys: vec![],
            search_stats: SearchStats::new(),
            root_moves: RootMoveList::new(),

            mg: MoveGen::default(),

            history: HistoryTable::default(),

            path: Vec::new(),

        }

    }

    // Checks to see if the engine should stop the search.
    pub fn check_stop(&mut self) {
        // Shared atomic stop — set by UGI `stop` or by the orchestrator on IDS exit.
        if self.stop_signal.load(AtomicOrdering::Relaxed) {
            self.stop = true;

        }

        // Check if the max time has been reached.
        if let Some(maxtime) = self.options.maxtime {
            if self.search_stats.start_time.elapsed().as_secs_f64() >= maxtime {
                self.stop = true;

            }

        }

        // Check shared node total so maxnodes is honest across the pool.
        if let Some(maxnodes) = self.options.maxnodes {
            if self.shared_nodes.load(AtomicOrdering::Relaxed) >= maxnodes {
                self.stop = true;

            }

        }

    }

    /// Update search data based on the current search stats and results.
    pub fn update_search_data(&mut self, ply_data: &mut SearchData) {
        // Gather current search stats
        ply_data.elapsed_time = self.search_stats.start_time.elapsed().as_secs_f64();
        ply_data.nodes = self.search_stats.nodes;
        ply_data.nps = (ply_data.nodes as f64 / ply_data.elapsed_time) as usize;
        
        // Results
        ply_data.pv = get_pv(&mut self.options.board.clone(), ply_data.ply);
        ply_data.best_move = ply_data.pv.get(0).unwrap_or(&RootMove::new_null()).clone();

        if ply_data.best_move.score >= WIN_THRESHOLD {
            ply_data.game_over = true;
            ply_data.winner = 1;

        } else if ply_data.best_move.score <= LOSS_THRESHOLD {
            ply_data.game_over = true;
            ply_data.winner = 2;

            // Handle best move when their are no valid moves (best losing move)
            if self.completed_plys.len() > 0 {
                let mut prev_ply_mv = self.completed_plys.last().unwrap().best_move.clone();
                prev_ply_mv.score = LOSS_SCORE;

                ply_data.best_move = prev_ply_mv;
       
            } else { 
                ply_data.best_move = self.root_moves.moves.first().unwrap().clone();

            }

        }

    } 

    /// Displays the final output of the search.
    pub fn final_output(&self) {
        // Fall back to the heuristically-best root move when no ply finished.
        // This happens if a `stop` (UGI command or quit) races search startup
        // before ply 1 completes — under cooperative-root SMP that race is
        // observable; the original serial path was protected by `check_stop`
        // waiting for a non-empty `completed_plys`.
        let mut best_search_data = match self.completed_plys.last() {
            Some(p) => p.clone(),
            None => {
                let mut sd = SearchData::new(0);
                if let Some(rm) = self.root_moves.moves.first() {
                    sd.best_move = rm.clone();

                }
                sd

            }
        };

        // When randomize is on, pick a random move from all root moves.
        if self.options.randomize && !best_search_data.game_over && !self.root_moves.moves.is_empty() {
            let mut rng = thread_rng();
            let chosen = self.root_moves.moves.choose(&mut rng).unwrap();
            best_search_data.best_move = chosen.clone();

        }

        // Update final time
        best_search_data.elapsed_time = self.search_stats.start_time.elapsed().as_secs_f64();

        ugi::best_move_output(best_search_data);

    }

    /// One cooperative root iteration — main forward, helpers backward, stop when pointers cross.
    pub fn cooperative_root_iteration(
        &mut self,
        moves: &[Move],
        dispatch: &RootDispatch,
        is_main: bool,
    ) -> Vec<(Move, f64)> {
        let mut local_results: Vec<(Move, f64)> = Vec::new();

        if moves.is_empty() {
            return local_results;

        }

        let mut board = self.options.board.clone();
        let player = Player::One;
        let start_ply = dispatch.start_ply;
        let beta = dispatch.beta;

        // Push root hash for cycle detection — child cycle checks against this.
        self.path.clear();
        self.path.push(board.hash());

        loop {
            if self.stop_signal.load(AtomicOrdering::Relaxed) {
                self.stop = true;
                break;

            }

            // Claim a move — main forward, helpers backward, both check for crossing.
            let i: usize = if is_main {
                let val = dispatch.next_top.fetch_add(1, AtomicOrdering::AcqRel);
                if val >= dispatch.next_back.load(AtomicOrdering::Acquire) {
                    break;

                }
                val

            } else {
                let prev = dispatch.next_back.fetch_sub(1, AtomicOrdering::AcqRel);
                if prev == 0 {
                    break;

                }
                let val = prev - 1;
                if val < dispatch.next_top.load(AtomicOrdering::Acquire) {
                    break;

                }
                val

            };

            if i >= moves.len() {
                break;

            }

            let mv = moves[i];
            board.make_move(&mv);

            // Skip moves that cycle back into the root position.
            if self.path.contains(&board.hash()) {
                board.unmake_move(&mv);
                continue;

            }

            // Main's PV move gets full window; everything else null-window with shared alpha.
            let score: f64 = if is_main && i == 0 {
                -self.search(&mut board, -beta, -dispatch.initial_alpha, player.other(), start_ply - 1, start_ply)

            } else {
                let alpha = dispatch.current_alpha();
                let mut s = -self.search(&mut board, -alpha - 1.0, -alpha, player.other(), start_ply - 1, start_ply);
                if s > alpha && s < beta {
                    s = -self.search(&mut board, -beta, -alpha, player.other(), start_ply - 1, start_ply);

                }
                s

            };

            board.unmake_move(&mv);

            if self.stop {
                break;

            }

            dispatch.try_update_alpha(score);
            local_results.push((mv, score));

        }

        self.path.pop();
        local_results

    }

    /// Main search function.
    fn search(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: Player, ply: i8, start_ply: i8) -> f64 {
        let is_root = ply == start_ply;
        let is_leaf = ply == 0;
        let board_hash = board.hash();

        // Check if the search should stop.
        if self.stop {
            return 0.0;
    
        } else if self.search_stats.nodes % 1000 == 0 {
            self.check_stop();

        }

        // Generate the Raw move list for this node.
        let data: GenResult = unsafe { self.mg.gen::<GenMoves, QuitOnThreat>(board, player) };
        let (has_threat, mut move_list) = (data.threat, data.move_list);
    
        // If there is the threat for the current player return a depth-adjusted win score.
        // Shallower wins score higher, so the engine always picks the quickest forced win.
        if has_threat {
            return WIN_SCORE - (start_ply - ply) as f64;

        }

        self.search_stats.nodes += 1;
        self.shared_nodes.fetch_add(1, AtomicOrdering::Relaxed);

        // Base case, if the node is a leaf node, return the evaluation.
        if is_leaf {
            if self.options.nn && network_loaded() {
                return get_evalulation_nn(board, player);

            } else {
                // Classical evaluation fallback
                return get_evalulation(board, &mut self.mg) * player.eval_multiplier();

            }

        }

        // Handle Transposition Table
        if let Some(entry) = unsafe { tt().probe(board_hash) } {
            if entry.depth >= ply {
                match entry.bound {
                    NodeBound::ExactValue => {
                        return entry.score

                    },
                    NodeBound::LowerBound => {
                        alpha = alpha.max(entry.score);

                    },
                    NodeBound::UpperBound => {
                        beta = beta.min(entry.score);

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
        
            // No raw moves at all means the current player has no legal moves: draw.
            if moves.is_empty() {
                return DRAW_SCORE;

            }

            self.order_moves(moves, board, player)

        };

        // All moves were filtered out (every move hands opponent an immediate win): loss.
        // Returns -(WIN_SCORE - depth) so parent sees WIN_SCORE - depth after negation.
        if current_player_moves.len() == 0 {
            return LOSS_SCORE + (start_ply - ply) as f64;

        }
        
        // Loop through valid moves and search them.
        self.path.push(board_hash);
        let original_alpha = alpha;
        let mut best_move = Move::new_null();
        let mut best_score: f64 = f64::NEG_INFINITY;
        for (i, mv) in current_player_moves.iter().enumerate() {
            board.make_move(mv);

            // Skip moves that cycle back to a position already on the path: cycle.
            if self.path.contains(&board.hash()) {
                board.unmake_move(mv);
                continue;

            }

            // Principal Variation Search
            let score: f64 = if i < 5 {
                -self.search(board, -beta, -alpha, player.other(), ply - 1, start_ply) // Full search

            } else {
                let mut score = -self.search(board, -alpha - 1.0, -alpha, player.other(), ply - 1, start_ply); // Null window search
                if score > alpha && score < beta {
                    score = -self.search(board, -beta, -alpha, player.other(), ply - 1, start_ply);
                }
                score

            };

            board.unmake_move(mv);

            // Update the score of the rootnode.
            if is_root {
                self.root_moves.update_move(*mv, score, start_ply);

            }

            if score > best_score {
                best_score = score;
                best_move = *mv;

            }
            if best_score > alpha {
                // self.history.log_alpha_increase(mv, ply);
                alpha = best_score;

            }

            if alpha >= beta {
                self.history.log_beta_cutoff(mv, ply);
                break;

            }

        }

        if !self.stop {
            let node_bound: NodeBound = if best_score >= beta {
                NodeBound::LowerBound

            } else if best_score <= original_alpha {
                NodeBound::UpperBound

            } else {
                NodeBound::ExactValue

            };

            let new_entry = Entry::new(board_hash, best_score, ply, best_move, node_bound);
            unsafe { tt().insert(new_entry) };

        }

        self.path.pop();
        best_score

    }

    /// Orders a list of moves.
    pub fn order_moves(&mut self, mut moves: Vec<Move>, board: &mut BoardState, player: Player) -> Vec<Move> {
        let mut out: Vec<Move> = Vec::with_capacity(moves.len());

        // Gather Transposition Table move
        let mut tt_move: Option<Move> = None;
        if let Some(entry) = unsafe { tt().probe(board.hash()) } {
            if entry.bestmove != Move::new_null() {
                tt_move = Some(entry.bestmove);

            }

        }

        // Store TT move first
        if let Some(tt) = tt_move {
            if let Some(idx) = moves.iter().position(|&m| m == tt) {
                out.push(moves.swap_remove(idx));

            }

        }

        let mut moves_to_sort: Vec<(Move, f64, f64)> = moves.into_iter().filter_map(|mv| {
            let mut sort_val: f64 = 0.0;
            board.make_move(&mv);

            let data = unsafe { self.mg.gen::<GenMoveCount, QuitOnThreat>(board, player.other()) };
            if data.threat {
                board.unmake_move(&mv);
                return None;

            }
            let opp_movecount = data.move_count;

            // If the move has a threat then increase the sort value.
            let data = unsafe { self.mg.gen::<GenNone, QuitOnThreat>(board, player) };
            if data.threat {
                sort_val += 1000.0;

            }

            sort_val -= opp_movecount as f64;

            board.unmake_move(&mv);

            return Some((mv, sort_val, self.history.fetch(&mv)));

        }).collect();

        moves_to_sort.sort_by(|a, b| {
            let a_score = a.1;
            let b_score = b.1;

            let a_hist = a.2;
            let b_hist = b.2;

            // If more than 3% difference use that to sort
            if (a_score - b_score).abs() > (0.03 * a_score.abs().max(b_score.abs())) {
                if a_score > b_score {
                    Ordering::Less

                } else if a_score < b_score {
                    Ordering::Greater

                } else {
                    Ordering::Equal
                    
                }

            } else { // Otherwise use history to sort
                if a_hist > b_hist {
                    Ordering::Less

                } else if a_hist < b_hist {
                    Ordering::Greater

                } else {
                    Ordering::Equal

                }

            }

        });

        out.extend(moves_to_sort.into_iter().map(|(mv, _, _)| mv));

        return out;

    }

    /// Setups up the RootMoveList from a [BoardState].
    /// 
    /// Generates all moves, sorts them, and calculates the number of threats that they each have.
    /// 
    pub fn setup_rootmoves(&mut self, board: &mut BoardState) {
        let moves = unsafe { self.mg.gen::<GenMoves, NoQuit>(board, Player::One).move_list.moves(board) };
        let ordered: Vec<Move> = self.order_moves(moves, board, Player::One);
        
        let root_moves: Vec<RootMove> = ordered.iter().map( |mv| {
            let mut new_board = board.make_move_clone(mv);
            let threats: usize = unsafe { self.mg.gen::<GenThreatCount, NoQuit>(&mut new_board, Player::One).threat_count };

            RootMove::new(*mv, 0.0, 0, threats)

        }).collect();

        let mut rootmove_list = RootMoveList::new();
        rootmove_list.moves = root_moves;

        self.root_moves = rootmove_list;

    }

}

/// Cooperative-root iterative deepening across N searcher threads.
///
/// `searchers[0]` is Main; the rest are Helpers. The IDS loop runs once
/// over Main's state (board, completed_plys, root_moves, output). Per
/// iteration, every searcher cooperates at the root via `thread::scope`:
/// they atomically claim root-move indices from a shared `RootDispatch`
/// and CAS-update a shared alpha. Whoever grabs index 0 runs full-window
/// PVS for the PV move; subsequent indices use null-window with the
/// shared alpha. Below the root each thread recurses independently with
/// its own `MoveGen`, `HistoryTable`, and `path`, sharing only the TT.
///
/// Works for `searchers.len() >= 1`. At N=1 it's equivalent to the
/// previous serial path but goes through a single scoped-thread spawn.
pub fn parallel_iterative_deepening_search(searchers: &mut [Searcher]) {
    assert!(!searchers.is_empty(), "parallel search requires at least one Searcher");

    // ---- Setup phase: validate, no-move/win/loss checks, root move setup. ----

    // All threads start fresh.
    for s in searchers.iter_mut() {
        s.stop = false;
        s.completed_plys.clear();
        s.search_stats = SearchStats::new();

    }
    let start_time = Instant::now();
    for s in searchers.iter_mut() {
        s.search_stats.start_time = start_time;

    }

    let initial_board = searchers[0].options.board.clone();
    let maxply = searchers[0].options.maxply;

    if initial_board.piece_bb.pop_count() != 12 {
        panic!("SEARCH ERROR: INVALID BOARD. Board must have exactly 12 pieces to start a search.");

    }

    // Generate the raw legal move list for the initial position.
    let mut working_board = initial_board.clone();
    let mut move_list: GenResult = unsafe { searchers[0].mg.gen::<GenMoves, NoQuit>(&mut working_board, Player::One) };
    let initial_moves = move_list.move_list.moves(&working_board);

    if initial_moves.is_empty() {
        let mut ply_data = SearchData::new(1);
        ply_data.elapsed_time = start_time.elapsed().as_secs_f64();
        ply_data.best_move.score = DRAW_SCORE;
        ply_data.game_over = true;
        ply_data.is_draw = true;
        searchers[0].completed_plys.push(ply_data);
        searchers[0].final_output();
        return;

    }

    for mv in initial_moves.iter() {
        if mv.is_win() {
            let mut ply_data = SearchData::new(1);
            ply_data.best_move = RootMove::new(*mv, WIN_SCORE, 1, 0);
            searchers[0].completed_plys.push(ply_data.clone());
            searchers[0].final_output();
            return;

        }

    }

    // Setup root moves on Main; this seeds threats and order_moves ordering.
    searchers[0].root_moves = RootMoveList::new();
    {
        let mut board_for_setup = initial_board.clone();
        searchers[0].setup_rootmoves(&mut board_for_setup);

    }

    if searchers[0].root_moves.moves.is_empty() {
        let mut ply_data = SearchData::new(1);
        ply_data.best_move = RootMove::new(initial_moves[0].clone(), LOSS_SCORE, 1, 0);
        ply_data.game_over = true;
        ply_data.winner = 2;
        searchers[0].completed_plys.push(ply_data.clone());
        searchers[0].final_output();
        return;

    }

    for s in searchers.iter_mut() {
        s.path.clear();

    }

    let mut current_ply: i8 = 1;

    // ---- IDS loop ----

    'iterative_deepening: loop {
        if searchers[0].completed_plys.last().map(|p| p.game_over).unwrap_or(false) {
            break 'iterative_deepening;

        }

        for s in searchers.iter_mut() {
            s.history.decay();

        }

        let prev_score = searchers[0].completed_plys.last().map(|p| p.best_move.score);
        let (mut alpha, mut beta) = match prev_score {
            Some(sc) => (sc - 1000.0, sc + 1000.0),
            None => (f64::NEG_INFINITY, f64::INFINITY),
        };

        // Aspiration retry loop. Each retry runs a fresh cooperative iteration.
        'aspiration_windows: loop {
            let moves_snapshot: Vec<Move> = searchers[0].root_moves.clone().into();
            if moves_snapshot.is_empty() {
                break 'aspiration_windows;

            }

            let dispatch = RootDispatch::new(alpha, beta, current_ply, moves_snapshot.len());

            // NEG_INFINITY alpha (first IDS iter) — run main alone, null-window is degenerate.
            let active_count: usize = if alpha.is_finite() { searchers.len() } else { 1 };

            let all_results: Vec<Vec<(Move, f64)>> = thread::scope(|sc| {
                let dispatch_ref = &dispatch;
                let moves_ref = moves_snapshot.as_slice();
                let (active, _) = searchers.split_at_mut(active_count);
                let handles: Vec<_> = active
                    .iter_mut()
                    .enumerate()
                    .map(|(idx, s)| {
                        let is_main = idx == 0;
                        sc.spawn(move || s.cooperative_root_iteration(moves_ref, dispatch_ref, is_main))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            // Fold into Main's root_moves and pick iteration best.
            let mut iteration_best: Option<(Move, f64)> = None;
            for results in all_results.iter() {
                for (mv, score) in results {
                    searchers[0].root_moves.update_move(*mv, *score, current_ply);
                    if iteration_best.map_or(true, |(_, s)| *score > s) {
                        iteration_best = Some((*mv, *score));

                    }

                }

            }
            let (iteration_best_move, best_score) =
                iteration_best.unwrap_or((Move::new_null(), alpha));

            // TT entry for the root position so future iterations / searches
            // can use the best move and bound at this depth.
            if !iteration_best_move.is_null() {
                let bound = if best_score >= beta {
                    NodeBound::LowerBound

                } else if best_score <= alpha {
                    NodeBound::UpperBound

                } else {
                    NodeBound::ExactValue

                };
                let entry = Entry::new(initial_board.hash(), best_score, current_ply, iteration_best_move, bound);
                unsafe { tt().insert(entry) };

            }

            // Refresh main's stop so maxtime / maxnodes can escape aspiration retries.
            searchers[0].check_stop();

            // Propagate stop / decisive scores out of the aspiration retry.
            if searchers[0].stop
                || searchers[0].stop_signal.load(AtomicOrdering::Relaxed)
                || best_score >= WIN_THRESHOLD
                || best_score <= LOSS_THRESHOLD
            {
                break 'aspiration_windows;

            }

            if best_score <= alpha {
                alpha -= 1000.0;

            } else if best_score >= beta {
                beta += 1000.0;

            } else {
                break 'aspiration_windows;

            }

        }

        // Time-based stop check (only the atomic-stop case set self.stop above;
        // also honor maxtime/maxnodes via Main's check).
        searchers[0].check_stop();
        if searchers[0].stop {
            break 'iterative_deepening;

        }

        let mut ply_data: SearchData = SearchData::new(current_ply);
        searchers[0].update_search_data(&mut ply_data);

        // Report cumulative sum without writing back — write-back would double-count next iter.
        let total_nodes: usize = searchers.iter().map(|s| s.search_stats.nodes).sum();
        ply_data.nodes = total_nodes;
        ply_data.nps = (total_nodes as f64 / ply_data.elapsed_time) as usize;

        ugi::info_output(ply_data.clone());
        searchers[0].completed_plys.push(ply_data);

        // Filter losing moves from the root list before the next iteration.
        searchers[0].root_moves.sort();
        searchers[0].root_moves.moves = searchers[0]
            .root_moves
            .moves
            .iter()
            .filter(|mv| mv.score > LOSS_THRESHOLD)
            .cloned()
            .collect();

        current_ply += 2;

        if let Some(maxply) = maxply {
            if current_ply > maxply {
                break 'iterative_deepening;

            }

        }

    }

    searchers[0].final_output();

}

/// History table for move ordering.
#[derive(Clone)]
pub struct HistoryTable {
    h_bounce: [[f64; 38]; 38],
    h_drop: [[[f64; 38]; 38]; 38],

}

impl Default for HistoryTable {
    fn default() -> Self {
        Self {
            h_bounce: [[0.0; 38]; 38],
            h_drop: [[[0.0; 38]; 38]; 38],

        }

    }

}

impl HistoryTable {
    #[inline(always)]
    pub fn clear(&mut self) {
        *self = Self::default();

    }

    #[inline]
    pub fn decay(&mut self) {
        for i in 0..38 {
            for j in 0..38 {
                self.h_bounce[i][j] *= 0.5;

                for k in 0..38 {
                    self.h_drop[i][j][k] *= 0.5;

                }

            }

        }

    }

    /// Fetches the history score for a move.
    #[inline(always)]
    pub fn fetch(&self, mv: &Move) -> f64 {
        let step1 = mv.data[0];
        let step2 = mv.data[1];
        let step3 = mv.data[2];

        if mv.flag != MoveType::Drop {
            let s = step1.1.0 as usize;
            let e = step2.1.0 as usize;
            self.h_bounce[s][e]

        } else {
            let s = step1.1.0 as usize;
            let p = step2.1.0 as usize;
            let d = step3.1.0 as usize;
            self.h_drop[s][p][d]

        }

    }

    /// Update history on a beta cutoff / fail-high.
    #[inline(always)]
    pub fn log_beta_cutoff(&mut self, mv: &Move, depth: i8) {
        let step1 = mv.data[0];
        let step2 = mv.data[1];
        let step3 = mv.data[2];

        // Standard depth weighting: deeper cutoffs matter more
        let bonus = depth as f64 * depth as f64;

        if mv.flag != MoveType::Drop {
            let s = step1.1.0 as usize;
            let e = step2.1.0 as usize;

            self.h_bounce[s][e] = self.h_bounce[s][e].add(bonus);

        } else {
            let s = step1.1.0 as usize;
            let p = step2.1.0 as usize;
            let d = step3.1.0 as usize;

            self.h_drop[s][p][d] = self.h_drop[s][p][d].add(bonus);

        }

    }
    
    /// Update history on a alpha increase
    #[inline(always)]
    pub fn log_alpha_increase(&mut self, mv: &Move, depth: i8) {
        let step1 = mv.data[0];
        let step2 = mv.data[1];
        let step3 = mv.data[2];

        // Standard depth weighting: deeper cutoffs matter more
        let bonus = depth as f64 * 0.5;

        if mv.flag != MoveType::Drop {
            let s = step1.1.0 as usize;
            let e = step2.1.0 as usize;

            self.h_bounce[s][e] = self.h_bounce[s][e].add(bonus);

        } else {
            let s = step1.1.0 as usize;
            let p = step2.1.0 as usize;
            let d = step3.1.0 as usize;

            self.h_drop[s][p][d] = self.h_drop[s][p][d].add(bonus);

        }

    }

}

/// Gets the principle variation from the transposition table.
pub fn get_pv(board: &mut BoardState, max_ply: i8) -> Vec<RootMove> {
    let mut pv: Vec<RootMove> = vec![];

    let mut current_board = board.clone();
    for _ in 0..max_ply {
        if let Some(entry) = unsafe { tt().probe(current_board.hash()) } {
            let current_move = entry.bestmove;
            current_board = current_board.make_move_clone(&current_move);

            pv.push(RootMove::new(current_move, entry.score, 0, 0));

        } else {
            break;

        }

    }

    pv

}


/// Structure that holds all of the results from a specific search ply.
#[derive(Debug, Clone)]
pub struct SearchData {
    pub best_move: RootMove,
    pub pv: Vec<RootMove>,
    pub ply: i8,

    pub nodes: usize,
    pub nps: usize,
    pub elapsed_time: f64,

    pub game_over: bool,
    pub winner: usize,
    pub is_draw: bool,

}

impl SearchData {
    pub fn new(ply: i8) -> SearchData {
        SearchData {
            best_move: RootMove::new_null(),
            pv: vec![],
            ply,
            nodes: 0,
            nps: 0,
            elapsed_time: 0.0,
            game_over: false,
            winner: 0,
            is_draw: false,

        }

    }

}

/// Structure that holds real time stats during a search. This data is stored into a [SearchData] object whenever a ply is completed.
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub nodes: usize,
    pub nps: usize,

    pub start_time: Instant,

}

impl SearchStats {
    pub fn new() -> SearchStats {
        SearchStats {
            nodes: 0,
            nps: 0,

            start_time: Instant::now(),

        }

    }

}


/// Holds all of the settings for a spsific search.
#[derive(Clone)]
pub struct SearchOptions {
    pub board: BoardState,
    pub maxply: Option<i8>,
    pub maxtime: Option<f64>,
    pub maxnodes: Option<usize>,
    pub randomize: bool,
    pub nn: bool, // Use NN if set, fallback to old evaluation if not loaded
    /// Total Lazy SMP thread count (1 = serial). Helper count is `threads - 1`.
    pub threads: usize,

}

impl SearchOptions {
    pub fn new() -> SearchOptions {
        SearchOptions {
            board: BoardState::from(STARTING_BOARD),
            maxply: Option::None,
            maxtime: Option::None,
            maxnodes: Option::None,
            randomize: false,
            nn: true,
            threads: 4,

        }

    }

}

impl Default for SearchOptions {
    fn default() -> Self {
        Self::new()

    }

}
