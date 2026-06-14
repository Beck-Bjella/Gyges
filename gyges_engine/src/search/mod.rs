//! Main strutures, and concepts related to searching.
//! 

pub mod evaluation;
pub mod eval_display;
pub mod network;

use core::f64;
use std::cmp::Ordering;
use std::ops::Add;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
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

// YBWC tuning: minimum remaining ply to open a split, and a registry cap.
const MIN_SPLIT_PLY: i8 = 2;
const MAX_SPLITS: usize = 32;

// The first N moves of a node get a full-window PVS search
const PVS_FULL_WINDOW: usize = 5;

/// Structure that owns all workers and runs a parallel search across them.
pub struct Searcher {
    pub master: Worker,
    pub helpers: Vec<Worker>,
    pub stop_signal: Arc<AtomicBool>,

}

impl Searcher {
    /// Creates a new searcher configured by a [SearchOptions] struct.
    pub fn new(options: SearchOptions, stop_signal: Arc<AtomicBool>) -> Searcher {
        let shared_nodes = Arc::new(AtomicUsize::new(0));
        let master = Worker::new(stop_signal.clone(), shared_nodes.clone(), options.clone());
        let helpers: Vec<Worker> = (1..options.threads.max(1))
            .map(|_| Worker::new(stop_signal.clone(), shared_nodes.clone(), options.clone()))
            .collect();

        Searcher { master, helpers, stop_signal }

    }

    /// Entry point for the search.
    pub fn go(&mut self) {
        // Initialize master and workers
        let start_time = Instant::now();
        for w in std::iter::once(&mut self.master).chain(self.helpers.iter_mut()) {
            w.stop = false;
            w.completed_plys.clear();
            w.start_time = start_time;
            w.path.clear();

        }

        let master = &mut self.master;
        let mut board = master.options.board.clone();

        if board.piece_bb.pop_count() != 12 {
            panic!("SEARCH ERROR: INVALID BOARD. Board must have exactly 12 pieces to start a search.");

        }

        // Generate the raw legal move list for the initial position.
        let mut gen: GenResult = unsafe { master.mg.gen::<GenMoves, NoQuit>(&mut board, Player::One) };
        let initial_moves = gen.move_list.moves(&board);

        // No legal moves: draw.
        if initial_moves.is_empty() {
            let mut ply_data = SearchData::new(1);
            ply_data.elapsed_time = master.start_time.elapsed().as_secs_f64();
            ply_data.best_move.score = DRAW_SCORE;
            ply_data.game_over = true;
            ply_data.is_draw = true;
            master.completed_plys.push(ply_data);
            master.final_output();
            return;

        }

        // Immediate win.
        for mv in initial_moves.iter() {
            if mv.is_win() {
                let mut ply_data = SearchData::new(1);
                ply_data.best_move = RootMove::new(*mv, WIN_SCORE, 1, 0);
                master.completed_plys.push(ply_data.clone());
                master.final_output();
                return;

            }

        }

        // Setup root moves; losing moves get filtered out, so none left means loss.
        master.root_moves = RootMoveList::new();
        let mut board_for_setup = master.options.board.clone();
        master.setup_rootmoves(&mut board_for_setup);

        if master.root_moves.moves.is_empty() {
            let mut ply_data = SearchData::new(1);
            ply_data.best_move = RootMove::new(initial_moves[0].clone(), LOSS_SCORE, 1, 0);
            ply_data.game_over = true;
            ply_data.winner = 2;
            master.completed_plys.push(ply_data.clone());
            master.final_output();
            return;

        }

        self.start_threads();

        self.master.final_output();

    }

    /// Attaches the shared YBWC context, spawns the helpers, and runs the
    /// master's iterative deepening until completion or stop.
    fn start_threads(&mut self) {
        // Create shared YBWC context
        let ctx = Arc::new(YbwcPool::new(self.helpers.len()));
        for w in std::iter::once(&mut self.master).chain(self.helpers.iter_mut()) {
            w.pool = Some(ctx.clone());

        }

        let maxply = self.master.options.maxply;
        let signal = self.stop_signal.clone();

        thread::scope(|sc| {
            // Spawn helpers into threads
            for h in self.helpers.iter_mut() {
                sc.spawn(move || h.helper_loop());

            }

            self.master.iterative_deepening_search(maxply);

            // Set stop signal forcing helpers exit
            signal.store(true, AtomicOrdering::Relaxed);

        });

        for w in std::iter::once(&mut self.master).chain(self.helpers.iter_mut()) {
            w.pool = None;

        }

    }

}

/// Structure that holds all needed information for one search thread, and contains all of the main searching functions.
pub struct Worker {
    pub options: SearchOptions,
    pub stop_signal: Arc<AtomicBool>,
    pub stop: bool,
    
    pub shared_nodes: Arc<AtomicUsize>,

    pub completed_plys: Vec<SearchData>,
    pub start_time: Instant,
    pub root_moves: RootMoveList,

    pub mg: MoveGen,

    pub history: HistoryTable,

    pub path: Vec<u64>,

    pub pool: Option<Arc<YbwcPool>>,

    /// Cutoff flag of the split this worker is currently draining, if any.
    /// Checked per-node in `search` so an in-flight subtree bails the moment
    /// a sibling fails high, instead of finishing wasted work.
    pub active_cutoff: Option<Arc<AtomicBool>>,

}

impl Worker {
    /// Creates a new worker.
    pub fn new(stop_signal: Arc<AtomicBool>, shared_nodes: Arc<AtomicUsize>, options: SearchOptions) -> Worker {
        Worker {
            options,
            stop_signal,
            stop: false,
            shared_nodes,

            completed_plys: vec![],
            start_time: Instant::now(),
            root_moves: RootMoveList::new(),

            mg: MoveGen::default(),

            history: HistoryTable::default(),

            path: Vec::new(),

            pool: None,

            active_cutoff: None,

        }

    }

    /// Starts a Iterative deepening Search
    /// 
    /// Master Thread Only
    /// 
    fn iterative_deepening_search(&mut self, maxply: Option<i8>) {
        let mut current_ply: i8 = 1;

        'iterative_deepening: loop {
            if self.completed_plys.last().map(|p| p.game_over).unwrap_or(false) {
                break 'iterative_deepening;

            }

            self.history.decay();
            if let Some(pool) = self.pool.as_ref() {
                pool.generation.fetch_add(1, AtomicOrdering::Relaxed);

            }

            let prev_score = self.completed_plys.last().map(|p| p.best_move.score);
            let (alpha, beta) = match prev_score {
                Some(sc) => (sc - 1000.0, sc + 1000.0),
                None => (f64::NEG_INFINITY, f64::INFINITY),
            };

            self.root_search(alpha, beta, current_ply);

            self.check_stop();
            if self.stop {
                break 'iterative_deepening;

            }

            let mut ply_data: SearchData = SearchData::new(current_ply);
            self.update_search_data(&mut ply_data);

            ugi::info_output(ply_data.clone());
            self.completed_plys.push(ply_data);

            // Filter losing moves from the root list before the next iteration.
            self.root_moves.sort();
            self.root_moves.moves = self.root_moves.moves.iter()
                .filter(|mv| mv.score > LOSS_THRESHOLD)
                .cloned()
                .collect();

            current_ply += 2;

            // Max ply check
            if let Some(maxply) = maxply {
                if current_ply > maxply {
                    break 'iterative_deepening;

                }

            }

        }

    }

    /// Starts a singular search to a specific ply. Handles the aspiration window and caches final result to the TT.
    /// 
    /// Master Thread Only
    /// 
    fn root_search(&mut self, mut alpha: f64, mut beta: f64, start_ply: i8) {
        let player = Player::One;
        let root_hash = self.options.board.hash();

        'aspiration_windows: loop {
            let moves: Vec<Move> = self.root_moves.clone().into();
            if moves.is_empty() {
                break 'aspiration_windows;

            }

            let mut board = self.options.board.clone();
            
            self.path.clear();
            self.path.push(board.hash());

            let mv0 = moves[0];
            let mut best_score = f64::NEG_INFINITY;
            let mut best_move = mv0;

            // Move 0 gets the full-window serial search to seed alpha.
            board.make_move(&mv0);
            if !self.path.contains(&board.hash()) {
                let score = -self.search(&mut board, -beta, -alpha, player.other(), start_ply - 1, start_ply);
                board.unmake_move(&mv0);

                if self.stop {
                    self.path.pop();
                    return;

                }

                self.root_moves.update_move(mv0, score, start_ply);
                best_score = score;

            } else {
                board.unmake_move(&mv0);

            }

            // Search the siblings through a split, unless move 0 already failed
            // high / found a win / is the only move (siblings pointless then).
            if !(moves.len() == 1 || best_score >= beta || best_score >= WIN_THRESHOLD) {
                let pool = self.pool.as_ref().unwrap().clone();
                let sp = Arc::new(SplitPoint::new(
                    board.clone(),
                    self.path.clone(),
                    moves[1..].to_vec(),
                    alpha.max(best_score),
                    beta,
                    start_ply,
                    start_ply,
                    player,
                ));

                pool.add(sp.clone());
                let saved_cutoff = self.active_cutoff.replace(sp.cutoff.clone());
                self.search_split(&sp, &mut board);
                self.close_split(&pool, &sp);
                self.active_cutoff = saved_cutoff;

                let shared = sp.shared.lock().unwrap();
                for (mv, score) in shared.results.iter() {
                    self.root_moves.update_move(*mv, *score, start_ply);
                    if *score > best_score {
                        best_score = *score;
                        best_move = *mv;

                    }

                }
                drop(shared);

            }

            self.path.pop();

            // Root TT entry so future iterations can use the best move and bound.
            if !best_move.is_null() && best_score.is_finite() {
                let bound = if best_score >= beta {
                    NodeBound::LowerBound

                } else if best_score <= alpha {
                    NodeBound::UpperBound

                } else {
                    NodeBound::ExactValue

                };
                let entry = Entry::new(root_hash, best_score, start_ply, best_move, bound);
                unsafe { tt().insert(entry) };

            }

            // Stop / decisive result: no point widening.
            self.check_stop();
            if self.stop || best_score == f64::NEG_INFINITY || best_score >= WIN_THRESHOLD || best_score <= LOSS_THRESHOLD {
                break 'aspiration_windows;

            }

            // Widen on a fail, otherwise the window held — done.
            if best_score <= alpha {
                alpha -= 1000.0;

            } else if best_score >= beta {
                beta += 1000.0;

            } else {
                break 'aspiration_windows;

            }

        }

    }

    /// Main search function. Root moves are handled by `root_iteration`,
    /// so this is only ever called below the root.
    fn search(&mut self, board: &mut BoardState, mut alpha: f64, mut beta: f64, player: Player, ply: i8, start_ply: i8) -> f64 {
        let is_leaf = ply == 0;
        let board_hash = board.hash();

        // Bail out instantly once this worker has stopped.
        if self.stop {
            return 0.0;

        }

        // Generate the Raw move list for this node.
        let data: GenResult = unsafe { self.mg.gen::<GenMoves, QuitOnThreat>(board, player) };
        let (has_threat, mut move_list) = (data.threat, data.move_list);
    
        // If there is the threat for the current player return a depth-adjusted win score.
        // Shallower wins score higher, so the engine always picks the quickest forced win.
        if has_threat {
            return WIN_SCORE - (start_ply - ply) as f64;

        }

        // Count this node, and pace the stop check off the shared total.
        if self.shared_nodes.fetch_add(1, AtomicOrdering::Relaxed) % 1000 == 0 {
            self.check_stop();

            // Bail the in-flight subtree if our split has been cut off by a
            // sibling. Sets `self.stop` so the existing discard-points (TT
            // insert, sp.store) drop this node's garbage score; the worker
            // un-stops when it picks up its next split.
            if let Some(cutoff) = &self.active_cutoff {
                if cutoff.load(AtomicOrdering::Acquire) {
                    self.stop = true;

                }

            }

        }

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

        // Generate and order the moves for this node.
        let moves = move_list.moves(board);

        // No raw moves at all means the current player has no legal moves: draw.
        if moves.is_empty() {
            return DRAW_SCORE;

        }

        let current_player_moves: Vec<Move> = self.order_moves(moves, board, player);

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
            let score: f64 = if i < PVS_FULL_WINDOW {
                -self.search(board, -beta, -alpha, player.other(), ply - 1, start_ply) // Full search

            } else {
                let mut score = -self.search(board, -alpha - 1.0, -alpha, player.other(), ply - 1, start_ply); // Null window search
                if score > alpha && score < beta {
                    score = -self.search(board, -beta, -alpha, player.other(), ply - 1, start_ply);
                }
                score

            };

            board.unmake_move(mv);

            if score > best_score {
                best_score = score;
                best_move = *mv;

            }
            if best_score > alpha {
                alpha = best_score;

            }

            if alpha >= beta {
                self.history.log_beta_cutoff(mv, ply);
                break;

            }

            // YBWC: with an idle helper and â‰¥2 children left, split the rest of this node.
            if ply >= MIN_SPLIT_PLY && i + 2 < current_player_moves.len() && !self.stop {
                if let Some(pool) = self.pool.clone() {
                    if pool.can_split() {
                        let (split_score, split_move) =
                            self.open_split(board, &current_player_moves[i + 1..], alpha, beta, ply, start_ply, player);

                        if split_score > best_score {
                            best_score = split_score;
                            best_move = split_move;

                        }
                        if best_score > alpha {
                            alpha = best_score;

                        }
                        if alpha >= beta {
                            self.history.log_beta_cutoff(&best_move, ply);

                        }
                        break;

                    }

                }

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

    /// Allows for multiple threads to search a split point by pulling children until it drains, cuts off, or is stoped.
    /// The `board` must be at the split node; `self.path` must end with its hash.
    fn search_split(&mut self, sp: &SplitPoint, board: &mut BoardState) {
        loop {
            // Check stop
            if self.stop || sp.cutoff.load(AtomicOrdering::Acquire) || self.stop_signal.load(AtomicOrdering::Relaxed) {
                break;

            }

            // Grab next open move
            let i = sp.next.fetch_add(1, AtomicOrdering::AcqRel);
            if i >= sp.moves.len() {
                break;

            }

            let mv = sp.moves[i];
            board.make_move(&mv);

            // Skip moves that cycle back to a position already on the path.
            if self.path.contains(&board.hash()) {
                board.unmake_move(&mv);
                continue;

            }

            // Principal Variation Search
            let alpha = sp.current_alpha();
            let score: f64 = if i < PVS_FULL_WINDOW - 1 {
                -self.search(board, -sp.beta, -alpha, sp.player.other(), sp.ply - 1, sp.start_ply)

            } else {
                let mut score = -self.search(board, -alpha - 1.0, -alpha, sp.player.other(), sp.ply - 1, sp.start_ply);
                if score > alpha && score < sp.beta && !self.stop {
                    score = -self.search(board, -sp.beta, -alpha, sp.player.other(), sp.ply - 1, sp.start_ply);

                }
                score

            };

            board.unmake_move(&mv);

            if self.stop {
                break;

            }

            sp.store(mv, score);

            if score >= sp.beta {
                sp.cutoff.store(true, AtomicOrdering::Release);
                break;

            }

        }

    }

    /// Open a split for the remaining children of the current node and run it to completion.
    /// Returns the best (score, move) found across all workers.
    fn open_split(&mut self, board: &mut BoardState, rest: &[Move], alpha: f64, beta: f64, ply: i8, start_ply: i8, player: Player) -> (f64, Move) {
        let pool = self.pool.as_ref().unwrap().clone();
        let sp = Arc::new(SplitPoint::new(board.clone(), self.path.clone(), rest.to_vec(), alpha, beta, ply, start_ply, player));

        pool.add(sp.clone());
        let saved_cutoff = self.active_cutoff.replace(sp.cutoff.clone());
        self.search_split(&sp, board);
        self.close_split(&pool, &sp);
        self.active_cutoff = saved_cutoff;

        let shared = sp.shared.lock().unwrap();
        (shared.best_score, shared.best_move)

    }

    /// Deregister `sp`, help other split points while its stragglers finish.
    fn close_split(&mut self, pool: &YbwcPool, sp: &Arc<SplitPoint>) {
        pool.remove(sp);

        while sp.active.load(AtomicOrdering::Acquire) > 0 {
            if let Some(other) = pool.try_claim() { // Help another split with work remaining
                let saved_path = std::mem::take(&mut self.path);
                self.path.extend_from_slice(&other.path);
                let saved_cutoff = self.active_cutoff.replace(other.cutoff.clone());
                // Helping a different split — our own split's cutoff/stop must
                // not abort it. Clear, then restore our state afterward.
                let saved_stop = self.stop;
                self.stop = false;

                let mut board = other.board.clone();
                self.search_split(&other, &mut board);

                other.active.fetch_sub(1, AtomicOrdering::AcqRel);

                self.stop = saved_stop;
                self.path = saved_path;
                self.active_cutoff = saved_cutoff;

            } else { // No splits to help: wait
                thread::yield_now();

            }

        }

    }

    /// Helper thread loop: claim split points until the stop signal fires.
    pub fn helper_loop(&mut self) {
        let pool = self.pool.as_ref().unwrap().clone();
        let mut last_generation = 0usize;
        let mut spins = 0u32;
        loop {
            // Stop check
            if self.stop_signal.load(AtomicOrdering::Relaxed) {
                break;

            }

            // Decay history in step with the master's IDS iterations.
            let generation = pool.generation.load(AtomicOrdering::Relaxed);
            if generation != last_generation {
                last_generation = generation;
                self.history.decay();

            }

            match pool.try_claim() {
                Some(sp) => {
                    spins = 0;
                    pool.idle.fetch_sub(1, AtomicOrdering::AcqRel);

                    self.stop = false;
                    self.path.clear();
                    self.path.extend_from_slice(&sp.path);
                    self.active_cutoff = Some(sp.cutoff.clone());
                    let mut board = sp.board.clone();

                    self.search_split(&sp, &mut board);

                    self.active_cutoff = None;
                    sp.active.fetch_sub(1, AtomicOrdering::AcqRel);
                    pool.idle.fetch_add(1, AtomicOrdering::AcqRel);

                }
                None => {
                    // Spin briefly for fast wakeup, then back off to spare the cores.
                    spins += 1;
                    if spins < 100 {
                        thread::yield_now();

                    } else {
                        thread::sleep(std::time::Duration::from_micros(100));

                    }

                }

            }

        }

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

    /// Setups up the RootMoveList from a [BoardState]. (master only)
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

    /// Checks to see if the engine should stop the search.
    pub fn check_stop(&mut self) {
        // Stop signal from UGI
        if self.stop_signal.load(AtomicOrdering::Relaxed) {
            self.stop = true;

        }

        // Check if the max time has been reached.
        if let Some(maxtime) = self.options.maxtime {
            if self.start_time.elapsed().as_secs_f64() >= maxtime {
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

    /// Update search data based on the current search stats and results. (master only)
    pub fn update_search_data(&mut self, ply_data: &mut SearchData) {
        // Gather current search stats — node total is shared across all workers.
        ply_data.elapsed_time = self.start_time.elapsed().as_secs_f64();
        ply_data.nodes = self.shared_nodes.load(AtomicOrdering::Relaxed);
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

    /// Displays the final output of the search. (master only)
    pub fn final_output(&self) {
        // Fall back to the heuristically-best root move when no ply finished —
        // a `stop` (UGI command or quit) can race search startup before ply 1 completes.
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
        best_search_data.elapsed_time = self.start_time.elapsed().as_secs_f64();

        ugi::best_move_output(best_search_data);

    }

}

/// A node whose remaining children are searched by multiple threads (YBWC split).
/// Created only after the first child was searched serially, so `alpha` is seeded.
pub struct SplitPoint {
    board: BoardState,
    path: Vec<u64>,
    moves: Vec<Move>,
    next: AtomicUsize,
    beta: f64,
    ply: i8,
    start_ply: i8,
    player: Player,
    active: AtomicUsize,
    cutoff: Arc<AtomicBool>,

    // Live shared state for split
    shared: Mutex<SplitShared>,

}

/// The split's live results.
/// Contains the shared alpha bound, the best child, and the per-child scores found so far.
struct SplitShared {
    alpha: f64,
    best_score: f64,
    best_move: Move,
    results: Vec<(Move, f64)>,

}

impl SplitPoint {
    /// Creates a new split point
    fn new(board: BoardState, path: Vec<u64>, moves: Vec<Move>, alpha: f64, beta: f64, ply: i8, start_ply: i8, player: Player) -> Self {
        Self {
            board,
            path,
            moves,
            next: AtomicUsize::new(0),
            beta,
            ply,
            start_ply,
            player,
            active: AtomicUsize::new(0),
            cutoff: Arc::new(AtomicBool::new(false)),
            shared: Mutex::new(SplitShared { 
                alpha, 
                best_score: f64::NEG_INFINITY, 
                best_move: Move::new_null(), 
                results: Vec::new() 
            }),

        }

    }

    /// Fetches the current best alpha for the split
    fn current_alpha(&self) -> f64 {
        self.shared.lock().unwrap().alpha

    }

    /// Stores a finished child's result to the shared split state
    fn store(&self, mv: Move, score: f64) {
        let mut shared = self.shared.lock().unwrap();
        if score > shared.alpha {
            shared.alpha = score;

        }
        if score > shared.best_score {
            shared.best_score = score;
            shared.best_move = mv;

        }
        shared.results.push((mv, score));

    }

}

/// Shared YBWC state: the active split points plus an idle-helper count.
pub struct YbwcPool {
    pool: Mutex<Vec<Arc<SplitPoint>>>,
    idle: AtomicUsize,
    splits: AtomicUsize,
    generation: AtomicUsize,

}

impl YbwcPool {
    /// Creates a new YBWC pool with the given number of helpers.
    fn new(helper_count: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            idle: AtomicUsize::new(helper_count),
            splits: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),

        }

    }

    /// Checks if there are idle helpers and the pool has room.
    fn can_split(&self) -> bool {
        self.idle.load(AtomicOrdering::Relaxed) > 0 && self.splits.load(AtomicOrdering::Relaxed) < MAX_SPLITS

    }
    
    /// Adds a split point to the pool.
    fn add(&self, sp: Arc<SplitPoint>) {
        self.splits.fetch_add(1, AtomicOrdering::AcqRel);
        self.pool.lock().unwrap().push(sp);

    }

    /// Removes a split point from the pool.
    fn remove(&self, sp: &Arc<SplitPoint>) {
        self.pool.lock().unwrap().retain(|x| !Arc::ptr_eq(x, sp));
        self.splits.fetch_sub(1, AtomicOrdering::AcqRel);

    }

    /// Claim a split point with work remaining. Increments the active count if successfully claimed.
    fn try_claim(&self) -> Option<Arc<SplitPoint>> {
        let pool = self.pool.lock().unwrap();
        for sp in pool.iter() {
            if !sp.cutoff.load(AtomicOrdering::Acquire) && sp.next.load(AtomicOrdering::Acquire) < sp.moves.len() {
                sp.active.fetch_add(1, AtomicOrdering::AcqRel);
                return Some(sp.clone());

            }

        }
        None

    }

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

/// Holds all of the settings for a spsific search.
#[derive(Clone)]
pub struct SearchOptions {
    pub board: BoardState,
    pub maxply: Option<i8>,
    pub maxtime: Option<f64>,
    pub maxnodes: Option<usize>,
    pub randomize: bool,
    pub nn: bool, // Use NN if set, fallback to old evaluation if not loaded
    /// Total search thread count (1 = serial). Helper count is `threads - 1`.
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
            threads: 1,

        }

    }

}

impl Default for SearchOptions {
    fn default() -> Self {
        Self::new()

    }

}
