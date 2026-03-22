//! All logic and functions related to the evaluation of a board.
//!

use std::path;

use gyges::board::bitboard::*;
use gyges::core::masks::RANKS;
use gyges::{AllResults, core::*};
use gyges::moves::movegen::{GenControlMoveCount, MoveGen, NoQuit};
use gyges::{board::*, GenResult, PathResult};

////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// OLD EVALUATION ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// pub const UNIQUE_PIECE_CONTROL_SCORES: [f64; 3] = [500.0, 100.0, 50.0];
// pub const SHARED_PIECE_CONTROL_SCORES: [f64; 3] = [75.0, 50.0, 25.0];

// pub const UNIQUE_SQUARE_CONTROL_SCORE: f64 = 10.0;
// pub const SHARED_SQUARE_CONTROL_SCORE: f64 = 5.0;

// // BEST EVALUATION FUNCTION
// pub fn get_evalulation(board: &mut BoardState, mg: &mut MoveGen) -> f64 {
//     let p1 = unsafe { mg.gen::<GenControlMoveCount, NoQuit>(board, Player::One) };
//     let p2 = unsafe { mg.gen::<GenControlMoveCount, NoQuit>(board, Player::Two) };
//     let control_squares = [p1.controlled_squares, p2.controlled_squares];
//     let control_pieces = [p1.controlled_pieces, p2.controlled_pieces];
//     let move_counts = [p1.move_count, p2.move_count];

//     let mut eval = 0.0;

//     eval += mobility_eval(Player::One, move_counts) - mobility_eval(Player::Two, move_counts);
//     eval += (control_eval(board, Player::One, control_pieces, control_squares) - control_eval(board, Player::Two, control_pieces, control_squares)) * 3.0;

//     eval

// }

// pub fn unique_controlled_pieces_score(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2]) -> f64 {
//     let pieces = control_pieces[player as usize];
//     let opp_pieces = control_pieces[player.other() as usize];

//     let mut unique_controlled_pieces = (pieces & !opp_pieces).0;

//     let mut score = 0.0;
//     while unique_controlled_pieces != 0 {
//         let pos = unique_controlled_pieces.trailing_zeros() as usize;
//         unique_controlled_pieces &= unique_controlled_pieces - 1;

//         let piece = board.data[pos];
//         score += UNIQUE_PIECE_CONTROL_SCORES[piece as usize];

//     }

//     score

// }

// pub fn shared_controlled_pieces_score(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2]) -> f64 {
//     let mut pieces = control_pieces[player as usize].0;

//     let mut score = 0.0;
//     while pieces != 0 {
//         let pos = pieces.trailing_zeros() as usize;
//         pieces &= pieces - 1;

//         let piece = board.data[pos];
//         score += SHARED_PIECE_CONTROL_SCORES[piece as usize];

//     }

//     score

// }

// pub fn unique_controlled_squares_score(player: Player, control_squares: [BitBoard; 2]) -> f64 {
//     let squares = control_squares[player as usize];
//     let opp_squares = control_squares[player.other() as usize];

//     let unique_squares = squares & !opp_squares;

//     unique_squares.pop_count() as f64 * UNIQUE_SQUARE_CONTROL_SCORE

// }

// pub fn shared_controlled_squares_score(player: Player, control_squares: [BitBoard; 2]) -> f64 {
//     let squares = control_squares[player as usize];
//     squares.pop_count() as f64 * SHARED_SQUARE_CONTROL_SCORE

// }

// pub fn mobility_eval(player: Player, move_counts: [usize; 2]) -> f64 {
//     let mut eval = 0.0;

//     eval += move_counts[player as usize] as f64;

//     eval

// }

// pub fn control_eval(board: &mut BoardState, player: Player, control_pieces: [BitBoard; 2], control_squares: [BitBoard; 2]) -> f64 {
//     let mut eval = 0.0;

//     eval +=  unique_controlled_pieces_score(board, player, control_pieces);
//     eval +=  unique_controlled_squares_score(player, control_squares);

//     // eval +=  shared_controlled_pieces_score(board, player, control_pieces);
//     // eval +=  shared_controlled_squares_score(player, control_squares);

//     eval

// }

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// Drop in replacement for the old function
pub fn get_evalulation(board: &mut BoardState, mg: &mut MoveGen) -> f64 {
    let evaluation_ctx = EvaluationContext::new(board, mg);
    evaluation_ctx.get_evaluation().total

}

pub const WEIGHTS: EvalWeights = EvalWeights {
    final_scale: 1.0,

    // Activity Weights
    activity_weight: 0.0,
    activity_type_weights: [1.0, 1.0, 1.0],
    activity_shared_modifier: 0.25,

    // Network Weights
    per_path_weight: 1.0,

    // Piece Control Weights
    piece_control_weight: 1.0,
    unique_piece_control_weights: [500.0, 100.0, 50.0],
    shared_piece_control_weights: [75.0, 50.0, 25.0],

    // Square Control Weights
    square_control_weight: 0.25,
    unique_square_control_weight: 10.0,
    shared_square_control_weight: 5.0,

    // Penalties
    backline_trapped_penalty: [-1500.0, -300.0, -150.0],
    backline_stranded_penalty: [-1500.0, -300.0, -150.0],

};

/// The data related to one players position and all of the data
pub struct EvaluationContext {
    pub p1_gen_data: AllResults,
    pub p2_gen_data: AllResults,

    pub board: BoardState,

    // Network Data
    pub total_paths: [usize; 2],
    pub total_bounces: [usize; 2],
    pub total_continuations: [usize; 2],

    // Piece data
    pub unique_piece_control: [BitBoard; 2],
    pub shared_piece_control: BitBoard,
    pub piece_data: Vec<PieceData>,

    // Square data
    pub unique_square_control: [BitBoard; 2],
    pub shared_square_control: BitBoard, 

}

impl EvaluationContext {
    /// Creates all relevant data for the evaluation
    pub fn new(board: &mut BoardState, mg: &mut MoveGen) -> Self {
        let active_lines = board.get_active_lines();
        let active_line_bbs = [
            RANKS[active_lines[0] as usize],
            RANKS[active_lines[1] as usize],
        ];

        let p1 = unsafe { mg.gen_all(board, Player::One) };
        let p2 = unsafe { mg.gen_all(board, Player::Two) };

        // Collect general total path data
        let total_paths = [
            p1.start_count.iter().sum(),
            p2.start_count.iter().sum(),
            
        ];
        let total_bounces = [
            p1.bounce_count.iter().sum(),
            p2.bounce_count.iter().sum(),

        ];
        let total_continuations = [
            p1.continuation_count.iter().sum(),
            p2.continuation_count.iter().sum(),
        ];

        // Piece Control
        let unique_piece_control = [
            p1.controlled_pieces & !p2.controlled_pieces,
            p2.controlled_pieces & !p1.controlled_pieces,
        ];
        let shared_piece_control = p1.controlled_pieces & p2.controlled_pieces;

        // Square Control
        let unique_square_control = [
            p1.controlled_squares & !p2.controlled_squares,
            p2.controlled_squares & !p1.controlled_squares,
        ];
        let shared_square_control = p1.controlled_squares & p2.controlled_squares;

        // Per Piece Data
        let mut piece_data: Vec<PieceData> = Vec::new();
        for pos in board.piece_bb.clone().get_data() {
            let piece = board.data[pos];
            let sq = SQ(pos as u8);

            // Other Info
            let shared = (shared_piece_control.0 & sq.bit()) != 0;
            let on_active_line = [
                active_line_bbs[0] & sq.bit() != 0,
                active_line_bbs[1] & sq.bit() != 0,
            ];

            // Raw signals
            let path_start_counts = [p1.start_count[pos], p2.start_count[pos]];
            let path_bounce_counts = [
                p1.bounce_count[pos],
                p2.bounce_count[pos],
            ];
            let path_continuation_counts = [
                p1.continuation_count[pos],
                p2.continuation_count[pos],
            ];
            let path_termination_counts = [
                p1.bounce_count[pos] - p1.continuation_count[pos],
                p2.bounce_count[pos] - p2.continuation_count[pos],
            ];
            let path_average_depths = [
                p1.depth[pos] as f64 / p1.bounce_count[pos] as f64,
                p2.depth[pos] as f64 / p2.bounce_count[pos] as f64,
            ];
            let path_min_depths = [
                if p1.min_depth[pos] == usize::MAX { f64::NAN } else { p1.min_depth[pos] as f64 },
                if p2.min_depth[pos] == usize::MAX { f64::NAN } else { p2.min_depth[pos] as f64 },
            ];

            // Computed signals
            let starter_percentage: [f64; 2] = [
                path_start_counts[0] as f64 / total_paths[0] as f64,
                path_start_counts[1] as f64 / total_paths[1] as f64,
            ];
            let flow_percentage = [
                path_bounce_counts[0] as f64 / total_bounces[0] as f64,
                path_bounce_counts[1] as f64 / total_bounces[1] as f64,
            ];
            let continuation_rates = [
                path_continuation_counts[0] as f64 / path_bounce_counts[0] as f64,
                path_continuation_counts[1] as f64 / path_bounce_counts[1] as f64,
            ];

            // Interpretations
            let activity_powers = [
                path_bounce_counts[0] as f64 + 10.0 * path_continuation_counts[0] as f64,
                path_bounce_counts[1] as f64 + 10.0 * path_continuation_counts[1] as f64,
            ];

            let trapped = path_bounce_counts[0] == 0 && path_bounce_counts[1] == 0; // Cant do anything
            let stranded = path_continuation_counts[0] == 0 && path_continuation_counts[1] == 0; // Cant reach other pieces

            let data = PieceData {
                piece,
                sq,

                shared,
                on_active_line,

                path_start_counts,
                path_bounce_counts,
                path_continuation_counts,
                path_termination_counts,
                path_average_depths,
                path_min_depths,

                starter_percentage,
                flow_percentage,
                continuation_rates,

                trapped,
                stranded,
                activity_powers,
                
            };

            piece_data.push(data);

        }

        Self {
            p1_gen_data: p1,
            p2_gen_data: p2,

            board: board.clone(),

            total_paths,
            total_bounces,
            total_continuations,

            unique_piece_control,
            shared_piece_control,
            piece_data,

            unique_square_control,
            shared_square_control,

        }

    }

    pub fn get_evaluation(&self) -> EvaluationScore {
        let p1_activity_score: f64 = self.get_activity_score(Player::One) * WEIGHTS.final_scale;
        let p2_activity_score: f64 = self.get_activity_score(Player::Two) * WEIGHTS.final_scale;

        let p1_piece_control = self.get_piece_control_score(Player::One) * WEIGHTS.final_scale;
        let p2_piece_control = self.get_piece_control_score(Player::Two) * WEIGHTS.final_scale;

        let p1_square_control = self.get_square_control_score(Player::One) * WEIGHTS.final_scale;
        let p2_square_control = self.get_square_control_score(Player::Two) * WEIGHTS.final_scale;

        let p1_path_score: f64 = self.get_total_path_score(Player::One) * WEIGHTS.final_scale;
        let p2_path_score: f64 = self.get_total_path_score(Player::Two) * WEIGHTS.final_scale;

        let p1_backline_penalty = self.backline_penalties(Player::One) * WEIGHTS.final_scale;
        let p2_backline_penalty = self.backline_penalties(Player::Two) * WEIGHTS.final_scale;

        let p1_total: f64 = 0.0 +
            0.0 * p1_activity_score +
            p1_piece_control +
            p1_square_control +
            p1_path_score +
            p1_backline_penalty;

        let p2_total: f64 = 0.0 +
            0.0 * p2_activity_score +
            p2_piece_control +
            p2_square_control +
            p2_path_score +
            p2_backline_penalty;

        let total = p1_total - p2_total;

        EvaluationScore {
            total,
            p1_total,
            p2_total,

            p1_activity_score,
            p2_activity_score,

            p1_piece_control,
            p2_piece_control,

            p1_square_control,
            p2_square_control,

            p1_path_score,
            p2_path_score,

            p1_backline_penalty,
            p2_backline_penalty,

        }

    }

    /// 
    pub fn get_activity_score(&self, player: Player) -> f64 {
        let mut activity_score = 0.0;

        for pd in self.piece_data.iter() {
            let activity_power: f64 = pd.activity_powers[player as usize];
            if !activity_power.is_finite() {
                continue;

            }

            let type_weight = WEIGHTS.activity_type_weights[pd.piece as usize];
            let shared_modifier = if pd.shared {
                WEIGHTS.activity_shared_modifier

            } else {
                1.0

            };

            activity_score += activity_power 
                                * type_weight 
                                * WEIGHTS.activity_weight 
                                * shared_modifier;

        }

        activity_score

    }

    /// Piece control score, weighted by depth-based ownership for shared pieces.
    pub fn get_piece_control_score(&self, player: Player) -> f64 {
        let mut score = 0.0;

        for pd in self.piece_data.iter() {
            let piece_value = if (self.unique_piece_control[player as usize].0 & pd.sq.bit()) != 0 {
                WEIGHTS.unique_piece_control_weights[pd.piece as usize]

            } else if (self.shared_piece_control.0 & pd.sq.bit()) != 0 {
                let depth_p1 = pd.path_min_depths[0];
                let depth_p2 = pd.path_min_depths[1];

                let access_p1 = if !depth_p1.is_nan() { 1.0 / depth_p1 } else { 0.0 };
                let access_p2 = if !depth_p2.is_nan() { 1.0 / depth_p2 } else { 0.0 };
                let total_access = access_p1 + access_p2;

                let ownership = if total_access == 0.0 {
                    0.0

                } else if player == Player::One {
                    access_p1 / total_access

                } else {
                    access_p2 / total_access

                };

                WEIGHTS.shared_piece_control_weights[pd.piece as usize] * ownership

            } else {
                0.0

            };

            score += piece_value * WEIGHTS.piece_control_weight;

        }

        score
    }

    /// Square control score
    pub fn get_square_control_score(&self, player: Player) -> f64 {
        let unique_squares = self.unique_square_control[player as usize].pop_count() as f64;
        let shared_squares = self.shared_square_control.pop_count() as f64;

        unique_squares * WEIGHTS.unique_square_control_weight + shared_squares * WEIGHTS.shared_square_control_weight
        
    }

    /// Backline penalties for pieces that are trapped or stranded
    pub fn backline_penalties(&self, player: Player) -> f64 {
        let mut penalty = 0.0;

        for pd in self.piece_data.iter() {
            if pd.on_active_line[player as usize] {
                if pd.trapped {
                    penalty += WEIGHTS.backline_trapped_penalty[pd.piece as usize];

                } else if pd.stranded {
                    penalty += WEIGHTS.backline_stranded_penalty[pd.piece as usize];

                }

            }

        }

        penalty

    }

    /// Total path score based on the number of paths
    pub fn get_total_path_score(&self, player: Player) -> f64 {
        let mut path_score = 0.0;

        path_score += self.total_paths[player as usize] as f64 * WEIGHTS.per_path_weight;

        path_score

    }

}

/// Storage structure for evaluation signals related to a specific piece.
pub struct PieceData {
    pub piece: Piece,
    pub sq: SQ,

    pub shared: bool,
    pub on_active_line: [bool; 2],

    // ========== SIGNALS ==========

    // RAW PATHING SIGNALS
    pub path_start_counts: [usize; 2],
    pub path_bounce_counts: [usize; 2],
    pub path_continuation_counts: [usize; 2],
    pub path_termination_counts: [usize; 2],
    pub path_average_depths: [f64; 2],
    pub path_min_depths: [f64; 2],

    // COMPUTED SIGNALS
    pub starter_percentage: [f64; 2], // Percentage of total paths that start at this piece
    pub flow_percentage: [f64; 2],    // Percentage of total bounces that are at this piece
    pub continuation_rates: [f64; 2], // Percentage of bounces that are continuations

    // ========== INTREPRETATIONS ==========
    pub trapped: bool,  // Cant do anything
    pub stranded: bool, // Cant reach other pieces
    pub activity_powers: [f64; 2], // Bounces + 10 * continuations, representing how "active" this piece is in the network

}

pub struct EvaluationScore {
    pub total: f64,
    pub p1_total: f64,
    pub p2_total: f64,

    // Components
    pub p1_activity_score: f64,
    pub p2_activity_score: f64,

    pub p1_piece_control: f64,
    pub p2_piece_control: f64,

    pub p1_square_control: f64,
    pub p2_square_control: f64,

    pub p1_path_score: f64,
    pub p2_path_score: f64,

    pub p1_backline_penalty: f64,
    pub p2_backline_penalty: f64,

}

#[derive(Debug, Clone, Copy)]
pub struct EvalWeights {
    pub final_scale: f64,

    // Activity Weights
    pub activity_weight: f64,
    pub activity_type_weights: [f64; 3],
    pub activity_shared_modifier: f64,

    // Network Weights
    pub per_path_weight: f64,

    // Piece Control Weights
    pub piece_control_weight: f64,
    pub unique_piece_control_weights: [f64; 3],
    pub shared_piece_control_weights: [f64; 3],

    // Square Control Weights
    pub square_control_weight: f64,
    pub unique_square_control_weight: f64,
    pub shared_square_control_weight: f64,

    // Penalties
    pub backline_trapped_penalty: [f64; 3],
    pub backline_stranded_penalty: [f64; 3],

}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

impl EvaluationContext {
    pub fn print(&self) {
        println!( "==================================================================================");
        println!("============================== EVALUATION BREAKDOWN ==============================");
        println!( "==================================================================================");
        println!();
        println!("BoardState: {}", self.board);
        println!();
        println!("Total Paths: P1 = {}, P2 = {}", self.total_paths[0], self.total_paths[1]);
        println!("Total Bounces: P1 = {}, P2 = {}", self.total_bounces[0], self.total_bounces[1]);
        println!("Total Continuations: P1 = {}, P2 = {}", self.total_continuations[0], self.total_continuations[1]);
        println!();
        println!("Unique Piece Control:");
        println!("P1 Piece Control: {}", self.unique_piece_control[0]);
        println!("P2 Piece Control: {}", self.unique_piece_control[1]);
        println!("Shared Piece Control: {}", self.shared_piece_control);
        println!();
        println!("Per Piece Data:");
        for pd in self.piece_data.iter() {
            println!("[ POS: {} TYPE: {} ]:", pd.sq.0, pd.piece);
            println!("    - Shared: {}", pd.shared);
            println!("    - Raw Pathing Signals:");
            println!("        - P1 Pathing: [ starts: {}  bounces: {}  continuations: {}  terminations: {}  avg_depth: {:.2} ]", pd.path_start_counts[0], pd.path_bounce_counts[0], pd.path_continuation_counts[0], pd.path_termination_counts[0], pd.path_average_depths[0]);
            println!("        - P2 Pathing: [ starts: {}  bounces: {}  continuations: {}  terminations: {}  avg_depth: {:.2} ]", pd.path_start_counts[1], pd.path_bounce_counts[1], pd.path_continuation_counts[1], pd.path_termination_counts[1], pd.path_average_depths[1]);
            println!("    - Computed Signals:");
            println!("        - Starter Percentages:   [ P1: {:.3}  P2: {:.3} ]", pd.starter_percentage[0], pd.starter_percentage[1]);
            println!("        - Flow Percentages:    [ P1: {:.3}  P2: {:.3} ]", pd.flow_percentage[0], pd.flow_percentage[1]);
            println!("        - Continuation Rates:  [ P1: {:.3}  P2: {:.3} ]", pd.continuation_rates[0], pd.continuation_rates[1]);
            println!("   - Interpretations:");
            println!("        - Activity Powers:         [ P1: {:.3}  P2: {:.3} ]", pd.activity_powers[0], pd.activity_powers[1]);
        }
        println!();
        self.print_extra();
        println!();
        let eval = self.get_evaluation();
        eval.print();

    }

}

impl EvaluationScore {
    pub fn print(&self) {
        println!("Evaluation Score Breakdown:");
        println!("    - Total: {:.3}", self.total);
        println!("        - P1 Total: {:.3}", self.p1_total);
        println!("        - P2 Total: {:.3}", self.p2_total);
        println!("    - Activity:");
        println!("        - P1 Activity: {:.3}", self.p1_activity_score);
        println!("        - P2 Activity: {:.3}", self.p2_activity_score);
        println!("    - Piece Control:");
        println!("        - P1 Piece Control: {:.3}", self.p1_piece_control);
        println!("        - P2 Piece Control: {:.3}", self.p2_piece_control);
        println!("    - Square Control:");
        println!("        - P1 Square Control: {:.3}", self.p1_square_control);
        println!("        - P2 Square Control: {:.3}", self.p2_square_control);
        println!("    - Total Path Counts:");
        println!("        - P1 Paths: {:.3}", self.p1_path_score);
        println!("        - P2 Paths: {:.3}", self.p2_path_score);
        println!("    - Backline Penalties:");
        println!("        - P1 Penalty: {:.3}", self.p1_backline_penalty);
        println!("        - P2 Penalty: {:.3}", self.p2_backline_penalty);

    }

}
