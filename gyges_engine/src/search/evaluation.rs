//! All logic and functions related to the evaluation of a board.
//! 

use std::path;

use gyges::board::*;
use gyges::board::bitboard::*;
use gyges::core::*;
use gyges::moves::movegen::{GenControlMoveCount, MoveGen, NoQuit};


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
    activity_weight: 2000.0,
    activity_type_weights: [1.0, 1.0, 1.0],
    activity_shared_modifier: 0.75,
    
    // Network Weights
    total_paths_weight: 0.01,

    // Material Weights
    material_weight: 1.0,
    unique_piece_control_weights: [40.0, 4.0, 2.0],
    shared_piece_control_weights: [3.0, 2.0, 1.0],
    
    backline_isolation_penalty: [-0.0, -0.0, -0.0],

};

/// The data related to one players position and all of the data
pub struct EvaluationContext {
    pub board: BoardState,

    // Network Data
    pub total_paths: [usize; 2],
    pub total_bounces: [usize; 2],
    pub total_continuations: [usize; 2],

    // Piece data
    pub unique_piece_control: [BitBoard; 2],
    pub shared_piece_control: BitBoard,
    pub piece_data: Vec<PieceData>,

}

impl EvaluationContext {
    /// Creates all relevant data for the given player
    /// 
    /// **NOTE**: Most likely will be moved to a EvalationContext struct later. So that we can reuse data bewteen players.
    /// 
    pub fn new(board: &mut BoardState, mg: &mut MoveGen) -> Self {
        let p1 = unsafe { mg.gen::<GenControlMoveCount, NoQuit>(board, Player::One) };
        let p2 = unsafe { mg.gen::<GenControlMoveCount, NoQuit>(board, Player::Two) };
        let p1_path_data = unsafe { mg.gen_path_data::<NoQuit>(board, Player::One) };
        let p2_path_data = unsafe { mg.gen_path_data::<NoQuit>(board, Player::Two) };

        // Collect general total path data
        let total_paths = [
            p1_path_data.start_count.iter().sum(),
            p2_path_data.start_count.iter().sum(),
        ];
        let total_bounces = [
            p1_path_data.bounce_count.iter().sum(),
            p2_path_data.bounce_count.iter().sum(),
        ];
        let total_continuations = [
            p1_path_data.continuation_count.iter().sum(),
            p2_path_data.continuation_count.iter().sum(),
        ];
        
        // Piece Control
        let unique_piece_control = [
            p1.controlled_pieces & !p2.controlled_pieces,
            p2.controlled_pieces & !p1.controlled_pieces,
        ];
        let shared_piece_control = p1.controlled_pieces & p2.controlled_pieces;

        // Per Piece Data
        let mut piece_data: Vec<PieceData> = Vec::new();
        for pos in board.piece_bb.clone().get_data() {
            let piece = board.data[pos];
            let sq = SQ(pos as u8);

            let shared = (shared_piece_control.0 & sq.bit()) != 0;

            // Raw signals
            let path_start_counts = [p1_path_data.start_count[pos], p2_path_data.start_count[pos]];
            let path_bounce_counts = [p1_path_data.bounce_count[pos], p2_path_data.bounce_count[pos]];
            let path_continuation_counts = [p1_path_data.continuation_count[pos], p2_path_data.continuation_count[pos]];
            let path_termination_counts = [
                p1_path_data.bounce_count[pos] - p1_path_data.continuation_count[pos], 
                p2_path_data.bounce_count[pos] - p2_path_data.continuation_count[pos]
            ];
            let path_average_depths = [
                p1_path_data.depth[pos] as f64 / p1_path_data.bounce_count[pos] as f64,
                p2_path_data.depth[pos] as f64 / p2_path_data.bounce_count[pos] as f64,
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
                path_continuation_counts[1] as f64 / path_bounce_counts[1] as f64
            ];

            // Interpretations
            let activity_powers = [
                flow_percentage[0] * continuation_rates[0],
                flow_percentage[1] * continuation_rates[1],
            ];  

            let data = PieceData {
                piece,
                sq,
                shared,

                path_start_counts,
                path_bounce_counts,
                path_continuation_counts,
                path_termination_counts,
                path_average_depths,

                starter_percentage,
                flow_percentage,
                continuation_rates,

                activity_powers,

            };

            piece_data.push(data);

        }

        Self {
            board: board.clone(),

            total_paths,
            total_bounces,
            total_continuations,

            unique_piece_control,
            shared_piece_control,
            piece_data,

        }

    }

    pub fn get_evaluation(&self) -> EvaluationScore {
        let p1_activity_score: f64 = self.get_activity_score(Player::One) * WEIGHTS.final_scale;
        let p2_activity_score: f64 = self.get_activity_score(Player::Two) * WEIGHTS.final_scale;

        let p1_material_score: f64 = self.get_material_score(Player::One) * WEIGHTS.final_scale;
        let p2_material_score: f64 = self.get_material_score(Player::Two) * WEIGHTS.final_scale;

        let p1_path_score: f64 = self.get_total_path_score(Player::One) * WEIGHTS.final_scale;
        let p2_path_score: f64 = self.get_total_path_score(Player::Two) * WEIGHTS.final_scale;

        let backline_isolation_penalty_p1 = self.backline_isolation_penalty(Player::One) * WEIGHTS.final_scale;
        let backline_isolation_penalty_p2 = self.backline_isolation_penalty(Player::Two) * WEIGHTS.final_scale;

        let p1_total: f64 = 0.0 + 
            p1_activity_score +
            p1_material_score +
            p1_path_score +
            backline_isolation_penalty_p1;

        let p2_total: f64 = 0.0 +
            p2_activity_score +
            p2_material_score +
            p2_path_score +
            backline_isolation_penalty_p2;

        let total = p1_total - p2_total;

        EvaluationScore {
            total,
            p1_total,
            p2_total,

            // Components
            p1_activity_score,
            p2_activity_score,

            p1_material_score,
            p2_material_score,

            p1_path_score,
            p2_path_score,

            backline_isolation_penalty_p1,
            backline_isolation_penalty_p2,
        }

    }

    /// How active pieces are
    pub fn get_activity_score(&self, player: Player) -> f64 {
        let mut activity_score = 0.0;

        for pd in self.piece_data.iter() {
            let activity_power: f64 = pd.activity_powers[player as usize];
            if !activity_power.is_finite() {
                continue;
            }

            let type_weight = WEIGHTS.activity_type_weights[pd.piece as usize];
            let shared_modifier = if pd.shared { WEIGHTS.activity_shared_modifier } else { 1.0 };

            activity_score += activity_power
                * type_weight
                * WEIGHTS.activity_weight
                * shared_modifier;
   
        }

        activity_score

    }

    /// Material score based on piece control
    pub fn get_material_score(&self, player: Player) -> f64 {
        let mut material_score = 0.0;

        for pd in self.piece_data.iter() {
            let piece_value = if (self.unique_piece_control[player as usize].0 & pd.sq.bit()) != 0 {
                WEIGHTS.unique_piece_control_weights[pd.piece as usize]

            } else if (self.shared_piece_control.0 & pd.sq.bit()) != 0 {
                WEIGHTS.shared_piece_control_weights[pd.piece as usize]

            } else {
                0.0

            };

            material_score += piece_value * WEIGHTS.material_weight;

        }

        material_score


    }
    
    pub fn backline_isolation_penalty(&self, player: Player) -> f64 {
        let mut penalty = 0.0;

        for pd in self.piece_data.iter() {
            if pd.path_start_counts[player as usize] > 0 {
                if (pd.path_continuation_counts[player as usize]) == 0 {
                    penalty += WEIGHTS.backline_isolation_penalty[pd.piece as usize];

                }
                               
            }

        }

        penalty

    }

    /// Total path score based on the number of paths
    pub fn get_total_path_score(&self, player: Player) -> f64 {
        let mut path_score = 0.0;

        path_score += self.total_paths[player as usize] as f64 * WEIGHTS.total_paths_weight;

        path_score

    }
    

}

/// Storage structure for evaluation signals related to a specific piece.
pub struct PieceData {
    pub piece: Piece,
    pub sq: SQ,

    pub shared: bool,

    // ===============================
    // ===== SIGNALS =====
    // ===============================

    // RAW PATHING SIGNALS
    pub path_start_counts: [usize; 2],
    pub path_bounce_counts: [usize; 2],
    pub path_continuation_counts: [usize; 2],
    pub path_termination_counts: [usize; 2],
    pub path_average_depths: [f64; 2],

    // COMPUTED SIGNALS
    pub starter_percentage: [f64; 2], // Percentage of total paths that start at this piece
    pub flow_percentage: [f64; 2], // Percentage of total bounces that are at this piece
    pub continuation_rates: [f64; 2], // Percentage of bounces that are continuations

    // ===========================
    // ===== INTREPRETATIONS =====
    // ===========================
 
    pub activity_powers: [f64; 2], // Flow percentage * continuation rate, representing how "active" this piece is in the network

}

pub struct EvaluationScore {
    pub total: f64,
    pub p1_total: f64,
    pub p2_total: f64,

    // Components
    pub p1_activity_score: f64,
    pub p2_activity_score: f64,

    pub p1_material_score: f64,
    pub p2_material_score: f64,

    pub p1_path_score: f64,
    pub p2_path_score: f64,

    pub backline_isolation_penalty_p1: f64,
    pub backline_isolation_penalty_p2: f64,

}

#[derive(Debug, Clone, Copy)]
pub struct EvalWeights {
    pub final_scale: f64,

    // Activity Weights
     pub activity_weight: f64,
    pub activity_type_weights: [f64; 3],
    pub activity_shared_modifier: f64,

    // Network Weights
    pub total_paths_weight: f64,

    // Material Weights
    pub material_weight: f64,
    pub unique_piece_control_weights: [f64; 3],
    pub shared_piece_control_weights: [f64; 3],

    // Penalties
    pub backline_isolation_penalty: [f64; 3],
    
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

impl EvaluationContext {
    pub fn print(&self) {
        println!("==================================================================================");
        println!("============================== EVALUATION BREAKDOWN ==============================");
        println!("==================================================================================");
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
        println!("    - Material:");
        println!("        - P1 Material: {:.3}", self.p1_material_score);
        println!("        - P2 Material: {:.3}", self.p2_material_score);
        println!("    - Total Path Counts:");
        println!("        - P1 Paths: {:.3}", self.p1_path_score);
        println!("        - P2 Paths: {:.3}", self.p2_path_score);
        println!("    - Backline Isolation Penalty:");
        println!("        - P1 Penalty: {:.3}", self.backline_isolation_penalty_p1);
        println!("        - P2 Penalty: {:.3}", self.backline_isolation_penalty_p2);

    }
}