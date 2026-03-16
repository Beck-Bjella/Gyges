//! All logic and functions related to the evaluation of a board.
//! 

use gyges::board::*;
use gyges::board::bitboard::*;
use gyges::core::*;
use gyges::moves::movegen::{GenControlMoveCount, MoveGen, NoQuit};

// Drop in replacement for the old function
pub fn get_evalulation(board: &mut BoardState, mg: &mut MoveGen) -> f64 {
    let evaluation_ctx = EvaluationContext::new(board, mg);
    evaluation_ctx.get_evaluation().total

}

pub const WEIGHTS: EvalWeights = EvalWeights {
    final_scale: 1.0,

    // Activity Weights
    activity_type_weights: [5.0, 2.0, 1.0],
    activity_weight: 100.0,
    activity_shared_modifier: 0.7,
    
    // Network Weights
    total_paths_weight: 0.01,

};


/// The data related to one players position and all of the data
pub struct EvaluationContext {
    pub board: BoardState,

    // General Path data
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

        let p1_network_score: f64 = self.get_network_score(Player::One) * WEIGHTS.final_scale;
        let p2_network_score: f64 = self.get_network_score(Player::Two) * WEIGHTS.final_scale;

        let p1_total = 0.0 +
            p1_activity_score +
            p1_network_score;

        let p2_total = 0.0 + 
            p2_activity_score +
            p2_network_score;

        let total = p1_total - p2_total;

        EvaluationScore {
            total,
            p1_total,
            p2_total,

            // Components
            p1_activity_score,
            p2_activity_score,

            p1_network_score,
            p2_network_score,

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

    // FUTURE: WILL HAVE MORE NETWORK FACTORS
    pub fn get_network_score(&self, player: Player) -> f64 {
        let mut network_score = 0.0;

        network_score += self.total_paths[player as usize] as f64 * WEIGHTS.total_paths_weight;

        network_score

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

    pub activity_powers: [f64; 2],

}

pub struct EvaluationScore {
    pub total: f64,
    pub p1_total: f64,
    pub p2_total: f64,

    // Components
    pub p1_activity_score: f64,
    pub p2_activity_score: f64,

    pub p1_network_score: f64,
    pub p2_network_score: f64,

}

#[derive(Debug, Clone, Copy)]
pub struct EvalWeights {
    pub final_scale: f64,

    // Activity Weights
    pub activity_type_weights: [f64; 3],
    pub activity_weight: f64,
    pub activity_shared_modifier: f64,

    // Network Weights
    pub total_paths_weight: f64,
    
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
        println!("    - Network:");
        println!("        - P1 Network: {:.3}", self.p1_network_score);
        println!("        - P2 Network: {:.3}", self.p2_network_score);

    }
}