//! All logic and functions related to the evaluation of a board.
//!

use gyges::board::bitboard::*;
use gyges::core::masks::RANKS;
use gyges::{AllResults, core::*};
use gyges::moves::movegen::{MoveGen};
use gyges::{board::*};

// Drop in replacement for the old function
pub fn get_evalulation(board: &mut BoardState, mg: &mut MoveGen) -> f64 {
    let evaluation_ctx = EvaluationContext::new(board, mg);
    evaluation_ctx.get_evaluation().total

}

/// Ones piece-square table 
#[rustfmt::skip]
pub const PST_ONE: [f64; 36] = [
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  // rank 0
     25.0,  50.0,  75.0,  75.0,  50.0,  25.0,  // rank 1
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,  // rank 2+
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
];

// Twos piece-square table
#[rustfmt::skip]
pub const PST_TWO: [f64; 36] = [
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  // rank 0
     50.0, 100.0, 100.0, 100.0, 100.0,  50.0,  // rank 1
     25.0,  50.0,  75.0,  75.0,  50.0,  25.0,  // rank 2+
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
];

// Threes piece-square table
#[rustfmt::skip]
pub const PST_THREE: [f64; 36] = [
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  // rank 0
     50.0, 100.0, 100.0, 100.0, 100.0,  50.0,  // rank 1
     25.0,  50.0,  75.0,  75.0,  50.0,  25.0,  // rank 2+
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
];

pub const WEIGHTS: EvalWeights = EvalWeights {
    phase: PhaseWeights {
        phase_table: [1.0, 1.0, 1.0, 1.0, 0.66, 0.33, 0.0],

    },

    earlygame: EarlyGameWeights {
        final_scale: 5.0,

        per_path_weight: 1.0,
        low_backline_penalty: -400.0,

    },

    midgame: MidGameWeights {
        final_scale: 1.0,

        // Activity Weights
        activity_weight: 0.0,
        activity_type_weights: [1.0, 1.0, 1.0],
        activity_shared_modifier: 0.25,

        // Network Weights
        per_path_weight: 1.0,

        // Piece Control Weights
        piece_control_weight: 1.0,
        unique_piece_control_weights: [100.0, 500.0, 500.0],
        shared_piece_control_weights: [0.0, 0.0, 0.0],

        // Square Control Weights
        square_control_weight: 0.25,
        unique_square_control_weight: 10.0,
        shared_square_control_weight: 5.0,

        // One Control (Exponential)
        one_control_base: 400.0,
        one_control_exp: 2.0,
        shared_one_weight: 0.3,

        // Penalties
        backline_trapped_penalty: [-1500.0, -300.0, -300.0],
        backline_stranded_penalty: [-1500.0, -300.0, -300.0],

    }

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
            let start_continuation_counts = [
                p1.start_continuation_count[pos],
                p2.start_continuation_count[pos],
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
            
            let trapped = [
                on_active_line[0] && path_start_counts[0] == 0,
                on_active_line[1] && path_start_counts[1] == 0,
            ];
            let stranded = [
                on_active_line[0] && start_continuation_counts[0] == 0,
                on_active_line[1] && start_continuation_counts[1] == 0,
            ];

            let material_score = {
                let mut scores = [0.0, 0.0];

                for player in [Player::One, Player::Two] {
                    let piece_value = if (unique_piece_control[player as usize].0 & sq.bit()) != 0 {
                        WEIGHTS.midgame.unique_piece_control_weights[piece as usize]

                    } else if (shared_piece_control & sq.bit()).0 != 0 {
                        let depth_p1 = path_min_depths[0];
                        let depth_p2 = path_min_depths[1];

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

                        WEIGHTS.midgame.shared_piece_control_weights[piece as usize] * ownership

                    } else {
                        0.0

                    };

                    scores[player as usize] = piece_value * WEIGHTS.midgame.piece_control_weight;

                }

                scores

            };

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

                material_score,
                
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

    /// Evaluation entry point
    pub fn get_evaluation(&self) -> EvaluationScore {
        let phase = self.get_phase();
    
        let mg = self.mg_eval();
        let eg = self.eg_eval();

        let total = phase * mg.total + (1.0 - phase) * eg.total;
        
        EvaluationScore { 
            total, 
            phase, 
            mg, 
            eg

        }

    }

    /// Gets the game phase based on opp piece count
    pub fn get_phase(&self) -> f64 {
        let count = (self.board.piece_bb & RANKS[5]).pop_count().min(6) as usize;
        WEIGHTS.phase.phase_table[count]

    }

}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// EARLY GAME EVAL ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


impl EvaluationContext {
    /// Early game evaluation
    pub fn eg_eval(&self) -> EarlyGameScore {
        let p1_path_score = self.eg_path_score(Player::One) * WEIGHTS.earlygame.final_scale;
        let p2_path_score = self.eg_path_score(Player::Two) * WEIGHTS.earlygame.final_scale;

        let p1_development = self.eg_development_score(Player::One) * WEIGHTS.earlygame.final_scale;
        let p2_development = self.eg_development_score(Player::Two) * WEIGHTS.earlygame.final_scale;

        let p1_total = 0.0 +
            p1_path_score +
            p1_development;

        let p2_total = 0.0 +
            p2_path_score +
            p2_development;

        let total    = p1_total - p2_total;

        EarlyGameScore {
            total,
            p1_total,
            p2_total,

            p1_path_score,
            p2_path_score,

            p1_development,
            p2_development,

        }

    }

    pub fn eg_path_score(&self, player: Player) -> f64 {
        self.total_paths[player as usize] as f64 * WEIGHTS.earlygame.per_path_weight

    }

    pub fn eg_development_score(&self, player: Player) -> f64 {
        let mut score = 0.0;

        let psts = [&PST_ONE, &PST_TWO, &PST_THREE];

        let mut pieces = self.board.piece_bb.0;
        while pieces != 0 {
            let sq = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;

            let piece = self.board.data[sq] as usize;
            let idx = if player == Player::One { sq } else { 35 - sq };
            score += psts[piece][idx];

        }

        let home_rank = if player == Player::One { RANKS[0] } else { RANKS[5] };
        let home_count = (self.board.piece_bb & home_rank).pop_count() as f64;
        let below_three = (3.0 - home_count).max(0.0);
        
        score += below_three * WEIGHTS.earlygame.low_backline_penalty;

        score

    }

}


///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// MID-END GAME EVAL ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

impl EvaluationContext {
    pub fn mg_eval(&self) -> MidGameScore {
        let p1_material = self.mg_material_score(Player::One) * WEIGHTS.midgame.final_scale;
        let p2_material = self.mg_material_score(Player::Two) * WEIGHTS.midgame.final_scale;

        let p1_square_control = self.mg_square_control_score(Player::One) * WEIGHTS.midgame.final_scale;
        let p2_square_control = self.mg_square_control_score(Player::Two) * WEIGHTS.midgame.final_scale;

        let p1_path_score: f64 = self.mg_path_score(Player::One) * WEIGHTS.midgame.final_scale;
        let p2_path_score: f64 = self.mg_path_score(Player::Two) * WEIGHTS.midgame.final_scale;

        let p1_backline_penalty = self.mg_backline_penalties(Player::One) * WEIGHTS.midgame.final_scale;
        let p2_backline_penalty = self.mg_backline_penalties(Player::Two) * WEIGHTS.midgame.final_scale;

        let one_control_score = self.mg_one_control_score() * WEIGHTS.midgame.final_scale;

        let p1_total: f64 = 0.0 +
            p1_material +
            p1_square_control +
            p1_path_score +
            p1_backline_penalty;
    
        let p2_total: f64 = 0.0 +
            p2_material +
            p2_square_control +
            p2_path_score +
            p2_backline_penalty;

        let total = (p1_total - p2_total) + one_control_score;

        MidGameScore {
            total,
            p1_total,
            p2_total,

            p1_material,
            p2_material,

            p1_square_control,
            p2_square_control,

            p1_path_score,
            p2_path_score,

            p1_backline_penalty,
            p2_backline_penalty,

            one_control_score

        }

    }

    /// Material score
    pub fn mg_material_score(&self, player: Player) -> f64 {
        let mut score = 0.0;

        for pd in self.piece_data.iter() {
            score += pd.material_score[player as usize];

        }

        score
    }

    /// Square control score
    pub fn mg_square_control_score(&self, player: Player) -> f64 {
        let unique_squares = self.unique_square_control[player as usize].pop_count() as f64;
        let shared_squares = self.shared_square_control.pop_count() as f64;

        unique_squares * WEIGHTS.midgame.unique_square_control_weight + shared_squares * WEIGHTS.midgame.shared_square_control_weight
        
    }

    /// Backline penalties for pieces that are trapped or stranded
    pub fn mg_backline_penalties(&self, player: Player) -> f64 {
        let mut penalty = 0.0;

        for pd in self.piece_data.iter() {
            if pd.trapped[player as usize] {
                penalty += WEIGHTS.midgame.backline_trapped_penalty[pd.piece as usize];

            } else if pd.stranded[player as usize] {
                penalty += WEIGHTS.midgame.backline_stranded_penalty[pd.piece as usize];

            }

        }

        penalty

    }

    /// Exponential score based on unique 1-piece control differential.
    /// Shared 1s contribute fractionally via depth-based ownership.
    pub fn mg_one_control_score(&self) -> f64 {
        let mut p1_ones: f64 = 0.0;
        let mut p2_ones: f64 = 0.0;

        for pd in self.piece_data.iter() {
            if pd.piece != Piece::One { continue; }

            if (self.unique_piece_control[0].0 & pd.sq.bit()) != 0 {
                p1_ones += 1.0;

            } else if (self.unique_piece_control[1].0 & pd.sq.bit()) != 0 {
                p2_ones += 1.0;

            } else if (self.shared_piece_control.0 & pd.sq.bit()) != 0 {
                let d1 = pd.path_min_depths[0];
                let d2 = pd.path_min_depths[1];
                let a1 = if d1.is_finite() { 1.0 / d1.max(1.0) } else { 0.0 };
                let a2: f64 = if d2.is_finite() { 1.0 / d2.max(1.0) } else { 0.0 };
                let total = a1 + a2;

                if total > 0.0 {
                    p1_ones += (a1 / total) * WEIGHTS.midgame.shared_one_weight;
                    p2_ones += (a2 / total) * WEIGHTS.midgame.shared_one_weight;

                }

            }

        }

        let base = WEIGHTS.midgame.one_control_base;
        let exp = WEIGHTS.midgame.one_control_exp;
        base * (exp.powf(p1_ones) - exp.powf(p2_ones))

    }

    /// Total path score based on the number of paths
    pub fn mg_path_score(&self, player: Player) -> f64 {
        let mut path_score = 0.0;

        path_score += self.total_paths[player as usize] as f64 * WEIGHTS.midgame.per_path_weight;

        path_score

    }

}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/// Top-level weights container.
#[derive(Debug, Clone, Copy)]
pub struct EvalWeights {
    pub phase: PhaseWeights,
    pub earlygame: EarlyGameWeights,
    pub midgame: MidGameWeights,

}

/// Weights specifically for phase detection and early game evaluation.
#[derive(Debug, Clone, Copy)]
pub struct PhaseWeights {
    pub phase_table: [f64; 7],

}

/// Early-game weights.
#[derive(Debug, Clone, Copy)]
pub struct EarlyGameWeights {
    pub final_scale: f64,

    pub per_path_weight: f64,
    pub low_backline_penalty: f64,

}

#[derive(Debug, Clone, Copy)]
pub struct MidGameWeights {
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

    // One Control (Exponential)
    pub one_control_base: f64,
    pub one_control_exp: f64,
    pub shared_one_weight: f64,

    // Penalties
    pub backline_trapped_penalty: [f64; 3],
    pub backline_stranded_penalty: [f64; 3],

}

pub struct EvaluationScore {
    pub total: f64,
    pub phase: f64,
    pub eg: EarlyGameScore,
    pub mg: MidGameScore,
    
}

pub struct EarlyGameScore {
    pub total: f64,
    pub p1_total: f64,
    pub p2_total: f64,

    pub p1_path_score: f64,
    pub p2_path_score: f64,

    pub p1_development: f64,
    pub p2_development: f64,

}

pub struct MidGameScore {
    pub total: f64,
    pub p1_total: f64,
    pub p2_total: f64,

    pub p1_material: f64,
    pub p2_material: f64,

    pub p1_square_control: f64,
    pub p2_square_control: f64,

    pub p1_path_score: f64,
    pub p2_path_score: f64,

    pub p1_backline_penalty: f64,
    pub p2_backline_penalty: f64,

    pub one_control_score: f64,

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

    // ========== INTERPRETATIONS ==========
    pub trapped: [bool; 2],  // per-player: on active line and can't bounce
    pub stranded: [bool; 2], // per-player: on active line and can't reach other pieces
    pub activity_powers: [f64; 2], // Bounces + 10 * continuations, representing how "active" this piece is in the network

    // ========== SCORES ==========
    pub material_score: [f64; 2], // Material value for each player

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
        println!();
        self.print_material_breakdown();
        println!();
        let eval = self.get_evaluation();
        eval.print();

    }

}

impl EvaluationScore {
    pub fn print(&self) {
        println!("Evaluation Score Breakdown:");
        println!("    - Total: {:.3}  (phase: {:.3})", self.total, self.phase);
        println!();
        println!("    [Earlygame weight: {:.3}]", 1.0 - self.phase);
        self.eg.print();
        println!("    [Midgame  weight: {:.3}]", self.phase);
        self.mg.print();
        println!();
       

    }

}

impl EarlyGameScore {
    pub fn print(&self) {
        println!("        - Total: {:.3}", self.total);
        println!("            P1: {:.3}  P2: {:.3}", self.p1_total, self.p2_total);
        println!("        - Path Score:        P1: {:.3}  P2: {:.3}", self.p1_path_score, self.p2_path_score);
        println!("        - Development:       P1: {:.3}  P2: {:.3}", self.p1_development, self.p2_development);

    }

}

impl MidGameScore {
    pub fn print(&self) {
        println!("        - Total: {:.3}", self.total);
        println!("            P1: {:.3}  P2: {:.3}", self.p1_total, self.p2_total);
        println!("        - Material:        P1: {:.3}  P2: {:.3}", self.p1_material, self.p2_material);
        println!("        - Square Control:   P1: {:.3}  P2: {:.3}", self.p1_square_control, self.p2_square_control);
        println!("        - Path Score:       P1: {:.3}  P2: {:.3}", self.p1_path_score, self.p2_path_score);
        println!("        - Backline Penalty: P1: {:.3}  P2: {:.3}", self.p1_backline_penalty, self.p2_backline_penalty);
        println!("        - One Control (exp): {:.3}", self.one_control_score);

    }

}
