//! Neural network evaluation — fast board eval without gen_all.
//!
//! Architecture: 180 → 64 (ReLU) → 1 (tanh)
//! Weights loaded from `weights.bin` exported by the Python training script.
//!
//! The board is always encoded from the current player's perspective:
//! their home rank maps to rank 0. Output in [-NETWORK_SCALE, +NETWORK_SCALE],
//! positive = current player winning.

use gyges::{board::*, core::*};
use std::fs;
use std::sync::OnceLock;

/// Global network instance — initialised once at startup via `init_network`.
static NETWORK: OnceLock<GygesNet> = OnceLock::new();

/// Load weights from `path` and store in the global instance.
/// Call this once at startup before any search begins.
pub fn init_network(path: &str) -> Result<(), String> {
    let net = GygesNet::load(path)?;
    NETWORK.set(net).map_err(|_| "Network already initialised".to_string())
}

/// Returns the raw network score for both players, or None if not loaded.
/// `p1_control` / `p2_control` are the unique-piece-control bitboards from EvaluationContext.
pub fn try_evalulation_nn(board: &BoardState, p1_control: u64, p2_control: u64) -> Option<(f64, f64)> {
    let net = NETWORK.get()?;
    Some((
        net.eval(board, Player::One,  p1_control, p2_control),
        -net.eval(board, Player::Two,  p1_control, p2_control), // negate to P1-relative for display
    ))
}

/// Drop-in replacement for `get_evalulation` — requires control bitboards from EvaluationContext.
///
/// Returns a score from the current player's perspective (negamax-compatible):
///   positive = current player is winning
///   negative = current player is losing
pub fn get_evalulation_nn(board: &BoardState, player: Player, p1_control: u64, p2_control: u64) -> f64 {
    let net = NETWORK.get().expect("Network not initialised — call init_network first");
    net.eval(board, player, p1_control, p2_control)
}

/// Scale the tanh output to match the hand-crafted eval's magnitude.
pub const NETWORK_SCALE: f64 = 10000.0;

/// Two-layer MLP: 180 → 64 (ReLU) → 1 (tanh)
///
/// Input encoding — 5 features per square × 36 squares = 180:
///   Board is always oriented so the current player's home rank is at rank 0.
///   features[sq*5 + 0..3] = one-hot piece type (empty, 1-ring, 2-ring, 3-ring)
///   features[sq*5 + 4]    = control scalar (+1 my unique, 0 shared/empty, -1 opponent unique)
///
/// Weight layout in `weights.bin` (float32, little-endian):
///   net.0.weight  [64 × 180]   11520 floats  offset 0
///   net.0.bias    [64]             64 floats  offset 11520
///   net.3.weight  [1  × 64]       64 floats  offset 11584
///   net.3.bias    [1]               1 float   offset 11648
///   Total: 11649 floats = 46596 bytes
pub struct GygesNet {
    /// First-layer weights, transposed for cache-friendly access during sparse evaluation.
    /// Layout: w1t[input_feature][neuron] — each column (all 64 neuron weights for one input)
    /// is contiguous in memory, enabling vectorized addition in the fused forward pass.
    w1t: Box<[[f32; 64]; 180]>,
    b1: [f32; 64],
    w2: [f32; 64],
    b2: f32,
}

impl GygesNet {
    /// Load weights from the binary file produced by the Python training script.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("Cannot read weight file '{}': {}", path, e))?;

        let expected_bytes = 11649 * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Weight file is {} bytes, expected {} (11649 × float32). \
                 Check that the Python export matches the 180→64→1 architecture.",
                bytes.len(),
                expected_bytes
            ));
        }

        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // net.0.weight: [64, 180] row-major in the file — transpose to [180, 64] for
        // cache-friendly sparse access: w1t[feature][neuron]
        let mut w1t = Box::new([[0f32; 64]; 180]);
        for i in 0..64 {
            for j in 0..180 {
                w1t[j][i] = floats[i * 180 + j];
            }
        }

        // net.0.bias: [64] — offset 11520
        let mut b1 = [0f32; 64];
        for i in 0..64 {
            b1[i] = floats[11520 + i];
        }

        // net.3.weight: [1, 64] — offset 11584
        let mut w2 = [0f32; 64];
        for i in 0..64 {
            w2[i] = floats[11584 + i];
        }

        // net.3.bias: scalar — offset 11648
        let b2 = floats[11648];

        Ok(Self { w1t, b1, w2, b2 })
    }

    /// Fused encode + forward pass.
    ///
    /// The board is encoded from the active player's perspective (P2 flipped vertically).
    /// For each square sq in 0..36:
    ///   features[sq*5 + 0] = 1.0  →  empty
    ///   features[sq*5 + 1] = 1.0  →  Piece::One
    ///   features[sq*5 + 2] = 1.0  →  Piece::Two
    ///   features[sq*5 + 3] = 1.0  →  Piece::Three
    ///   features[sq*5 + 4] = control scalar (+1.0 my / 0.0 shared / -1.0 opponent)
    ///
    /// Instead of building this 180-element feature vector and then multiplying by the weight
    /// matrix, we exploit the sparsity of the input: each square contributes exactly one
    /// non-zero one-hot feature (+1.0) and optionally one control feature (+1.0 or -1.0).
    /// That means only ~36-72 of the 180 inputs are non-zero.
    ///
    /// For each non-zero feature, we directly add (or subtract) the corresponding weight
    /// column into the hidden accumulator. This replaces 64×180 = 11,520 multiply-adds
    /// with ~36-72 vectorizable additions of 64-element arrays — no multiplies needed.
    ///
    /// The transposed weight layout (w1t[feature][neuron]) ensures each column is contiguous
    /// in memory, enabling auto-vectorization of the inner loop.
    pub fn eval(&self, board: &BoardState, player: Player, p1_control: u64, p2_control: u64) -> f64 {
        // Start with bias — equivalent to the constant term in each neuron
        let mut hidden = self.b1;

        let (my_control, opp_control) = match player {
            Player::One => (p1_control, p2_control),
            Player::Two => (p2_control, p1_control),
        };

        for sq in 0..36usize {
            // For P2, flip the board vertically so active player's home rank is always rank 0
            let board_sq = match player {
                Player::One => sq,
                Player::Two => (5 - sq / 6) * 6 + sq % 6,
            };

            let bit = 1u64 << board_sq;

            // Determine which piece (if any) is on this square
            let piece_idx = if board.piece_bb.0 & bit != 0 {
                board.data[board_sq] as usize + 1
            } else {
                0
            };

            // One-hot piece feature: add the weight column for this feature
            // (equivalent to multiplying by 1.0 — no multiply needed)
            let piece_col = &self.w1t[sq * 5 + piece_idx];
            for i in 0..64 {
                hidden[i] += piece_col[i];
            }

            // Control feature: only applies to occupied squares
            if piece_idx != 0 {
                let ctrl_col = &self.w1t[sq * 5 + 4];
                if my_control & bit != 0 {
                    // +1.0 control — add the weight column
                    for i in 0..64 {
                        hidden[i] += ctrl_col[i];
                    }
                } else if opp_control & bit != 0 {
                    // -1.0 control — subtract the weight column
                    for i in 0..64 {
                        hidden[i] -= ctrl_col[i];
                    }
                }
                // 0.0 control (shared/no control) — skip entirely
            }
        }

        // ReLU activation
        for i in 0..64 {
            hidden[i] = hidden[i].max(0.0);
        }

        // Layer 2: out = tanh(w2 · hidden + b2) — dense, since hidden is fully populated
        let mut out = self.b2;
        for i in 0..64 {
            out += self.w2[i] * hidden[i];
        }

        (out.tanh() as f64) * NETWORK_SCALE
    }
}
