//! NN evaluation. FEATURE_COUNT sparse pair-factored inputs → L1_SIZE (ReLU) → 1 (tanh).
//! Board encoded from the side-to-move's perspective (home rank = rank 0).
//! Output in [-NETWORK_SCALE, +NETWORK_SCALE], positive = side-to-move winning.

use gyges::{board::*, core::*};
use std::fs;

/// Global network instance — loaded via `load_network`, read during search.
/// Must not be replaced while a search is in progress (same discipline as the TT).
static mut NETWORK: Option<GygesNet> = None;
static mut NETWORK_NAME: Option<String> = None;

/// Load weights from `path` and install them as the active network.
/// Can be called multiple times to swap weights — must only be called when no
/// search is running.
pub fn load_network(path: &str) -> Result<(), String> {
    let net = GygesNet::load(path)?;
    let name = std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    unsafe {
        NETWORK = Some(net);
        NETWORK_NAME = Some(name);
    }
    Ok(())
}

/// Returns true if a network is currently loaded.
pub fn network_loaded() -> bool {
    unsafe { NETWORK.is_some() }
}

/// Returns the filename of the loaded network, if any.
pub fn network_name() -> Option<&'static str> {
    unsafe { NETWORK_NAME.as_deref() }
}

/// Returns the raw network score for both players (P1-relative), or None if not loaded.
pub fn try_evalulation_nn(board: &BoardState) -> Option<(f64, f64)> {
    let net = unsafe { NETWORK.as_ref() }?;
    Some((
        net.eval(board, Player::One),
        -net.eval(board, Player::Two),
    ))
}

/// Drop-in replacement for `get_evalulation`.
///
/// Returns a score from the current player's perspective (negamax-compatible).
/// Panics if no network is loaded — gate with `network_loaded()` or a config flag.
pub fn get_evalulation_nn(board: &BoardState, player: Player) -> f64 {
    let net = unsafe { NETWORK.as_ref() }.expect("Network not loaded — call load_network first");
    net.eval(board, player)
}

/// Scales tanh output to match the hand-crafted eval's magnitude.
pub const NETWORK_SCALE: f64 = 10000.0;

/// Layer 1 width.
pub const L1_SIZE: usize = 1024;

/// Total input features: 108 singletons + 3×630 same-type pairs + 3×1296 cross-type pairs.
pub const FEATURE_COUNT: usize = 5886;

// Offsets into the sparse feature index space (must match the Python encoder).
const PAIR_11_OFFSET: u32 = 108;
const PAIR_22_OFFSET: u32 = 738;
const PAIR_33_OFFSET: u32 = 1368;
const PAIR_12_OFFSET: u32 = 1998;
const PAIR_13_OFFSET: u32 = 3294;
const PAIR_23_OFFSET: u32 = 4590;

/// FEATURE_COUNT → L1_SIZE (ReLU) → 1 (tanh).
/// Sparse input: singletons + same-type pairs + cross-type pairs.
/// Weights file (float32 LE): w1 [L1 × F], b1 [L1], w2 [1 × L1], b2 [1].
pub struct GygesNet {
    /// Layer-1 weights, transposed: w1t[feature][neuron]. Column-major sparse adds.
    w1t: Box<[[f32; L1_SIZE]; FEATURE_COUNT]>,
    b1: [f32; L1_SIZE],
    w2: [f32; L1_SIZE],
    b2: f32,

}

impl GygesNet {
    /// Load weights from the binary file produced by the Python training script.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("Cannot read weight file '{}': {}", path, e))?;

        let expected_floats = L1_SIZE * FEATURE_COUNT + L1_SIZE + L1_SIZE + 1;
        let expected_bytes = expected_floats * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Weight file is {} bytes, expected {} ({} × float32). \
                 Check the Python export matches {}→{}→1.",
                bytes.len(),
                expected_bytes,
                expected_floats,
                FEATURE_COUNT,
                L1_SIZE
            ));

        }

        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // fc1.weight: [L1_SIZE, FEATURE_COUNT] row-major in the file — transpose to
        // [FEATURE_COUNT, L1_SIZE] for cache-friendly sparse access: w1t[feature][neuron].
        // Allocate via Vec to avoid blowing the stack on the multi-MB array literal.
        let mut w1t_vec: Vec<[f32; L1_SIZE]> = vec![[0.0f32; L1_SIZE]; FEATURE_COUNT];
        for i in 0..L1_SIZE {
            for j in 0..FEATURE_COUNT {
                w1t_vec[j][i] = floats[i * FEATURE_COUNT + j];

            }

        }
        let w1t: Box<[[f32; L1_SIZE]; FEATURE_COUNT]> = w1t_vec
            .into_boxed_slice()
            .try_into()
            .map_err(|_| "internal: w1t length mismatch".to_string())?;

        let b1_off = L1_SIZE * FEATURE_COUNT;
        let w2_off = b1_off + L1_SIZE;
        let b2_off = w2_off + L1_SIZE;

        // fc1.bias: [L1_SIZE]
        let mut b1 = [0f32; L1_SIZE];
        for i in 0..L1_SIZE {
            b1[i] = floats[b1_off + i];

        }

        // fc2.weight: [1, L1_SIZE]
        let mut w2 = [0f32; L1_SIZE];
        for i in 0..L1_SIZE {
            w2[i] = floats[w2_off + i];

        }

        // fc2.bias: scalar
        let b2 = floats[b2_off];

        Ok(Self { w1t, b1, w2, b2 })

    }

    /// Encode + forward pass from `player`'s perspective (P2 sees board mirrored 180°).
    pub fn eval(&self, board: &BoardState, player: Player) -> f64 {
        let mut hidden = self.b1;

        let mirror = matches!(player, Player::Two);
        encode_position(board, mirror, |feat| {
            let col = &self.w1t[feat as usize];
            for i in 0..L1_SIZE {
                hidden[i] += col[i];

            }

        });

        // ReLU
        for i in 0..L1_SIZE {
            hidden[i] = hidden[i].max(0.0);

        }

        // Layer 2: out = tanh(w2 · hidden + b2)
        let mut out = self.b2;
        for i in 0..L1_SIZE {
            out += self.w2[i] * hidden[i];

        }

        (out.tanh() as f64) * NETWORK_SCALE

    }

}

/// Unordered-pair index: PAIR_IDX[a][b] = PAIR_IDX[b][a] ∈ [0, 630) for a ≠ b.
/// Diagonal is unused.
const PAIR_IDX: [[u16; 36]; 36] = {
    let mut m = [[0u16; 36]; 36];
    let mut k: u16 = 0;
    let mut a = 0usize;
    while a < 36 {
        let mut b = a + 1;
        while b < 36 {
            m[a][b] = k;
            m[b][a] = k;
            k += 1;
            b += 1;
        }
        a += 1;
    }
    m
};

/// Feature index for a singleton piece of type `t` at (feature-space) square `sq`.
/// `t` must be 1, 2, or 3.
#[inline(always)]
#[allow(dead_code)]
fn feature_singleton(sq: u32, t: u32) -> u32 {
    sq * 3 + (t - 1)
}

/// Feature index for the pair {(sq_a, t_a), (sq_b, t_b)}, sq_a ≠ sq_b.
/// Types are 1/2/3. Same-type is unordered; cross-type is ordered (lower type first).
#[inline]
#[allow(dead_code)]
fn feature_pair(sq_a: u32, t_a: u32, sq_b: u32, t_b: u32) -> u32 {
    if t_a == t_b {
        let off = match t_a {
            1 => PAIR_11_OFFSET,
            2 => PAIR_22_OFFSET,
            3 => PAIR_33_OFFSET,
            _ => unreachable!(),
        };
        off + PAIR_IDX[sq_a as usize][sq_b as usize] as u32
    } else if t_a < t_b {
        let off = match (t_a, t_b) {
            (1, 2) => PAIR_12_OFFSET,
            (1, 3) => PAIR_13_OFFSET,
            (2, 3) => PAIR_23_OFFSET,
            _ => unreachable!(),
        };
        off + sq_a * 36 + sq_b
    } else {
        feature_pair(sq_b, t_b, sq_a, t_a)
    }
}

/// Emit the active feature indices for `board` via `out`. When `mirror` is true,
/// every square is transformed by `35 - sq` (P2 view: 180° rotation).
///
/// Order: singletons (in raw-square iteration order), then same-type pairs grouped
/// by piece type, then cross-type pairs. Matches the Python encoder ordering.
fn encode_position<F: FnMut(u32)>(board: &BoardState, mirror: bool, mut out: F) {
    // Feature-space squares grouped by piece type. Index 0 = type 1, etc.
    let mut grouped: [[u8; 12]; 3] = [[0; 12]; 3];
    let mut counts = [0usize; 3];

    for raw_sq in 0..36u8 {
        if board.piece_bb.0 & (1u64 << raw_sq) == 0 { continue; }
        let t = match board.data[raw_sq as usize] {
            Piece::One => 1u32,
            Piece::Two => 2u32,
            Piece::Three => 3u32,
            Piece::None => continue,
        };
        let feat_sq = if mirror { 35 - raw_sq } else { raw_sq } as u32;

        out(feat_sq * 3 + (t - 1));

        let g = (t - 1) as usize;
        grouped[g][counts[g]] = feat_sq as u8;
        counts[g] += 1;

    }

    // Same-type pairs (unordered)
    let same_offsets = [PAIR_11_OFFSET, PAIR_22_OFFSET, PAIR_33_OFFSET];
    for g in 0..3 {
        let n = counts[g];
        let off = same_offsets[g];
        for i in 0..n {
            for j in (i + 1)..n {
                out(off + PAIR_IDX[grouped[g][i] as usize][grouped[g][j] as usize] as u32);

            }

        }

    }

    // Cross-type pairs (ordered: lower piece-type's square first)
    let cross: [(usize, usize, u32); 3] = [
        (0, 1, PAIR_12_OFFSET),
        (0, 2, PAIR_13_OFFSET),
        (1, 2, PAIR_23_OFFSET),
    ];
    for (lo, hi, off) in cross {
        let nlo = counts[lo];
        let nhi = counts[hi];
        for i in 0..nlo {
            for j in 0..nhi {
                out(off + grouped[lo][i] as u32 * 36 + grouped[hi][j] as u32);

            }

        }

    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encoder self-consistency: emitted indices are unique, < FEATURE_COUNT, and
    /// the total count matches the piece-group combinatorics.
    #[test]
    fn encoder_emits_unique_valid_indices() {
        for board_arr in [STARTING_BOARD, BENCH_BOARD, TEST_BOARD] {
            let board = BoardState::from(board_arr);
            for mirror in [false, true] {
                let mut indices: Vec<u32> = Vec::new();
                encode_position(&board, mirror, |f| indices.push(f));

                let mut n = [0usize; 3];
                for sq in 0..36 {
                    if board.piece_bb.0 & (1u64 << sq) == 0 { continue; }
                    match board.data[sq] {
                        Piece::One => n[0] += 1,
                        Piece::Two => n[1] += 1,
                        Piece::Three => n[2] += 1,
                        _ => {}
                    }
                }
                let singles = n[0] + n[1] + n[2];
                let same = n[0]*n[0].saturating_sub(1)/2
                         + n[1]*n[1].saturating_sub(1)/2
                         + n[2]*n[2].saturating_sub(1)/2;
                let cross = n[0]*n[1] + n[0]*n[2] + n[1]*n[2];
                assert_eq!(
                    indices.len(), singles + same + cross,
                    "feature count mismatch (mirror={})", mirror
                );

                let mut sorted = indices.clone();
                sorted.sort();
                sorted.dedup();
                assert_eq!(sorted.len(), indices.len(), "duplicate features emitted");

                assert!(
                    indices.iter().all(|&f| (f as usize) < FEATURE_COUNT),
                    "feature index out of range"
                );
            }
        }
    }

    /// `feature_pair` must canonicalize: same-type order-invariant, cross-type
    /// reduces to (lower_type, higher_type) with lower-type's square first.
    #[test]
    fn feature_pair_canonicalizes() {
        // Same-type: pair(a, t, b, t) == pair(b, t, a, t)
        assert_eq!(feature_pair(3, 1, 17, 1), feature_pair(17, 1, 3, 1));
        assert_eq!(feature_pair(0, 2, 35, 2), feature_pair(35, 2, 0, 2));

        // Cross-type: pair(a, 1, b, 2) == pair(b, 2, a, 1), and lands in PAIR_12 range.
        let idx = feature_pair(5, 1, 10, 2);
        assert_eq!(idx, feature_pair(10, 2, 5, 1));
        assert!(idx >= PAIR_12_OFFSET && idx < PAIR_13_OFFSET);
        assert_eq!(idx, PAIR_12_OFFSET + 5 * 36 + 10);
    }
    
}
