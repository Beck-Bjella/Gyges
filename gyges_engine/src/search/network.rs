//! NN evaluation. 144 → L1_SIZE (ReLU) → 1 (tanh).
//! Board encoded from the side-to-move's perspective (home rank = rank 0).
//! Output in [-NETWORK_SCALE, +NETWORK_SCALE], positive = side-to-move winning.

use gyges::{board::*, core::*, moves::*};
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

/// Build both perspectives' accumulators from scratch using the active network.
/// Panics if no network is loaded.
pub fn accumulator_from_scratch_active(board: &BoardState) -> Accumulator {
    let net = unsafe { NETWORK.as_ref() }.expect("Network not loaded");
    net.accumulator_from_scratch(board)
}

/// Patch `next` from `prev` for the move `mv` applied to `board_before`.
/// Panics if no network is loaded.
pub fn patch_make_active(prev: &Accumulator, next: &mut Accumulator, board_before: &BoardState, mv: &Move) {
    let net = unsafe { NETWORK.as_ref() }.expect("Network not loaded");
    net.patch_make(prev, next, board_before, mv);
}

/// Layer-2 + tanh applied to a pre-built accumulator, picking the perspective for `player`.
/// Panics if no network is loaded.
pub fn eval_from_accumulator_active(acc: &Accumulator, player: Player) -> f64 {
    let net = unsafe { NETWORK.as_ref() }.expect("Network not loaded");
    net.eval_from_accumulator(acc, player)
}

/// Scales tanh output to match the hand-crafted eval's magnitude.
pub const NETWORK_SCALE: f64 = 10000.0;

/// Layer 1 width.
pub const L1_SIZE: usize = 256;

/// 144 → L1_SIZE (ReLU) → 1 (tanh).
/// Input: one-hot piece per square (4 features × 36 squares = 144).
/// Weights file (float32 LE): w1 [L1 × 144], b1 [L1], w2 [1 × L1], b2 [1].
pub struct GygesNet {
    /// Layer-1 weights, transposed: w1t[feature][neuron]. Each column is contiguous
    /// for vectorized column-adds in the sparse forward pass.
    w1t: Box<[[f32; L1_SIZE]; 144]>,
    b1: [f32; L1_SIZE],
    w2: [f32; L1_SIZE],
    b2: f32,

}

impl GygesNet {
    /// Load weights from the binary file produced by the Python training script.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("Cannot read weight file '{}': {}", path, e))?;

        let expected_floats = L1_SIZE * 144 + L1_SIZE + L1_SIZE + 1;
        let expected_bytes = expected_floats * 4;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Weight file is {} bytes, expected {} ({} × float32). \
                 Check that the Python export matches the 144→{}→1 architecture.",
                bytes.len(),
                expected_bytes,
                expected_floats,
                L1_SIZE
            ));

        }

        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // net.0.weight: [L1_SIZE, 144] row-major in the file — transpose to [144, L1_SIZE] for
        // cache-friendly sparse access: w1t[feature][neuron]
        let mut w1t = Box::new([[0f32; L1_SIZE]; 144]);
        for i in 0..L1_SIZE {
            for j in 0..144 {
                w1t[j][i] = floats[i * 144 + j];

            }

        }

        let w1_len = L1_SIZE * 144;
        let b1_off = w1_len;
        let w2_off = b1_off + L1_SIZE;
        let b2_off = w2_off + L1_SIZE;

        // net.0.bias: [L1_SIZE]
        let mut b1 = [0f32; L1_SIZE];
        for i in 0..L1_SIZE {
            b1[i] = floats[b1_off + i];

        }

        // net.3.weight: [1, L1_SIZE]
        let mut w2 = [0f32; L1_SIZE];
        for i in 0..L1_SIZE {
            w2[i] = floats[w2_off + i];

        }

        // net.3.bias: scalar
        let b2 = floats[b2_off];

        Ok(Self { w1t, b1, w2, b2 })

    }

    /// Encode + forward pass from `player`'s perspective (P2 rotated 180°).
    /// Sparse: only 36 of 144 inputs are non-zero, so we add weight columns
    /// instead of running a full matmul.
    pub fn eval(&self, board: &BoardState, player: Player) -> f64 {
        // Start with bias — equivalent to the constant term in each neuron
        let mut hidden = self.b1;

        for sq in 0..36usize {
            // For P2, rotate the board 180° so active player's home rank is always rank 0
            let board_sq = match player {
                Player::One => sq,
                Player::Two => 35 - sq,
            };

            let bit = 1u64 << board_sq;

            let piece_idx = if board.piece_bb.0 & bit != 0 {
                board.data[board_sq] as usize + 1

            } else {
                0

            };

            // One-hot piece feature: add the weight column for this feature
            // (equivalent to multiplying by 1.0 — no multiply needed)
            let piece_col = &self.w1t[sq * 4 + piece_idx];
            for i in 0..L1_SIZE {
                hidden[i] += piece_col[i];

            }

        }

        // ReLU activation
        for i in 0..L1_SIZE {
            hidden[i] = hidden[i].max(0.0);

        }

        // Layer 2: out = tanh(w2 · hidden + b2) — dense, since hidden is fully populated
        let mut out = self.b2;
        for i in 0..L1_SIZE {
            out += self.w2[i] * hidden[i];

        }

        (out.tanh() as f64) * NETWORK_SCALE

    }

    /// Rebuild both perspective accumulators from scratch. Iteration order matches
    /// `eval` so the seed is bit-identical to the dense forward pass.
    pub fn accumulator_from_scratch(&self, board: &BoardState) -> Accumulator {
        let mut p1 = self.b1;
        let mut p2 = self.b1;

        for sq in 0..36usize {
            // P1 view: display square == raw square
            let p1_idx = if board.piece_bb.0 & (1u64 << sq) != 0 {
                board.data[sq] as usize + 1

            } else {
                0

            };
            let col_p1 = &self.w1t[sq * 4 + p1_idx];
            for i in 0..L1_SIZE {
                p1[i] += col_p1[i];

            }

            // P2 view: read piece at the mirrored raw square, deposit into feature `sq`
            let p2_raw = MIRROR[sq] as usize;
            let p2_idx = if board.piece_bb.0 & (1u64 << p2_raw) != 0 {
                board.data[p2_raw] as usize + 1

            } else {
                0

            };
            let col_p2 = &self.w1t[sq * 4 + p2_idx];
            for i in 0..L1_SIZE {
                p2[i] += col_p2[i];

            }

        }

        Accumulator { p1, p2 }

    }

    /// Derive `next` from `prev` by patching only the features that change under `mv`.
    /// Patches both perspectives in lockstep (same weights, mirrored feature index).
    /// `board_before` is the state BEFORE the move is applied.
    pub fn patch_make(
        &self,
        prev: &Accumulator,
        next: &mut Accumulator,
        board_before: &BoardState,
        mv: &Move,
    ) {
        *next = *prev;

        let s1 = mv.data[0].1.0 as usize;
        let s2 = mv.data[1].1.0 as usize;
        let s3 = mv.data[2].1.0 as usize;

        // Touched squares: 2 for Bounce, 3 for Drop. Drop's sq3 may equal sq1
        // (displaced piece bouncing back to the start square), so dedup is required.
        let sqs = [s1, s2, s3];
        let len = if mv.flag == MoveType::Drop { 3 } else { 2 };

        for i in 0..len {
            let sq = sqs[i];

            // Skip duplicate squares already patched on a prior step
            let mut dup = false;
            for j in 0..i {
                if sqs[j] == sq { dup = true; break; }

            }
            if dup { continue; }

            // BEFORE state: empty if bb bit is clear, else the piece in `data`
            let before = if board_before.piece_bb.0 & (1u64 << sq) != 0 {
                board_before.data[sq]

            } else {
                Piece::None

            };

            // AFTER state: simulate placements in order (later writes win for sq1==sq3)
            let mut after = before;
            if s1 == sq { after = mv.data[0].0; }
            if s2 == sq { after = mv.data[1].0; }
            if mv.flag == MoveType::Drop && s3 == sq { after = mv.data[2].0; }

            if before == after { continue; }

            let mq = MIRROR[sq] as usize;
            let f_old_p1 = sq * 4 + piece_idx(before);
            let f_new_p1 = sq * 4 + piece_idx(after);
            let f_old_p2 = mq * 4 + piece_idx(before);
            let f_new_p2 = mq * 4 + piece_idx(after);

            let sub_p1 = &self.w1t[f_old_p1];
            let add_p1 = &self.w1t[f_new_p1];
            let sub_p2 = &self.w1t[f_old_p2];
            let add_p2 = &self.w1t[f_new_p2];

            for j in 0..L1_SIZE {
                next.p1[j] += add_p1[j] - sub_p1[j];
                next.p2[j] += add_p2[j] - sub_p2[j];

            }

        }

    }

    /// Layer 2 + tanh applied to a pre-built accumulator. Picks the perspective for `player`.
    pub fn eval_from_accumulator(&self, acc: &Accumulator, player: Player) -> f64 {
        let pre = match player {
            Player::One => &acc.p1,
            Player::Two => &acc.p2,

        };

        let mut out = self.b2;
        for i in 0..L1_SIZE {
            out += self.w2[i] * pre[i].max(0.0);

        }

        (out.tanh() as f64) * NETWORK_SCALE

    }

}

/// Pre-ReLU layer-1 output, one per perspective.
/// `p1` indexes by raw square, `p2` by `MIRROR[q]`. Eval reads whichever matches STM.
#[derive(Clone, Copy)]
pub struct Accumulator {
    pub p1: [f32; L1_SIZE],
    pub p2: [f32; L1_SIZE],

}

impl Accumulator {
    /// All-zeros accumulator — used only to size the search stack. Real values
    /// come from `accumulator_from_scratch` (root) and `patch_make` (children).
    pub fn zero() -> Self {
        Accumulator { p1: [0.0; L1_SIZE], p2: [0.0; L1_SIZE] }

    }

}

/// 180° rotation: P1 square `q` maps to P2 square `35 - q`.
const MIRROR: [u8; 36] = {
    let mut m = [0u8; 36];
    let mut q = 0;
    while q < 36 {
        m[q] = (35 - q) as u8;
        q += 1;

    }
    m

};

/// Maps a `Piece` to its one-hot feature index within a square's 4-element block.
/// Matches the encoding in `GygesNet::eval`: empty=0, One=1, Two=2, Three=3.
#[inline(always)]
fn piece_idx(p: Piece) -> usize {
    match p {
        Piece::None => 0,
        Piece::One => 1,
        Piece::Two => 2,
        Piece::Three => 3,

    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use gyges::moves::movegen::*;

    /// Build a deterministic synthetic GygesNet so tests don't depend on a weights file
    /// being present. Uses a simple LCG to fill all weights with pseudo-random values
    /// in roughly [-1, 1].
    fn synth_net() -> GygesNet {
        let mut state: u32 = 0xDEAD_BEEF;
        let mut next = || -> f32 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut w1t = Box::new([[0f32; L1_SIZE]; 144]);
        for j in 0..144 {
            for i in 0..L1_SIZE {
                w1t[j][i] = next();
            }
        }
        let mut b1 = [0f32; L1_SIZE];
        for v in b1.iter_mut() { *v = next(); }
        let mut w2 = [0f32; L1_SIZE];
        for v in w2.iter_mut() { *v = next(); }
        let b2 = next();

        GygesNet { w1t, b1, w2, b2 }
    }

    fn accs_close(a: &Accumulator, b: &Accumulator, tol: f32) -> bool {
        (0..L1_SIZE).all(|i| (a.p1[i] - b.p1[i]).abs() <= tol && (a.p2[i] - b.p2[i]).abs() <= tol)
    }

    /// `eval_from_accumulator` is just layer 2 + tanh on a pre-built layer-1 output.
    /// It must match the dense `eval` (which fuses layer 1 from features) to float precision.
    #[test]
    fn accumulator_eval_matches_dense_eval() {
        let net = synth_net();

        for board_arr in [STARTING_BOARD, BENCH_BOARD, TEST_BOARD] {
            let board = BoardState::from(board_arr);
            let acc = net.accumulator_from_scratch(&board);

            for player in [Player::One, Player::Two] {
                let dense = net.eval(&board, player);
                let from_acc = net.eval_from_accumulator(&acc, player);
                assert!(
                    (dense - from_acc).abs() < 1e-3,
                    "{:?}: dense={} from_acc={}", player, dense, from_acc
                );
            }
        }
    }

    /// Walk a real game: at every step, the patched accumulator must agree with
    /// a from-scratch rebuild of the post-move board, element-wise. This is the
    /// oracle test for `patch_make` correctness.
    #[test]
    fn patch_matches_full_recomputation_over_random_walk() {
        let net = synth_net();

        // Walk from several starting positions to exercise different piece layouts.
        for board_arr in [STARTING_BOARD, BENCH_BOARD, TEST_BOARD] {
            let mut board = BoardState::from(board_arr);
            let mut acc = net.accumulator_from_scratch(&board);
            let mut mg = MoveGen::default();
            let mut player = Player::One;

            for _step in 0..30 {
                let mut data: GenResult = unsafe { mg.gen::<GenMoves, NoQuit>(&mut board, player) };
                let moves = data.move_list.moves(&board);
                if moves.is_empty() { break; }

                // Use the first non-winning move so the walk stays inside the game.
                let Some(&mv) = moves.iter().find(|m| !m.is_win()) else { break };

                let mut patched = Accumulator::zero();
                net.patch_make(&acc, &mut patched, &board, &mv);

                board.make_move(&mv);

                let expected = net.accumulator_from_scratch(&board);
                assert!(
                    accs_close(&expected, &patched, 1e-4),
                    "patch_make diverged from from_scratch after {} (flag {:?})",
                    mv, mv.flag
                );

                acc = patched;
                player = player.other();
            }
        }
    }

    /// Exercises the rare drop case where the displaced piece bounces back to the
    /// starting square (`sq3 == sq1`). The dedup logic in `patch_make` must handle
    /// this without double-patching the square.
    #[test]
    fn patch_handles_drop_with_sq3_equals_sq1() {
        let net = synth_net();
        let board = BoardState::from(STARTING_BOARD);

        // Construct a synthetic Drop move where sq3 == sq1: piece moves from a3 to a1
        // (picking up whatever's at a1) and the displaced piece bounces back to a3.
        // We don't care if this is a *legal* Gyges move — only that the patch
        // bookkeeping handles the duplicate-square case.
        let s1 = SQ(12); // a3 — has no piece in STARTING_BOARD, so we'll set one up
        let s2 = SQ(0);  // a1 — has a 3
        let s3 = SQ(12); // back to a3

        let mut board = board;
        // Put a piece at sq1 so the move makes sense
        board.data[s1.0 as usize] = Piece::One;
        board.piece_bb.set_bit(s1.0 as usize);
        let acc = net.accumulator_from_scratch(&board);

        let mv = Move {
            data: [
                (Piece::None, s1),         // sq1 emptied (then re-filled by step 3)
                (Piece::One,  s2),         // moving piece lands at sq2 (replaces the 3)
                (Piece::Three, s3),        // displaced 3 bounces back to sq1
            ],
            flag: MoveType::Drop,
        };

        let mut patched = Accumulator::zero();
        net.patch_make(&acc, &mut patched, &board, &mv);

        let mut after = board;
        after.make_move(&mv);
        let expected = net.accumulator_from_scratch(&after);

        assert!(
            accs_close(&expected, &patched, 1e-4),
            "patch_make mishandled sq3==sq1 drop"
        );
    }
}
