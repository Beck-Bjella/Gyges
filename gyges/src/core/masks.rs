//! Bitboard masks

/// Rank 1 mask.
pub const RANK_1: u64 = 0b111111000000000000000000000000000000;
/// Rank 2 mask.
pub const RANK_2: u64 = 0b000000111111000000000000000000000000;
/// Rank 3 mask.
pub const RANK_3: u64 = 0b000000000000111111000000000000000000;
/// Rank 4 mask.
pub const RANK_4: u64 = 0b000000000000000000111111000000000000;
/// Rank 5 mask.
pub const RANK_5: u64 = 0b000000000000000000000000111111000000;
/// Rank 6 mask.
pub const RANK_6: u64 = 0b000000000000000000000000000000111111;

/// File A mask.
pub const FILE_A: u64 = 0b100000100000100000100000100000100000;
/// File B mask.
pub const FILE_B: u64 = 0b010000010000010000010000010000010000;
/// File C mask.
pub const FILE_C: u64 = 0b001000001000001000001000001000001000;
/// File D mask.
pub const FILE_D: u64 = 0b000100000100000100000100000100000100;
/// File E mask.
pub const FILE_E: u64 = 0b000010000010000010000010000010000010;
/// File F mask.
pub const FILE_F: u64 = 0b000001000001000001000001000001000001;

/// Array of all ranks.
pub const RANKS: [u64; 6] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6];

/// Array of all files.
pub const FILES: [u64; 6] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F];

/// Full board mask.
pub const FULL: u64 = 0b111111111111111111111111111111111111;
/// Empty board mask.
pub const EMPTY: u64 = 0;

/// Each element represents a different active line for P1 and is a mask of the squares that are in their back zone.
pub const P1_BACK_ZONE: [u64; 6] = [
    0b000000000000000000000000000000000000,
    0b000000000000000000000000000000111111,
    0b000000000000000000000000111111111111,
    0b000000000000000000111111111111111111,
    0b000000000000111111111111111111111111,
    0b000000111111111111111111111111111111
];
/// Each element represents a different active line for P2 and is a mask of the squares that are in their back zone.
pub const P2_BACK_ZONE: [u64; 6] = [
    0b111111111111111111111111111111000000,
    0b111111111111111111111111000000000000,
    0b111111111111111111000000000000000000,
    0b111111111111000000000000000000000000,
    0b111111000000000000000000000000000000,
    0b000000000000000000000000000000000000
];

// Array of each player's back zones.
pub const BACK_ZONES: [[u64; 6]; 2] = [P1_BACK_ZONE, P2_BACK_ZONE];

/// The mask of P1's starting zone.
pub const P1_STARTING_ZONE: u64 = 0b111111111111111111000000000000000000;
/// The mask of P2's starting zone.
pub const P2_STARTING_ZONE: u64 = 0b000000000000000000111111111111111111;

/// Array of each player's starting zones.
pub const STARTING_ZONES: [u64; 2] = [P1_STARTING_ZONE, P2_STARTING_ZONE];