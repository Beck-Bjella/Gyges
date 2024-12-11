use std::ops::{Not, BitOr, BitOrAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign};
use std::fmt::Display;

use gyges::board::BoardState;

/// Bits 0..37 are the positions of the pieces on the board.
/// Bits 38..62 are the types of pieces in order.
/// Bit 63 is the player to move.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BitState(pub u64);

impl BitState {
    /// Creates a new BitState from an existing BoardState
    pub fn from(board: &BoardState) -> Self {
        let mut bb: u64 = 0u64;
        let mut piece_idx = 0;
        for i in 0..38 {
            let piece = board.data[i] as u8;
            if piece != 3 {
                // Set the piece bit
                bb |= 1 << i;
                // Set the piece type
                bb |= ((piece + 1) as u64) << (38 + (piece_idx * 2));
                piece_idx += 1;
            }
        }
        // Set the player to move
        bb |= (board.player as u64) << 63;

        BitState(bb)

    }

    pub fn make_bounce_mv(&self, start_pos: u64, end_pos: u64) -> BitState {
        let mut new_state = *self;

        // Update the piece bitboard
        new_state.0 ^= (1 << start_pos) | (1 << end_pos);

        // Starting piece index and type
        let starting_idx = self.piece_idx(start_pos as usize);
        let starting_piece = self.piece_type(starting_idx);

        // Remove the piece type at starting index
        new_state.remove_type(starting_idx);

        // Since we've removed a piece type, adjust the ending index
        let ending_idx = if start_pos < end_pos {
            self.piece_idx(end_pos as usize) - 1
        } else {
            self.piece_idx(end_pos as usize)
        };

        // Add the starting piece type at the new index
        new_state.add_type(ending_idx, starting_piece);

        new_state

    }

    pub fn make_drop_mv(&self, start_pos: u64, pickup_pos: u64, drop_pos: u64) -> BitState {
        let mut new_state = *self;

        // Starting piece index and type
        let starting_idx = self.piece_idx(start_pos as usize);
        let starting_piece = self.piece_type(starting_idx);

        // Remove the starting piece
        new_state.remove_type(starting_idx);
        new_state.0 ^= 1 << start_pos;

        // Pickup piece index and type
        let pickup_idx = new_state.piece_idx(pickup_pos as usize);
        let pickup_piece = new_state.piece_type(pickup_idx);

        // Set the pickup piece's type to the starting piece's type
        new_state.set_type_data(pickup_idx, starting_piece);

        // Drop piece index
        let drop_idx = new_state.piece_idx(drop_pos as usize);

        // Add the pickup piece type at the drop index
        new_state.add_type(drop_idx, pickup_piece);

        // Update the piece bitboard
        new_state.0 ^= 1 << drop_pos;

        new_state
        
    }

    // ======================== MOVE MAKING HELPERS ========================

    /// Removes an existing piece type
    pub fn remove_type(&mut self, piece_idx: usize) {
        let type_pos = 38 + (piece_idx * 2);

        // Clear the two bits at type_pos
        let clear_mask = !(0b11u64 << type_pos);
        self.0 &= clear_mask;

        // Shift higher bits down by 2
        let higher_bits_mask = !0u64 << (type_pos + 2);
        let higher_bits = (self.0 & higher_bits_mask) >> 2;
        self.0 = (self.0 & !higher_bits_mask) | higher_bits;

    }

    /// Adds a new piece type
    pub fn add_type(&mut self, piece_idx: usize, piece_type: u8) {
        let type_pos = 38 + (piece_idx * 2);

        // Shift higher bits up by 2 to make space
        let higher_bits_mask = !0u64 << type_pos;
        let higher_bits = (self.0 & higher_bits_mask) << 2;
        self.0 = (self.0 & !higher_bits_mask) | higher_bits;

        // Set the new piece type
        self.0 |= (piece_type as u64) << type_pos;
        
    }

    /// Changes the type data at one of the type data slots
    pub fn set_type_data(&mut self, piece_idx: usize, piece_type: u8) {
        let type_pos = 38 + (piece_idx * 2);

        // Mask to clear the two bits at type_pos
        let clear_mask = !(0b11u64 << type_pos);

        // Clear and set the piece type bits at piece_idx
        self.0 = (self.0 & clear_mask) | ((piece_type as u64) << type_pos);

    }
    
    // ======================== GETTERS ========================

    /// Gets the piece at a square
    /// 0 = None
    /// 1 = One
    /// 2 = Two
    /// 3 = Three
    pub fn piece_at(&self, pos: usize) -> u8 {
        if (self.0 & (1 << pos)) == 0 {
            return 0;

        }

        let idx = self.piece_idx(pos);
        self.piece_type(idx)

    }

    /// Gets the piece bitboard
    pub fn piece_bb(&self) -> u64 {
        self.0 & ((1u64 << 38) - 1)

    }

    /// Gets the index of the piece at a position
    pub fn piece_idx(&self, pos: usize) -> usize {
        let mask = (1u64 << pos) - 1;
        let bits_before = self.piece_bb() & mask;
        bits_before.count_ones() as usize

    }

    /// Gets the piece type at a given index
    pub fn piece_type(&self, piece_idx: usize) -> u8 {
        ((self.0 >> (38 + (piece_idx * 2))) & 0b11) as u8

    }

    /// Gets the player to move
    pub fn player(&self) -> u8 {
        (self.0 >> 63) as u8

    }

}

impl Display for BitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.piece_at(37) == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.piece_at(37))?;

        }
        writeln!(f, " ")?;
        writeln!(f, " ")?;

        for y in (0..6).rev() {
            for x in 0..6 {
                if self.piece_at(y * 6 + x) == 0 {
                    write!(f, "    .")?;
                } else {
                    write!(f, "    {}", self.piece_at(y * 6 + x))?;

                }
               
            }
            writeln!(f, " ")?;
            writeln!(f, " ")?;

        }

        writeln!(f, " ")?;
        if self.piece_at(36) == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.piece_at(36))?;

        }

        Result::Ok(())

    }

}

// Impl bit opperators
impl Not for BitState {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }

}

impl BitOr<BitState> for BitState {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }

}

impl BitOr<u64> for BitState {
    type Output = Self;

    fn bitor(self, rhs: u64) -> Self::Output {
        Self(self.0 | rhs)
    }

}

impl BitOrAssign<BitState> for BitState {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }

}

impl BitOrAssign<u64> for BitState {
    fn bitor_assign(&mut self, rhs: u64) {
        self.0 |= rhs;
    }

}

impl BitAnd<BitState> for BitState {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }

}

impl BitAnd<u64> for BitState {
    type Output = Self;

    fn bitand(self, rhs: u64) -> Self::Output {
        Self(self.0 & rhs)
    }

}

impl BitAndAssign<BitState> for BitState {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }

}

impl BitAndAssign<u64> for BitState {
    fn bitand_assign(&mut self, rhs: u64) {
        self.0 &= rhs;
    }

}

impl BitXor<BitState> for BitState {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }

}

impl BitXor<u64> for BitState {
    type Output = Self;

    fn bitxor(self, rhs: u64) -> Self::Output {
        Self(self.0 ^ rhs)
    }

}

impl BitXorAssign<BitState> for BitState {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }

}

impl BitXorAssign<u64> for BitState {
    fn bitxor_assign(&mut self, rhs: u64) {
        self.0 ^= rhs;
    }

}

impl Shl<usize> for BitState {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.0 << rhs)
    }

}

impl ShlAssign<usize> for BitState {
    fn shl_assign(&mut self, rhs: usize) {
        self.0 <<= rhs;
    }

}

impl Shr<usize> for BitState {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        Self(self.0 >> rhs)
    }

}

impl ShrAssign<usize> for BitState {
    fn shr_assign(&mut self, rhs: usize) {
        self.0 >>= rhs;
    }

}
