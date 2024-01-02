//! Contains the bitboard struct and impls.

use std::{ops::{Not, BitOr, BitOrAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign}, fmt::Display};

use crate::core::bit_twiddles::*;
use crate::core::masks::*;
use crate::core::*;

/// A bitboard is a 64 bit integer that represents if a square is occupied or not.  
/// 
/// The mapping of bits of the bitboard to their corosponding [squares] is the same as the mapping documented on the [boardstate] struct.
/// 
/// [squares]: 
/// [boardstate]: 
/// 
#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct BitBoard(pub u64);

impl BitBoard {
    pub const EMPTY: BitBoard = BitBoard(EMPTY);
    pub const FULL: BitBoard = BitBoard(FULL);

    /// Returns an vector of the indexes of the set bits in the bitboard.
    pub fn get_data(&mut self) -> Vec<usize> {
        let mut indexs = vec![];

        while self.0 != 0 {
            indexs.push(self.pop_lsb());

        }

        indexs

    }

    /// Starts from the start of the bitboard and returns the index of the first set bit.
    pub fn bit_scan_forward(&self) -> usize {
        bit_scan_forward(self.0) as usize

    }

    /// Starts from the end of the bitboard and returns the index of the last set bit.
    pub fn bit_scan_reverse(&self) -> usize {
        bit_scan_reverse(self.0) as usize

    }  

    /// Returns the index of the first set bit and clears it.
    pub fn pop_lsb(&mut self) -> usize {
        let bit = self.bit_scan_forward();
        self.0 &= !(1 << bit);
        bit
        
    }

    /// Returns the index of the last set bit and clears it.
    pub fn pop_msb(&mut self) -> usize {
        let bit = self.bit_scan_reverse();
        self.0 &= !(1 << bit);
        bit
        
    }

    /// Returns the number of set bits in the bitboard.
    pub fn pop_count(&self) -> usize {
        self.0.count_ones() as usize

    }

    /// Sets a specific bit in the bitboard.
    pub fn set_bit(&mut self, bit: usize) {
        let mask = 1 << bit;

        self.0 |= mask;

    }

    /// Clears a specific bit in the bitboard.
    pub fn clear_bit(&mut self, bit: usize) {
        let mask = 1 << bit;

        self.0 &= !mask;

    }

    /// Toggles a specific bit in the bitboard.
    pub fn toggle_bit(&mut self, bit: usize) {
        let mask = 1 << bit;

        self.0 ^= mask;

    }

    /// Checks if there are no set bits in the bitboard.
    pub fn is_empty(self) -> bool {
        self.0 == 0
        
    }
    
    /// Checks if there are set bits in the bitboard.
    pub fn is_not_empty(self) -> bool {
        self.0 != 0
        
    }

}

impl Display for BitBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.clone().get_data();
        
        if data.contains(&(SQ::P2_GOAL.0 as usize)) {
            writeln!(f, "          1")?;

        } else {
            writeln!(f, "          0")?;
            
        }

        for y in (0..6).rev() {
            for x in 0..6 {
                if data.contains(&((y * 6) + x)) {
                    write!(f, "  1")?;

                } else {
                    write!(f, "  0")?;

                }

            }

            writeln!(f)?;

        }

        if data.contains(&(SQ::P1_GOAL.0 as usize)) {
            writeln!(f, "          1")?;

        } else {
            writeln!(f, "          0")?;
            
        }
    
       Ok(())

    }

}

// Impl bit opperators
impl Not for BitBoard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }

}

impl BitOr<BitBoard> for BitBoard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }

}

impl BitOr<u64> for BitBoard {
    type Output = Self;

    fn bitor(self, rhs: u64) -> Self::Output {
        Self(self.0 | rhs)
    }

}

impl BitOrAssign<BitBoard> for BitBoard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }

}

impl BitOrAssign<u64> for BitBoard {
    fn bitor_assign(&mut self, rhs: u64) {
        self.0 |= rhs;
    }

}

impl BitAnd<BitBoard> for BitBoard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }

}

impl BitAnd<u64> for BitBoard {
    type Output = Self;

    fn bitand(self, rhs: u64) -> Self::Output {
        Self(self.0 & rhs)
    }

}

impl BitAndAssign<BitBoard> for BitBoard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }

}

impl BitAndAssign<u64> for BitBoard {
    fn bitand_assign(&mut self, rhs: u64) {
        self.0 &= rhs;
    }

}

impl BitXor<BitBoard> for BitBoard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }

}

impl BitXor<u64> for BitBoard {
    type Output = Self;

    fn bitxor(self, rhs: u64) -> Self::Output {
        Self(self.0 ^ rhs)
    }

}

impl BitXorAssign<BitBoard> for BitBoard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }

}

impl BitXorAssign<u64> for BitBoard {
    fn bitxor_assign(&mut self, rhs: u64) {
        self.0 ^= rhs;
    }

}

impl Shl<usize> for BitBoard {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.0 << rhs)
    }

}

impl ShlAssign<usize> for BitBoard {
    fn shl_assign(&mut self, rhs: usize) {
        self.0 <<= rhs;
    }

}

impl Shr<usize> for BitBoard {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        Self(self.0 >> rhs)
    }

}

impl ShrAssign<usize> for BitBoard {
    fn shr_assign(&mut self, rhs: usize) {
        self.0 >>= rhs;
    }

}
