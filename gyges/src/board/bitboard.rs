use std::ops::{Not, BitOr, BitOrAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign};

use crate::core::bit_twiddles::*;

#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct BitBoard(pub u64);

impl BitBoard {
    pub fn get_data(&mut self) -> Vec<usize> {
        let mut indexs = vec![];

        while self.0 != 0 {
            indexs.push(self.pop_lsb());

        }

        indexs

    }

    pub fn bit_scan_forward(&self) -> usize {
        bit_scan_forward(self.0) as usize

    }

    pub fn bit_scan_reverse(&self) -> usize {
        bit_scan_reverse(self.0) as usize

    }

    pub fn pop_lsb(&mut self) -> usize {
        let bit = self.bit_scan_forward();
        self.0 &= !(1 << bit);
        bit
        
    }

    pub fn pop_msb(&mut self) -> usize {
        let bit = self.bit_scan_reverse();
        self.0 &= !(1 << bit);
        bit
        
    }

    pub fn pop_count(&self) -> usize {
        self.0.count_ones() as usize

    }

    pub fn set_bit(&mut self, bit: usize) {
        let mask = 1 << bit;

        self.0 |= mask;

    }

    pub fn clear_bit(&mut self, bit: usize) {
        let mask = 1 << bit;

        self.0 &= !mask;

    }

    pub fn toggle_bit(&mut self, bit: usize) {
        let mask = 1 << bit;

        self.0 ^= mask;

    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
        
    }
    
    pub fn is_not_empty(self) -> bool {
        self.0 != 0
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
