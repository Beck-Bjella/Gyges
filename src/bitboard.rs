use std::ops::*;
use crate::bit_twiddles::*;

#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct BitBoard(pub u64);

impl_bit_ops!(BitBoard, u64);

impl BitBoard {
    pub fn print(&self) {
        println!("{:#b}", self.0);

    }

    pub fn bit_scan_forward(&self) -> usize {
        return bit_scan_forward(self.0);

    }

    pub fn pop_lsb(&mut self) -> usize {
        let bit = self.bit_scan_forward();
        *self &= *self - 1;
        bit
        
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


// impl BitAndAssign for BitBoard {
//     fn bitand_assign(&mut self, rhs: Self) {
//         self.0 = self.0 & rhs.0;

//     }

// }

// impl BitAnd for BitBoard {
//     type Output = Self;

//     fn bitand(self, rhs: Self) -> BitBoard {
//         BitBoard(self.0 & rhs.0)

//     }

// }

// impl BitOrAssign for BitBoard {
//     fn bitor_assign(&mut self, rhs: Self) {
//         self.0 = self.0 | rhs.0;
//     }

// }

// impl BitOr for BitBoard {
//     type Output = Self;

//     fn bitor(self, rhs: Self) -> BitBoard {
//         BitBoard(self.0 | rhs.0)
        
//     }

// }

// impl BitXorAssign for BitBoard {
//     fn bitxor_assign(&mut self, rhs: Self) {
//         self.0 = self.0 ^ rhs.0;
//     }

// }

// impl BitXor for BitBoard {
//     type Output = Self;

//     fn bitxor(self, rhs: Self) -> BitBoard {
//         BitBoard(self.0 ^ rhs.0)
//     }

// }

// impl ShlAssign for BitBoard {
//     fn shl_assign(&mut self, rhs: u8) {
//         self.0 <<= rhs;

//     }

// }

// impl Shl for BitBoard {
//     type Output = Self;

//     fn shl(self, rhs: u8) -> Self::Output {
//         self.0 << rhs;

//     }

// }

// impl ShrAssign for BitBoard {
//     fn shr_assign(&mut self, rhs: Self) {
//         self.0 >>= rhs;

//     }

// }

// impl Shr for BitBoard {
//     type Output = Self;

//     fn shr(self, rhs: u8) -> Self::Output {
//         self.0 >> rhs;

//     }

// }

// impl AddAssign for BitBoard {
//     fn add_assign(&mut self, rhs: Self) {
        
//     }

// }

// impl Add for BitBoard {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
        
//     }
    
// }

// impl SubAssign for BitBoard {
//     fn sub_assign(&mut self, rhs: Self) {
        
//     }

// }

// impl Sub for BitBoard {
//     type Output = Self;

//     fn sub(self, rhs: Self) -> Self::Output {
        
//     }

// }

// impl MulAssign for BitBoard {
//     fn mul_assign(&mut self, rhs: Self) {
        
//     }

// }

// impl Mul for BitBoard {
//     type Output = Self;

//     fn mul(self, rhs: Self) -> Self::Output {
        
//     }

// }

// impl DivAssign for BitBoard {
//     fn div_assign(&mut self, rhs: Self) {
        
//     }

// }

// impl Div for BitBoard {
//     type Output = Self;

//     fn div(self, rhs: Self) -> Self::Output {
        
//     }

// }

// impl RemAssign for BitBoard {
//     fn rem_assign(&mut self, rhs: Self) {
        
//     }

// }

// impl Rem for BitBoard {
//     type Output = Self;

//     fn rem(self, rhs: Self) -> Self::Output {
        
//     }

// }
