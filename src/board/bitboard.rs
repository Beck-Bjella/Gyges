use std::ops::{Not, Rem, RemAssign, BitOr, BitOrAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Shl, ShlAssign, Shr, ShrAssign};
use std::fmt::Display;

use crate::consts::*;
use crate::helper::bit_twiddles::*;

// Allows for shifting operations to be applied to a struct consisting of a singular tuple
/// containing a type that implements that bit operation.
macro_rules! impl_indv_shift_ops {
    ($t:ty, $tname:ident, $fname:ident, $w:ident, $ta_name:ident, $fa_name:ident) => (

        impl $tname<usize> for $t {
            type Output = $t;

            #[inline]
            fn $fname(self, rhs: usize) -> $t {
                Self::from((self.0).$w(rhs as u32))
            }
        }

        impl $ta_name<usize> for $t {

            #[inline]
            fn $fa_name(&mut self, rhs: usize) {
                *self = Self::from((self.0).$w(rhs as u32));

            }

        }

    )

}

/// Allows for bit operations to be applied to a struct consisting of a singular tuple
/// containing a type that implements that bit operation.
macro_rules! impl_indv_bit_ops {
    ($t:ty, $b:ty, $tname:ident, $fname:ident, $w:ident, $ta_name:ident, $fa_name:ident) => (

        impl $tname for $t {
            type Output = $t;

            #[inline]
            fn $fname(self, rhs: $t) -> $t {
                Self::from((self.0).$w(rhs.0))

            }

        }

        impl $ta_name for $t {

            #[inline]
            fn $fa_name(&mut self, rhs: $t) {
                *self = Self::from((self.0).$w(rhs.0));

            }

        }

        impl $tname<$b> for $t {
            type Output = $t;

            #[inline]
            fn $fname(self, rhs: $b) -> $t {
                Self::from((self.0).$w(rhs))

            }

        }

        impl $ta_name<$b> for $t {

            #[inline]
            fn $fa_name(&mut self, rhs: $b) {
                *self = Self::from((self.0).$w(rhs));

            }

        }

    )

}

/// Implies bit operations `&, |, ^, !`, shifting operations `<< >>`,1
/// math operations `+, -, *, /, %` and `From` trait to a struct consisting of a
/// singular tuple. This tuple must contain a type that implements these bit operations.
macro_rules! impl_bit_ops {
    ($t:tt, $b:tt) => (
        impl From<$b> for $t {
            fn from(bit_type: $b) -> Self {
                $t(bit_type)

            }

        }

        impl From<$t> for $b {
            fn from(it:$t) -> Self {
                it.0

            }

        }

        impl_indv_bit_ops!( $t, $b,  Rem,    rem,    rem,             RemAssign,    rem_assign);
        impl_indv_bit_ops!( $t, $b,  BitOr,  bitor,  bitor,           BitOrAssign,  bitor_assign);
        impl_indv_bit_ops!( $t, $b,  BitAnd, bitand, bitand,          BitAndAssign, bitand_assign);
        impl_indv_bit_ops!( $t, $b,  BitXor, bitxor, bitxor,          BitXorAssign, bitxor_assign);

        impl_indv_bit_ops!( $t, $b,  Add,    add,    wrapping_add,    AddAssign, add_assign);
        impl_indv_bit_ops!( $t, $b,  Div,    div,    wrapping_div,    DivAssign, div_assign);
        impl_indv_bit_ops!( $t, $b,  Mul,    mul,    wrapping_mul,    MulAssign, mul_assign);
        impl_indv_bit_ops!( $t, $b,  Sub,    sub,    wrapping_sub,    SubAssign, sub_assign);

        impl_indv_shift_ops!($t, Shl, shl, wrapping_shl,    ShlAssign, shl_assign);
        impl_indv_shift_ops!($t, Shr, shr, wrapping_shr,    ShrAssign, shr_assign);

        impl Not for $t {
            type Output = $t;

            #[inline]
            fn not(self) -> $t {
                $t(!self.0)

            }

        }

    )
    
}

#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct BitBoard(pub u64);

impl_bit_ops!(BitBoard, u64);

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

impl Display for BitBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.clone().get_data();
        
        if data.contains(&PLAYER_2_GOAL) {
            println!("          1");

        } else {
            println!("          0");
            
        }

        for y in (0..6).rev() {
            for x in 0..6 {
                if data.contains(&((y * 6) + x)) {
                    print!("  1");

                } else {
                    print!("  0");

                }

            }

            println!("")

        }

        if data.contains(&PLAYER_1_GOAL) {
            println!("          1");

        } else {
            println!("          0");
            
        }
    
        return Result::Ok(());

    }

}
