use std::ops::*;
use crate::bit_twiddles::*;

#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct BitBoard(pub u64);

impl_bit_ops!(BitBoard, u64);

impl BitBoard {
    pub fn new(set_bit: usize) -> BitBoard {
        return BitBoard(0 << set_bit);

    }

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
