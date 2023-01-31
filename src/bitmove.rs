use std::ops::*;

#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct BitMove(pub u128);

impl_bit_ops!(BitMove, u128);

impl BitMove {
    pub fn new(data_0: usize, data_1: usize, data_2: usize, data_3: usize, data_4: usize, data_5: usize) -> BitMove {
        let mut bit_data: u128 = 0;

        bit_data ^= 1 << data_0;
        bit_data ^= 1 << (data_1 + 4);

        bit_data ^= 1 << (data_2 + 40);
        bit_data ^= 1 << (data_3 + 44);

        bit_data ^= 1 << (data_4 + 80);
        bit_data ^= 1 << (data_5 + 84);

        return BitMove(bit_data);

    }

    pub fn translate(&mut self) -> [u32; 6] {
        let mut data = [0; 6];

        let mut indexs: Vec<u32> = vec![];
        for _ in 0..6 {
            indexs.push(self.pop_lsb());
    
        }

        data[0] = indexs[0];
        data[1] = indexs[1] - 4;
        data[2] = indexs[2] - 40;
        data[3] = indexs[3] - 44;
        data[4] = indexs[4] - 80;
        data[5] = indexs[5] - 84;

        return data;

    }

    pub fn pop_lsb(&mut self) -> u32 {
        let i = self.0.trailing_zeros();

        self.0 ^= 1 << i;

        return i;

    }

    pub fn print(&self) {
        println!("{:#b}", self.0);

    }

    // pub fn bit_scan_forward(&self) -> usize {
    //     return bit_scan_forward(self.0);

    // }

    // pub fn pop_lsb(&mut self) -> usize {
    //     let bit = self.bit_scan_forward();
    //     *self &= *self - 1;
    //     bit
        
    // }

    // pub fn set_bit(&mut self, bit: usize) {
    //     let mask = 1 << bit;

    //     self.0 |= mask;

    // }

    // pub fn clear_bit(&mut self, bit: usize) {
    //     let mask = 1 << bit;

    //     self.0 &= !mask;

    // } 

}
