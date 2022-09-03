const INDEX64: [usize; 64] = [
    0, 47,  1, 56, 48, 27,  2, 60,
   57, 49, 41, 37, 28, 16,  3, 61,
   54, 58, 35, 52, 50, 42, 21, 44,
   38, 32, 29, 23, 17, 11,  4, 62,
   46, 55, 26, 59, 40, 36, 15, 53,
   34, 51, 20, 43, 31, 22, 10, 45,
   25, 39, 14, 33, 19, 30,  9, 24,
   13, 18,  8, 12,  7,  6,  5, 63
];

const DEBRUIJN64: u64 = 0x03f79d71b4cb0a89;

pub fn bit_scan_forward(bits: u64) -> usize {
    unsafe {
        return *INDEX64.get_unchecked((((bits ^ bits.wrapping_sub(1)).wrapping_mul(DEBRUIJN64)).wrapping_shr(58)) as usize);

    }
    
}
