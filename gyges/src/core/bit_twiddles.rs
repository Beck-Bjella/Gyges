//! Bit twiddling functions and related constants.

static DEBRUIJ_T: &[u8] = &[
    0, 47, 1, 56, 48, 27, 2, 60, 57, 49, 41, 37, 28, 16, 3, 61, 54, 58, 35, 52, 50, 42, 21, 44, 38,
    32, 29, 23, 17, 11, 4, 62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45, 25,
    39, 14, 33, 19, 30, 9, 24, 13, 18, 8, 12, 7, 6, 5, 63,
];

const DEBRUIJ_M: u64 = 0x03f7_9d71_b4cb_0a89;

/// Returns the index of the least significant bit.
#[inline(always)]
pub fn bit_scan_forward(bits: u64) -> u8 {
    unsafe {
        *DEBRUIJ_T.get_unchecked(
            (((bits ^ bits.wrapping_sub(1)).wrapping_mul(DEBRUIJ_M)).wrapping_shr(58)) as usize,
        )

    }
    
}

/// Returns the index of the most significant bit.
#[inline(always)]
pub fn bit_scan_reverse(mut bb: u64) -> u8 {
    bb |= bb >> 1;
    bb |= bb >> 2;
    bb |= bb >> 4;
    bb |= bb >> 8;
    bb |= bb >> 16;
    bb |= bb >> 32;
    unsafe { *DEBRUIJ_T.get_unchecked((bb.wrapping_mul(DEBRUIJ_M)).wrapping_shr(58) as usize) }

}
