
// Matrix Types
pub type BinaryMatrix = [u64; 38];
pub type BinaryPieceMatrix = [u16; 14];

// ======================== DISPLAY FUNCTIONS ========================

/// Displays a binary adjacency matrix
pub fn disp(mat: &BinaryMatrix) {
    println!("┌─{}─┐",  " ".repeat((38 * 2) - 1).as_str());
    for i in 0..38 {
        let mut row = String::from("│ ");

        for j in 0..38 {
            if (mat[i] & (1 << j)) != 0 {
                row.push('■');
                row.push(' ');

            } else {
                row.push('□');
                row.push(' ');

            }
        }

        // Print a pipe character at the end of each row
        row.push('│');

        println!("{}", row);

    }

    println!("└─{}─┘",  " ".repeat((38 * 2) - 1).as_str());

}

/// Displays a binary piece adjacency matrix
pub fn disp_piece(mat: &BinaryPieceMatrix) {
    println!("┌─{}─┐",  " ".repeat((14 * 2) - 1).as_str());
    for i in 0..14 {
        let mut row = String::from("│ ");

        for j in 0..14 {
            if (mat[i] & (1 << j)) != 0 {
                row.push('1');
                row.push(' ');

            } else {
                row.push('0');
                row.push(' ');

            }
        }

        // Print a pipe character at the end of each row
        row.push('│');

        println!("{}", row);

    }

    println!("└─{}─┘",  " ".repeat((14 * 2) - 1).as_str());

}

// ======================== MATRIX MULTIPLICAITON  ========================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Efficiently multiplies two binary adjacency matrices using SIMD
#[target_feature(enable = "avx2")]
pub unsafe fn simd_binary_matrix_multiply(a: &BinaryMatrix, b: &BinaryMatrix) -> BinaryMatrix {
    let mut result = [0u64; 38];

    for i in 0..38 {
        if a[i] == 0 {
            continue; // Skip rows with no connections
        }

        let mut row_result = 0u64;

        // Main loop to process elements in groups of 4 using AVX2
        for j in (0..33).step_by(4) {
            // Load four `u64` elements from `b` into an AVX register
            let b_values = _mm256_set_epi64x(
                *b.get(j + 3).unwrap_or(&0) as i64,
                *b.get(j + 2).unwrap_or(&0) as i64,
                *b.get(j + 1).unwrap_or(&0) as i64,
                *b.get(j).unwrap_or(&0) as i64,
            );

            // Create a mask based on bits in `a[i]` at positions `j, j+1, j+2, j+3`
            let a_mask = _mm256_set_epi64x(
                if a[i] & (1 << (j + 3)) != 0 { -1 } else { 0 },
                if a[i] & (1 << (j + 2)) != 0 { -1 } else { 0 },
                if a[i] & (1 << (j + 1)) != 0 { -1 } else { 0 },
                if a[i] & (1 << j) != 0 { -1 } else { 0 },
            );

            // Perform SIMD AND to mask `b_values`, then OR the masked values into `row_result`
            let masked_values = _mm256_and_si256(a_mask, b_values);

            // Extract and accumulate each element into `row_result`
            row_result |= _mm256_extract_epi64(masked_values, 0) as u64;
            row_result |= _mm256_extract_epi64(masked_values, 1) as u64;
            row_result |= _mm256_extract_epi64(masked_values, 2) as u64;
            row_result |= _mm256_extract_epi64(masked_values, 3) as u64;
        }

        // Handle the remaining elements 36 and 37 w/o SIMD
        for j in 36..38 {
            if (a[i] & (1 << j)) != 0 {
                row_result |= b[j];
            }
        }

        // Store the accumulated row result into `result[i]`
        result[i] = row_result;
        
    }

    result

}

/// Efficiently multiplies two binary adjacency matrices
#[inline(always)]
pub unsafe fn binary_matrix_multiply(a: &BinaryMatrix, b: &BinaryMatrix) -> BinaryMatrix {
    let mut result = [0u64; 38];
    for i in 0..38 {
        if a[i] == 0 { // Skip empty rows
            continue;
        }

        for j in 0..38 {
            if (a[i] & (1 << j)) != 0 {
                result[i] |= b[j];
            }
        }

    }

    result

}

/// Efficiently multiplies two binary piece adjacency matrices
#[target_feature(enable = "avx2")]
pub unsafe fn simd_piece_binary_matrix_multiply(a: &BinaryPieceMatrix, b: &BinaryPieceMatrix) -> BinaryPieceMatrix {
    let mut result = [0u16; 14];

    for i in 0..14 {
        if a[i] == 0 {
            continue; // Skip rows with no connections
        }

        // Load all 16 columns from `b` at once
        let b_values = _mm256_set_epi16(
            *b.get(13).unwrap_or(&0) as i16,
            *b.get(12).unwrap_or(&0) as i16,
            *b.get(11).unwrap_or(&0) as i16,
            *b.get(10).unwrap_or(&0) as i16,
            *b.get(9).unwrap_or(&0) as i16,
            *b.get(8).unwrap_or(&0) as i16,
            *b.get(7).unwrap_or(&0) as i16,
            *b.get(6).unwrap_or(&0) as i16,
            *b.get(5).unwrap_or(&0) as i16,
            *b.get(4).unwrap_or(&0) as i16,
            *b.get(3).unwrap_or(&0) as i16,
            *b.get(2).unwrap_or(&0) as i16,
            *b.get(1).unwrap_or(&0) as i16,
            *b.get(0).unwrap_or(&0) as i16,
            0, 0, // Padding
        );

        // Create a mask based on each bit in `a[i]`
        let a_mask = _mm256_set_epi16(
            if a[i] & (1 << 13) != 0 { -1 } else { 0 },
            if a[i] & (1 << 12) != 0 { -1 } else { 0 },
            if a[i] & (1 << 11) != 0 { -1 } else { 0 },
            if a[i] & (1 << 10) != 0 { -1 } else { 0 },
            if a[i] & (1 << 9) != 0 { -1 } else { 0 },
            if a[i] & (1 << 8) != 0 { -1 } else { 0 },
            if a[i] & (1 << 7) != 0 { -1 } else { 0 },
            if a[i] & (1 << 6) != 0 { -1 } else { 0 },
            if a[i] & (1 << 5) != 0 { -1 } else { 0 },
            if a[i] & (1 << 4) != 0 { -1 } else { 0 },
            if a[i] & (1 << 3) != 0 { -1 } else { 0 },
            if a[i] & (1 << 2) != 0 { -1 } else { 0 },
            if a[i] & (1 << 1) != 0 { -1 } else { 0 },
            if a[i] & (1 << 0) != 0 { -1 } else { 0 },
            0, 0, // Padding
        );

        // Apply mask with AND operation
        let masked_values = _mm256_and_si256(a_mask, b_values);

        // Repeditly OR bits into one u16
        let lower_half: __m128i = _mm256_extracti128_si256(masked_values, 0);
        let upper_half = _mm256_extracti128_si256(masked_values, 1);
        let reduced = _mm_or_si128(lower_half, upper_half);
        let reduced = _mm_or_si128(reduced, _mm_srli_si128(reduced, 8));
        let reduced = _mm_or_si128(reduced, _mm_srli_si128(reduced, 4));
        let reduced = _mm_or_si128(reduced, _mm_srli_si128(reduced, 2));

        // Extract final result
        let row_result = _mm_extract_epi16(reduced, 0) as u16;

        // Store the accumulated row result into `result[i]`
        result[i] = row_result;

    }

    result

}


