#include <stdint.h>

// ==================== CONSTANTS ====================

#define MATRIX_DIM 38
#define BIT_36_MASK (1ULL << 36)
#define BIT_37_MASK (1ULL << 37)
#define GOALS_MASK (BIT_36_MASK | BIT_37_MASK)


// ==================== HELPER FUNCTIONS ====================

extern "C" __device__ void print_bb(uint64_t bb) {
    for (int i = 0; i < 38; i++) {
        if (bb & ((uint64_t)1 << i)) {
            printf("1 ");

        } else {
            printf("0 ");

        }

    }

    printf("\n");

}

extern "C" __device__ uint64_t get_reach(
    // Inputs
    const uint8_t piece_type,
    const int pos,
    uint64_t piece_bb,

    // Lookup tables
    const uint64_t* unique_one_reachs,
    const uint64_t* unique_two_reachs,
    const uint64_t* unique_three_reachs,
    const uint64_t* all_two_intercepts,
    const uint64_t* all_three_intercepts,
    const uint8_t* one_map,
    const uint16_t* two_map,
    const uint16_t* three_map
) {    
    uint64_t intercept;
    int idx;
    uint64_t reach;
    switch (piece_type) {
        case 0: // Ones 
            idx = one_map[pos];
            reach = unique_one_reachs[idx];
            break;
            
        case 1: // Twos
            intercept = piece_bb & all_two_intercepts[pos];
            idx = two_map[(pos * 29) + intercept % 29];
            reach = unique_two_reachs[idx];
            break;

        case 2: // Threes
            intercept = piece_bb & all_three_intercepts[pos];
            idx = three_map[(pos * 11007) + intercept % 11007];
            reach = unique_three_reachs[idx];
            break;

    }

    return reach;

}


// ==================== KERNELS ======================

extern "C" __global__ void unified_kernel(
    const uint8_t* boards,       // Input boards
    const uint64_t* piece_bbs,   // Input bitboards for each board
    float* routes,               // Output routes

    // Lookup tables
    const uint64_t* one_reach,
    const uint64_t* two_reach,
    const uint64_t* three_reach,
    const uint64_t* all_two_intercepts,
    const uint64_t* all_three_intercepts,
    const uint8_t* one_map,
    const uint16_t* two_map,
    const uint16_t* three_map

) {
    uint64_t matrix_id = blockIdx.x;       // Each block processes one matrix
    uint64_t thread_id = threadIdx.x;      // Each thread processes one row

    // Shared memory for the adjacency matrixes
    __shared__ uint64_t adj_matrix[38];
    __shared__ uint64_t result_matrix[38];
    adj_matrix[thread_id] = 0;
    result_matrix[thread_id] = 0;

    // Step 1: Adjacency Matrix Generation
    uint8_t piece_type = boards[(matrix_id * 38) + thread_id];
    uint64_t piece_bb = piece_bbs[matrix_id];

    if (piece_type != 3 && thread_id < 36) { // Non empty spot   
        uint64_t reach = get_reach(
            piece_type,
            thread_id,
            piece_bb,
            
            one_reach,
            two_reach,
            three_reach,
            all_two_intercepts,
            all_three_intercepts,
            one_map,
            two_map,
            three_map
        );

        uint64_t masked_reach = reach & (piece_bb | GOALS_MASK);

        // Store result in shared memory
        adj_matrix[thread_id] = masked_reach;
        result_matrix[thread_id] = masked_reach;

    }

    __syncthreads(); // Sync threads

    // Step 2: Bitwise Matrix Multiplication
    for (uint64_t exp = 0; exp < 8; exp++) {
        // Each thread computes its rows result
        uint64_t row_result = 0;
        for (uint64_t k = 0; k < 38; k++) {
            row_result |= adj_matrix[k] * ((result_matrix[thread_id] >> k) & 1ULL);

        }
 
        // Store result
        result_matrix[thread_id] = row_result;

        __syncthreads(); // Sync before the next power calculation

    }

    // Step 3: Validate Routes
    if (thread_id == 0) {
        uint64_t mask = (uint64_t)1 << 36;
        for (int row = 30; row < 36; row++) {
            if (result_matrix[row] & mask) {
                routes[matrix_id] = 1.0;
                return;

            }

        }

        routes[matrix_id] = 0.0;

    }

}
