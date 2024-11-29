#include <stdint.h>
#include <cuda_runtime.h>
#include <cstdio>

// ==================== CONSTANTS ====================

#define MATRIX_DIM 38
#define BIT_36_MASK (1ULL << 36)
#define BIT_37_MASK (1ULL << 37)
#define GOALS_MASK (BIT_36_MASK | BIT_37_MASK)

// Lookup table for intercepts. 0-35 are for twos, 36-71 are for threes.
__constant__ uint64_t ALL_INTERCEPTS[72] {
    // TWOS
    0b00000000000000000000000000000000000000000000000000000001000010ULL,
    0b00000000000000000000000000000000000000000000000000000010000101ULL,
    0b00000000000000000000000000000000000000000000000000000100001010ULL,
    0b00000000000000000000000000000000000000000000000000001000010100ULL,
    0b00000000000000000000000000000000000000000000000000010000101000ULL,
    0b00000000000000000000000000000000000000000000000000100000010000ULL,
    0b00000000000000000000000000000000000000000000000001000010000001ULL,
    0b00000000000000000000000000000000000000000000000010000101000010ULL,
    0b00000000000000000000000000000000000000000000000100001010000100ULL,
    0b00000000000000000000000000000000000000000000001000010100001000ULL,
    0b00000000000000000000000000000000000000000000010000101000010000ULL,
    0b00000000000000000000000000000000000000000000100000010000100000ULL,
    0b00000000000000000000000000000000000000000001000010000001000000ULL,
    0b00000000000000000000000000000000000000000010000101000010000000ULL,
    0b00000000000000000000000000000000000000000100001010000100000000ULL,
    0b00000000000000000000000000000000000000001000010100001000000000ULL,
    0b00000000000000000000000000000000000000010000101000010000000000ULL,
    0b00000000000000000000000000000000000000100000010000100000000000ULL,
    0b00000000000000000000000000000000000001000010000001000000000000ULL,
    0b00000000000000000000000000000000000010000101000010000000000000ULL,
    0b00000000000000000000000000000000000100001010000100000000000000ULL,
    0b00000000000000000000000000000000001000010100001000000000000000ULL,
    0b00000000000000000000000000000000010000101000010000000000000000ULL,
    0b00000000000000000000000000000000100000010000100000000000000000ULL,
    0b00000000000000000000000000000001000010000001000000000000000000ULL,
    0b00000000000000000000000000000010000101000010000000000000000000ULL,
    0b00000000000000000000000000000100001010000100000000000000000000ULL,
    0b00000000000000000000000000001000010100001000000000000000000000ULL,
    0b00000000000000000000000000010000101000010000000000000000000000ULL,
    0b00000000000000000000000000100000010000100000000000000000000000ULL,
    0b00000000000000000000000000000010000001000000000000000000000000ULL,
    0b00000000000000000000000000000101000010000000000000000000000000ULL,
    0b00000000000000000000000000001010000100000000000000000000000000ULL,
    0b00000000000000000000000000010100001000000000000000000000000000ULL,
    0b00000000000000000000000000101000010000000000000000000000000000ULL,
    0b00000000000000000000000000010000100000000000000000000000000000ULL,
    // THREES
    0b0000000000000000000000000000000000000000000000000001000011000110ULL,
    0b0000000000000000000000000000000000000000000000000010000111001101ULL,
    0b0000000000000000000000000000000000000000000000000100001110011011ULL,
    0b0000000000000000000000000000000000000000000000001000011100110110ULL,
    0b0000000000000000000000000000000000000000000000010000111000101100ULL,
    0b0000000000000000000000000000000000000000000000100000110000011000ULL,
    0b0000000000000000000000000000000000000000000001000011000110000011ULL,
    0b0000000000000000000000000000000000000000000010000111001101000111ULL,
    0b0000000000000000000000000000000000000000000100001110011011001110ULL,
    0b0000000000000000000000000000000000000000001000011100110110011100ULL,
    0b0000000000000000000000000000000000000000010000111000101100111000ULL,
    0b0000000000000000000000000000000000000000100000110000011000110000ULL,
    0b0000000000000000000000000000000000000001000011000110000011000001ULL,
    0b0000000000000000000000000000000000000010000111001101000111000010ULL,
    0b0000000000000000000000000000000000000100001110011011001110000100ULL,
    0b0000000000000000000000000000000000001000011100110110011100001000ULL,
    0b0000000000000000000000000000000000010000111000101100111000010000ULL,
    0b0000000000000000000000000000000000100000110000011000110000100000ULL,
    0b0000000000000000000000000000000001000011000110000011000001000000ULL,
    0b0000000000000000000000000000000010000111001101000111000010000000ULL,
    0b0000000000000000000000000000000100001110011011001110000100000000ULL,
    0b0000000000000000000000000000001000011100110110011100001000000000ULL,
    0b0000000000000000000000000000010000111000101100111000010000000000ULL,
    0b0000000000000000000000000000100000110000011000110000100000000000ULL,
    0b0000000000000000000000000000000011000110000011000001000000000000ULL,
    0b0000000000000000000000000000000111001101000111000010000000000000ULL,
    0b0000000000000000000000000000001110011011001110000100000000000000ULL,
    0b0000000000000000000000000000011100110110011100001000000000000000ULL,
    0b0000000000000000000000000000111000101100111000010000000000000000ULL,
    0b0000000000000000000000000000110000011000110000100000000000000000ULL,
    0b0000000000000000000000000000000110000011000001000000000000000000ULL,
    0b0000000000000000000000000000001101000111000010000000000000000000ULL,
    0b0000000000000000000000000000011011001110000100000000000000000000ULL,
    0b0000000000000000000000000000110110011100001000000000000000000000ULL,
    0b0000000000000000000000000000101100111000010000000000000000000000ULL,
    0b0000000000000000000000000000011000110000100000000000000000000000ULL

};

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

// ==================== BIT STATE HELPERS ====================

// Remove an existing piece type
__device__ void remove_type(uint64_t* state, uint8_t piece_idx) {
    uint8_t type_pos = 38 + (piece_idx * 2);

    // Clear the two bits at type_pos
    uint64_t clear_mask = ~((uint64_t)0b11 << type_pos);
    *state &= clear_mask;

    // Shift higher bits down by 2
    uint64_t higher_bits_mask = ~((uint64_t)0) << (type_pos + 2);
    uint64_t higher_bits = (*state & higher_bits_mask) >> 2;
    *state = (*state & ~higher_bits_mask) | higher_bits;
}

// Add a new piece type
__device__ void add_type(uint64_t* state, uint8_t piece_idx, uint8_t piece_type) {
    uint8_t type_pos = 38 + (piece_idx * 2);

    // Shift higher bits up by 2 to make space
    uint64_t higher_bits_mask = ~((uint64_t)0) << type_pos;
    uint64_t higher_bits = (*state & higher_bits_mask) << 2;
    *state = (*state & ~higher_bits_mask) | higher_bits;

    // Set the new piece type
    *state |= ((uint64_t)piece_type << type_pos);
}

// Change the type data at a specific index
__device__ void set_type_data(uint64_t* state, uint8_t piece_idx, uint8_t piece_type) {
    uint8_t type_pos = 38 + (piece_idx * 2);

    // Mask to clear the two bits at type_pos
    uint64_t clear_mask = ~((uint64_t)0b11 << type_pos);

    // Clear and set the piece type bits at piece_idx
    *state = (*state & clear_mask) | ((uint64_t)piece_type << type_pos);
}

// Get the piece bitboard
__device__ uint64_t get_piece_bb(uint64_t state) {
    return state & (((uint64_t)1 << 38) - 1);
}

// Get the index of the piece at a position
__device__ uint8_t piece_idx(uint64_t state, uint8_t pos) {
    uint64_t mask = ((uint64_t)1 << pos) - 1;
    uint64_t bits_before = get_piece_bb(state) & mask;

    // Use CUDA intrinsic to count set bits
    return __popcll(bits_before);
}

// Get the piece type at a given piece index
__device__ uint8_t piece_type(uint64_t state, uint8_t piece_idx) {
    return (uint8_t)((state >> (38 + (piece_idx * 2))) & 0b11);
}

// Get the piece at a square
// 0 = None, 1 = One, 2 = Two, 3 = Three
__device__ uint8_t piece_at(uint64_t state, uint8_t pos) {
    if ((state & ((uint64_t)1 << pos)) == 0) {
        return 0;
    }

    uint8_t idx = piece_idx(state, pos);
    return piece_type(state, idx);
}

// ==================== BIT STATE ====================

// Get the player to move 
__device__ uint8_t player(uint64_t state) {
    return (uint8_t)(state >> 63);
}

// Make a bounce move
__device__ uint64_t make_bounce_mv(uint64_t state, uint8_t start_pos, uint8_t end_pos) {
    uint64_t new_state = state;

    // Update the piece bitboard
    new_state ^= ((uint64_t)1 << start_pos) | ((uint64_t)1 << end_pos);

    // Starting piece index and type
    uint8_t starting_idx = piece_idx(state, start_pos);
    uint8_t starting_piece = piece_type(state, starting_idx);

    // Remove the piece type at starting index
    remove_type(&new_state, starting_idx);

    // Since we've removed a piece type, adjust the ending index
    uint8_t ending_idx;
    if (start_pos < end_pos) {
        ending_idx = piece_idx(state, end_pos) - 1;
    } else {
        ending_idx = piece_idx(state, end_pos);
    }

    // Add the starting piece type at the new index
    add_type(&new_state, ending_idx, starting_piece);

    return new_state;
}

// Make a drop move
__device__ uint64_t make_drop_mv(uint64_t state, uint8_t start_pos, uint8_t pickup_pos, uint8_t drop_pos) {
    uint64_t new_state = state;

    // Starting piece index and type
    uint8_t starting_idx = piece_idx(state, start_pos);
    uint8_t starting_piece = piece_type(state, starting_idx);

    // Remove the starting piece
    remove_type(&new_state, starting_idx);
    new_state ^= (uint64_t)1 << start_pos;

    // Pickup piece index and type
    uint8_t pickup_idx = piece_idx(new_state, pickup_pos);
    uint8_t pickup_piece = piece_type(new_state, pickup_idx);

    // Set the pickup piece's type to the starting piece's type
    set_type_data(&new_state, pickup_idx, starting_piece);

    // Drop piece index
    uint8_t drop_idx = piece_idx(new_state, drop_pos);

    // Add the pickup piece type at the drop index
    add_type(&new_state, drop_idx, pickup_piece);

    // Update the piece bitboard
    new_state ^= (uint64_t)1 << drop_pos;

    return new_state;
}

// ==================== HELPERS ====================

extern "C" __device__ uint64_t reach(
    uint64_t piece_bb,
    uint8_t piece_type,
    uint64_t piece_pos,

    // Lookup tables
    const uint64_t* __restrict__ one_reach,
    const uint64_t* __restrict__ two_reach,
    const uint64_t* __restrict__ three_reach
) {
    uint64_t reach;
    uint64_t intercepts;
    if (piece_type == 1) {
        reach = one_reach[piece_pos];
        
    } else if (piece_type == 2) {
        intercepts = piece_bb & ALL_INTERCEPTS[piece_pos];
        reach = two_reach[(piece_pos * 29) + (intercepts % 29)];
        
    } else if (piece_type == 3) {
        intercepts = piece_bb & ALL_INTERCEPTS[piece_pos + 36];
        reach = three_reach[(piece_pos * 11007) + (intercepts % 11007)];

    }

    // Mask reach
    uint64_t masked_reach = reach & (piece_bb | GOALS_MASK);

    return masked_reach;

}

// ==================== BLOCKING KERNELS ======================

// Adj matrix vairant
extern "C" __global__ void adj_kernel(
    uint64_t* init_state, // Starting state
    uint8_t* move_data,   // Move data
    float* routes,        // Output 

    // Lookup tables
    const uint64_t* __restrict__ one_reach,
    const uint64_t* __restrict__ two_reach,
    const uint64_t* __restrict__ three_reach

) {
    uint64_t matrix_id = blockIdx.x;       // Each block processes one matrix
    uint64_t thread_id = threadIdx.x;      // Each thread processes one row

    // Init shared memory
    __shared__ uint64_t adj_matrix[38];
    __shared__ uint64_t result_matrix[38];
    __shared__ uint64_t shared_new_state;

    if (thread_id == 0) {
        uint8_t start_pos = move_data[(matrix_id * 3)];
        uint8_t pickup_pos = move_data[(matrix_id * 3) + 1];
        uint8_t end_pos = move_data[(matrix_id * 3) + 2];

        uint64_t new_state = (end_pos == 100)
            ? make_bounce_mv(init_state[0], start_pos, pickup_pos)
            : make_drop_mv(init_state[0], start_pos, pickup_pos, end_pos);

        shared_new_state = new_state;

    }

    __syncthreads(); // Sync threads

    // Step 1: Generate Adj Matrix
    uint8_t piece_type = piece_at(shared_new_state, thread_id);
    if (piece_type != 0) {
        uint64_t piece_bb = get_piece_bb(shared_new_state);

        adj_matrix[thread_id] = reach(
            piece_bb,
            piece_type,
            thread_id,
            one_reach,
            two_reach,
            three_reach
        );

    } else {
        adj_matrix[thread_id] = 0;

    }
    
    result_matrix[thread_id] = adj_matrix[thread_id];

    __syncthreads(); // Sync threads

    // Step 2: Bitwise Matrix Multiplication
    for (uint64_t exp = 0; exp < 8; exp++) {
        // Each thread computes its rows result
        uint64_t row_result = 0;
        for (uint64_t k = 0; k < 38; k++) {
            row_result |= adj_matrix[k] * ((result_matrix[thread_id] >> k) & 1ULL);

        }

        // Store result
        result_matrix[thread_id] |= row_result;

        __syncthreads(); // Sync before the next power calculation

    }

    // Step 3: Validate Routes
    if (thread_id == 0) {
        for (int row = 30; row < 36; row++) {
            if (result_matrix[row] & BIT_36_MASK) {
                routes[matrix_id] = 1.0; // There is a path to the goal -> did not block the threat
                return;

            }

        }

        routes[matrix_id] = 0.0; // No path to the goal -> blocked the threat

    }

}

// Wavefront variant
extern "C" __global__ void wavefront_kernel(
    uint64_t* init_state, // Starting state
    uint8_t* move_data,   // Move data
    float* routes,        // Output 

    // Lookup tables
    const uint64_t* __restrict__ one_reach,
    const uint64_t* __restrict__ two_reach,
    const uint64_t* __restrict__ three_reach

) {
    uint64_t block_id = blockIdx.x;       // Each block processes one matrix
    uint64_t thread_id = threadIdx.x;     // Each thread processes one row

    // Shared memory
    __shared__ uint64_t current_frontier; // Current wavefront of nodes
    __shared__ uint64_t next_frontier;    // Next wavefront of nodes
    __shared__ uint64_t reached;          // Reached nodes

    __shared__ uint64_t shared_state;
    __shared__ uint64_t shared_bb;

    if (thread_id == 0) {
        // Frontiers
        current_frontier = 0ULL;
        current_frontier |= (1ULL << 31) | (1ULL << 33); // NEED TO CHANGE

        next_frontier = 0ULL;

        reached = current_frontier;
        
        // Shared state
        uint8_t start_pos = move_data[(block_id * 3)];
        uint8_t pickup_pos = move_data[(block_id * 3) + 1];
        uint8_t end_pos = move_data[(block_id * 3) + 2];

        uint64_t new_state = (end_pos == 100)
            ? make_bounce_mv(init_state[0], start_pos, pickup_pos)
            : make_drop_mv(init_state[0], start_pos, pickup_pos, end_pos);

        shared_state = new_state;
        shared_bb = get_piece_bb(new_state);

    }

    __syncthreads(); // Sync threads

    for (int depth = 0; depth < 8; depth++) {
        // Exit condition
        if (current_frontier == 0ULL || current_frontier & BIT_36_MASK) {
            break;
        }

        if ((1ULL << thread_id) & current_frontier) {
            uint8_t piece_type = piece_at(shared_state, thread_id);
            if (piece_type != 0) {
                uint64_t reachable = reach(
                    shared_bb,
                    piece_type,
                    thread_id,
                    one_reach,
                    two_reach,
                    three_reach
                );

                // Remove already reached positions
                reachable &= ~reached;

                // Update next frontier
                atomicOr(&next_frontier, reachable);

            }

        }

        __syncthreads(); // Sync threads

        // Swap frontiers
        if (thread_id == 0) {
            current_frontier = next_frontier;
            next_frontier = 0ULL;

            atomicOr(&reached, current_frontier);

        }

        __syncthreads(); // Sync threads

    }

    // Store result
    if (thread_id == 0) {
        routes[block_id] = (current_frontier & BIT_36_MASK) ? 1.0f : 0.0f;

    }

}

// ==============================================================
// ==============================================================
// =======================   NEW STUFF  =========================  
// ==============================================================
// ==============================================================

// Remove an existing piece type
__device__ uint64_t new_remove_type(uint64_t state, uint8_t piece_idx) {
    uint64_t new_state = state;

    uint8_t type_pos = 38 + (piece_idx * 2);

    // Clear the two bits at type_pos
    uint64_t clear_mask = ~((uint64_t)0b11 << type_pos);
    new_state &= clear_mask;

    // Shift higher bits down by 2
    uint64_t higher_bits_mask = ~((uint64_t)0) << (type_pos + 2);
    uint64_t higher_bits = (new_state & higher_bits_mask) >> 2;
    new_state = (new_state & ~higher_bits_mask) | higher_bits;

    return new_state;

}

__device__ uint64_t remove_piece(uint64_t state, uint8_t pos) {
    uint64_t new_state = state;

    // Clear the bit at pos
    new_state &= ~((uint64_t)1 << pos);

    // Remove the piece type
    new_state = new_remove_type(new_state, piece_idx(state, pos));

    return new_state;

}

// ==============================================================

#define MAX_STACK_SIZE 1000

struct OnePath {
    uint64_t backtrack_bb;
    uint8_t p1;
    uint8_t p2;
};

struct TwoPath {
    uint64_t backtrack_bb;
    uint8_t p1;
    uint8_t p2;
    uint8_t p3;
};

struct ThreePath {
    uint64_t backtrack_bb;
    uint8_t p1;
    uint8_t p2;
    uint8_t p3;
    uint8_t p4;
};

struct StackData {
    uint64_t banned_bb;
    uint64_t backtrack_bb;
    uint32_t active_line_idx;
    uint8_t current_pos;
    uint8_t current_piece;

};

struct GenRequest {
    uint64_t state;
    uint64_t active_bb;
    uint8_t flag;

};

struct GenResult {
    uint64_t end_positions[6];
    uint64_t pickup_positions[6];
    uint64_t drop_positions;
    
};

// ==============================================================

// Pushs to a stach w/o any overflow handling
__device__ void push(StackData* stack, uint32_t* stack_height, uint32_t block_id, uint32_t type, StackData data) {
    uint32_t current_height = atomicAdd(stack_height, 1); // Get the position to push to
    stack[(block_id * MAX_STACK_SIZE * 3) + (type * MAX_STACK_SIZE) + current_height] = data;

}

// Pops from a stack w/o any underflow handling
__device__ StackData pop(StackData* stack, uint32_t* stack_height, uint32_t block_id, uint32_t type) {
    uint32_t current_height = atomicSub(stack_height, 1) - 1; // Get the position to pop from
    return stack[(block_id * MAX_STACK_SIZE * 3) + (type * MAX_STACK_SIZE) + current_height];

}

// ==============================================================

// Lookup tables
__device__ OnePath* one_paths;
__device__ TwoPath* two_paths;
__device__ ThreePath* three_paths;
__device__ uint16_t* one_path_lists;
__device__ uint16_t* two_path_lists;
__device__ uint16_t* three_path_lists;
__device__ uint8_t* one_map;
__device__ uint16_t* two_map;
__device__ uint16_t* three_map;

// ==============================================================

__device__ void process_one(
    StackData* stack, 
    uint32_t* one_stack_height, 
    uint32_t* two_stack_height,
    uint32_t* three_stack_height,
    StackData current_data,
    uint64_t current_state,
    uint64_t end_positions,
    uint64_t* local_end_positions, 
    uint64_t* local_pickup_positions,
    uint64_t block_id
) {
    uint16_t path_list_idx = one_map[current_data.current_pos];
    uint16_t path_list_len = one_path_lists[(path_list_idx * 5) + 4];

    for (int i = 0; i < path_list_len; i++) {
        uint16_t path_idx = one_path_lists[(path_list_idx * 5) + i];
        OnePath path = one_paths[path_idx];

        uint8_t end_pos = path.p2;
        uint64_t end_bit = (uint64_t)1 << end_pos;

        uint64_t end_pos_banned = (current_data.banned_bb | end_positions) & end_bit;
        uint64_t backtrack_conflict = current_data.backtrack_bb & path.backtrack_bb; 
        bool valid_player = !(((end_bit & BIT_36_MASK) && 0 == 0) || ((end_bit & BIT_37_MASK) && 0 == 1));

        if (backtrack_conflict || end_pos_banned || !valid_player) {
            continue;

        }           
            
        uint64_t end_piece = piece_at(current_state, end_pos);
        bool is_empty = (end_piece == 0);

        if (is_empty) {
            *local_end_positions |= end_bit;

        } else {
            uint64_t new_banned_bb = current_data.banned_bb ^ end_bit;
            uint64_t new_backtrack_bb = current_data.backtrack_bb ^ path.backtrack_bb;
            
            *local_pickup_positions |= end_bit;

            StackData new_data = {
                new_banned_bb,
                new_backtrack_bb,
                current_data.active_line_idx,
                end_pos,
                (uint8_t)end_piece,
            };

            int stack_type = end_piece - 1;
            push(
                stack, 
                (stack_type == 0) ? one_stack_height : (stack_type == 1) ? two_stack_height : three_stack_height, 
                block_id, 
                stack_type, 
                new_data
            );

        }
            
    }

}

__device__ void process_two(
    StackData* stack, 
    uint32_t* one_stack_height, 
    uint32_t* two_stack_height,
    uint32_t* three_stack_height,
    StackData current_data, 
    uint64_t current_state,
    uint64_t end_positions,
    uint64_t* local_end_positions, 
    uint64_t* local_pickup_positions,
    uint64_t block_id
) {
    uint64_t intercept_bb = get_piece_bb(current_state) & ALL_INTERCEPTS[current_data.current_pos];

    uint16_t path_list_idx = two_map[(current_data.current_pos * 29) + (intercept_bb % 29)];
    uint16_t path_list_len = two_path_lists[(path_list_idx * 13) + 12];

    for (int i = 0; i < path_list_len; i++) {
        uint16_t path_idx = two_path_lists[(path_list_idx * 13) + i];
        TwoPath path = two_paths[path_idx];

        uint8_t end_pos = path.p3;
        uint64_t end_bit = (uint64_t)1 << end_pos;

        uint64_t end_pos_banned = (current_data.banned_bb | end_positions) & end_bit;
        uint64_t backtrack_conflict = current_data.backtrack_bb & path.backtrack_bb; 
        bool valid_player = !(((end_bit & BIT_36_MASK) && 0 == 0) || ((end_bit & BIT_37_MASK) && 0 == 1));

        if (backtrack_conflict || end_pos_banned || !valid_player) {
            continue;

        }           
            
        uint64_t end_piece = piece_at(current_state, end_pos);
        bool is_empty = (end_piece == 0);

        if (is_empty) {
            *local_end_positions |= end_bit;

        } else {
            uint64_t new_banned_bb = current_data.banned_bb ^ end_bit;
            uint64_t new_backtrack_bb = current_data.backtrack_bb ^ path.backtrack_bb;
            
            *local_pickup_positions |= end_bit;

            StackData new_data = {
                new_banned_bb,
                new_backtrack_bb,
                current_data.active_line_idx,
                end_pos,
                (uint8_t)end_piece,
            };

            int stack_type = end_piece - 1;
            push(
                stack, 
                (stack_type == 0) ? one_stack_height : (stack_type == 1) ? two_stack_height : three_stack_height, 
                block_id, 
                stack_type, 
                new_data
            );

        }

    }

}

__device__ void process_three(
    StackData* stack, 
    uint32_t* one_stack_height, 
    uint32_t* two_stack_height,
    uint32_t* three_stack_height,
    StackData current_data, 
    uint64_t current_state,
    uint64_t end_positions,
    uint64_t* local_end_positions, 
    uint64_t* local_pickup_positions,
    uint64_t block_id
) {
    uint64_t intercept_bb = get_piece_bb(current_state) & ALL_INTERCEPTS[current_data.current_pos + 36];

    uint16_t path_list_idx = three_map[(current_data.current_pos * 11007) + (intercept_bb % 11007)];
    uint16_t path_list_len = three_path_lists[(path_list_idx * 36) + 35];

    for (int i = 0; i < path_list_len; i++) {
        uint16_t path_idx = three_path_lists[(path_list_idx * 36) + i];
        ThreePath path = three_paths[path_idx];

        uint8_t end_pos = path.p4;
        uint64_t end_bit = (uint64_t)1 << end_pos;

        uint64_t end_pos_banned = (current_data.banned_bb | end_positions) & end_bit;
        uint64_t backtrack_conflict = current_data.backtrack_bb & path.backtrack_bb; 
        bool valid_player = !(((end_bit & BIT_36_MASK) && 0 == 0) || ((end_bit & BIT_37_MASK) && 0 == 1));

        if (backtrack_conflict || end_pos_banned || !valid_player) {
            continue;

        }           
            
        uint64_t end_piece = piece_at(current_state, end_pos);
        bool is_empty = (end_piece == 0);

        if (is_empty) {
            *local_end_positions |= end_bit;

        } else {
            uint64_t new_banned_bb = current_data.banned_bb ^ end_bit;
            uint64_t new_backtrack_bb = current_data.backtrack_bb ^ path.backtrack_bb;
            
            *local_pickup_positions |= end_bit;

            StackData new_data = {
                new_banned_bb,
                new_backtrack_bb,
                current_data.active_line_idx,
                end_pos,
                (uint8_t)end_piece,
            };

            int stack_type = end_piece - 1;
            push(
                stack, 
                (stack_type == 0) ? one_stack_height : (stack_type == 1) ? two_stack_height : three_stack_height, 
                block_id, 
                stack_type, 
                new_data
            );

        }

    } 

}

extern "C" __global__ void gen_kernel(
    const GenRequest* __restrict__ in_data,
    GenResult* out_data,  
    StackData* stack
    
) {
    uint32_t block_id = blockIdx.x;       // Each block processes one generation request
    uint32_t thread_id = threadIdx.x;     // 
    uint32_t warp_id = threadIdx.x / 32;  //
    uint32_t lane_id = threadIdx.x % 32;  // 

    // Init Shared Memory
    __shared__ uint32_t one_stack_height;       // Stack height for ones
    __shared__ uint32_t two_stack_height;       // Stack height for twos
    __shared__ uint32_t three_stack_height;     // Stack height for threes
    __shared__ uint64_t init_state;             // Initial state
    __shared__ uint64_t end_positions[6];       // End positions
    __shared__ uint64_t pickup_positions[6];    // Pickup positions
    __shared__ uint64_t drop_positions;         // Drop positions
    if (thread_id == 0) {
        one_stack_height = 0;
        two_stack_height = 0;
        three_stack_height = 0;
        init_state = in_data[block_id].state;

        for (int i = 0; i < 6; i++) {
            end_positions[i] = 0ULL;
            pickup_positions[i] = 0ULL;

        }
        drop_positions = (~init_state & 0b111111111111111111111111111111111111ULL);

    }

    __syncthreads(); // Sync threads
    
    // Setup
    __shared__ uint64_t starting_states[6]; // Starting states for active line
    uint64_t start_bb = in_data[block_id].active_bb;
    if (((1ULL << thread_id) & start_bb) && thread_id < 6) {
        uint64_t new_state = remove_piece(init_state, thread_id);
        starting_states[thread_id] = new_state; // WRONG INDEX -> ONLY WORKS WHEN STARTING LINE IS ON THE FIRST ROW

        // Create init stack data
        uint8_t piece = piece_at(init_state, thread_id);
        StackData data = {
            0ULL,
            0ULL,
            thread_id, // WRONG INDEX -> ONLY WORKS WHEN STARTING LINE IS ON THE FIRST ROW
            (uint8_t)thread_id, 
            piece,
        };

        // Push to stack
        int stack_type = piece - 1;
        push(
            stack, 
            (stack_type == 0) ? &one_stack_height : (stack_type == 1) ? &two_stack_height : &three_stack_height, 
            block_id, 
            stack_type, 
            data
        );

    }
    
    __syncthreads(); // Sync threads

    // MAIN PROCESSING LOOP
    while (true) {
        if (warp_id == 0 && lane_id < one_stack_height) { // ONES -> warp 0
            StackData current_data = pop(stack, &one_stack_height, block_id, 0);
            uint64_t current_state = starting_states[current_data.active_line_idx];

            uint64_t local_end_positions = 0ULL;
            uint64_t local_pickup_positions = 0ULL;

            process_one(
                stack,
                &one_stack_height,
                &two_stack_height,
                &three_stack_height,
                current_data,
                current_state,
                end_positions[current_data.active_line_idx],
                &local_end_positions,
                &local_pickup_positions,
                block_id
            );

            atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
            atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

        } else if (warp_id == 1 && lane_id < two_stack_height) { // TWOS -> warp 1
            StackData current_data = pop(stack, &two_stack_height, block_id, 1);
            uint64_t current_state = starting_states[current_data.active_line_idx];

            uint64_t local_end_positions = 0ULL;
            uint64_t local_pickup_positions = 0ULL;

            process_two(
                stack,
                &one_stack_height,
                &two_stack_height,
                &three_stack_height,
                current_data,
                current_state,
                end_positions[current_data.active_line_idx],
                &local_end_positions,
                &local_pickup_positions,
                block_id
            );

            atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
            atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

        } else if (warp_id == 2 && lane_id < three_stack_height) { // THREES -> warp 2
            StackData current_data = pop(stack, &three_stack_height, block_id, 2);
            uint64_t current_state = starting_states[current_data.active_line_idx];

            uint64_t local_end_positions = 0ULL;
            uint64_t local_pickup_positions = 0ULL;

            process_three(
                stack,
                &one_stack_height,
                &two_stack_height,
                &three_stack_height,
                current_data,
                current_state,
                end_positions[current_data.active_line_idx],
                &local_end_positions,
                &local_pickup_positions,
                block_id
            );

            atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
            atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

        }

        __syncthreads(); // Sync threads

        // Exit condition
        if (one_stack_height == 0 && two_stack_height == 0 && three_stack_height == 0) {
            break;

        }

    }

    __syncthreads(); // Sync threads

    // Store results
    if (thread_id == 0) {
        GenResult result;
        for (int i = 0; i < 6; i++) {
            result.end_positions[i] = end_positions[i];
            result.pickup_positions[i] = pickup_positions[i];

        }
        result.drop_positions = drop_positions;

        out_data[block_id] = result;

    }

}

// ==============================================================


// extern "C" __global__ void gen_kernel(
//     const GenRequest* __restrict__ in_data,
//     GenResult* out_data,   
//     StackData* stack
    
// ) {
//     uint32_t block_id = blockIdx.x;       // Each block processes one generation request
//     uint32_t thread_id = threadIdx.x;     // Each thread processes one postion

//     // Init Shared Memory
//     __shared__ uint32_t stack_height;           // Stack height
//     __shared__ uint64_t init_state;             // Initial state
//     __shared__ uint64_t end_positions[6];       // End positions
//     __shared__ uint64_t pickup_positions[6];    // Pickup positions
//     __shared__ uint64_t drop_positions;         // Drop positions
//     if (thread_id == 0) {
//         stack_height = 0;
//         init_state = in_data[block_id].state;

//         for (int i = 0; i < 6; i++) {
//             end_positions[i] = 0ULL;
//             pickup_positions[i] = 0ULL;

//         }
//         drop_positions = (~init_state & 0b111111111111111111111111111111111111ULL);

//     }

//     __syncthreads(); // Sync threads
    
//     // Setup
//     __shared__ uint64_t starting_states[6]; // Starting states
//     uint64_t start_bb = in_data[block_id].active_bb;
//     if ((1ULL << thread_id) & start_bb) {
//         // Save starting info to shared memory
//         uint64_t new_state = remove_piece(init_state, thread_id);
//         starting_states[thread_id] = new_state; // WRONG INDEX -> ONLY WORKS WHEN STARTING LINE IS ON THE FIRST ROW

//         // Store starting data into stack
//         StackData data = {
//             0ULL,
//             0ULL,
//             thread_id, // WRONG INDEX -> ONLY WORKS WHEN STARTING LINE IS ON THE FIRST ROW
//             (uint8_t)thread_id, 
//             piece_at(init_state, thread_id),
//         };

//         push(stack, &stack_height, block_id, data);

//     }

//     __syncthreads(); // Sync threads

//     // TESTING PURPOSES
//     uint16_t player = 0;    

//     // MAIN PROCESSING LOOP
//     while (true) {
//         if (thread_id < stack_height) {
//             StackData current_data = pop(stack, &stack_height, block_id);
//             uint64_t current_state = starting_states[current_data.active_line_idx];

//             uint64_t local_end_positions = 0ULL;
//             uint64_t local_pickup_positions = 0ULL;

//             if (current_data.current_piece == 1) { // ONES
//                 uint16_t path_list_idx = one_map[current_data.current_pos];
//                 uint16_t path_list_len = one_path_lists[(path_list_idx * 5) + 4];

//                 for (int i = 0; i < path_list_len; i++) {
//                     uint16_t path_idx = one_path_lists[(path_list_idx * 5) + i];
//                     OnePath path = one_paths[path_idx];

//                     if (current_data.backtrack_bb & path.backtrack_bb) {
//                         continue;

//                     }

//                     uint8_t end_pos = path.p2;
//                     uint64_t end_bit = (uint64_t)1 << end_pos;

//                     if (end_positions[current_data.active_line_idx] & end_bit) {
//                         continue;

//                     }

//                     if (end_bit & BIT_36_MASK) {
//                         if (player == 0) {
//                             continue;
//                         }
//                         local_end_positions |= end_bit;

//                     } else if (end_bit & BIT_37_MASK) {
//                         if (player == 1) {
//                             continue;
//                         }
//                         local_end_positions |= end_bit;

//                     }

//                     uint64_t end_piece = piece_at(current_state, end_pos);
//                     if (end_piece != 0) {
//                         if ((current_data.banned_bb & end_bit) == 0) {
//                             uint64_t new_banned_bb = current_data.banned_bb ^ end_bit;
//                             uint64_t new_backtrack_bb = current_data.backtrack_bb ^ path.backtrack_bb;

//                             local_pickup_positions |= end_bit;

//                             StackData new_data = {
//                                 new_banned_bb,
//                                 new_backtrack_bb,
//                                 current_data.active_line_idx,
//                                 end_pos,
//                                 (uint8_t)end_piece,
//                             };

//                             push(stack, &stack_height, block_id, new_data);

//                         }

//                     } else {
//                         local_end_positions |= end_bit;

//                     }
                        
//                 }

//             } else if (current_data.current_piece == 2) { // TWOS
//                 uint64_t intercept_bb = get_piece_bb(current_state) & ALL_INTERCEPTS[current_data.current_pos];

//                 uint16_t path_list_idx = two_map[(current_data.current_pos * 29) + (intercept_bb % 29)];
//                 uint16_t path_list_len = two_path_lists[(path_list_idx * 13) + 12];

//                 for (int i = 0; i < path_list_len; i++) {
//                     uint16_t path_idx = two_path_lists[(path_list_idx * 13) + i];
//                     TwoPath path = two_paths[path_idx];

//                     if (current_data.backtrack_bb & path.backtrack_bb) {
//                         continue;

//                     }

//                     uint8_t end_pos = path.p3;
//                     uint64_t end_bit = (uint64_t)1 << end_pos;

//                     if (end_positions[current_data.active_line_idx] & end_bit) {
//                         continue;

//                     }

//                     if (end_bit & BIT_36_MASK) {
//                         if (player == 0) {
//                             continue;
//                         }
//                         local_end_positions |= end_bit;
                        
//                     } else if (end_bit & BIT_37_MASK) {
//                         if (player == 1) {
//                             continue;
//                         }
//                         local_end_positions |= end_bit;

//                     }

//                     uint64_t end_piece = piece_at(current_state, end_pos);
//                     if (end_piece != 0) {
//                         if ((current_data.banned_bb & end_bit) == 0) {
//                             uint64_t new_banned_bb = current_data.banned_bb ^ end_bit;
//                             uint64_t new_backtrack_bb = current_data.backtrack_bb ^ path.backtrack_bb;

//                             local_pickup_positions |= end_bit;

//                             StackData new_data = {
//                                 new_banned_bb,
//                                 new_backtrack_bb,
//                                 current_data.active_line_idx,
//                                 end_pos,
//                                 (uint8_t)end_piece,
//                             };

//                             push(stack, &stack_height, block_id, new_data);

//                         }

//                     } else {
//                         local_end_positions |= end_bit;

//                     }

//                 }

//             } else if (current_data.current_piece == 3) {
//                 uint64_t intercept_bb = get_piece_bb(current_state) & ALL_INTERCEPTS[current_data.current_pos + 36];

//                 uint16_t path_list_idx = three_map[(current_data.current_pos * 11007) + (intercept_bb % 11007)];
//                 uint16_t path_list_len = three_path_lists[(path_list_idx * 36) + 35];

//                 for (int i = 0; i < path_list_len; i++) {
//                     uint16_t path_idx = three_path_lists[(path_list_idx * 36) + i];
//                     ThreePath path = three_paths[path_idx];

//                     if (current_data.backtrack_bb & path.backtrack_bb) {
//                         continue;

//                     }

//                     uint8_t end_pos = path.p4;
//                     uint64_t end_bit = (uint64_t)1 << end_pos;

//                     if (end_positions[current_data.active_line_idx] & end_bit) {
//                         continue;

//                     }

//                     if (end_bit & BIT_36_MASK) {
//                         if (player == 0) {
//                             continue;
//                         }
//                         local_end_positions |= end_bit;
       
//                     } else if (end_bit & BIT_37_MASK) {
//                         if (player == 1) {
//                             continue;
//                         }
//                         local_end_positions |= end_bit;
               
//                     }
                        
//                     uint64_t end_piece = piece_at(current_state, end_pos);
//                     if (end_piece != 0) {
//                         if ((current_data.banned_bb & end_bit) == 0) {
//                             uint64_t new_banned_bb = current_data.banned_bb ^ end_bit;
//                             uint64_t new_backtrack_bb = current_data.backtrack_bb ^ path.backtrack_bb;
                            
//                             local_pickup_positions |= end_bit;

//                             StackData new_data = {
//                                 new_banned_bb,
//                                 new_backtrack_bb,
//                                 current_data.active_line_idx,
//                                 end_pos,
//                                 (uint8_t)end_piece,
//                             };

//                             push(stack, &stack_height, block_id, new_data);
                            
//                         }

//                     } else {
//                         local_end_positions |= end_bit;

//                     }

//                 } 

//             }

//             // Sync Data
//             if (current_data.current_piece != 0) {
//                 atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
//                 atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

//             }
            
//         }

//         __syncthreads(); // Sync threads

//         // Exit condition
//         if (stack_height == 0) {
//             break;

//         }

//     }

//     __syncthreads(); // Sync threads

//     // Store results
//     if (thread_id == 0) {
//         GenResult result;
//         for (int i = 0; i < 6; i++) {
//             result.end_positions[i] = end_positions[i];
//             result.pickup_positions[i] = pickup_positions[i];

//         }
//         result.drop_positions = drop_positions;

//         out_data[block_id] = result;

//     }

// }
