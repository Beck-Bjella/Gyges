#include <stdint.h>
#include <cuda_runtime.h>
#include <cstdio>

// ==================== CONSTANTS ====================

// Masks
#define BIT_36_MASK (1ULL << 36)
#define BIT_37_MASK (1ULL << 37)
#define GOALS_MASK (BIT_36_MASK | BIT_37_MASK)

// Intercept table
__constant__ uint64_t ALL_INTERCEPTS[72] {
    0b00000000000000000000000000000000000000000000000000000001000010ULL, // TWOS: (0-35)
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
    0b0000000000000000000000000000000000000000000000000001000011000110ULL, // THREES: (36-71)
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

// Backzones
__constant__ uint64_t BACK_ZONES[2][6] {
    {
        0b000000000000000000000000000000000000ULL,
        0b000000000000000000000000000000111111ULL,
        0b000000000000000000000000111111111111ULL,
        0b000000000000000000111111111111111111ULL,
        0b000000000000111111111111111111111111ULL,
        0b000000111111111111111111111111111111ULL 
    },
    {
        0b111111111111111111111111111111000000ULL,
        0b111111111111111111111111000000000000ULL,
        0b111111111111111111000000000000000000ULL,
        0b111111111111000000000000000000000000ULL,
        0b111111000000000000000000000000000000ULL,
        0b000000000000000000000000000000000000ULL
    }
    
};


// Stack
const uint32_t MAX_STACK_SIZE = 10000;
#define STACKIDX(block_id, type) (block_id * MAX_STACK_SIZE * 3) + (type * MAX_STACK_SIZE)
#define HEIGHTIDX(block_id, type) (block_id * 3) + type

// ==================== STRUCTS ====================

struct GenRequest {
    uint64_t state;
    uint64_t active_bb;
    uint8_t p1_activeline;
    uint8_t p2_activeline;
    uint8_t player;
    uint8_t flag;

};

struct GenResult {
    uint64_t end_positions[6];
    uint64_t pickup_positions[6];
    uint64_t drop_positions;
    uint64_t has_threat;
    
};

struct StackData {
    uint64_t banned_bb;
    uint64_t backtrack_bb;
    uint8_t active_line_idx;
    uint8_t current_pos;
    uint8_t current_piece;

};

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

// ==================== DEVICE MEMORY ====================

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

// ====================  NEW STUFF ====================  

// Remove an existing piece type and returns the new state
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

// ==================== MOVEGEN ====================

// Pushs to a stack w/o any overflow handling
__device__ void push(StackData* stack, uint32_t* stack_height, uint32_t block_id, uint32_t type, StackData data) {
    uint32_t old_height = atomicAdd(&stack_height[HEIGHTIDX(block_id, type)], 1);
    if (old_height >= MAX_STACK_SIZE) {
        printf("Error: Stack overflow at block %u, type %u, height %u\n", block_id, type, old_height);
        atomicSub(stack_height, 1); // revert
        return; 
    }

    stack[STACKIDX(block_id, type) + old_height] = data;

}

// Pops from a stack w/o any underflow handling
__device__ StackData pop(StackData* stack, uint32_t* stack_height, uint32_t block_id, uint32_t type) {
    uint32_t old_height = atomicSub(&stack_height[HEIGHTIDX(block_id, type)], 1) - 1;
    if ((int32_t)old_height < 0) {
        printf("Error: Stack underflow at block %u, type %u\n", block_id, type);
        atomicAdd(&stack_height[HEIGHTIDX(block_id, type)], 1);
        return {0ULL, 0ULL, 0ULL, 0, 0};

    }
    
    StackData val = stack[STACKIDX(block_id, type) + old_height];

    return val;

}

// ====== VALID MOVES ======

__device__ void valid_moves_one(
    StackData* stack, 
    uint32_t* stack_height, 
    StackData current_data,
    uint64_t starting_state,
    uint64_t end_positions,
    uint64_t* local_has_threat,
    uint64_t* local_end_positions, 
    uint64_t* local_pickup_positions,
    uint64_t block_id,
    uint8_t player
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
        bool invalid_player = ((end_bit & BIT_36_MASK) && (player == 0)) || ((end_bit & BIT_37_MASK) && (player == 1));

        if (backtrack_conflict || end_pos_banned || invalid_player) {
            continue;

        }

        uint64_t end_piece = piece_at(starting_state, end_pos);
        bool is_empty = (end_piece == 0);

        if (is_empty) {
            *local_end_positions |= end_bit;
            if ((end_bit & GOALS_MASK) != 0) {
                *local_has_threat = 1;
            }

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
                stack_height,
                block_id, 
                stack_type, 
                new_data
            );

        }
            
    }

}

__device__ void valid_moves_two(
    StackData* stack, 
    uint32_t* stack_height,
    StackData current_data, 
    uint64_t starting_state,
    uint64_t end_positions,
    uint64_t* local_has_threat,
    uint64_t* local_end_positions, 
    uint64_t* local_pickup_positions,
    uint64_t block_id,
    uint8_t player
) {
    uint64_t intercept_bb = get_piece_bb(starting_state) & ALL_INTERCEPTS[current_data.current_pos];

    uint16_t path_list_idx = two_map[(current_data.current_pos * 29) + (intercept_bb % 29)];
    uint16_t path_list_len = two_path_lists[(path_list_idx * 13) + 12];

    for (int i = 0; i < path_list_len; i++) {
        uint16_t path_idx = two_path_lists[(path_list_idx * 13) + i];
        TwoPath path = two_paths[path_idx];

        uint8_t end_pos = path.p3;
        uint64_t end_bit = (uint64_t)1 << end_pos;

        uint64_t end_pos_banned = (current_data.banned_bb | end_positions) & end_bit;
        uint64_t backtrack_conflict = current_data.backtrack_bb & path.backtrack_bb; 
        bool invalid_player = ((end_bit & BIT_36_MASK) && (player == 0)) || ((end_bit & BIT_37_MASK) && (player == 1));

        if (backtrack_conflict || end_pos_banned || invalid_player) {
            continue;

        }
         
            
        uint64_t end_piece = piece_at(starting_state, end_pos);
        bool is_empty = (end_piece == 0);

        if (is_empty) {
            *local_end_positions |= end_bit;
            if ((end_bit & GOALS_MASK) != 0) {
                *local_has_threat = 1;
            }

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
                stack_height,
                block_id, 
                stack_type, 
                new_data
            );

        }

    }

}

__device__ void valid_moves_three(
    StackData* stack, 
    uint32_t* stack_height,
    StackData current_data, 
    uint64_t starting_state,
    uint64_t end_positions,
    uint64_t* local_has_threat,
    uint64_t* local_end_positions, 
    uint64_t* local_pickup_positions,
    uint64_t block_id,
    uint8_t player
) {
    uint64_t intercept_bb = get_piece_bb(starting_state) & ALL_INTERCEPTS[current_data.current_pos + 36];

    uint16_t path_list_idx = three_map[(current_data.current_pos * 11007) + (intercept_bb % 11007)];
    uint16_t path_list_len = three_path_lists[(path_list_idx * 36) + 35];

    for (int i = 0; i < path_list_len; i++) {
        uint16_t path_idx = three_path_lists[(path_list_idx * 36) + i];
        ThreePath path = three_paths[path_idx];

        uint8_t end_pos = path.p4;
        uint64_t end_bit = (uint64_t)1 << end_pos;

        uint64_t end_pos_banned = (current_data.banned_bb | end_positions) & end_bit;
        uint64_t backtrack_conflict = current_data.backtrack_bb & path.backtrack_bb; 
        bool invalid_player = ((end_bit & BIT_36_MASK) && (player == 0)) || ((end_bit & BIT_37_MASK) && (player == 1));

        if (backtrack_conflict || end_pos_banned || invalid_player) {
            continue;

        }

        uint64_t end_piece = piece_at(starting_state, end_pos);
        bool is_empty = (end_piece == 0);

        if (is_empty) {
            *local_end_positions |= end_bit;
            if ((end_bit & GOALS_MASK) != 0) {
                *local_has_threat = 1;
            }

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
                stack_height,
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
    StackData* stack,
    uint32_t* stack_height
    
) {
    uint32_t block_id = blockIdx.x;       // Each block processes one generation request
    uint32_t thread_id = threadIdx.x;     // 
    uint32_t warp_id = threadIdx.x / 32;  //
    uint32_t lane_id = threadIdx.x % 32;  // 

    // Init Shared Memory
    __shared__ uint64_t end_positions[6];       // End positions
    __shared__ uint64_t pickup_positions[6];    // Pickup positions
    __shared__ uint64_t has_threat;                 // Threat flag

    __shared__ uint64_t starting_states[6];     // Starting states
    __shared__ GenRequest request;              // Generation request 
    __shared__ uint8_t player;                  // Player to move
    __shared__ uint8_t flag;                    // Generation flag
    if (thread_id == 0) {
        request = in_data[block_id];
        player = request.player;
        flag = request.flag;

        has_threat = 0;

        // Get drops
        uint8_t other_player = player == 0 ? 1 : 0;
        uint8_t other_activeline = player == 0 ? request.p2_activeline : request.p1_activeline;
        
        uint64_t piece_bb = request.state & 0b111111111111111111111111111111111111ULL;
        uint64_t other_backzone = BACK_ZONES[other_player][other_activeline];
        uint64_t drop_bb = ~piece_bb & (0b111111111111111111111111111111111111ULL ^ other_backzone);

        out_data[block_id].drop_positions = drop_bb;

    }

    __syncthreads();
    
    // Setup
    if (thread_id < 6) {
        end_positions[thread_id] = 0ULL;
        pickup_positions[thread_id] = 0ULL;

        uint8_t current_pos;
        if (player == 0) {
            current_pos = request.p1_activeline * 6 + thread_id;
        } else {
            current_pos = request.p2_activeline * 6 + thread_id;

        }

        uint64_t active_bb = request.active_bb;
        if ((1ULL << current_pos) & active_bb) {
            uint64_t new_state = remove_piece(request.state, current_pos);
            starting_states[thread_id] = new_state;

            // Create init stack data
            uint8_t piece = piece_at(request.state, current_pos);
            StackData data = {
                0ULL,
                0ULL,
                (uint8_t)thread_id,
                current_pos, 
                piece,
            };

            // Push to stack
            int stack_type = piece - 1;
            push(
                stack, 
                stack_height, 
                block_id, 
                stack_type, 
                data
            );

        }

    }

    __syncthreads();

    // MAIN PROCESSING LOOP
    while (true) {
        // Sync threads
        __threadfence();
        __syncthreads();

        // Current stack heights
        uint32_t one_stack_height = stack_height[HEIGHTIDX(block_id, 0)];
        uint32_t two_stack_height = stack_height[HEIGHTIDX(block_id, 1)];
        uint32_t three_stack_height = stack_height[HEIGHTIDX(block_id, 2)];
        if ((one_stack_height == 0 && two_stack_height == 0 && three_stack_height == 0)) {
            break;

        }

        // Process stack
        switch(warp_id) { 
            case 0: {
                if (lane_id < one_stack_height) {
                    StackData current_data = pop(stack, stack_height, block_id, 0);
                    uint64_t starting_state = starting_states[current_data.active_line_idx];

                    uint64_t local_has_threat = 0ULL;
                    uint64_t local_end_positions = 0ULL;
                    uint64_t local_pickup_positions = 0ULL;

                    valid_moves_one(
                        stack,
                        stack_height,
                        current_data,
                        starting_state,
                        end_positions[current_data.active_line_idx],
                        &local_has_threat,
                        &local_end_positions,
                        &local_pickup_positions,
                        block_id,
                        player
                    );

                    if (!has_threat && local_has_threat) {
                        atomicOr(&has_threat, 1);

                    }
                    
                    atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
                    atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

                }

                break;

            }
            case 1: {
                if (lane_id < two_stack_height) {
                    StackData current_data = pop(stack, stack_height, block_id, 1);
                    uint64_t starting_state = starting_states[current_data.active_line_idx];

                    uint64_t local_has_threat = 0ULL;
                    uint64_t local_end_positions = 0ULL;
                    uint64_t local_pickup_positions = 0ULL;

                    valid_moves_two(
                        stack,
                        stack_height,
                        current_data,
                        starting_state,
                        end_positions[current_data.active_line_idx],
                        &local_has_threat,
                        &local_end_positions,
                        &local_pickup_positions,
                        block_id,
                        player
                    );

                    if (!has_threat && local_has_threat) {
                        atomicOr(&has_threat, 1);

                    }

                    atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
                    atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

                }

                break;

            }
            case 2: {
                if (lane_id < three_stack_height) {
                    StackData current_data = pop(stack, stack_height, block_id, 2);
                    uint64_t starting_state = starting_states[current_data.active_line_idx];

                    uint64_t local_has_threat = 0ULL;
                    uint64_t local_end_positions = 0ULL;
                    uint64_t local_pickup_positions = 0ULL;

                    valid_moves_three(
                        stack,
                        stack_height,
                        current_data,
                        starting_state,
                        end_positions[current_data.active_line_idx],
                        &local_has_threat,
                        &local_end_positions,
                        &local_pickup_positions,
                        block_id,
                        player
                    );

                    if (!has_threat && local_has_threat) {
                        atomicOr(&has_threat, 1);

                    }

                    atomicOr(&end_positions[current_data.active_line_idx], local_end_positions);
                    atomicOr(&pickup_positions[current_data.active_line_idx], local_pickup_positions);

                }

                break;
            
            }

        }

    }

    __syncthreads(); 

    // Store results
    if (thread_id < 6) {
        out_data[block_id].end_positions[thread_id] = end_positions[thread_id];
        out_data[block_id].pickup_positions[thread_id] = pickup_positions[thread_id];

        if (thread_id == 0) {
            out_data[block_id].has_threat = has_threat;

        }

    }

}
