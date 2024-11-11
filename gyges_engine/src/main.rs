extern crate gyges_engine;

// use gyges_engine::ugi::Ugi;

// pub fn main() {
//     Ugi::new().start();

// }

// =================================== BLOCKING MOVE GEN ===================================

use cust::stream;
use gyges::moves::movegen_consts::*;
use gyges_engine::gpu_reach_consts;
use gyges_engine::matrix::*;
use core::num;
use std::{f32::consts::E, io::Write, mem, thread, time::Duration};

use gyges::{board::{self, TEST_BOARD}, core::masks::RANKS, moves::{gen_reach_consts::{UNIQUE_ONE_REACHS, UNIQUE_THREE_REACHS, UNIQUE_TWO_REACHS}, movegen::{has_threat, piece_control_sqs, valid_moves, valid_threat_count}, movegen_consts::ALL_THREE_INTERCEPTS, Move}, BitBoard, BoardState, Piece, Player, BENCH_BOARD, SQ, STARTING_BOARD};


use cust::{error::CudaError, memory::{AsyncCopyDestination, LockedBuffer}};
use cust::prelude::*;

fn main() -> Result<(), CudaError> {
    // Intialize CUDA
    let _ctx = cust::quick_init().expect("Failed to quick initialize CUDA");
    let device = Device::get_device(0).expect("Failed to retrieve device");
    let name = device.name().expect("Failed to retrieve device name");
    let mem = device.total_memory().expect("Failed to retrieve device memory");

    println!("CUDA Device: {}", name);
    println!("Total VRAM: {}mb", mem / 1_000_000);
    println!("");

    // Initialize board
    let mut board = BoardState::from(TEST_BOARD);
    // let mut board = BoardState::from([
    //     0, 0, 2, 1, 0, 2,
    //     0, 0, 0, 0, 0, 0,
    //     3, 0, 2, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0,
    //     3, 0, 1, 0, 0, 1,
    //     0, 0
    // ]);
    let player = Player::One;
    println!("{}", board);

    let mut mv_gen = BlockingMoveGen::new().expect("Failed to initialize blocking move generator");

    let norm = unsafe { gen_all_blocking_moves(&mut board, player) };
    let batch = unsafe { mv_gen.gen_all(&mut board, player) };

    println!("Norm: {}", norm.len());
    println!("Batch: {}", batch.len());
    println!("");

    unsafe {
        // MAIN BENCHMARKS

        // NEW
        let mut num = 0;
        let start = std::time::Instant::now();
        for _ in 0..4000 {
            let moves = mv_gen.gen_all(&mut board, player);
            num += moves.len();

        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("New Elapsed: {:?}, {}", elapsed, num);

        // Native
        let mut num = 0;
        let start = std::time::Instant::now();
        for _ in 0..4000 {
            let moves = valid_moves(&mut board, player).moves(&board);
            let mut pruned = vec![];
            for mv in moves.iter() {
                let mut new_board = board.clone().make_move(mv);
                let has_threat = has_threat(&mut new_board, player.other());
                if !has_threat {
                    pruned.push(mv);
                    num += 1;
                }

            }

        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("Native Elapsed: {:?}, {}", elapsed, num);

    }

    Ok(())
        
}

pub struct BlockingMoveGen {
    // Lookup tables
    one_reach_d: DeviceBuffer<u64>,
    two_reach_d: DeviceBuffer<u64>,
    three_reach_d: DeviceBuffer<u64>,

    all_two_intercepts_d: DeviceBuffer<u64>,
    all_three_intercepts_d: DeviceBuffer<u64>,

    one_map_d: DeviceBuffer<[u8; 1]>,
    two_map_d: DeviceBuffer<[u16; 29]>,
    three_map_d: DeviceBuffer<[u16; 11007]>,

    // GPU and Host buffers
    board_input_buffer_d: DeviceBuffer<u8>,
    board_bb_input_buffer_d: DeviceBuffer<u64>,

    board_input_buffer_h: LockedBuffer<u8>,
    board_bb_input_buffer_h: LockedBuffer<u64>,

    final_d: DeviceBuffer<f32>,
    final_h: LockedBuffer<f32>,

    // CUDA
    stream: Stream,
    module: Module

}

impl BlockingMoveGen {
    pub fn new() -> Result<Self, CudaError> {
        // Create a stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?; 

        // Load kernels from PTX
        let ptx = std::fs::read_to_string("kernels.ptx").expect("Failed to read PTX file");
        let module = Module::from_ptx(ptx, &[])?;

        // Load lookup tables into GPU
        let one_reach_d = DeviceBuffer::from_slice(&gpu_reach_consts::UNIQUE_ONE_REACHS)?;
        let two_reach_d = DeviceBuffer::from_slice(&gpu_reach_consts::UNIQUE_TWO_REACHS)?;
        let three_reach_d = DeviceBuffer::from_slice(&gpu_reach_consts::UNIQUE_THREE_REACHS)?;

        let all_two_intercepts_d = DeviceBuffer::from_slice(&gpu_reach_consts::ALL_TWO_INTERCEPTS)?;
        let all_three_intercepts_d = DeviceBuffer::from_slice(&gpu_reach_consts::ALL_THREE_INTERCEPTS)?;

        let one_map_d: DeviceBuffer<[u8; 1]> = DeviceBuffer::from_slice(&gpu_reach_consts::ONE_MAP)?;
        let two_map_d = DeviceBuffer::from_slice(&gpu_reach_consts::TWO_MAP)?;
        let three_map_d = DeviceBuffer::from_slice(&gpu_reach_consts::THREE_MAP)?;

        // Board and BB input buffers
        let board_input_buffer_d = DeviceBuffer::from_slice(&vec![0; 38 * 1000])?;
        let board_bb_input_buffer_d = DeviceBuffer::from_slice(&vec![0; 1000])?;

        let board_input_buffer_h = LockedBuffer::new(&0, 38 * 1000)?;
        let board_bb_input_buffer_h = LockedBuffer::new(&0, 1000)?;

        // Results
        let final_d =  DeviceBuffer::from_slice(&vec![0.0; 1000])?;
        let final_h = LockedBuffer::new(&0.0, 1000)?;


        // Create the BMG instance  
        Ok(Self {
            one_reach_d,
            two_reach_d,
            three_reach_d,

            all_two_intercepts_d,
            all_three_intercepts_d,

            one_map_d,
            two_map_d,
            three_map_d,

            board_input_buffer_d,
            board_bb_input_buffer_d,

            board_input_buffer_h,
            board_bb_input_buffer_h,

            final_d,
            final_h,

            stream,
            module

        })

    }

    pub fn batch_check(&mut self, num_boards: u32) -> Result<(), CudaError> {
        // Constants
        let stream = &self.stream;
            
        // Copy data to the GPU
        self.board_input_buffer_d.copy_from(&self.board_input_buffer_h)?;
        self.board_bb_input_buffer_d.copy_from(&self.board_bb_input_buffer_h)?;

        // Load kernel
        let unified_kernel = self.module.get_function("unified_kernel")?;

        // Start
        unsafe{
            launch!(unified_kernel<<<(num_boards, 1, 1), (38, 1, 1), 0, stream>>>(
                self.board_input_buffer_d.as_device_ptr(),
                self.board_bb_input_buffer_d.as_device_ptr(),
                self.final_d.as_device_ptr(),

                // Lookup tables
                self.one_reach_d.as_device_ptr(),
                self.two_reach_d.as_device_ptr(),
                self.three_reach_d.as_device_ptr(),
                self.all_two_intercepts_d.as_device_ptr(),
                self.all_three_intercepts_d.as_device_ptr(),
                self.one_map_d.as_device_ptr(),
                self.two_map_d.as_device_ptr(),
                self.three_map_d.as_device_ptr()

            ))?;

        }

        // Copy final results back to the host
        self.final_d.copy_to(&mut self.final_h)?;

        Ok(())

        
    }

    pub fn gen_all(&mut self, board: &mut BoardState, player: Player) -> Vec<Move> {
        let mut moves = Vec::with_capacity(600);
        let mut idx = 0;

        // Calculate boards
        let mut piece_control: [BitBoard; 6] = unsafe { piece_control_sqs(board, player) };

        let active_lines = board.get_active_lines();
        let drops = board.get_drops(active_lines, player).get_data();

        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            let starting_piece = board.piece_at(starting_sq);

            if starting_piece == Piece::None {
                continue;
            }

            // BRUTE FORCE: Try all valid placements
            for block_pos in piece_control[x].get_data() {
                let block_sq = SQ(block_pos as u8);
                let block_piece = board.piece_at(block_sq);

                match block_piece {
                    Piece::None => { // Empty square: Try placing piece
                        let mut new_board = board.clone();
                        new_board.place(starting_piece, block_sq);
                        new_board.remove(starting_sq);

                        let new_piece_bb = board.piece_bb ^ (starting_sq.bit() | block_sq.bit());

                        // Store
                        for (i, x) in new_board.data.iter().enumerate() {
                            let input_idx = 38 * idx + i;
                            self.board_input_buffer_h[input_idx] = *x as u8;
                        }
                        self.board_bb_input_buffer_h[idx] = new_piece_bb.0;
                        moves.push(Move::new([(Piece::None, starting_sq), (starting_piece, block_sq), (Piece::None, SQ::NONE)], gyges::moves::MoveType::Bounce));

                        idx += 1;

                    },
                    Piece::One | Piece::Two | Piece::Three => { // Occupied square: Try replacement
                        for empty_pos in drops.iter() {
                            let empty_sq = SQ(*empty_pos as u8);

                            let mut new_board = board.clone();
                            new_board.place(starting_piece, block_sq);
                            new_board.place(block_piece, empty_sq);
                            new_board.remove(starting_sq);

                            let new_piece_bb = board.piece_bb ^ (starting_sq.bit() | empty_sq.bit());

                            // Store    
                            for (i, x) in new_board.data.iter().enumerate() {
                                let input_idx = 38 * idx + i;
                                self.board_input_buffer_h[input_idx] = *x as u8;
                            }
                            self.board_bb_input_buffer_h[idx] = new_piece_bb.0;
                            moves.push(Move::new([(Piece::None, starting_sq), (starting_piece, block_sq), (block_piece, empty_sq)], gyges::moves::MoveType::Bounce));

                            idx += 1;

                        }

                    }

                }

            }

        }

        // Run on GPU
        let board_count = idx;
        self.batch_check(board_count as u32).expect("Failed to batch check routes");

        // Check and filter blocking moves
        let mut blocking_moves = vec![];
        for i in 0..board_count {
            if self.final_h[i] == 0.0 {
                blocking_moves.push(moves[i]);

            }

        }

        blocking_moves

    }

}


// ================== ROUTE CHECKING FUNCTIONS ==================

/// Checks if there is a theorical route between two nodes in a 
/// binary adjacency matrix in up to 'max_pow' steps.
/// 
/// If True is returned, there is a chance that the route is a false positive, and
/// it backtracked.
/// 
/// If False is returned, there is no route between the two nodes garanteed.
pub unsafe fn check_route(adj: &BinaryMatrix, max_pow: usize, start_idxs: &Vec<usize>, end: usize) -> bool {
    let mut current_power_matrix = adj.clone();

    for _ in 0..max_pow {
        current_power_matrix = simd_binary_matrix_multiply(&current_power_matrix, &adj);

        // Get full col of the matrix [end] and check if it has any 1s
            for i in start_idxs.iter() {
            if (current_power_matrix[*i] & (1 << end)) != 0 { // ROUTE FOUND
                return true;
            }
        }
        
    }

    false

}


pub unsafe fn gen_all_blocking_moves(board: &mut BoardState, player: Player) -> Vec<Move> {
    let mut moves = Vec::with_capacity(1000);
    
    let active_lines = board.get_active_lines();
    let mut active_pieces = board.piece_bb & RANKS[active_lines[player.other() as usize] as usize];
    let active_piece_idxs = active_pieces.get_data();

    let piece_control: [BitBoard; 6] = unsafe { piece_control_sqs(board, player) };

    let active_lines = board.get_active_lines();
    let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);

    for x in 0..6 {
        let starting_sq = active_line_sq + x;
        let starting_piece = board.piece_at(starting_sq);

        if starting_piece == Piece::None {
            continue;
        }

        // BRUTE FORCE: Try all valid placements
        for block_pos in piece_control[x].clone().get_data() {
            let block_sq = SQ(block_pos as u8);
            let block_piece = board.piece_at(block_sq);

            match block_piece {
                Piece::None => { // Empty square: Try placing piece
                    board.place(starting_piece, block_sq);
                    board.remove(starting_sq);
                    board.piece_bb ^= starting_sq.bit() | block_sq.bit();

                    // Check threat
                    if !check_route(&unsafe{adj_matrix(board)}, 8, &active_piece_idxs, 36) {
                        let mv = Move::new([(Piece::None, starting_sq), (starting_piece, block_sq), (Piece::None, SQ::NONE)], gyges::moves::MoveType::Bounce);
                        moves.push(mv);
                    }

                    board.remove(block_sq);
                    board.place(starting_piece, starting_sq);
                    board.piece_bb ^= starting_sq.bit() | block_sq.bit();

                },
                Piece::One | Piece::Two | Piece::Three => { // Occupied square: Try replacement     ;    !!!!!missing starting position!!!!!! - todo
                    let mut drops = board.get_drops(active_lines, player);
                    for empty_pos in drops.get_data() {
                        let empty_sq = SQ(empty_pos as u8);

                        board.place(starting_piece, block_sq);
                        board.place(block_piece, empty_sq);
                        board.remove(starting_sq);
                        board.piece_bb ^= starting_sq.bit() | empty_sq.bit();

                        if !check_route(&unsafe{adj_matrix(board)}, 8, &active_piece_idxs, 36) {
                            let mv = Move::new([(Piece::None, starting_sq), (starting_piece, block_sq), (block_piece, empty_sq)], gyges::moves::MoveType::Drop);
                            moves.push(mv);
                        }

                        board.remove(empty_sq);
                        board.place(block_piece, block_sq);
                        board.place(starting_piece, starting_sq);
                        board.piece_bb ^= starting_sq.bit() | empty_sq.bit();

                    }

                }

            }

        }

    }

    moves

}

// ================== ADJACENCY MATRIX FUNCTIONS ==================

/// Generates a binary adjacency matrix
#[inline(always)]
pub unsafe fn adj_matrix(board: &mut BoardState) -> BinaryMatrix {
    // Initialize the binary adjacency matrix, with each u64 representing a row
    let mut adj = [0; 38];

    // Loop through each square (node) in the board
    for i in 0..36 {
        let sq = SQ(i as u8);
        let piece = board.piece_at(sq);
        if piece == Piece::None {
            continue;
        }

        // Get reachable positions for this piece
        let reach = fast_reach(board, sq) & (board.piece_bb | (SQ::P1_GOAL.bit() | SQ::P2_GOAL.bit()));
        adj[i] = reach.0;

    }   

    adj

}

/// Generates a binary piece adjacency matrix
#[inline(always)]
pub unsafe fn piece_adj_matrix(board: &mut BoardState) -> [u16; 14] {
    // Create a map of squares to piece IDs
    let mut piece_id_map = std::collections::HashMap::new();
    let mut curr_id = 0;
    for i in 0..38 {
        let sq = SQ(i as u8);
        let piece = board.piece_at(sq);
        
        if sq == SQ::P1_GOAL || sq == SQ::P2_GOAL { // Goals get an ID
            piece_id_map.insert(sq.0, curr_id);

            curr_id += 1;

        }
        
        if piece == Piece::None {
            continue;

        }

        // Add id to piece_id_map
        piece_id_map.insert(sq.0, curr_id);

        curr_id += 1;

    }

    // Initialize a 14x14 adjacency matrix
    let mut adj = [0u16; 14];

    // Precompute goal bits and piece bitboard
    let goal_bits = SQ::P1_GOAL.bit() | SQ::P2_GOAL.bit();
    let reachable_mask = board.piece_bb | goal_bits;

    // Loop through each square (node) in the board
    for i in 0..36 {
        let sq = SQ(i as u8);
        let piece = board.piece_at(sq);

        if piece == Piece::None {
            continue;

        }

        if let Some(curr_id) = piece_id_map.get(&sq.0) {
            // Get reachable positions for this piece, filtered by reachable_mask
            let mut reach = fast_reach(board, sq) & reachable_mask;

            // Map each reachable board position to its piece ID
            for pos in reach.get_data() {
                let reach_sq = SQ(pos as u8);
                if let Some(&reach_id) = piece_id_map.get(&reach_sq.0) {
                    // Set the bit in the adjacency matrix for the piece-to-piece connection
                    adj[*curr_id] |= 1 << reach_id;
                }

            }
        
        }

    }

    adj

}


/// Writes the calculated adj matrix into a pre-allocated buffer
#[inline(always)]
pub unsafe fn adj_matrix_flat(board: &mut BoardState, matrix_id: usize, buffer: *mut f32) {
    // Loop through each square (node) in the board
    for i in 0..36 {
        let sq = SQ(i as u8);
        let piece = board.piece_at(sq);
        if piece == Piece::None {
            continue;

        }

        // Get reachable positions for this piece
        let mut reach = fast_reach(board, sq) & (board.piece_bb | (SQ::P1_GOAL.bit() | SQ::P2_GOAL.bit()));
        for pos in reach.get_data() {
            let buffer_offset = (matrix_id * 1444) + (i * 38) + pos as usize;
            buffer.add(buffer_offset).write(1.0);

        }

    }   

}

// ================== HELPER FUNCTIONS ==================

/// Returns the squares reachable by a piece
pub unsafe fn reach(board: &mut BoardState, sq: SQ) -> BitBoard {
    let mut reachable = BitBoard::EMPTY;

    // Get contol of this piece
    match board.piece_at(sq) {
        Piece::One => {
            let valid_paths_idx = ONE_MAP.get_unchecked(sq.0 as usize).get_unchecked(0);
            let valid_paths = UNIQUE_ONE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..valid_paths[ONE_PATH_COUNT_IDX] {
                let path_idx = valid_paths[i as usize];
                let path = UNIQUE_ONE_PATHS.get_unchecked(path_idx as usize);

                let end = SQ(path.0[1]);
                let end_bit = end.bit();

                reachable |= end_bit;
                
            }

        },
        Piece::Two => {
            let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[sq.0 as usize];

            let valid_paths_idx = TWO_MAP.get_unchecked(sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
            let valid_paths = UNIQUE_TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..valid_paths[TWO_PATH_COUNT_IDX] {
                let path_idx = valid_paths[i as usize];
                let path = UNIQUE_TWO_PATHS.get_unchecked(path_idx as usize);

                let end = SQ(path.0[2]);
                let end_bit = end.bit();

                reachable |= end_bit;
            
            }

        },
        Piece::Three => {
            let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[sq.0 as usize];

            let valid_paths_idx = THREE_MAP.get_unchecked(sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
            let valid_paths = UNIQUE_THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

            for i in 0..valid_paths[THREE_PATH_COUNT_IDX] {
                let path_idx = valid_paths[i as usize];
                let path = UNIQUE_THREE_PATHS.get_unchecked(path_idx as usize);

                let end = SQ(path.0[3]);
                let end_bit = end.bit();

                reachable |= end_bit;

            }

        }
        _ => {}
        
    }

    reachable

}

/// Looksup the correct reach positions for a piece.
#[inline(always)]
pub unsafe fn fast_reach(board: &mut BoardState, sq: SQ) -> BitBoard {
    let piece: Piece = board.piece_at(sq);
    if piece == Piece::One {
        let valid_paths_idx = ONE_MAP.get_unchecked(sq.0 as usize).get_unchecked(0);
        return *UNIQUE_ONE_REACHS.get_unchecked(*valid_paths_idx as usize);

    } else if piece == Piece::Two {
        let intercept_bb = board.piece_bb & ALL_TWO_INTERCEPTS[sq.0 as usize];

        let valid_paths_idx = TWO_MAP.get_unchecked(sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 29);
        return *UNIQUE_TWO_REACHS.get_unchecked(*valid_paths_idx as usize);

    } else if piece == Piece::Three {
        let intercept_bb = board.piece_bb & ALL_THREE_INTERCEPTS[sq.0 as usize];
        
        let valid_paths_idx = THREE_MAP.get_unchecked(sq.0 as usize).get_unchecked(intercept_bb.0 as usize % 11007);
        return *UNIQUE_THREE_REACHS.get_unchecked(*valid_paths_idx as usize);

    } else {
        return BitBoard::EMPTY;
    }

}

// ======================================================