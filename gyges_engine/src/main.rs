extern crate gyges_engine;

// use gyges_engine::ugi::Ugi;

// pub fn main() {
//     Ugi::new().start();

// }

// =================================== BLOCKING MOVE GEN ===================================

use cust::stream;
use gyges::moves::movegen_consts::*;
use gyges::BitBoard;
use gyges_engine::gpu_reach_consts;
use core::num;
use std::{f32::consts::E, io::Write, mem, thread, time::Duration};
use gyges::{board::{self, TEST_BOARD}, core::masks::RANKS, moves::{gen_reach_consts::{UNIQUE_ONE_REACHS, UNIQUE_THREE_REACHS, UNIQUE_TWO_REACHS}, movegen::{has_threat, piece_control_sqs, valid_moves, valid_threat_count}, movegen_consts::ALL_THREE_INTERCEPTS, Move}, BoardState, Piece, Player, BENCH_BOARD, SQ, STARTING_BOARD};


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
    // let mut board = BoardState::from(
    // [
    //     0, 0, 1, 0, 0, 0,
    //     0, 3, 3, 0, 2, 0,
    //     0, 2, 2, 0, 1, 0,
    //     0, 0, 0, 0, 3, 0,
    //     0, 0, 2, 3, 0, 0,
    //     0, 0, 1, 0, 2, 0,
    //     0, 0
    // ]);
    let player = Player::One;
    println!("{}", board);

    let mut mv_gen = BlockingMoveGen::new().expect("Failed to initialize blocking move generator");

    let new_batch = unsafe { mv_gen.new_gen_all(&mut board, player) };
    println!("New Batch: {}", new_batch.len());

    let new_batch = unsafe { mv_gen.new_gen_all(&mut board, player) };
    println!("New Batch: {}", new_batch.len());
    

    let batch = unsafe { mv_gen.gen_all(&mut board, player) };
    println!("Batch: {}", batch.len());
    println!("");

    unsafe {
        // MAIN BENCHMARKS

        // NEWNEW
        let mut num = 0;
        let start: std::time::Instant = std::time::Instant::now();
        for _ in 0..40000 {
            let moves = mv_gen.new_gen_all(&mut board, player);
            num += moves.len();

        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("NewNew Elapsed: {:?}, {}", elapsed, num);

        // NEW
        let mut num = 0;
        let start = std::time::Instant::now();
        for _ in 0..40000 {
            let moves = mv_gen.gen_all(&mut board, player);
            num += moves.len();

        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("New Elapsed: {:?}, {}", elapsed, num);

        // Native
        let mut num = 0;
        let start = std::time::Instant::now();
        for _ in 0..40000 {
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
    input_buffer_d: DeviceBuffer<u64>,
    input_buffer_h: LockedBuffer<u64>,

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
        let input_buffer_d: DeviceBuffer<u64> = DeviceBuffer::from_slice(&vec![0; 1000])?;
        let input_buffer_h = LockedBuffer::new(&0, 1000)?;

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

            input_buffer_d,
            input_buffer_h,

            final_d,
            final_h,
    
            stream,
            module

        })

    }

    pub fn new_batch_check(&mut self, num_boards: u32) -> Result<(), CudaError> {
        // Constants
        let stream = &self.stream;
            
        // Copy data to the GPU
        self.input_buffer_d.copy_from(&self.input_buffer_h)?;
        
        // Load kernel
        let new_unified_kernel = self.module.get_function("unified_kernel")?;

        // Start
        unsafe{
            launch!(new_unified_kernel<<<(num_boards, 1, 1), (38, 1, 1), 0, stream>>>(
                self.input_buffer_d.as_device_ptr(),
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

    pub fn new_gen_all(&mut self, board: &mut BoardState, player: Player) -> Vec<Move> {
        let mut bit_state = BitState::from(board);

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
                        let new_state = bit_state.make_bounce_mv(starting_sq.0 as u64, block_sq.0 as u64);
                        self.input_buffer_h[idx] = new_state.0;

                        moves.push(Move::new([(Piece::None, starting_sq), (starting_piece, block_sq), (Piece::None, SQ::NONE)], gyges::moves::MoveType::Bounce));

                        idx += 1;

                    },
                    _ => { // Occupied square: Try replacement
                        for empty_pos in drops.iter() {
                            let empty_sq = SQ(*empty_pos as u8);

                            let mut new_state = bit_state.make_drop_mv(starting_sq.0 as u64, block_sq.0 as u64, empty_sq.0 as u64);
                            self.input_buffer_h[idx] = new_state.0;
      
                            moves.push(Move::new([(Piece::None, starting_sq), (starting_piece, block_sq), (block_piece, empty_sq)], gyges::moves::MoveType::Bounce));

                            idx += 1;

                        }

                    }

                }

            }

        }

        // Run on GPU
        let board_count = idx;
        self.new_batch_check(board_count as u32).expect("Failed to batch check routes");

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


// ================== BitState Representation ==================

use std::{ops::{Not, BitOr, BitOrAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign}, fmt::Display};

// /// Bits 0..38 are the positions of the pieces on the board
// /// Bits 38..62 are the types of pieces in order.
// /// Bit 63 is the player to move.
// #[derive(Debug, Copy, Clone, PartialEq)]
// pub struct BitState(pub u64);

// impl BitState {
//     /// Creates a new BitState from an existing BoardState
//     pub fn from(board: &BoardState) -> Self {
//         let mut bb = BitState(0);
//         let mut piece_idx = 0;
//         for i in 0..38 {
//             let piece: u8 = board.data[i] as u8;
           
//             if piece != 3 {
//                 // Set the piece
//                 bb |= 1 << i;

//                 // Set the piece type
//                 bb |= ((piece + 1) as u64) << (38 + (piece_idx * 2));

//                 piece_idx += 1;

//             }

//         }

//         // Set the player to move
//         bb |= (board.player as u64) << 63;

//         bb

//     }

//     pub fn make_bounce_mv(&self, start_pos: u64, end_pos: u64) -> BitState {
//         // Copy the current state
//         let mut new_state = *self;

//         // Update the piece bitboard
//         new_state ^= 1 << start_pos;

//         // Starting piece
//         let starting_idx = self.piece_idx(start_pos as usize);
//         let starting_piece = self.piece_at(start_pos as usize) as u8;

//         new_state.remove_type(starting_idx);

//         // Ending piece
//         let ending_idx = new_state.piece_idx(end_pos as usize);
//         new_state.add_type(ending_idx, starting_piece);

//         new_state ^= 1 << end_pos;

//         new_state

//     }

//     pub fn make_drop_mv(&self, start_pos: u64, pickup_pos: u64, drop_pos: u64) -> BitState {
//         // Copy the current state
//         let mut new_state = *self;

      

//         // Starting piece
//         let starting_idx = self.piece_idx(start_pos as usize);
//         let starting_piece = self.piece_type(starting_idx) as u8;

//         new_state.remove_type(starting_idx);
//         new_state ^= 1 << start_pos;

//         // Pickup piece
//         let pickup_idx = new_state.piece_idx(pickup_pos as usize);
//         let pickup_piece = new_state.piece_type(pickup_idx) as u8;
//         new_state.set_type_data(pickup_idx, starting_piece);

        

//         // Drop piece   
//         let drop_idx = new_state.piece_idx(drop_pos as usize);
//         new_state.add_type(drop_idx, pickup_piece);
        
//         // Update the piece bitboard
//         let piece_bb_mask = 1 << drop_pos;
//         new_state ^= piece_bb_mask;

//         new_state

//     }

//     // ======================== MOVE MAKING HELPERS ========================

//     /// Removes a existing piece type
//     pub fn remove_type(&mut self, piece_idx: usize) {
//         // Save the data to the right of the removed piece    
//         let save_mask: u64 = !0 << (38 + (piece_idx * 2) + 2); 
//         let saved = self.0 & save_mask;
    
//         // Clear all data from the removed piece to the left
//         let clear_mask = !0 << (38 + (piece_idx * 2));     
//         let cleared = self.0 & !clear_mask; 

//         self.0 = (saved >> 2) | cleared;

//     }

//     /// Adds in a new piece type 
//     pub fn add_type(&mut self, piece_idx: usize, piece_type: u8) {
//         // Save the data to the right of the removed piece    
//         let save_mask: u64 = !0 << (38 + (piece_idx * 2)); 
//         let saved = self.0 & save_mask;
    
//         // Clear all data from the removed piece to the left
//         let clear_mask = !0 << (38 + (piece_idx * 2));     
//         let cleared = self.0 & !clear_mask; 

//         // Create the new data
//         let type_data = (piece_type as u64) << (38 + (piece_idx * 2));

//         self.0 = (saved << 2) | type_data | cleared;

//     }

//     /// Changes the type data at one of the type data slots
//     pub fn set_type_data(&mut self, piece_idx: usize, piece_type: u8) {
//         // Save the data to the right of the removed piece    
//         let save_mask: u64 = !0 << (38 + (piece_idx * 2) + 2); 
//         let saved = self.0 & save_mask;     

//         // Clear all data from the removed piece to the left
//         let clear_mask = !0 << (38 + (piece_idx * 2));     
//         let cleared = self.0 & !clear_mask; 

    
//         // Create the new data
//         let type_data = (piece_type as u64) << (38 + (piece_idx * 2));
//         let new_data = saved | type_data;

//         self.0 = new_data | cleared;

//     }

//     // ======================== GETTERS ========================

//     /// Gets a u8 of the piece at a square
//     /// 0 = None
//     /// 1 = One
//     /// 2 = Two
//     /// 3 = Three
//     pub const fn piece_at(&self, pos: usize) -> u8 {
//         // Empty
//         if (self.0 & (1 << pos)) == 0 {
//             return 0;

//         }

//         self.piece_type(self.piece_idx(pos))
        
//     }

//     /// Gets the piece bitboard
//     pub const fn piece_bb(&self) -> u64 {
//         self.0 & 0b111111_111111_111111_111111_111111_111111 // First 38 bits
    
//     }

//     /// Gets the idx of the piece at a position
//     pub const fn piece_idx(&self, pos: usize) -> usize {
//         // Pieces before the requested position
//         // (<< 26) : cuts off the piece type bits and the player bit
//         // (38 - pos) : cuts off the pieces after the requested position
//         if pos == 0 { // Piece at pos 0 allways is idx 0
//             return 0;

//         }

//         let before = self.piece_bb() << (26 + 2 + (36 - pos));  

//         // Number of pieces before the requested position == idx
//         before.count_ones() as usize

//     }

//     /// Idx is the index of the piece not its position. 
//     /// Ex. 0 = first piece, 1 = second piece, etc.
//     pub const fn piece_type(&self, piece_idx: usize) -> u8 {
//         ((self.0 >> (38 + (piece_idx * 2))) & 0b11) as u8

//     }

//     /// Gets the player to move
//     /// 0 = Player One
//     /// 1 = Player Two
//     pub const fn player(&self) -> u8 {
//         (self.0 >> 63) as u8

//     }

// }

/// Bits 0..37 are the positions of the pieces on the board.
/// Bits 38..62 are the types of pieces in order.
/// Bit 63 is the player to move.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BitState(pub u64);

impl BitState {
    /// Creates a new BitState from an existing BoardState
    pub fn from(board: &BoardState) -> Self {
        let mut bb = 0u64;
        let mut piece_idx = 0;
        for i in 0..38 {
            let piece = board.data[i] as u8;
            if piece != 3 {
                // Set the piece bit
                bb |= 1 << i;
                // Set the piece type
                bb |= ((piece + 1) as u64) << (38 + (piece_idx * 2));
                piece_idx += 1;
            }
        }
        // Set the player to move
        bb |= (board.player as u64) << 63;

        BitState(bb)

    }

    pub fn make_bounce_mv(&self, start_pos: u64, end_pos: u64) -> BitState {
        let mut new_state = *self;

        // Update the piece bitboard
        new_state.0 ^= (1 << start_pos) | (1 << end_pos);

        // Starting piece index and type
        let starting_idx = self.piece_idx(start_pos as usize);
        let starting_piece = self.piece_type(starting_idx);

        // Remove the piece type at starting index
        new_state.remove_type(starting_idx);

        // Since we've removed a piece type, adjust the ending index
        let ending_idx = if start_pos < end_pos {
            self.piece_idx(end_pos as usize) - 1
        } else {
            self.piece_idx(end_pos as usize)
        };

        // Add the starting piece type at the new index
        new_state.add_type(ending_idx, starting_piece);

        new_state

    }

    pub fn make_drop_mv(&self, start_pos: u64, pickup_pos: u64, drop_pos: u64) -> BitState {
        let mut new_state = *self;

        // Starting piece index and type
        let starting_idx = self.piece_idx(start_pos as usize);
        let starting_piece = self.piece_type(starting_idx);

        // Remove the starting piece
        new_state.remove_type(starting_idx);
        new_state.0 ^= 1 << start_pos;

        // Pickup piece index and type
        let pickup_idx = new_state.piece_idx(pickup_pos as usize);
        let pickup_piece = new_state.piece_type(pickup_idx);

        // Set the pickup piece's type to the starting piece's type
        new_state.set_type_data(pickup_idx, starting_piece);

        // Drop piece index
        let drop_idx = new_state.piece_idx(drop_pos as usize);

        // Add the pickup piece type at the drop index
        new_state.add_type(drop_idx, pickup_piece);

        // Update the piece bitboard
        new_state.0 ^= 1 << drop_pos;

        new_state
        
    }

    // ======================== MOVE MAKING HELPERS ========================

    /// Removes an existing piece type
    pub fn remove_type(&mut self, piece_idx: usize) {
        let type_pos = 38 + (piece_idx * 2);

        // Clear the two bits at type_pos
        let clear_mask = !(0b11u64 << type_pos);
        self.0 &= clear_mask;

        // Shift higher bits down by 2
        let higher_bits_mask = !0u64 << (type_pos + 2);
        let higher_bits = (self.0 & higher_bits_mask) >> 2;
        self.0 = (self.0 & !higher_bits_mask) | higher_bits;

    }

    /// Adds a new piece type
    pub fn add_type(&mut self, piece_idx: usize, piece_type: u8) {
        let type_pos = 38 + (piece_idx * 2);

        // Shift higher bits up by 2 to make space
        let higher_bits_mask = !0u64 << type_pos;
        let higher_bits = (self.0 & higher_bits_mask) << 2;
        self.0 = (self.0 & !higher_bits_mask) | higher_bits;

        // Set the new piece type
        self.0 |= (piece_type as u64) << type_pos;
        
    }

    /// Changes the type data at one of the type data slots
    pub fn set_type_data(&mut self, piece_idx: usize, piece_type: u8) {
        let type_pos = 38 + (piece_idx * 2);

        // Mask to clear the two bits at type_pos
        let clear_mask = !(0b11u64 << type_pos);

        // Clear and set the piece type bits at piece_idx
        self.0 = (self.0 & clear_mask) | ((piece_type as u64) << type_pos);

    }
    
    // ======================== GETTERS ========================

    /// Gets the piece at a square
    /// 0 = None
    /// 1 = One
    /// 2 = Two
    /// 3 = Three
    pub fn piece_at(&self, pos: usize) -> u8 {
        if (self.0 & (1 << pos)) == 0 {
            return 0;

        }

        let idx = self.piece_idx(pos);
        self.piece_type(idx)

    }

    /// Gets the piece bitboard
    pub fn piece_bb(&self) -> u64 {
        self.0 & ((1u64 << 38) - 1)

    }

    /// Gets the index of the piece at a position
    pub fn piece_idx(&self, pos: usize) -> usize {
        let mask = (1u64 << pos) - 1;
        let bits_before = self.piece_bb() & mask;
        bits_before.count_ones() as usize

    }

    /// Gets the piece type at a given index
    pub fn piece_type(&self, piece_idx: usize) -> u8 {
        ((self.0 >> (38 + (piece_idx * 2))) & 0b11) as u8

    }

    /// Gets the player to move
    pub fn player(&self) -> u8 {
        (self.0 >> 63) as u8

    }

}



// Impl bit opperators
impl Not for BitState {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }

}

impl BitOr<BitState> for BitState {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }

}

impl BitOr<u64> for BitState {
    type Output = Self;

    fn bitor(self, rhs: u64) -> Self::Output {
        Self(self.0 | rhs)
    }

}

impl BitOrAssign<BitState> for BitState {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }

}

impl BitOrAssign<u64> for BitState {
    fn bitor_assign(&mut self, rhs: u64) {
        self.0 |= rhs;
    }

}

impl BitAnd<BitState> for BitState {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }

}

impl BitAnd<u64> for BitState {
    type Output = Self;

    fn bitand(self, rhs: u64) -> Self::Output {
        Self(self.0 & rhs)
    }

}

impl BitAndAssign<BitState> for BitState {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }

}

impl BitAndAssign<u64> for BitState {
    fn bitand_assign(&mut self, rhs: u64) {
        self.0 &= rhs;
    }

}

impl BitXor<BitState> for BitState {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }

}

impl BitXor<u64> for BitState {
    type Output = Self;

    fn bitxor(self, rhs: u64) -> Self::Output {
        Self(self.0 ^ rhs)
    }

}

impl BitXorAssign<BitState> for BitState {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }

}

impl BitXorAssign<u64> for BitState {
    fn bitxor_assign(&mut self, rhs: u64) {
        self.0 ^= rhs;
    }

}

impl Shl<usize> for BitState {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.0 << rhs)
    }

}

impl ShlAssign<usize> for BitState {
    fn shl_assign(&mut self, rhs: usize) {
        self.0 <<= rhs;
    }

}

impl Shr<usize> for BitState {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        Self(self.0 >> rhs)
    }

}

impl ShrAssign<usize> for BitState {
    fn shr_assign(&mut self, rhs: usize) {
        self.0 >>= rhs;
    }

}
