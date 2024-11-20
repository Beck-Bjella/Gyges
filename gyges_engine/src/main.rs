extern crate gyges_engine;

// use gyges_engine::ugi::Ugi;

// pub fn main() {
//     Ugi::new().start();

// }

// =================================== BLOCKING MOVE GEN ===================================


use cust::sys::cuMemHostGetDevicePointer_v2;
use gyges_engine::merged_reach_consts::*;
use gyges::{
    board::{
        self, TEST_BOARD
        
    }, moves::{
        movegen::{
            has_threat, piece_control_sqs, valid_moves 

        }, movegen_consts::ALL_THREE_INTERCEPTS, Move
    }, BitBoard, BoardState, Piece, Player, SQ

};


use cuda_sys::{cuda::*, cudart::cudaFreeHost};
use std::{ffi::{c_void, CString}, num, ptr};

pub fn cuda_init() -> Result<CUcontext, CUresult> {
    // Initialize the CUDA driver
    let init_result = unsafe { cuInit(0) };
    if init_result != cudaError_t::CUDA_SUCCESS {
        return Err(init_result);

    }

    // Get the first available device
    let mut device: CUdevice = 0;
    let device_result = unsafe { cuDeviceGet(&mut device, 0) };
    if device_result != cudaError_t::CUDA_SUCCESS {
        return Err(device_result);

    }

    // Create a context for the device
    let mut context: CUcontext = ptr::null_mut();
    let context_result = unsafe { cuCtxCreate_v2(&mut context, 0, device) };
    if context_result == cudaError_t::CUDA_SUCCESS {
        Ok(context)  // Return the context if successful

    } else {
        Err(context_result)  // Return the error code on failure

    }

}

pub fn device_mem_alloc<T>(size: usize) -> Result<CUdeviceptr, CUresult> {
    let mut device_ptr: CUdeviceptr = CUdeviceptr::default();

    let result: cudaError_t = unsafe {
        cuMemAlloc_v2(
            &mut device_ptr,
            size * std::mem::size_of::<T>(),

        )

    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(device_ptr)

    } else {
        Err(result)

    }

}

pub fn device_mem_free(device_ptr: CUdeviceptr) -> Result<(), CUresult> {
    let result: cudaError_t = unsafe { cuMemFree_v2(device_ptr as CUdeviceptr) };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(())

    } else {
        Err(result)

    }

}

pub fn allocate_zero_copy_memory<T>(size: usize) -> Result<(*mut T, CUdeviceptr), CUresult> {
    let mut host_ptr: *mut c_void = std::ptr::null_mut();
    let bytes = size * std::mem::size_of::<T>();

    // Allocate pinned host memory
    let result: cudaError_t = unsafe {
        cuMemHostAlloc(
            &mut host_ptr as *mut *mut c_void,
            bytes,
            CU_MEMHOSTALLOC_DEVICEMAP
        )
    };
    if result != cudaError_t::CUDA_SUCCESS {
        return Err(result);

    }

    // Retrieve the device pointer for the mapped memory
    let mut device_ptr: CUdeviceptr = 0;
    unsafe { cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0) };

    Ok((host_ptr as *mut T, device_ptr))

}

pub fn mem_copy_to_device<T>(device_ptr: CUdeviceptr, host_data: &[T]) -> Result<(), CUresult> {
    let size_in_bytes = host_data.len() * std::mem::size_of::<T>();

    let result: cudaError_t = unsafe {
        cuMemcpyHtoD_v2(
            device_ptr,
            host_data.as_ptr() as *const std::ffi::c_void,
            size_in_bytes

        )

    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(())

    } else {
        Err(result)

    }

}

pub fn mem_copy_to_host<T>(device_ptr: CUdeviceptr, host_data: *mut T, size: usize) -> Result<(), CUresult> {
    let size_in_bytes = size * std::mem::size_of::<T>();

    let result: cudaError_t = unsafe {
        cuMemcpyDtoH_v2(host_data as *mut c_void, device_ptr, size_in_bytes)
    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(())

    } else {
        Err(result)

    }

}

pub fn load_module_from_ptx(ptx: &str) -> Result<CUmodule, cudaError_t> {
    let c_file_path = CString::new(ptx).expect("Failed to convert PTX to CString");

    let mut module: CUmodule = ptr::null_mut();

    let result: cudaError_t = unsafe {
        cuModuleLoad(&mut module, c_file_path.as_ptr())
    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(module)

    } else {
        Err(result)

    }

}

pub fn get_kernel_function(module: CUmodule, kernel_name: &str) -> Result<CUfunction, CUresult> {
    let c_kernel_name = CString::new(kernel_name).expect("Failed to create CString for kernel name");

    let mut function: CUfunction = ptr::null_mut();

    let result: cudaError_t = unsafe { cuModuleGetFunction(&mut function, module, c_kernel_name.as_ptr()) };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(function)

    } else {
        Err(result)

    }

}

fn main() -> Result<(), CUresult> {
    // Initialize CUDA
    let _ctx: CUcontext = cuda_init().expect("Failed to initialize CUDA context");


    // Initialize board
    let mut board = BoardState::from(TEST_BOARD);
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

    let new_batch = mv_gen.gen(&mut board, player);
    println!("New Batch: {}", new_batch.len());

    unsafe {
        // MAIN BENCHMARKS
        let iters = 100000;

        // NEW
        let mut num = 0;
        let start: std::time::Instant = std::time::Instant::now();
        for _ in 0..iters {
            let moves = mv_gen.gen(&mut board, player);
            num += moves.len();

        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("New Elapsed: {:?}, {}", elapsed, num);

        // Native
        let mut num = 0;
        let start = std::time::Instant::now();
        for _ in 0..iters {
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

    mv_gen.mem_free();

    Ok(())
        
}


pub struct BlockingMoveGen {
    // Lookup tables
    one_reach_d: CUdeviceptr,
    two_reach_d: CUdeviceptr,
    three_reach_d: CUdeviceptr,

    // Zero-copy buffers
    state_input_d: CUdeviceptr,
    state_input_h: *mut u64,
    move_input_d: CUdeviceptr,
    move_input_h: *mut u8,
    final_d: CUdeviceptr,
    final_h: *mut f32,

    // CUDA
    module: CUmodule,
    kernel: CUfunction,

}

impl BlockingMoveGen {
    pub fn new() -> Result<Self, CUresult> {
        // Load kernel from PTX
        let module: CUmodule = load_module_from_ptx("kernels.ptx")?;
        let kernel = get_kernel_function(module, "unified_kernel")?;

        // Allocate lookup tables
        let one_reach_d = device_mem_alloc::<u64>(36)?;
        mem_copy_to_device(one_reach_d, MERGED_ONE_REACHS.as_ref())?;
        let two_reach_d = device_mem_alloc::<u64>(29 * 36)?;
        mem_copy_to_device(two_reach_d, MERGED_TWO_REACHS.as_flattened())?;
        let three_reach_d = device_mem_alloc::<u64>(11007 * 36)?;
        mem_copy_to_device(three_reach_d, MERGED_THREE_REACHS.as_flattened())?;
      
        // Allocate buffers
        let (state_input_h, state_input_d) = allocate_zero_copy_memory::<u64>(1)?;
        let (move_input_h, move_input_d) = allocate_zero_copy_memory::<u8>(1000*3)?;
        let (final_h, final_d) = allocate_zero_copy_memory::<f32>(1000)?;

        // Create the instance  
        Ok(Self {
            one_reach_d,
            two_reach_d,
            three_reach_d,
    
            move_input_d,
            move_input_h,
            final_d,
            final_h,
            state_input_d,
            state_input_h,

            module,
            kernel

        })

    }

    pub fn gen(&mut self, board: &mut BoardState, player: Player) -> Vec<(u8, u8, u8)> {
        let bit_state = BitState::from(board);
        unsafe { *self.state_input_h = bit_state.0; } // Set the input state
        
        let mut piece_control: [BitBoard; 6] = unsafe { piece_control_sqs(board, player) };

        let active_lines = board.get_active_lines();
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);
        let drops = board.get_drops(active_lines, player).get_data();

        let mut idx = 0;
        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            let starting_piece = board.piece_at(starting_sq);
            if starting_piece != Piece::None {
                for block_pos in piece_control[x].get_data() {
                    let block_sq = SQ(block_pos as u8);
                    let block_piece = board.piece_at(block_sq);
    
                    if block_piece == Piece::None {
                        unsafe { self.move_input_h.add(idx * 3).copy_from([starting_sq.0, block_sq.0, 100].as_ptr(), 3); } // Copy move data 
                        idx += 1;
    
                    } else {
                        for empty_pos in drops.iter() {
                            unsafe { self.move_input_h.add(idx * 3).copy_from([starting_sq.0, block_sq.0, *empty_pos as u8].as_ptr(), 3); } // Copy move data 
                            idx += 1;
    
                        }
    
                    }
    
                }

            }

        }
        let num_boards: u32 = idx as u32;

        // Launch kernel
        unsafe {
            cuLaunchKernel(
                self.kernel,
                num_boards as u32, 1, 1,
                38, 1, 1,
                0,
                ptr::null_mut(),
                [
                    &mut self.state_input_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.move_input_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.final_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.one_reach_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.two_reach_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.three_reach_d as *mut CUdeviceptr as *mut c_void
                ].as_ptr() as *mut *mut c_void,
                ptr::null_mut(),  // No extra arguments
            );

        }

        // Sync results
        unsafe { cuCtxSynchronize(); }

        // IMPACT ON PERFORMANCE: MINIMAL 
        // Filter blocking moves from the results
        let mut blocking_moves = Vec::with_capacity(50);
        for i in 0..num_boards {
            if unsafe{ self.final_h.add(i as usize).read() == 0.0 } {
                unsafe {
                    let mv = self.move_input_h.add(i as usize * 3);
                    blocking_moves.push((mv.read(), mv.add(1).read(), mv.add(2).read()));

                }
     
            }

        }

        // Return the blocking moves
        blocking_moves

    }

    /// Frees all allocated GPU memory
    pub fn mem_free(&mut self) {
        // Free lookup tables
        device_mem_free(self.one_reach_d).expect("Failed to free one reach table");
        device_mem_free(self.two_reach_d).expect("Failed to free two reach table");
        device_mem_free(self.three_reach_d).expect("Failed to free three reach table");

        // Free buffers
        unsafe {
            cuMemFreeHost(self.state_input_h as *mut c_void);
            cuMemFreeHost(self.move_input_h as *mut c_void);
            cuMemFreeHost(self.final_h as *mut c_void);

        }

    }

}


// ================== BitState Representation ==================

use std::ops::{Not, BitOr, BitOrAssign, BitAnd, BitAndAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign};

/// Bits 0..37 are the positions of the pieces on the board.
/// Bits 38..62 are the types of pieces in order.
/// Bit 63 is the player to move.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BitState(pub u64);

impl BitState {
    /// Creates a new BitState from an existing BoardState
    pub fn from(board: &BoardState) -> Self {
        let mut bb: u64 = 0u64;
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
