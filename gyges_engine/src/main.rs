extern crate gyges_engine;

// use gyges_engine::ugi::Ugi;

// pub fn main() {
//     Ugi::new().start();

// }

//

use gyges_engine::merged_reach_consts::*;
use gyges::{
    board::TEST_BOARD, core::masks::RANKS, moves::{movegen::{piece_control_sqs, valid_moves}, movegen_consts::{ONE_MAP, THREE_MAP, TWO_MAP, UNIQUE_ONE_PATHS, UNIQUE_ONE_PATH_LISTS, UNIQUE_THREE_PATHS, UNIQUE_THREE_PATH_LISTS, UNIQUE_TWO_PATHS, UNIQUE_TWO_PATH_LISTS}, Move, MoveType}, BitBoard, BoardState, Piece, Player, SQ

};
use cuda_sys::cuda::*;
use std::{ffi::{c_void, CString}, fmt::Display, ptr};

// =================================== MOVE GEN ===================================

fn main() -> Result<(), CUresult> {
    // Initialize CUDA
    let _ctx: CUcontext = cuda_init().expect("Failed to initialize CUDA context");

    let mut board = BoardState::from(TEST_BOARD);
    // let mut board = BoardState::from([ // SPARCE CASE
    //     0, 0, 0, 0, 0, 3,
    //     0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 2, 0,
    //     0, 0, 0, 0, 0, 3,
    //     0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 1,
    //     0, 0
    // ]);
    let player = Player::One;
    println!("{}", board);

    let mut mg = MoveGen::new().expect("Failed to initialize move generator");

    let moves = mg.gen(&mut board, player);
    let new: Vec<Move> = decode_moves(&mut board, &moves[0]);

    let mut moves = unsafe { valid_moves(&mut board, player) };
    let real: Vec<Move> = moves.moves(&board);

    println!("Real: {}, New: {}", real.len(), new.len());
    
    // BENCHMARKS
    unsafe {
        let iters = 50000;

        for batch in 0..3 {
            // NEW
            let mut num = 0;
            let start: std::time::Instant = std::time::Instant::now();
            for _ in 0..iters {
                let moves: Vec<GenResult> = mg.gen(&mut board, player);
                // let dec = decode_moves(&mut board, &moves[0]);
                // num += dec.len();

                num += 1;
               
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!("{}: New Elapsed: {:?}, {}", batch, elapsed / iters as f64, num);

        }


        for batch in 0..3 {
            // Native
            let mut num = 0;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let mut moves = valid_moves(&mut board, player);
                // let dec = moves.moves(&board);
                // num += dec.len();

                num += 1;
               
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!("{}: Native Elapsed: {:?}, {}", batch, elapsed / iters as f64, num);
    
        }

    }
    
    mg.mem_free();

    Ok(())

}



// CHATGPT GENERATED
pub unsafe fn set_symbol_device_pointer(
    module: CUmodule,               // The loaded CUDA module
    symbol_name: &str,              // The symbol name (device variable)
    device_ptr: CUdeviceptr,        // The device pointer to copy to the symbol
) -> Result<(), CUresult> {
    let mut symbol_address: CUdeviceptr = 0; // Pointer to the symbol in device memory
    let mut symbol_size: usize = 0;          // Size of the symbol

    // Convert the symbol name to a C-compatible string
    let c_symbol_name = CString::new(symbol_name).expect("Symbol name conversion failed");

    // Retrieve the symbol address and size in the module
    let result = cuModuleGetGlobal_v2(
        &mut symbol_address as *mut CUdeviceptr,
        &mut symbol_size as *mut usize,
        module,
        c_symbol_name.as_ptr(),
    );

    if result != CUresult::CUDA_SUCCESS {
        return Err(result); // Symbol not found or invalid
    }

    // Ensure the symbol size matches the size of a CUdeviceptr
    if symbol_size != std::mem::size_of::<CUdeviceptr>() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE); // Size mismatch
    }

    // Copy the device pointer to the symbol's memory location
    let result = cuMemcpyHtoD_v2(
        symbol_address,
        &device_ptr as *const _ as *const std::ffi::c_void,
        symbol_size,
    );

    if result != CUresult::CUDA_SUCCESS {
        return Err(result); // Copy operation failed
    }

    Ok(())

}


// Structs for the CUDA kernel

#[repr(C)]
struct StackData {
    banned_bb: u64,
    backtrack_bb: u64,
    current_pos: u8,
    current_piece: u8,

}

#[repr(C)]
pub struct GenRequest {
    pub state: u64,
    pub active_bb: u64,
    pub flag: u8

}

#[repr(C)]
pub struct GenResult {
    end_positions: [u64; 6],
    pickup_positions: [u64; 6],
    drop_positions: u64,

}

#[repr(C)]
struct OnePath {
    backtrack_bb: u64,
    pos_1: u8,
    pos_2: u8

}

#[repr(C)]
struct TwoPath {
    backtrack_bb: u64,
    pos_1: u8,
    pos_2: u8,
    pos_3: u8

}

#[repr(C)]
struct ThreePath {
    backtrack_bb: u64,
    pos_1: u8,
    pos_2: u8,
    pos_3: u8,
    pos_4: u8

}


pub struct MoveGen {
    // GPU Stack buffer
    stack_d: CUdeviceptr,

    // Input & Output Buffers
    input_d: CUdeviceptr,
    input_h: *mut GenRequest,
    final_d: CUdeviceptr,
    final_h: *mut GenResult,

    // CUDA
    module: CUmodule,
    gen_kernel: CUfunction,

}

impl MoveGen {
    pub const MAX_REQUESTS: usize = 1;

    pub fn new() -> Result<Self, CUresult> {
        // Load kernel from PTX
        let module: CUmodule = load_module_from_ptx("kernels.ptx")?;
        let gen_kernel = get_kernel_function(module, "gen_kernel")?;

        let mut one_path_vec = vec![];
        for i in 0..UNIQUE_ONE_PATHS.len() {
            let path: ([u8; 2], u64) = UNIQUE_ONE_PATHS[i];
            one_path_vec.push(OnePath {
                backtrack_bb: path.1,
                pos_1: path.0[0],
                pos_2: path.0[1]

            });

        }

        let mut two_path_vec = vec![];
        for i in 0..UNIQUE_TWO_PATHS.len() {
            let path: ([u8; 3], u64) = UNIQUE_TWO_PATHS[i];
            two_path_vec.push(TwoPath {
                backtrack_bb: path.1,
                pos_1: path.0[0],
                pos_2: path.0[1],
                pos_3: path.0[2]

            });

        }

        let mut three_path_vec = vec![];
        for i in 0..UNIQUE_THREE_PATHS.len() {
            let path: ([u8; 4], u64) = UNIQUE_THREE_PATHS[i];
            three_path_vec.push(ThreePath {
                backtrack_bb: path.1,
                pos_1: path.0[0],
                pos_2: path.0[1],
                pos_3: path.0[2],
                pos_4: path.0[3]

            });

        }

        // Allocate and copy the lookup tables to GPU
        unsafe {
            // ========== 'UNIQUE PATHS' LOOKUP TABLES ==========
            let unique_one_paths_d = device_mem_alloc::<OnePath>(UNIQUE_ONE_PATHS.len())?;
            mem_copy_to_device(unique_one_paths_d, &one_path_vec)?;
            set_symbol_device_pointer(module, "one_paths", unique_one_paths_d).expect("Failed to set one paths symbol");
            
            let unique_two_paths_d = device_mem_alloc::<TwoPath>(UNIQUE_TWO_PATHS.len())?;
            mem_copy_to_device(unique_two_paths_d, &two_path_vec)?;
            set_symbol_device_pointer(module, "two_paths", unique_two_paths_d).expect("Failed to set two paths symbol");

            let unique_three_paths_d = device_mem_alloc::<ThreePath>(UNIQUE_THREE_PATHS.len())?;
            mem_copy_to_device(unique_three_paths_d, &three_path_vec)?;
            set_symbol_device_pointer(module, "three_paths", unique_three_paths_d).expect("Failed to set three paths symbol");

            // ========== 'UNIQUE PATH LISTS' LOOKUP TABLES ==========
            let unique_one_paths_list_d = device_mem_alloc::<u16>(UNIQUE_ONE_PATH_LISTS.len() * 5)?;
            mem_copy_to_device(unique_one_paths_list_d, UNIQUE_ONE_PATH_LISTS.as_flattened())?;
            set_symbol_device_pointer(module, "one_path_lists", unique_one_paths_list_d).expect("Failed to set one path lists symbol");

            let unique_two_paths_list_d = device_mem_alloc::<u16>(UNIQUE_TWO_PATH_LISTS.len() * 13)?;
            mem_copy_to_device(unique_two_paths_list_d, UNIQUE_TWO_PATH_LISTS.as_flattened())?;
            set_symbol_device_pointer(module, "two_path_lists", unique_two_paths_list_d).expect("Failed to set two path lists symbol");

            let unique_three_paths_list_d = device_mem_alloc::<u16>(UNIQUE_THREE_PATH_LISTS.len() * 36)?;
            mem_copy_to_device(unique_three_paths_list_d, UNIQUE_THREE_PATH_LISTS.as_flattened())?;
            set_symbol_device_pointer(module, "three_path_lists", unique_three_paths_list_d).expect("Failed to set three path lists symbol");

            // ========== 'MAPS' LOOKUP TABLES ==========
            let one_map_d = device_mem_alloc::<u8>(36)?;
            mem_copy_to_device(one_map_d, ONE_MAP.as_flattened())?;
            set_symbol_device_pointer(module, "one_map", one_map_d).expect("Failed to set one map symbol");

            let two_map_d = device_mem_alloc::<u16>(29 * 36)?;
            mem_copy_to_device(two_map_d, TWO_MAP.as_flattened())?;
            set_symbol_device_pointer(module, "two_map", two_map_d).expect("Failed to set two map symbol");

            let three_map_d = device_mem_alloc::<u16>(11007 * 36)?;
            mem_copy_to_device(three_map_d, THREE_MAP.as_flattened())?;
            set_symbol_device_pointer(module, "three_map", three_map_d).expect("Failed to set three map symbol");

        }
       
        // Allocate stack buffers
        let stack_d = device_mem_alloc::<StackData>(1 * 1000 * 3)?;

        // Allocate input & output buffers
        let (input_h, input_d) = allocate_zero_copy_memory::<GenRequest>(MoveGen::MAX_REQUESTS)?;
        let (final_h, final_d) = allocate_zero_copy_memory::<GenResult>(MoveGen::MAX_REQUESTS)?;
        
        // Create the instance  
        Ok(Self {
            stack_d,

            input_d,
            input_h,
            final_d,
            final_h,

            module,
            gen_kernel,

        })

    }

    pub fn create_request(&self, board: &mut BoardState, flag: u8) -> GenRequest {
        let bit_state = BitState::from(board);

        let active_lines = board.get_active_lines();
        let active_bb: BitBoard = board.piece_bb & RANKS[active_lines[0]];

        GenRequest {
            state: bit_state.0,
            active_bb: active_bb.0,
            flag

        }

    }

    pub fn gen(&mut self, board: &mut BoardState, _player: Player) -> Vec<GenResult> {
        let num_requests = 1;

        // Create and save requests
        for i in 0..num_requests {
            let request = self.create_request(board, 0);
            unsafe { self.input_h.add(i).write(request); }

        }

        // Launch kernel
        unsafe {
            cuLaunchKernel(
                self.gen_kernel,
                num_requests as u32, 1, 1,
                32 * 3, 1, 1,
                0,
                ptr::null_mut(),
                [
                    &mut self.input_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.final_d as *mut CUdeviceptr as *mut c_void,
                    &mut self.stack_d as *mut CUdeviceptr as *mut c_void,
                ].as_ptr() as *mut *mut c_void,
                ptr::null_mut(),  // No extra arguments

            );

        }

        // Sync results
        unsafe { cuCtxSynchronize(); }

        let mut results: Vec<GenResult> = Vec::with_capacity(num_requests);
        for i in 0..num_requests {
            unsafe {
                results.push(self.final_h.add(i).read());

            }

        }

        results

    }

    /// Frees all allocated GPU memory
    pub fn mem_free(&mut self) {
        device_mem_free(self.stack_d).expect("Failed to free stack buffer");
        
        // Free Host buffers
        unsafe {
            cuMemFreeHost(self.input_h as *mut c_void);
            cuMemFreeHost(self.final_h as *mut c_void);

        }

    }

    
}

pub fn decode_moves(board: &mut BoardState, gen_result: &GenResult) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::with_capacity(1000);

    let drop_positions = BitBoard(gen_result.drop_positions).get_data();

    for i in 0..6 {
        let start_sq = SQ(i as u8);
        let start_piece = board.piece_at(start_sq);
        if start_piece == Piece::None {
            continue;

        }

        let start_position = (start_piece, start_sq);

        for end_pos in BitBoard(gen_result.end_positions[i]).get_data() {
            let data = [(Piece::None, start_position.1), (start_position.0, SQ(end_pos as u8)), (Piece::None, SQ::NONE)];
            moves.push(Move::new(data, MoveType::Bounce));

        }

        for pick_up_pos in BitBoard(gen_result.pickup_positions[i]).get_data() {
            let data = [(Piece::None, start_position.1), (start_position.0, SQ(pick_up_pos as u8)), (board.piece_at(SQ(pick_up_pos as u8)), start_position.1)];
            moves.push(Move::new(data, MoveType::Drop));

            for drop_pos in drop_positions.iter() {
                let data = [(Piece::None, start_position.1), (start_position.0, SQ(pick_up_pos as u8)), (board.piece_at(SQ(pick_up_pos as u8)), SQ(*drop_pos as u8))];
                moves.push(Move::new(data, MoveType::Drop));

            }
    
        }
    
    }

    moves

}

// =================================== CUDA HELPERS ===================================

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

// =================================== BLOCKING MOVE GEN ===================================

// fn main() -> Result<(), CUresult> {
//     // Initialize CUDA
//     let _ctx: CUcontext = cuda_init().expect("Failed to initialize CUDA context");

//     // Initialize board
//     let mut board = BoardState::from(TEST_BOARD); // TESTING CASE
//     // let mut board2 = BoardState::from([ // REAL CASE
//     //     0, 0, 1, 0, 0, 0,
//     //     0, 3, 3, 0, 2, 0,
//     //     0, 2, 2, 0, 1, 0,
//     //     0, 0, 0, 0, 3, 0,
//     //     0, 0, 2, 3, 0, 0,
//     //     0, 0, 1, 0, 2, 0,
//     //     0, 0
//     // ]);
//     // let mut board = BoardState::from([ // SPARCE CASE
//     //     0, 0, 2, 0, 0, 0,
//     //     0, 0, 0, 0, 0, 0,
//     //     3, 0, 0, 0, 0, 0,
//     //     0, 0, 0, 0, 0, 0,
//     //     0, 0, 0, 0, 0, 0,
//     //     3, 0, 0, 0, 0, 0,
//     //     0, 0
//     // ]);
//     let player = Player::One;
//     println!("{}", board);
    
//     let mut mv_gen = BlockingMoveGen::new().expect("Failed to initialize blocking move generator");

//     let new_batch = mv_gen.gen(&mut board, player);
//     println!("{:?}", new_batch.len());

//     unsafe {
//         // MAIN BENCHMARKS
//         let iters = 10000;

//         for batch in 0..2 {
//             // NEW
//             let mut num = 0;
//             let start: std::time::Instant = std::time::Instant::now();
//             for _ in 0..iters {
//                 let moves = mv_gen.gen(&mut board, player);
//                 num += moves.len();
    
//             }
//             let elapsed = start.elapsed().as_secs_f64();
//             println!("{}: WAVE Elapsed: {:?}, {}", batch, elapsed / iters as f64, num);

//         }

//         for batch in 0..2 {
//             // Native
//             let mut num = 0;
//             let start = std::time::Instant::now();
//             for _ in 0..iters {
//                 let moves = valid_moves(&mut board, player).moves(&board);
//                 let mut pruned = vec![];
//                 for mv in moves.iter() {
//                     let mut new_board = board.clone().make_move(mv);
//                     let has_threat = has_threat(&mut new_board, player.other());
//                     if !has_threat {
//                         pruned.push(mv);
//                         num += 1;
//                     }

//                 }

//             }
//             let elapsed = start.elapsed().as_secs_f64();
//             println!("{}: Native Elapsed: {:?}, {}", batch, elapsed / iters as f64, num);

//         }

//     }

//     mv_gen.mem_free();

//     Ok(())
        
// }

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
    _module: CUmodule,
    wavefront_kernel: CUfunction,

}

impl BlockingMoveGen {
    pub fn new() -> Result<Self, CUresult> {
        // Load kernel from PTX
        let _module: CUmodule = load_module_from_ptx("kernels.ptx")?;
        let wavefront_kernel = get_kernel_function(_module, "wavefront_kernel")?; // Old kernel: "adj_kernel"
        
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

            state_input_d,
            state_input_h,
            move_input_d,
            move_input_h,
            final_d,
            final_h,

            _module,
            wavefront_kernel,

        })

    }

    pub fn gen(&mut self, board: &mut BoardState, player: Player) -> Vec<(u8, u8, u8)> {
        let bit_state = BitState::from(board);
        unsafe { *self.state_input_h = bit_state.0; } // Set the input state
        
        let piece_control: [BitBoard; 6] = unsafe { piece_control_sqs(board, player) };

        let active_lines = board.get_active_lines();
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);
        let drops = board.get_drops(active_lines, player).get_data();

        let mut idx = 0;
        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) == Piece::None {
                continue;

            }

            // Cant reach the starting square
            let mut new_piece_control = piece_control[x] & !(1 << starting_sq.0);

            for block_pos in new_piece_control.get_data() {
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
        let num_moves = idx as u32;
    
        // Launch kernel
        unsafe {
            cuLaunchKernel(
                self.wavefront_kernel,
                num_moves, 1, 1,
                36, 1, 1,
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

        // Filter blocking moves from the results
        let mut blocking_moves = Vec::with_capacity(50);
        for i in 0..num_moves {
            unsafe {
                if self.final_h.add(i as usize).read() == 0.0 {
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

impl Display for BitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.piece_at(37) == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.piece_at(37))?;

        }
        writeln!(f, " ")?;
        writeln!(f, " ")?;

        for y in (0..6).rev() {
            for x in 0..6 {
                if self.piece_at(y * 6 + x) == 0 {
                    write!(f, "    .")?;
                } else {
                    write!(f, "    {}", self.piece_at(y * 6 + x))?;

                }
               
            }
            writeln!(f, " ")?;
            writeln!(f, " ")?;

        }

        writeln!(f, " ")?;
        if self.piece_at(36) == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.piece_at(36))?;

        }

        Result::Ok(())

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
