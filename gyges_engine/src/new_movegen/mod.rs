pub mod cuda_helpers;
pub mod bitstate;

use std::{ffi::c_void, ptr};

use gyges::{
    core::masks::RANKS, moves::{
        movegen_consts::*, 
        Move, 
        MoveType
    }, BitBoard, BoardState, Piece, Player, SQ 

};

use cuda_sys::cuda::*;
use cuda_helpers::*;

use bitstate::*;

pub struct MoveGen {
    inner: InnerMoveGen
    
}

impl MoveGen {
    pub fn new() -> Self {
        Self {
            inner: InnerMoveGen::new().expect("Failed to initialize inner move generator")

        }

    }

    pub fn batch(&mut self, boards: &Vec<BoardState>, player: Player) -> Vec<GenResult> {
        for (i, board) in boards.iter().enumerate() {
            let bitstate = BitState::from(board);

            let active_lines = board.get_active_lines();
            let active_bb: BitBoard = board.piece_bb & RANKS[active_lines[0]];

            let request = GenRequest {
                state: bitstate.0,
                active_bb: active_bb.0,
                flag: player as u8

            };

            unsafe { self.inner.store(request, i); }

        }

        unsafe { 
            self.inner.gen(boards.len() as u32); 
            self.inner.sync();

        }

        let mut results = Vec::with_capacity(boards.len());
        for i in 0..boards.len() {
            results.push(unsafe { self.inner.fetch(i) });

        }

        results

    }
    
}

impl Drop for MoveGen {
    fn drop(&mut self) {
        self.inner.mem_free();

    }

}


pub struct InnerMoveGen {
    // GPU Stack buffer
    stack_d: CUdeviceptr,
    stack_heights_d: CUdeviceptr,

    // Input & Output Buffers
    input_d: CUdeviceptr,
    input_h: *mut GenRequest,
    final_d: CUdeviceptr,
    final_h: *mut GenResult,

    // CUDA
    _ctx: CUcontext,
    _module: CUmodule,
    gen_kernel: CUfunction,
    stream: CUstream

}

impl InnerMoveGen {
    pub const MAX_REQUESTS: usize = 10000;
    pub const MAX_STACK_SIZE: usize = 75;

    pub fn new() -> Result<Self, CUresult> {
        // Initialize CUDA
        let _ctx: CUcontext = cuda_init()?;

        // Load kernel from PTX
        let _module: CUmodule = load_module_from_ptx("kernels.ptx")?;
        let gen_kernel = get_kernel_function(_module, "gen_kernel")?;

        let mut stream = ptr::null_mut();
        unsafe { cuStreamCreate(&mut stream, 0) };

        // Format lookup tables
        let one_path_vec = UNIQUE_ONE_PATHS.iter().map(|path| {
            OnePath {
                backtrack_bb: path.1,
                pos_1: path.0[0],
                pos_2: path.0[1]

            }

        }).collect::<Vec<OnePath>>();

        let two_path_vec = UNIQUE_TWO_PATHS.iter().map(|path| {
            TwoPath {
                backtrack_bb: path.1,
                pos_1: path.0[0],
                pos_2: path.0[1],
                pos_3: path.0[2]

            }

        }).collect::<Vec<TwoPath>>();

        // Simplify the three paths
        let three_path_vec: Vec<ThreePath> = UNIQUE_THREE_PATHS.iter().map(|path| {
            ThreePath {
                backtrack_bb: path.1,
                pos_1: path.0[0],
                pos_2: path.0[1],
                pos_3: path.0[2],
                pos_4: path.0[3]

            }

        }).collect::<Vec<ThreePath>>();

        // Allocate and copy the lookup tables to GPU
        unsafe {
            // ========== 'UNIQUE PATHS' LOOKUP TABLES ==========
            let unique_one_paths_d = device_mem_alloc::<OnePath>(UNIQUE_ONE_PATHS.len())?;
            mem_copy_to_device(unique_one_paths_d, &one_path_vec)?;
            set_symbol_device_pointer(_module, "one_paths", unique_one_paths_d)?;
            
            let unique_two_paths_d = device_mem_alloc::<TwoPath>(UNIQUE_TWO_PATHS.len())?;
            mem_copy_to_device(unique_two_paths_d, &two_path_vec)?;
            set_symbol_device_pointer(_module, "two_paths", unique_two_paths_d)?;

            let unique_three_paths_d = device_mem_alloc::<ThreePath>(UNIQUE_THREE_PATHS.len())?;
            mem_copy_to_device(unique_three_paths_d, &three_path_vec)?;
            set_symbol_device_pointer(_module, "three_paths", unique_three_paths_d)?;

            // ========== 'UNIQUE PATH LISTS' LOOKUP TABLES ==========
            let unique_one_paths_list_d = device_mem_alloc::<u16>(UNIQUE_ONE_PATH_LISTS.len() * 5)?;
            mem_copy_to_device(unique_one_paths_list_d, UNIQUE_ONE_PATH_LISTS.as_flattened())?;
            set_symbol_device_pointer(_module, "one_path_lists", unique_one_paths_list_d)?;

            let unique_two_paths_list_d = device_mem_alloc::<u16>(UNIQUE_TWO_PATH_LISTS.len() * 13)?;
            mem_copy_to_device(unique_two_paths_list_d, UNIQUE_TWO_PATH_LISTS.as_flattened())?;
            set_symbol_device_pointer(_module, "two_path_lists", unique_two_paths_list_d)?;

            let unique_three_paths_list_d = device_mem_alloc::<u16>(UNIQUE_THREE_PATH_LISTS.len() * 36)?;
            mem_copy_to_device(unique_three_paths_list_d, UNIQUE_THREE_PATH_LISTS.as_flattened())?;
            set_symbol_device_pointer(_module, "three_path_lists", unique_three_paths_list_d)?;

            // ========== 'MAPS' LOOKUP TABLES ==========
            let one_map_d = device_mem_alloc::<u8>(36)?;
            mem_copy_to_device(one_map_d, ONE_MAP.as_flattened())?;
            set_symbol_device_pointer(_module, "one_map", one_map_d)?;

            let two_map_d = device_mem_alloc::<u16>(29 * 36)?;
            mem_copy_to_device(two_map_d, TWO_MAP.as_flattened())?;
            set_symbol_device_pointer(_module, "two_map", two_map_d)?;

            let three_map_d = device_mem_alloc::<u16>(11007 * 36)?;
            mem_copy_to_device(three_map_d, THREE_MAP.as_flattened())?;
            set_symbol_device_pointer(_module, "three_map", three_map_d)?;

        }
       
        // Allocate stack buffer
        let stack_d = device_mem_alloc::<StackData>(InnerMoveGen::MAX_REQUESTS * InnerMoveGen::MAX_STACK_SIZE * 3)?;
        let stack_heights_d = device_mem_alloc::<u32>(InnerMoveGen::MAX_REQUESTS * 3)?;

        // Allocate input & output buffers
        let (input_h, input_d) = allocate_zero_copy_memory::<GenRequest>(InnerMoveGen::MAX_REQUESTS)?;
        let (final_h, final_d) = allocate_zero_copy_memory::<GenResult>(InnerMoveGen::MAX_REQUESTS)?;
        
        // Create the instance  
        Ok(Self {
            stack_d,
            stack_heights_d,

            input_d,
            input_h,
            final_d,
            final_h,

            _ctx,
            _module,
            gen_kernel,
            stream

        })

    }

    /// Stores a request into the input buffer
    pub unsafe fn store(&mut self, request: GenRequest, idx: usize) {
        if idx >= InnerMoveGen::MAX_REQUESTS {
            panic!("Request index out of bounds");

        }

        self.input_h.add(idx).write_volatile(request);

    }

    /// Fetches a result from the output buffer
    pub unsafe fn fetch(&mut self, idx: usize) -> GenResult {
        if idx >= InnerMoveGen::MAX_REQUESTS {
            panic!("Request index out of bounds");

        }

        self.final_h.add(idx).read_volatile()

    }

    /// Processes 'n' number of requests stored in the input buffer, and loads there results into the output buffer.
    pub unsafe fn gen(&mut self, n: u32) {
        // Launch kernel
        cuLaunchKernel(
            self.gen_kernel,
            n as u32, 1, 1,
            32 * 3, 1, 1,
            0,
            self.stream,
            [
                &mut self.input_d as *mut CUdeviceptr as *mut c_void,
                &mut self.final_d as *mut CUdeviceptr as *mut c_void,
                &mut self.stack_d as *mut CUdeviceptr as *mut c_void,
                &mut self.stack_heights_d as *mut CUdeviceptr as *mut c_void
            ].as_ptr() as *mut *mut c_void,
            ptr::null_mut(),  // No extra arguments

        );

    }

    /// Blocks untill done processing
    pub unsafe fn sync(&mut self) {
        cuStreamSynchronize(self.stream);

    }

    /// Frees all allocated GPU memory
    pub fn mem_free(&mut self) {
        device_mem_free(self.stack_d).expect("Failed to free stack buffer");
        device_mem_free(self.stack_heights_d).expect("Failed to free stack heights buffer");
        
        // Free Host buffers
        unsafe {
            cuMemFreeHost(self.input_h as *mut c_void);
            cuMemFreeHost(self.final_h as *mut c_void);

        }

    }
    
}

#[repr(C)]
pub struct GenRequest {
    pub state: u64,
    pub active_bb: u64,
    pub flag: u8

}

#[derive(Debug)]
#[repr(C)]
pub struct GenResult {
    end_positions: [u64; 6],
    pickup_positions: [u64; 6],
    drop_positions: u64,

}

impl GenResult {
    pub fn new() -> Self {
        Self {
            end_positions: [0; 6],
            pickup_positions: [0; 6],
            drop_positions: 0

        }

    }

    pub fn moves(&mut self, board: &BoardState) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::with_capacity(1000);

        let drop_positions = BitBoard(self.drop_positions).get_data();

        for i in 0..6 {
            let start_sq = SQ(i as u8);
            let start_piece = board.piece_at(start_sq);
            if start_piece == Piece::None {
                continue;

            }

            let start_position = (start_piece, start_sq);

            for end_pos in BitBoard(self.end_positions[i]).get_data() {
                let data = [(Piece::None, start_position.1), (start_position.0, SQ(end_pos as u8)), (Piece::None, SQ::NONE)];
                moves.push(Move::new(data, MoveType::Bounce));

            }

            for pick_up_pos in BitBoard(self.pickup_positions[i]).get_data() {
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

    pub fn print(&self) {
        for i in 0..6 {
            println!("End {}: {}", i, BitBoard(self.end_positions[i]));
            println!("Pickup {}: {}", i, BitBoard(self.pickup_positions[i]));

        }

        println!("Drops: {}", BitBoard(self.drop_positions));

    }

}



#[repr(C)]
struct StackData {
    banned_bb: u64,
    backtrack_bb: u64,
    active_line_idx: u8,
    current_pos: u8,
    current_piece: u8,

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
