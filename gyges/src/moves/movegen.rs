//! This module contains all of the components needed for generating moves. 
//!

use crate::core::*;
use crate::board::*;
use crate::board::bitboard::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;

/// The maximum size of the stack.
const MAX_STACK_SIZE: usize = 1000;

//////////////////////////////////////////////
//////////////////// CORE ////////////////////
//////////////////////////////////////////////

/// Contains all of the core logic needed for generating moves.
/// 
/// Generation options:
///  - Moves
///  - Move count
///  - Threat count
///  - Control & Move count
///
/// Other:
/// - Quit on threat detection
/// 
pub struct MoveGen {
    stack: FixedStack<StackData>,

}

impl MoveGen {
    /// Generate the specified data.
    #[inline(always)]
    pub unsafe fn gen<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
        self.stack.clear();
        let player_bit = 1 << player as u64;

        let active_lines: [usize; 2] = board.get_active_lines();
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);
        let drop_bb = board.get_drops(active_lines, player);

        let mut result = GenResult::new(drop_bb);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            if board.piece_at(starting_sq) != Piece::None {
                let starting_piece = board.piece_at(starting_sq);

                G::init(&mut result, x, starting_sq, starting_piece);

                self.stack.push(StackData::new(Action::End, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));
                self.stack.push(StackData::new(Action::Gen, BitBoard::EMPTY, BitBoard::EMPTY, starting_sq, starting_piece, starting_sq, starting_piece, x, player));
                self.stack.push(StackData::new(Action::Start, BitBoard::EMPTY, BitBoard::EMPTY, SQ::NONE, Piece::None, starting_sq, starting_piece, 0, player));

            }

        }

        while !self.stack.is_empty() {
            let data = self.stack.pop();

            let action = data.action;
            let backtrack_board = data.backtrack_board;
            let banned_positions = data.banned_positions;
            let current_sq = data.current_sq;
            let current_piece = data.current_piece;
            let starting_sq = data.starting_sq;
            let starting_piece = data.starting_piece;
            let active_line_idx = data.active_line_idx;
            let player: Player = data.player;

            match action {
                Action::Start => {
                    board.remove(starting_sq);
                    board.piece_bb ^= starting_sq.bit();
                    continue;

                },
                Action::End => {
                    board.place(starting_piece, starting_sq);
                    board.piece_bb ^= starting_sq.bit();
                    continue;

                },
                Action::Gen => {
                    match current_piece {
                        Piece::One => {
                            let path_list = ONE_PATH_LISTS.get_unchecked(current_sq.0 as usize);
                            for i in 0..(path_list.count as usize) {
                                let path = &path_list.paths[i];
                
                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }
                
                                let end = SQ(path.0[1]);
                                let end_bit = end.bit();

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() {
                                        continue;
    
                                    }
                                    
                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    self.stack.push(StackData::new(Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                                    
                                    continue;

                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 {
                                        continue;

                                    }

                                    G::store_goal(&mut result, active_line_idx, end_bit);

                                    if Q::check_quit() {
                                        board.place(starting_piece, starting_sq);
                                        board.piece_bb ^= starting_sq.bit();
                                        
                                        result.threat = true;

                                        return result;

                                    }

                                    continue;

                                }

                                G::store_end(&mut result, active_line_idx, end_bit);

                            }

                        },
                        Piece::Two => {
                            let intercepts = ALL_TWO_INTERCEPTS[current_sq.0 as usize];
                            let intercept_bb = board.piece_bb & intercepts;

                            let key = unsafe { compress_pext(intercepts, intercept_bb.0) };            
                            let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(key as usize);

                            let path_list = TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
                            for i in 0..(path_list.count as usize) {
                                let path = &path_list.paths[i];

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                let end = SQ(path.0[2]);
                                let end_bit = end.bit();

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() {
                                        continue;
    
                                    }

                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    self.stack.push(StackData::new(Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                                    
                                    continue;
                                
                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 {
                                        continue;

                                    }

                                    G::store_goal(&mut result, active_line_idx, end_bit);

                                    if Q::check_quit() {
                                        board.place(starting_piece, starting_sq);
                                        board.piece_bb ^= starting_sq.bit();

                                        result.threat = true;

                                        return result;

                                    }

                                    continue;

                                }

                                G::store_end(&mut result, active_line_idx, end_bit);        

                            }

                        },
                        Piece::Three => {
                            let intercepts = ALL_THREE_INTERCEPTS[current_sq.0 as usize];
                            let intercept_bb = board.piece_bb & intercepts;
                            
                            let key = unsafe { compress_pext(intercepts, intercept_bb.0) };            
                            let valid_paths_idx: &u16 = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(key as usize);
        
                            let path_list = THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);
                            for i in 0..(path_list.count as usize) {
                                let path = &path_list.paths[i];

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                let end = SQ(path.0[3]);
                                let end_bit = end.bit();

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() {
                                        continue;
    
                                    }

                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    self.stack.push(StackData::new(Action::Gen, new_backtrack_board, new_banned_positions, end, end_piece, starting_sq, starting_piece, active_line_idx, player));
                                    
                                    continue;

                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 {
                                        continue;

                                    }

                                    G::store_goal(&mut result, active_line_idx, end_bit);

                                    if Q::check_quit() {
                                        board.place(starting_piece, starting_sq);
                                        board.piece_bb ^= starting_sq.bit();

                                        result.threat = true;

                                        return result;

                                    }

                                    continue;

                                }

                                G::store_end(&mut result, active_line_idx, end_bit);

                            }

                        },
                        Piece::None => {}

                    }

                }

            }

        }

        G::exit(&mut result, board);

        result

    }

}

impl Default for MoveGen {
    fn default() -> Self {
        Self {
            stack: FixedStack::new(MAX_STACK_SIZE),

        }

    }

}

//////////////////////////////////////////////////////////
//////////////////// GENERATION TYPES ////////////////////
//////////////////////////////////////////////////////////

/// The type of generation to perform.
pub trait GenType {
    fn init(result: &mut GenResult, x: usize, starting_sq: SQ, starting_piece: Piece); // Initializes the generation
    fn store_bounce(result: &mut GenResult, active_line_idx: usize, end_bit: u64); // Stores relevant data for bounce
    fn store_end(result: &mut GenResult, active_line_idx: usize, end_bit: u64);    // Stores relevant data for the end of a path
    fn store_goal(result: &mut GenResult, active_line_idx: usize, end_bit: u64);   // Stores relevant data for a goal
    fn exit(result: &mut GenResult, board: &mut BoardState); // Exits the generation

}

/// Generate the valid moves.
pub struct GenMoves;
impl GenType for GenMoves {
    fn init(result: &mut GenResult, x: usize, starting_sq: SQ, starting_piece: Piece) {
        result.move_list.set_start(x, starting_sq, starting_piece);
    }
    fn store_bounce(result: &mut GenResult, active_line_idx: usize, end_bit: u64) {
        result.move_list.set_pickup_position(active_line_idx, end_bit);
    }
    fn store_end(result: &mut GenResult, active_line_idx: usize, end_bit: u64) {
        result.move_list.set_end_position(active_line_idx, end_bit);
    }
    fn store_goal(result: &mut GenResult, active_line_idx: usize, end_bit: u64) {
        result.move_list.set_end_position(active_line_idx, end_bit);
    }
    fn exit(_: &mut GenResult, _: &mut BoardState) {}

}

/// Generate the move count.
pub struct GenMoveCount;
impl GenType for GenMoveCount {
    fn init(_: &mut GenResult, _: usize, _: SQ, _: Piece) {}
    fn store_bounce(result: &mut GenResult, _: usize, _: u64) {
        result.move_count += 25;
    }
    fn store_end(result: &mut GenResult, _: usize, _: u64) {
        result.move_count += 1;
    }
    fn store_goal(result: &mut GenResult, _: usize, _: u64) {
        result.move_count += 1;
    }
    fn exit(_: &mut GenResult, _: &mut BoardState) {}

}

pub struct GenThreatCount;
impl GenType for GenThreatCount {
    fn init(_: &mut GenResult, _: usize, _: SQ, _: Piece) {}
    fn store_bounce(_: &mut GenResult, _: usize, _: u64) {}
    fn store_end(_: &mut GenResult, _: usize, _: u64) {}
    fn store_goal(result: &mut GenResult, _: usize, _: u64) {
        result.threat_count += 1;
    }
    fn exit(_: &mut GenResult, _: &mut BoardState) {}

}

/// Generate the controlled squares, controlled pieces, and the move count.
pub struct GenControlMoveCount;
impl GenType for GenControlMoveCount {
    fn init(result: &mut GenResult, _: usize, starting_sq: SQ, _: Piece) {
        result.controlled_pieces |= starting_sq.bit();

    }
    fn store_bounce(result: &mut GenResult, _: usize, end_bit: u64) {
        result.move_count += 25;
        result.controlled_pieces |= end_bit;
    }
    fn store_end(result: &mut GenResult, _: usize, end_bit: u64) {
        result.move_count += 1;
        result.controlled_squares |= end_bit;
    }
    fn store_goal(result: &mut GenResult, _: usize, _: u64) {
        result.move_count += 1;
    }
    fn exit(result: &mut GenResult, board: &mut BoardState) {
        result.controlled_squares &= !board.piece_bb;
    }

}

/// Generates no data. Used when only threat detection is needed.
pub struct GenNone;
impl GenType for GenNone {
    fn init(_: &mut GenResult, _: usize, _: SQ, _: Piece) {}
    fn store_bounce(_: &mut GenResult, _: usize, _: u64) {}
    fn store_end(_: &mut GenResult, _: usize, _: u64) {}
    fn store_goal(_: &mut GenResult, _: usize, _: u64) {}
    fn exit(_: &mut GenResult, _: &mut BoardState) {}

}

////////////////////////////////////////////////////
//////////////////// QUIT TYPES ////////////////////
////////////////////////////////////////////////////


/// The type of quit to perform.
pub trait QuitType {
    fn check_quit() -> bool;
}

/// Quit when the first threat is found.
pub struct QuitOnThreat;
impl QuitType for QuitOnThreat {
    fn check_quit() -> bool {
        true
    }

}

/// Do not quit until full generation is complete.
pub struct NoQuit;
impl QuitType for NoQuit {
    fn check_quit() -> bool {
        false
    }

}


///////////////////////////////////////////////////////
//////////////////// RANDOM HELPER ////////////////////
///////////////////////////////////////////////////////

/// The type of action specific to an item on the stack.
#[derive(PartialEq, Debug, Clone, Copy)]
enum Action {
    Gen,
    Start,
    End

}

/// The data stored on the stack.
struct StackData {
    pub action: Action,
    pub backtrack_board: BitBoard,
    pub banned_positions: BitBoard,
    pub current_sq: SQ,
    pub current_piece: Piece,
    pub starting_sq: SQ,
    pub starting_piece: Piece,
    pub active_line_idx: usize,
    pub player: Player
}

impl StackData {
    pub fn new(action: Action, backtrack_board: BitBoard, banned_positions: BitBoard, current_sq: SQ, current_piece: Piece, starting_sq: SQ, starting_piece: Piece, active_line_idx: usize, player: Player) -> Self {
        Self {
            action,
            backtrack_board,
            banned_positions,
            current_sq,
            current_piece,
            starting_sq,
            starting_piece,
            active_line_idx,
            player
        }

    }

}

/// A hyper efficient fixed-size stack allocated on the heap.
struct FixedStack<T> {
    buffer: Box<[T]>,
    top: usize,
}

impl<T> FixedStack<T> {
    /// Creates a new fixed-size stack with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let buffer = unsafe {
            let mut vec = Vec::with_capacity(capacity);
            vec.set_len(capacity);
            vec.into_boxed_slice()
        };

        Self { buffer, top: 0 }

    }

    /// Pushes a value onto the top of the stack.
    #[inline(always)]
    pub unsafe fn push(&mut self, value: T) {
        if self.top >= self.buffer.len() {
            panic!("Stack overflow!");
        }

        *self.buffer.get_unchecked_mut(self.top) = value;
        self.top += 1;

    }

    /// Pops a value from the top of the stack.
    #[inline(always)]
    pub unsafe fn pop(&mut self) -> T {
        if self.top == 0 {
            panic!("Stack underflow!");
        }

        self.top -= 1;
        std::ptr::read(self.buffer.get_unchecked(self.top))

    }

    /// Returns true if the stack is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.top == 0

    }

    /// Clears the stack without modifying memory.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.top = 0;

    }

}

impl<T> Drop for FixedStack<T> {
    // Custom drop to properly clean up memory
    fn drop(&mut self) {
        for i in 0..self.top {
            unsafe {
                std::ptr::drop_in_place(self.buffer.get_unchecked_mut(i));

            }

        }

    }

}

/// Extracts specific bits from a 64-bit integer using a mask. 
#[inline(always)]
pub unsafe fn compress_pext(mask: u64, val: u64) -> u16 {
    core::arch::x86_64::_pext_u64(val, mask) as u16

}

/// The result of a move generation.
/// 
/// Threat is mututally exclusive with all other fields.
#[derive(Debug, Clone)]
pub struct GenResult {
    pub threat: bool,
    pub threat_count: usize,
    pub move_count: usize,
    pub move_list: RawMoveList,
    pub controlled_squares: BitBoard,
    pub controlled_pieces: BitBoard

}

impl GenResult {
    pub fn new(drop_bb: BitBoard) -> Self {
        Self {
            threat: false,
            threat_count: 0,
            move_count: 0,
            move_list: RawMoveList::new(drop_bb),
            controlled_squares: BitBoard::EMPTY,
            controlled_pieces: BitBoard::EMPTY

        }

    }

}
