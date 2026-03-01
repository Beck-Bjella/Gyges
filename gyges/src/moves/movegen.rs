//! This module contains all of the components needed for generating moves. 
//!
//! 

use crate::core::*;
use crate::board::*;
use crate::board::bitboard::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;
use crate::moves::new_movegen_consts::*;

use crate::moves::path_data::*;
use crate::moves::three_dir_tables::*;
use crate::moves::two_dir_tables::*;
use crate::moves::one_dir_tables::*;

/// The maximum size of the stack.
const MAX_STACK_SIZE: usize = 1000;

//////////////////////////////////////////////
//////////////////// CORE ////////////////////
//////////////////////////////////////////////

/// Contains all of the core logic needed for generating moves.
/// 
pub struct MoveGen {
    stack: FixedStack<StackData>,
    stack2: FixedStack<StackData2>,

}

impl MoveGen {
    /// Generate the specified data. - V1 (OLD CONSTS)
    #[inline(always)]
    pub unsafe fn gen_old<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
        self.stack.clear();
        let player_bit = 1 << player as u64;

        let active_lines: [usize; 2] = board.get_active_lines();
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);
        let drop_bb = board.get_drops(active_lines, player);

        let mut result = GenResult::new(drop_bb);

        for x in 0..6 {
            let starting_sq = active_line_sq + x;
            let starting_piece = board.piece_at(starting_sq);
            if starting_piece != Piece::None {
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

                                    if Q::QUIT {
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

                                    if Q::QUIT {
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

                                    if Q::QUIT {
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

    #[inline(always)]
    pub unsafe fn gen<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
        self.stack2.clear();
        let player_bit = 1 << player as u64;

        let active_lines: [usize; 2] = board.get_active_lines();
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);
        let drop_bb = board.get_drops(active_lines, player);

        let mut result = GenResult::new(drop_bb);

        for active_line_idx in 0..6 {
            let starting_sq = active_line_sq + active_line_idx;
            let starting_piece = board.piece_at(starting_sq);
            if starting_piece != Piece::None {
                G::init(&mut result, active_line_idx, starting_sq, starting_piece);

                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();

                self.stack2.push(StackData2::new(Action2::from_piece(starting_piece), BitBoard::EMPTY, starting_sq));
                
                let mut quit = false;
                let mut start = true;
                while !self.stack2.is_empty() {
                    let data: StackData2 = self.stack2.pop();

                    let action = data.action;
                    let backtrack_board = data.backtrack_board;
                    let current_sq = data.current_sq;

                    quit = match action {
                        Action2::Gen1 => MoveGen::process_paths::<OneBounce, G, Q>(board, &mut result, &mut self.stack2, current_sq, backtrack_board, active_line_idx, player_bit, start),
                        Action2::Gen2 => MoveGen::process_paths::<TwoBounce, G, Q>(board, &mut result, &mut self.stack2, current_sq, backtrack_board, active_line_idx, player_bit, start),
                        Action2::Gen3 => MoveGen::process_paths::<ThreeBounce, G, Q>(board, &mut result, &mut self.stack2, current_sq, backtrack_board, active_line_idx, player_bit, start),
                    };

                    if start {

                        start = false;

                    }

                    if quit {
                        break;
                    }

                }

                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();

                if quit {
                    result.threat = true;
                    return result;

                }

            }

        }

        G::exit(&mut result, board);

      

        result

    }

    
    #[inline(always)]
    pub unsafe fn gen_new<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
        self.stack2.clear();
        let player_bit = 1 << player as u64;

        let active_lines: [usize; 2] = board.get_active_lines();
        let active_line_sq = SQ((active_lines[player as usize] * 6) as u8);
        let drop_bb = board.get_drops(active_lines, player);

        let mut result = GenResult::new(drop_bb);

        for active_line_idx in 0..6 {
            let starting_sq = active_line_sq + active_line_idx;
            let starting_piece = board.piece_at(starting_sq);
            if starting_piece != Piece::None {
                G::init(&mut result, active_line_idx, starting_sq, starting_piece);

                board.remove(starting_sq);
                board.piece_bb ^= starting_sq.bit();

                self.stack2.push(StackData2::new(Action2::from_piece(starting_piece), BitBoard::EMPTY, starting_sq));
                
                let mut quit = false;
                let mut start = true;
                while !self.stack2.is_empty() {
                    let data: StackData2 = self.stack2.pop();

                    let action = data.action;
                    let backtrack_board = data.backtrack_board;
                    let current_sq = data.current_sq;

                    quit = match action {
                        Action2::Gen1 => MoveGen::process_paths2::<One, G, Q>(board, &mut result, &mut self.stack2, current_sq, backtrack_board, active_line_idx, player_bit, start),
                        Action2::Gen2 => MoveGen::process_paths2::<Two, G, Q>(board, &mut result, &mut self.stack2, current_sq, backtrack_board, active_line_idx, player_bit, start),
                        Action2::Gen3 => MoveGen::process_paths2::<Three, G, Q>(board, &mut result, &mut self.stack2, current_sq, backtrack_board, active_line_idx, player_bit, start),
                    };

                    if start {

                        start = false;

                    }

                    if quit {
                        break;
                    }

                }

                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();

                if quit {
                    result.threat = true;
                    return result;

                }

            }

        }

        G::exit(&mut result, board);

      

        result

    }

    #[inline(always)]
    unsafe fn process_paths<P: PieceLookup, G: GenType, Q: QuitType>(
        board:            &mut BoardState,
        result:           &mut GenResult,
        stack:            &mut FixedStack<StackData2>,
        current_sq:       SQ,
        backtrack_board:  BitBoard,
        active_line_idx:  usize,
        player_bit:       u64,
        start: bool
    ) -> bool {
        let (count, ends, backs) = P::get_path_data(current_sq, board.piece_bb);

        let position_intercepts_mask = if !start {
            POSITION_INTERCEPTIONS[current_sq.0 as usize]

        } else {
            0

        };

        for i in 0..(count as usize) {
            let end = SQ(*ends.add(i));
            let end_bit = end.bit();
            let back: u64 = *backs.add(i);

            if (backtrack_board & back).is_not_empty() {
                continue;

            }

            if (board.piece_bb & end_bit).is_not_empty() {
                G::store_bounce(result, active_line_idx, end_bit);

                stack.push(StackData2::new(
                    Action2::from_piece(board.piece_at(end)),
                    backtrack_board | back | position_intercepts_mask,
                    end
                ));

                continue; 

            }

            let goal_bit = BitBoard(end_bit >> 36);
            if goal_bit.is_not_empty() {
                if (goal_bit & player_bit).is_not_empty() {
                    continue;

                }

                G::store_goal(result, active_line_idx, end_bit);

                if Q::QUIT {
                    return true;
                    
                }

                continue;

            }

            G::store_end(result, active_line_idx, end_bit);

        }

        false

    }

    #[inline(always)]
    unsafe fn process_paths2<P: PieceTables, G: GenType, Q: QuitType>(
        board:           &mut BoardState,
        result:          &mut GenResult,
        stack:           &mut FixedStack<StackData2>,
        current_sq:      SQ,
        backtrack_board: BitBoard,
        active_line_idx: usize,
        player_bit:      u64,
        start:           bool,
    ) -> bool {
        let sq  = current_sq.0 as usize;
        let pbb = board.piece_bb.0;

        
        let blocked = P::piece_blocked(sq, pbb) | backtrack_board.0;
        let (n, s, e, w) = P::dir_indices(sq, blocked);

        let n = P::path_data(n);
        let s = P::path_data(s);
        let e = P::path_data(e);
        let w = P::path_data(w);

        // bulk ends and goals
        let ends  = (n.end_bits | s.end_bits | e.end_bits | w.end_bits) & !pbb;
        let goals = (n.goal_bits | s.goal_bits | e.goal_bits | w.goal_bits) & !player_bit;

        G::store_end(result,  active_line_idx, ends);

        let valid_goals = (goals >> 36) & !player_bit;
        if valid_goals != 0{
            G::store_goal(result, active_line_idx, goals);

            if Q::QUIT {
                return true;
            }

        }

        // bounces
        let position_intercepts_mask = if !start { POSITION_INTERCEPTIONS[sq] } else { 0 };
        for d in [n, s, e, w] {
            for p in 0..d.count {
                let end = SQ(*d.ends.add(p));
                let end_bit = 1u64 << end.0;

                if (pbb & end_bit) != 0 {
                    G::store_bounce(result, active_line_idx, end_bit);
                    stack.push(StackData2::new(
                        Action2::from_piece(board.piece_at(end)),
                        backtrack_board | *d.backs.add(p) | position_intercepts_mask,
                        end,
                    ));
                }
            }
        }

        false
    }

}

impl Default for MoveGen {
    fn default() -> Self {
        Self {
            stack: FixedStack::new(MAX_STACK_SIZE),
            stack2: FixedStack::new(MAX_STACK_SIZE),

        }

    }

}

//////////////////////////////////////////////////////////
////////////////////// LOOKUP TYPES //////////////////////
//////////////////////////////////////////////////////////
/// 


#[derive(Copy, Clone)]
pub struct PathEntry {
    pub end_bits:  u64,
    pub goal_bits: u64,
    pub count:     usize,
    pub ends:      *const u8,
    pub backs:     *const u64,
}

unsafe impl Send for PathEntry {}
unsafe impl Sync for PathEntry {}

pub trait PieceTables {
    fn north_mask(sq: usize) -> u64;
    fn south_mask(sq: usize) -> u64;
    fn east_mask(sq:  usize) -> u64;
    fn west_mask(sq:  usize) -> u64;

    fn north_idx(sq: usize, key: usize) -> usize;
    fn south_idx(sq: usize, key: usize) -> usize;
    fn east_idx(sq:  usize, key: usize) -> usize;
    fn west_idx(sq:  usize, key: usize) -> usize;

    // individual accessors — kept for other uses outside process_paths
    fn path_count(idx: usize)          -> usize;
    fn path_end(idx: usize, p: usize)  -> u8;
    fn path_back(idx: usize, p: usize) -> u64;
    fn end_bits(idx: usize)            -> u64;
    fn goal_bits(idx: usize)           -> u64;

    // single fetch — all data needed for one path list, named fields
    fn path_data(idx: usize) -> PathEntry;

    fn piece_blocked(sq: usize, piece_bb: u64) -> u64;

    #[inline(always)]
    fn dir_indices(sq: usize, blocked: u64) -> (usize, usize, usize, usize) {
        (
            Self::north_idx(sq, unsafe { compress_pext(Self::north_mask(sq), blocked) as usize }),
            Self::south_idx(sq, unsafe { compress_pext(Self::south_mask(sq), blocked) as usize }),
            Self::east_idx(sq,  unsafe { compress_pext(Self::east_mask(sq),  blocked) as usize }),
            Self::west_idx(sq,  unsafe { compress_pext(Self::west_mask(sq),  blocked) as usize }),
        )
    }

    #[inline(always)]
    fn lookup(sq: usize, piece_bb: u64, backtrack: u64) -> (usize, usize, usize, usize, u64, u64) {
        let blocked = Self::piece_blocked(sq, piece_bb) | backtrack;
        let (n, s, e, w) = Self::dir_indices(sq, blocked);
        (
            n, s, e, w,
            Self::end_bits(n)  | Self::end_bits(s)  | Self::end_bits(e)  | Self::end_bits(w),
            Self::goal_bits(n) | Self::goal_bits(s) | Self::goal_bits(e) | Self::goal_bits(w),
        )
    }
}

// ── ONE ───────────────────────────────────────────────────────────────────────
pub struct One;
impl PieceTables for One {
    #[inline(always)] fn north_mask(sq: usize) -> u64 { unsafe { *ONE_NORTH_MASK.get_unchecked(sq) } }
    #[inline(always)] fn south_mask(sq: usize) -> u64 { unsafe { *ONE_SOUTH_MASK.get_unchecked(sq) } }
    #[inline(always)] fn east_mask(sq:  usize) -> u64 { unsafe { *ONE_EAST_MASK.get_unchecked(sq)  } }
    #[inline(always)] fn west_mask(sq:  usize) -> u64 { unsafe { *ONE_WEST_MASK.get_unchecked(sq)  } }

    #[inline(always)] fn north_idx(sq: usize, key: usize) -> usize { unsafe { *ONE_NORTH_TABLE.get_unchecked(sq).get_unchecked(key) as usize } }
    #[inline(always)] fn south_idx(sq: usize, key: usize) -> usize { unsafe { *ONE_SOUTH_TABLE.get_unchecked(sq).get_unchecked(key) as usize } }
    #[inline(always)] fn east_idx(sq:  usize, key: usize) -> usize { unsafe { *ONE_EAST_TABLE.get_unchecked(sq).get_unchecked(key)  as usize } }
    #[inline(always)] fn west_idx(sq:  usize, key: usize) -> usize { unsafe { *ONE_WEST_TABLE.get_unchecked(sq).get_unchecked(key)  as usize } }

    #[inline(always)] fn path_count(idx: usize)          -> usize { unsafe { *ONE_PATH_COUNT.get_unchecked(idx) as usize } }
    #[inline(always)] fn path_end(idx: usize, p: usize)  -> u8    { unsafe { *ONE_PATH_ENDS.get_unchecked(idx).get_unchecked(p) } }
    #[inline(always)] fn path_back(idx: usize, p: usize) -> u64   { unsafe { *ONE_PATH_BACKS.get_unchecked(idx).get_unchecked(p) } }
    #[inline(always)] fn end_bits(idx: usize)            -> u64   { unsafe { *ONE_PATH_END_BITS.get_unchecked(idx) } }
    #[inline(always)] fn goal_bits(idx: usize)           -> u64   { unsafe { *ONE_PATH_GOAL_BITS.get_unchecked(idx) } }

    #[inline(always)]
    fn path_data(idx: usize) -> PathEntry {
        unsafe {
            let d = ONE_PATH_DATA.get_unchecked(idx);
            PathEntry { end_bits: d.end_bits, goal_bits: d.goal_bits, count: d.count as usize, ends: d.ends.as_ptr(), backs: d.backs.as_ptr() }
        }
    }

    #[inline(always)] fn piece_blocked(_sq: usize, _piece_bb: u64) -> u64 { 0 }
}

// ── TWO ───────────────────────────────────────────────────────────────────────
pub struct Two;
impl PieceTables for Two {
    #[inline(always)] fn north_mask(sq: usize) -> u64 { unsafe { *TWO_NORTH_MASK.get_unchecked(sq) } }
    #[inline(always)] fn south_mask(sq: usize) -> u64 { unsafe { *TWO_SOUTH_MASK.get_unchecked(sq) } }
    #[inline(always)] fn east_mask(sq:  usize) -> u64 { unsafe { *TWO_EAST_MASK.get_unchecked(sq)  } }
    #[inline(always)] fn west_mask(sq:  usize) -> u64 { unsafe { *TWO_WEST_MASK.get_unchecked(sq)  } }

    #[inline(always)] fn north_idx(sq: usize, key: usize) -> usize { unsafe { *TWO_NORTH_TABLE.get_unchecked(sq).get_unchecked(key) as usize } }
    #[inline(always)] fn south_idx(sq: usize, key: usize) -> usize { unsafe { *TWO_SOUTH_TABLE.get_unchecked(sq).get_unchecked(key) as usize } }
    #[inline(always)] fn east_idx(sq:  usize, key: usize) -> usize { unsafe { *TWO_EAST_TABLE.get_unchecked(sq).get_unchecked(key)  as usize } }
    #[inline(always)] fn west_idx(sq:  usize, key: usize) -> usize { unsafe { *TWO_WEST_TABLE.get_unchecked(sq).get_unchecked(key)  as usize } }

    #[inline(always)] fn path_count(idx: usize)          -> usize { unsafe { *TWO_PATH_COUNT.get_unchecked(idx) as usize } }
    #[inline(always)] fn path_end(idx: usize, p: usize)  -> u8    { unsafe { *TWO_PATH_ENDS.get_unchecked(idx).get_unchecked(p) } }
    #[inline(always)] fn path_back(idx: usize, p: usize) -> u64   { unsafe { *TWO_PATH_BACKS.get_unchecked(idx).get_unchecked(p) } }
    #[inline(always)] fn end_bits(idx: usize)            -> u64   { unsafe { *TWO_PATH_END_BITS.get_unchecked(idx) } }
    #[inline(always)] fn goal_bits(idx: usize)           -> u64   { unsafe { *TWO_PATH_GOAL_BITS.get_unchecked(idx) } }

    #[inline(always)]
    fn path_data(idx: usize) -> PathEntry {
        unsafe {
            let d = TWO_PATH_DATA.get_unchecked(idx);
            PathEntry { end_bits: d.end_bits, goal_bits: d.goal_bits, count: d.count as usize, ends: d.ends.as_ptr(), backs: d.backs.as_ptr() }
        }
    }

    #[inline(always)]
    fn piece_blocked(sq: usize, piece_bb: u64) -> u64 {
        let mut extra = 0u64;
        let mut bb = unsafe { piece_bb & *ALL_TWO_INTERCEPTS.get_unchecked(sq) };
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            extra |= unsafe { *TWO_PIECE_INTERCEPTS.get_unchecked(sq).get_unchecked(s) };
            bb &= bb - 1;
        }
        extra
    }
}

// ── THREE ─────────────────────────────────────────────────────────────────────
pub struct Three;
impl PieceTables for Three {
    #[inline(always)] fn north_mask(sq: usize) -> u64 { unsafe { *THREE_NORTH_MASK.get_unchecked(sq) } }
    #[inline(always)] fn south_mask(sq: usize) -> u64 { unsafe { *THREE_SOUTH_MASK.get_unchecked(sq) } }
    #[inline(always)] fn east_mask(sq:  usize) -> u64 { unsafe { *THREE_EAST_MASK.get_unchecked(sq)  } }
    #[inline(always)] fn west_mask(sq:  usize) -> u64 { unsafe { *THREE_WEST_MASK.get_unchecked(sq)  } }

    #[inline(always)] fn north_idx(sq: usize, key: usize) -> usize { unsafe { *THREE_NORTH_TABLE.get_unchecked(sq).get_unchecked(key) as usize } }
    #[inline(always)] fn south_idx(sq: usize, key: usize) -> usize { unsafe { *THREE_SOUTH_TABLE.get_unchecked(sq).get_unchecked(key) as usize } }
    #[inline(always)] fn east_idx(sq:  usize, key: usize) -> usize { unsafe { *THREE_EAST_TABLE.get_unchecked(sq).get_unchecked(key)  as usize } }
    #[inline(always)] fn west_idx(sq:  usize, key: usize) -> usize { unsafe { *THREE_WEST_TABLE.get_unchecked(sq).get_unchecked(key)  as usize } }

    #[inline(always)] fn path_count(idx: usize)          -> usize { unsafe { *THREE_PATH_COUNT.get_unchecked(idx) as usize } }
    #[inline(always)] fn path_end(idx: usize, p: usize)  -> u8    { unsafe { *THREE_PATH_ENDS.get_unchecked(idx).get_unchecked(p) } }
    #[inline(always)] fn path_back(idx: usize, p: usize) -> u64   { unsafe { *THREE_PATH_BACKS.get_unchecked(idx).get_unchecked(p) } }
    #[inline(always)] fn end_bits(idx: usize)            -> u64   { unsafe { *THREE_PATH_END_BITS.get_unchecked(idx) } }
    #[inline(always)] fn goal_bits(idx: usize)           -> u64   { unsafe { *THREE_PATH_GOAL_BITS.get_unchecked(idx) } }

    #[inline(always)]
    fn path_data(idx: usize) -> PathEntry {
        unsafe {
            let d = THREE_PATH_DATA.get_unchecked(idx);
            PathEntry { end_bits: d.end_bits, goal_bits: d.goal_bits, count: d.count as usize, ends: d.ends.as_ptr(), backs: d.backs.as_ptr() }
        }
    }

    #[inline(always)]
    fn piece_blocked(sq: usize, piece_bb: u64) -> u64 {
        let mut extra = 0u64;
        let mut bb = unsafe { piece_bb & *ALL_THREE_INTERCEPTS.get_unchecked(sq) };
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            extra |= unsafe { *THREE_PIECE_INTERCEPTS.get_unchecked(sq).get_unchecked(s) };
            bb &= bb - 1;
        }
        extra
    }
}







pub trait PieceLookup {
    unsafe fn get_path_data(sq: SQ, piece_bb: BitBoard) -> (u8, *const u8, *const u64);
}

pub struct OneBounce;
impl PieceLookup for OneBounce {
    #[inline(always)]
    unsafe fn get_path_data(sq: SQ, _piece_bb: BitBoard) -> (u8, *const u8, *const u64) {
        (
            *ONE_COUNTS.get_unchecked(sq.0 as usize),
            ONE_ENDS.get_unchecked(sq.0 as usize).as_ptr(),
            ONE_BACKS.get_unchecked(sq.0 as usize).as_ptr(),
        )

    }

}

pub struct TwoBounce;
impl PieceLookup for TwoBounce {
    #[inline(always)]
    unsafe fn get_path_data(sq: SQ, piece_bb: BitBoard) -> (u8, *const u8, *const u64) {
        let intercepts = ALL_TWO_INTERCEPTS[sq.0 as usize];
        let intercept_bb = piece_bb & intercepts;
        let key = compress_pext(intercepts, intercept_bb.0);
        let idx = *TWO_MAP.get_unchecked(sq.0 as usize).get_unchecked(key as usize) as usize;

        (
            *TWO_COUNTS.get_unchecked(idx),
            TWO_ENDS.get_unchecked(idx).as_ptr(),
            TWO_BACKS.get_unchecked(idx).as_ptr(),
        )

    }

}

pub struct ThreeBounce;
impl PieceLookup for ThreeBounce {
    #[inline(always)]
    unsafe fn get_path_data(sq: SQ, piece_bb: BitBoard) -> (u8, *const u8, *const u64) {
        let intercepts   = ALL_THREE_INTERCEPTS[sq.0 as usize];
        let intercept_bb = piece_bb & intercepts;
        let key = compress_pext(intercepts, intercept_bb.0);
        let idx = *THREE_MAP.get_unchecked(sq.0 as usize).get_unchecked(key as usize) as usize;

        (
            *THREE_COUNTS.get_unchecked(idx),
            THREE_ENDS.get_unchecked(idx).as_ptr(),
            THREE_BACKS.get_unchecked(idx).as_ptr(),
        )

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
    const QUIT: bool;
}

/// Quit when the first threat is found.
pub struct QuitOnThreat;
impl QuitType for QuitOnThreat {
    const QUIT: bool = true;

}

/// Do not quit until full generation is complete.
pub struct NoQuit;
impl QuitType for NoQuit {
    const QUIT: bool = false;

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

/// The type of action specific to an item on the stack.
#[derive(PartialEq, Debug, Clone, Copy)]
enum Action2 {
    Gen1,
    Gen2,
    Gen3,

}

impl Action2 {
    #[inline(always)]
    pub fn from_piece(piece: Piece) -> Self {
        match piece {
            Piece::One => Action2::Gen1,
            Piece::Two => Action2::Gen2,
            Piece::Three => Action2::Gen3,
            Piece::None => panic!("Invalid piece for Action2"),
        }

    }
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

struct StackData2 {
    pub backtrack_board: BitBoard,
    pub current_sq: SQ,
    pub action: Action2

}

impl StackData2 {
    pub fn new(action: Action2, backtrack_board: BitBoard, current_sq: SQ) -> Self {
        Self {
            backtrack_board,
            current_sq,
            action
            
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

    #[inline(always)]
    pub unsafe fn push_in_place(&mut self) -> &mut T {
        if self.top >= self.buffer.len() {
            panic!("Stack overflow!");
        }
        let slot = self.buffer.get_unchecked_mut(self.top);
        self.top += 1;
        slot
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
