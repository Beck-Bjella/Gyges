//! This module contains all of the components needed for generating moves. 
//!
//! 
use crate::core::*;
use crate::board::*;
use crate::board::bitboard::*;
use crate::moves::move_list::*;
use crate::moves::movegen_consts::*;
use crate::moves::new_movegen_consts::*;

// use crate::moves::new_consts::ONE_BLOB;
// use crate::moves::new_consts::ONE_META;
// use crate::moves::new_consts::PackedPath;
// use crate::moves::new_consts::Meta;
// use crate::moves::new_consts::TWO_BLOB;
// use crate::moves::new_consts::TWO_META;
// use crate::moves::threes_compact::THREE_META;
// use crate::moves::threes_compact::THREE_BLOB;
// use crate::moves::new_consts::ALL_INTERCEPTS;
// use crate::moves::new_consts::COMBINED_MAP;
// use crate::moves::new_consts::COMBINED_META;
// use crate::moves::new_consts::COMBINED_BLOB;
// use crate::moves::new_consts::MAP_BASE;
// use crate::moves::new_consts::MAP_STRIDE;


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
    pub unsafe fn gen<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
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

    // REALLY GOOD 
    #[inline(always)]
    pub unsafe fn gen4<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
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

                self.stack2.push(StackData2::new(Action2::from_piece(starting_piece), BitBoard::EMPTY, BitBoard::EMPTY, starting_sq));
                
                while !self.stack2.is_empty() {
                    let data = self.stack2.pop();

                    let action = data.action;
                    let backtrack_board = data.backtrack_board;
                    let banned_positions = data.banned_positions;
                    let current_sq = data.current_sq;

                    match action {
                        Action2::Gen1 => {
                            let path_list = ONE_PATH_LISTS.get_unchecked(current_sq.0 as usize);

                            for i in 0..(path_list.count as usize) {
                                let path = &path_list.paths[i];
                                let end = SQ(path.0[1]);
                                let end_bit = end.bit();

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() || (backtrack_board & path.1).is_not_empty() {
                                        continue;

                                    }
                                    
                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    self.stack2.push(StackData2::new(Action2::from_piece(end_piece), new_backtrack_board, new_banned_positions, end));

                                    continue;

                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 || (backtrack_board & path.1).is_not_empty() {
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

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                G::store_end(&mut result, active_line_idx, end_bit);

                            }
                    
                        }
                        Action2::Gen2 => {
                            let intercepts = ALL_TWO_INTERCEPTS[current_sq.0 as usize];
                            let intercept_bb = board.piece_bb & intercepts;

                            let key = unsafe { compress_pext(intercepts, intercept_bb.0) };            
                            let valid_paths_idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(key as usize);

                            let path_list = TWO_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                            for i in 0..(path_list.count as usize) {
                                let path = &path_list.paths[i];
                                let end = SQ(path.0[2]);
                                let end_bit = end.bit();

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() || (backtrack_board & path.1).is_not_empty() {
                                        continue;

                                    }

                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    self.stack2.push(StackData2::new(Action2::from_piece(end_piece), new_backtrack_board, new_banned_positions, end));

                                    continue;
                                
                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 || (backtrack_board & path.1).is_not_empty() {
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

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                G::store_end(&mut result, active_line_idx, end_bit);        

                            }

                        },
                        Action2::Gen3 => {
                            let intercepts = ALL_THREE_INTERCEPTS[current_sq.0 as usize];
                            let intercept_bb = board.piece_bb & intercepts;
                            
                            let key = unsafe { compress_pext(intercepts, intercept_bb.0) };            
                            let valid_paths_idx: &u16 = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(key as usize);

                            let path_list = THREE_PATH_LISTS.get_unchecked(*valid_paths_idx as usize);

                            for i in 0..(path_list.count as usize) {
                                let path = &path_list.paths[i];
                                let end = SQ(path.0[3]);
                                let end_bit = end.bit();

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() || (backtrack_board & path.1).is_not_empty() {
                                        continue;

                                    }

                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ path.1;
                                    
                                    self.stack2.push(StackData2::new(Action2::from_piece(end_piece), new_backtrack_board, new_banned_positions, end));
    
                                    continue;
                                
                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 || (backtrack_board & path.1).is_not_empty() {
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

                                if (backtrack_board & path.1).is_not_empty() {
                                    continue;
                    
                                }

                                G::store_end(&mut result, active_line_idx, end_bit);

                            }

                        }

                    }

                }
                
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();

            }

        }

        G::exit(&mut result, board);

        result

    }

    // REALLY GOOD + new consts
    #[inline(always)]
    pub unsafe fn gen5<G: GenType, Q: QuitType>(&mut self, board: &mut BoardState, player: Player) -> GenResult {
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

                self.stack2.push(StackData2::new(Action2::from_piece(starting_piece), BitBoard::EMPTY, BitBoard::EMPTY, starting_sq));
                
                while !self.stack2.is_empty() {
                    let data = self.stack2.pop();

                    let action = data.action;
                    let backtrack_board = data.backtrack_board;
                    let banned_positions = data.banned_positions;
                    let current_sq = data.current_sq;

                    match action {
                        Action2::Gen1 => {
                            let count = *ONE_COUNTS.get_unchecked(current_sq.0 as usize) as usize;
                            let ends = ONE_ENDS.get_unchecked(current_sq.0 as usize);
                            let backs = ONE_BACKS.get_unchecked(current_sq.0 as usize);

                            for i in 0..count {
                                let end = SQ(*ends.get_unchecked(i));
                                let end_bit = end.bit();
                                let back = *backs.get_unchecked(i);

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() || (backtrack_board & back).is_not_empty() {
                                        continue;

                                    }
                                    
                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ back;
                                    
                                    self.stack2.push(StackData2::new(Action2::from_piece(end_piece), new_backtrack_board, new_banned_positions, end));

                                    continue;

                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 || (backtrack_board & back).is_not_empty() {
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

                                if (backtrack_board & back).is_not_empty() {
                                    continue;
                    
                                }

                                G::store_end(&mut result, active_line_idx, end_bit);

                            }
                    
                        }
                        Action2::Gen2 => {
                            let intercepts = ALL_TWO_INTERCEPTS[current_sq.0 as usize];
                            let intercept_bb = board.piece_bb & intercepts;

                            let key = unsafe { compress_pext(intercepts, intercept_bb.0) };            
                            let idx = TWO_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(key as usize);

                            let count = *TWO_COUNTS.get_unchecked(*idx as usize) as usize;
                            let ends = TWO_ENDS.get_unchecked(*idx as usize);
                            let backs = TWO_BACKS.get_unchecked(*idx as usize);

                            for i in 0..count {
                                let end = SQ(*ends.get_unchecked(i));
                                let end_bit = end.bit();
                                let back = *backs.get_unchecked(i);

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() || (backtrack_board & back).is_not_empty() {
                                        continue;

                                    }

                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ back;
                                    
                                    self.stack2.push(StackData2::new(Action2::from_piece(end_piece), new_backtrack_board, new_banned_positions, end));

                                    continue;
                                
                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 || (backtrack_board & back).is_not_empty() {
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

                                if (backtrack_board & back).is_not_empty() {
                                    continue;
                    
                                }

                                G::store_end(&mut result, active_line_idx, end_bit);        

                            }

                        },
                        Action2::Gen3 => {
                            let intercepts = ALL_THREE_INTERCEPTS[current_sq.0 as usize];
                            let intercept_bb = board.piece_bb & intercepts;
                            
                            let key = unsafe { compress_pext(intercepts, intercept_bb.0) };            
                            let idx: &u16 = THREE_MAP.get_unchecked(current_sq.0 as usize).get_unchecked(key as usize);

                            let count = *THREE_COUNTS.get_unchecked(*idx as usize) as usize;
                            let ends = THREE_ENDS.get_unchecked(*idx as usize);
                            let backs = THREE_BACKS.get_unchecked(*idx as usize);

                            for i in 0..count {
                                let end = SQ(*ends.get_unchecked(i));
                                let end_bit = end.bit();
                                let back = *backs.get_unchecked(i);

                                if (board.piece_bb & end_bit).is_not_empty() {
                                    if (banned_positions & end_bit).is_not_empty() || (backtrack_board & back).is_not_empty() {
                                        continue;

                                    }

                                    G::store_bounce(&mut result, active_line_idx, end_bit);

                                    let end_piece = board.piece_at(end);
                                    let new_banned_positions = banned_positions ^ end_bit;
                                    let new_backtrack_board = backtrack_board ^ back;
                                    
                                    self.stack2.push(StackData2::new(Action2::from_piece(end_piece), new_backtrack_board, new_banned_positions, end));
    
                                    continue;
                                
                                }

                                let goal_bit = end_bit >> 36;
                                if goal_bit != 0 {
                                    if (goal_bit & player_bit) != 0 || (backtrack_board & back).is_not_empty() {
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

                                if (backtrack_board & back).is_not_empty() {
                                    continue;
                    
                                }

                                G::store_end(&mut result, active_line_idx, end_bit);

                            }

                        }

                    }

                }
                
                board.place(starting_piece, starting_sq);
                board.piece_bb ^= starting_sq.bit();

            }

        }

        G::exit(&mut result, board);

        result

    }


    // #[inline(always)]
    // unsafe fn process_paths<P: PieceLookup, G: GenType, Q: QuitType>(
    //     board:            &mut BoardState,
    //     result:           &mut GenResult,
    //     stack:            &mut Stack2,
    //     current_sq:       SQ,
    //     backtrack_board:  BitBoard,
    //     banned_positions: BitBoard,
    //     active_line_idx:  usize,
    //     player_bit:       u64,
    //     // starting_piece and starting_sq gone
    // ) -> bool {
    //     let data      = P::get_path_list(current_sq, board.piece_bb);
    //     let path_list = &*data.list;

    //     for i in 0..(path_list.count as usize) {
    //         let path    = path_list.paths.get_unchecked(i);
    //         let end     = SQ(path.0[P::END_IDX]);
    //         let end_bit = end.bit();

    //         if (board.piece_bb & end_bit).is_not_empty() {
    //             if (banned_positions & end_bit).is_not_empty()
    //                 || (backtrack_board & path.1).is_not_empty()
    //             {
    //                 continue;
    //             }

    //             G::store_bounce(result, active_line_idx, end_bit);

    //             stack.push(StackData2::new(
    //                 Action2::from_piece(board.piece_at(end)),
    //                 backtrack_board ^ path.1,
    //                 banned_positions ^ end_bit,
    //                 end,
    //             ));

    //             continue;
    //         }

    //         let goal_bit = end_bit >> 36;
    //         if goal_bit != 0 {
    //             if (goal_bit & player_bit) != 0
    //                 || (backtrack_board & path.1).is_not_empty()
    //             {
    //                 continue;
    //             }

    //             G::store_goal(result, active_line_idx, end_bit);

    //             if Q::QUIT {
    //                 return true;  // just signal, let gen4 handle cleanup
    //             }

    //             continue;
    //         }

    //         G::store_end(result, active_line_idx, end_bit);

    //     }

    //     false
        
    // }

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

// pub trait PieceLookup {
//     unsafe fn get_path_list(sq: SQ, piece_bb: BitBoard) -> PathData;
// }

// pub struct OneBounce;
// impl PieceLookup for OneBounce {
//     #[inline(always)]
//     unsafe fn get_path_list(sq: SQ, _piece_bb: BitBoard) -> PathData {
//         PathData {
//             list: ONE_PATH_LISTS.get_unchecked(sq.0 as usize),
//         }
//     }
// }

// pub struct TwoBounce;
// impl PieceLookup for TwoBounce {
//     #[inline(always)]
//     unsafe fn get_path_list(sq: SQ, piece_bb: BitBoard) -> PathData {
//         let intercepts = ALL_TWO_INTERCEPTS[sq.0 as usize];
//         let intercept_bb = piece_bb & intercepts;
//         let key = compress_pext(intercepts, intercept_bb.0);
//         let idx = TWO_MAP
//             .get_unchecked(sq.0 as usize)
//             .get_unchecked(key as usize);
//         PathData {
//             list: TWO_PATH_LISTS.get_unchecked(*idx as usize),
//         }
//     }
// }

// pub struct ThreeBounce;
// impl PieceLookup for ThreeBounce {
//     #[inline(always)]
//     unsafe fn get_path_list(sq: SQ, piece_bb: BitBoard) -> PathData {
//         let intercepts = ALL_THREE_INTERCEPTS[sq.0 as usize];
//         let intercept_bb = piece_bb & intercepts;
//         let key = compress_pext(intercepts, intercept_bb.0);
//         let idx = THREE_MAP
//             .get_unchecked(sq.0 as usize)
//             .get_unchecked(key as usize);
//         PathData {
//             list: THREE_PATH_LISTS.get_unchecked(*idx as usize),
//         }
//     }
// }

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
    pub banned_positions: BitBoard,
    pub current_sq: SQ,
    pub action: Action2

}

impl StackData2 {
    pub fn new(action: Action2, backtrack_board: BitBoard, banned_positions: BitBoard, current_sq: SQ) -> Self {
        Self {
            backtrack_board,
            banned_positions,
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
