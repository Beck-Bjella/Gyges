//! This module contains different movelists. 
//! 
//! There is the RawMoveList and the RootMoveList, and both follow the same pattern of storing groups of moves.
//! 

use std::cmp::Ordering;

use crate::board::*;
use crate::board::bitboard::*;
use crate::core::*;
use crate::moves::*;

/// An Encoded list of moves.
/// 
/// A RawMoveList stores the starting, pickup, and end postions of moves in BitBoards. 
/// These are easy and efficent to set and can decoded into a `Vec<Move>` when the real moves need to be used.
/// 
/// The RawMoveList has some major advantages over just using a `Vec<Move>`. The main reason is that it is much faster to generate.
/// The other main reason is that you can get general infomation about the types of moves that will be in the list before doing the costly transformation into a ```Vec<Move>```.
/// The most common example of this is checking if any move in the list will move into the opponents goal.
/// 
#[derive(Debug, Clone)]
pub struct RawMoveList {
    pub drop_positions: BitBoard,
    pub start_indexs: Vec<usize>,
    pub start_positions: [(Piece, SQ); 6],
    pub end_positions: [BitBoard; 6],
    pub pickup_positions: [BitBoard; 6],

}

impl RawMoveList {
    /// Creates an empty RawMoveList for a board with the drop positions already known.
    pub fn new(drop_positions: BitBoard) -> RawMoveList {
        RawMoveList {
            drop_positions,
            start_indexs: vec![],
            start_positions: [(Piece::None, SQ::NONE); 6],
            end_positions: [BitBoard::EMPTY; 6],
            pickup_positions: [BitBoard::EMPTY; 6],

        }

    }

    /// Defines a new piece that can move on a player's active line.
    pub fn set_start(&mut self, active_line_idx: usize, start_sq: SQ, start_piece: Piece) {
        self.start_indexs.push(active_line_idx);
        self.start_positions[active_line_idx] = (start_piece, start_sq);

    }

    /// Sets a possible end position for a piece.
    pub fn set_end_position(&mut self, active_line_idx: usize, end_bit: u64) {
        self.end_positions[active_line_idx] |= end_bit;

    }

    /// Sets a possible pickup position for a piece.
    pub fn set_pickup_position(&mut self, active_line_idx: usize, pickup_bit: u64) {
        self.pickup_positions[active_line_idx] |= pickup_bit;

    }

    /// Decodes the RawMoveList into a `Vec<Move>`
    ///
    /// Removes all data from the RawMoveList in the process of decoding.
    /// Do not try and use data in the list after this process.
    ///
    pub fn moves(&mut self, board: &BoardState) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::with_capacity(1000);

        let drop_positions = self.drop_positions.get_data();

        for idx in self.start_indexs.iter() {
            let start_position = self.start_positions[*idx];

            for end_pos in self.end_positions[*idx].get_data() {
                let data = [(Piece::None, start_position.1), (start_position.0, SQ(end_pos as u8)), (Piece::None, SQ::NONE)];
                moves.push(Move::new(data, MoveType::Bounce));

            }

            for pick_up_pos in self.pickup_positions[*idx].get_data() {
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

    /// Decode only moves matching the criticality-tiered blocker rules.
    ///
    /// `crit_partial` = squares on at least one threat path.
    /// `crit_full` = squares on every threat path (chokepoints).
    /// `shift_mask` = opp's back zone (only used to keep counter-threat bounces).
    pub fn moves_filtered(&mut self, board: &BoardState, crit_partial: BitBoard, crit_full: BitBoard, shift_mask: BitBoard) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::with_capacity(64);

        let mut crit_drop_bb = self.drop_positions & crit_partial;
        let crit_drop_positions = crit_drop_bb.get_data();
        let mut full_drop_bb = self.drop_positions & crit_full;
        let full_drop_positions = full_drop_bb.get_data();
        let all_drop_positions = self.drop_positions.get_data();

        for idx in self.start_indexs.iter() {
            let start_position = self.start_positions[*idx];
            let start_bit = BitBoard(start_position.1.bit());
            let start_full = (crit_full & start_bit).is_not_empty();
            let start_crit = (crit_partial & start_bit).is_not_empty();

            // Bounce: depends on start state. Shift mask keeps counter-threat bounces.
            let mut bounce_ends = if start_full {
                self.end_positions[*idx]

            } else if start_crit {
                self.end_positions[*idx] & (crit_partial | shift_mask)

            } else {
                self.end_positions[*idx] & (crit_full | shift_mask)

            };
            while bounce_ends.is_not_empty() {
                let end_pos = bounce_ends.pop_lsb();
                moves.push(Move::new(
                    [
                        (Piece::None, start_position.1),
                        (start_position.0, SQ(end_pos as u8)),
                        (Piece::None, SQ::NONE),
                    ],
                    MoveType::Bounce,
                ));

            }

            // Drop: shift mask not used (drops can't land in opp's back zone).
            let mut pickups = self.pickup_positions[*idx];
            while pickups.is_not_empty() {
                let pick_up_pos = pickups.pop_lsb();
                let pickup_sq = SQ(pick_up_pos as u8);
                let pickup_crit = (crit_partial & BitBoard(pickup_sq.bit())).is_not_empty();
                let pickup_full = (crit_full & BitBoard(pickup_sq.bit())).is_not_empty();
                let displaced = board.piece_at(pickup_sq);

                // Pickup-only drop: drop_dest = start. Keep if pickup is crit
                // OR start (= drop_dest) is crit.
                let keep_pickup_only = pickup_crit || start_crit;

                if keep_pickup_only {
                    moves.push(Move::new(
                        [
                            (Piece::None, start_position.1),
                            (start_position.0, pickup_sq),
                            (displaced, start_position.1),
                        ],
                        MoveType::Drop,
                    ));

                }

                // Drop-elsewhere: full crit on start or pickup frees drop_dest.
                let drops: &Vec<usize> = if start_full || pickup_full {
                    &all_drop_positions

                } else if start_crit {
                    if pickup_crit { &all_drop_positions } else { &crit_drop_positions }

                } else if pickup_crit {
                    &crit_drop_positions

                } else {
                    &full_drop_positions

                };

                for drop_pos in drops.iter() {
                    moves.push(Move::new(
                        [
                            (Piece::None, start_position.1),
                            (start_position.0, pickup_sq),
                            (displaced, SQ(*drop_pos as u8)),
                        ],
                        MoveType::Drop,
                    ));

                }

            }

        }

        moves

    }

    /// Checks if there is a move in the RawMoveList that would move into the opponents goal.
    pub fn has_threat(&mut self, player: Player) -> bool {
        for idx in self.start_indexs.iter() {
            if (self.end_positions[*idx] & SQ::GOALS[player.other() as usize].bit()).is_not_empty() {
                return true;

            } 
        
        }

        false

    }

}

/// A sortable list of RootMove's
/// 
/// Very simmilar to using `Vec<RootMove>` except implements custom functions for setup and sorting.
/// 
#[derive(Clone)]
pub struct RootMoveList {
    pub moves: Vec<RootMove>,

}

impl RootMoveList {
    pub fn new() -> RootMoveList {
        RootMoveList {
            moves: vec![],

        }

    }
    
    /// Sorts the RootMoveList by whatever move has the greatest score.
    pub fn sort(&mut self) {
        self.moves.sort_by(|a, b| {
            if a.score > b.score {
                Ordering::Less
                
            } else if a.score == b.score {
                match a.threats.cmp(&b.threats) {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Equal => Ordering::Equal,
                    Ordering::Greater => Ordering::Less,

                }

            } else {
                Ordering::Greater
    
            }
    
        });
    
    }

    /// Updates the score and ply fields of a specific move.
    pub fn update_move(&mut self, mv: Move, score: f64, ply: i8) {
        for root_move in self.moves.iter_mut() {
            if root_move.mv == mv {
                root_move.set_score_and_ply(score, ply);

            }

        }

        self.sort();

    }

    /// Returns the first move
    pub fn first(&self) -> RootMove {
        self.moves[0]

    }

}

impl From<RootMoveList> for Vec<Move> {
    fn from(val: RootMoveList) -> Vec<Move> {
        let moves: Vec<Move> = val.moves.into_iter().map( |mv| {
            mv.mv

        }).collect();

        moves

    }

}

impl Default for RootMoveList {
    fn default() -> Self {
        Self::new()

    }

}
