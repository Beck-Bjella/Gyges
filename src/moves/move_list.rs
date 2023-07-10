use std::cmp::Ordering;

use crate::board::bitboard::*;
use crate::moves::moves::*;
use crate::board::board::*;
use crate::moves::move_gen::*;
use crate::consts::*;
use crate::tools::tt::Entry;

#[derive(Clone)]
pub struct RawMoveList {
    pub drop_positions: BitBoard,
    pub start_indexs: Vec<usize>,
    pub start_positions: [(usize, usize); 6],
    pub end_positions: [BitBoard; 6],
    pub pickup_positions: [BitBoard; 6],

}

impl RawMoveList {
    pub fn new(drop_positions: BitBoard) -> RawMoveList {
        RawMoveList {
            drop_positions,
            start_indexs: vec![],
            start_positions: [(NULL, NULL); 6],
            end_positions: [BitBoard(0); 6],
            pickup_positions: [BitBoard(0); 6],

        }

    }

    pub fn add_start_index(&mut self, index: usize) {
        self.start_indexs.push(index);

    }

    pub fn set_start(&mut self, active_line_idx: usize, start_pos: usize, start_piece_type: usize) {
        self.start_positions[active_line_idx] = (start_pos, start_piece_type);

    }

    pub fn set_end_position(&mut self, active_line_idx: usize, end_position: usize) {
        self.end_positions[active_line_idx].set_bit(end_position);

    }

    pub fn set_pickup_position(&mut self, active_line_idx: usize, pickup_position: usize) {
        self.pickup_positions[active_line_idx].set_bit(pickup_position);

    }

    pub fn moves(&mut self, board: &BoardState) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::with_capacity(1000);

        let drop_positions = self.drop_positions.get_data();

        for idx in self.start_indexs.iter() {
            let start_position = self.start_positions[*idx];
            
            for end_pos in self.end_positions[*idx].get_data() {
                moves.push(Move::new([0, start_position.0, start_position.1, end_pos, NULL, NULL], MoveType::Bounce));

            }

            for pick_up_pos in self.pickup_positions[*idx].get_data() {
                moves.push(Move::new([0, start_position.0, start_position.1, pick_up_pos, board.data[pick_up_pos], start_position.0], MoveType::Drop));

                for drop_pos in drop_positions.iter() {
                    moves.push(Move::new([0, start_position.0, start_position.1, pick_up_pos, board.data[pick_up_pos], *drop_pos], MoveType::Drop));

                }
        
            }
        
        }

        moves

    }

    pub fn has_threat(&mut self, player: f64) -> bool {
        for idx in self.start_indexs.iter() {
            if player == PLAYER_1 {
                if (self.end_positions[*idx] & (1 << PLAYER_2_GOAL)).is_not_empty() {
                    return true;

                } 

            } else if player == PLAYER_2 && (self.end_positions[*idx] & (1 << PLAYER_1_GOAL)).is_not_empty() {
                return true;

            }
        
        }

        false

    }

}

pub struct RootMoveList {
    pub moves: Vec<RootMove>,

}

impl RootMoveList {
    pub fn new() -> RootMoveList {
        RootMoveList {
            moves: vec![],

        }

    }
    
    pub fn sort(&mut self) {
        self.moves.sort_by(|a, b| {
            if a.score > b.score {
                Ordering::Less
                
            } else if a.score == b.score {
                // match a.threats.cmp(&b.threats) {
                //     Ordering::Greater => Ordering::Less,
                    
                //     Ordering::Less => Ordering::Greater,

                //     Ordering::Equal => Ordering::Equal

                // }

                Ordering::Equal
    
            } else {
                Ordering::Greater
    
            }
    
        });
    
    }

    pub fn update_move(&mut self, mv: Move, score: f64, ply: i8) {
        for root_move in self.moves.iter_mut() {
            if root_move.mv == mv {
                root_move.set_score_and_ply(score, ply);

            }

        }

        self.sort();

    }

    pub fn setup(&mut self, board: &mut BoardState) {
        let moves = order_moves(unsafe { valid_moves(board, PLAYER_1) }.moves(board), board, PLAYER_1, &vec![]);

        let root_moves: Vec<RootMove> = moves.iter().map( |mv| {
            let mut new_board = board.make_move(mv);
            let threats = unsafe { valid_threat_count(&mut new_board, PLAYER_1) };

            RootMove::new(*mv, 0.0, 0, threats)

        }).collect();

        self.moves = root_moves;

    }

    pub fn first(&self) -> RootMove {
        self.moves[0]

    }

    pub fn as_vec(&self) -> Vec<Move> {
        let moves: Vec<Move> = self.moves.clone().into_iter().map( |mv| {
            mv.mv

        }).collect();

        moves

    }

}
