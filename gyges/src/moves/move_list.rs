use std::cmp::Ordering;

use crate::board::board::*;
use crate::board::bitboard::*;
use crate::core::player::*;
use crate::core::sq::*;
use crate::core::piece::*;
use crate::moves::moves::*;
use crate::moves::movegen::*;

#[derive(Clone)]
pub struct RawMoveList {
    pub drop_positions: BitBoard,
    pub start_indexs: Vec<usize>,
    pub start_positions: [(Piece, SQ); 6],
    pub end_positions: [BitBoard; 6],
    pub pickup_positions: [BitBoard; 6],

}

impl RawMoveList {
    pub fn new(drop_positions: BitBoard) -> RawMoveList {
        RawMoveList {
            drop_positions,
            start_indexs: vec![],
            start_positions: [(Piece::None, SQ::NONE); 6],
            end_positions: [BitBoard::EMPTY; 6],
            pickup_positions: [BitBoard::EMPTY; 6],

        }

    }

    pub fn add_start_index(&mut self, index: usize) {
        self.start_indexs.push(index);

    }

    pub fn set_start(&mut self, active_line_idx: usize, start_sq: SQ, start_piece: Piece) {
        self.start_positions[active_line_idx] = (start_piece, start_sq);

    }

    pub fn set_end_position(&mut self, active_line_idx: usize, end_bit: u64) {
        self.end_positions[active_line_idx] |= end_bit;

    }

    pub fn set_pickup_position(&mut self, active_line_idx: usize, pickup_bit: u64) {
        self.pickup_positions[active_line_idx] |= pickup_bit;

    }

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

    pub fn has_threat(&mut self, player: Player) -> bool {
        for idx in self.start_indexs.iter() {
            if (self.end_positions[*idx] & SQ::GOALS[player.other() as usize].bit()).is_not_empty() {
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
        let moves = order_moves(unsafe { valid_moves(board, Player::One) }.moves(board), board, Player::One);

        let root_moves: Vec<RootMove> = moves.iter().map( |mv| {
            let mut new_board = board.make_move(mv);
            let threats: usize = unsafe { valid_threat_count(&mut new_board, Player::One) };

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
