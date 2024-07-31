//! Structures to represent moves, lists of moves, and move generation.
//! 
//! This module contains the move, rootmove structures.
//! 

pub mod move_list;

pub mod movegen;
pub mod movegen_consts;

use std::cmp::Ordering;
use std::fmt::Display;

use crate::board::*;
use crate::moves::movegen::*;
use crate::core::*;

/// Designates the type of move.
/// 
/// A move can either be a drop or a bounce.
/// A drop is a move that has three stages: the staring position, a piece that is replaced, and where that piece is dropped.
/// A bounce is a move that only has two stages: the starting position and the ending position.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum MoveType {
    Drop,
    Bounce,
    None

}

/// Structure that defines a move
/// 
/// A move has the data which represents how the move effects the board. 
/// The format of each of the tuples in the data array is a (Piece, Square) and it directly means put that piece at that square.
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Move {
    pub data: [(Piece, SQ); 3],
    pub flag: MoveType,

}

impl Move {
    /// Create a new move from its individual components.
    pub fn new(data: [(Piece, SQ); 3], flag: MoveType) -> Move {
        Move {
            data,
            flag,

        }

    }

    /// Creates a new null move.
    pub fn new_null() -> Move {
        Move {
            data: [(Piece::None, SQ::NONE); 3],
            flag: MoveType::None,

        }

    }

    /// Checks if a move is null.
    pub fn is_null(&self) -> bool {
        self.data == [(Piece::None, SQ::NONE); 3] && self.flag == MoveType::None

    }

    /// Checks if a move wins the game
    pub fn is_win(&self) -> bool {
        if self.flag == MoveType::Bounce {
            self.data[1].1 == SQ::P1_GOAL || self.data[1].1 == SQ::P2_GOAL

        } else {
             self.data[2].1 == SQ::P1_GOAL || self.data[2].1 == SQ::P2_GOAL

        }

    }

}

impl Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.flag == MoveType::Bounce {
            write!(f, "({}, {}) : ({}, {})", self.data[0].0, self.data[0].1, self.data[1].0, self.data[1].1)

        } else {
            write!(f, "({}, {}) : ({}, {}) : ({}, {})", self.data[0].0, self.data[0].1, self.data[1].0, self.data[1].1, self.data[2].0, self.data[2].1)

        }

    }

}

/// Structure that defines a rootmove.
/// 
/// A rootmove is a move that can store other data associated with the root of a search tree. This data includes the ply of the move and  
/// the score of the move.
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct RootMove {
    pub mv: Move,
    pub score: f64,
    pub threats: usize,
    pub ply: i8

}

impl RootMove {
    /// Creates a rootmove from its indivudal components.
    pub fn new(mv: Move, score: f64, ply: i8, threats: usize) -> RootMove {
        RootMove {
            mv,
            score,
            ply,
            threats,

        }

    }

    /// Creates a null rootmove.
    pub fn new_null() -> RootMove {
        RootMove {
            mv: Move::new_null(),
            score: 0.0,
            ply: 0,
            threats: 0,

        }

    }

    /// Checks if a rootmove is null.
    pub fn is_null(&self) -> bool {
        self.mv.is_null() && self.score == 0.0 && self.ply == 0

    }

    /// Sets the score and the search ply of a rootmove.
    pub fn set_score_and_ply(&mut self, score: f64, ply: i8) {
        self.score = score;
        self.ply = ply;

    }

}

impl Display for RootMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.mv.flag == MoveType::Bounce {
            write!(f, "{}|{}", self.mv.data[0].1, self.mv.data[1].1)

        } else {
            write!(f, "{}|{}|{}", self.mv.data[0].1, self.mv.data[1].1, self.mv.data[2].1)

        }

    }

}

/// Orders a list of moves.
pub fn order_moves(moves: Vec<Move>, board: &mut BoardState, player: Player) -> Vec<Move> {
    // For every move calculate a value to sort it by.
    let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
        let mut sort_val: f64 = 0.0;
        let mut new_board = board.make_move(&mv);
        
        sort_val -= unsafe{ valid_move_count(&mut new_board, player.other())} as f64;

        // If a move has less then 5 threats then penalize it.
        let threat_count = unsafe{ valid_threat_count(&mut new_board, player) };
        if threat_count <= 5_usize {
            sort_val -= 1000.0 * (5 - threat_count) as f64;

        }

        (mv, sort_val)

    }).collect();

    // Sort the moves based on their predicted values.
    moves_to_sort.sort_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    // Collect the moves.
    let ordered_moves: Vec<Move> = moves_to_sort.into_iter().map(|x| x.0).collect();

    ordered_moves
 
}
