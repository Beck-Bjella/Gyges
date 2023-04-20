use std::cmp::Ordering;

use crate::consts::*;
use crate::board::*;
use crate::move_gen::*;
use crate::tt::*;

/// Designates the type of move.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum MoveType {
    Drop,
    Bounce,
    None

}

/// Structure that defines a move
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Move {
    pub data: [usize; 6],
    pub flag: MoveType,

}

impl Move {
    // Create a new move from its indivudal components.
    pub fn new(data: [usize; 6], flag: MoveType) -> Move {
        return Move {
            data,
            flag,

        };

    }

    /// Creates a new null move.
    pub fn new_null() -> Move {
        return Move {
            data: [NULL; 6],
            flag: MoveType::None,

        };

    }

    /// Checks if a move is null.
    pub fn is_null(&self) -> bool {
        if self.data == [NULL; 6] && self.flag == MoveType::None {
            return true;

        } else {
            return false;

        }

    }

}

/// Structure that defines a rootmove.
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
        return RootMove {
            mv,
            score,
            ply,
            threats

        };

    }

    /// Creates a null rootmove.
    pub fn new_null() -> RootMove {
        return RootMove {
            mv: Move::new_null(),
            score: 0.0,
            ply: 0,
            threats: 0

        };

    }

    /// Checks if a rootmove is null.
    pub fn is_null(&self) -> bool {
        if self.mv.is_null() && self.score == 0.0 && self.ply == 0 {
            return true;

        } else {
            return false;

        }

    }

    /// Sets the score and the search ply of a rootmove..-
    pub fn set_score_and_ply(&mut self, score: f64, ply: i8) {
        self.score = score;
        self.ply = ply;

    }

}

/// Orders a list of moves.
pub fn order_moves(moves: Vec<Move>, board: &mut BoardState, player: f64, pv: &Vec<Entry>) -> Vec<Move> {
    // For every move calculate a value to sort it by
    let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
        let mut sort_val: f64;
        
        // Check if move is in the PV
        for e in pv {
            if e.bestmove == mv {
                sort_val = 500_000.0;
                return (mv, sort_val);

            }

        }

        let mut new_board = board.make_move(&mv);

        // If move is not the PV then guess how good it is
        sort_val = -1.0 * unsafe { valid_move_count(&mut new_board, -player)} as f64;

        // If a move has less then 5 threats then penalize it
        let threat_count = unsafe{ valid_threat_count(&mut new_board, player) };
        if threat_count <= 5 as usize {
            sort_val -= 1000.0 * (5 - threat_count) as f64;

        }
       
        (mv, sort_val)

    }).collect();
    
    // Sort the moves based on their predicted values
    moves_to_sort.sort_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    // Collect the moves
    let ordered_moves: Vec<Move> = moves_to_sort.into_iter().map(|x| x.0).collect();

    ordered_moves
 
}
