use std::cmp::Ordering;

use crate::consts::*;
use crate::board::*;
use crate::move_gen::*;
use crate::tt::*;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum MoveType {
    Drop,
    Bounce,
    None

}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Move {
    pub data: [usize; 6],
    pub flag: MoveType,

}

impl Move {
    pub fn new(data: [usize; 6], flag: MoveType) -> Move {
        return Move {
            data,
            flag,

        };

    }

    pub fn new_null() -> Move {
        return Move {
            data: [NULL; 6],
            flag: MoveType::None,

        };

    }

    pub fn is_null(&self) -> bool {
        if self.data == [NULL; 6] && self.flag == MoveType::None {
            return true;

        } else {
            return false;

        }

    }

}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct RootMove {
    pub mv: Move,
    pub score: f64,
    pub threats: usize,
    pub ply: i8

}

impl RootMove {
    pub fn new(mv: Move, score: f64, ply: i8, threats: usize) -> RootMove {
        return RootMove {
            mv,
            score,
            ply,
            threats

        };

    }

    pub fn new_null() -> RootMove {
        return RootMove {
            mv: Move::new_null(),
            score: 0.0,
            ply: 0,
            threats: 0

        };

    }

    pub fn is_null(&self) -> bool {
        if self.mv.is_null() && self.score == 0.0 && self.ply == 0 {
            return true;

        } else {
            return false;

        }

    }

    pub fn set_score_and_ply(&mut self, score: f64, ply: i8) {
        self.score = score;
        self.ply = ply;

    }

}

pub fn order_moves(moves: Vec<Move>, board: &mut BoardState, player: f64, pv: &Vec<Entry>) -> Vec<Move> {
    // let mut best_move = Move::new_null();
    // let (vaild, entry) = unsafe{ tt().probe(board.hash()) };
    // if vaild && entry.bound == NodeBound::ExactValue {
    //     best_move = entry.bestmove;
        
    // }

    let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
        let mut sort_val: f64;
        
        // Check if move is in the PV
        for e in pv {
            if e.bestmove == mv {
                sort_val = 500_000.0;
                return (mv, sort_val);

            }

        }

        // If move is not the PV then guess how good it is
        let mut new_board = board.make_move(&mv);

        sort_val = -1.0 * unsafe { valid_move_count(&mut new_board, -player)} as f64;

        let threat_count = unsafe{ valid_threat_count(&mut new_board, player) };
        if threat_count <= 5 {
            sort_val -= 1000.0 * (5 - threat_count) as f64;

        }
       
        return (mv, sort_val);

    }).collect();
    
    moves_to_sort.sort_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    let ordered_moves: Vec<Move> = moves_to_sort.into_iter().map(|x| x.0).collect();
    
    // println!("{:?}", ordered_moves.len());

    return ordered_moves;
 
}
