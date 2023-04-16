use std::cmp::Ordering;
use std::ops::Deref;

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
    let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
        let predicted_score: f64;

        let mut new_board = board.make_move(&mv);

        for e in pv {
            if e.bestmove == mv {
                predicted_score = -500_000.0;
                return (mv, predicted_score);

            }

        }

        predicted_score = unsafe{valid_move_count(&mut new_board, -player)} as f64;
       
        return (mv, predicted_score);

    }).collect();

    moves_to_sort.sort_unstable_by(|a, b| {
        if a.1 < b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    let ordered_moves: Vec<Move> = moves_to_sort.into_iter().map(|x| x.0).collect();
    
    return ordered_moves;
 
}

pub fn tt_order_moves(moves: Vec<Move>, board: &mut BoardState) -> Vec<Move> {
    let mut moves_to_sort: Vec<(Move, f64)> = moves.into_iter().map(|mv| {
        let mut sort_val: f64 = f64::NEG_INFINITY;

        let new_board = board.make_move(&mv);
        let (vaild, entry) = unsafe{ tt().probe(new_board.hash()) };
        if vaild {
            sort_val = entry.score as f64;
        
        }

        return (mv, sort_val);

    }).collect();

    moves_to_sort.sort_unstable_by(|a, b| {
        if a.1 > b.1 {
            Ordering::Less
            
        } else if a.1 == b.1 {
            Ordering::Equal

        } else {
            Ordering::Greater

        }

    });

    let ordered_moves: Vec<Move> = moves_to_sort.into_iter().map(|x| x.0).collect();

    return ordered_moves;
    
}
